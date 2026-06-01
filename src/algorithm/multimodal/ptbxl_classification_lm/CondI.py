from ...fedbase import BasicServer, BasicClient
import utils.system_simulator as ss
from utils import fmodule
import copy
import collections
import utils.fflow as flw
import os
import torch
import numpy as np


class Server(BasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)
        self.n_leads = 12
        self.list_testing_leads = [
            list(range(self.n_leads))
        ]
        self.num_outer_loops = option['num_outer_loops']
        self.pm = option['pm']
        self.ps = option['ps']
        self.list_testing_rates = [
            [self.ps, self.pm]
        ]
        self.quantile = option['graph_quantile']
        self.checkpoints_dir = os.path.join('fedtask', option['task'], 'checkpoints', f"{option['ps']}_{option['pm']}_{option['graph_quantile']}")
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        
        self.test_metric = None # update after each iteration
        self.best_metric = 0.0    # acc
        self.visualize=option['visualize']
        if self.visualize:
            print("[WARN] Visualize mode for debugging...")
        self.option = option
        self.correlation_matrix = None
        

    def run(self):
        """
        Start the federated learning symtem where the global model is trained iteratively.
        """
        flw.logger.time_start('Total Time Cost')
        for round in range(1, self.num_rounds+1):
            self.current_round = round
            ss.clock.step()
            # using logger to evaluate the model
            flw.logger.info("--------------Round {}--------------".format(round))
            flw.logger.time_start('Time Cost')
            
            # get current best acc
            if self.test_metric is not None: 
                self.best_metric = max(self.best_metric, self.test_metric['acc1']) 
            
            if flw.logger.check_if_log(round, self.eval_interval) and round >= 1:   # initial check before any training starts
                flw.logger.time_start('Eval Time Cost')
                flw.logger.log_once()   # testing phase happen here !
                flw.logger.time_end('Eval Time Cost') 
                
            # check condition for saving best weight
            if self.test_metric is not None:
                if self.test_metric['acc1'] > self.best_metric:
                    self.best_metric = self.test_metric['acc1']
                    print(f'New best metric: {self.best_metric}')
                    self.save_checkpoints() # save current checkpoints
                
            # check if early stopping
            if flw.logger.early_stop(): break
            # federated train
            self.iterate(round)
            # decay learning rate
            self.global_lr_scheduler(round)
            flw.logger.time_end('Time Cost')
        flw.logger.info("--------------Final Evaluation--------------")
        flw.logger.time_start('Eval Time Cost')
        flw.logger.log_once()
        flw.logger.time_end('Eval Time Cost')
        flw.logger.info("=================End==================")
        flw.logger.time_end('Total Time Cost')
        # save results as .json file
        flw.logger.save_output_as_json()
        return
    
    def save_checkpoints(self):
        print("Saving global model checkpoints!")
        outdir = os.path.join(self.checkpoints_dir, 'FedMac1_na', 'global-model')
        os.makedirs(outdir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(outdir, f'model.pt'))

    def load_checkpoints(self):
        if os.path.exists(os.path.join(self.checkpoints_dir, 'Original_MTM', 'global-model')):
            print("Loading global model checkpoints!")
            self.model.load_state_dict(torch.load(os.path.join(self.checkpoints_dir, 'global-model', 'model.pt')))

    def iterate(self, round):
        """
        The standard iteration of each federated round that contains three
        necessary procedure in FL: client selection, communication and model aggregation.
        :param
            t: the number of current round
        """
        # sample clients: MD sampling as default
        self.selected_clients = self.sample()
        # training
        conmmunitcation_result = self.communicate(self.selected_clients)
        models = conmmunitcation_result['model']
        modalities_list = conmmunitcation_result['modalities']
        
        self.model = self.aggregate(models, modalities_list)
        return

    @torch.no_grad()
    def aggregate(self, models: list, modalities_list: list):
        print(f"\n[Server] Started Aggregating {len(self.selected_clients)}/{len(self.clients)} clients: {self.selected_clients}")
        
        n_models = len(models)
        new_model = copy.deepcopy(self.model)
        
        # We perform standard FedAvg for all shared components across all clients.
        # This allows clients to receive weights for modalities they didn't train on.
        p = [self.clients[client_id].datavol for client_id in self.selected_clients]
        
        for m in range(self.n_leads):
            # Only average from models that trained this modality to avoid diluting with untrained weights
            pm_m = list()
            chosen_models_m = list()
            for k, client_id in enumerate(self.selected_clients):
                if m in modalities_list[k]:
                    pm_m.append(self.clients[client_id].datavol)
                    chosen_models_m.append(models[k])
            if len(pm_m) > 0:
                # w_ins encoder (per-lead)
                new_model.pre_extractors[m] = fmodule._model_sum([
                    model.pre_extractors[m] * pk for model, pk in zip(chosen_models_m, pm_m)
                ]) / sum(pm_m)
                # w_imputed encoder (per-lead, independent)
                new_model.imputed_pre_extractors[m] = fmodule._model_sum([
                    model.imputed_pre_extractors[m] * pk for model, pk in zip(chosen_models_m, pm_m)
                ]) / sum(pm_m)
                # f_main encoder (per-lead, independent)
                new_model.f_main_pre_extractors[m] = fmodule._model_sum([
                    model.f_main_pre_extractors[m] * pk for model, pk in zip(chosen_models_m, pm_m)
                ]) / sum(pm_m)

        # Shared Inception for w_ins
        new_model.classifier = fmodule._model_sum([
            model.classifier * pk for model, pk in zip(models, p)
        ]) / sum(p)
        new_model.feature_extractors[0] = fmodule._model_sum([
            model.feature_extractors[0] * pk for model, pk in zip(models, p)
        ]) / sum(p)
        # Independent Inception for w_imputed
        new_model.imputed_feature_extractors[0] = fmodule._model_sum([
            model.imputed_feature_extractors[0] * pk for model, pk in zip(models, p)
        ]) / sum(p)
        # Independent Inception for f_main (output_dim=250)
        new_model.f_main_feature_extractors[0] = fmodule._model_sum([
            model.f_main_feature_extractors[0] * pk for model, pk in zip(models, p)
        ]) / sum(p)

        # Aggregate per_modality_imputer (diffusion imputer)
        print("[VERIFICATION] Server is aggregating per_modality_imputer parameters via FedAvg.")
        new_model.per_modality_imputer = fmodule._model_sum([
            model.per_modality_imputer * pk for model, pk in zip(models, p)
        ]) / sum(p)

        # Aggregate cond_encoder
        total_p = float(sum(p))
        cond_enc_state = copy.deepcopy(new_model.cond_encoder.state_dict())
        for key in cond_enc_state:
            cond_enc_state[key] = sum(model.cond_encoder.state_dict()[key] * pk for model, pk in zip(models, p)) / total_p
        new_model.cond_encoder.load_state_dict(cond_enc_state)

        # Aggregate modality_embeddings (w_mod)
        new_model.modality_embeddings.data = sum(
            model.modality_embeddings.data * pk for model, pk in zip(models, p)
        ) / total_p

        # Aggregate aux_gate
        aux_state = copy.deepcopy(new_model.aux_gate.state_dict())
        for key in aux_state:
            aux_state[key] = sum(model.aux_gate.state_dict()[key] * pk for model, pk in zip(models, p)) / total_p
        new_model.aux_gate.load_state_dict(aux_state)

        # Aggregate ins_imputed_encoder (kept for compat)
        ins_enc_state = copy.deepcopy(new_model.ins_imputed_encoder.state_dict())
        for key in ins_enc_state:
            ins_enc_state[key] = sum(model.ins_imputed_encoder.state_dict()[key] * pk for model, pk in zip(models, p)) / total_p
        new_model.ins_imputed_encoder.load_state_dict(ins_enc_state)

        return new_model
    
    def test(self, model=None):
        """
        Evaluate the model on the test dataset owned by the server.
        :param
            model: the model need to be evaluated
        :return:
            metrics: specified by the task during running time (e.g. metric = [mean_accuracy, mean_loss] when the task is classification)
        """
        # return dict()
        if model is None: model=self.model
        if self.test_data:
            return self.calculator.lm_server_test(
                model=model,
                dataset=self.test_data,
                batch_size=self.option['test_batch_size'],
                leads=self.list_testing_leads,
                rates=self.list_testing_rates,
                ps=self.ps,
                pm=self.pm,
                quantile=self.quantile,
                visualize=self.visualize,
                contrastive_weight=self.option['contrastive_weight']
            )
        else:
            return None

    def test_on_clients(self, dataflag='train'):
        """
        Validate accuracies and losses on clients' local datasets
        :param
            dataflag: choose train data or valid data to evaluate
        :return
            metrics: a dict contains the lists of each metric_value of the clients
        """
        all_metrics = collections.defaultdict(list)
        for client_id in self.selected_clients:
            c = self.clients[client_id]
            # Skip clients with empty datasets
            if dataflag == 'train' and len(c.train_data) == 0:
                continue
            elif dataflag == 'valid' and len(c.valid_data) == 0:
                continue
            client_metrics = c.test(self.model, dataflag)
            for met_name, met_val in client_metrics.items():
                all_metrics[met_name].append(met_val)
        return all_metrics


class Client(BasicClient):
    def __init__(self, option, modalities, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)
        self.n_leads = 12
        self.pm = option['pm']
        self.ps = option['ps']
        self.quantile = option['graph_quantile']
        self.train_data = train_data
        self.train_data.local_missing_setup(modalities, self.ps, self.pm)   # local missing setting -  store the indices of missing as -1
        self.valid_data.local_missing_setup(modalities, self.ps, self.pm)
        self.fedmsplit_prox_lambda = option['fedmsplit_prox_lambda']
        self.modalities = modalities
        self.local_model = None
        
        self.correlation_matrix = None
        self.contrastive_weight = option['contrastive_weight']
        self.num_diff_steps = option.get('num_diff_steps', 1)
        # Some runs pass diff_lr=None explicitly; fall back to learning_rate in that case.
        self.diff_lr = option.get('diff_lr')
        if self.diff_lr is None:
            self.diff_lr = option['learning_rate']
        
        # self.train_data.save_data_dist(pm=self.pm, ps=self.ps, fn=f'client_{self.name}_{self.pm}_{self.ps}.png')

    def pack(self, model):
        """
        Packing the package to be send to the server. The operations of compression
        of encryption of the package should be done here.
        :param
            model: the locally trained model
        :return
            package: a dict that contains the necessary information for the server
        """
        return {
            "model" : model,
            "modalities": self.modalities
        }

    def reply(self, svr_pkg):
        """
        Reply to server with the transmitted package.
        The whole local procedure should be planned here.
        The standard form consists of three procedure:
        unpacking the server_package to obtain the global model,
        training the global model, and finally packing the updated
        model into client_package.
        :param
            svr_pkg: the package received from the server
        :return:
            client_pkg: the package to be send to the server
        """
        model = self.unpack(svr_pkg)
        # if self.local_model is None:
        #     self.local_model = copy.deepcopy(model)
        self.local_model = copy.deepcopy(model)
        self.train(self.local_model)
        cpkg = self.pack(self.local_model)
        return cpkg

    @ss.with_completeness
    @fmodule.with_multi_gpus
    def train(self, model):
        """
        Two-phase local training:
          Phase A — Imputation (diffusion imputer + cond_encoder)
          Phase B — Classification (all non-imputer params)
        """
        self.train_data.local_missing_setup(self.modalities, self.ps, self.pm)
        self.valid_data.local_missing_setup(self.modalities, self.ps, self.pm)

        # ---- Parameter groups ----
        diff_prefixes = ('per_modality_imputer',)
        cond_prefixes = ('cond_encoder', 'ins_imputed_encoder')

        diff_params = [p for n, p in model.named_parameters() if n.startswith(diff_prefixes)]
        cond_params = [p for n, p in model.named_parameters() if n.startswith(cond_prefixes)]
        condi_params = [p for n, p in model.named_parameters()
                        if not n.startswith(diff_prefixes) and not n.startswith(cond_prefixes)]

        if not hasattr(self, "_logged_param_check"):
            n_d, n_co, n_p = len(diff_params), len(cond_params), len(condi_params)
            n_total = len(list(model.parameters()))
            print(f"[VERIFICATION] Client {self.name} - diff:{n_d} cond:{n_co} condi:{n_p} total:{n_total}")
            assert n_d + n_co + n_p == n_total
            self._logged_param_check = True

        from tqdm import tqdm
        round_id = getattr(self.server, "current_round", -1)

        def _set_bn_mode(prefixes_train):
            for n, m in model.named_modules():
                if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                    if any(n.startswith(pf) for pf in prefixes_train):
                        m.train()
                    else:
                        m.eval()

        # ==============================================================
        # PHASE A: Imputation (train diff_params + cond_params)
        # ==============================================================
        model.train()
        model._detach_imputation_loss = False
        model._contrastive_lambda = 0.0  # no contrastive in Phase A
        for p in model.parameters():
            p.requires_grad = False
        for p in diff_params + cond_params:
            p.requires_grad = True
        _set_bn_mode(prefixes_train=['per_modality_imputer', 'cond_encoder', 'ins_imputed_encoder'])

        optimizer_A = torch.optim.AdamW(
            diff_params + cond_params, lr=self.diff_lr, weight_decay=self.weight_decay)

        phaseA_losses = []
        pbar = tqdm(range(self.num_diff_steps), desc=f"Client {self.name} Phase A (Diff)", leave=False)
        for it in pbar:
            batch_data = self.get_batch_data()
            if batch_data[-1].shape[0] == 1:
                continue
            optimizer_A.zero_grad()
            _ = self.calculator.train_one_step(
                model=model, data=batch_data, leads=self.modalities,
                quantile=self.quantile, matrix=self.correlation_matrix,
                contrastive_weight=self.contrastive_weight)
            loss_diff = getattr(model, "loss_imputation", 0.0)
            if isinstance(loss_diff, torch.Tensor):
                loss_diff.backward()
                optimizer_A.step()
                phaseA_losses.append(float(loss_diff.detach().item()))
                pbar.set_postfix({"loss": f"{phaseA_losses[-1]:.4f}"})
        pbar.close()
        print(f"[PHASE-A][Round {round_id}][Client {self.name}] "
              f"mean_loss_diff={np.mean(phaseA_losses) if phaseA_losses else float('nan'):.6f}")

        # ==============================================================
        # PHASE B: Classification (train condi_params only; cond frozen)
        # ==============================================================
        model.train()
        model._detach_imputation_loss = True
        model._contrastive_lambda = 0.2  # supervised contrastive on w_ins
        for p in model.parameters():
            p.requires_grad = False
        for p in condi_params:            # cond_params NOT included → frozen
            p.requires_grad = True
        # Restore all BN to train mode for Phase B
        for n, m in model.named_modules():
            if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                if not n.startswith('per_modality_imputer'):
                    m.train()

        optimizer_B = self.calculator.get_optimizer(
            model=model, lr=self.learning_rate,
            weight_decay=self.weight_decay, momentum=self.momentum)

        phaseB_losses = []
        pbar = tqdm(range(self.num_steps), desc=f"Client {self.name} Phase B (Classify)", leave=False)
        for it in pbar:
            batch_data = self.get_batch_data()
            if batch_data[-1].shape[0] == 1:
                continue
            model.zero_grad()
            res = self.calculator.train_one_step(
                model=model, data=batch_data, leads=self.modalities,
                quantile=self.quantile, matrix=self.correlation_matrix,
                contrastive_weight=self.contrastive_weight)
            loss = res['loss']
            loss.backward()
            optimizer_B.step()
            phaseB_losses.append(float(loss.detach().item()))
            pbar.set_postfix({"loss": f"{phaseB_losses[-1]:.4f}"})
        pbar.close()
        print(f"[PHASE-B][Round {round_id}][Client {self.name}] "
              f"mean_loss_cls={np.mean(phaseB_losses) if phaseB_losses else float('nan'):.6f}")

        # Restore
        for p in model.parameters():
            p.requires_grad = True
        model._detach_imputation_loss = False
        return

    @fmodule.with_multi_gpus
    def test(self, model, dataflag='train'):
        """
        Evaluate the model with local data (e.g. training data or validating data).
        :param
            model:
            dataflag: choose the dataset to be evaluated on
        :return:
            metric: specified by the task during running time (e.g. metric = [mean_accuracy, mean_loss] when the task is classification)
        """
        if dataflag == "train":
            dataset = self.train_data
        elif dataflag == "valid":
            dataset = self.valid_data
        return self.calculator.test(
            model=model,
            dataset=dataset,
            leads=self.modalities,
            quantile=self.quantile,
            contrastive_weight=self.contrastive_weight
        )
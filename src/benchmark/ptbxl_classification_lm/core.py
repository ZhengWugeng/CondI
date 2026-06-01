from .dataset import PTBXLReduceDataset
from benchmark.toolkits import DefaultTaskGen
from benchmark.toolkits import ClassificationCalculator
from benchmark.toolkits import IDXTaskPipe
from benchmark.ptbxl_classification_lm.subset import ClientSubset, ImputedClientSubset
import os
import ujson
import importlib
import random
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import numpy as np
from tqdm import tqdm
from icecream import ic

import warnings
warnings.filterwarnings('ignore')

class TaskPipe(IDXTaskPipe):
    TaskDataset = ClientSubset
    @classmethod
    def load_task(cls, task_path):
        with open(os.path.join(task_path, 'data.json'), 'r') as inf:
            feddata = ujson.load(inf)
        class_path = feddata['datasrc']['class_path']
        class_name = feddata['datasrc']['class_name']
        if feddata['datasrc']['imputation']: 
            cls.TaskDataset = ImputedClientSubset
        
        origin_class = getattr(importlib.import_module(class_path), class_name)
        origin_train_data = cls.args_to_dataset(origin_class, feddata['datasrc']['train_args'])
        origin_test_data = cls.args_to_dataset(origin_class, feddata['datasrc']['test_args'])
        test_data = cls.TaskDataset(origin_test_data, [_ for _ in range(len(origin_test_data))])
        train_datas = []
        valid_datas = []
        modalities_list = []
        for i, name in enumerate(feddata['client_names']):
            train_data = feddata[name]['dtrain']    # sample idx
            valid_data = feddata[name]['dvalid']
            if cls._cross_validation:
                k = len(train_data)
                train_data.extend(valid_data)
                random.shuffle(train_data)
                all_data = train_data
                train_data = all_data[:k]
                valid_data = all_data[k:]
            if cls._train_on_all:
                train_data.extend(valid_data)
            train_datas.append(cls.TaskDataset(origin_train_data, train_data, seed=i))
            valid_datas.append(cls.TaskDataset(origin_train_data, valid_data, seed=i))
            modalities_list.append(feddata[name]['modalities'])
            # modalities_list.append(list(range(12)))
        return train_datas, valid_datas, test_data, feddata['client_names'], modalities_list

def save_task(generator):
    """
    Store the splited indices of the local data in the original dataset (source dataset) into the disk as .json file
    The input 'generator' must have attributes:
        :taskpath: string. the path of storing
        :train_data: the training dataset which is a dict {'x':..., 'y':...}
        :test_data: the testing dataset which is a dict {'x':..., 'y':...}
        :train_cidxs: a list of lists of integer. The splited indices in train_data of the training part of each local dataset
        :valid_cidxs: a list of lists of integer. The splited indices in train_data of the valiadtion part of each local dataset
        :client_names: a list of strings. The names of all the clients, which is used to index the clients' data in .json file
        :source_dict: a dict that contains parameters which is necessary to dynamically importing the original Dataset class and generating instances
                For example, for MNIST using this task pipe, the source_dict should be like:
                {'class_path': 'torchvision.datasets',
                    'class_name': 'MNIST',
                    'train_args': {'root': '"'+MNIST_rawdata_path+'"', 'download': 'True', 'transform': 'transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])','train': 'True'},
                    'test_args': {'root': '"'+MNIST_rawdata_path+'"', 'download': 'True', 'transform': 'transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])', 'train': 'False'}
                }
        :return:
    """
    feddata = {
        'store': 'IDX',
        'client_names': generator.cnames,
        'dtest': [i for i in range(len(generator.test_data))],
        'datasrc': generator.source_dict
    }
    for cid in range(len(generator.cnames)):
        if generator.specific_training_leads:
            feddata[generator.cnames[cid]] = {
                'modalities': generator.specific_training_leads[cid],
                'dtrain': generator.train_cidxs[cid],
                'dvalid': generator.valid_cidxs[cid]
            }
        else:
            feddata[generator.cnames[cid]] = {
                'dtrain': generator.train_cidxs[cid],
                'dvalid': generator.valid_cidxs[cid]
            }
    with open(os.path.join(generator.taskpath, 'data.json'), 'w') as outf:
        ujson.dump(feddata, outf, default=convert)
    return
    
def convert(o):
    if isinstance(o, np.int64): return int(o)  
    raise TypeError

def iid_divide(l, g):
    """
    https://github.com/TalwalkarLab/leaf/blob/master/data/utils/sample.py
    divide list `l` among `g` groups
    each group has either `int(len(l)/g)` or `int(len(l)/g)+1` elements
    returns a list of groups
    """
    num_elems = len(l)
    group_size = int(len(l) / g)
    num_big_groups = num_elems - g * group_size
    num_small_groups = g - num_big_groups
    glist = []
    for i in range(num_small_groups):
        glist.append(l[group_size * i: group_size * (i + 1)])
    bi = group_size * num_small_groups
    group_size += 1
    for i in range(num_big_groups):
        glist.append(l[bi + group_size * i:bi + group_size * (i + 1)])
    return glist


def split_list_by_indices(l, indices):
    """
    divide list `l` given indices into `len(indices)` sub-lists
    sub-list `i` starts from `indices[i]` and stops at `indices[i+1]`
    returns a list of sub-lists
    """
    res = []
    current_index = 0
    for index in indices:
        res.append(l[current_index: index])
        current_index = index

    return res


def by_labels_non_iid_split(dataset, n_classes, n_clients, n_clusters, alpha, frac, seed=1234):
    """
    split classification dataset among `n_clients`. The dataset is split as follow:
        1) classes are grouped into `n_clusters`
        2) for each cluster `c`, samples are partitioned across clients using dirichlet distribution

    Inspired by the split in "Federated Learning with Matched Averaging"__(https://arxiv.org/abs/2002.06440)

    :param dataset:
    :type dataset: torch.utils.Dataset
    :param n_classes: number of classes present in `dataset`
    :param n_clients: number of clients
    :param n_clusters: number of clusters to consider; if it is `-1`, then `n_clusters = n_classes`
    :param alpha: parameter controlling the diversity among clients
    :param frac: fraction of dataset to use
    :param seed:
    :return: list (size `n_clients`) of subgroups, each subgroup is a list of indices.
    """
    if n_clusters == -1:
        n_clusters = n_classes

    rng_seed = (seed if (seed is not None and seed >= 0) else int(time.time()))
    rng = random.Random(rng_seed)
    np.random.seed(rng_seed)

    all_labels = list(range(n_classes))
    rng.shuffle(all_labels)
    clusters_labels = iid_divide(all_labels, n_clusters)        #cluster=class

    label2cluster = dict()  # maps label to its cluster
    for group_idx, labels in enumerate(clusters_labels):
        for label in labels:
            label2cluster[label] = group_idx
    # get subset
    n_samples = int(len(dataset) * frac)
    selected_indices = np.random.randint(0, len(dataset), n_samples)

    clusters_sizes = np.zeros(n_clusters, dtype=int)
    clusters = {k: [] for k in range(n_clusters)}       # class:[indices]
    for idx in selected_indices:
        _, label = dataset[idx]
        group_id = label2cluster[int(label)]
        clusters_sizes[group_id] += 1
        clusters[group_id].append(idx)

    for _, cluster in clusters.items():
        rng.shuffle(cluster)

    clients_indices = [[] for _ in range(n_clients)]        
    clients_counts = np.zeros((n_clusters, n_clients), dtype=np.int64)  # number of samples by client from each cluster
    
    for cluster_id in range(n_clusters):
            weights = np.random.dirichlet(alpha=alpha * np.ones(n_clients))
            clients_counts[cluster_id] = np.random.multinomial(clusters_sizes[cluster_id], weights)

    
    clients_counts = np.cumsum(clients_counts, axis=1)

    for cluster_id in range(n_clusters):
        cluster_split = split_list_by_indices(clusters[cluster_id], clients_counts[cluster_id])

        for client_id, indices in enumerate(cluster_split):
            # clients_indices[client_id] += list(indices)
            clients_indices[client_id] += indices

                    
    return clients_indices


def noniid_partition(generator):
    print(generator)
    labels = np.unique(generator.train_data.y)
    clients_indices = by_labels_non_iid_split(generator.train_data, labels.shape[0], generator.num_clients, labels.shape[0], generator.skewness, frac=1, seed=generator.seed)
    return clients_indices    


def iid_partition(generator):
    print(generator)
    labels = np.unique(generator.train_data.y)
    local_datas = [[] for _ in range(generator.num_clients)]
    for label in labels:
        permutation = np.random.permutation(np.where(generator.train_data.y == label)[0])
        split = np.array_split(permutation, generator.num_clients)
        for i, idxs in enumerate(split):
            local_datas[i] += idxs.tolist()
    return local_datas
    

class TaskGen(DefaultTaskGen):
    def __init__(self, dist_id, num_clients=1, skewness=0.5, local_hld_rate=0.0, seed=0,
                 percentages=None, missing=False, modal_equality=False, modal_missing_case3=False,
                 modal_missing_case4=False, imputation=False, sample_missing_ratio=0.5,
                 client_visible_modalities=8):
        super(TaskGen, self).__init__(benchmark='ptbxl_classification_lm',
                                      dist_id=dist_id,
                                      num_clients=num_clients,
                                      skewness=skewness,
                                      rawdata_path='./benchmark/RAW_DATA/PTBXL',
                                      local_hld_rate=local_hld_rate,
                                      seed = seed
                                      )
        if self.dist_id == 1:
            self.partition = noniid_partition
        else:
            self.partition = iid_partition
        # self.local_holdout = local_holdout
        self.num_classes = 5
        self.save_task = save_task
        self.visualize = self.visualize_by_class
        self.sample_missing_ratio = float(max(0.0, min(1.0, sample_missing_ratio)))
        self.client_visible_modalities = int(client_visible_modalities)
        self.source_dict = {
            'class_path': 'benchmark.ptbxl_classification_lm.dataset',
            'class_name': 'PTBXLReduceDataset',
            'train_args': {
                'root': '"'+self.rawdata_path+'"',
                'download': 'True',
                'standard_scaler': 'True',
                'train':'True',
                'sample_missing_ratio': str(self.sample_missing_ratio),
            },
            'test_args': {
                'root': '"'+self.rawdata_path+'"',
                'download': 'True',
                'standard_scaler': 'True',
                'train': 'False',
                'sample_missing_ratio': str(self.sample_missing_ratio),
            },
            'imputation': imputation
        }
        
        self.missing = missing
        self.modal_equality = modal_equality
        self.modal_missing_case3 = modal_missing_case3
        self.modal_missing_case4 = modal_missing_case4
        self.specific_training_leads = None
        self.local_holdout_rate = 0.1

        if self.num_clients == 3:
            # Client 1: 缂?0,1,2,3 -> 鎷ユ湁 4-11
            # Client 2: 缂?4,5,6,7 -> 鎷ユ湁 0-3, 8-11
            # Client 3: 缂?8,9,10,11 -> 鎷ユ湁 0-7
            self.specific_training_leads = [
                list(range(4, 12)),
                list(range(0, 4)) + list(range(8, 12)),
                list(range(0, 8))
            ]
        else:
            # Each client only trains on a configurable number of modalities.
            keep = max(1, min(12, self.client_visible_modalities))
            self.specific_training_leads = []
            rng = np.random.RandomState(self.seed)
            for _ in range(self.num_clients):
                kept = sorted(rng.choice(12, keep, replace=False).tolist())
                self.specific_training_leads.append(kept)

        print(f"[TaskGen] client_visible_modalities={len(self.specific_training_leads[0])} (unseen={12-len(self.specific_training_leads[0])})")
        for cid, leads in enumerate(self.specific_training_leads):
            print(f"[TaskGen] Client{cid} visible modalities: {leads}")
        self.taskname = self.taskname + '_full_modal_local_missing'
        self.taskpath = os.path.join(self.task_rootpath, self.taskname)

    def load_data(self):
        self.train_data = PTBXLReduceDataset(
            root=self.rawdata_path,
            download=True,
            standard_scaler=True,
            train=True, 
        )
        
        self.test_data = PTBXLReduceDataset(
            root=self.rawdata_path,
            download=True,
            standard_scaler=True,
            train=False,
        )
    
class TaskCalculator(ClassificationCalculator):
    def __init__(self, device, optimizer_name='sgd'):
        super(TaskCalculator, self).__init__(device, optimizer_name)
        self.n_leads = 12
        self.DataLoader = DataLoader

    def train_one_step(self, model, data, leads,
                       quantile=0.75, matrix=None, contrastive_weight=0.2):
        """
        :param model: the model to train
        :param data: the training dataset
        :param matrix: weight for adjacency matrix A
        :return: dict of train-one-step's result, which should at least contains the key 'loss'
        """
        tdata = self.data_to_device(data)
        _, loss, outputs = model(
            tdata[0],
            tdata[-1],
            leads,
            quantile=quantile,
            matrix=matrix,
            contrastive_weight=contrastive_weight,
        )
        return {'loss': loss, 'outputs': outputs}

    @torch.no_grad()
    def test(self, model, dataset, leads, 
             quantile=0.75, batch_size=64, num_workers=0, 
             matrix=None, contrastive_weight=0.2):
        """
        Metric = [mean_accuracy, mean_loss]
        :param model:
        :param dataset:
        :param batch_size:
        :return: [mean_accuracy, mean_loss]
        """
        model.eval()
        if batch_size==-1:batch_size=len(dataset)
        data_loader = self.get_data_loader(dataset, batch_size=batch_size, num_workers=num_workers)
        total_loss = 0.0
        labels = list()
        predicts = list()
        for batch_id, batch_data in enumerate(data_loader):
            batch_data = self.data_to_device(batch_data)
            labels.extend(batch_data[1].cpu().tolist())
            
            loss_leads, loss, outputs = model(batch_data[0], batch_data[-1],
                                              leads, quantile=quantile, matrix=matrix,
                                              contrastive_weight=contrastive_weight)
            total_loss += loss.item() * len(batch_data[-1])
            cur_predict = torch.softmax(outputs, dim=1)
            predicts.extend(torch.argmax(torch.softmax(outputs, dim=1), dim=1).cpu().tolist())
        
        labels = np.array(labels)
        predicts = np.array(predicts)
        accuracy = accuracy_score(labels, predicts)
        return {
            'loss': total_loss / len(dataset),
            'acc': accuracy
        }
        
    @torch.no_grad()
    def pretrain_test(self, model, dataset, leads, quantile=0.75, batch_size=64, num_workers=0):
        """
        Metric = [mean_accuracy, mean_loss]
        :param model:
        :param dataset:
        :param batch_size:
        :return: [mean_accuracy, mean_loss]
        """
        model.eval()
        if batch_size==-1:batch_size=len(dataset)
        data_loader = self.get_data_loader(dataset, batch_size=batch_size, num_workers=num_workers)
        total_loss = 0.0
        labels = list()
        predicts = list()
        for batch_id, batch_data in enumerate(data_loader):
            batch_data = self.data_to_device(batch_data)
            labels.extend(batch_data[1].cpu().tolist())
            
            _, loss, _ = model(batch_data[0], batch_data[-1], leads, quantile=quantile)
            total_loss += loss.item() * len(batch_data[-1])

        return {
            'loss': total_loss / len(dataset),

        }
        
    @torch.no_grad()
    def evaluate(self, model, dataset, leads, batch_size=64, num_workers=0):
        """
        Metric = [mean_accuracy, mean_loss]
        :param model:
        :param dataset:
        :param batch_size:
        :return: [mean_accuracy, mean_loss]
        """
        model.eval()
        if batch_size==-1:batch_size=len(dataset)
        data_loader = self.get_data_loader(dataset, batch_size=batch_size, num_workers=num_workers)

        # total_loss = []
        labels = list()
        predicts = list()
        total_loss = list()
        for batch_id, batch_data in enumerate(data_loader):
            batch_data = self.data_to_device(batch_data)
            labels.extend(batch_data[1].cpu().tolist())
            logvar, loss_, outputs = model(batch_data[0], batch_data[-1], leads)
            logvar = torch.exp(torch.tensor(logvar).to(outputs.device))
            
            total_loss.append(torch.exp(logvar))
            
        
        loss_eval = torch.mean(torch.cat(total_loss, dim=0))
        return loss_eval


    @torch.no_grad()
    def server_test(self, model, dataset, leads, batch_size=64, num_workers=0):
        """
        Metric = [mean_accuracy, mean_loss]
        :param model:
        :param dataset:
        :param batch_size:
        :return: [mean_accuracy, mean_loss]
        """
        model.eval()
        if batch_size==-1:batch_size=len(dataset)
        data_loader = self.get_data_loader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
        result = dict() 
        for test_combi_index in range(len(leads)):
            total_loss = 0.0
            labels = list()
            predicts = list()   
            # loss_each_modal = [[] for i in range(self.n_leads)]
            # loss_each_modal = [0]*12
            for batch_id, batch_data in enumerate(data_loader):
                batch_data = self.data_to_device(batch_data)
                labels.extend(batch_data[1].cpu().tolist())
                loss_leads, loss, outputs = model(batch_data[0], batch_data[-1], leads[test_combi_index])
                # for i in range(self.n_leads):
                #     loss_each_modal[i] += loss_leads[i] * len(batch_data[-1])
                total_loss += loss.item() * len(batch_data[-1])
                predicts.extend(torch.argmax(torch.softmax(outputs, dim=1), dim=1).cpu().tolist())
            labels = np.array(labels)
            predicts = np.array(predicts)
            accuracy = accuracy_score(labels, predicts)
            # for i in range(self.n_leads):
            #     result['loss_modal_combi'+str(test_combi_index+1)+'_modal'+str(i+1)] = loss_each_modal[i] / len(dataset)
            result['loss'+str(test_combi_index+1)] = total_loss / len(dataset)
            result['acc'+str(test_combi_index+1)] = accuracy
        # return {
        #     'loss': total_loss / len(dataset),
        #     'acc': accuracy
        # }
        return result
    
    @torch.no_grad()
    def lm_server_test(self, model, dataset, leads, rates,
                       quantile=0.75, visualize=False, matrix=None,
                       batch_size=64, num_workers=0, ps=0.0,
                       pm=0.0, contrastive_weight=0.2, local_missing_leads=None):
        """
        Metric = [mean_accuracy, mean_loss]
        :param model:
        :param dataset:
        :param batch_size:
        :return: [mean_accuracy, mean_loss]
        """
        model.eval()
        if batch_size==-1:batch_size=len(dataset)

        result = dict() 

        for index in range(len(rates)):
            dataset.local_missing_setup(leads[0], ps=rates[index][0], pm=rates[index][1],
                                        local_missing_leads=local_missing_leads)
            # dataset.save_data_dist(pm=rates[index][1], ps=rates[index][0])
            data_loader = self.get_data_loader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            
            total_loss = 0.0
            labels = list()
            predicts = list()   
            for batch_id, batch_data in enumerate(data_loader):
                batch_data = self.data_to_device(batch_data)
                labels.extend(batch_data[1].cpu().tolist())
                test_unc, loss, outputs = model(batch_data[0], batch_data[-1], 
                                                leads[0], quantile=quantile, 
                                                visualize=visualize, matrix=matrix,
                                                contrastive_weight=contrastive_weight)
                
                total_loss += loss.item() * len(batch_data[-1])
                predicts.extend(torch.argmax(torch.softmax(outputs, dim=1), dim=1).cpu().tolist())
            labels = np.array(labels)
            predicts = np.array(predicts)
            accuracy = accuracy_score(labels, predicts)

            # --- Per-class accuracy ---
            from sklearn.metrics import confusion_matrix
            unique_cls = sorted(set(labels.tolist()) | set(predicts.tolist()))
            cm = confusion_matrix(labels, predicts, labels=unique_cls)
            per_cls_acc = cm.diagonal() / cm.sum(axis=1).clip(min=1)
            parts = [f"c{int(c)}={a:.3f}({int(n)})"
                     for c, a, n in zip(unique_cls, per_cls_acc, cm.sum(axis=1))]
            print(f"  [Per-class] {' | '.join(parts)}  overall={accuracy:.4f}")

            if visualize:
                stored_features = {}
                for k, v in model.stored_features.items():
                    if len(v) > 0:
                        stored_features[k] = torch.cat(v, dim=0).cpu().numpy()
                np.save('./visualize_notebooks/features.npy', stored_features)
            
            result['loss'+str(index+1)] = total_loss / len(dataset)
            result['acc'+str(index+1)] = accuracy
        # return {
        #     'loss': total_loss / len(dataset),
        #     'acc': accuracy
        # }
        return result
    
    @torch.no_grad()
    def lm_server_test_all_cases(self, model, dataset, leads, rates,
                       quantile=0.75, visualize=False, matrix=None,
                       batch_size=64, num_workers=0, ps=0.0,
                       pm=0.0, contrastive_weight=0.2, local_missing_leads=None):
        """
        Metric = [mean_accuracy, mean_loss]
        :param model:
        :param dataset:
        :param batch_size:
        :return: [mean_accuracy, mean_loss]
        """
        model.eval()
        model.to(self.device)
        if batch_size==-1:batch_size=len(dataset)
        dataset = ClientSubset(dataset, range(len(dataset)), 10000)
        
        result = dict() 
        # for test_combi_index in range(len(leads)):
        #     dataset.local_missing_setup(leads[test_combi_index], ps=ps, pm=pm)
        stored_features = [] 
        for lead in tqdm(leads, total=len(leads)):
            dataset.global_missing_setup(lead, ps=0.0, pm=0.0)
            # dataset.save_data_dist(pm=rates[index][1], ps=rates[index][0])
            data_loader = self.get_data_loader(dataset, batch_size=1, shuffle=False, num_workers=num_workers)
            full_lead = list(range(12))
            for batch_data in data_loader:
 
                batch_data[0] = batch_data[0].float().to(self.device)
                batch_data[1] = batch_data[1].to(self.device)
                # batch_data = self.data_to_device(batch_data)
                # labels.extend(batch_data[1].cpu().tolist())
                visualize_feature, loss, outputs = model(batch_data[0], batch_data[-1], 
                                                full_lead, quantile=quantile, 
                                                visualize=visualize, matrix=matrix,
                                                contrastive_weight=contrastive_weight, examine_emb=True)
                stored_features.append(visualize_feature.cpu().numpy())
                break
        stored_features = np.concatenate(stored_features, axis=0)
        print(stored_features.shape)
            
        np.save('./visualize_notebooks/motivation_features.npy', stored_features)
        return result
    
    @torch.no_grad()
    def lm_pretrain_server_test(self, model, dataset, leads, quantile=0.75,
                                visualize=False, matrix=None, batch_size=64,
                                num_workers=0, ps=0.0, pm=0.0, contrastive_weight=0.2):
        """
        Metric = [mean_accuracy, mean_loss]
        :param model:
        :param dataset:
        :param batch_size:
        :return: [mean_accuracy, mean_loss]
        """
        model.eval()
        if batch_size==-1:batch_size=len(dataset)
        dataset = ClientSubset(dataset, range(len(dataset)), 10000)
        
        result = dict() 
        for test_combi_index in range(len(leads)):
            dataset.local_missing_setup(leads[test_combi_index], ps=ps, pm=pm)
            data_loader = self.get_data_loader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            
            total_loss = 0.0
            labels = list()

            for batch_id, batch_data in enumerate(data_loader):
                batch_data = self.data_to_device(batch_data)
                labels.extend(batch_data[1].cpu().tolist())
                _, loss, _ = model(batch_data[0], batch_data[-1], leads[test_combi_index],
                                                quantile=quantile, visualize=visualize,
                                                matrix=matrix, contrastive_weight=contrastive_weight)
                
                total_loss += loss.item() * len(batch_data[-1])
            
            if visualize: 
                stored_features = model.stored_features
                stored_features = {
                    k: torch.cat(v, dim=0).cpu().numpy() for k, v in stored_features.items()}
                np.save('./visualize_notebooks/features.npy', stored_features)
            
            result['loss'+str(test_combi_index+1)] = total_loss / len(dataset)

        return result


    @torch.no_grad()
    def full_modal_server_test(self, model, dataset, leads, batch_size=64, num_workers=0):
        """
        Metric = [mean_accuracy, mean_loss]
        :param model:
        :param dataset:
        :param batch_size:
        :return: [mean_accuracy, mean_loss]
        """
        model.eval()
        if batch_size==-1:batch_size=len(dataset)
        data_loader = self.get_data_loader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        total_loss = 0.0
        labels = list()
        predicts = list()
        for batch_id, batch_data in enumerate(data_loader):
            batch_data = self.data_to_device(batch_data)
            labels.extend(batch_data[1].cpu().tolist())
            loss, outputs = model(batch_data[0], batch_data[-1], leads)
            total_loss += loss.item() * len(batch_data[-1])
            predicts.extend(torch.argmax(torch.softmax(outputs, dim=1), dim=1).cpu().tolist())
        labels = np.array(labels)
        predicts = np.array(predicts)
        accuracy = accuracy_score(labels, predicts)
        return {
            'loss': total_loss / len(dataset),
            'acc': accuracy
        }
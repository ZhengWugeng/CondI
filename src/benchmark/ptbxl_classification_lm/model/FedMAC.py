from typing import Any
import torch
import os
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import random

# from torch_geometric import nn as gnn
# import torch_geometric.utils as gutils

from utils.fmodule import FModule
# import torch_geometric.typing
# from torch_geometric.utils import coalesce, cumsum


def dense_to_sparse(adj, mask = None):
    r"""Converts a dense adjacency matrix to a sparse adjacency matrix defined
    by edge indices and edge attributes.

    Args:
        adj (torch.Tensor): The dense adjacency matrix of shape
            :obj:`[num_nodes, num_nodes]` or
            :obj:`[batch_size, num_nodes, num_nodes]`.
        mask (torch.Tensor, optional): A boolean tensor of shape
            :obj:`[batch_size, num_nodes]` holding information about which
            nodes are in each example are valid. (default: :obj:`None`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """
    if adj.dim() < 2 or adj.dim() > 3:
        raise ValueError(f"Dense adjacency matrix 'adj' must be two- or "
                         f"three-dimensional (got {adj.dim()} dimensions)")

    if mask is not None and adj.dim() == 2:
        warnings.warn("Mask should not be provided in case the dense "
                      "adjacency matrix is two-dimensional")
        mask = None

    if mask is not None and mask.dim() != 2:
        raise ValueError(f"Mask must be two-dimensional "
                         f"(got {mask.dim()} dimensions)")

    if mask is not None and adj.size(-2) != adj.size(-1):
        raise ValueError(f"Mask is only supported on quadratic adjacency "
                         f"matrices (got [*, {adj.size(-2)}, {adj.size(-1)}])")

    if adj.dim() == 2:
        edge_index = adj.nonzero().t()
        edge_attr = adj[edge_index[0], edge_index[1]]
        return edge_index, edge_attr
    else:
        flatten_adj = adj.view(-1, adj.size(-1))
        if mask is not None:
            flatten_adj = flatten_adj[mask.view(-1)]
        edge_index = flatten_adj.nonzero().t()
        edge_attr = flatten_adj[edge_index[0], edge_index[1]]

        if mask is None:
            offset = torch.arange(
                start=0,
                end=adj.size(0) * adj.size(2),
                step=adj.size(2),
                device=adj.device,
            )
            offset = offset.repeat_interleave(adj.size(1))
        else:
            count = mask.sum(dim=-1)
            offset = cumsum(count)[:-1]
            offset = offset.repeat_interleave(count)

        edge_index[1] += offset[edge_index[0]]

        return edge_index, edge_attr
    
def sampling_from_mat(mat, n_samples):
    p = mat.clone().detach()
    index = p.multinomial(num_samples=n_samples, replacement=False)
    mask = torch.zeros_like(mat)
    mask[index] = 1.0
    return mat * mask
    

def adjacency_matrix_to_edge_indices(adj_matrix):
    edge_list = []
    num_vertices = len(adj_matrix)
    # Iterate over each element in the adjacency matrix
    for i in range(num_vertices):
        for j in range(num_vertices):  # Change 'range(i, num_vertices)' to 'range(num_vertices)' for directed graphs
            if adj_matrix[i][j] != 0:
                edge_list.append((i, j))  # Include weight if necessary
    
    edge_indices = np.zeros((2, len(edge_list)), dtype=np.int32)
    for i, pair in enumerate(edge_list):
        edge_indices[:, i] = np.array(pair)
        
    return edge_indices

class InceptionBlock1D(FModule):
    def __init__(self, input_channels):
        super(InceptionBlock1D, self).__init__()
        self.input_channels = input_channels
        self.bottleneck = nn.Conv1d(self.input_channels, 32, kernel_size=1, stride=1, bias=False)
        self.convs_conv1 = nn.Conv1d(32, 32, kernel_size=39, stride=1, padding=19, bias=False)
        self.convs_conv2 = nn.Conv1d(32, 32, kernel_size=19, stride=1, padding=9, bias=False)
        self.convs_conv3 = nn.Conv1d(32, 32, kernel_size=9, stride=1, padding=4, bias=False)
        self.convbottle_maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
        self.convbottle_conv = nn.Conv1d(self.input_channels, 32, kernel_size=1, stride=1, bias=False)
        self.bnrelu_bn = nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bnrelu_relu = nn.ReLU()
    def forward(self, x):
        bottled = self.bottleneck(x)
        y = torch.cat([
            self.convs_conv1(bottled),
            self.convs_conv2(bottled),
            self.convs_conv3(bottled),
            self.convbottle_conv(self.convbottle_maxpool(x))
        ], dim=1)
        out = self.bnrelu_relu(self.bnrelu_bn(y))
        return out

class Shortcut1D(FModule):
    def __init__(self, input_channels):
        super(Shortcut1D, self).__init__()
        self.input_channels = input_channels
        self.act_fn = nn.ReLU(inplace=True)
        self.conv = nn.Conv1d(self.input_channels, 128, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    def forward(self, inp, out):
        return self.act_fn(out + self.bn(self.conv(inp)))
        
class Inception1DBase(FModule):
    def __init__(self, input_channels=1):
        super(Inception1DBase, self).__init__()
        self.input_channels = input_channels
        # inception backbone
        self.inceptionbackbone_1 = InceptionBlock1D(input_channels=self.input_channels)
        self.inceptionbackbone_2 = InceptionBlock1D(input_channels=128)
        self.inceptionbackbone_3 = InceptionBlock1D(input_channels=128)
        self.inceptionbackbone_4 = InceptionBlock1D(input_channels=128)
        self.inceptionbackbone_5 = InceptionBlock1D(input_channels=128)
        self.inceptionbackbone_6 = InceptionBlock1D(input_channels=128)
        # shortcut
        self.shortcut_1 = Shortcut1D(input_channels=self.input_channels)
        self.shortcut_2 = Shortcut1D(input_channels=128)
        # pooling
        self.ap = nn.AdaptiveAvgPool1d(output_size=1)
        self.mp = nn.AdaptiveMaxPool1d(output_size=1)
        # flatten
        self.flatten = nn.Flatten()
        self.bn_1 = nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.dropout_1 = nn.Dropout(p=0.25, inplace=False)
        self.ln_1 = nn.Linear(256, 128, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.bn_2 = nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.dropout_2 = nn.Dropout(p=0.5, inplace=False)
        # self.ln_2 = nn.Linear(128, 71, bias=True)
    def forward(self, x):
        # inception backbone
        input_res = x
        x = self.inceptionbackbone_1(x)
        x = self.inceptionbackbone_2(x)
        x = self.inceptionbackbone_3(x)
        x = self.shortcut_1(input_res, x)
        input_res = x.clone()
        x = self.inceptionbackbone_4(x)
        x = self.inceptionbackbone_5(x)
        x = self.inceptionbackbone_6(x)
        x = self.shortcut_2(input_res, x)
        # input_res = x.clone()
        # head
        x = torch.cat([self.mp(x), self.ap(x)], dim=1)
        x = self.flatten(x)
        x = self.bn_1(x)
        x = self.dropout_1(x)
        x = self.ln_1(x)
        x = self.relu(x)
        x = self.bn_2(x)
        x = self.dropout_2(x)
        return x
        
class ContrastiveWeight(nn.Module):

    def __init__(self):
        super(ContrastiveWeight, self).__init__()
        self.temperature = 0.2

        self.bce = torch.nn.BCELoss()
        self.softmax = torch.nn.Softmax(dim=-1)
        self.log_softmax = torch.nn.LogSoftmax(dim=-1)
        self.kl = torch.nn.KLDivLoss(reduction='batchmean')
        self.num_leads = 12

    def get_positive_and_negative_mask(self, similarity_matrix, cur_batch_size, batch_emb_mask=None):
        # batch_emb_mask = missing_mask -> 1- to get having mask 
        diag = np.eye(cur_batch_size)
        mask = torch.from_numpy(diag)
        mask = mask.type(torch.bool)

        oral_batch_size = cur_batch_size // 12

        positives_mask = np.zeros(similarity_matrix.size())
        for i in range(12):
            ll = np.eye(cur_batch_size, cur_batch_size, k=oral_batch_size * i)
            lr = np.eye(cur_batch_size, cur_batch_size, k=-oral_batch_size * i)
            positives_mask += ll
            positives_mask += lr
            

        positives_mask = torch.from_numpy(positives_mask).to(similarity_matrix.device)
        if batch_emb_mask is not None:
            batch_emb_mask = 1 - batch_emb_mask
            positives_mask = (positives_mask * batch_emb_mask)*(batch_emb_mask.T)
        positives_mask[mask] = 0

        negatives_mask = 1 - positives_mask
        negatives_mask[mask] = 0

        return positives_mask.type(torch.bool), negatives_mask.type(torch.bool)

    def forward(self, batch_emb_om, batch_emb_mask=None):
        cur_batch_shape = batch_emb_om.shape

        # get similarity matrix among mask samples
        norm_emb = F.normalize(batch_emb_om, dim=1)
        similarity_matrix = torch.matmul(norm_emb, norm_emb.transpose(0, 1))

        # get positives and negatives similarity
        positives_mask, negatives_mask = self.get_positive_and_negative_mask(similarity_matrix, cur_batch_shape[0], batch_emb_mask)

        positives = similarity_matrix[positives_mask].view(cur_batch_shape[0], -1)
        negatives = similarity_matrix[negatives_mask].view(cur_batch_shape[0], -1)

        # generate predict and target probability distributions matrix
        logits = torch.cat((positives, negatives), dim=-1)
        y_true = torch.cat(
            (torch.ones(cur_batch_shape[0], positives.shape[-1]), torch.zeros(cur_batch_shape[0], negatives.shape[-1])),
            dim=-1).to(batch_emb_om.device).float()

        # multiple positives - KL divergence
        predict = self.log_softmax(logits / self.temperature)
        loss = self.kl(predict, y_true)

        return loss, similarity_matrix, logits, positives_mask
    
class AggregationRebuild(nn.Module):

    def __init__(self):
        super(AggregationRebuild, self).__init__()
        self.temperature = 0.2
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mse = torch.nn.MSELoss()

    def forward(self, similarity_matrix, batch_emb_om):
        cur_batch_shape = batch_emb_om.shape

        # get the weight among (oral, oral's masks, others, others' masks)
        similarity_matrix /= self.temperature

        similarity_matrix = similarity_matrix - torch.eye(cur_batch_shape[0]).to(
            similarity_matrix.device).float() * 1e12
        rebuild_weight_matrix = self.softmax(similarity_matrix) # (BxM)x(BxM)

        batch_emb_om = batch_emb_om.reshape(cur_batch_shape[0], -1) # (BxM)xD

        # generate the rebuilt batch embedding (oral, others, oral's masks, others' masks)
        rebuild_batch_emb = torch.matmul(rebuild_weight_matrix, batch_emb_om)

        # get oral' rebuilt batch embedding
        rebuild_oral_batch_emb = rebuild_batch_emb.reshape(cur_batch_shape[0], cur_batch_shape[1], -1)

        return rebuild_weight_matrix, rebuild_oral_batch_emb

class SimilarityProjector(FModule):
    def __init__(self):
        super(SimilarityProjector, self).__init__()
        self.ln = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
    
    def forward(self, x):
        return self.ln(x)
    
class ModalityEncoder(FModule):
    def __init__(self, input_channels=1):
        super(ModalityEncoder, self).__init__()
        self.input_channels = input_channels
        self.bottleneck = nn.Conv1d(self.input_channels, 32, kernel_size=1, stride=1, bias=False)
        self.convs_conv1 = nn.Conv1d(32, 32, kernel_size=39, stride=1, padding=19, bias=False)
        self.convs_conv2 = nn.Conv1d(32, 32, kernel_size=39, stride=1, padding=19, bias=False)
        self.finalizer = nn.Sequential(
            nn.Conv1d(32, 16, kernel_size=39, stride=1, padding=19, bias=False), 
            nn.ReLU(), 
            nn.Conv1d(16, 1, kernel_size=39, stride=1, padding=19, bias=True)
        )
    def forward(self, x):
        shortcut = x.clone()
        x = self.bottleneck(x)
        x = F.relu(self.convs_conv1(x))
        x = F.relu(self.convs_conv2(x))
        
        out = self.finalizer(x) + shortcut
        
        return out

class GlobalFuser(FModule):
    def __init__(self):
        super(GlobalFuser, self).__init__()
        self.masking = nn.Sequential(
            nn.Conv2d(2, 32, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(32, 8, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(8, 1, 3, 1, 1))
        # self.finalizer = nn.Conv1d(1, 1, 1, 1, 0)
    
    def forward(self, old, new):
        old = old.unsqueeze(1)
        new = new.unsqueeze(1)
        all_feat = torch.cat([old, new], dim=1)
        mask = torch.sigmoid(self.masking(all_feat))
        all_feat = old * mask + new * (1-mask)
        # all_feat = self.finalizer(all_feat)
        return all_feat.squeeze(1), mask.squeeze(1)
    
class FinalGCN(FModule):
    def __init__(self):
        super(FinalGCN, self).__init__()
        self.gcn = gnn.models.GCN(in_channels=128, 
                                     hidden_channels=32,
                                     out_channels=128, 
                                     num_layers = 3, dropout= 0.2,)
        
    def forward(self, x, edge_index, edge_weight=None):
        x = self.gcn(x, edge_index, edge_weight)
        out = torch.mean(x, dim=0, keepdim=True)
        
        return out, x

class ImputationEmbedding(FModule):
    def __init__(self):
        super(ImputationEmbedding, self).__init__()
        self.hidden_dim = 128
        self.imputation = nn.Parameter(torch.randn(1, self.hidden_dim))
    
    def forward(self, x):
        return self.imputation
    
class Classifier(FModule):
    def __init__(self):
        super(Classifier, self).__init__()
        self.ln1 = nn.Linear(128*12, 128, True)
        self.ln2 = nn.Linear(128, 5, True)
    
    def forward(self, x):       #()
        return self.ln2(F.relu(self.ln1(x)))
    
class Model(FModule):
    def __init__(self):
        super(Model, self).__init__()
        self.n_leads = 12
        self.hidden_dim = 128
        self.feature_extractors = nn.ModuleList(
            [Inception1DBase(input_channels=1)]
        )
        # self.relation_embedders = nn.ModuleList()
        self.sim_projectors = nn.ModuleList()
        self.embeddings = nn.ModuleList()
        self.pre_extractors = nn.ModuleList() 
        for i in range(self.n_leads):
            self.sim_projectors.append(SimilarityProjector())
            self.embeddings.append(ImputationEmbedding())
            self.pre_extractors.append(ModalityEncoder(input_channels=1))
        self.classifier = Classifier()
        self.criterion = nn.CrossEntropyLoss()
        
        self.contrastive_weight = ContrastiveWeight() 
        self.rebuilder = AggregationRebuild()
        self.global_fuser = GlobalFuser()
        # self.graph_conv = FinalGCN()
        
        self.stored_features = {
            'final_feature': [],
            'before_fuse': [],
            'after_rebuild': [],
            'after_fuse': [],
            'label': [],
            'missing_mask': [],
            'alpha_old': []
        }
        
        
    def forward(self, x, y, leads, quantile=0.75, visualize=False, matrix=None, contrastive_weight=0.1, examine_emb=False): # x: B x 12 x C
        batch_size = y.shape[0]
        features = torch.zeros(size=(batch_size, 12, self.hidden_dim), dtype=torch.float32, device=y.device)
        # sim_features_inter = torch.zeros(size=(batch_size*12, self.hidden_dim), dtype=torch.float32, device=y.device)
        sim_features_intra = torch.zeros(size=(batch_size+1, 12, self.hidden_dim), dtype=torch.float32, device=y.device)    # +1 for embedding
        ori_features_intra = torch.zeros(size=(batch_size+1, 12, self.hidden_dim), dtype=torch.float32, device=y.device)
        
        total_lead_ind = [*range(12)]
        leads_features = []
        feature_extractor_outputs = torch.zeros(size=(batch_size, self.hidden_dim), dtype=torch.float32, device=y.device)
        
        missing_mask_total = list()
                
        for lead in total_lead_ind:    
            
            # filter indices of missing modality
            pseudo_x = x[..., lead, 0].clone()
            
            pseudo_id = pseudo_x == -1
            missing_indices = pseudo_id.nonzero()   # Nx[id1,idx] -> Nx2: N is num of missing points
            missing_mask = torch.zeros([batch_size])
            for pair in missing_indices:
                missing_mask[pair] = 1
                
            missing_mask = missing_mask.reshape(-1, 1).to(y.device)
            missing_mask_total.append(missing_mask)
            
            # pre-extract
            feature = self.pre_extractors[lead](x[:, lead, :].view(batch_size, 1, -1))  # Bx1xC
            feature = feature * (1-missing_mask.reshape(-1, 1, 1))
            feature = self.feature_extractors[0](feature)
            
            # feature = self.feature_extractors[0](x[:, lead, :].view(batch_size, 1, -1))  # B x 1 x C
            feature = feature * (1-missing_mask) + missing_mask * self.embeddings[lead](x).repeat(batch_size, 1)
            
            features[:,lead:(lead+1), :] = feature.unsqueeze(1)
            
            dense_feature = self.sim_projectors[lead](torch.cat([feature, self.embeddings[lead](x)]))
            # sim_features_inter[lead*batch_size:(lead+1)*batch_size, :] = dense_feature
            sim_features_intra[:, lead: (lead+1), :] = dense_feature.unsqueeze(1)
            ori_features_intra[:, lead: (lead+1), :] = torch.cat([feature.unsqueeze(1), self.embeddings[lead](x).unsqueeze(1)])
        
        if examine_emb:
            features = features.clone().detach().requires_grad_()
        
        missing_mask_total = torch.cat(missing_mask_total, dim=1)   # BxM
        if visualize:
            self.stored_features['missing_mask'].append(missing_mask_total.cpu().detach())
        
        sim_features_intra = sim_features_intra.permute(1,0,2).reshape(-1, self.hidden_dim)  # BxMxD
        ori_features_intra = ori_features_intra.permute(1,0,2).reshape(-1, self.hidden_dim)
        
        # intra - inter contrastive loss
        # inter_loss, inter_similarity_matrix, _, _ = self.contrastive_weight(sim_features_inter)
        intra_loss, intra_similarity_matrix, _, _ = self.contrastive_weight(sim_features_intra)
            
        ori_intra_loss, ori_intra_similarity_matrx, _, _ = self.contrastive_weight(ori_features_intra)
        contrastive_loss = (intra_loss + ori_intra_loss) * 0.5
        
        _, rebuilt_features = self.rebuilder(intra_similarity_matrix[:batch_size*12, :batch_size*12],     # skip last 12 embeddings
                                             features.reshape(-1, self.hidden_dim))
        # rebuilt_features = rebuilt_features.reshape(batch_size, -1)
        
        if visualize:
            self.stored_features['before_fuse'].append(features.cpu().detach())
            self.stored_features['after_rebuild'].append(rebuilt_features.cpu().detach().reshape(batch_size, -1, self.hidden_dim))
        
        rebuilt_features, alpha_old = self.global_fuser(
            features.reshape(batch_size, -1, self.hidden_dim), 
            rebuilt_features.reshape(batch_size, -1, self.hidden_dim))  # BxMxD
        
        if visualize:
            self.stored_features['after_fuse'].append(rebuilt_features.cpu().detach())
            self.stored_features['alpha_old'].append(alpha_old.cpu().detach())
        
        if visualize:   # store feature to visualize later
            self.stored_features['final_feature'].append(rebuilt_features.detach().cpu().reshape(batch_size, -1))
            self.stored_features['label'].append(y.detach().cpu())
        
        outputs = self.classifier(rebuilt_features.reshape(batch_size, -1))
        loss = self.criterion(outputs, y.type(torch.int64))
        loss = loss + contrastive_weight*contrastive_loss

        loss_leads = 0
        return features, loss, outputs

from typing import Any
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import random
from utils.fmodule import FModule


def init_xavier(param):
    nn.init.kaiming_normal_(param.data)

def initialize_weights_xavier(lstm):
    for name, param in lstm.named_parameters():
        if 'weight_ih' in name:
            init_xavier(param)
        elif 'weight_hh' in name:
            init_xavier(param)
        elif 'bias' in name:
            # Initialize biases to 0.1 to ensure that forget gates are initially open
            nn.init.constant_(param.data, 0.1)


class TFEM(FModule):
    """Temporal feature extraction module (per modality)"""
    def __init__(self):
        super(TFEM, self).__init__()
        self.input_channels = 1
        self.extract = nn.Sequential(
            nn.Linear(1, 256), nn.LeakyReLU(0.1), 
            nn.Linear(256, 512), nn.LeakyReLU(0.1), 
            nn.Linear(512, 128)
        )
        self.lstm1 = nn.LSTM(input_size=128, hidden_size=128,
                             num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=128,
                             num_layers=1, batch_first=True)
        
        initialize_weights_xavier(self.lstm1)
        initialize_weights_xavier(self.lstm2)
        
    def forward(self, x, hn=None):
        x = self.extract(x)
        x, (h_t, c_t) = self.lstm1(x)
        x, (h_t, c_t) = self.lstm2(x, (h_t, c_t))
        return h_t[-1,...]
        
        return h_t
    
class FMSM(FModule):
    """Fine-grained multimodal scale module"""
    def __init__(self):
        super(FMSM, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(12, 64, 3, 1, 1), nn.ReLU(), 
            nn.Conv1d(64, 12, 1, 1, 0))
        self.sigmoid = nn.Sigmoid()
        self.finalizer = nn.Sequential(
            nn.Conv1d(12, 64, 3, 1, 1), nn.ReLU(),
            nn.Conv1d(64, 128, 3, 1, 1), nn.ReLU(),
            nn.Conv1d(128, 12, 1, 1, 0))
    def forward(self, h):
        # h (BxMxd)
        masks = self.sigmoid(self.conv(h))
        # h = self.finalizer(h)
        h = h * masks
        
        return h
    
class CGFM(FModule):
    """Coarse-grained multimodal scale module"""
    def __init__(self):
        super(CGFM, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(12, 32, 3, 1, 1), nn.ReLU(),
            nn.Conv1d(32, 32, 3, 1, 1), nn.ReLU(),
            nn.Conv1d(32, 1, 1, 1, 0))
        self.sigmoid = nn.Sigmoid()
        self.act = nn.LeakyReLU(0.1)
    
    def forward(self, h):
        # h (BxMxd)
        reweight = self.sigmoid(self.conv(h))
        h = h * reweight
        h = torch.mean(h, dim=1)
        
        return h

class Classifier(FModule):
    def __init__(self):
        super(Classifier, self).__init__()
        self.ln1 = nn.Linear(128, 128, True)
        self.ln2 = nn.Linear(128, 5, True)
        # self.act = nn.LeakyReLU(0.1)
    
    def forward(self, x):       #()
        # x = F.relu(self.ln1(x))
        # x = self.ln2(x)
        x = self.ln2(x)
        return x
    
class Model(FModule):
    def __init__(self):
        super(Model, self).__init__()
        self.n_leads = 12
        self.hidden_dim = 128
        self.window_size = 250
        self.feature_extractors = nn.ModuleList([
            TFEM() for _ in range(self.n_leads)])
        # self.relation_embedders = nn.ModuleList()
        self.fmsm = FMSM()
        self.cgfm = CGFM()
        
        self.classifier = Classifier()
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, x, y, leads, quantile=0.75, visualize=False, matrix=None, contrastive_weight=None): # x: B x 12 x C
        batch_size = y.shape[0]
        total_lead_ind = range(self.n_leads)
        
        seq_len = x.size(2)
        num_windows = seq_len - self.window_size + 1
        features = torch.zeros(size=(batch_size*num_windows, 12, self.hidden_dim), dtype=torch.float32, device=y.device)
        
        for lead in total_lead_ind:
            
            pseudo_x = x[..., lead, 0].clone()
            
            pseudo_id = pseudo_x == -1
            missing_indices = pseudo_id.nonzero()
            missing_mask = torch.zeros([batch_size])
            for pair in missing_indices:
                missing_mask[pair] = 1
            missing_mask = missing_mask.reshape(-1, 1, 1).to(y.device) # B, 1
            
            # convert missing points to 0
            lead_x = x[:, lead:(lead+1), :].permute(0, 2, 1)   # B,D,L -> B,L,D (D=1)
            
            lead_x = [lead_x[:, i: i + self.window_size, :] for i in range(num_windows)] 
            lead_x = torch.stack(lead_x, dim=1).reshape(batch_size*num_windows, self.window_size, 1) # Bxnum_windowsxSxD -> (BxNW) x S x D
            
            h_t_final = self.feature_extractors[lead](lead_x)
            
            features[:, lead:(lead+1), :] = h_t_final.unsqueeze(1) # 1xBNWxH ~ last final state
            
        # features = torch.cat(features, dim=1)
        features = self.fmsm(features)   
        features = self.cgfm(features)
        
        # print(features)
        
        outputs = self.classifier(features.reshape(batch_size*num_windows, -1))
        
        expand_y = y.repeat(1, num_windows).reshape(-1)
        loss = self.criterion(outputs, expand_y.type(torch.int64))
        
        outputs = outputs.reshape(batch_size, num_windows, -1)[:, 0, :]
        
        loss_leads = 0
        
        return loss_leads, loss, outputs

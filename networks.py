import torch
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self, AE, AE_2, feature_dim, class_num, windowsize, Patch_channel):
        super(Network, self).__init__()
        self.AE = AE
        self.AE_2 = AE_2
        self.feature_dim = feature_dim
        self.cluster_num = class_num
        self.windowsize = windowsize
        self.Patch_channel = Patch_channel
        example_input = torch.randn(self.feature_dim, 1, self.Patch_channel, self.windowsize, self.windowsize).to('cuda')
        with torch.no_grad():
            rep_dim = AE(example_input).shape[1]
        self.rep_dim = rep_dim
        self.cluster_projector = nn.Sequential(
            nn.Linear(self.rep_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, self.cluster_num)
        )
        self.temperature = 0.5

    def forward(self, x_i, x_j):
        z_i = self.AE(x_i)
        z_j = self.AE_2(x_j)
        z_i_norm = F.normalize(z_i.to('cuda'), dim=1)
        z_j_norm = F.normalize(z_j.to('cuda'), dim=1)
        logits_i = self.cluster_projector(z_i_norm)
        logits_j = self.cluster_projector(z_j_norm)
        c_i = F.softmax(logits_i / self.temperature, dim=1)
        c_j = F.softmax(logits_j / self.temperature, dim=1)

        return z_i, z_j, c_i, c_j
    
    def forward_cluster(self, x):
        h = self.AE(x)
        h_norm = F.normalize(h, dim=1)
        logits = self.cluster_projector(h_norm)
        c = F.softmax(logits / self.temperature, dim=1)
        cluster_assignments = torch.argmax(c, dim=1)
        return cluster_assignments

    def forward_feature_map(self, x):
        h = self.AE(x)
        h_norm = F.normalize(h, dim=1)
        logits = self.cluster_projector(h_norm)
        c = F.softmax(logits / self.temperature, dim=1)
        return h_norm, c

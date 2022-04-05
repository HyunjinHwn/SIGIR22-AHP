import torch
import torch.nn as nn
import numpy as np
device = torch.device("cuda")

class MLPgenerator(nn.Module):
    def __init__(self, dim, num_nodes, device, size_dist):
        super(MLPgenerator, self).__init__()
        
        def block(in_feat, out_feat, normalize=True, is_last = False):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            if not is_last :
                layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        layer_list = []
        for i, d in enumerate(dim):
            if i == 0:
                layer_list += block(dim[i], dim[i+1], normalize=False)
            elif i < len(dim)-2 :
                layer_list += block(dim[i], dim[i+1])
            elif i == len(dim)-2:
                layer_list += block(dim[i], dim[i+1], is_last = True)
                break
                
        self.model = nn.Sequential(*layer_list)
        self.num_nodes = num_nodes
        self.device = device
        self.size_dist = size_dist
        self.dim = dim

    def forward(self, bs, z=None):
        # sample the neg_sample's size from ground distribution
        if not self.size_dist :
            sampled_size = [5] * bs
        else :
            vals = list(self.size_dist.keys())
            p = [self.size_dist[v] for v in vals]
            sampled_size = np.random.choice(vals, bs, p=p)
        
        # sample nodes
        indices = []
        neg_samples_onehot = []
        z = torch.randn(bs, self.dim[0], device=self.device)
            
        gen_hedge = self.model(z) 
        del z
        for neg_i in range(bs):
            onehots = torch.zeros(sampled_size[neg_i], self.num_nodes).to(self.device)
            values, idx = torch.topk(gen_hedge[neg_i].squeeze(), k=sampled_size[neg_i])
            for i in range(sampled_size[neg_i]):
                onehots[i, idx[i]] = 1 + values[i] - values[i].detach()
            neg_samples_onehot.append(onehots)
            del onehots
            indices.append(idx.detach().to('cpu'))
        return neg_samples_onehot, indices

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class MeanAggregator(nn.Module):
    def __init__(self, dim_vertex, layers):
        super(MeanAggregator, self).__init__()
        Layers = []
        for i in range(len(layers)-1):
            Layers.append(nn.Linear(layers[i], layers[i+1]))
            if i != len(layers)-2:
                Layers.append(nn.ReLU(True))
        self.cls = nn.Sequential(*Layers)
        
    def aggregate(self, embeddings, weights = None):
        if not weights :
            embedding = embeddings.mean(dim=0).squeeze()
        else :
            weights = weights.unsqueeze(-1).repeat(1, embeddings.shape[1])
            embeddings = torch.mul(embeddings, weights)
            embedding = embeddings.mean(dim=0).squeeze()
        return embedding
    
    def classify(self, embedding) :
#         pdb.set_trace()
        embedding = torch.linalg.norm(embedding.unsqueeze(0),dim=0)      
        return self.cls(embedding)
    
    def forward(self, embeddings, weights = None):
        embedding = self.aggregate(embeddings)
        pred = self.classify(embedding)
        return pred, embedding
    

    
class MaxminAggregator(nn.Module):
    def __init__(self, dim_vertex, layers):
        super(MaxminAggregator, self).__init__()
        Layers = []
        for i in range(len(layers)-1):
            Layers.append(nn.Linear(layers[i], layers[i+1]))
            if i != len(layers)-2:
                Layers.append(nn.ReLU(True))
        self.cls = nn.Sequential(*Layers)
    
    def aggregate(self, embeddings):
        max_val, _ = torch.max(embeddings, dim=0)
        min_val, _ = torch.min(embeddings, dim=0)
        return max_val - min_val
    
    def classify(self, embedding):
        return F.sigmoid(self.cls(embedding))
    
    def forward(self, embeddings):
        embedding = self.aggregate(embeddings)
        pred = self.classify(embedding)
        return pred, embedding
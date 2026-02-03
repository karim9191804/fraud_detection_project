"""
GNN Model - Version Légère pour Training Rapide
Optimisé pour dataset avec GPU P100
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, BatchNorm, global_mean_pool


class LightGNNModel(nn.Module):
    """
    GNN Léger avec GAT
    - 2 couches seulement
    - 64 hidden channels
    - Dropout 0.3
    """
    
    def __init__(self, config):
        super().__init__()
        
        # ✅ CORRIGÉ: Pas de default, prend direct du config
        self.in_channels = config['in_channels']
        self.hidden_channels = config.get('hidden_channels', 64)
        self.num_layers = config.get('num_layers', 2)
        self.dropout = config.get('dropout', 0.3)
        self.heads = config.get('heads', 2)
        
        # Couche d'entrée
        self.conv1 = GATConv(
            self.in_channels,
            self.hidden_channels,
            heads=self.heads,
            dropout=self.dropout
        )
        self.bn1 = BatchNorm(self.hidden_channels * self.heads)
        
        # Couche de sortie
        self.conv2 = GATConv(
            self.hidden_channels * self.heads,
            self.hidden_channels,
            heads=1,
            concat=False,
            dropout=self.dropout
        )
        self.bn2 = BatchNorm(self.hidden_channels)
        
        # Classification
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_channels, 32),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(32, 2)  # Fraud/Not Fraud
        )
    
    def forward(self, x, edge_index, batch=None):
        """Forward pass"""
        
        # Layer 1
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Layer 2
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        
        # Global pooling
        if batch is not None:
            x = global_mean_pool(x, batch)
        
        # Classification
        out = self.classifier(x)
        
        return out, x  # logits + embeddings
    
    def get_embeddings(self, x, edge_index, batch=None):
        """Extraire embeddings pour LLM"""
        with torch.no_grad():
            _, embeddings = self.forward(x, edge_index, batch)
        return embeddings

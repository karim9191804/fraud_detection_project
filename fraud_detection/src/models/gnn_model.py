"""
GNN Model - Version Légère pour Training Rapide
Optimisé pour dataset 25% avec GPU P100
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, BatchNorm, global_mean_pool


class LightGNNModel(nn.Module):
    """
    GNN Léger avec GAT
    - 2 couches seulement
    - 64 hidden channels (au lieu de 256+)
    - Dropout 0.3
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.in_channels = config.get('in_channels', 432)
        self.hidden_channels = config.get('hidden_channels', 64)  # LÉGER !
        self.num_layers = config.get('num_layers', 2)  # 2 couches max
        self.dropout = config.get('dropout', 0.3)
        self.heads = config.get('heads', 2)  # 2 attention heads
        
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
        """Forward pass léger"""
        
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


def create_light_gnn(num_features=432):
    """
    Créer GNN léger pour training rapide
    
    ~100K paramètres (vs 1M+ pour version complète)
    """
    config = {
        'in_channels': num_features,
        'hidden_channels': 64,  # Réduit de 256
        'num_layers': 2,  # Réduit de 3-4
        'dropout': 0.3,
        'heads': 2  # Réduit de 4-8
    }
    
    model = LightGNNModel(config)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ GNN Léger créé: {total_params:,} paramètres")
    
    return model

"""
Graph Neural Network pour la détection de fraude
Utilise Graph Attention Networks (GAT) pour analyser les relations entre transactions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool


class GNNModel(nn.Module):
    """
    Graph Attention Network pour l'apprentissage de représentations de graphes de transactions
    """
    
    def __init__(self, config):
        super(GNNModel, self).__init__()
        
        self.config = config
        self.hidden_channels = config.get('hidden_channels', 256)
        self.num_layers = config.get('num_layers', 3)
        self.dropout = config.get('dropout', 0.3)
        self.heads = config.get('heads', 8)
        
        # Couches GAT
        self.convs = nn.ModuleList()
        
        # Première couche (input_dim sera défini dynamiquement)
        self.input_dim = None  # Sera initialisé lors du premier forward
        
        # Couches intermédiaires
        for i in range(self.num_layers):
            if i == 0:
                # Première couche (sera créée dynamiquement)
                pass
            else:
                in_channels = self.hidden_channels * self.heads if i == 1 else self.hidden_channels
                out_channels = self.hidden_channels
                heads = self.heads if i < self.num_layers - 1 else 1
                
                self.convs.append(
                    GATConv(in_channels, out_channels, heads=heads, dropout=self.dropout)
                )
        
        # Couche de classification
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_channels, self.hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_channels // 2, 64),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(64, 1)
        )
        
        # Flag pour initialisation dynamique
        self.initialized = False
    
    def _init_first_layer(self, input_dim):
        """Initialise la première couche GAT avec la bonne dimension d'entrée"""
        if not self.initialized:
            self.input_dim = input_dim
            first_conv = GATConv(
                input_dim, 
                self.hidden_channels, 
                heads=self.heads, 
                dropout=self.dropout
            )
            self.convs.insert(0, first_conv)
            self.initialized = True
            
            # Déplacer sur le bon device
            device = next(self.parameters()).device
            self.convs[0] = self.convs[0].to(device)
    
    def forward(self, x, edge_index, batch=None):
        """
        Forward pass
        
        Args:
            x: Features des nœuds [num_nodes, input_dim]
            edge_index: Indices des arêtes [2, num_edges]
            batch: Assignation des nœuds aux graphes [num_nodes]
        
        Returns:
            logits: Prédictions [num_graphs, 1] ou [num_nodes, 1]
        """
        # Initialisation dynamique de la première couche
        if not self.initialized:
            self._init_first_layer(x.size(1))
        
        # Appliquer les couches GAT
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            
            if i < len(self.convs) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Pooling global si batch est fourni (mode graphe)
        if batch is not None:
            x = global_mean_pool(x, batch)
        
        # Classification
        logits = self.classifier(x)
        
        return logits
    
    def get_embeddings(self, x, edge_index, batch=None):
        """
        Obtenir les embeddings (sans la couche de classification)
        Utile pour le modèle hybride
        """
        if not self.initialized:
            self._init_first_layer(x.size(1))
        
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            
            if i < len(self.convs) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        if batch is not None:
            x = global_mean_pool(x, batch)
        
        return x


# Test du modèle (ne s'exécute que si ce fichier est lancé directement)
if __name__ == "__main__":
    print("🧪 Test du GNNModel...")
    
    config = {
        'hidden_channels': 128,
        'num_layers': 3,
        'dropout': 0.3,
        'heads': 4
    }
    
    model = GNNModel(config)
    print(f"✅ Modèle créé : {model}")
    print(f"📊 Paramètres : {sum(p.numel() for p in model.parameters()):,}")

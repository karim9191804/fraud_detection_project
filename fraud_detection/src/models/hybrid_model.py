"""
Modèle Hybride GNN+LLM - Version Légère
Fusion optimisée pour training rapide
"""

import torch
import torch.nn as nn


class LightHybridModel(nn.Module):
    """
    Modèle Hybride Léger
    Total: ~170M params
    Avec LoRA: seulement ~1.1M params entraînables
    """
    
    def __init__(self, gnn_model, llm_model):
        super().__init__()
        
        self.gnn = gnn_model
        self.llm = llm_model
        
        # Dimensions
        self.gnn_hidden = 64
        self.llm_hidden = 768
        
        # Projecteur GNN → LLM
        self.projector = nn.Sequential(
            nn.Linear(self.gnn_hidden, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, self.llm_hidden)
        )
        
        # Fusion finale
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.llm_hidden * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
    
    def forward(self, graph_data, text_data=None):
        """
        Forward pass hybride
        
        Args:
            graph_data: (x, edge_index, batch) pour GNN
            text_data: Liste de textes pour LLM (optionnel)
        
        Returns:
            logits: Prédiction finale (batch_size, 2)
        """
        
        x, edge_index, batch = graph_data
        
        # 1. GNN Forward
        gnn_logits, gnn_embeddings = self.gnn(x, edge_index, batch)
        
        # 2. Si pas de texte, utiliser juste GNN
        if text_data is None:
            return gnn_logits
        
        # 3. Projeter GNN vers espace LLM
        gnn_projected = self.projector(gnn_embeddings)
        
        # 4. LLM Forward
        inputs = self.llm.tokenizer(
            text_data,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )
        inputs = {k: v.to(x.device) for k, v in inputs.items()}
        
        llm_logits, llm_embeddings = self.llm(**inputs)
        
        # 5. Fusion GNN + LLM
        fused = torch.cat([gnn_projected, llm_embeddings], dim=-1)
        final_logits = self.fusion_layer(fused)
        
        return final_logits

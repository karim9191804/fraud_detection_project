"""
Modèle Hybride GNN+LLM - Version Légère
Fusion optimisée pour training rapide
"""

import torch
import torch.nn as nn


class LightHybridModel(nn.Module):
    """
    Modèle Hybride Léger
    - GNN: 64 hidden channels
    - LLM: DistilBERT (66M params)
    - Fusion: Projection simple
    
    Total: ~170M params (vs 2.8B+ pour version complète)
    Avec LoRA: seulement ~1.1M params entraînables
    """
    
    def __init__(self, gnn_model, llm_model):
        super().__init__()
        
        self.gnn = gnn_model
        self.llm = llm_model
        
        # Dimensions
        self.gnn_hidden = 64  # GNN output
        self.llm_hidden = 768  # DistilBERT hidden size
        
        # Projecteur GNN → LLM
        self.projector = nn.Sequential(
            nn.Linear(self.gnn_hidden, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, self.llm_hidden)
        )
        
        # Fusion finale
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.llm_hidden * 2, 256),  # GNN + LLM
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # Fraud/Not Fraud
        )
    
    def forward(self, graph_data, text_data=None):
        """
        Forward pass hybride
        
        Args:
            graph_data: (x, edge_index, batch) pour GNN
            text_data: (input_ids, attention_mask) pour LLM (optionnel)
        
        Returns:
            logits: Prédiction finale
            embeddings: Dict avec GNN + LLM embeddings
        """
        
        x, edge_index, batch = graph_data
        
        # 1. GNN Forward
        gnn_logits, gnn_embeddings = self.gnn(x, edge_index, batch)
        
        # 2. Projeter GNN embeddings vers espace LLM
        gnn_projected = self.projector(gnn_embeddings)
        
        # 3. Si texte fourni, utiliser LLM
        if text_data is not None:
            input_ids, attention_mask = text_data
            llm_logits, llm_embeddings = self.llm(input_ids, attention_mask)
            
            # Fusion GNN + LLM
            fused = torch.cat([gnn_projected, llm_embeddings], dim=-1)
            final_logits = self.fusion_layer(fused)
            
            embeddings = {
                'gnn': gnn_embeddings,
                'llm': llm_embeddings,
                'fused': fused
            }
        else:
            # Mode GNN seulement (plus rapide)
            final_logits = gnn_logits
            embeddings = {
                'gnn': gnn_embeddings,
                'llm': None,
                'fused': gnn_projected
            }
        
        return final_logits, embeddings
    
    def predict_with_explanation(self, graph_data, transaction_info):
        """
        Prédiction + Explication
        
        Utilisé en MODE JOUR pour inférence rapide
        """
        
        with torch.no_grad():
            # Prédiction GNN rapide
            logits, embeddings = self.forward(graph_data, text_data=None)
            
            # Probabilités
            probs = torch.softmax(logits, dim=-1)
            pred_class = torch.argmax(probs, dim=-1)
            confidence = probs.max(dim=-1).values
            
            # Générer explication simple
            explanation = self.llm.generate_explanation(
                embeddings['gnn'],
                transaction_info
            )
        
        return {
            'prediction': pred_class,
            'confidence': confidence,
            'probabilities': probs,
            'explanation': explanation,
            'embeddings': embeddings
        }
    
    def freeze_gnn(self):
        """Geler GNN pour fine-tuning LLM seulement"""
        for param in self.gnn.parameters():
            param.requires_grad = False
        print("🔒 GNN gelé")
    
    def unfreeze_gnn(self):
        """Dégeler GNN"""
        for param in self.gnn.parameters():
            param.requires_grad = True
        print("🔓 GNN dégelé")
    
    def freeze_llm(self):
        """Geler LLM pour fine-tuning GNN seulement"""
        for param in self.llm.parameters():
            param.requires_grad = False
        print("🔒 LLM gelé")
    
    def unfreeze_llm(self):
        """Dégeler LLM"""
        for param in self.llm.parameters():
            param.requires_grad = True
        print("🔓 LLM dégelé")


def create_light_hybrid_model():
    """
    Créer modèle hybride léger complet
    
    - GNN: ~100K params
    - LLM: ~66M params (DistilBERT)
    - Projector + Fusion: ~600K params
    
    Total: ~170M params
    Entraînables avec LoRA: ~1.1M params
    """
    
    from gnn_model_light import create_light_gnn
    from llm_wrapper_light import create_light_llm
    
    # Créer composants
    gnn = create_light_gnn()
    llm = create_light_llm()
    
    # Créer hybride
    hybrid_model = LightHybridModel(gnn, llm)
    
    # Stats
    total_params = sum(p.numel() for p in hybrid_model.parameters())
    trainable_params = sum(p.numel() for p in hybrid_model.parameters() if p.requires_grad)
    
    print("\n" + "="*60)
    print("✅ MODÈLE HYBRIDE LÉGER CRÉÉ")
    print("="*60)
    print(f"  Total paramètres: {total_params:,}")
    print(f"  Entraînables: {trainable_params:,}")
    print(f"  Ratio: {trainable_params/total_params*100:.1f}%")
    print("="*60)
    
    return hybrid_model

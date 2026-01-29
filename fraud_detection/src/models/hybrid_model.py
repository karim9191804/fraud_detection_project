"""
Modèle hybride GNN + LLM pour la détection de fraude
Combine les embeddings du GNN avec un LLM (Phi-2) pour l'analyse contextuelle
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

from .gnn_model import GNNModel


class HybridFraudDetector(nn.Module):
    """
    Détecteur de fraude hybride combinant :
    - GNN (GAT) pour l'analyse structurelle du graphe de transactions
    - LLM (Phi-2 + LoRA) pour l'analyse contextuelle et le raisonnement
    """
    
    def __init__(self, config, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(HybridFraudDetector, self).__init__()
        
        self.config = config
        self.device = device
        
        # 1. Composant GNN
        print("🔧 Initialisation du GNN (GAT)...")
        self.gnn = GNNModel(config.get('gnn', {}))
        
        # 2. Composant LLM (Phi-2 avec LoRA)
        print("🔧 Chargement du LLM (microsoft/phi-2)...")
        llm_config = config.get('llm', {})
        model_name = llm_config.get('model_name', 'microsoft/phi-2')
        
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side='left'
        )
        
        # Ajouter un pad_token si manquant
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Charger le modèle de base
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
            device_map='auto' if device == 'cuda' else None
        )
        
        # Appliquer LoRA si activé
        if llm_config.get('use_lora', True):
            print("🔧 Application de LoRA...")
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=llm_config.get('lora_r', 8),
                lora_alpha=llm_config.get('lora_alpha', 32),
                lora_dropout=llm_config.get('lora_dropout', 0.1),
                target_modules=["q_proj", "v_proj"],  # Phi-2 architecture
                bias="none"
            )
            self.llm = get_peft_model(self.llm, lora_config)
            self.llm.print_trainable_parameters()
        
        # 3. Couche de fusion GNN → LLM
        gnn_dim = config.get('gnn', {}).get('hidden_channels', 256)
        llm_dim = self.llm.config.hidden_size
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(gnn_dim, llm_dim),
            nn.LayerNorm(llm_dim),
            nn.Dropout(0.1)
        )
        
        # 4. Tête de classification finale
        self.fraud_head = nn.Sequential(
            nn.Linear(llm_dim, llm_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(llm_dim // 2, 1)
        )
        
        self.to(device)
    
    def forward(self, graph_data, text_inputs=None, mode='inference'):
        """
        Forward pass du modèle hybride
        
        Args:
            graph_data: PyTorch Geometric Data object (x, edge_index, batch)
            text_inputs: Dict avec 'input_ids' et 'attention_mask' (optionnel)
            mode: 'inference', 'gnn_only', 'hybrid'
        
        Returns:
            logits: Prédictions de fraude [batch_size, 1]
            embeddings: Embeddings combinés (pour analyse)
        """
        
        # 1. Extraire les embeddings du GNN
        x = graph_data.x
        edge_index = graph_data.edge_index
        batch = graph_data.batch if hasattr(graph_data, 'batch') else None
        
        gnn_embeddings = self.gnn.get_embeddings(x, edge_index, batch)
        
        # Mode GNN uniquement (baseline)
        if mode == 'gnn_only':
            logits = self.gnn.classifier(gnn_embeddings)
            return logits, gnn_embeddings
        
        # 2. Fusionner avec le LLM
        # Projeter les embeddings GNN dans l'espace du LLM
        fused_embeddings = self.fusion_layer(gnn_embeddings)
        
        # 3. Si des inputs texte sont fournis, les combiner
        if text_inputs is not None and mode == 'hybrid':
            # Passer les inputs texte dans le LLM
            with torch.no_grad():
                llm_outputs = self.llm(
                    input_ids=text_inputs['input_ids'].to(self.device),
                    attention_mask=text_inputs['attention_mask'].to(self.device),
                    output_hidden_states=True
                )
                text_embeddings = llm_outputs.hidden_states[-1][:, -1, :]  # Dernier token
            
            # Combiner (moyenne pondérée ou concaténation)
            combined_embeddings = (fused_embeddings + text_embeddings) / 2
        else:
            combined_embeddings = fused_embeddings
        
        # 4. Prédiction finale
        logits = self.fraud_head(combined_embeddings)
        
        return logits, combined_embeddings
    
    def generate_explanation(self, graph_data, prediction, max_length=100):
        """
        Générer une explication textuelle de la prédiction
        Utilise le LLM en mode génération
        """
        # Extraire les embeddings
        _, embeddings = self.forward(graph_data, mode='inference')
        
        # Construire le prompt
        fraud_prob = torch.sigmoid(prediction).item()
        prompt = f"""Analyze this transaction:
Fraud probability: {fraud_prob:.2%}
Reasoning:"""
        
        # Tokenizer
        inputs = self.tokenizer(
            prompt,
            return_tensors='pt',
            padding=True,
            truncation=True
        ).to(self.device)
        
        # Générer
        with torch.no_grad():
            outputs = self.llm.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        explanation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return explanation
    
    def save_pretrained(self, save_path):
        """Sauvegarder le modèle complet"""
        torch.save({
            'gnn_state_dict': self.gnn.state_dict(),
            'fusion_layer': self.fusion_layer.state_dict(),
            'fraud_head': self.fraud_head.state_dict(),
            'config': self.config
        }, f"{save_path}/hybrid_model.pt")
        
        # Sauvegarder le LLM (avec LoRA)
        self.llm.save_pretrained(f"{save_path}/llm")
        self.tokenizer.save_pretrained(f"{save_path}/llm")
        
        print(f"✅ Modèle sauvegardé dans {save_path}")


# Test du modèle
if __name__ == "__main__":
    print("🧪 Test du HybridFraudDetector...")
    
    config = {
        'gnn': {
            'hidden_channels': 128,
            'num_layers': 2,
            'dropout': 0.3,
            'heads': 4
        },
        'llm': {
            'model_name': 'microsoft/phi-2',
            'use_lora': True,
            'lora_r': 8,
            'lora_alpha': 32
        }
    }
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  Device: {device}")
    
    # Modèle (attention: télécharge Phi-2 ~3GB)
    # model = HybridFraudDetector(config, device=device)
    # print("✅ Modèle hybride créé")
    print("⚠️ Décommentez pour tester (télécharge Phi-2)")

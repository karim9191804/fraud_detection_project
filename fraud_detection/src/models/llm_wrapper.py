"""
LLM Wrapper - Version Légère avec DistilBERT
66M paramètres au lieu de 2.7B (Phi-2)
Training 40x plus rapide !
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model


class LightLLMWrapper(nn.Module):
    """
    LLM Léger basé sur DistilBERT
    - 66M paramètres (vs 2.7B pour Phi-2)
    - LoRA pour fine-tuning efficace
    - Optimisé pour explications courtes
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.model_name = config.get('model_name', 'distilbert-base-uncased')
        self.max_length = config.get('max_length', 128)  # Court !
        self.use_lora = config.get('use_lora', True)
        self.lora_r = config.get('lora_r', 4)  # Très léger
        self.lora_alpha = config.get('lora_alpha', 8)
        
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Modèle de base
        self.base_model = AutoModel.from_pretrained(self.model_name)
        
        # LoRA pour fine-tuning léger
        if self.use_lora:
            lora_config = LoraConfig(
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                target_modules=['q_lin', 'v_lin'],  # DistilBERT layers
                lora_dropout=0.1,
                bias="none",
                task_type="CAUSAL_LM"
            )
            self.base_model = get_peft_model(self.base_model, lora_config)
            print(f"✅ LoRA activé: {self.lora_r}r, {self.lora_alpha}α")
        
        # Projection pour fusion avec GNN
        self.hidden_size = self.base_model.config.hidden_size  # 768 pour DistilBERT
        
        # Tête de classification pour fraud
        self.fraud_head = nn.Sequential(
            nn.Linear(self.hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 2)
        )
    
    def forward(self, input_ids, attention_mask=None):
        """Forward pass"""
        
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # [CLS] token embedding
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        # Classification
        logits = self.fraud_head(pooled_output)
        
        return logits, pooled_output
    
    def generate_explanation(self, gnn_embedding, transaction_features):
        """
        Générer explication de fraude
        
        Simplifié: templates + scores au lieu de génération complète
        Plus rapide et suffisant pour explainability
        """
        
        # Scores de risque basés sur GNN
        risk_score = torch.sigmoid(gnn_embedding).mean().item()
        
        # Templates d'explication simples
        if risk_score > 0.8:
            explanation = f"⚠️ RISQUE ÉLEVÉ ({risk_score:.2%}): Transaction suspecte détectée. "
            explanation += "Patterns anormaux dans le graphe de transactions."
        elif risk_score > 0.5:
            explanation = f"⚡ RISQUE MOYEN ({risk_score:.2%}): Activité à surveiller. "
            explanation += "Certains indicateurs sont inhabituels."
        else:
            explanation = f"✅ RISQUE FAIBLE ({risk_score:.2%}): Transaction normale."
        
        return explanation
    
    def prepare_text_for_rlhf(self, transaction_data, fraud_label, explanation):
        """
        Préparer texte pour RLHF
        Format: [Transaction] → [Prediction] → [Explanation]
        """
        
        text = f"Transaction Amount: ${transaction_data.get('amount', 0):.2f}\n"
        text += f"Prediction: {'FRAUD' if fraud_label == 1 else 'NORMAL'}\n"
        text += f"Explanation: {explanation}"
        
        return text


def create_light_llm():
    """
    Créer LLM léger pour training rapide
    
    DistilBERT: 66M paramètres
    Avec LoRA: seulement ~1M paramètres entraînables
    """
    config = {
        'model_name': 'distilbert-base-uncased',
        'max_length': 128,
        'use_lora': True,
        'lora_r': 4,  # Très léger
        'lora_alpha': 8
    }
    
    model = LightLLMWrapper(config)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✅ LLM Léger créé (DistilBERT)")
    print(f"   Total: {total_params:,} paramètres")
    print(f"   Entraînables (LoRA): {trainable_params:,} paramètres")
    
    return model

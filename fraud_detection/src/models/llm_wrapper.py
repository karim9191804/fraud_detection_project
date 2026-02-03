"""
LLM Wrapper - Version Légère avec DistilBERT
66M paramètres - Training 40x plus rapide
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model, TaskType


class LightLLMWrapper(nn.Module):
    """
    LLM Léger basé sur DistilBERT
    - 66M paramètres
    - LoRA pour fine-tuning efficace
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.model_name = config.get('model_name', 'distilbert-base-uncased')
        self.max_length = config.get('max_length', 128)
        self.use_lora = config.get('use_lora', True)
        self.lora_r = config.get('lora_r', 4)
        self.lora_alpha = config.get('lora_alpha', 8)
        
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Modèle de base
        self.base_model = AutoModel.from_pretrained(self.model_name)
        
        # ✅ CORRIGÉ: SEQ_CLS au lieu de CAUSAL_LM
        if self.use_lora:
            lora_config = LoraConfig(
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                target_modules=['q_lin', 'v_lin'],
                lora_dropout=0.1,
                bias="none",
                task_type=TaskType.SEQ_CLS  # ✅ CORRIGÉ
            )
            self.base_model = get_peft_model(self.base_model, lora_config)
            print(f"✅ LoRA activé: {self.lora_r}r, {self.lora_alpha}α")
        
        self.hidden_size = self.base_model.config.hidden_size
        
        # Tête de classification
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

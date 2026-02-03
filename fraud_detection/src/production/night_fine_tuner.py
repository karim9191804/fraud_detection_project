"""
Night Fine-Tuner
Fine-tuning automatique pendant la nuit + RLHF
MODE NUIT: Apprentissage continu (2h-6h du matin)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import json
import os


class NightFineTuner:
    """
    Fine-tuning MODE NUIT
    
    • S'exécute pendant heures creuses (2h-6h)
    • Fine-tune sur cas critiques accumulés
    • RLHF avec feedbacks experts
    • Validation automatique
    """
    
    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        
        # Optimizer pour fine-tuning (LR plus petit)
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.get('night_lr', 1e-5),
            weight_decay=1e-6
        )
        
        self.criterion = nn.CrossEntropyLoss()
        self.fine_tuning_history = []
        
        print("✅ Night Fine-Tuner initialisé")
        print(f"   LR: {config.get('night_lr', 1e-5)}")
    
    def fine_tune_on_critical_cases(
        self, 
        critical_cases, 
        num_epochs=5,
        validation_data=None
    ):
        """
        Fine-tuning sur cas critiques accumulés pendant la journée
        """
        
        print(f"\n{'='*60}")
        print(f"🌙 NIGHT MODE - FINE-TUNING")
        print(f"{'='*60}")
        print(f"   Cas critiques: {len(critical_cases)}")
        print(f"   Epochs: {num_epochs}")
        
        if len(critical_cases) == 0:
            print("⚠️  Aucun cas critique, skip fine-tuning")
            return {'skipped': True}
        
        # Filtrer cas avec feedback
        cases_with_feedback = [
            c for c in critical_cases 
            if c.get('human_feedback') is not None
        ]
        
        print(f"   Avec feedback humain: {len(cases_with_feedback)}")
        
        if len(cases_with_feedback) == 0:
            print("⚠️  Aucun feedback humain, utilisation auto-feedback")
            cases_with_feedback = critical_cases
        
        # Préparer données
        features_list = []
        labels_list = []
        
        for case in cases_with_feedback:
            features_list.append(case['features'])
            
            if case.get('human_feedback'):
                label = case['human_feedback']['corrected_label']
            else:
                label = 1 if case['fraud_probability'] > 0.5 else 0
            
            labels_list.append(label)
        
        # Convertir en tensors
        X = torch.tensor(features_list, dtype=torch.float32).to(self.device)
        y = torch.tensor(labels_list, dtype=torch.long).to(self.device)
        
        # Fine-tuning loop
        self.model.train()
        
        history = {
            'epochs': [],
            'losses': [],
            'val_metrics': []
        }
        
        for epoch in range(num_epochs):
            # Forward
            if hasattr(self.model, 'gnn'):
                embeddings = self.model.gnn.conv1.lin(X)
                logits = self.model.gnn.classifier(embeddings)
            else:
                logits, _ = self.model(X, None, None)
            
            loss = self.criterion(logits, y)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Validation
            val_metrics = {}
            if validation_data is not None:
                val_metrics = self.validate(validation_data)
            
            history['epochs'].append(epoch + 1)
            history['losses'].append(float(loss.item()))
            history['val_metrics'].append(val_metrics)
            
            print(f"\n   Epoch {epoch+1}/{num_epochs}:")
            print(f"      Loss: {loss.item():.4f}")
            if val_metrics:
                print(f"      Val F1: {val_metrics.get('f1_score', 0):.4f}")
        
        # Sauvegarder historique
        self.fine_tuning_history.append({
            'timestamp': datetime.now().isoformat(),
            'num_cases': len(cases_with_feedback),
            'num_epochs': num_epochs,
            'history': history
        })
        
        print(f"\n✅ Fine-tuning terminé")
        
        return {
            'num_cases': len(cases_with_feedback),
            'num_epochs': num_epochs,
            'final_loss': float(history['losses'][-1]),
            'history': history
        }
    
    def rlhf_update(self, critical_cases):
        """
        RLHF (Reinforcement Learning from Human Feedback)
        """
        
        print(f"\n{'='*60}")
        print(f"🎯 RLHF - APPRENTISSAGE PAR RENFORCEMENT")
        print(f"{'='*60}")
        
        # Filtrer cas avec feedback
        cases_with_feedback = [
            c for c in critical_cases 
            if c.get('human_feedback') is not None
        ]
        
        if len(cases_with_feedback) == 0:
            print("⚠️  Pas de feedback humain disponible")
            return {'skipped': True}
        
        print(f"   Feedbacks disponibles: {len(cases_with_feedback)}")
        
        # Calculer rewards
        rewards = []
        for case in cases_with_feedback:
            feedback = case['human_feedback']
            
            is_correct = (
                (case['fraud_probability'] > 0.5) == 
                (feedback['corrected_label'] == 1)
            )
            
            if is_correct:
                if case['confidence'] > 0.8:
                    reward = 2.0
                else:
                    reward = 1.0
            else:
                if case['confidence'] > 0.8:
                    reward = -2.0
                else:
                    reward = -0.5
            
            rewards.append(reward)
        
        avg_reward = sum(rewards) / len(rewards)
        
        print(f"   Reward moyen: {avg_reward:.4f}")
        print(f"   Rewards positifs: {sum(1 for r in rewards if r > 0)}")
        print(f"   Rewards négatifs: {sum(1 for r in rewards if r < 0)}")
        
        # Policy gradient update
        self.model.train()
        
        total_loss = 0
        for case, reward in zip(cases_with_feedback, rewards):
            x = torch.tensor(case['features'], dtype=torch.float32).unsqueeze(0).to(self.device)
            
            if hasattr(self.model, 'gnn'):
                embeddings = self.model.gnn.conv1.lin(x)
                logits = self.model.gnn.classifier(embeddings)
            else:
                logits, _ = self.model(x, None, None)
            
            probs = torch.softmax(logits, dim=1)
            log_prob = torch.log(probs[0, case['human_feedback']['corrected_label']] + 1e-8)
            
            loss = -reward * log_prob
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(cases_with_feedback)
        
        print(f"   Avg RLHF loss: {avg_loss:.4f}")
        print(f"✅ RLHF update terminé")
        
        return {
            'num_feedbacks': len(cases_with_feedback),
            'avg_reward': avg_reward,
            'avg_loss': avg_loss,
            'rewards': rewards
        }
    
    def validate(self, val_data):
        """Validation rapide"""
        
        try:
            from src.utils.metrics import compute_all_metrics
        except:
            return {}
        
        self.model.eval()
        
        with torch.no_grad():
            val_data = val_data.to(self.device)
            
            if hasattr(self.model, 'gnn'):
                logits, _ = self.model.gnn(val_data.x, val_data.edge_index, None)
            else:
                logits, _ = self.model(val_data.x, val_data.edge_index, None)
            
            pred = logits.argmax(dim=1).cpu().numpy()
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            
            metrics = compute_all_metrics(val_data.y.cpu().numpy(), pred, probs)
        
        return metrics
    
    def save_checkpoint(self, path):
        """Sauvegarder modèle amélioré"""
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'fine_tuning_history': self.fine_tuning_history,
            'timestamp': datetime.now().isoformat()
        }, path)
        
        print(f"💾 Checkpoint sauvegardé: {path}")

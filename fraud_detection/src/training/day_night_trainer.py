"""
Enhanced Training System for Fraud Detection
Inspired by Day/Night/Morning Cycle - Simplified for Kaggle
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix
)
from datetime import datetime
import os
from typing import Dict
import warnings
warnings.filterwarnings('ignore')


class FraudDetectionTrainer:
    """
    Trainer simplifié pour Kaggle avec concepts Day/Night/Morning
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Mémoire pour cas critiques (inspiré du mode Day)
        self.critical_cases = []
        
        # Historique d'entraînement
        self.history = {
            'train': [],
            'val': [],
            'learning_rates': [],
            'critical_cases_count': []
        }
        
        # Meilleurs résultats
        self.best_val_loss = float('inf')
        self.best_val_f1 = 0.0
        self.best_epoch = 0
        
        # Setup optimizer et loss
        self._setup_optimizer()
        self._setup_loss()
        self._setup_scheduler()
        
        print(f"✅ Trainer initialisé sur {device}")
    
    def _setup_optimizer(self):
        """Configure l'optimizer avec learning rates différentiels"""
        param_groups = [
            {'params': self.model.gnn.parameters(), 'lr': self.config['gnn_lr'], 'name': 'gnn'},
            {'params': self.model.llm.parameters(), 'lr': self.config['llm_lr'], 'name': 'llm'},
            {'params': self.model.projection.parameters(), 'lr': self.config.get('projection_lr', 1e-3), 'name': 'projection'}
        ]
        
        self.optimizer = optim.AdamW(
            param_groups,
            betas=self.config.get('adam_betas', (0.9, 0.999)),
            eps=self.config.get('adam_eps', 1e-8),
            weight_decay=self.config.get('weight_decay', 1e-5)
        )
        
        print(f"⚙️ Optimizer configuré:")
        for g in param_groups:
            print(f"   {g['name']} LR: {g['lr']}")
    
    def _setup_loss(self):
        """Configure la fonction de loss"""
        if self.config.get('use_focal_loss', True):
            self.criterion = FocalLoss(
                alpha=self.config.get('focal_alpha', 0.25),
                gamma=self.config.get('focal_gamma', 2.0),
                pos_weight=torch.tensor([self.config.get('pos_weight', 10.0)]).to(self.device)
            )
            print(f"🎯 Loss: Focal Loss (α={self.config.get('focal_alpha', 0.25)}, γ={self.config.get('focal_gamma', 2.0)})")
        else:
            pos_weight = torch.tensor([self.config.get('pos_weight', 10.0)]).to(self.device)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            print(f"🎯 Loss: BCEWithLogitsLoss (pos_weight={self.config.get('pos_weight', 10.0)})")
    
    def _setup_scheduler(self):
        """Configure le learning rate scheduler"""
        if self.config.get('use_scheduler', True):
            self.scheduler = WarmupReduceLROnPlateau(
                self.optimizer,
                warmup_epochs=self.config.get('warmup_epochs', 2),
                patience=self.config.get('lr_patience', 3),
                factor=self.config.get('lr_factor', 0.5),
                min_lr=self.config.get('min_lr', 1e-6)
            )
            print(f"📉 Scheduler: Warm-up + ReduceLROnPlateau")
        else:
            self.scheduler = None
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int):
        """Boucle d'entraînement principale"""
        print("\n" + "="*80)
        print("🚀 DÉBUT DE L'ENTRAÎNEMENT")
        print("="*80)
        
        for epoch in range(num_epochs):
            print(f"\n📅 Epoch {epoch + 1}/{num_epochs}")
            
            # Entraînement
            train_metrics = self._train_epoch(train_loader, epoch + 1)
            
            # Validation
            val_metrics = self._validate(val_loader, epoch + 1)
            
            # Sauvegarder l'historique
            self.history['train'].append(train_metrics)
            self.history['val'].append(val_metrics)
            
            # Learning rates
            lrs = {f"lr_{group['name']}": group['lr'] for group in self.optimizer.param_groups}
            self.history['learning_rates'].append(lrs)
            self.history['critical_cases_count'].append(len(self.critical_cases))
            
            # Afficher les résultats
            self._print_epoch_results(epoch + 1, train_metrics, val_metrics)
            
            # Scheduler step
            if self.scheduler:
                self.scheduler.step(val_metrics['loss'])
            
            # Sauvegarder le meilleur modèle
            self._save_best_model(epoch + 1, train_metrics, val_metrics)
            
            # Early stopping
            if self._check_early_stopping(val_metrics['loss'], epoch + 1):
                print(f"\n⚠️ Early stopping à l'epoch {epoch + 1}")
                break
        
        print("\n✅ ENTRAÎNEMENT TERMINÉ!")
        self._print_final_results()
    
    def _train_epoch(self, loader: DataLoader, epoch: int) -> Dict:
        """Une epoch d'entraînement"""
        self.model.train()
        total_loss = 0
        all_preds, all_labels = [], []
        
        for data in tqdm(loader, desc=f"Training", leave=False):
            data = data.to(self.device)
            logits = self.model(data).squeeze()
            if logits.dim() == 0:
                logits = logits.unsqueeze(0)
            labels = data.y.float()
            loss = self.criterion(logits, labels)
            
            self.optimizer.zero_grad()
            loss.backward()
            if self.config.get('grad_clip', 1.0) > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
            self.optimizer.step()
            
            total_loss += loss.item()
            preds = torch.sigmoid(logits).detach().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
            self._detect_critical_cases(data, preds, labels)
        
        metrics = compute_metrics(np.array(all_preds), np.array(all_labels))
        metrics['loss'] = total_loss / len(loader)
        return metrics
    
    def _validate(self, loader: DataLoader, epoch: int) -> Dict:
        """Validation"""
        self.model.eval()
        total_loss = 0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for data in tqdm(loader, desc=f"Validation", leave=False):
                data = data.to(self.device)
                logits = self.model(data).squeeze()
                if logits.dim() == 0:
                    logits = logits.unsqueeze(0)
                labels = data.y.float()
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                preds = torch.sigmoid(logits).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
        
        metrics = compute_metrics(np.array(all_preds), np.array(all_labels))
        metrics['loss'] = total_loss / len(loader)
        return metrics
    
    def _detect_critical_cases(self, data, preds, labels):
        """Détecter et sauvegarder les cas critiques"""
        if not self.config.get('save_critical_cases', True):
            return
        threshold = self.config.get('critical_confidence_threshold', 0.6)
        max_cases = self.config.get('max_critical_cases', 1000)
        for i in range(len(preds)):
            conf = max(preds[i], 1 - preds[i])
            if conf < threshold and len(self.critical_cases) < max_cases:
                self.critical_cases.append({
                    'prediction': int(preds[i] > 0.5),
                    'confidence': float(conf),
                    'fraud_prob': float(preds[i]),
                    'true_label': int(labels[i].item())
                })
    
    def _print_epoch_results(self, epoch: int, train_metrics: Dict, val_metrics: Dict):
        """Afficher les résultats de l'epoch"""
        print(f"\n📊 Résultats Epoch {epoch}:")
        print(f"   Train Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['loss']:.4f}")
        print(f"   Train F1: {train_metrics['f1']:.4f} | Val F1: {val_metrics['f1']:.4f}")
        can_deploy, _ = check_deployment_criteria(val_metrics, self.config)
        print("   ✅ Modèle prêt pour le déploiement!" if can_deploy else "   ⚠️ Critères de déploiement non atteints")
    
    def _save_best_model(self, epoch: int, train_metrics: Dict, val_metrics: Dict):
        """Sauvegarder le meilleur modèle"""
        if val_metrics['loss'] < self.best_val_loss:
            self.best_val_loss = val_metrics['loss']
            self.best_val_f1 = val_metrics['f1']
            self.best_epoch = epoch
            os.makedirs(self.config['checkpoint_dir'], exist_ok=True)
            checkpoint_path = os.path.join(self.config['checkpoint_dir'], "best_model.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'config': self.config
            }, checkpoint_path)
            print(f"\n💾 Meilleur modèle sauvegardé! (Loss: {val_metrics['loss']:.4f}, F1: {val_metrics['f1']:.4f})")
    
    def _check_early_stopping(self, val_loss: float, epoch: int) -> bool:
        if not hasattr(self, 'early_stopping_counter'):
            self.early_stopping_counter = 0
            self.early_stopping_best = val_loss
        patience = self.config.get('patience', 5)
        min_delta = self.config.get('min_delta', 1e-4)
        if val_loss < self.early_stopping_best - min_delta:
            self.early_stopping_best = val_loss
            self.early_stopping_counter = 0
            return False
        else:
            self.early_stopping_counter += 1
            return self.early_stopping_counter >= patience
    
    def _print_final_results(self):
        print(f"\n🏆 Meilleure performance: Epoch {self.best_epoch} | Val Loss {self.best_val_loss:.4f} | Val F1 {self.best_val_f1:.4f}")
        if self.critical_cases:
            print(f"🔍 Cas critiques détectés: {len(self.critical_cases)}")


# ============================================
# UTILITAIRES
# ============================================

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, pos_weight=self.pos_weight, reduction='none')
        probs = torch.sigmoid(inputs)
        pt = torch.where(targets==1, probs, 1-probs)
        focal_weight = (1-pt)**self.gamma
        if self.alpha is not None:
            alpha_t = torch.where(targets==1, self.alpha, 1-self.alpha)
            focal_weight = alpha_t * focal_weight
        return (focal_weight * bce_loss).mean()


class WarmupReduceLROnPlateau:
    def __init__(self, optimizer, warmup_epochs, patience, factor, min_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
        self.base_lrs = [g['lr'] for g in optimizer.param_groups]
        self.plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=factor, patience=patience, min_lr=min_lr, verbose=True)
    def step(self, val_loss=None):
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            factor = self.current_epoch / self.warmup_epochs
            for i, g in enumerate(self.optimizer.param_groups):
                g['lr'] = self.base_lrs[i]*factor
        else:
            if val_loss is not None:
                self.plateau_scheduler.step(val_loss)


def compute_metrics(predictions, labels, threshold=0.5):
    preds_binary = (predictions > threshold).astype(int)
    metrics = {
        'accuracy': accuracy_score(labels, preds_binary),
        'precision': precision_score(labels, preds_binary, zero_division=0),
        'recall': recall_score(labels, preds_binary, zero_division=0),
        'f1': f1_score(labels, preds_binary, zero_division=0),
        'auc': roc_auc_score(labels, predictions) if len(np.unique(labels))>1 else 0.0
    }
    cm = confusion_matrix(labels, preds_binary)
    if cm.shape==(2,2):
        tn, fp, fn, tp = cm.ravel()
        metrics['fpr'] = fp/(fp+tn) if (fp+tn)>0 else 0
        metrics['fnr'] = fn/(fn+tp) if (fn+tp)>0 else 0
    return metrics


def check_deployment_criteria(metrics: Dict, config: Dict):
    criteria = {
        'f1': metrics.get('f1',0) >= config.get('deploy_min_f1',0.75),
        'precision': metrics.get('precision',0) >= config.get('deploy_min_precision',0.70),
        'recall': metrics.get('recall',0) >= config.get('deploy_min_recall',0.70),
        'fpr': metrics.get('fpr',1.0) <= config.get('deploy_max_fpr',0.10)
    }
    return all(criteria.values()), criteria


if __name__ == "__main__":
    print("✅ Trainer module loaded successfully!")

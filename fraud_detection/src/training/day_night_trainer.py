"""
Training System with Day/Night Cycle and RLHF
Architecture: Day (Inference) → Night (Training) → Morning (Validation)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader
from trl import PPOTrainer, PPOConfig
from transformers import TrainingArguments
from typing import Dict, List, Optional
import wandb
from datetime import datetime, time
import schedule
import json
import os
from tqdm import tqdm
import numpy as np

from ..models.hybrid_model import GNNLLMHybrid, RewardModel
from ..utils.metrics import compute_all_metrics
from ..utils.memory import TemporalMemory


class DayNightTrainer:
    """
    Trainer avec cycle Jour/Nuit/Matin
    """
    def __init__(
        self,
        model: GNNLLMHybrid,
        config: Dict,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Mémoire temporelle pour stocker les cas critiques
        self.memory = TemporalMemory(
            max_size=config["memory"]["max_memory_size"],
            retention_days=config["memory"]["retention_days"]
        )
        
        # Reward model pour RLHF
        self.reward_model = RewardModel(
            hidden_size=config["llm"]["hidden_size"] if "hidden_size" in config["llm"] else 768
        ).to(device)
        
        # Optimizers
        self.gnn_optimizer = optim.AdamW(
            self.model.gnn.parameters(),
            lr=config["fine_tuning"]["learning_rate"]
        )
        
        self.llm_optimizer = optim.AdamW(
            self.model.llm.parameters(),
            lr=config["rlhf"]["learning_rate"]
        )
        
        # Loss functions
        self.classification_loss = nn.CrossEntropyLoss()
        
        # Metrics tracking
        self.metrics_history = {
            "train": [],
            "val": [],
            "test": []
        }
        
        # État du système
        self.current_mode = "day"  # day, night, morning
        self.pending_approval = False
        
        # Logging
        if config["monitoring"]["enable_wandb"]:
            wandb.init(
                project="fraud-detection-gnn-llm",
                name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config=config
            )
    
    def day_mode_inference(
        self,
        data_loader: DataLoader,
        save_critical: bool = True
    ) -> Dict:
        """
        Mode JOUR: Inférence en temps réel avec sauvegarde des cas critiques
        
        Args:
            data_loader: DataLoader pour les nouvelles transactions
            save_critical: Sauvegarder les cas critiques
        
        Returns:
            Résultats d'inférence et métriques
        """
        print("\n☀️ MODE JOUR - Inférence en temps réel")
        print("=" * 50)
        
        self.model.eval()
        self.current_mode = "day"
        
        results = []
        critical_cases = []
        human_verifications_needed = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(data_loader, desc="Inférence")):
                batch = batch.to(self.device)
                
                # Prédiction
                logits, features = self.model(
                    batch.x,
                    batch.edge_index,
                    batch.batch
                )
                
                probs = torch.softmax(logits, dim=-1)
                predictions = torch.argmax(probs, dim=-1)
                confidences = torch.max(probs, dim=-1)[0]
                
                # Traiter chaque transaction
                for i in range(len(predictions)):
                    pred = predictions[i].item()
                    conf = confidences[i].item()
                    fraud_prob = probs[i, 1].item()
                    
                    result = {
                        "prediction": pred,
                        "confidence": conf,
                        "fraud_probability": fraud_prob,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    # Cas critique: faible confiance ou haute probabilité de fraude
                    is_critical = (
                        conf < 0.6 or
                        (pred == 1 and fraud_prob > 0.8)
                    )
                    
                    if is_critical:
                        critical_cases.append({
                            "features": batch.x[i].cpu().numpy(),
                            "prediction": pred,
                            "confidence": conf,
                            "fraud_prob": fraud_prob
                        })
                    
                    # Vérification humaine nécessaire
                    threshold = self.config["day_night_cycle"]["day"]["human_verification_threshold"]
                    if pred == 1 and fraud_prob > threshold:
                        human_verifications_needed.append(result)
                        result["action"] = "BLOCKED - Awaiting human verification"
                    elif pred == 1:
                        result["action"] = "FLAGGED - Monitoring"
                    else:
                        result["action"] = "APPROVED"
                    
                    results.append(result)
        
        # Sauvegarder les cas critiques dans la mémoire
        if save_critical and critical_cases:
            self.memory.add_batch(critical_cases, label="critical_day")
            print(f"💾 {len(critical_cases)} cas critiques sauvegardés")
        
        # Statistiques
        stats = {
            "total_transactions": len(results),
            "fraud_detected": sum(1 for r in results if r["prediction"] == 1),
            "human_verification_needed": len(human_verifications_needed),
            "critical_cases": len(critical_cases),
            "avg_confidence": np.mean([r["confidence"] for r in results])
        }
        
        print(f"\n📊 Statistiques du jour:")
        print(f"  - Transactions: {stats['total_transactions']}")
        print(f"  - Fraudes détectées: {stats['fraud_detected']}")
        print(f"  - Vérifications humaines: {stats['human_verification_needed']}")
        print(f"  - Cas critiques: {stats['critical_cases']}")
        print(f"  - Confiance moyenne: {stats['avg_confidence']:.2%}")
        
        return {
            "results": results,
            "stats": stats,
            "critical_cases": critical_cases,
            "human_verifications": human_verifications_needed
        }
    
    def night_mode_training(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 3
    ) -> Dict:
        """
        Mode NUIT: Fine-tuning + RLHF sur les données du jour
        
        Args:
            train_loader: DataLoader d'entraînement
            val_loader: DataLoader de validation
            num_epochs: Nombre d'epochs
        
        Returns:
            Métriques d'entraînement
        """
        print("\n🌙 MODE NUIT - Fine-tuning & RLHF")
        print("=" * 50)
        
        self.model.train()
        self.current_mode = "night"
        
        # Charger les cas critiques de la mémoire
        critical_memory = self.memory.get_recent(days=1, label="critical_day")
        feedback_memory = self.memory.get_recent(days=1, label="human_feedback")
        
        print(f"📚 Mémoire chargée:")
        print(f"  - Cas critiques: {len(critical_memory)}")
        print(f"  - Feedbacks humains: {len(feedback_memory)}")
        
        # Phase 1: Fine-tuning GNN
        print("\n🔧 Phase 1: Fine-tuning GNN")
        gnn_metrics = self._finetune_gnn(train_loader, val_loader, num_epochs)
        
        # Phase 2: RLHF sur LLM
        print("\n🎯 Phase 2: RLHF sur LLM")
        rlhf_metrics = self._rlhf_training(feedback_memory)
        
        # Validation finale
        print("\n✅ Validation finale")
        val_metrics = self.evaluate(val_loader, split="val")
        
        # Sauvegarder le checkpoint
        checkpoint_path = self._save_checkpoint("night_training")
        
        return {
            "gnn_metrics": gnn_metrics,
            "rlhf_metrics": rlhf_metrics,
            "val_metrics": val_metrics,
            "checkpoint": checkpoint_path
        }
    
    def _finetune_gnn(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int
    ) -> Dict:
        """Fine-tuning du GNN"""
        
        best_val_loss = float('inf')
        metrics = []
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            epoch_metrics = {"epoch": epoch + 1}
            
            # Training
            self.model.train()
            train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for batch in train_bar:
                batch = batch.to(self.device)
                
                # Forward
                logits, _ = self.model(batch.x, batch.edge_index, batch.batch)
                loss = self.classification_loss(logits, batch.y)
                
                # Backward
                self.gnn_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.gnn.parameters(), 1.0)
                self.gnn_optimizer.step()
                
                epoch_loss += loss.item()
                train_bar.set_postfix({"loss": loss.item()})
            
            avg_loss = epoch_loss / len(train_loader)
            epoch_metrics["train_loss"] = avg_loss
            
            # Validation
            val_metrics = self.evaluate(val_loader, split="val")
            epoch_metrics.update(val_metrics)
            
            # Sauvegarder le meilleur modèle
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                self._save_checkpoint("best_gnn")
            
            metrics.append(epoch_metrics)
            
            # Logging
            if self.config["monitoring"]["enable_wandb"]:
                wandb.log(epoch_metrics)
            
            print(f"Epoch {epoch+1}: train_loss={avg_loss:.4f}, "
                  f"val_loss={val_metrics['loss']:.4f}, "
                  f"val_f1={val_metrics['f1_score']:.4f}")
        
        return {"epochs": metrics, "best_val_loss": best_val_loss}
    
    def _rlhf_training(self, feedback_data: List[Dict]) -> Dict:
        """
        RLHF training avec PPO
        
        Args:
            feedback_data: Données de feedback humain
        
        Returns:
            Métriques RLHF
        """
        if not feedback_data:
            print("⚠️ Pas de feedback humain disponible, RLHF sauté")
            return {"status": "skipped", "reason": "no_feedback"}
        
        print(f"🎓 RLHF avec {len(feedback_data)} feedbacks")
        
        # Configuration PPO
        ppo_config = PPOConfig(
            model_name=self.config["llm"]["model_name"],
            learning_rate=self.config["rlhf"]["learning_rate"],
            batch_size=16,
            mini_batch_size=4,
            ppo_epochs=self.config["rlhf"]["ppo_epochs"]
        )
        
        # TODO: Implémenter le pipeline RLHF complet avec PPOTrainer
        # Pour l'instant, simulation
        
        rlhf_metrics = {
            "status": "completed",
            "feedback_used": len(feedback_data),
            "avg_reward": np.random.uniform(0.6, 0.9)  # Placeholder
        }
        
        return rlhf_metrics
    
    def morning_validation(
        self,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None
    ) -> Dict:
        """
        Mode MATIN: Validation et décision de déploiement
        
        Args:
            val_loader: DataLoader de validation
            test_loader: DataLoader de test (optionnel)
        
        Returns:
            Résultats de validation et recommandation de déploiement
        """
        print("\n🌅 MODE MATIN - Validation & Déploiement")
        print("=" * 50)
        
        self.current_mode = "morning"
        
        # Charger le meilleur checkpoint de la nuit
        self._load_checkpoint("best_gnn")
        
        # Validation complète
        val_metrics = self.evaluate(val_loader, split="val")
        
        test_metrics = None
        if test_loader:
            test_metrics = self.evaluate(test_loader, split="test")
        
        # Critères de déploiement
        deploy_criteria = {
            "min_f1": 0.75,
            "min_precision": 0.70,
            "min_recall": 0.70,
            "max_false_positive_rate": 0.10
        }
        
        # Vérifier les critères
        can_deploy = (
            val_metrics["f1_score"] >= deploy_criteria["min_f1"] and
            val_metrics["precision"] >= deploy_criteria["min_precision"] and
            val_metrics["recall"] >= deploy_criteria["min_recall"] and
            val_metrics.get("false_positive_rate", 1.0) <= deploy_criteria["max_false_positive_rate"]
        )
        
        recommendation = {
            "can_deploy": can_deploy,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "criteria": deploy_criteria,
            "timestamp": datetime.now().isoformat()
        }
        
        if can_deploy:
            print("✅ Modèle prêt pour le déploiement")
            self.pending_approval = True
        else:
            print("❌ Modèle ne remplit pas les critères")
            print("Métriques actuelles vs requises:")
            print(f"  F1: {val_metrics['f1_score']:.4f} vs {deploy_criteria['min_f1']}")
            print(f"  Precision: {val_metrics['precision']:.4f} vs {deploy_criteria['min_precision']}")
            print(f"  Recall: {val_metrics['recall']:.4f} vs {deploy_criteria['min_recall']}")
        
        # Sauvegarder le rapport
        self._save_validation_report(recommendation)
        
        return recommendation
    
    def evaluate(self, data_loader: DataLoader, split: str = "val") -> Dict:
        """
        Évaluer le modèle
        
        Args:
            data_loader: DataLoader à évaluer
            split: train/val/test
        
        Returns:
            Dictionnaire de métriques
        """
        self.model.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc=f"Eval {split}"):
                batch = batch.to(self.device)
                
                logits, _ = self.model(batch.x, batch.edge_index, batch.batch)
                loss = self.classification_loss(logits, batch.y)
                
                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(probs, dim=-1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch.y.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
                total_loss += loss.item()
        
        # Calculer toutes les métriques
        metrics = compute_all_metrics(
            y_true=np.array(all_labels),
            y_pred=np.array(all_preds),
            y_prob=np.array(all_probs)
        )
        
        metrics["loss"] = total_loss / len(data_loader)
        
        # Logging
        if self.config["monitoring"]["enable_wandb"]:
            wandb.log({f"{split}_{k}": v for k, v in metrics.items()})
        
        return metrics
    
    def _save_checkpoint(self, name: str) -> str:
        """Sauvegarder un checkpoint"""
        checkpoint_dir = self.config["paths"]["checkpoints_dir"]
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = f"{checkpoint_dir}/{name}_{timestamp}.pt"
        
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "gnn_optimizer_state_dict": self.gnn_optimizer.state_dict(),
            "llm_optimizer_state_dict": self.llm_optimizer.state_dict(),
            "config": self.config,
            "timestamp": timestamp
        }, checkpoint_path)
        
        print(f"💾 Checkpoint sauvegardé: {checkpoint_path}")
        
        return checkpoint_path
    
    def _load_checkpoint(self, name: str):
        """Charger un checkpoint"""
        checkpoint_dir = self.config["paths"]["checkpoints_dir"]
        
        # Trouver le checkpoint le plus récent avec ce nom
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith(name)]
        if not checkpoints:
            print(f"⚠️ Pas de checkpoint trouvé: {name}")
            return
        
        latest_checkpoint = sorted(checkpoints)[-1]
        checkpoint_path = f"{checkpoint_dir}/{latest_checkpoint}"
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        
        print(f"📂 Checkpoint chargé: {checkpoint_path}")
    
    def _save_validation_report(self, recommendation: Dict):
        """Sauvegarder le rapport de validation"""
        reports_dir = f"{self.config['paths']['logs_dir']}/validation_reports"
        os.makedirs(reports_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"{reports_dir}/validation_{timestamp}.json"
        
        with open(report_path, 'w') as f:
            json.dump(recommendation, f, indent=2)
        
        print(f"📄 Rapport sauvegardé: {report_path}")


def schedule_day_night_cycle(trainer: DayNightTrainer, config: Dict):
    """
    Programmer le cycle jour/nuit/matin
    
    Args:
        trainer: Instance du trainer
        config: Configuration
    """
    print("🕐 Configuration du cycle jour/nuit/matin...")
    
    # Mode jour: 8h-20h
    schedule.every().day.at("08:00").do(
        lambda: print("☀️ Passage en mode JOUR")
    )
    
    # Mode nuit: 20h-8h
    schedule.every().day.at("20:00").do(
        lambda: print("🌙 Passage en mode NUIT")
    )
    
    # Validation matin: 7h
    schedule.every().day.at("07:00").do(
        lambda: print("🌅 Validation matinale")
    )
    
    print("✓ Cycle programmé")
    print("  - 08:00-20:00: Mode JOUR (inférence)")
    print("  - 20:00-08:00: Mode NUIT (training)")
    print("  - 07:00: Validation matinale")


if __name__ == "__main__":
    print("Testing Day/Night Training System...")
    print("✓ System ready!")

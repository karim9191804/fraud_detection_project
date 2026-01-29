"""
Temporal Memory System for Critical Cases and Human Feedback
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pickle
import numpy as np


class TemporalMemory:
    """
    Système de mémoire temporelle pour stocker:
    - Cas critiques détectés pendant le jour
    - Feedback humain sur les prédictions
    - Historique des transactions suspectes
    """
    def __init__(
        self,
        storage_dir: str = "memory",
        max_size: int = 10000,
        retention_days: int = 30
    ):
        self.storage_dir = storage_dir
        self.max_size = max_size
        self.retention_days = retention_days
        
        os.makedirs(storage_dir, exist_ok=True)
        
        self.memory = {
            "critical_day": [],
            "human_feedback": [],
            "suspicious": [],
            "verified_fraud": [],
            "false_positives": []
        }
        
        self._load_from_disk()
    
    def add(
        self,
        data: Dict,
        label: str = "critical_day",
        metadata: Optional[Dict] = None
    ):
        """
        Ajouter un élément à la mémoire
        
        Args:
            data: Données à stocker
            label: Type de mémoire
            metadata: Métadonnées supplémentaires
        """
        entry = {
            "data": data,
            "timestamp": datetime.now().isoformat(),
            "label": label
        }
        
        if metadata:
            entry["metadata"] = metadata
        
        if label not in self.memory:
            self.memory[label] = []
        
        self.memory[label].append(entry)
        
        # Limiter la taille
        if len(self.memory[label]) > self.max_size:
            self.memory[label] = self.memory[label][-self.max_size:]
        
        # Sauvegarder
        self._save_to_disk()
    
    def add_batch(
        self,
        data_list: List[Dict],
        label: str = "critical_day",
        metadata: Optional[Dict] = None
    ):
        """
        Ajouter plusieurs éléments
        
        Args:
            data_list: Liste de données
            label: Type de mémoire
            metadata: Métadonnées communes
        """
        for data in data_list:
            self.add(data, label, metadata)
    
    def get_recent(
        self,
        days: int = 1,
        label: Optional[str] = None
    ) -> List[Dict]:
        """
        Récupérer les éléments récents
        
        Args:
            days: Nombre de jours à récupérer
            label: Filtrer par type (None = tous)
        
        Returns:
            Liste d'éléments
        """
        cutoff = datetime.now() - timedelta(days=days)
        recent = []
        
        memories_to_check = [label] if label else self.memory.keys()
        
        for mem_label in memories_to_check:
            if mem_label not in self.memory:
                continue
            
            for entry in self.memory[mem_label]:
                timestamp = datetime.fromisoformat(entry["timestamp"])
                if timestamp >= cutoff:
                    recent.append(entry)
        
        return recent
    
    def get_all(self, label: Optional[str] = None) -> List[Dict]:
        """
        Récupérer tous les éléments
        
        Args:
            label: Filtrer par type (None = tous)
        
        Returns:
            Liste d'éléments
        """
        if label:
            return self.memory.get(label, [])
        
        all_entries = []
        for entries in self.memory.values():
            all_entries.extend(entries)
        
        return all_entries
    
    def add_human_feedback(
        self,
        transaction_id: str,
        predicted_label: int,
        true_label: int,
        confidence: float,
        feedback_text: Optional[str] = None
    ):
        """
        Ajouter un feedback humain
        
        Args:
            transaction_id: ID de la transaction
            predicted_label: Label prédit (0=légitime, 1=fraude)
            true_label: Vrai label après vérification humaine
            confidence: Confiance du modèle
            feedback_text: Commentaire textuel (optionnel)
        """
        feedback = {
            "transaction_id": transaction_id,
            "predicted_label": predicted_label,
            "true_label": true_label,
            "confidence": confidence,
            "correct": predicted_label == true_label,
            "feedback_text": feedback_text
        }
        
        self.add(feedback, label="human_feedback")
        
        # Classer aussi dans les catégories appropriées
        if predicted_label == 1 and true_label == 0:
            self.add(feedback, label="false_positives")
        elif predicted_label == 0 and true_label == 1:
            # Faux négatif, ajouter comme fraude manquée
            self.add(feedback, label="verified_fraud")
        elif true_label == 1:
            # Vrai positif
            self.add(feedback, label="verified_fraud")
    
    def get_feedback_stats(self, days: int = 7) -> Dict:
        """
        Statistiques sur le feedback humain
        
        Args:
            days: Nombre de jours à analyser
        
        Returns:
            Statistiques
        """
        feedbacks = self.get_recent(days=days, label="human_feedback")
        
        if not feedbacks:
            return {
                "total": 0,
                "correct": 0,
                "incorrect": 0,
                "accuracy": 0.0
            }
        
        total = len(feedbacks)
        correct = sum(1 for f in feedbacks if f["data"]["correct"])
        
        return {
            "total": total,
            "correct": correct,
            "incorrect": total - correct,
            "accuracy": correct / total if total > 0 else 0.0,
            "false_positives": len(self.get_recent(days=days, label="false_positives")),
            "verified_frauds": len(self.get_recent(days=days, label="verified_fraud"))
        }
    
    def cleanup_old_entries(self):
        """Nettoyer les anciennes entrées"""
        cutoff = datetime.now() - timedelta(days=self.retention_days)
        
        for label in self.memory:
            self.memory[label] = [
                entry for entry in self.memory[label]
                if datetime.fromisoformat(entry["timestamp"]) >= cutoff
            ]
        
        self._save_to_disk()
        print(f"✓ Nettoyage effectué (rétention: {self.retention_days} jours)")
    
    def get_critical_patterns(self, days: int = 7) -> Dict:
        """
        Analyser les patterns des cas critiques
        
        Args:
            days: Nombre de jours à analyser
        
        Returns:
            Patterns détectés
        """
        critical_cases = self.get_recent(days=days, label="critical_day")
        
        if not critical_cases:
            return {"patterns": [], "count": 0}
        
        # Extraire les features
        features_list = []
        for case in critical_cases:
            if "features" in case["data"]:
                features_list.append(case["data"]["features"])
        
        if not features_list:
            return {"patterns": [], "count": len(critical_cases)}
        
        # Statistiques basiques
        features_array = np.array(features_list)
        
        patterns = {
            "count": len(critical_cases),
            "avg_features": features_array.mean(axis=0).tolist(),
            "std_features": features_array.std(axis=0).tolist(),
            "avg_confidence": np.mean([
                case["data"]["confidence"]
                for case in critical_cases
                if "confidence" in case["data"]
            ])
        }
        
        return patterns
    
    def export_for_training(
        self,
        label: str,
        days: Optional[int] = None,
        format: str = "list"
    ) -> List[Dict]:
        """
        Exporter les données pour le training
        
        Args:
            label: Type de mémoire à exporter
            days: Nombre de jours (None = tout)
            format: Format de sortie ('list', 'numpy')
        
        Returns:
            Données formatées
        """
        if days:
            entries = self.get_recent(days=days, label=label)
        else:
            entries = self.get_all(label=label)
        
        if format == "list":
            return [entry["data"] for entry in entries]
        elif format == "numpy":
            # Extraire les features si disponibles
            features = []
            labels = []
            
            for entry in entries:
                if "features" in entry["data"]:
                    features.append(entry["data"]["features"])
                if "true_label" in entry["data"]:
                    labels.append(entry["data"]["true_label"])
            
            return {
                "features": np.array(features) if features else None,
                "labels": np.array(labels) if labels else None
            }
        
        return entries
    
    def _save_to_disk(self):
        """Sauvegarder la mémoire sur disque"""
        for label, entries in self.memory.items():
            filepath = os.path.join(self.storage_dir, f"{label}.pkl")
            with open(filepath, 'wb') as f:
                pickle.dump(entries, f)
    
    def _load_from_disk(self):
        """Charger la mémoire depuis le disque"""
        for label in self.memory.keys():
            filepath = os.path.join(self.storage_dir, f"{label}.pkl")
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    self.memory[label] = pickle.load(f)
                print(f"✓ Chargé {len(self.memory[label])} entrées de {label}")
    
    def get_summary(self) -> Dict:
        """Résumé de la mémoire"""
        summary = {
            "total_entries": sum(len(entries) for entries in self.memory.values()),
            "by_type": {
                label: len(entries)
                for label, entries in self.memory.items()
            },
            "oldest_entry": None,
            "newest_entry": None
        }
        
        # Trouver les entrées les plus anciennes/récentes
        all_entries = self.get_all()
        if all_entries:
            timestamps = [
                datetime.fromisoformat(e["timestamp"])
                for e in all_entries
            ]
            summary["oldest_entry"] = min(timestamps).isoformat()
            summary["newest_entry"] = max(timestamps).isoformat()
        
        return summary
    
    def clear(self, label: Optional[str] = None):
        """
        Vider la mémoire
        
        Args:
            label: Type à vider (None = tout)
        """
        if label:
            self.memory[label] = []
        else:
            for key in self.memory:
                self.memory[key] = []
        
        self._save_to_disk()
        print(f"✓ Mémoire vidée: {label if label else 'tout'}")


class FeedbackCollector:
    """
    Collecteur de feedback pour RLHF
    """
    def __init__(self, memory: TemporalMemory):
        self.memory = memory
    
    def collect_feedback(
        self,
        transaction_id: str,
        predicted_label: int,
        confidence: float,
        explanation: str
    ) -> Dict:
        """
        Collecter un feedback utilisateur
        
        Args:
            transaction_id: ID de la transaction
            predicted_label: Prédiction du modèle
            confidence: Confiance du modèle
            explanation: Explication générée par le LLM
        
        Returns:
            Feedback structuré
        """
        print(f"\n{'=' * 60}")
        print(f"🔍 Vérification Humaine Requise")
        print(f"{'=' * 60}")
        print(f"Transaction ID: {transaction_id}")
        print(f"Prédiction: {'FRAUDE' if predicted_label == 1 else 'LÉGITIME'}")
        print(f"Confiance: {confidence:.2%}")
        print(f"\nExplication:")
        print(f"{explanation}")
        print(f"{'=' * 60}")
        
        # Simuler une réponse humaine (dans un système réel, interface web)
        print("\nVotre décision:")
        print("1. FRAUDE confirmée")
        print("2. FAUX POSITIF (légitime)")
        print("3. BESOIN DE PLUS D'INFOS")
        
        # Pour le test, retourner un feedback simulé
        true_label = predicted_label  # Placeholder
        
        feedback = {
            "transaction_id": transaction_id,
            "predicted_label": predicted_label,
            "true_label": true_label,
            "confidence": confidence,
            "explanation": explanation,
            "feedback_quality": "good"  # good, neutral, poor
        }
        
        self.memory.add_human_feedback(
            transaction_id=transaction_id,
            predicted_label=predicted_label,
            true_label=true_label,
            confidence=confidence,
            feedback_text=f"Explanation quality: good"
        )
        
        return feedback


if __name__ == "__main__":
    # Test du système de mémoire
    print("Testing Temporal Memory System...")
    
    # Créer une mémoire
    memory = TemporalMemory(storage_dir="test_memory")
    
    # Ajouter des cas critiques
    for i in range(5):
        memory.add(
            {
                "features": np.random.rand(10).tolist(),
                "prediction": 1,
                "confidence": 0.5 + i * 0.05
            },
            label="critical_day"
        )
    
    # Ajouter du feedback
    memory.add_human_feedback(
        transaction_id="TX123",
        predicted_label=1,
        true_label=0,
        confidence=0.85,
        feedback_text="Clear false positive"
    )
    
    # Récupérer les données récentes
    recent = memory.get_recent(days=1)
    print(f"\n✓ Entrées récentes: {len(recent)}")
    
    # Statistiques de feedback
    stats = memory.get_feedback_stats(days=7)
    print(f"✓ Statistiques de feedback: {stats}")
    
    # Résumé
    summary = memory.get_summary()
    print(f"\n✓ Résumé de la mémoire:")
    print(f"  Total: {summary['total_entries']} entrées")
    print(f"  Par type: {summary['by_type']}")
    
    # Nettoyer
    import shutil
    shutil.rmtree("test_memory")
    
    print("\n✓ Memory system tests passed!")

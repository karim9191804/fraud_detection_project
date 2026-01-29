"""
Metrics Utilities for Fraud Detection
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray = None
) -> Dict:
    """
    Calculer toutes les métriques pertinentes pour la détection de fraude
    
    Args:
        y_true: Labels réels
        y_pred: Prédictions (classes)
        y_prob: Probabilités de la classe positive (optionnel)
    
    Returns:
        Dictionnaire de métriques
    """
    metrics = {}
    
    # Métriques de base
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
    metrics["f1_score"] = f1_score(y_true, y_pred, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    metrics["true_negatives"] = int(tn)
    metrics["false_positives"] = int(fp)
    metrics["false_negatives"] = int(fn)
    metrics["true_positives"] = int(tp)
    
    # Taux spécifiques
    metrics["false_positive_rate"] = fp / (fp + tn) if (fp + tn) > 0 else 0
    metrics["false_negative_rate"] = fn / (fn + tp) if (fn + tp) > 0 else 0
    metrics["true_negative_rate"] = tn / (tn + fp) if (tn + fp) > 0 else 0  # Specificity
    
    # Fraud Detection Rate (même que recall)
    metrics["fraud_detection_rate"] = metrics["recall"]
    
    # Métriques basées sur les probabilités
    if y_prob is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
            metrics["average_precision"] = average_precision_score(y_true, y_prob)
        except ValueError:
            # Peut arriver si une seule classe est présente
            metrics["roc_auc"] = 0.0
            metrics["average_precision"] = 0.0
    
    return metrics


def print_metrics(metrics: Dict, title: str = "Metrics"):
    """
    Afficher les métriques de façon lisible
    
    Args:
        metrics: Dictionnaire de métriques
        title: Titre de l'affichage
    """
    print(f"\n{'=' * 60}")
    print(f"{title:^60}")
    print(f"{'=' * 60}")
    
    # Métriques principales
    print("\n📊 Métriques Principales:")
    main_metrics = ["accuracy", "precision", "recall", "f1_score"]
    for metric in main_metrics:
        if metric in metrics:
            print(f"  {metric:.<30} {metrics[metric]:>6.2%}")
    
    # Confusion Matrix
    print("\n📈 Confusion Matrix:")
    if all(k in metrics for k in ["true_negatives", "false_positives", "false_negatives", "true_positives"]):
        print(f"  True Negatives (TN)........... {metrics['true_negatives']:>6}")
        print(f"  False Positives (FP).......... {metrics['false_positives']:>6}")
        print(f"  False Negatives (FN).......... {metrics['false_negatives']:>6}")
        print(f"  True Positives (TP)........... {metrics['true_positives']:>6}")
    
    # Taux
    print("\n📉 Taux:")
    rate_metrics = ["false_positive_rate", "false_negative_rate", "fraud_detection_rate"]
    for metric in rate_metrics:
        if metric in metrics:
            print(f"  {metric:.<30} {metrics[metric]:>6.2%}")
    
    # AUC metrics
    print("\n🎯 AUC Metrics:")
    auc_metrics = ["roc_auc", "average_precision"]
    for metric in auc_metrics:
        if metric in metrics:
            print(f"  {metric:.<30} {metrics[metric]:>6.4f}")
    
    print(f"{'=' * 60}\n")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str = None,
    class_names: list = ["Legitimate", "Fraud"]
) -> plt.Figure:
    """
    Tracer la matrice de confusion
    
    Args:
        y_true: Labels réels
        y_pred: Prédictions
        save_path: Chemin de sauvegarde (optionnel)
        class_names: Noms des classes
    
    Returns:
        Figure matplotlib
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )
    
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title('Confusion Matrix')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix sauvegardée: {save_path}")
    
    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: str = None
) -> Tuple[plt.Figure, float]:
    """
    Tracer la courbe ROC
    
    Args:
        y_true: Labels réels
        y_prob: Probabilités prédites
        save_path: Chemin de sauvegarde (optionnel)
    
    Returns:
        (Figure, AUC score)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})', linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ ROC curve sauvegardée: {save_path}")
    
    return fig, auc


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: str = None
) -> Tuple[plt.Figure, float]:
    """
    Tracer la courbe Precision-Recall
    
    Args:
        y_true: Labels réels
        y_prob: Probabilités prédites
        save_path: Chemin de sauvegarde (optionnel)
    
    Returns:
        (Figure, Average Precision score)
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(recall, precision, label=f'PR Curve (AP = {ap:.4f})', linewidth=2)
    
    # Baseline (proportion de positifs)
    baseline = y_true.sum() / len(y_true)
    ax.plot([0, 1], [baseline, baseline], 'k--', label=f'Baseline ({baseline:.4f})')
    
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ PR curve sauvegardée: {save_path}")
    
    return fig, ap


def get_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list = ["Legitimate", "Fraud"]
) -> str:
    """
    Générer un rapport de classification complet
    
    Args:
        y_true: Labels réels
        y_pred: Prédictions
        class_names: Noms des classes
    
    Returns:
        Rapport en texte
    """
    return classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    )


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f1"
) -> Tuple[float, Dict]:
    """
    Trouver le seuil optimal pour maximiser une métrique
    
    Args:
        y_true: Labels réels
        y_prob: Probabilités prédites
        metric: Métrique à optimiser ('f1', 'precision', 'recall')
    
    Returns:
        (optimal_threshold, metrics_at_threshold)
    """
    thresholds = np.linspace(0, 1, 101)
    best_threshold = 0.5
    best_score = 0
    best_metrics = {}
    
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        
        if metric == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == "precision":
            score = precision_score(y_true, y_pred, zero_division=0)
        elif metric == "recall":
            score = recall_score(y_true, y_pred, zero_division=0)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
            best_metrics = compute_all_metrics(y_true, y_pred, y_prob)
    
    print(f"\n🎯 Seuil optimal pour {metric}: {best_threshold:.4f}")
    print(f"   Score: {best_score:.4f}")
    
    return best_threshold, best_metrics


class MetricsTracker:
    """
    Suivi des métriques au fil du temps
    """
    def __init__(self):
        self.history = {
            "train": [],
            "val": [],
            "test": []
        }
    
    def add(self, metrics: Dict, split: str, epoch: int = None):
        """Ajouter des métriques"""
        entry = metrics.copy()
        if epoch is not None:
            entry["epoch"] = epoch
        
        self.history[split].append(entry)
    
    def get_best(self, split: str, metric: str = "f1_score") -> Dict:
        """Obtenir la meilleure métrique"""
        if not self.history[split]:
            return {}
        
        return max(self.history[split], key=lambda x: x.get(metric, 0))
    
    def plot_history(self, metric: str = "f1_score", save_path: str = None) -> plt.Figure:
        """Tracer l'évolution d'une métrique"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for split in ["train", "val", "test"]:
            if self.history[split]:
                epochs = [entry.get("epoch", i) for i, entry in enumerate(self.history[split])]
                values = [entry.get(metric, 0) for entry in self.history[split]]
                
                ax.plot(epochs, values, marker='o', label=split.capitalize())
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric.replace("_", " ").title()} Over Time')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


if __name__ == "__main__":
    # Test des métriques
    print("Testing Metrics...")
    
    # Données de test
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 1000)
    y_pred = np.random.randint(0, 2, 1000)
    y_prob = np.random.random(1000)
    
    # Calculer métriques
    metrics = compute_all_metrics(y_true, y_pred, y_prob)
    print_metrics(metrics, "Test Metrics")
    
    # Rapport
    report = get_classification_report(y_true, y_pred)
    print("\nClassification Report:")
    print(report)
    
    # Seuil optimal
    threshold, _ = find_optimal_threshold(y_true, y_prob, metric="f1")
    
    print("\n✓ Metrics tests passed!")

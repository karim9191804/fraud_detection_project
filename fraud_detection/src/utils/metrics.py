"""
Métriques d'évaluation pour la détection de fraude
"""

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
import numpy as np


def compute_all_metrics(y_true, y_pred, y_probs=None):
    """
    Calculer toutes les métriques de classification
    
    Args:
        y_true: Labels vrais
        y_pred: Prédictions
        y_probs: Probabilités (optionnel, pour ROC-AUC)
    
    Returns:
        dict: Dictionnaire avec toutes les métriques
    """
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0)
    }
    
    # ROC-AUC si probabilités fournies
    if y_probs is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_probs)
        except:
            metrics['roc_auc'] = 0.0
    
    # Matrice de confusion
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['true_positives'] = int(tp)
        
        # Spécificité
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return metrics


def print_metrics(metrics, title="Métriques"):
    """
    Afficher les métriques de manière formatée
    
    Args:
        metrics: Dict de métriques
        title: Titre à afficher
    """
    
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key:20s}: {value:.4f}")
        elif isinstance(value, int):
            print(f"  {key:20s}: {value:,}")
    
    print(f"{'='*60}\n")


def calculate_fraud_detection_score(metrics):
    """
    Calculer un score global de détection de fraude
    
    Pondération:
    - F1-Score: 40%
    - Recall: 30% (important pour détecter les fraudes)
    - Precision: 20% (éviter faux positifs)
    - ROC-AUC: 10%
    
    Args:
        metrics: Dict de métriques
    
    Returns:
        float: Score global entre 0 et 1
    """
    
    f1 = metrics.get('f1_score', 0)
    recall = metrics.get('recall', 0)
    precision = metrics.get('precision', 0)
    roc_auc = metrics.get('roc_auc', 0)
    
    score = (
        0.40 * f1 +
        0.30 * recall +
        0.20 * precision +
        0.10 * roc_auc
    )
    
    return score


def is_model_deployable(metrics, thresholds=None):
    """
    Vérifier si le modèle est déployable selon des seuils
    
    Args:
        metrics: Dict de métriques
        thresholds: Dict de seuils (optionnel)
    
    Returns:
        tuple: (bool déployable, list raisons)
    """
    
    if thresholds is None:
        thresholds = {
            'f1_score': 0.75,
            'precision': 0.70,
            'recall': 0.70
        }
    
    deployable = True
    reasons = []
    
    for metric, threshold in thresholds.items():
        value = metrics.get(metric, 0)
        if value < threshold:
            deployable = False
            reasons.append(f"{metric} = {value:.4f} < {threshold}")
    
    return deployable, reasons

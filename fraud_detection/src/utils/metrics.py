"""
Métriques pour évaluation
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)


def compute_all_metrics(y_true, y_pred, y_probs=None, threshold=0.5):
    """
    Calculer toutes les métriques
    
    Args:
        y_true: Labels vrais
        y_pred: Prédictions (0 ou 1)
        y_probs: Probabilités (optionnel, pour ROC-AUC)
        threshold: Seuil de classification
    
    Returns:
        dict: Toutes les métriques
    """
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
    }
    
    # ROC-AUC si probas fournies
    if y_probs is not None and len(np.unique(y_true)) > 1:
        metrics['roc_auc'] = roc_auc_score(y_true, y_probs)
    else:
        metrics['roc_auc'] = 0.0
    
    # Matrice de confusion
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['true_positives'] = int(tp)
        
        # Taux d'erreur
        metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    return metrics

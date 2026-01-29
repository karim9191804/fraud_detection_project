"""
Modèles de détection de fraude
"""

from .gnn_model import GNNModel
from .hybrid_model import HybridFraudDetector

__all__ = ['GNNModel', 'HybridFraudDetector']

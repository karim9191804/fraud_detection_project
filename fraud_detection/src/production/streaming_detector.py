"""
Streaming Fraud Detector
Détection de fraude en temps réel transaction par transaction
MODE JOUR: Traitement streaming 24/7
"""

import torch
import numpy as np
from datetime import datetime
from collections import deque
import json
import threading
import queue
import time


class StreamingFraudDetector:
    """
    Détecteur de fraude en streaming
    
    • Traite transactions une par une en temps réel
    • Génère explications LLM
    • Détecte et stocke cas critiques
    • Thread-safe pour production
    """
    
    def __init__(self, model, llm_wrapper, config, device):
        self.model = model
        self.llm_wrapper = llm_wrapper
        self.config = config
        self.device = device
        
        # Buffer pour cas critiques (MODE JOUR)
        self.critical_cases_buffer = deque(maxlen=config.get('max_buffer_size', 10000))
        
        # Queue pour processing asynchrone
        self.transaction_queue = queue.Queue()
        self.results_queue = queue.Queue()
        
        # Statistiques temps réel
        self.stats = {
            'total_transactions': 0,
            'frauds_detected': 0,
            'critical_cases': 0,
            'avg_latency_ms': 0,
            'throughput_per_sec': 0
        }
        
        # Lock pour thread-safety
        self.lock = threading.Lock()
        
        # Model en mode eval permanent
        self.model.eval()
        
        print("✅ Streaming Fraud Detector initialisé")
        print(f"   Device: {device}")
        print(f"   Buffer size: {config.get('max_buffer_size', 10000)}")
    
    def preprocess_transaction(self, transaction):
        """
        Préparer une transaction pour inférence
        
        Args:
            transaction: dict avec features de la transaction
        
        Returns:
            Tensor prêt pour le modèle
        """
        
        # Extraire features
        features = transaction.get('features', [])
        
        # Convertir en tensor
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        
        return x.to(self.device)
    
    def detect_fraud(self, transaction):
        """
        Détecter fraude sur UNE transaction en temps réel
        
        Args:
            transaction: dict {
                'transaction_id': str,
                'features': list[float],
                'amount': float,
                'merchant': str,
                ...
            }
        
        Returns:
            dict: Résultat de détection avec explication
        """
        
        start_time = time.time()
        
        with torch.no_grad():
            # 1. Preprocessing
            x = self.preprocess_transaction(transaction)
            
            # 2. GNN Forward
            if hasattr(self.model, 'gnn'):
                embeddings = self.model.gnn.conv1.lin(x)
                logits = self.model.gnn.classifier(embeddings)
            else:
                logits, embeddings = self.model(x, None, None)
            
            # 3. Prédiction
            probs = torch.softmax(logits, dim=1)[0]
            fraud_prob = float(probs[1])
            is_fraud = fraud_prob > 0.5
            confidence = float(probs.max())
            
            # 4. Explication
            explanation = self.generate_explanation(
                transaction, 
                fraud_prob, 
                confidence
            )
            
            # 5. Détection cas critique
            is_critical = confidence < self.config.get('confidence_threshold', 0.75)
            
            if is_critical:
                self.save_critical_case(transaction, fraud_prob, confidence, explanation)
        
        # Latence
        latency_ms = (time.time() - start_time) * 1000
        
        # Stats
        with self.lock:
            self.stats['total_transactions'] += 1
            if is_fraud:
                self.stats['frauds_detected'] += 1
            if is_critical:
                self.stats['critical_cases'] += 1
            
            alpha = 0.1
            self.stats['avg_latency_ms'] = (
                alpha * latency_ms + 
                (1 - alpha) * self.stats['avg_latency_ms']
            )
        
        result = {
            'transaction_id': transaction.get('transaction_id', 'unknown'),
            'is_fraud': is_fraud,
            'fraud_probability': fraud_prob,
            'confidence': confidence,
            'explanation': explanation,
            'is_critical': is_critical,
            'latency_ms': latency_ms,
            'timestamp': datetime.now().isoformat()
        }
        
        return result
    
    def generate_explanation(self, transaction, fraud_prob, confidence):
        """
        Générer explication humaine avec LLM
        """
        
        if fraud_prob > 0.8:
            risk_level = "ÉLEVÉ"
            emoji = "🚨"
        elif fraud_prob > 0.5:
            risk_level = "MOYEN"
            emoji = "⚠️"
        else:
            risk_level = "FAIBLE"
            emoji = "✅"
        
        explanation = f"{emoji} Risque de fraude: {risk_level}\n"
        explanation += f"Probabilité: {fraud_prob*100:.1f}%\n"
        explanation += f"Confiance: {confidence*100:.1f}%\n\n"
        
        amount = transaction.get('amount', 0)
        merchant = transaction.get('merchant', 'Unknown')
        
        explanation += f"Montant: {amount}€\n"
        explanation += f"Marchand: {merchant}\n"
        
        if fraud_prob > 0.5:
            explanation += "\n🔍 Signaux suspects:\n"
            if amount > 1000:
                explanation += "  • Montant inhabituel\n"
            if confidence < 0.75:
                explanation += "  • Pattern ambigu nécessitant validation\n"
        else:
            explanation += "\n✓ Transaction normale\n"
        
        return explanation
    
    def save_critical_case(self, transaction, fraud_prob, confidence, explanation):
        """
        Sauvegarder cas critique pour review MODE NUIT
        """
        
        critical_case = {
            'transaction_id': transaction.get('transaction_id'),
            'features': transaction.get('features'),
            'fraud_probability': fraud_prob,
            'confidence': confidence,
            'explanation': explanation,
            'timestamp': datetime.now().isoformat(),
            'reviewed': False,
            'human_feedback': None
        }
        
        with self.lock:
            self.critical_cases_buffer.append(critical_case)
    
    def get_critical_cases(self, clear=False):
        """
        Récupérer cas critiques accumulés (pour MODE NUIT)
        """
        
        with self.lock:
            cases = list(self.critical_cases_buffer)
            
            if clear:
                self.critical_cases_buffer.clear()
        
        return cases
    
    def get_stats(self):
        """Statistiques temps réel"""
        
        with self.lock:
            return self.stats.copy()
    
    def process_stream(self, transaction_stream):
        """
        Traiter un flux de transactions
        """
        
        for transaction in transaction_stream:
            result = self.detect_fraud(transaction)
            yield result


class TransactionStreamSimulator:
    """
    Simulateur de flux de transactions pour tests
    """
    
    def __init__(self, dataset, delay_ms=100):
        self.dataset = dataset
        self.delay_ms = delay_ms
        self.index = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.index >= len(self.dataset):
            raise StopIteration
        
        time.sleep(self.delay_ms / 1000.0)
        
        transaction = {
            'transaction_id': f"TX_{self.index:08d}",
            'features': self.dataset[self.index]['features'],
            'amount': np.random.uniform(10, 5000),
            'merchant': f"Merchant_{np.random.randint(1, 100)}",
            'timestamp': datetime.now().isoformat()
        }
        
        self.index += 1
        return transaction

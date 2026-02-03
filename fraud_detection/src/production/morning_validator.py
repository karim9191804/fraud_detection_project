"""
Morning Validator
Validation et décision de déploiement le matin
MODE MATIN: Validation humaine + décision déploiement (6h-8h)
"""

import torch
from datetime import datetime
import json
import time


class MorningValidator:
    """
    Validation MODE MATIN
    
    • Valide modèle amélioré pendant la nuit
    • Présente résultats à l'expert humain
    • Expert décide: déployer nouvelle version ou rollback
    • Génère rapport quotidien
    """
    
    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        
        # Thresholds de déploiement
        self.min_f1 = config.get('deploy_min_f1', 0.75)
        self.min_precision = config.get('deploy_min_precision', 0.70)
        self.min_recall = config.get('deploy_min_recall', 0.70)
        
        print("✅ Morning Validator initialisé")
        print(f"   Min F1: {self.min_f1}")
        print(f"   Min Precision: {self.min_precision}")
        print(f"   Min Recall: {self.min_recall}")
    
    def validate_improved_model(self, val_data, test_data=None):
        """
        Validation complète du modèle amélioré
        """
        
        print(f"\n{'='*60}")
        print(f"🌅 MORNING VALIDATION")
        print(f"{'='*60}")
        print(f"   Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        
        try:
            from src.utils.metrics import compute_all_metrics
        except:
            print("⚠️  Module metrics non disponible")
            return {}
        
        self.model.eval()
        
        # Validation set
        with torch.no_grad():
            val_data = val_data.to(self.device)
            
            if hasattr(self.model, 'gnn'):
                logits, _ = self.model.gnn(val_data.x, val_data.edge_index, None)
            else:
                logits, _ = self.model(val_data.x, val_data.edge_index, None)
            
            pred = logits.argmax(dim=1).cpu().numpy()
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            
            val_metrics = compute_all_metrics(val_data.y.cpu().numpy(), pred, probs)
        
        # Test set (si disponible)
        test_metrics = None
        if test_data is not None:
            with torch.no_grad():
                test_data = test_data.to(self.device)
                
                if hasattr(self.model, 'gnn'):
                    logits, _ = self.model.gnn(test_data.x, test_data.edge_index, None)
                else:
                    logits, _ = self.model(test_data.x, test_data.edge_index, None)
                
                pred = logits.argmax(dim=1).cpu().numpy()
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                
                test_metrics = compute_all_metrics(test_data.y.cpu().numpy(), pred, probs)
        
        # Affichage
        print(f"\n📊 MÉTRIQUES VALIDATION:")
        print(f"   Accuracy:  {val_metrics['accuracy']:.4f}")
        print(f"   Precision: {val_metrics['precision']:.4f}")
        print(f"   Recall:    {val_metrics['recall']:.4f}")
        print(f"   F1-Score:  {val_metrics['f1_score']:.4f}")
        print(f"   ROC-AUC:   {val_metrics['roc_auc']:.4f}")
        
        if test_metrics:
            print(f"\n📊 MÉTRIQUES TEST:")
            print(f"   Accuracy:  {test_metrics['accuracy']:.4f}")
            print(f"   Precision: {test_metrics['precision']:.4f}")
            print(f"   Recall:    {test_metrics['recall']:.4f}")
            print(f"   F1-Score:  {test_metrics['f1_score']:.4f}")
            print(f"   ROC-AUC:   {test_metrics['roc_auc']:.4f}")
        
        # Décision automatique
        auto_decision = self.make_deployment_decision(val_metrics)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'auto_decision': auto_decision
        }
    
    def make_deployment_decision(self, metrics):
        """
        Décision automatique de déploiement
        """
        
        criteria_met = {
            'f1_score': metrics['f1_score'] >= self.min_f1,
            'precision': metrics['precision'] >= self.min_precision,
            'recall': metrics['recall'] >= self.min_recall
        }
        
        all_criteria_met = all(criteria_met.values())
        
        decision = {
            'recommend_deployment': all_criteria_met,
            'criteria_met': criteria_met,
            'reasons': []
        }
        
        if all_criteria_met:
            decision['reasons'].append("✅ Tous les critères sont satisfaits")
            decision['reasons'].append(f"✅ F1-Score: {metrics['f1_score']:.4f} >= {self.min_f1}")
            decision['reasons'].append(f"✅ Precision: {metrics['precision']:.4f} >= {self.min_precision}")
            decision['reasons'].append(f"✅ Recall: {metrics['recall']:.4f} >= {self.min_recall}")
        else:
            decision['reasons'].append("⚠️ Certains critères non satisfaits:")
            if not criteria_met['f1_score']:
                decision['reasons'].append(f"   ✗ F1-Score: {metrics['f1_score']:.4f} < {self.min_f1}")
            if not criteria_met['precision']:
                decision['reasons'].append(f"   ✗ Precision: {metrics['precision']:.4f} < {self.min_precision}")
            if not criteria_met['recall']:
                decision['reasons'].append(f"   ✗ Recall: {metrics['recall']:.4f} < {self.min_recall}")
        
        print(f"\n{'='*60}")
        if decision['recommend_deployment']:
            print("✅ RECOMMANDATION: DÉPLOYER NOUVELLE VERSION")
        else:
            print("⚠️  RECOMMANDATION: GARDER VERSION ACTUELLE")
        print(f"{'='*60}")
        
        for reason in decision['reasons']:
            print(f"   {reason}")
        
        return decision
    
    def generate_daily_report(self, validation_results, night_results, stream_stats):
        """
        Générer rapport quotidien pour expert humain
        """
        
        report = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'timestamp': datetime.now().isoformat(),
            'streaming_stats': stream_stats,
            'night_fine_tuning': night_results,
            'morning_validation': validation_results,
            'recommendation': validation_results.get('auto_decision', {})
        }
        
        # Sauvegarder
        report_path = f"/kaggle/working/daily_report_{datetime.now().strftime('%Y%m%d')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n📄 Rapport quotidien généré: {report_path}")
        
        return report
    
    def await_human_confirmation(self, timeout_hours=2):
        """
        Attendre confirmation humaine
        """
        
        print(f"\n{'='*60}")
        print("👤 ATTENTE CONFIRMATION HUMAINE")
        print(f"{'='*60}")
        print("   Expert humain doit review et confirmer déploiement")
        print(f"   Timeout: {timeout_hours}h")
        print()
        
        # Simulation pour démo
        print("   Simulation: Auto-confirmation après 5 secondes...")
        time.sleep(5)
        
        confirmed = True
        
        if confirmed:
            print("\n✅ CONFIRMATION REÇUE - Déploiement autorisé")
        else:
            print("\n⚠️  REJET - Garder version actuelle")
        
        return confirmed

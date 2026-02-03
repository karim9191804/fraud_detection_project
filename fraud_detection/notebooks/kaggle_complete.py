"""
╔══════════════════════════════════════════════════════════════╗
║  FRAUD DETECTION - PIPELINE COMPLET GNN + LLM                ║
║  Version finale optimisée GPU + DEMO PRODUCTION              ║
╚══════════════════════════════════════════════════════════════╝
"""

# ============================================================
# 🔧 SETUP ET CLONE GITHUB
# ============================================================

import os
import sys
import torch

print("="*60)
print("🚀 FRAUD DETECTION PIPELINE - VERSION FINALE")
print("="*60)

# Vérifier GPU FIRST
if not torch.cuda.is_available():
    print("⚠️  WARNING: GPU non disponible, utilisation CPU (très lent)")
    device = torch.device('cpu')
else:
    device = torch.device('cuda')
    print(f"✅ GPU détecté: {torch.cuda.get_device_name(0)}")
    print(f"   Mémoire: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Clone GitHub
if not os.path.exists('/kaggle/working/fraud_detection_project'):
    print("\n📥 Clonage depuis GitHub...")
    get_ipython().system('git clone https://github.com/karim9191804/fraud_detection_project.git /kaggle/working/fraud_detection_project')

os.chdir('/kaggle/working/fraud_detection_project/fraud_detection')

# Installation dépendances
print("\n📦 Installation dépendances...")
get_ipython().system('pip install -q -r requirements.txt')

sys.path.insert(0, '/kaggle/working/fraud_detection_project/fraud_detection')
print("✅ Setup terminé\n")

# ============================================================
# 📦 IMPORTS
# ============================================================

import pandas as pd
import numpy as np
import yaml
import json
import time
from datetime import datetime

# Imports des modules GitHub
from src.data.ieee_dataset import prepare_ieee_dataset
from src.models.gnn_model import LightGNNModel
from src.models.llm_wrapper import LightLLMWrapper
from src.models.hybrid_model import LightHybridModel
from src.utils.metrics import compute_all_metrics

import torch.nn as nn
import torch.optim as optim

print("✅ Tous les modules importés\n")

# ============================================================
# ⚙️ CONFIGURATION
# ============================================================

CONFIG = {
    'dataset_percent': 0.25,   # 0.25 = 25% (rapide), 1.0 = 100% (meilleur)
    'num_epochs': 8,
    'learning_rate': 1e-3,
    'batch_size': None,        # None = full graph en mémoire
    'use_hybrid': False,       # False = GNN seul, True = GNN+LLM
}

print(f"📋 Configuration:")
print(f"   Dataset: {int(CONFIG['dataset_percent']*100)}%")
print(f"   Epochs: {CONFIG['num_epochs']}")
print(f"   LR: {CONFIG['learning_rate']}")
print(f"   Mode: {'GNN+LLM Hybrid' if CONFIG['use_hybrid'] else 'GNN Seul'}")
print(f"   Device: {device}")
print()

# ============================================================
# 📊 ÉTAPE 1: CHARGEMENT DATASET
# ============================================================

print("="*60)
print("📊 ÉTAPE 1/5: CHARGEMENT DATASET")
print("="*60)

start_time = time.time()

train_trans = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv')
train_ident = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_identity.csv')

print(f"✅ Dataset original: {len(train_trans):,} transactions")

# Échantillonnage si demandé
if CONFIG['dataset_percent'] < 1.0:
    print(f"🔄 Échantillonnage {int(CONFIG['dataset_percent']*100)}%...")
    train_trans = train_trans.sample(frac=CONFIG['dataset_percent'], random_state=42)
    train_ident = train_ident[train_ident['TransactionID'].isin(train_trans['TransactionID'])]

print(f"✅ Dataset utilisé: {len(train_trans):,} transactions")
print(f"   Fraudes: {train_trans['isFraud'].sum():,} ({train_trans['isFraud'].mean()*100:.2f}%)")

# Sauvegarder pour pipeline
os.makedirs('/kaggle/working/temp_data', exist_ok=True)
train_trans.to_csv('/kaggle/working/temp_data/train_transaction.csv', index=False)
train_ident.to_csv('/kaggle/working/temp_data/train_identity.csv', index=False)

elapsed = time.time() - start_time
print(f"⏱️  Temps: {elapsed:.1f}s\n")

# ============================================================
# 🔗 ÉTAPE 2: CONSTRUCTION GRAPHE GNN
# ============================================================

print("="*60)
print("🔗 ÉTAPE 2/5: CONSTRUCTION GRAPHE GNN")
print("="*60)
print("⏱️  Cela peut prendre 10-45 minutes selon taille dataset...")
print("💡 Le graphe est construit sur CPU (normal)\n")

start_time = time.time()

dataset = prepare_ieee_dataset(
    data_dir='/kaggle/working/temp_data',
    output_dir='/kaggle/working/data/processed',
    test_size=0.15,
    val_size=0.15
)

elapsed = time.time() - start_time

print(f"\n✅ Graphe créé en {elapsed/60:.1f} minutes:")
print(f"   Train: {dataset['train'].num_nodes:,} nodes, {dataset['train'].num_edges:,} edges")
print(f"   Val:   {dataset['val'].num_nodes:,} nodes, {dataset['val'].num_edges:,} edges")
print(f"   Test:  {dataset['test'].num_nodes:,} nodes, {dataset['test'].num_edges:,} edges")
print(f"   Features: {dataset['train'].x.shape[1]}")

torch.save(dataset, '/kaggle/working/ieee_graph.pt')
print(f"💾 Graphe sauvegardé: /kaggle/working/ieee_graph.pt\n")

# ============================================================
# 🧠 ÉTAPE 3: CRÉATION MODÈLES
# ============================================================

print("="*60)
print("🧠 ÉTAPE 3/5: CRÉATION MODÈLES")
print("="*60)

with open('configs/config_light.yaml', 'r') as f:
    model_config = yaml.safe_load(f)

# Corriger dimensions features
model_config['gnn']['in_channels'] = dataset['train'].x.shape[1]

# GNN
gnn_model = LightGNNModel(model_config['gnn']).to(device)
print(f"✅ GNN créé: {sum(p.numel() for p in gnn_model.parameters()):,} params")

# Hybrid (optionnel)
if CONFIG['use_hybrid']:
    llm_model = LightLLMWrapper(model_config['llm']).to(device)
    print("✅ LLM créé (LoRA)")

    model = LightHybridModel(gnn_model, llm_model).to(device)
    print(f"✅ Hybrid Model: {sum(p.numel() for p in model.parameters()):,} params")
else:
    model = gnn_model
    print("✅ Mode GNN seul")

print(f"   Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")

# ============================================================
# 🏋️ ÉTAPE 4: TRAINING
# ============================================================

print("="*60)
print("🏋️ ÉTAPE 4/5: TRAINING")
print("="*60)
print(f"⏱️  Temps estimé: {CONFIG['num_epochs'] * 2}-{CONFIG['num_epochs'] * 5} minutes\n")

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE)
        return (self.alpha * (1 - pt) ** self.gamma * BCE).mean()

optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=0.01)
criterion = FocalLoss(alpha=0.25, gamma=2.0)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=2, verbose=True
)

# Envoyer sur device
print(f"📥 Chargement données sur {device}...")
train_data = dataset['train'].to(device)
val_data   = dataset['val'].to(device)
print("✅ Données prêtes\n")

best_f1 = 0
training_start = time.time()
history = {'train_loss': [], 'val_loss': [], 'val_f1': []}

for epoch in range(CONFIG['num_epochs']):
    epoch_start = time.time()

    print("="*60)
    print(f"Epoch {epoch+1}/{CONFIG['num_epochs']}")
    print("="*60)

    # TRAIN
    model.train()
    if CONFIG['use_hybrid']:
        logits = model((train_data.x, train_data.edge_index, None), None)
    else:
        logits, _ = model(train_data.x, train_data.edge_index, None)

    loss = criterion(logits, train_data.y)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    train_loss = loss.item()

    # VAL
    model.eval()
    with torch.no_grad():
        if CONFIG['use_hybrid']:
            val_logits = model((val_data.x, val_data.edge_index, None), None)
        else:
            val_logits, _ = model(val_data.x, val_data.edge_index, None)

        val_loss = criterion(val_logits, val_data.y).item()

        val_pred  = val_logits.argmax(dim=1).cpu().numpy()
        val_true  = val_data.y.cpu().numpy()
        val_probs = torch.softmax(val_logits, dim=1)[:, 1].cpu().numpy()

        metrics = compute_all_metrics(val_true, val_pred, val_probs)

    scheduler.step(metrics['f1_score'])

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['val_f1'].append(metrics['f1_score'])

    epoch_time = time.time() - epoch_start
    print(f"\n📊 Résultats (temps: {epoch_time:.1f}s):")
    print(f"   Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    print(f"   Accuracy:   {metrics['accuracy']:.4f}")
    print(f"   F1-Score:   {metrics['f1_score']:.4f}")
    print(f"   Precision:  {metrics['precision']:.4f}")
    print(f"   Recall:     {metrics['recall']:.4f}")
    print(f"   ROC-AUC:    {metrics['roc_auc']:.4f}")

    if metrics['f1_score'] > best_f1:
        best_f1 = metrics['f1_score']
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch + 1,
            'metrics': metrics,
            'config': CONFIG
        }, '/kaggle/working/best_model.pt')
        print(f"\n   🏆 Meilleur modèle sauvegardé (F1: {best_f1:.4f})")

    print()

training_time = time.time() - training_start

print("="*60)
print(f"✅ Training terminé en {training_time/60:.1f} minutes")
print(f"🏆 Meilleur F1 validation: {best_f1:.4f}")
print("="*60 + "\n")

# ============================================================
# 🌅 ÉTAPE 5: TEST FINAL
# ============================================================

print("="*60)
print("🌅 ÉTAPE 5/5: TEST FINAL")
print("="*60)

checkpoint = torch.load('/kaggle/working/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"✅ Meilleur modèle chargé (Epoch {checkpoint['epoch']})\n")

test_data = dataset['test'].to(device)

with torch.no_grad():
    if CONFIG['use_hybrid']:
        test_logits = model((test_data.x, test_data.edge_index, None), None)
    else:
        test_logits, _ = model(test_data.x, test_data.edge_index, None)

    test_pred  = test_logits.argmax(dim=1).cpu().numpy()
    test_true  = test_data.y.cpu().numpy()
    test_probs = torch.softmax(test_logits, dim=1)[:, 1].cpu().numpy()

    test_metrics = compute_all_metrics(test_true, test_pred, test_probs)

print("📊 RÉSULTATS FINAUX:")
print("="*60)
print(f"  Accuracy:   {test_metrics['accuracy']:.4f}")
print(f"  Precision:  {test_metrics['precision']:.4f}")
print(f"  Recall:     {test_metrics['recall']:.4f}")
print(f"  F1-Score:   {test_metrics['f1_score']:.4f}")
print(f"  ROC-AUC:    {test_metrics['roc_auc']:.4f}")
print("="*60)

if 'true_positives' in test_metrics:
    print("\n📈 Confusion Matrix:")
    print(f"   True Positives:  {test_metrics['true_positives']:,}")
    print(f"   False Positives: {test_metrics['false_positives']:,}")
    print(f"   True Negatives:  {test_metrics['true_negatives']:,}")
    print(f"   False Negatives: {test_metrics['false_negatives']:,}")
    print(f"   FPR: {test_metrics['fpr']:.4f}")
    print(f"   FNR: {test_metrics['fnr']:.4f}")

deployable = (
    test_metrics['f1_score'] >= 0.70 and
    test_metrics['precision'] >= 0.65 and
    test_metrics['recall'] >= 0.65
)

print("\n" + "="*60)
if deployable:
    print("✅ MODÈLE VALIDÉ POUR DÉPLOIEMENT PRODUCTION!")
    print(f"   ✓ F1-Score ≥ 0.70: {test_metrics['f1_score']:.4f}")
    print(f"   ✓ Precision ≥ 0.65: {test_metrics['precision']:.4f}")
    print(f"   ✓ Recall ≥ 0.65: {test_metrics['recall']:.4f}")
else:
    print("⚠️  MODÈLE À AMÉLIORER")
    print("   Suggestions:")
    if test_metrics['f1_score'] < 0.70:
        print("   • Augmenter nombre d'epochs")
    if CONFIG['dataset_percent'] < 1.0:
        print("   • Utiliser dataset complet (100%)")
    if not CONFIG['use_hybrid']:
        print("   • Activer mode hybrid (GNN+LLM)")
    print("\n   Résultats actuels:")
    print(f"   {'✓' if test_metrics['f1_score'] >= 0.70 else '✗'} F1-Score ≥ 0.70: {test_metrics['f1_score']:.4f}")
    print(f"   {'✓' if test_metrics['precision'] >= 0.65 else '✗'} Precision ≥ 0.65: {test_metrics['precision']:.4f}")
    print(f"   {'✓' if test_metrics['recall'] >= 0.65 else '✗'} Recall ≥ 0.65: {test_metrics['recall']:.4f}")
print("="*60)

# ============================================================
# 💾 SAUVEGARDE RÉSULTATS
# ============================================================

print("\n💾 Sauvegarde résultats...")

results = {
    'timestamp': datetime.now().isoformat(),
    'config': CONFIG,
    'dataset': {
        'total_transactions': len(train_trans),
        'train_nodes': dataset['train'].num_nodes,
        'val_nodes': dataset['val'].num_nodes,
        'test_nodes': dataset['test'].num_nodes,
        'features': dataset['train'].x.shape[1]
    },
    'training': {
        'epochs': CONFIG['num_epochs'],
        'best_epoch': checkpoint['epoch'],
        'training_time_minutes': training_time / 60,
        'best_val_f1': float(best_f1)
    },
    'test_metrics': {k: float(v) if isinstance(v, (int, float, np.number)) else v
                     for k, v in test_metrics.items()},
    'deployable': deployable,
    'device': str(device),
    'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
}

with open('/kaggle/working/results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("✅ Résultats sauvegardés: /kaggle/working/results.json")

# ============================================================
# 🎉 RÉSUMÉ FINAL
# ============================================================

print("\n" + "="*60)
print("🎉 PIPELINE TERMINÉ AVEC SUCCÈS!")
print("="*60)
print("\n📊 Résumé:")
print(f"   Dataset: {len(train_trans):,} transactions ({int(CONFIG['dataset_percent']*100)}%)")
print(f"   Training: {training_time/60:.1f} minutes")
print(f"   Meilleur F1 Val: {best_f1:.4f}")
print(f"   F1-Score Test: {test_metrics['f1_score']:.4f}")
print(f"   Déploiement: {'✅ OUI' if deployable else '⚠️ NON'}")
print("\n📁 Fichiers générés:")
print("   • /kaggle/working/best_model.pt")
print("   • /kaggle/working/results.json")
print("   • /kaggle/working/ieee_graph.pt")
print("\n✨ Merci d'avoir utilisé le système de détection de fraude!")
print("="*60 + "\n")


# ============================================================
# ✅ CELLULE 13 BONUS: Démonstration Système Production Complet
# ============================================================

print("\n" + "="*80)
print("🎯 DÉMONSTRATION SYSTÈME COMPLET - PRODUCTION WORKFLOW")
print("="*80)

# Imports
import importlib.util

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Charger modules production
base_path = "/kaggle/working/fraud_detection_project/fraud_detection/src/production"
stream_module   = load_module("streaming_detector", f"{base_path}/streaming_detector.py")
night_module    = load_module("night_fine_tuner", f"{base_path}/night_fine_tuner.py")
morning_module  = load_module("morning_validator", f"{base_path}/morning_validator.py")

StreamingFraudDetector      = stream_module.StreamingFraudDetector
TransactionStreamSimulator  = stream_module.TransactionStreamSimulator
NightFineTuner              = night_module.NightFineTuner
MorningValidator            = morning_module.MorningValidator

# Configuration
production_config = {
    'max_buffer_size': 10000,
    'confidence_threshold': 0.75,
    'night_lr': 1e-5,
    'deploy_min_f1': 0.75,
    'deploy_min_precision': 0.70,
    'deploy_min_recall': 0.70
}

# ☀️ MODE JOUR
print(f"\n{'='*80}")
print("☀️ MODE JOUR - STREAMING")
print(f"{'='*80}")

streaming_detector = StreamingFraudDetector(
    model=model, llm_wrapper=None, config=production_config, device=device
)

# Simuler 100 transactions
test_dataset = []
for i in range(min(100, dataset['test'].num_nodes)):
    test_dataset.append({'features': dataset['test'].x[i].cpu().numpy().tolist()})

stream_simulator = TransactionStreamSimulator(test_dataset, delay_ms=10)

results_stream = []
for i, result in enumerate(streaming_detector.process_stream(stream_simulator)):
    results_stream.append(result)
    if i < 3 or result.get('is_fraud', False):
        print(f"TX #{i+1}: {'🚨 FRAUDE' if result['is_fraud'] else '✅ OK'} "
              f"(P={result['fraud_probability']:.2f}, {result['latency_ms']:.1f}ms)")

day_stats = streaming_detector.get_stats()
critical_cases = streaming_detector.get_critical_cases(clear=False)

print(f"\n📊 STATS: {day_stats['total_transactions']} TX, "
      f"{day_stats['frauds_detected']} fraudes, {day_stats['critical_cases']} critiques")

# 🌙 MODE NUIT
print(f"\n{'='*80}")
print("🌙 MODE NUIT - FINE-TUNING + RLHF")
print(f"{'='*80}")

# Ajouter feedback humain (simulation)
for case in critical_cases[:20]:
    case['human_feedback'] = {
        'expert_id': 'expert_001',
        'corrected_label': 1 if case['fraud_probability'] > 0.5 else 0,
        'confidence': 'high',
        'timestamp': datetime.now().isoformat()
    }

cases_with_feedback = [c for c in critical_cases if c.get('human_feedback')]
print(f"Feedbacks: {len(cases_with_feedback)}")

night_tuner = NightFineTuner(model, production_config, device)

fine_tuning_results = night_tuner.fine_tune_on_critical_cases(
    cases_with_feedback,
    num_epochs=3,
    validation_data=dataset['val']
)

rlhf_results = night_tuner.rlhf_update(cases_with_feedback)
night_tuner.save_checkpoint('/kaggle/working/model_improved_night.pt')

# 🌅 MODE MATIN
print(f"\n{'='*80}")
print("🌅 MODE MATIN - VALIDATION")
print(f"{'='*80}")

morning_validator = MorningValidator(model, production_config, device)

validation_results = morning_validator.validate_improved_model(
    val_data=dataset['val'],
    test_data=dataset['test']
)

daily_report = morning_validator.generate_daily_report(
    validation_results,
    {'fine_tuning': fine_tuning_results, 'rlhf': rlhf_results},
    day_stats
)

if validation_results['auto_decision']['recommend_deployment']:
    print("\n✅ RECOMMANDATION: Déployer")
    confirmed = morning_validator.await_human_confirmation(timeout_hours=2)
    print(f"\n{'🚀 DÉPLOIEMENT CONFIRMÉ' if confirmed else '🔄 ROLLBACK'}")
else:
    print("\n⚠️  RECOMMANDATION: Garder version actuelle")

# RÉSUMÉ
print(f"\n{'='*80}")
print("📊 RÉSUMÉ CYCLE COMPLET")
print(f"{'='*80}")
print(f"☀️ JOUR: {day_stats['total_transactions']} TX, {day_stats['critical_cases']} critiques")
print(f"🌙 NUIT: {len(cases_with_feedback)} feedbacks, RLHF reward={rlhf_results.get('avg_reward', 0):.2f}")
print(f"🌅 MATIN: F1={validation_results['val_metrics']['f1_score']:.4f}")
print("\n✅ Système complet opérationnel - Prêt pour PFE ! 🎓")

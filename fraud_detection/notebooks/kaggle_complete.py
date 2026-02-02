"""
Kaggle Notebook: Fraud Detection avec GNN + LLM Hybrid
Dataset: IEEE-CIS Fraud Detection
Architecture: GNN (GAT) + DistilGPT2 avec LoRA
"""

# ============================================
# CELLULE 1: INSTALLATION ET SETUP
# ============================================

print("="*80)
print("🚀 FRAUD DETECTION SYSTEM - GNN + LLM HYBRID")
print("="*80)

# Cloner le repository
get_ipython().system('git clone https://github.com/karim9191804/fraud_detection_project.git')
get_ipython().magic('cd fraud_detection_project')

# Installer les dépendances
get_ipython().system('pip install -r requirements.txt -q')

print("\n✅ Installation terminée!")
print("📦 Packages installés:")
print("   - torch, torch-geometric")
print("   - transformers, peft")
print("   - scikit-learn, matplotlib, seaborn")

# ============================================
# CELLULE 2: CHARGEMENT DES DONNÉES
# ============================================

import sys
sys.path.insert(0, '/kaggle/working/fraud_detection_project')

from src.data.ieee_dataset import load_ieee_dataset

print("\n" + "="*80)
print("📊 CHARGEMENT DU DATASET IEEE-CIS")
print("="*80)

# Charger le dataset
dataset = load_ieee_dataset(
    data_dir='/kaggle/input/ieee-fraud-detection',
    use_sample=False,  # False pour dataset complet, True pour test rapide
    test_size=0.2,
    val_size=0.1,
    random_state=42
)

print(f"\n✅ Dataset chargé:")
print(f"   Train: {len(dataset['train'])} samples")
print(f"   Val:   {len(dataset['val'])} samples")
print(f"   Test:  {len(dataset['test'])} samples")

# Distribution des classes
train_labels = [data.y.item() for data in dataset['train']]
fraud_count = sum(train_labels)
legit_count = len(train_labels) - fraud_count

print(f"\n📈 Distribution des classes:")
print(f"   Fraudes: {fraud_count:,} ({100*fraud_count/len(train_labels):.2f}%)")
print(f"   Légitimes: {legit_count:,} ({100*legit_count/len(train_labels):.2f}%)")
print(f"   Ratio: 1:{legit_count/fraud_count:.1f}")

# ============================================
# CELLULE 3: CRÉATION DU MODÈLE HYBRIDE
# ============================================

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from torch_geometric.data import Data, Batch

print("\n" + "="*80)
print("🧠 CRÉATION DU MODÈLE HYBRIDE GNN + DistilGPT2")
print("="*80)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️  Device: {device}")

# 1. GNN (Graph Neural Network)
print("\n1️⃣ Création du GNN...")
from src.models.gnn_model import GNNModel

gnn_config = {
    'input_dim': 64,
    'hidden_channels': 128,
    'num_layers': 2,
    'dropout': 0.3,
    'heads': 4
}

gnn = GNNModel(gnn_config).to(device)
print(f"   ✅ GNN créé: {sum(p.numel() for p in gnn.parameters()):,} paramètres")

# 2. LLM (DistilGPT2 avec LoRA)
print("\n2️⃣ Chargement de DistilGPT2...")
tokenizer = AutoTokenizer.from_pretrained('distilgpt2', padding_side='left')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

llm = AutoModelForCausalLM.from_pretrained(
    'distilgpt2',
    torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
    device_map='auto' if device == 'cuda' else None
)

# Application de LoRA
print("   🔧 Application de LoRA...")
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["c_attn", "c_proj"],
    bias="none"
)

llm = get_peft_model(llm, lora_config)
llm.print_trainable_parameters()

# 3. Couche de fusion
print("\n3️⃣ Création de la couche de fusion...")
fusion_layer = nn.Sequential(
    nn.Linear(128, llm.config.n_embd),
    nn.LayerNorm(llm.config.n_embd),
    nn.Dropout(0.1)
).to(device)

# 4. Tête de classification
print("\n4️⃣ Création de la tête de classification...")
fraud_head = nn.Sequential(
    nn.Linear(llm.config.n_embd, llm.config.n_embd // 2),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(llm.config.n_embd // 2, 1)
).to(device)

# 5. Modèle hybride complet
print("\n5️⃣ Assemblage du modèle hybride...")

class HybridModel(nn.Module):
    def __init__(self, gnn, llm, fusion, classifier):
        super().__init__()
        self.gnn = gnn
        self.llm = llm
        self.fusion = fusion
        self.classifier = classifier
    
    def forward(self, graph_data, mode='train'):
        device = next(self.gnn.parameters()).device
        
        x = graph_data.x.to(device)
        edge_index = graph_data.edge_index.to(device)
        batch = graph_data.batch.to(device) if hasattr(graph_data, 'batch') else None
        
        gnn_embeddings = self.gnn.get_embeddings(x, edge_index, batch)
        fused_embeddings = self.fusion(gnn_embeddings)
        logits = self.classifier(fused_embeddings)
        
        return logits

model = HybridModel(gnn, llm, fusion_layer, fraud_head).to(device)

# Statistiques
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\n📊 Statistiques du modèle:")
print(f"   Paramètres totaux: {total_params:,}")
print(f"   Paramètres entraînables: {trainable_params:,}")
print(f"   Ratio: {100 * trainable_params / total_params:.2f}%")

print("\n✅ Modèle hybride prêt!")

# ============================================
# CELLULE 4: CONFIGURATION D'ENTRAÎNEMENT
# ============================================

from torch_geometric.loader import DataLoader
import os

print("\n" + "="*80)
print("⚙️  CONFIGURATION D'ENTRAÎNEMENT")
print("="*80)

# DataLoaders
BATCH_SIZE = 32

train_loader = DataLoader(
    dataset['train'], 
    batch_size=BATCH_SIZE, 
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

val_loader = DataLoader(
    dataset['val'], 
    batch_size=BATCH_SIZE, 
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

test_loader = DataLoader(
    dataset['test'], 
    batch_size=BATCH_SIZE, 
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

print(f"✅ DataLoaders créés (batch_size={BATCH_SIZE})")

# Configuration du trainer
training_config = {
    # Learning rates différentiels
    'gnn_lr': 2e-4,
    'llm_lr': 2e-5,
    'classifier_lr': 2e-4,
    
    # Optimizer
    'adam_betas': (0.9, 0.999),
    'adam_eps': 1e-8,
    'weight_decay': 1e-5,
    
    # Loss function
    'use_focal_loss': True,
    'focal_alpha': 0.25,
    'focal_gamma': 2.0,
    'pos_weight': 10.0,
    
    # Scheduler
    'use_scheduler': True,
    'warmup_epochs': 2,
    'lr_patience': 3,
    'lr_factor': 0.5,
    'min_lr': 1e-6,
    
    # Training
    'grad_clip': 1.0,
    'patience': 5,
    'min_delta': 1e-4,
    
    # Cas critiques
    'save_critical_cases': True,
    'critical_confidence_threshold': 0.6,
    'max_critical_cases': 1000,
    
    # Critères de déploiement
    'deploy_min_f1': 0.75,
    'deploy_min_precision': 0.70,
    'deploy_min_recall': 0.70,
    'deploy_max_fpr': 0.10,
    
    # Chemins
    'checkpoint_dir': '/kaggle/working/checkpoints',
    'logs_dir': '/kaggle/working/logs',
    'device': device
}

# Créer les dossiers
os.makedirs(training_config['checkpoint_dir'], exist_ok=True)
os.makedirs(training_config['logs_dir'], exist_ok=True)

print("\n📋 Configuration:")
print(f"   GNN LR: {training_config['gnn_lr']}")
print(f"   LLM LR: {training_config['llm_lr']}")
print(f"   Focal Loss: {training_config['use_focal_loss']}")
print(f"   Warm-up: {training_config['warmup_epochs']} epochs")
print(f"   Patience: {training_config['patience']}")

# ============================================
# CELLULE 5: ENTRAÎNEMENT
# ============================================

from src.training.trainer import FraudDetectionTrainer

print("\n" + "="*80)
print("🏋️ ENTRAÎNEMENT DU MODÈLE")
print("="*80)

# Créer le trainer
trainer = FraudDetectionTrainer(model, training_config)

# Entraîner
NUM_EPOCHS = 15

trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=NUM_EPOCHS
)

print("\n✅ Entraînement terminé!")

# ============================================
# CELLULE 6: VISUALISATION DES RÉSULTATS
# ============================================

print("\n" + "="*80)
print("📊 VISUALISATION DES RÉSULTATS")
print("="*80)

# Créer les graphiques
trainer.plot_results(
    save_path=f"{training_config['logs_dir']}/training_results.png"
)

print(f"\n🏆 Meilleure performance:")
print(f"   Epoch: {trainer.best_epoch}")
print(f"   Val Loss: {trainer.best_val_loss:.4f}")
print(f"   Val F1: {trainer.best_val_f1:.4f}")

# Statistiques des cas critiques
if trainer.critical_cases:
    avg_conf = sum(c['confidence'] for c in trainer.critical_cases) / len(trainer.critical_cases)
    avg_fraud = sum(c['fraud_prob'] for c in trainer.critical_cases) / len(trainer.critical_cases)
    
    print(f"\n🔍 Cas critiques détectés:")
    print(f"   Total: {len(trainer.critical_cases)}")
    print(f"   Confiance moyenne: {avg_conf:.4f}")
    print(f"   Prob. fraude moyenne: {avg_fraud:.4f}")

# ============================================
# CELLULE 7: ÉVALUATION FINALE (MODE MORNING)
# ============================================

from src.training.trainer import compute_metrics, check_deployment_criteria
import numpy as np

print("\n" + "="*80)
print("🌅 VALIDATION FINALE (MODE MORNING)")
print("="*80)

# Charger le meilleur modèle
best_checkpoint = torch.load(f"{training_config['checkpoint_dir']}/best_model.pt")
model.load_state_dict(best_checkpoint['model_state_dict'])

print(f"📂 Meilleur modèle chargé (Epoch {best_checkpoint['epoch']})")

# Évaluation sur le test set
model.eval()
test_preds = []
test_labels = []

with torch.no_grad():
    for data in test_loader:
        data = data.to(device)
        logits = model(data).squeeze()
        if logits.dim() == 0:
            logits = logits.unsqueeze(0)
        
        preds = torch.sigmoid(logits).cpu().numpy()
        test_preds.extend(preds)
        test_labels.extend(data.y.cpu().numpy())

# Calculer les métriques
test_metrics = compute_metrics(
    predictions=np.array(test_preds),
    labels=np.array(test_labels)
)

print(f"\n📊 Métriques Test Set:")
print(f"   Accuracy:  {test_metrics['accuracy']:.4f}")
print(f"   Precision: {test_metrics['precision']:.4f}")
print(f"   Recall:    {test_metrics['recall']:.4f}")
print(f"   F1:        {test_metrics['f1']:.4f}")
print(f"   AUC:       {test_metrics['auc']:.4f}")
if 'fpr' in test_metrics:
    print(f"   FPR:       {test_metrics['fpr']:.4f}")
    print(f"   FNR:       {test_metrics['fnr']:.4f}")

# Vérifier les critères de déploiement
can_deploy, criteria = check_deployment_criteria(test_metrics, training_config)

print(f"\n{'='*80}")
if can_deploy:
    print("✅ MODÈLE VALIDÉ POUR LE DÉPLOIEMENT!")
    print("="*80)
    print("\nTous les critères sont remplis:")
    for criterion, passed in criteria.items():
        print(f"   ✓ {criterion}")
else:
    print("⚠️ MODÈLE NON VALIDÉ POUR LE DÉPLOIEMENT")
    print("="*80)
    print("\nCritères non remplis:")
    for criterion, passed in criteria.items():
        status = "✓" if passed else "✗"
        print(f"   {status} {criterion}")

# Sauvegarder le rapport final
import json
from datetime import datetime

validation_report = {
    'timestamp': datetime.now().isoformat(),
    'best_epoch': trainer.best_epoch,
    'val_metrics': best_checkpoint['val_metrics'],
    'test_metrics': test_metrics,
    'can_deploy': can_deploy,
    'deploy_criteria': criteria,
    'critical_cases_count': len(trainer.critical_cases)
}

report_path = f"{training_config['logs_dir']}/validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(report_path, 'w') as f:
    json.dump(validation_report, f, indent=2, default=str)

print(f"\n📄 Rapport de validation sauvegardé: {report_path}")

# ============================================
# RÉSUMÉ FINAL
# ============================================

print("\n" + "="*80)
print("🎉 ENTRAÎNEMENT ET VALIDATION TERMINÉS!")
print("="*80)

print(f"\n📁 Fichiers générés:")
print(f"   Checkpoints: {training_config['checkpoint_dir']}")
print(f"   Logs: {training_config['logs_dir']}")
print(f"   Visualisation: {training_config['logs_dir']}/training_results.png")

print(f"\n🏆 Résultats finaux:")
print(f"   Meilleur epoch: {trainer.best_epoch}")
print(f"   Val Loss: {trainer.best_val_loss:.4f}")
print(f"   Val F1: {trainer.best_val_f1:.4f}")
print(f"   Test F1: {test_metrics['f1']:.4f}")
print(f"   Déploiement: {'✅ OUI' if can_deploy else '⚠️ NON'}")

print("\n✨ Merci d'avoir utilisé le système de détection de fraude GNN+LLM!")

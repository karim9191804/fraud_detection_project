# 🚀 GUIDE COMPLET STEP-BY-STEP - FRAUD DETECTION PROJECT

Ce guide vous accompagne **étape par étape** pour tout mettre en place.

---

## 📋 TABLE DES MATIÈRES

1. [Configuration Initiale](#1-configuration-initiale)
2. [Setup GitHub](#2-setup-github)
3. [Setup Kaggle](#3-setup-kaggle)
4. [GitHub → Kaggle Workflow](#4-github--kaggle-workflow)
5. [Exécution sur Kaggle GPU P100](#5-exécution-sur-kaggle-gpu-p100)
6. [Kaggle → GitHub Sync](#6-kaggle--github-sync)
7. [Déploiement Hugging Face](#7-déploiement-hugging-face)
8. [Monitoring et Métriques](#8-monitoring-et-métriques)
9. [RLHF et Fine-tuning Continu](#9-rlhf-et-fine-tuning-continu)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. CONFIGURATION INITIALE

### 1.1 Prérequis

✅ Vérifiez que vous avez :
- [ ] Python 3.10+
- [ ] Git installé
- [ ] Compte GitHub
- [ ] Compte Kaggle
- [ ] (Optionnel) Compte Hugging Face
- [ ] (Optionnel) Compte WandB

### 1.2 Obtenir les Tokens

#### A. GitHub Personal Access Token

```bash
1. Allez sur: https://github.com/settings/tokens
2. Cliquez: "Generate new token" → "Generate new token (classic)"
3. Nom: fraud-detection-token
4. Permissions à cocher:
   ✅ repo (tous)
   ✅ workflow
   ✅ admin:repo_hook
5. Cliquez "Generate token"
6. COPIEZ ET SAUVEGARDEZ LE TOKEN (ghp_...)
```

#### B. Kaggle API Credentials

```bash
1. Allez sur: https://www.kaggle.com/settings/account
2. Section "API" → Cliquez "Create New API Token"
3. Un fichier kaggle.json sera téléchargé
4. Contenu du fichier:
   {
     "username": "votre_username",
     "key": "votre_key_ici"
   }
5. GARDEZ CE FICHIER
```

#### C. Hugging Face Token (Optionnel)

```bash
1. Allez sur: https://huggingface.co/settings/tokens
2. Cliquez "New token"
3. Name: fraud-detection
4. Type: Write
5. Cliquez "Generate"
6. COPIEZ LE TOKEN (hf_...)
```

#### D. WandB API Key (Optionnel)

```bash
1. Allez sur: https://wandb.ai/authorize
2. Copiez votre API Key
```

---

## 2. SETUP GITHUB

### 2.1 Créer le Repository

```bash
# Sur GitHub.com
1. Allez sur: https://github.com/new
2. Repository name: fraud-detection-project
3. Description: "Fraud Detection with GNN+LLM+RLHF - PFE"
4. Public ou Private (votre choix)
5. ❌ NE PAS cocher "Add README"
6. ❌ NE PAS cocher "Add .gitignore"
7. ❌ NE PAS choisir de license
8. Cliquez "Create repository"
```

### 2.2 Initialiser Git Localement

```bash
# Extraire votre projet
unzip fraud_detection.zip
cd fraud_detection

# Initialiser Git
git init

# Configurer votre identité
git config user.name "Votre Nom"
git config user.email "votre.email@example.com"

# Ajouter tous les fichiers
git add .

# Premier commit
git commit -m "Initial commit: Fraud Detection System with GNN+LLM+RLHF"

# Renommer la branche en main (si nécessaire)
git branch -M main

# Ajouter le remote (REMPLACEZ votre-username)
git remote add origin https://github.com/votre-username/fraud-detection-project.git

# Pousser vers GitHub
git push -u origin main
```

**✅ Vérification:** Allez sur votre repo GitHub, vous devriez voir tous vos fichiers.

### 2.3 Configurer les GitHub Secrets

```bash
1. Sur votre repo GitHub, allez dans: Settings → Secrets and variables → Actions
2. Cliquez "New repository secret"
3. Ajoutez ces secrets:

Secret 1:
  Name: KAGGLE_USERNAME
  Value: votre_username_kaggle

Secret 2:
  Name: KAGGLE_KEY
  Value: votre_kaggle_api_key (depuis kaggle.json)

Secret 3 (optionnel):
  Name: HF_TOKEN
  Value: hf_votre_token

Secret 4 (optionnel):
  Name: WANDB_API_KEY
  Value: votre_wandb_key
```

**✅ Vérification:** Settings → Secrets → Actions devrait afficher vos secrets.

---

## 3. SETUP KAGGLE

### 3.1 Accepter la Compétition IEEE-CIS

```bash
1. Allez sur: https://www.kaggle.com/c/ieee-fraud-detection/rules
2. Lisez les règles
3. Cliquez "I Understand and Accept"
```

**✅ Vérification:** Vous devriez voir "You're in!" en haut de la page.

### 3.2 Installer Kaggle API Localement

```bash
# Installer
pip install kaggle

# Configurer
mkdir -p ~/.kaggle
cp ~/Downloads/kaggle.json ~/.kaggle/kaggle.json  # Ajustez le chemin
chmod 600 ~/.kaggle/kaggle.json

# Tester
kaggle competitions list

# Vous devriez voir une liste de compétitions
```

### 3.3 Télécharger le Dataset Localement (Optionnel)

```bash
# Créer le dossier
mkdir -p data/raw/ieee-fraud

# Télécharger
kaggle competitions download -c ieee-fraud-detection

# Dézipper
unzip ieee-fraud-detection.zip -d data/raw/ieee-fraud/

# Vérifier
ls data/raw/ieee-fraud/
# Devrait afficher: train_transaction.csv, train_identity.csv, etc.
```

---

## 4. GITHUB → KAGGLE WORKFLOW

### 4.1 Push automatique vers Kaggle

Le workflow GitHub Actions (`.github/workflows/sync-kaggle.yml`) fait ceci automatiquement :

**Déclencheurs:**
- ✅ À chaque push sur main
- ✅ Toutes les 6 heures (configurable)
- ✅ Manuellement via GitHub Actions

**Ce qu'il fait:**
1. Checkout du code
2. Upload du code vers Kaggle Dataset
3. Upload/Update du notebook Kaggle
4. Commit du status de sync

### 4.2 Déclencher Manuellement

```bash
# Sur GitHub:
1. Allez dans: Actions
2. Cliquez sur "Sync with Kaggle"
3. Cliquez "Run workflow"
4. Sélectionnez "main"
5. Cliquez "Run workflow"
```

**✅ Vérification:** Allez sur Kaggle → Your Datasets → Vous devriez voir "fraud-detection-code"

### 4.3 Vérifier le Notebook Kaggle

```bash
1. Allez sur: https://www.kaggle.com/code
2. Vous devriez voir: "fraud-detection-training"
3. Cliquez dessus pour l'ouvrir
```

---

## 5. EXÉCUTION SUR KAGGLE GPU P100

### 5.1 Ouvrir le Notebook sur Kaggle

```bash
1. Allez sur votre notebook: 
   https://www.kaggle.com/code/votre-username/fraud-detection-training

2. En haut à droite, vérifiez:
   - Accelerator: GPU P100 ✅
   - Internet: ON ✅
   
3. Si pas activé:
   - Cliquez sur "⚙️" (Settings)
   - Accelerator → GPU P100
   - Internet → ON
   - Save
```

### 5.2 Configurer les Secrets Kaggle

```bash
1. Dans le notebook, cliquez "Add-ons" → "Secrets"
2. Ajoutez ces secrets:

Secret 1:
  Label: GITHUB_TOKEN
  Value: ghp_votre_token_github

Secret 2:
  Label: WANDB_API_KEY  (optionnel)
  Value: votre_wandb_key
```

### 5.3 Modifier le Notebook

Dans la cellule de configuration, remplacez :

```python
# REMPLACEZ par vos vraies valeurs
GITHUB_REPO = "votre-username/fraud-detection-project"
GITHUB_TOKEN = "ghp_..."  # Ou utilisez Kaggle Secrets (recommandé)
```

**Avec Secrets (recommandé):**

```python
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()

GITHUB_REPO = "votre-username/fraud-detection-project"
GITHUB_TOKEN = user_secrets.get_secret("GITHUB_TOKEN")
```

### 5.4 Lancer l'Entraînement !

```bash
1. Cliquez "Run All" ou Shift+Enter sur chaque cellule
2. L'entraînement va:
   - Cloner votre code GitHub
   - Télécharger IEEE-CIS depuis Kaggle
   - Préparer les données
   - Créer le modèle sur GPU P100
   - Entraîner (Mode Jour → Nuit → Matin)
   - Sauvegarder les résultats
   - Push vers GitHub

⏰ Durée estimée: 1-2 heures sur GPU P100
```

### 5.5 Surveiller l'Exécution

**GPU Utilization:**
```python
# Cellule de monitoring (ajoutez-la si besoin)
!nvidia-smi

import torch
print(f"GPU Memory Used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"GPU Memory Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```

**Logs en Temps Réel:**
- WandB (si configuré): Dashboard temps réel
- Kaggle Output: Scroll pour voir les logs

---

## 6. KAGGLE → GITHUB SYNC

### 6.1 Automatique via le Notebook

Le notebook push automatiquement vers GitHub à la fin :

```python
# Cette cellule dans le notebook fait le push
try:
    repo = Repo('.')
    repo.git.add('checkpoints/')
    repo.git.add('logs/')
    repo.git.add('memory/')
    repo.git.add('final_model/')
    
    commit_msg = f"Kaggle P100 training - {datetime.now()}"
    repo.index.commit(commit_msg)
    
    origin = repo.remote('origin')
    origin.push()
    
    print("✅ Résultats poussés vers GitHub")
except Exception as e:
    print(f"⚠️ Erreur: {e}")
```

### 6.2 Téléchargement Manuel (si push échoue)

```bash
# Dans Kaggle Notebook:
1. Cliquez "Save Version"
2. Type: "Save & Run All"
3. Une fois terminé, cliquez sur la version
4. Cliquez "Output" (à droite)
5. Téléchargez les fichiers:
   - checkpoints/
   - logs/
   - memory/
   - final_model/

# Localement:
# Copiez les fichiers dans votre repo local
cp -r ~/Downloads/checkpoints ./
cp -r ~/Downloads/logs ./
cp -r ~/Downloads/memory ./
cp -r ~/Downloads/final_model ./

# Commit et push
git add checkpoints/ logs/ memory/ final_model/
git commit -m "Add training results from Kaggle P100"
git push origin main
```

### 6.3 Vérifier sur GitHub

```bash
1. Allez sur: https://github.com/votre-username/fraud-detection-project
2. Vous devriez voir les nouveaux dossiers:
   - checkpoints/
   - logs/
   - memory/
   - final_model/
3. Vérifiez le dernier commit
```

---

## 7. DÉPLOIEMENT HUGGING FACE

### 7.1 Créer un Space

```bash
1. Allez sur: https://huggingface.co/new-space
2. Owner: votre-username
3. Space name: fraud-detection-app
4. License: MIT
5. Select the Space SDK: Gradio
6. Space hardware: CPU basic (gratuit) ou GPU (payant)
7. Cliquez "Create Space"
```

### 7.2 Déployer avec le Script

```bash
# Localement
cd fraud_detection

# Configurer les variables
export HF_USERNAME=votre-username-hf
export HF_SPACE_NAME=fraud-detection-app
export HF_TOKEN=hf_votre_token

# Déployer
chmod +x deploy_to_huggingface.sh
./deploy_to_huggingface.sh
```

**Le script va:**
1. Login à Hugging Face
2. Créer le Space (si n'existe pas)
3. Copier les fichiers nécessaires
4. Push vers HF

### 7.3 Configuration Manuelle (Alternative)

```bash
# Cloner le Space
git clone https://huggingface.co/spaces/votre-username/fraud-detection-app
cd fraud-detection-app

# Copier les fichiers
cp ../fraud_detection/app.py .
cp ../fraud_detection/requirements.txt .
cp -r ../fraud_detection/src .
cp -r ../fraud_detection/configs .
mkdir -p models
cp -r ../fraud_detection/final_model/* models/

# Créer README.md pour HF
cat > README.md <<EOF
---
title: Fraud Detection System
emoji: 🛡️
colorFrom: red
colorTo: blue
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
---

# 🛡️ Fraud Detection System

Système intelligent de détection de fraude avec GNN+LLM+RLHF.
EOF

# Push
git add .
git commit -m "Deploy Fraud Detection System"
git push
```

### 7.4 Vérifier le Déploiement

```bash
1. Votre app sera accessible sur:
   https://huggingface.co/spaces/votre-username/fraud-detection-app

2. Attendez le build (2-5 minutes)

3. Une fois "Running", testez l'interface:
   - Entrez une transaction
   - Cliquez "Analyser"
   - Vérifiez la prédiction et l'explication
```

---

## 8. MONITORING ET MÉTRIQUES

### 8.1 WandB Setup

```bash
# Localement
pip install wandb
wandb login
# Entrez votre API key

# Dans Kaggle Notebook:
# Ajoutez votre WANDB_API_KEY dans Secrets
# Le notebook l'utilisera automatiquement
```

**Dashboard:**
```bash
1. Pendant l'entraînement, allez sur: https://wandb.ai
2. Projet: fraud-detection-kaggle
3. Vous verrez en temps réel:
   - Loss curves
   - Metrics (F1, Precision, Recall, etc.)
   - GPU utilization
   - System metrics
```

### 8.2 Logs Locaux

```bash
# Voir les logs
cat logs/training.log

# Suivre en temps réel
tail -f logs/training.log

# Voir les métriques
cat logs/validation_reports/*.json | jq '.'
```

### 8.3 TensorBoard

```bash
# Lancer TensorBoard
tensorboard --logdir logs/tensorboard

# Ouvrir: http://localhost:6006
```

---

## 9. RLHF ET FINE-TUNING CONTINU

### 9.1 Collecter du Feedback Humain

**Via l'interface Gradio:**

```bash
1. Ouvrez l'interface: http://localhost:7860
   ou sur HF: https://huggingface.co/spaces/votre-username/fraud-detection-app

2. Analysez des transactions

3. Pour chaque prédiction:
   - Si CORRECTE: Cliquez "✅ Correct"
   - Si INCORRECTE: Cliquez "❌ Incorrect" et donnez raison
   - Si INCERTAINE: Cliquez "🤔 Pas sûr"

4. Le feedback est sauvegardé dans memory/human_feedback.pkl
```

### 9.2 Relancer le Fine-tuning avec Feedback

```bash
# Localement ou sur Kaggle

python -c "
from src.training.day_night_trainer import DayNightTrainer
from src.models.hybrid_model import create_hybrid_model
from src.utils.memory import TemporalMemory
import yaml

# Charger config
with open('configs/config.yaml') as f:
    config = yaml.safe_load(f)

# Charger modèle
model = create_hybrid_model(config['gnn'], config['llm'])
model.load_pretrained('final_model')

# Charger la mémoire
memory = TemporalMemory('memory')

# Créer trainer
trainer = DayNightTrainer(model, config)

# Fine-tuning avec feedback
feedback_data = memory.get_recent(days=7, label='human_feedback')
print(f'Feedback collecté: {len(feedback_data)} entrées')

# RLHF training
rlhf_results = trainer._rlhf_training(feedback_data)
print(f'RLHF terminé: {rlhf_results}')

# Sauvegarder
model.save_pretrained('final_model_v2')
print('✅ Modèle mis à jour sauvegardé')
"
```

### 9.3 Cycle Continu (Production)

```bash
# Script de surveillance continue
python -c "
import schedule
import time
from src.training.day_night_trainer import DayNightTrainer

def job():
    print('🔄 Démarrage du cycle jour/nuit...')
    # Votre code de training
    pass

# Programmer
schedule.every().day.at('20:00').do(job)  # Nuit
schedule.every().day.at('07:00').do(job)  # Matin

while True:
    schedule.run_pending()
    time.sleep(60)
"
```

---

## 10. TROUBLESHOOTING

### Erreur: CUDA Out of Memory

```yaml
# Dans configs/config.yaml
dataset:
  batch_size: 128  # Réduire de 256 à 128
  
gnn:
  hidden_channels: 256  # Réduire de 512 à 256
```

### Erreur: Git Push Failed

```bash
# Vérifier le remote
git remote -v

# Réinitialiser
git remote remove origin
git remote add origin https://github.com/votre-username/fraud-detection-project.git

# Avec token
git remote set-url origin https://ghp_votre_token@github.com/votre-username/fraud-detection-project.git

# Retry
git push -u origin main --force  # Attention: --force écrase l'historique
```

### Erreur: Kaggle API

```bash
# Vérifier credentials
cat ~/.kaggle/kaggle.json

# Permissions
chmod 600 ~/.kaggle/kaggle.json

# Tester
kaggle competitions list

# Si erreur 401:
# Régénérez un nouveau token sur Kaggle
```

### Erreur: Module Not Found

```bash
# Réinstaller
pip install -r requirements.txt --force-reinstall

# Ou individuellement
pip install torch torch-geometric transformers peft trl
```

### Erreur: Hugging Face Deployment

```bash
# Vérifier le token
echo $HF_TOKEN

# Login manuel
huggingface-cli login

# Retry deployment
./deploy_to_huggingface.sh
```

---

## 📊 RÉCAPITULATIF DES WORKFLOWS

### Workflow 1: Développement Local

```
Vous codez localement
     ↓
git add . && git commit -m "Update"
     ↓
git push origin main
     ↓
GitHub Actions déclenche sync Kaggle
     ↓
Notebook Kaggle mis à jour automatiquement
```

### Workflow 2: Training sur Kaggle

```
Ouvrir Notebook Kaggle
     ↓
Run All (sur GPU P100)
     ↓
Training automatique (Jour → Nuit → Matin)
     ↓
Résultats push vers GitHub
     ↓
Vous récupérez les résultats
```

### Workflow 3: Déploiement Continu

```
Training terminé sur Kaggle
     ↓
Modèle push vers GitHub
     ↓
Vous déployez sur Hugging Face
     ↓
Interface web accessible publiquement
     ↓
Collecte de feedback
     ↓
Re-training avec RLHF
```

---

## ✅ CHECKLIST FINALE

- [ ] Repository GitHub créé et code pushé
- [ ] GitHub Secrets configurés (KAGGLE_USERNAME, KAGGLE_KEY)
- [ ] Kaggle API configurée localement
- [ ] Compétition IEEE-CIS acceptée
- [ ] Notebook Kaggle créé et testé
- [ ] GitHub Actions fonctionnel (sync auto)
- [ ] Training sur GPU P100 réussi
- [ ] Résultats récupérés depuis Kaggle
- [ ] Hugging Face Space déployé
- [ ] Interface Gradio accessible
- [ ] WandB configuré (optionnel)
- [ ] Système de feedback RLHF actif

---

## 🎯 PROCHAINES ÉTAPES

1. **Optimiser les hyperparamètres**
2. **Tester différents modèles GNN** (GCN vs GAT vs GraphSAGE)
3. **Collecter plus de feedback** pour RLHF
4. **Ajouter des features** dans le preprocessing
5. **Créer une API REST** pour intégration
6. **Monitoring production** avec alertes
7. **Documentation utilisateur** finale
8. **Présentation PFE** 🎓

---

**Bon courage pour votre PFE ! 🚀**

Si vous avez des questions, référez-vous à cette documentation ou contactez-nous.

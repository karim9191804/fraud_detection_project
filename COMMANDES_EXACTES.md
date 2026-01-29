# 🚀 COMMANDES EXACTES - DÉMARRAGE IMMÉDIAT

Copiez-collez ces commandes pour démarrer en 30 minutes !

---

## 📥 ÉTAPE 1: TÉLÉCHARGEMENT (2 min)

```bash
# Téléchargez fraud_detection_complete.zip depuis cette conversation
# Puis:

unzip fraud_detection_complete.zip
cd fraud_detection

# Vérifier le contenu
ls -la
```

✅ Vous devriez voir: src/, configs/, main.py, etc.

---

## ⚙️ ÉTAPE 2: INSTALLATION (5 min)

```bash
# Créer environnement virtuel
python3 -m venv venv

# Activer (Linux/Mac)
source venv/bin/activate

# OU Activer (Windows)
venv\Scripts\activate

# Installer dépendances
pip install --upgrade pip
pip install -r requirements.txt

# Vérifier installation
python -c "import torch; print('✅ PyTorch:', torch.__version__)"
python -c "import torch_geometric; print('✅ PyG OK')"
python -c "import transformers; print('✅ Transformers OK')"
```

✅ Si tout affiche OK, continuez !

---

## 🔑 ÉTAPE 3: TOKENS (5 min)

### A. GitHub Token

1. Allez sur: https://github.com/settings/tokens
2. "Generate new token (classic)"
3. Nom: `fraud-detection`
4. Cochez: `repo`, `workflow`
5. Generate → **COPIEZ LE TOKEN**

### B. Kaggle Credentials

1. Allez sur: https://www.kaggle.com/settings/account
2. "Create New API Token"
3. Télécharge `kaggle.json`
4. **OUVREZ le fichier** et copiez username et key

### C. Créer .env

```bash
# Créer le fichier
cat > .env << 'EOF'
# GitHub
GITHUB_TOKEN=ghp_COLLEZ_VOTRE_TOKEN_ICI

# Kaggle
KAGGLE_USERNAME=votre_username_kaggle
KAGGLE_KEY=votre_key_depuis_kaggle_json

# Hugging Face (optionnel)
HF_TOKEN=hf_votre_token

# WandB (optionnel)
WANDB_API_KEY=votre_wandb_key
EOF

# Éditer le fichier
nano .env  # Ou code .env, vim .env, etc.
```

✅ Remplacez TOUS les placeholders par vos vraies valeurs !

---

## 📂 ÉTAPE 4: SETUP GITHUB (5 min)

### Créer le repo sur GitHub.com

1. Allez sur: https://github.com/new
2. Name: `fraud-detection-project`
3. ❌ Ne cochez RIEN
4. "Create repository"
5. **NOTEZ L'URL** (ex: https://github.com/username/fraud-detection-project.git)

### Push le code

```bash
# Dans le dossier fraud_detection

# Init Git
git init
git config user.name "Votre Nom"
git config user.email "votre.email@example.com"

# Ajouter tout
git add .
git commit -m "Initial commit: Fraud Detection System"

# Renommer branche
git branch -M main

# Ajouter remote (REMPLACEZ votre-username)
git remote add origin https://github.com/votre-username/fraud-detection-project.git

# Push
git push -u origin main
```

✅ Allez sur GitHub, vous devriez voir vos fichiers !

### Configurer GitHub Secrets

1. Sur votre repo GitHub: Settings → Secrets and variables → Actions
2. "New repository secret"

**Secret 1:**
- Name: `KAGGLE_USERNAME`
- Value: votre_username (depuis .env)

**Secret 2:**
- Name: `KAGGLE_KEY`
- Value: votre_key (depuis .env)

✅ Vérifiez: Settings → Secrets devrait afficher 2 secrets

---

## 📊 ÉTAPE 5: SETUP KAGGLE (5 min)

### Configurer API localement

```bash
# Créer dossier
mkdir -p ~/.kaggle

# Créer kaggle.json avec vos credentials
cat > ~/.kaggle/kaggle.json << EOF
{
  "username": "votre_username",
  "key": "votre_key"
}
EOF

# Sécuriser
chmod 600 ~/.kaggle/kaggle.json

# Tester
kaggle competitions list | head -5
```

✅ Vous devriez voir une liste de compétitions

### Accepter la compétition

1. Allez sur: https://www.kaggle.com/c/ieee-fraud-detection/rules
2. Cliquez "I Understand and Accept"

✅ Vous verrez "You're in!"

---

## 🚀 ÉTAPE 6: PUSH VERS KAGGLE (2 min)

```bash
# REMPLACEZ votre-username par votre vrai username Kaggle
python push_to_kaggle.py --username votre-username
```

Vous verrez:
```
🚀 PUSH VERS KAGGLE
==================================================
📦 Push du code vers Kaggle Dataset...
✅ Dataset créé: votre-username/fraud-detection-code

📓 Push du notebook vers Kaggle...
✅ Notebook pushé: https://www.kaggle.com/code/...

✅ PUSH TERMINÉ
```

✅ Vérifiez sur Kaggle:
- Datasets → Vous devriez voir "fraud-detection-code"
- Code → Vous devriez voir "fraud-detection-training"

---

## 🎮 ÉTAPE 7: TRAINING SUR GPU P100 (1-2h)

### Ouvrir le notebook

1. Allez sur: https://www.kaggle.com/code/votre-username/fraud-detection-training
2. Cliquez dessus pour l'ouvrir

### Configurer

**Activer GPU:**
- En haut à droite: ⚙️ → Accelerator → **GPU P100**
- Internet: **ON**
- Save

**Ajouter Secrets:**
1. Add-ons → Secrets
2. Ajouter:
   - Label: `GITHUB_TOKEN`
   - Value: Votre token GitHub (ghp_...)

### Modifier le notebook

Dans la 3ème cellule, remplacez:

```python
# AVANT
GITHUB_REPO = "votre-username/fraud-detection-project"
GITHUB_TOKEN = "ghp_your_token_here"

# APRÈS (avec Secrets, recommandé)
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()

GITHUB_REPO = "votre-vrai-username/fraud-detection-project"  # ← CHANGEZ
GITHUB_TOKEN = user_secrets.get_secret("GITHUB_TOKEN")
```

### Lancer !

```
Cliquez "Run All" en haut
```

⏰ **Durée:** 1-2 heures

**Ce qui va se passer:**
1. Clone votre code GitHub ✅
2. Télécharge IEEE-CIS depuis Kaggle ✅
3. Prépare les données (10-15 min) ✅
4. Crée le modèle GNN+LLM sur GPU ✅
5. Entraîne (45-60 min):
   - ☀️ Mode JOUR
   - 🌙 Mode NUIT (fine-tuning)
   - 🌅 Mode MATIN (validation)
6. Push résultats vers GitHub ✅

**Surveiller:**
- Output du notebook (scroll pour voir les logs)
- nvidia-smi pour GPU
- WandB (si configuré)

---

## 📥 ÉTAPE 8: RÉCUPÉRER LES RÉSULTATS (5 min)

### Automatique via GitHub

```bash
# Localement, pull les résultats
cd fraud_detection
git pull origin main
```

✅ Vous devriez voir:
- `checkpoints/` - Checkpoints d'entraînement
- `logs/` - Métriques et logs
- `memory/` - Mémoire RLHF
- `final_model/` - Modèle entraîné

### Manuel (si push GitHub a échoué)

Dans Kaggle:
1. Cliquez "Save Version" → "Save & Run All"
2. Une fois terminé, cliquez sur la version
3. Output (à droite) → Téléchargez les dossiers

Localement:
```bash
# Copier les fichiers téléchargés
cp -r ~/Downloads/checkpoints ./
cp -r ~/Downloads/logs ./
cp -r ~/Downloads/memory ./
cp -r ~/Downloads/final_model ./

# Commit
git add checkpoints/ logs/ memory/ final_model/
git commit -m "Add training results from Kaggle P100"
git push origin main
```

---

## 🌐 ÉTAPE 9: DÉPLOYER SUR HUGGING FACE (10 min)

### Obtenir HF Token

1. Allez sur: https://huggingface.co/settings/tokens
2. "New token"
3. Name: `fraud-detection`
4. Type: **Write**
5. Generate → **COPIEZ LE TOKEN**

### Déployer

```bash
# Exporter les variables
export HF_USERNAME=votre-username-hf
export HF_SPACE_NAME=fraud-detection-app
export HF_TOKEN=hf_votre_token

# Lancer le script
chmod +x deploy_to_huggingface.sh
./deploy_to_huggingface.sh
```

**Vous verrez:**
```
🚀 Déploiement sur Hugging Face Spaces
======================================
🔐 Connexion à Hugging Face...
🏗️  Création du Space...
📦 Préparation des fichiers...
⬆️  Push vers Hugging Face...
✅ Déploiement terminé!

🌐 Votre application est accessible sur:
   https://huggingface.co/spaces/votre-username/fraud-detection-app
```

### Vérifier

1. Allez sur l'URL affichée
2. Attendez le build (2-5 min)
3. Une fois "Running", testez:
   - Entrez une transaction
   - Cliquez "Analyser"
   - Vérifiez prédiction + explication

---

## ✅ ÉTAPE 10: TESTER LOCALEMENT (5 min)

```bash
# Lancer l'interface
python -m src.deployment.gradio_app

# Ou
python main.py --mode deploy
```

**Ouvrez:** http://localhost:7860

**Testez:**
1. Onglet "Analyse de Transaction"
2. Remplissez les champs
3. Cliquez "Analyser"
4. Vérifiez résultat

---

## 📊 VÉRIFIER LES MÉTRIQUES

```bash
# Voir les logs
cat logs/training.log | tail -50

# Voir le rapport de validation
cat logs/validation_reports/*.json | jq '.'

# Métriques principales
python -c "
import json
with open('logs/validation_reports/validation_latest.json') as f:
    metrics = json.load(f)
    print('📊 Métriques:')
    print(f\"  F1-Score: {metrics.get('val_metrics', {}).get('f1_score', 0):.4f}\")
    print(f\"  Precision: {metrics.get('val_metrics', {}).get('precision', 0):.4f}\")
    print(f\"  Recall: {metrics.get('val_metrics', {}).get('recall', 0):.4f}\")
    print(f\"  ROC-AUC: {metrics.get('val_metrics', {}).get('roc_auc', 0):.4f}\")
"
```

---

## 🔄 SYNCHRONISATION CONTINUE

### GitHub Actions (Automatique)

GitHub Actions synchronise automatiquement vers Kaggle:
- ✅ À chaque push sur main
- ✅ Toutes les 6 heures
- ✅ Manuellement via Actions tab

**Déclencher manuellement:**
1. GitHub → Actions
2. "Sync with Kaggle"
3. "Run workflow"

### Manuelle

```bash
# Push vers Kaggle
python push_to_kaggle.py --username votre-username

# Ou juste le notebook
python push_to_kaggle.py --username votre-username --skip-dataset
```

---

## 🎯 RÉCAPITULATIF

### Ce que vous avez maintenant:

✅ **Code complet** sur votre machine
✅ **Repository GitHub** avec tout le code
✅ **Dataset Kaggle** avec votre code
✅ **Notebook Kaggle** prêt pour GPU P100
✅ **Modèle entraîné** avec métriques
✅ **Interface web** sur Hugging Face
✅ **Sync automatique** GitHub ↔ Kaggle
✅ **Documentation complète**

### Temps total:
- Setup initial: **30 minutes**
- Training GPU P100: **1-2 heures**
- Déploiement HF: **10 minutes**
- **TOTAL: ~2-3 heures**

---

## 📚 DOCUMENTATION

Si vous bloquez, consultez:

1. **GUIDE_STEP_BY_STEP.md** - Guide détaillé complet
2. **GPU_GUIDE.md** - Optimisation GPU
3. **VERIFICATION_COMPLETE.md** - Récapitulatif projet
4. **README.md** - Documentation technique
5. **QUICKSTART.md** - Démarrage rapide

---

## 🆘 AIDE RAPIDE

### Erreur CUDA Out of Memory
```yaml
# configs/config.yaml
dataset:
  batch_size: 128  # Réduire
```

### Erreur Git
```bash
git remote set-url origin https://ghp_token@github.com/user/repo.git
```

### Erreur Kaggle API
```bash
chmod 600 ~/.kaggle/kaggle.json
```

---

## 🎉 SUCCÈS !

Si vous avez suivi toutes les étapes, vous avez:

✅ Un système complet de détection de fraude
✅ GNN + LLM + RLHF fonctionnel
✅ Entraîné sur GPU P100
✅ Déployé sur le cloud
✅ Prêt pour votre PFE ! 🎓

---

**Questions ? Consultez la documentation ou créez une issue GitHub !**

**Bon courage ! 🚀**

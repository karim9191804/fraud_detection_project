#!/bin/bash
# deploy_to_huggingface.sh - Déploiement automatique sur HF Spaces

echo "🚀 Déploiement sur Hugging Face Spaces"
echo "======================================"

# Variables
HF_USERNAME=${HF_USERNAME:-"your-username"}
HF_SPACE_NAME=${HF_SPACE_NAME:-"fraud-detection-app"}
HF_REPO="https://huggingface.co/spaces/$HF_USERNAME/$HF_SPACE_NAME"

echo "📝 Configuration:"
echo "  Username: $HF_USERNAME"
echo "  Space: $HF_SPACE_NAME"
echo "  Repo: $HF_REPO"
echo ""

# Vérifier si HF_TOKEN est défini
if [ -z "$HF_TOKEN" ]; then
    echo "⚠️  HF_TOKEN non défini"
    echo "Exécutez: export HF_TOKEN=your_token"
    echo "Ou définissez-le dans .env"
    exit 1
fi

# Vérifier huggingface-cli
if ! command -v huggingface-cli &> /dev/null; then
    echo "📦 Installation de huggingface-cli..."
    pip install huggingface-hub
fi

# Login HF
echo ""
echo "🔐 Connexion à Hugging Face..."
huggingface-cli login --token $HF_TOKEN

# Créer le Space (si n'existe pas)
echo ""
echo "🏗️  Création du Space..."
huggingface-cli repo create \
    --type space \
    --space_sdk gradio \
    $HF_SPACE_NAME \
    2>/dev/null || echo "Space déjà existant"

# Préparer les fichiers pour le déploiement
echo ""
echo "📦 Préparation des fichiers..."

# Créer un dossier temporaire
DEPLOY_DIR="./deploy_temp"
rm -rf $DEPLOY_DIR
mkdir -p $DEPLOY_DIR

# Copier les fichiers essentiels
echo "  ✓ app.py"
cp app.py $DEPLOY_DIR/

echo "  ✓ requirements.txt"
cp requirements.txt $DEPLOY_DIR/

echo "  ✓ src/"
cp -r src/ $DEPLOY_DIR/

echo "  ✓ configs/"
cp -r configs/ $DEPLOY_DIR/

# Créer README pour HF
cat > $DEPLOY_DIR/README.md << 'EOF'
---
title: Fraud Detection System
emoji: 🛡️
colorFrom: red
colorTo: blue
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: mit
---

# 🛡️ Fraud Detection System

Système intelligent de détection de fraude combinant **Graph Neural Networks**, **Large Language Models** et **RLHF**.

## Features

- 🔍 Analyse en temps réel
- 💡 Explications en langage naturel
- 📊 Prédictions précises
- 🤖 Apprentissage continu

## Usage

1. Entrez les détails de la transaction
2. Cliquez sur "Analyser"
3. Obtenez la prédiction et l'explication

## Architecture

- **GNN**: GCN/GAT/GraphSAGE pour la détection
- **LLM**: Microsoft Phi-2 pour les explications
- **RLHF**: Apprentissage depuis feedback humain
EOF

# Copier les modèles (si disponibles et pas trop gros)
if [ -d "models" ] && [ $(du -sm models | cut -f1) -lt 10000 ]; then
    echo "  ✓ models/ (taille acceptable)"
    cp -r models/ $DEPLOY_DIR/
else
    echo "  ⚠️  models/ trop gros ou absent (skippe)"
    mkdir -p $DEPLOY_DIR/models
fi

# Initialiser Git dans le dossier de déploiement
cd $DEPLOY_DIR
git init
git remote add space https://huggingface.co/spaces/$HF_USERNAME/$HF_SPACE_NAME
git add .
git commit -m "Deploy Fraud Detection System"

# Push vers HF
echo ""
echo "⬆️  Push vers Hugging Face..."
GIT_LFS_SKIP_SMUDGE=1 git push --force space main

cd ..
rm -rf $DEPLOY_DIR

echo ""
echo "✅ Déploiement terminé!"
echo ""
echo "🌐 Votre application est accessible sur:"
echo "   $HF_REPO"
echo ""
echo "⏳ Note: Le build peut prendre quelques minutes"
echo "   Vérifiez l'avancement sur le Space"
echo ""

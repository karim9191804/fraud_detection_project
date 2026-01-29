#!/bin/bash
# quick_start.sh - Script de démarrage rapide

echo "🚀 Fraud Detection System - Quick Start"
echo "========================================"

# Vérifier Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 n'est pas installé"
    exit 1
fi

echo "✓ Python détecté: $(python3 --version)"

# Créer l'environnement virtuel
echo ""
echo "📦 Création de l'environnement virtuel..."
python3 -m venv venv

# Activer l'environnement
echo "🔧 Activation de l'environnement..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Installer les dépendances
echo ""
echo "📥 Installation des dépendances..."
pip install --upgrade pip
pip install -r requirements.txt

# Créer les répertoires
echo ""
echo "📁 Création des répertoires..."
mkdir -p data/raw/ieee-fraud
mkdir -p data/processed
mkdir -p models
mkdir -p checkpoints
mkdir -p logs
mkdir -p memory

# Vérifier les tokens
echo ""
echo "🔑 Vérification des configurations..."
if [ ! -f .env ]; then
    echo "⚠️  Fichier .env manquant. Création..."
    cat > .env << EOF
# GitHub
GITHUB_TOKEN=your_github_token_here

# Kaggle
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key

# Hugging Face
HF_TOKEN=your_huggingface_token

# WandB (optionnel)
WANDB_API_KEY=your_wandb_key
EOF
    echo "✓ Fichier .env créé. Veuillez le compléter avec vos tokens."
fi

# Vérifier Kaggle credentials
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo ""
    echo "⚠️  Credentials Kaggle manquantes"
    echo "1. Allez sur https://www.kaggle.com/settings/account"
    echo "2. Cliquez sur 'Create New API Token'"
    echo "3. Placez kaggle.json dans ~/.kaggle/"
    echo "4. Exécutez: chmod 600 ~/.kaggle/kaggle.json"
fi

echo ""
echo "✅ Installation terminée!"
echo ""
echo "📋 Prochaines étapes:"
echo "1. Complétez le fichier .env avec vos tokens"
echo "2. Téléchargez les données: kaggle competitions download -c ieee-fraud-detection"
echo "3. Lancez le pipeline: python main.py --mode full"
echo ""
echo "🔗 Commandes utiles:"
echo "  python main.py --mode data      # Préparation données uniquement"
echo "  python main.py --mode train     # Entraînement uniquement"
echo "  python main.py --mode deploy    # Déploiement interface"
echo ""
echo "📚 Documentation: README.md"
echo ""

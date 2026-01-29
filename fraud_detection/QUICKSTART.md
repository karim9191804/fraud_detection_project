# 🚀 Guide de Démarrage Rapide

Ce guide vous permet de démarrer rapidement avec le système de détection de fraude.

## ⚡ Installation Ultra-Rapide (5 minutes)

### Prérequis
- Python 3.10+
- Git
- 8 GB RAM minimum

### Étapes

```bash
# 1. Cloner le projet
git clone https://github.com/votre-username/fraud-detection-project.git
cd fraud-detection-project

# 2. Installation automatique
chmod +x quick_start.sh
./quick_start.sh

# 3. Configurer les tokens (IMPORTANT!)
nano .env  # Ou utilisez votre éditeur préféré
```

Dans `.env`, ajoutez:
```bash
GITHUB_TOKEN=ghp_your_token_here
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_kaggle_api_key
HF_TOKEN=hf_your_token_here
```

## 📥 Obtenir les Données

### Option 1: Via Kaggle API (Recommandé)

```bash
# 1. Configurer Kaggle
mkdir -p ~/.kaggle
# Placer votre kaggle.json dans ~/.kaggle/

# 2. Télécharger
kaggle competitions download -c ieee-fraud-detection
unzip ieee-fraud-detection.zip -d data/raw/ieee-fraud/
```

### Option 2: Téléchargement Manuel

1. Aller sur [Kaggle IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection)
2. Télécharger les fichiers
3. Extraire dans `data/raw/ieee-fraud/`

## 🎯 Premier Entraînement

```bash
# Pipeline complet
python main.py --mode full

# Ou par étapes:

# 1. Préparer les données (10-15 min)
python main.py --mode data

# 2. Entraîner le modèle (30-60 min avec GPU, 2-3h CPU)
python main.py --mode train

# 3. Déployer l'interface (instantané)
python main.py --mode deploy
```

## 🌐 Lancer l'Interface Web

```bash
# Option 1: Localement
python -m src.deployment.gradio_app

# Option 2: Avec le script principal
python main.py --mode deploy

# Option 3: Déployer sur HF Spaces
chmod +x deploy_to_huggingface.sh
./deploy_to_huggingface.sh
```

L'interface sera accessible sur `http://localhost:7860`

## 🔄 Synchronisation GitHub-Kaggle

```bash
# Sync unique
python -c "
from src.deployment.github_kaggle_sync import setup_github_kaggle_sync
import yaml

with open('configs/config.yaml') as f:
    config = yaml.safe_load(f)

sync = setup_github_kaggle_sync(config)
sync.bidirectional_sync()
"

# Surveillance continue (sync automatique)
python -c "
from src.deployment.github_kaggle_sync import setup_github_kaggle_sync
import yaml

with open('configs/config.yaml') as f:
    config = yaml.safe_load(f)

sync = setup_github_kaggle_sync(config)
sync.watch_and_sync()  # Ctrl+C pour arrêter
"
```

## 📊 Surveiller l'Entraînement

### WandB (Recommandé)

```bash
# 1. Login
wandb login

# 2. Lancer training (auto-logging activé)
python main.py --mode train
```

Vos runs seront visibles sur [wandb.ai](https://wandb.ai)

### TensorBoard

```bash
# Terminal 1: Lancer training
python main.py --mode train

# Terminal 2: Lancer TensorBoard
tensorboard --logdir logs/
```

Accessible sur `http://localhost:6006`

## 🧪 Tests Rapides

```bash
# Test du modèle GNN
python src/models/gnn_model.py

# Test du modèle hybride
python src/models/hybrid_model.py

# Test des métriques
python src/utils/metrics.py

# Test de la mémoire
python src/utils/memory.py

# Tests complets
pytest tests/
```

## 🔧 Configuration Rapide

### Changer le Modèle GNN

Dans `configs/config.yaml`:
```yaml
gnn:
  model_type: "GAT"  # GCN, GAT, ou GraphSAGE
  hidden_channels: 256
  num_layers: 3
```

### Activer/Désactiver RLHF

```yaml
day_night_cycle:
  night:
    rlhf_training: true  # ou false
```

### Modifier le Cycle Jour/Nuit

```yaml
day_night_cycle:
  day:
    human_verification_threshold: 0.7  # 0-1
  morning:
    human_check_required: true
```

## 📝 Commandes Utiles

```bash
# Voir les logs
tail -f logs/training.log

# Nettoyer les fichiers temporaires
rm -rf __pycache__ *.pyc logs/* checkpoints/*

# Sauvegarder un checkpoint
python -c "
from src.models.hybrid_model import create_hybrid_model
import yaml

with open('configs/config.yaml') as f:
    config = yaml.safe_load(f)

# Créer et sauvegarder
model = create_hybrid_model(config['gnn'], config['llm'])
model.save_pretrained('models/my_model')
"

# Charger un checkpoint
python -c "
from src.models.hybrid_model import GNNLLMHybrid
model = GNNLLMHybrid(...)
model.load_pretrained('models/my_model')
"
```

## 🐛 Résolution de Problèmes

### Erreur: CUDA out of memory

```bash
# Réduire le batch size dans config.yaml
dataset:
  batch_size: 64  # au lieu de 128
```

### Erreur: Module not found

```bash
# Réinstaller les dépendances
pip install -r requirements.txt --force-reinstall
```

### Erreur: Kaggle API

```bash
# Vérifier les credentials
cat ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

### Erreur: GitHub push

```bash
# Vérifier le token
echo $GITHUB_TOKEN

# Réinitialiser le remote
git remote remove origin
git remote add origin https://$GITHUB_TOKEN@github.com/username/repo.git
```

## 📚 Ressources Supplémentaires

- [README Complet](README.md)
- [Documentation API](docs/api.md)
- [Architecture](docs/architecture.md)
- [RLHF Guide](docs/rlhf.md)

## 💡 Conseils

1. **Commencez petit**: Testez avec un subset des données
2. **Surveillez les métriques**: Utilisez WandB dès le début
3. **Sauvegardez souvent**: Les checkpoints sont vos amis
4. **GPU recommandé**: L'entraînement sera 10-20x plus rapide
5. **Feedback humain**: Essentiel pour RLHF

## 🎯 Workflow Recommandé

```
1. Installation (5 min)
   └─> quick_start.sh

2. Configuration (2 min)
   └─> Éditer .env et config.yaml

3. Données (15 min)
   └─> python main.py --mode data

4. Entraînement (1-3h)
   └─> python main.py --mode train
   └─> Surveiller avec WandB

5. Évaluation
   └─> Vérifier les métriques dans logs/

6. Déploiement (5 min)
   └─> ./deploy_to_huggingface.sh

7. Utilisation
   └─> Interface web + API
```

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/votre-username/fraud-detection-project/issues)
- **Email**: votre.email@example.com
- **Discord**: [Serveur Discord](https://discord.gg/your-server)

---

**Happy Coding! 🚀**

"""
Main Orchestration Script
Pipeline complet: Data → Training → Deployment
"""

import argparse
import yaml
import os
from pathlib import Path
import torch

from src.data.ieee_dataset import prepare_ieee_dataset, IEEEFraudPreprocessor
from src.models.hybrid_model import create_hybrid_model
from src.training.day_night_trainer import DayNightTrainer, schedule_day_night_cycle
from src.deployment.github_kaggle_sync import setup_github_kaggle_sync
from src.deployment.gradio_app import launch_app


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Charger la configuration"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_directories(config: dict):
    """Créer tous les répertoires nécessaires"""
    directories = [
        config["paths"]["data_dir"],
        config["paths"]["processed_dir"],
        config["paths"]["models_dir"],
        config["paths"]["logs_dir"],
        config["paths"]["checkpoints_dir"],
        config["paths"]["memory_dir"]
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ {directory}")


def prepare_data(config: dict):
    """Pipeline de préparation des données"""
    print("\n" + "=" * 60)
    print("ÉTAPE 1: PRÉPARATION DES DONNÉES")
    print("=" * 60)
    
    # Préparer le dataset IEEE-CIS
    dataset = prepare_ieee_dataset(
        data_dir=config["paths"]["data_dir"],
        output_dir=config["paths"]["processed_dir"],
        test_size=config["dataset"]["test_ratio"],
        val_size=config["dataset"]["val_ratio"]
    )
    
    return dataset


def create_model(config: dict):
    """Créer le modèle hybride GNN+LLM"""
    print("\n" + "=" * 60)
    print("ÉTAPE 2: CRÉATION DU MODÈLE")
    print("=" * 60)
    
    gnn_config = {
        "in_channels": 432,  # IEEE-CIS features
        "hidden_channels": config["gnn"]["hidden_channels"],
        "num_layers": config["gnn"]["num_layers"],
        "dropout": config["gnn"]["dropout"],
        "model_type": config["gnn"]["model_type"]
    }
    
    llm_config = {
        "model_name": config["llm"]["model_name"],
        "max_length": config["llm"]["max_length"],
        "temperature": config["llm"]["temperature"],
        "use_lora": config["llm"]["use_lora"],
        "lora_r": config["llm"]["lora_r"],
        "lora_alpha": config["llm"]["lora_alpha"]
    }
    
    model = create_hybrid_model(gnn_config, llm_config)
    
    print(f"\n✅ Modèle créé:")
    print(f"  - GNN: {config['gnn']['model_type']}")
    print(f"  - LLM: {config['llm']['model_name']}")
    print(f"  - LoRA: {'Activé' if config['llm']['use_lora'] else 'Désactivé'}")
    
    return model


def train_model(model, dataset, config: dict):
    """Entraîner le modèle avec le système jour/nuit"""
    print("\n" + "=" * 60)
    print("ÉTAPE 3: ENTRAÎNEMENT")
    print("=" * 60)
    
    # Créer le trainer
    trainer = DayNightTrainer(
        model=model,
        config=config
    )
    
    # Créer les data loaders
    from torch_geometric.loader import DataLoader
    
    train_loader = DataLoader(
        [dataset['train']],
        batch_size=config["dataset"]["batch_size"],
        shuffle=True
    )
    
    val_loader = DataLoader(
        [dataset['val']],
        batch_size=config["dataset"]["batch_size"],
        shuffle=False
    )
    
    test_loader = DataLoader(
        [dataset['test']],
        batch_size=config["dataset"]["batch_size"],
        shuffle=False
    )
    
    # Mode 1: Training complet (une fois)
    print("\n🔧 Mode: Training complet")
    
    # Jour: Inférence initiale
    day_results = trainer.day_mode_inference(train_loader)
    
    # Nuit: Training
    night_results = trainer.night_mode_training(
        train_loader,
        val_loader,
        num_epochs=config["fine_tuning"]["num_epochs"]
    )
    
    # Matin: Validation
    morning_results = trainer.morning_validation(val_loader, test_loader)
    
    return trainer, morning_results


def setup_sync(config: dict):
    """Configurer la synchronisation GitHub-Kaggle"""
    print("\n" + "=" * 60)
    print("ÉTAPE 4: SYNCHRONISATION GITHUB-KAGGLE")
    print("=" * 60)
    
    if not config.get("github", {}).get("auto_sync", False):
        print("⚠️ Auto-sync désactivé dans la config")
        return None
    
    sync = setup_github_kaggle_sync(config)
    
    # Synchronisation initiale
    print("\n🔄 Synchronisation initiale...")
    sync.bidirectional_sync()
    
    return sync


def deploy_app(config: dict, model_path: str = None):
    """Déployer l'application Gradio"""
    print("\n" + "=" * 60)
    print("ÉTAPE 5: DÉPLOIEMENT")
    print("=" * 60)
    
    print("\n🚀 Lancement de l'interface Gradio...")
    
    # Lancer l'app
    launch_app(
        share=True,  # Créer un lien public
        server_port=7860
    )


def main():
    """Pipeline principal"""
    parser = argparse.ArgumentParser(description="Fraud Detection System - Main Pipeline")
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Chemin vers le fichier de configuration"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["full", "data", "train", "deploy", "sync"],
        default="full",
        help="Mode d'exécution"
    )
    
    parser.add_argument(
        "--skip-data",
        action="store_true",
        help="Sauter la préparation des données"
    )
    
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Sauter l'entraînement"
    )
    
    args = parser.parse_args()
    
    # Charger la configuration
    print("📋 Chargement de la configuration...")
    config = load_config(args.config)
    
    # Setup des répertoires
    print("\n📁 Configuration des répertoires...")
    setup_directories(config)
    
    # Pipeline complet ou partiel
    if args.mode == "full" or args.mode == "data":
        if not args.skip_data:
            dataset = prepare_data(config)
        else:
            print("\n⏭️ Préparation des données sautée")
            # Charger le dataset existant
            dataset = {
                'train': torch.load(f"{config['paths']['processed_dir']}/train_data.pt"),
                'val': torch.load(f"{config['paths']['processed_dir']}/val_data.pt"),
                'test': torch.load(f"{config['paths']['processed_dir']}/test_data.pt")
            }
    
    if args.mode == "full" or args.mode == "train":
        if not args.skip_training:
            model = create_model(config)
            trainer, results = train_model(model, dataset, config)
            
            # Sauvegarder le modèle final
            model_path = f"{config['paths']['models_dir']}/final_model"
            model.save_pretrained(model_path)
            print(f"\n💾 Modèle sauvegardé: {model_path}")
        else:
            print("\n⏭️ Entraînement sauté")
            model_path = f"{config['paths']['models_dir']}/final_model"
    
    if args.mode == "full" or args.mode == "sync":
        sync = setup_sync(config)
        if sync and config.get("github", {}).get("auto_sync"):
            # Lancer la surveillance continue en arrière-plan
            print("\n👁️ Surveillance GitHub-Kaggle activée")
    
    if args.mode == "full" or args.mode == "deploy":
        deploy_app(config, model_path if 'model_path' in locals() else None)
    
    print("\n" + "=" * 60)
    print("✅ PIPELINE TERMINÉ")
    print("=" * 60)


if __name__ == "__main__":
    main()

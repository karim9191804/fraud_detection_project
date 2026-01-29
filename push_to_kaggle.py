#!/usr/bin/env python3
"""
Script pour pousser le code et le notebook vers Kaggle
"""

import os
import json
import argparse
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi


def push_dataset(username: str, dataset_name: str = "fraud-detection-code"):
    """
    Créer ou mettre à jour le dataset Kaggle avec le code
    
    Args:
        username: Kaggle username
        dataset_name: Nom du dataset
    """
    print(f"\n📦 Push du code vers Kaggle Dataset...")
    
    api = KaggleApi()
    api.authenticate()
    
    # Créer metadata
    metadata = {
        "title": "Fraud Detection Code",
        "id": f"{username}/{dataset_name}",
        "licenses": [{"name": "MIT"}],
        "resources": []
    }
    
    # Sauvegarder metadata
    with open("dataset-metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    try:
        # Vérifier si existe
        api.dataset_status(f"{username}/{dataset_name}")
        
        # Update
        api.dataset_create_version(
            folder=".",
            version_notes=f"Auto-update from local",
            quiet=False
        )
        print(f"✅ Dataset mis à jour: {username}/{dataset_name}")
        
    except Exception:
        # Créer nouveau
        api.dataset_create_new(
            folder=".",
            public=False,
            quiet=False
        )
        print(f"✅ Nouveau dataset créé: {username}/{dataset_name}")
    
    # Nettoyer
    if os.path.exists("dataset-metadata.json"):
        os.remove("dataset-metadata.json")


def push_notebook(username: str, notebook_name: str = "fraud-detection-training"):
    """
    Créer ou mettre à jour le notebook Kaggle
    
    Args:
        username: Kaggle username
        notebook_name: Nom du notebook
    """
    print(f"\n📓 Push du notebook vers Kaggle...")
    
    api = KaggleApi()
    api.authenticate()
    
    # Vérifier que le notebook existe
    notebook_path = Path("kaggle_train_gpu.ipynb")
    if not notebook_path.exists():
        print(f"❌ Notebook non trouvé: {notebook_path}")
        return
    
    # Créer metadata du kernel
    kernel_metadata = {
        "id": f"{username}/{notebook_name}",
        "title": "Fraud Detection Training GPU P100",
        "code_file": str(notebook_path),
        "language": "python",
        "kernel_type": "notebook",
        "is_private": False,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": ["ieee-fraud-detection"],
        "competition_sources": [],
        "kernel_sources": []
    }
    
    # Sauvegarder metadata
    with open("kernel-metadata.json", "w") as f:
        json.dump(kernel_metadata, f, indent=2)
    
    try:
        # Push
        api.kernels_push(".")
        print(f"✅ Notebook pushé: https://www.kaggle.com/code/{username}/{notebook_name}")
        
    except Exception as e:
        print(f"❌ Erreur lors du push: {e}")
    
    # Nettoyer
    if os.path.exists("kernel-metadata.json"):
        os.remove("kernel-metadata.json")


def main():
    parser = argparse.ArgumentParser(description="Push code to Kaggle")
    
    parser.add_argument(
        "--username",
        type=str,
        required=True,
        help="Kaggle username"
    )
    
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="fraud-detection-code",
        help="Nom du dataset Kaggle"
    )
    
    parser.add_argument(
        "--notebook-name",
        type=str,
        default="fraud-detection-training",
        help="Nom du notebook Kaggle"
    )
    
    parser.add_argument(
        "--skip-dataset",
        action="store_true",
        help="Sauter le push du dataset"
    )
    
    parser.add_argument(
        "--skip-notebook",
        action="store_true",
        help="Sauter le push du notebook"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 PUSH VERS KAGGLE")
    print("=" * 60)
    
    # Push dataset
    if not args.skip_dataset:
        push_dataset(args.username, args.dataset_name)
    
    # Push notebook
    if not args.skip_notebook:
        push_notebook(args.username, args.notebook_name)
    
    print("\n" + "=" * 60)
    print("✅ PUSH TERMINÉ")
    print("=" * 60)
    
    print(f"\n🔗 Liens utiles:")
    print(f"  Dataset: https://www.kaggle.com/datasets/{args.username}/{args.dataset_name}")
    print(f"  Notebook: https://www.kaggle.com/code/{args.username}/{args.notebook_name}")
    
    print(f"\n📝 Prochaines étapes:")
    print(f"  1. Allez sur le notebook Kaggle")
    print(f"  2. Vérifiez que GPU P100 est activé")
    print(f"  3. Configurez les Secrets (GITHUB_TOKEN, etc.)")
    print(f"  4. Cliquez 'Run All'")
    print(f"  5. Attendez ~1-2h pour l'entraînement")


if __name__ == "__main__":
    main()

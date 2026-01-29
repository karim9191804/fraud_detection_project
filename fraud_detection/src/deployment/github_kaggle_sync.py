"""
GitHub <-> Kaggle Bidirectional Synchronization
Synchronise automatiquement le code et les données entre GitHub et Kaggle
"""

import os
import time
import json
from typing import Optional, List
from datetime import datetime
from pathlib import Path

# GitHub
from github import Github
import git

# Kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile
import shutil


class GitHubKaggleSync:
    """
    Synchronisation bidirectionnelle GitHub <-> Kaggle
    """
    def __init__(
        self,
        github_repo: str,
        kaggle_dataset: str,
        kaggle_username: str,
        local_dir: str = ".",
        github_token: Optional[str] = None,
        sync_interval: int = 300
    ):
        self.github_repo = github_repo
        self.kaggle_dataset = kaggle_dataset
        self.kaggle_username = kaggle_username
        self.local_dir = Path(local_dir)
        self.sync_interval = sync_interval
        
        # GitHub setup
        self.github_token = github_token or os.getenv("GITHUB_TOKEN")
        self.github = Github(self.github_token) if self.github_token else None
        self.repo = None
        
        # Kaggle setup
        self.kaggle_api = KaggleApi()
        self.kaggle_api.authenticate()
        
        # Git repo local
        self.git_repo = None
        
        self._setup()
    
    def _setup(self):
        """Configuration initiale"""
        print("🔧 Configuration de la synchronisation...")
        
        # Setup GitHub
        if self.github:
            self.repo = self.github.get_repo(self.github_repo)
            print(f"✓ GitHub repo connecté: {self.github_repo}")
        
        # Setup Git local
        try:
            self.git_repo = git.Repo(self.local_dir)
            print(f"✓ Git repo local: {self.local_dir}")
        except git.InvalidGitRepositoryError:
            print("⚠️ Pas de repo Git local, initialisation...")
            self.git_repo = git.Repo.init(self.local_dir)
            
            # Ajouter le remote
            if self.github_token:
                remote_url = f"https://{self.github_token}@github.com/{self.github_repo}.git"
                try:
                    self.git_repo.create_remote("origin", remote_url)
                except git.exc.GitCommandError:
                    pass  # Remote déjà existant
        
        print("✓ Synchronisation configurée")
    
    def push_to_github(
        self,
        files: Optional[List[str]] = None,
        commit_message: Optional[str] = None
    ) -> bool:
        """
        Pousser les changements vers GitHub
        
        Args:
            files: Fichiers spécifiques à commiter (None = tous)
            commit_message: Message de commit
        
        Returns:
            Succès
        """
        try:
            print("\n📤 Push vers GitHub...")
            
            # Ajouter les fichiers
            if files:
                self.git_repo.index.add(files)
            else:
                self.git_repo.git.add(A=True)
            
            # Commit
            if not commit_message:
                commit_message = f"Auto-sync: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            try:
                self.git_repo.index.commit(commit_message)
                print(f"✓ Commit: {commit_message}")
            except git.exc.GitCommandError as e:
                if "nothing to commit" in str(e):
                    print("ℹ️ Pas de changements à commiter")
                    return True
                raise
            
            # Push
            origin = self.git_repo.remote("origin")
            origin.push()
            
            print("✅ Changements poussés vers GitHub")
            return True
            
        except Exception as e:
            print(f"❌ Erreur lors du push GitHub: {e}")
            return False
    
    def pull_from_github(self) -> bool:
        """
        Récupérer les changements depuis GitHub
        
        Returns:
            Succès
        """
        try:
            print("\n📥 Pull depuis GitHub...")
            
            origin = self.git_repo.remote("origin")
            origin.pull()
            
            print("✅ Changements récupérés depuis GitHub")
            return True
            
        except Exception as e:
            print(f"❌ Erreur lors du pull GitHub: {e}")
            return False
    
    def upload_to_kaggle(
        self,
        files_dir: str = ".",
        dataset_metadata: Optional[dict] = None
    ) -> bool:
        """
        Uploader les fichiers vers Kaggle dataset
        
        Args:
            files_dir: Dossier contenant les fichiers à uploader
            dataset_metadata: Métadonnées du dataset
        
        Returns:
            Succès
        """
        try:
            print("\n📤 Upload vers Kaggle...")
            
            # Créer les métadonnées si nécessaires
            if not dataset_metadata:
                dataset_metadata = {
                    "title": self.kaggle_dataset.split("/")[-1],
                    "id": f"{self.kaggle_username}/{self.kaggle_dataset.split('/')[-1]}",
                    "licenses": [{"name": "MIT"}]
                }
            
            # Sauvegarder les métadonnées
            metadata_path = Path(files_dir) / "dataset-metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(dataset_metadata, f, indent=2)
            
            # Upload/Update dataset
            try:
                # Vérifier si le dataset existe
                self.kaggle_api.dataset_status(self.kaggle_dataset)
                
                # Update existant
                self.kaggle_api.dataset_create_version(
                    folder=files_dir,
                    version_notes=f"Auto-sync: {datetime.now().isoformat()}",
                    quiet=False
                )
                print(f"✅ Dataset Kaggle mis à jour: {self.kaggle_dataset}")
                
            except Exception:
                # Créer nouveau dataset
                self.kaggle_api.dataset_create_new(
                    folder=files_dir,
                    public=False,
                    quiet=False
                )
                print(f"✅ Nouveau dataset Kaggle créé: {self.kaggle_dataset}")
            
            return True
            
        except Exception as e:
            print(f"❌ Erreur lors de l'upload Kaggle: {e}")
            return False
    
    def download_from_kaggle(
        self,
        output_dir: str = "data/kaggle",
        unzip: bool = True
    ) -> bool:
        """
        Télécharger le dataset depuis Kaggle
        
        Args:
            output_dir: Dossier de destination
            unzip: Dézipper automatiquement
        
        Returns:
            Succès
        """
        try:
            print("\n📥 Download depuis Kaggle...")
            
            os.makedirs(output_dir, exist_ok=True)
            
            # Télécharger
            self.kaggle_api.dataset_download_files(
                self.kaggle_dataset,
                path=output_dir,
                unzip=unzip
            )
            
            print(f"✅ Dataset téléchargé: {output_dir}")
            return True
            
        except Exception as e:
            print(f"❌ Erreur lors du download Kaggle: {e}")
            return False
    
    def sync_github_to_kaggle(self) -> bool:
        """
        Synchroniser GitHub → Kaggle
        
        Returns:
            Succès
        """
        print("\n🔄 Synchronisation GitHub → Kaggle")
        print("=" * 50)
        
        # 1. Pull depuis GitHub
        if not self.pull_from_github():
            return False
        
        # 2. Upload vers Kaggle
        if not self.upload_to_kaggle(self.local_dir):
            return False
        
        print("✅ Synchronisation GitHub → Kaggle réussie")
        return True
    
    def sync_kaggle_to_github(self, kaggle_data_dir: str = "data/kaggle") -> bool:
        """
        Synchroniser Kaggle → GitHub
        
        Args:
            kaggle_data_dir: Dossier où télécharger les données Kaggle
        
        Returns:
            Succès
        """
        print("\n🔄 Synchronisation Kaggle → GitHub")
        print("=" * 50)
        
        # 1. Download depuis Kaggle
        if not self.download_from_kaggle(kaggle_data_dir):
            return False
        
        # 2. Push vers GitHub
        if not self.push_to_github(
            commit_message=f"Sync from Kaggle: {datetime.now().isoformat()}"
        ):
            return False
        
        print("✅ Synchronisation Kaggle → GitHub réussie")
        return True
    
    def bidirectional_sync(self) -> bool:
        """
        Synchronisation bidirectionnelle complète
        
        Returns:
            Succès
        """
        print("\n🔄 SYNCHRONISATION BIDIRECTIONNELLE")
        print("=" * 60)
        
        # 1. GitHub → Local → Kaggle
        success_1 = self.sync_github_to_kaggle()
        
        time.sleep(2)
        
        # 2. Kaggle → Local → GitHub
        success_2 = self.sync_kaggle_to_github()
        
        if success_1 and success_2:
            print("\n✅ SYNCHRONISATION COMPLÈTE RÉUSSIE")
            return True
        else:
            print("\n⚠️ SYNCHRONISATION PARTIELLE")
            return False
    
    def watch_and_sync(self):
        """
        Surveiller les changements et synchroniser automatiquement
        """
        print(f"\n👁️ Surveillance active (intervalle: {self.sync_interval}s)")
        print("Appuyez sur Ctrl+C pour arrêter")
        
        try:
            while True:
                print(f"\n⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Vérifier les changements locaux
                if self.git_repo.is_dirty():
                    print("📝 Changements locaux détectés")
                    self.sync_github_to_kaggle()
                else:
                    print("✓ Pas de changements locaux")
                
                # Attendre
                time.sleep(self.sync_interval)
                
        except KeyboardInterrupt:
            print("\n\n🛑 Surveillance arrêtée")


class KaggleNotebookRunner:
    """
    Exécuter des notebooks sur Kaggle depuis le code local
    """
    def __init__(self, kaggle_username: str):
        self.kaggle_username = kaggle_username
        self.kaggle_api = KaggleApi()
        self.kaggle_api.authenticate()
    
    def push_notebook(
        self,
        notebook_path: str,
        title: str,
        dataset_sources: Optional[List[str]] = None,
        enable_gpu: bool = True,
        enable_internet: bool = True
    ) -> str:
        """
        Pousser et exécuter un notebook sur Kaggle
        
        Args:
            notebook_path: Chemin du notebook local
            title: Titre du notebook
            dataset_sources: Datasets à attacher
            enable_gpu: Activer GPU
            enable_internet: Activer Internet
        
        Returns:
            URL du notebook Kaggle
        """
        print(f"\n📤 Push notebook vers Kaggle: {title}")
        
        # Créer les métadonnées du kernel
        kernel_metadata = {
            "id": f"{self.kaggle_username}/{title.lower().replace(' ', '-')}",
            "title": title,
            "code_file": notebook_path,
            "language": "python",
            "kernel_type": "notebook",
            "is_private": False,
            "enable_gpu": enable_gpu,
            "enable_internet": enable_internet,
            "dataset_sources": dataset_sources or [],
            "competition_sources": [],
            "kernel_sources": []
        }
        
        # Sauvegarder les métadonnées
        metadata_file = Path(notebook_path).parent / "kernel-metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(kernel_metadata, f, indent=2)
        
        try:
            # Push le kernel
            self.kaggle_api.kernels_push(str(Path(notebook_path).parent))
            
            kernel_url = f"https://www.kaggle.com/code/{kernel_metadata['id']}"
            print(f"✅ Notebook poussé: {kernel_url}")
            
            return kernel_url
            
        except Exception as e:
            print(f"❌ Erreur lors du push: {e}")
            return ""
    
    def get_notebook_status(self, kernel_slug: str) -> dict:
        """
        Obtenir le statut d'un notebook
        
        Args:
            kernel_slug: Slug du kernel (username/kernel-name)
        
        Returns:
            Statut du kernel
        """
        try:
            status = self.kaggle_api.kernel_status(kernel_slug)
            return {
                "status": status,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            print(f"❌ Erreur: {e}")
            return {"status": "error", "error": str(e)}


def setup_github_kaggle_sync(config: dict) -> GitHubKaggleSync:
    """
    Configuration rapide de la synchronisation
    
    Args:
        config: Configuration du projet
    
    Returns:
        Instance de GitHubKaggleSync
    """
    sync = GitHubKaggleSync(
        github_repo=config["github"]["repo"],
        kaggle_dataset=config["kaggle"]["dataset"],
        kaggle_username=config["kaggle"]["username"],
        sync_interval=config["github"]["sync_interval"]
    )
    
    return sync


if __name__ == "__main__":
    # Test
    print("Testing GitHub-Kaggle Sync...")
    
    # Configuration de test
    test_config = {
        "github": {
            "repo": "username/fraud-detection-project",
            "sync_interval": 300
        },
        "kaggle": {
            "dataset": "ieee-fraud-detection",
            "username": "username"
        }
    }
    
    # Créer l'instance
    sync = setup_github_kaggle_sync(test_config)
    
    print("\n✓ Sync system configured!")
    print("\nPour utiliser:")
    print("  sync.bidirectional_sync()  # Sync complète")
    print("  sync.watch_and_sync()      # Surveillance continue")

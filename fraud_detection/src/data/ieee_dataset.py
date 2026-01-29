"""
IEEE-CIS Fraud Detection Dataset Loading and Preprocessing
"""

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import networkx as nx
from typing import Tuple, Dict, List
import os
from tqdm import tqdm
import pickle


class IEEEFraudDataset(Dataset):
    """
    Dataset PyTorch Geometric pour IEEE-CIS Fraud Detection
    """
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform=None,
        pre_transform=None
    ):
        self.split = split
        super().__init__(root, transform, pre_transform)
        self.data_list = torch.load(self.processed_paths[0])
    
    @property
    def raw_file_names(self) -> List[str]:
        return [
            'train_transaction.csv',
            'train_identity.csv',
            'test_transaction.csv',
            'test_identity.csv'
        ]
    
    @property
    def processed_file_names(self) -> List[str]:
        return [f'{self.split}_data.pt']
    
    def len(self) -> int:
        return len(self.data_list)
    
    def get(self, idx: int) -> Data:
        return self.data_list[idx]


class IEEEFraudPreprocessor:
    """
    Préprocesseur pour les données IEEE-CIS
    """
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.scalers = {}
        self.encoders = {}
        self.feature_names = []
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Charger les données brutes
        
        Returns:
            (train_df, test_df)
        """
        print("📂 Chargement des données IEEE-CIS...")
        
        # Charger transactions
        train_transaction = pd.read_csv(
            f"{self.data_dir}/train_transaction.csv"
        )
        train_identity = pd.read_csv(
            f"{self.data_dir}/train_identity.csv"
        )
        
        # Merge
        train_df = train_transaction.merge(
            train_identity,
            on='TransactionID',
            how='left'
        )
        
        print(f"✓ Train: {train_df.shape}")
        print(f"  - Fraud: {train_df['isFraud'].sum()} ({train_df['isFraud'].mean():.2%})")
        print(f"  - Legitimate: {(~train_df['isFraud'].astype(bool)).sum()}")
        
        # Test data (si disponible)
        test_df = None
        if os.path.exists(f"{self.data_dir}/test_transaction.csv"):
            test_transaction = pd.read_csv(
                f"{self.data_dir}/test_transaction.csv"
            )
            test_identity = pd.read_csv(
                f"{self.data_dir}/test_identity.csv"
            )
            test_df = test_transaction.merge(
                test_identity,
                on='TransactionID',
                how='left'
            )
            print(f"✓ Test: {test_df.shape}")
        
        return train_df, test_df
    
    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Feature engineering avancé
        
        Args:
            df: DataFrame brut
        
        Returns:
            DataFrame avec features engineerées
        """
        print("🔧 Feature engineering...")
        
        df = df.copy()
        
        # 1. Features temporelles
        if 'TransactionDT' in df.columns:
            df['hour'] = (df['TransactionDT'] / 3600) % 24
            df['day'] = (df['TransactionDT'] / (3600 * 24)) % 7
        
        # 2. Features de montant
        if 'TransactionAmt' in df.columns:
            df['TransactionAmt_log'] = np.log1p(df['TransactionAmt'])
            df['TransactionAmt_decimal'] = df['TransactionAmt'] % 1
        
        # 3. Agrégations par carte
        if 'card1' in df.columns:
            card_stats = df.groupby('card1')['TransactionAmt'].agg([
                'mean', 'std', 'min', 'max', 'count'
            ]).add_prefix('card1_')
            df = df.merge(card_stats, on='card1', how='left')
        
        # 4. Features de domaine email
        if 'P_emaildomain' in df.columns:
            email_stats = df.groupby('P_emaildomain')['isFraud'].agg([
                'mean', 'count'
            ]).add_prefix('email_')
            df = df.merge(email_stats, on='P_emaildomain', how='left')
        
        # 5. Device info
        device_cols = [col for col in df.columns if 'device' in col.lower()]
        for col in device_cols:
            if df[col].dtype == 'object':
                df[f'{col}_freq'] = df.groupby(col)[col].transform('count')
        
        print(f"✓ Features: {df.shape[1]} colonnes")
        
        return df
    
    def preprocess(
        self,
        df: pd.DataFrame,
        is_train: bool = True
    ) -> pd.DataFrame:
        """
        Prétraitement complet
        
        Args:
            df: DataFrame à prétraiter
            is_train: Mode train (fit) ou test (transform)
        
        Returns:
            DataFrame prétraité
        """
        print("⚙️ Prétraitement...")
        
        df = df.copy()
        
        # Feature engineering
        df = self.feature_engineering(df)
        
        # Séparation features/target
        if 'isFraud' in df.columns:
            y = df['isFraud']
            X = df.drop(['isFraud', 'TransactionID'], axis=1, errors='ignore')
        else:
            y = None
            X = df.drop(['TransactionID'], axis=1, errors='ignore')
        
        # Identifier les types de colonnes
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        
        print(f"  - Numériques: {len(numeric_cols)}")
        print(f"  - Catégorielles: {len(categorical_cols)}")
        
        # Traiter les valeurs manquantes
        for col in numeric_cols:
            if is_train:
                self.scalers[f'{col}_fillna'] = X[col].median()
            X[col].fillna(self.scalers[f'{col}_fillna'], inplace=True)
        
        for col in categorical_cols:
            X[col].fillna('missing', inplace=True)
        
        # Encoder les variables catégorielles
        for col in categorical_cols:
            if is_train:
                self.encoders[col] = LabelEncoder()
                X[col] = self.encoders[col].fit_transform(X[col].astype(str))
            else:
                # Gérer les nouvelles catégories
                X[col] = X[col].astype(str).apply(
                    lambda x: x if x in self.encoders[col].classes_ else 'missing'
                )
                X[col] = self.encoders[col].transform(X[col])
        
        # Normalisation
        if is_train:
            self.scalers['standard'] = StandardScaler()
            X_scaled = self.scalers['standard'].fit_transform(X)
        else:
            X_scaled = self.scalers['standard'].transform(X)
        
        # Reconstruire DataFrame
        X_processed = pd.DataFrame(
            X_scaled,
            columns=X.columns,
            index=X.index
        )
        
        if y is not None:
            X_processed['isFraud'] = y.values
        
        self.feature_names = X.columns.tolist()
        
        print(f"✓ Shape finale: {X_processed.shape}")
        
        return X_processed
    
    def create_graph(
        self,
        df: pd.DataFrame,
        edge_strategy: str = "card_similarity"
    ) -> Data:
        """
        Créer un graphe PyG à partir des transactions
        
        Args:
            df: DataFrame prétraité
            edge_strategy: Stratégie de création d'arêtes
        
        Returns:
            PyG Data object
        """
        print("📊 Construction du graphe...")
        
        # Features des nœuds
        if 'isFraud' in df.columns:
            y = torch.tensor(df['isFraud'].values, dtype=torch.long)
            X = df.drop('isFraud', axis=1)
        else:
            y = None
            X = df
        
        x = torch.tensor(X.values, dtype=torch.float32)
        
        # Création des arêtes
        if edge_strategy == "card_similarity":
            edge_index = self._create_edges_card_similarity(df)
        elif edge_strategy == "temporal":
            edge_index = self._create_edges_temporal(df)
        elif edge_strategy == "knn":
            edge_index = self._create_edges_knn(X, k=10)
        else:
            raise ValueError(f"Unknown edge strategy: {edge_strategy}")
        
        # Créer le graphe
        data = Data(
            x=x,
            edge_index=edge_index,
            y=y
        )
        
        print(f"✓ Graphe créé:")
        print(f"  - Nœuds: {data.num_nodes}")
        print(f"  - Arêtes: {data.num_edges}")
        print(f"  - Features: {data.num_node_features}")
        
        return data
    
    def _create_edges_card_similarity(self, df: pd.DataFrame) -> torch.Tensor:
        """Créer arêtes basées sur les cartes similaires"""
        edges = []
        
        if 'card1' in df.columns:
            # Connecter les transactions avec la même carte
            for card in tqdm(df['card1'].unique(), desc="Cartes"):
                indices = df[df['card1'] == card].index.tolist()
                if len(indices) > 1:
                    # Connecter toutes les paires
                    for i in range(len(indices)):
                        for j in range(i + 1, len(indices)):
                            edges.append([indices[i], indices[j]])
                            edges.append([indices[j], indices[i]])  # Bidirectionnel
        
        if not edges:
            # Graphe vide si pas d'arêtes
            return torch.zeros((2, 0), dtype=torch.long)
        
        edge_index = torch.tensor(edges, dtype=torch.long).t()
        return edge_index
    
    def _create_edges_temporal(self, df: pd.DataFrame) -> torch.Tensor:
        """Créer arêtes basées sur la proximité temporelle"""
        edges = []
        
        if 'TransactionDT' in df.columns:
            df_sorted = df.sort_values('TransactionDT').reset_index(drop=True)
            
            # Connecter transactions consécutives (fenêtre de temps)
            window_size = 100  # transactions
            
            for i in tqdm(range(len(df_sorted)), desc="Temporal edges"):
                for j in range(i + 1, min(i + window_size, len(df_sorted))):
                    edges.append([i, j])
                    edges.append([j, i])
        
        if not edges:
            return torch.zeros((2, 0), dtype=torch.long)
        
        edge_index = torch.tensor(edges, dtype=torch.long).t()
        return edge_index
    
    def _create_edges_knn(self, X: pd.DataFrame, k: int = 10) -> torch.Tensor:
        """Créer arêtes avec k plus proches voisins"""
        from sklearn.neighbors import kneighbors_graph
        
        print(f"  Computing {k}-NN...")
        
        # KNN graph
        knn_graph = kneighbors_graph(
            X.values,
            n_neighbors=k,
            mode='connectivity',
            include_self=False
        )
        
        # Convertir en edge_index
        edge_index = torch.tensor(
            np.array(knn_graph.nonzero()),
            dtype=torch.long
        )
        
        return edge_index
    
    def save_preprocessors(self, save_path: str):
        """Sauvegarder les preprocessors"""
        with open(f"{save_path}/scalers.pkl", 'wb') as f:
            pickle.dump(self.scalers, f)
        
        with open(f"{save_path}/encoders.pkl", 'wb') as f:
            pickle.dump(self.encoders, f)
        
        with open(f"{save_path}/feature_names.pkl", 'wb') as f:
            pickle.dump(self.feature_names, f)
        
        print(f"✓ Preprocessors sauvegardés dans {save_path}")
    
    def load_preprocessors(self, load_path: str):
        """Charger les preprocessors"""
        with open(f"{load_path}/scalers.pkl", 'rb') as f:
            self.scalers = pickle.load(f)
        
        with open(f"{load_path}/encoders.pkl", 'rb') as f:
            self.encoders = pickle.load(f)
        
        with open(f"{load_path}/feature_names.pkl", 'rb') as f:
            self.feature_names = pickle.load(f)
        
        print(f"✓ Preprocessors chargés depuis {load_path}")


def prepare_ieee_dataset(
    data_dir: str,
    output_dir: str,
    test_size: float = 0.15,
    val_size: float = 0.15
) -> Dict[str, Data]:
    """
    Pipeline complet de préparation du dataset
    
    Args:
        data_dir: Répertoire des données brutes
        output_dir: Répertoire de sortie
        test_size: Proportion test
        val_size: Proportion validation
    
    Returns:
        Dict contenant train, val, test Data objects
    """
    print("=" * 50)
    print("PRÉPARATION DATASET IEEE-CIS FRAUD DETECTION")
    print("=" * 50)
    
    # Créer le preprocessor
    preprocessor = IEEEFraudPreprocessor(data_dir)
    
    # Charger les données
    train_df, test_df = preprocessor.load_data()
    
    # Prétraiter
    train_processed = preprocessor.preprocess(train_df, is_train=True)
    
    # Split train/val/test
    train_val_df, test_internal_df = train_test_split(
        train_processed,
        test_size=test_size,
        random_state=42,
        stratify=train_processed['isFraud']
    )
    
    train_df_final, val_df = train_test_split(
        train_val_df,
        test_size=val_size / (1 - test_size),
        random_state=42,
        stratify=train_val_df['isFraud']
    )
    
    print(f"\n📊 Splits:")
    print(f"  - Train: {len(train_df_final)}")
    print(f"  - Val: {len(val_df)}")
    print(f"  - Test: {len(test_internal_df)}")
    
    # Créer les graphes
    train_data = preprocessor.create_graph(train_df_final)
    val_data = preprocessor.create_graph(val_df)
    test_data = preprocessor.create_graph(test_internal_df)
    
    # Sauvegarder
    os.makedirs(output_dir, exist_ok=True)
    
    torch.save(train_data, f"{output_dir}/train_data.pt")
    torch.save(val_data, f"{output_dir}/val_data.pt")
    torch.save(test_data, f"{output_dir}/test_data.pt")
    
    preprocessor.save_preprocessors(output_dir)
    
    print(f"\n✅ Dataset préparé et sauvegardé dans {output_dir}")
    
    return {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }


if __name__ == "__main__":
    # Test
    data_dir = "data/raw/ieee-fraud"
    output_dir = "data/processed"
    
    dataset = prepare_ieee_dataset(data_dir, output_dir)
    
    print("\n📊 Statistiques finales:")
    for split, data in dataset.items():
        print(f"\n{split.upper()}:")
        print(f"  Nodes: {data.num_nodes}")
        print(f"  Edges: {data.num_edges}")
        print(f"  Features: {data.num_node_features}")
        if data.y is not None:
            fraud_rate = data.y.float().mean().item()
            print(f"  Fraud rate: {fraud_rate:.2%}")

"""
Préparation du dataset IEEE-CIS avec sampling 25%
Construction du graphe GNN optimisé
"""

import pandas as pd
import torch
import numpy as np
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import os


def prepare_ieee_dataset(data_dir, output_dir, test_size=0.15, val_size=0.15):
    """
    Préparer le dataset IEEE-CIS avec construction de graphe
    
    Args:
        data_dir: Chemin vers les fichiers CSV (train_transaction.csv, train_identity.csv)
        output_dir: Chemin de sortie pour sauvegarder
        test_size: Proportion test (0.15 = 15%)
        val_size: Proportion validation (0.15 = 15%)
    
    Returns:
        dict: {'train': Data, 'val': Data, 'test': Data}
    """
    
    print('\n' + '='*60)
    print('📊 PRÉPARATION DATASET IEEE-CIS')
    print('='*60)
    
    # 1. Charger les données
    print('\n📥 Chargement des données...')
    
    trans_path = os.path.join(data_dir, 'train_transaction.csv')
    ident_path = os.path.join(data_dir, 'train_identity.csv')
    
    if not os.path.exists(trans_path):
        raise FileNotFoundError(f"Fichier introuvable: {trans_path}")
    
    train_trans = pd.read_csv(trans_path)
    
    # Identity est optionnel
    if os.path.exists(ident_path):
        train_ident = pd.read_csv(ident_path)
        df = train_trans.merge(train_ident, on='TransactionID', how='left')
        print(f'   ✅ Transactions: {len(train_trans):,}')
        print(f'   ✅ Identités: {len(train_ident):,}')
    else:
        df = train_trans
        print(f'   ✅ Transactions: {len(train_trans):,}')
        print(f'   ⚠️  train_identity.csv non trouvé (optionnel)')
    
    # 2. Sélectionner features
    print('\n🔧 Préparation des features...')
    
    # Colonnes à exclure
    exclude_cols = ['TransactionID', 'TransactionDT', 'isFraud']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Séparer features numériques et catégorielles
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    
    print(f'   ✅ {len(numeric_cols)} features numériques sélectionnées')
    
    # 3. Préparer X et y
    X = df[numeric_cols].fillna(0).values.astype(np.float32)
    y = df['isFraud'].values.astype(np.int64)
    
    # Normaliser
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    print(f'   ✅ Shape finale: {X.shape}')
    print(f'   ✅ Fraudes: {y.sum():,} ({y.mean()*100:.2f}%)')
    
    # 4. Split train/val/test
    print('\n✂️  Split des données...')
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, 
        test_size=(test_size + val_size), 
        random_state=42, 
        stratify=y
    )
    
    val_size_adjusted = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, 
        test_size=(test_size / (test_size + val_size)), 
        random_state=42, 
        stratify=y_temp
    )
    
    print(f'   ✅ Train: {len(X_train):,} samples ({y_train.sum():,} fraudes)')
    print(f'   ✅ Val: {len(X_val):,} samples ({y_val.sum():,} fraudes)')
    print(f'   ✅ Test: {len(X_test):,} samples ({y_test.sum():,} fraudes)')
    
    # 5. Construire les graphes
    print('\n🔗 Construction des graphes GNN...')
    print('   (Cela peut prendre quelques minutes)')
    
    def build_graph(X, y, k_neighbors=10):
        """
        Construire un graphe PyG
        Chaque transaction = node
        Edges = K plus proches voisins
        """
        num_nodes = len(X)
        
        # Limiter k si nécessaire
        k = min(k_neighbors, num_nodes - 1)
        
        print(f'      Building graph with {num_nodes} nodes, k={k}...')
        
        # KNN pour créer edges
        knn = NearestNeighbors(n_neighbors=k+1, metric='cosine', n_jobs=-1)
        knn.fit(X)
        
        # Trouver voisins pour chaque node
        distances, indices = knn.kneighbors(X)
        
        # Créer edge_index
        edges = []
        for i in range(num_nodes):
            neighbors = indices[i][1:]  # Exclure le node lui-même
            for j in neighbors:
                edges.append([i, j])
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        # Créer Data object
        data = Data(
            x=torch.tensor(X, dtype=torch.float),
            edge_index=edge_index,
            y=torch.tensor(y, dtype=torch.long),
            num_nodes=num_nodes
        )
        
        print(f'      ✅ {data.num_nodes} nodes, {data.num_edges} edges')
        
        return data
    
    # Construire les 3 graphes
    train_data = build_graph(X_train, y_train, k_neighbors=10)
    val_data = build_graph(X_val, y_val, k_neighbors=10)
    test_data = build_graph(X_test, y_test, k_neighbors=10)
    
    dataset = {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }
    
    # 6. Sauvegarder
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, 'ieee_dataset.pt')
        torch.save(dataset, save_path)
        print(f'\n💾 Dataset sauvegardé: {save_path}')
    
    print('\n' + '='*60)
    print('✅ PRÉPARATION TERMINÉE')
    print('='*60)
    
    return dataset


if __name__ == '__main__':
    # Test
    dataset = prepare_ieee_dataset(
        data_dir='/kaggle/input/ieee-fraud-detection',
        output_dir='/kaggle/working/data',
        test_size=0.15,
        val_size=0.15
    )
    
    print('\n📊 Statistiques finales:')
    for split, data in dataset.items():
        fraud_rate = (data.y == 1).sum().item() / len(data.y) * 100
        print(f'   {split}: {data.num_nodes} nodes, {data.num_edges} edges, {fraud_rate:.2f}% fraudes')

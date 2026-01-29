"""
Cellule Kaggle: Créer le Modèle Hybride Complet
GNN + DistilGPT2 avec LoRA pour la détection de fraude
"""

import sys
import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from torch_geometric.data import Data, Batch

# Configuration des chemins
sys.path.insert(0, '/kaggle/working/fraud_detection_project/fraud_detection')
os.chdir('/kaggle/working/fraud_detection_project/fraud_detection')

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️  Device: {device}")

# ============================================
# CRÉATION DU MODÈLE HYBRIDE COMPLET
# ============================================

print("\n" + "="*60)
print("🧠 CRÉATION DU MODÈLE HYBRIDE GNN + DistilGPT2")
print("="*60)

# 1. GNN (Graph Neural Network)
print("\n1️⃣ Création du GNN...")
from src.models.gnn_model import GNNModel

# Configuration complète du GNN
gnn_config = {
    'input_dim': 64,          # Dimension des features d'entrée
    'hidden_channels': 128,
    'num_layers': 2,
    'dropout': 0.3,
    'heads': 4
}

# Créer le GNN et s'assurer qu'il est sur le bon device
gnn = GNNModel(gnn_config)
gnn = gnn.to(device)

# Force tous les paramètres sur le device
for param in gnn.parameters():
    param.data = param.data.to(device)
    
print(f"   ✅ GNN créé: {sum(p.numel() for p in gnn.parameters()):,} paramètres")
print(f"   📍 Device: {next(gnn.parameters()).device}")

# 2. LLM (DistilGPT2 avec LoRA)
print("\n2️⃣ Chargement de DistilGPT2...")
tokenizer = AutoTokenizer.from_pretrained('distilgpt2', padding_side='left')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

llm = AutoModelForCausalLM.from_pretrained(
    'distilgpt2',
    torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
    device_map='auto' if device == 'cuda' else None
)

# Application de LoRA
print("   🔧 Application de LoRA...")
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["c_attn", "c_proj"],
    bias="none"
)

llm = get_peft_model(llm, lora_config)
llm.print_trainable_parameters()
print("   ✅ LLM créé avec LoRA")

# 3. Couche de fusion (GNN → LLM)
print("\n3️⃣ Création de la couche de fusion...")
gnn_dim = gnn_config['hidden_channels']
llm_dim = llm.config.n_embd

fusion_layer = nn.Sequential(
    nn.Linear(gnn_dim, llm_dim),
    nn.LayerNorm(llm_dim),
    nn.Dropout(0.1)
)

# S'assurer que fusion_layer est sur le bon device
fusion_layer = fusion_layer.to(device)
for param in fusion_layer.parameters():
    param.data = param.data.to(device)

print(f"   ✅ Fusion: {gnn_dim} → {llm_dim}")
print(f"   📍 Device: {next(fusion_layer.parameters()).device}")

# 4. Tête de classification
print("\n4️⃣ Création de la tête de classification...")
fraud_head = nn.Sequential(
    nn.Linear(llm_dim, llm_dim // 2),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(llm_dim // 2, 1)
)

# S'assurer que fraud_head est sur le bon device
fraud_head = fraud_head.to(device)
for param in fraud_head.parameters():
    param.data = param.data.to(device)

print(f"   ✅ Classification: {llm_dim} → {llm_dim//2} → 1")
print(f"   📍 Device: {next(fraud_head.parameters()).device}")

# 5. Classe Wrapper pour le modèle complet
print("\n5️⃣ Assemblage du modèle hybride...")

class HybridModel(nn.Module):
    def __init__(self, gnn, llm, fusion, classifier):
        super().__init__()
        self.gnn = gnn
        self.llm = llm
        self.fusion = fusion
        self.classifier = classifier
    
    def forward(self, graph_data, mode='train'):
        # Assurer que toutes les données sont sur le bon device
        device = next(self.gnn.parameters()).device
        
        x = graph_data.x.to(device)
        edge_index = graph_data.edge_index.to(device)
        batch = graph_data.batch.to(device) if hasattr(graph_data, 'batch') else None
        
        # Embeddings GNN
        gnn_embeddings = self.gnn.get_embeddings(x, edge_index, batch)
        
        # Vérifier que les embeddings sont sur le bon device
        gnn_embeddings = gnn_embeddings.to(device)
        
        # Fusion avec LLM
        fused_embeddings = self.fusion(gnn_embeddings)
        
        # Classification
        logits = self.classifier(fused_embeddings)
        
        return logits

# Créer le modèle hybride
model = HybridModel(gnn, llm, fusion_layer, fraud_head)
print("   ✅ Modèle hybride assemblé")

# Vérification finale des devices
print("\n🔍 Vérification des devices:")
print(f"   GNN:     {next(model.gnn.parameters()).device}")
print(f"   Fusion:  {next(model.fusion.parameters()).device}")
print(f"   Classif: {next(model.classifier.parameters()).device}")

# S'assurer que tout le modèle est sur le bon device
model = model.to(device)

# ============================================
# STATISTIQUES DU MODÈLE
# ============================================

print("\n" + "="*60)
print("📊 STATISTIQUES DU MODÈLE HYBRIDE")
print("="*60)

gnn_params = sum(p.numel() for p in gnn.parameters())
llm_params = sum(p.numel() for p in llm.parameters())
fusion_params = sum(p.numel() for p in fusion_layer.parameters())
head_params = sum(p.numel() for p in fraud_head.parameters())
total_params = gnn_params + llm_params + fusion_params + head_params

llm_trainable = sum(p.numel() for p in llm.parameters() if p.requires_grad)
total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\n📦 Paramètres par composant:")
print(f"   GNN:            {gnn_params:>15,}")
print(f"   LLM:            {llm_params:>15,}")
print(f"   Fusion:         {fusion_params:>15,}")
print(f"   Classification: {head_params:>15,}")
print(f"   {'─'*35}")
print(f"   TOTAL:          {total_params:>15,}")

print(f"\n🎯 Paramètres entraînables:")
print(f"   LLM (LoRA):     {llm_trainable:>15,}")
print(f"   Total:          {total_trainable:>15,}")
print(f"   Ratio:          {100 * total_trainable / total_params:>14.2f}%")

print("\n" + "="*60)
print("✅ MODÈLE HYBRIDE PRÊT POUR L'ENTRAÎNEMENT!")
print("="*60)

# ============================================
# TEST AVEC DONNÉES SYNTHÉTIQUES
# ============================================

print("\n🧪 Test rapide du modèle avec des données synthétiques...")

# Créer 5 graphes de test
test_graphs = []
num_test_graphs = 5
num_nodes_per_graph = 10

for i in range(num_test_graphs):
    # Générer des features aléatoires sur le bon device
    x = torch.randn(num_nodes_per_graph, gnn_config['input_dim'], device=device)
    
    # Créer des edges aléatoires (graphe connecté)
    edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 0],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 9]
    ], dtype=torch.long, device=device)
    
    graph = Data(x=x, edge_index=edge_index)
    test_graphs.append(graph)

# Créer un batch
sample_data = Batch.from_data_list(test_graphs)
print(f"   Batch créé: {sample_data.num_graphs} graphes, {sample_data.num_nodes} nœuds")
print(f"   Features shape: {sample_data.x.shape}")
print(f"   Device des données: {sample_data.x.device}")

# Test du modèle
model.eval()
with torch.no_grad():
    output = model(sample_data)
    predictions = torch.sigmoid(output).squeeze().cpu().numpy()
    
    print("✅ Test réussi!")
    print(f"   Output shape: {output.shape}")
    print(f"\n📊 Prédictions sur {num_test_graphs} transactions:")
    
    # Gérer le cas où predictions est un scalaire ou un vecteur
    if predictions.ndim == 0:
        predictions = [predictions.item()]
    
    for i, pred in enumerate(predictions):
        label = "🚨 FRAUDE" if pred > 0.5 else "✅ LÉGIT"
        confidence = pred if pred > 0.5 else (1 - pred)
        bar = "█" * int(confidence * 20)
        print(f"   Transaction {i+1}: {pred:.4f} {label} [{bar:<20}] {confidence:.1%}")

print("\n🎉 Le modèle hybride est complètement opérationnel!")

# ============================================
# SAUVEGARDE INITIALE DU MODÈLE
# ============================================

print("\n💾 Sauvegarde du modèle initial...")
save_dir = "/kaggle/working/initial_model"
os.makedirs(save_dir, exist_ok=True)

save_path = f"{save_dir}/hybrid_model_initial.pt"
torch.save({
    'model_state_dict': model.state_dict(),
    'gnn_config': gnn_config,
    'lora_config': {
        'r': lora_config.r,
        'lora_alpha': lora_config.lora_alpha,
        'lora_dropout': lora_config.lora_dropout,
        'target_modules': lora_config.target_modules
    },
    'total_params': total_params,
    'trainable_params': total_trainable
}, save_path)

print(f"   ✅ Modèle initial sauvegardé: {save_path}")

print("\n" + "="*60)
print("🎯 PRÊT POUR L'ENTRAÎNEMENT!")
print("="*60)
print("\n💡 Variables créées et disponibles:")
print("   - model: Le modèle hybride complet")
print("   - gnn: Le Graph Neural Network")
print("   - llm: Le DistilGPT2 avec LoRA")
print("   - fusion_layer: La couche de fusion")
print("   - fraud_head: La tête de classification")
print("   - gnn_config: Configuration du GNN")
print("\n➡️  Vous pouvez maintenant passer à la cellule d'entraînement!")

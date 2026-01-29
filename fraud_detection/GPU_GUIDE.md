# 🎮 Guide d'Utilisation GPU

Ce document explique comment utiliser le GPU efficacement pour l'entraînement.

## 📊 Configurations Optimales

### CPU (Local, Basique)
```yaml
dataset:
  batch_size: 32
  num_workers: 2

gnn:
  hidden_channels: 128
  num_layers: 2
```

**Temps estimé:** 4-6 heures

---

### GPU (NVIDIA GTX/RTX, ~8GB VRAM)
```yaml
dataset:
  batch_size: 128
  num_workers: 4

gnn:
  hidden_channels: 256
  num_layers: 3
```

**Temps estimé:** 45-90 minutes

---

### GPU P100 (Kaggle, 16GB VRAM)
```yaml
dataset:
  batch_size: 256
  num_workers: 0  # Important sur Kaggle !

gnn:
  hidden_channels: 512
  num_layers: 3
```

**Temps estimé:** 30-60 minutes

---

### GPU A100 (Colab Pro, 40GB VRAM)
```yaml
dataset:
  batch_size: 512
  num_workers: 4

gnn:
  hidden_channels: 1024
  num_layers: 4
```

**Temps estimé:** 20-40 minutes

---

## 🔧 Détection Automatique du Device

Le code détecte automatiquement le meilleur device :

```python
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

## 🚀 Optimisations GPU

### 1. Mixed Precision Training (FP16)

Pour GPUs avec Tensor Cores (RTX, V100, A100, P100) :

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in train_loader:
    optimizer.zero_grad()
    
    with autocast():  # Mixed precision
        output = model(batch)
        loss = criterion(output, batch.y)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**Gain:** 2-3x plus rapide + moins de mémoire

### 2. Gradient Accumulation

Si batch_size trop gros pour la mémoire :

```python
accumulation_steps = 4  # Effectif batch = batch_size * 4

for i, batch in enumerate(train_loader):
    output = model(batch)
    loss = criterion(output, batch.y) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 3. Pin Memory

Pour transferts CPU→GPU plus rapides :

```python
train_loader = DataLoader(
    dataset,
    batch_size=128,
    pin_memory=True,  # ← Important !
    num_workers=4
)
```

### 4. Checkpoint Gradient

Pour très gros modèles :

```python
from torch.utils.checkpoint import checkpoint

class MyModel(nn.Module):
    def forward(self, x):
        # Utilise checkpoint pour économiser mémoire
        x = checkpoint(self.layer1, x)
        x = checkpoint(self.layer2, x)
        return x
```

## 📈 Monitoring GPU

### Pendant l'entraînement

```python
import torch

def print_gpu_stats():
    if torch.cuda.is_available():
        print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"GPU Memory Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        print(f"Max Memory Allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

# Appeler régulièrement
print_gpu_stats()
```

### Avec nvidia-smi

```bash
# Terminal séparé
watch -n 1 nvidia-smi

# Ou dans le code
import subprocess
subprocess.run(['nvidia-smi'])
```

### Avec GPUtil

```python
import GPUtil

GPUs = GPUtil.getGPUs()
for gpu in GPUs:
    print(f"GPU {gpu.id}: {gpu.name}")
    print(f"  Load: {gpu.load * 100:.1f}%")
    print(f"  Memory: {gpu.memoryUsed}/{gpu.memoryTotal} MB")
    print(f"  Temp: {gpu.temperature}°C")
```

## ⚠️ Problèmes Courants

### CUDA Out of Memory

**Solutions:**
1. Réduire `batch_size`
2. Réduire `hidden_channels`
3. Activer gradient accumulation
4. Utiliser mixed precision (FP16)
5. Vider le cache: `torch.cuda.empty_cache()`

```python
# Nettoyer la mémoire
import gc
import torch

gc.collect()
torch.cuda.empty_cache()
```

### Slow Training

**Vérifications:**
1. GPU utilisé ? `print(next(model.parameters()).device)`
2. Données sur GPU ? `print(batch.x.device)`
3. num_workers optimal ? (2-4 pour GPU)
4. pin_memory activé ?

```python
# S'assurer que tout est sur GPU
model = model.to('cuda')
batch = batch.to('cuda')
```

### Multiple GPUs

Pour utiliser plusieurs GPUs :

```python
import torch.nn as nn

# DataParallel (simple)
model = nn.DataParallel(model)

# Ou DistributedDataParallel (meilleur)
from torch.nn.parallel import DistributedDataParallel as DDP
model = DDP(model, device_ids=[0, 1])
```

## 🎯 Configuration Recommandée par Plateforme

### Kaggle Kernels (GPU P100, 16GB)

```yaml
# configs/config_kaggle.yaml
dataset:
  batch_size: 256
  num_workers: 0  # Important !
  pin_memory: true

gnn:
  hidden_channels: 512
  num_layers: 3
  
fine_tuning:
  gradient_accumulation_steps: 1
  mixed_precision: true
```

### Google Colab (GPU T4, 16GB)

```yaml
# configs/config_colab.yaml
dataset:
  batch_size: 128
  num_workers: 2
  pin_memory: true

gnn:
  hidden_channels: 384
  num_layers: 3
  
fine_tuning:
  gradient_accumulation_steps: 2
  mixed_precision: true
```

### Local (RTX 3090, 24GB)

```yaml
# configs/config_local.yaml
dataset:
  batch_size: 256
  num_workers: 4
  pin_memory: true

gnn:
  hidden_channels: 768
  num_layers: 4
  
fine_tuning:
  gradient_accumulation_steps: 1
  mixed_precision: true
```

## 📊 Benchmarks

Performance sur IEEE-CIS (590k transactions) :

| Platform | GPU | VRAM | Batch | Time | F1-Score |
|----------|-----|------|-------|------|----------|
| CPU | - | - | 32 | 4h 30m | 0.89 |
| Colab | T4 | 16GB | 128 | 1h 15m | 0.91 |
| Kaggle | P100 | 16GB | 256 | 45m | 0.92 |
| Local | RTX 3090 | 24GB | 256 | 35m | 0.93 |
| Colab Pro | A100 | 40GB | 512 | 22m | 0.94 |

*Métriques approximatives, peuvent varier.*

## 🔗 Ressources

- [PyTorch GPU Documentation](https://pytorch.org/docs/stable/notes/cuda.html)
- [Mixed Precision Training](https://pytorch.org/docs/stable/amp.html)
- [PyTorch Performance Tuning](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [NVIDIA CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)

---

**Utilisez le GPU efficacement et entraînez plus rapidement ! 🚀**

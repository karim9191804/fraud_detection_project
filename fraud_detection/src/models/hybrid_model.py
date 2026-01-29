"""
Hybrid Model Builder
GNN + Transformer (LLM)
Importable depuis Kaggle / GitHub
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from torch_geometric.nn import GCNConv, global_mean_pool


def build_hybrid_model(device=None):

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    gnn_config = {
        "input_dim": 64,
        "hidden_dim": 128,
        "output_dim": 128
    }

    class GNNEncoder(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.conv1 = GCNConv(config["input_dim"], config["hidden_dim"])
            self.conv2 = GCNConv(config["hidden_dim"], config["output_dim"])

        def forward(self, x, edge_index, batch):
            x = self.conv1(x, edge_index).relu()
            x = self.conv2(x, edge_index)
            return global_mean_pool(x, batch)

    gnn = GNNEncoder(gnn_config)

    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    llm = AutoModelForCausalLM.from_pretrained(model_name)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1
    )

    llm = get_peft_model(llm, lora_config)

    class HybridModel(nn.Module):
        def __init__(self, gnn, llm):
            super().__init__()
            self.gnn = gnn
            self.llm = llm
            self.proj = nn.Linear(
                gnn_config["output_dim"],
                llm.config.n_embd
            )

        def forward(self, graph_data):
            gnn_out = self.gnn(
                graph_data.x,
                graph_data.edge_index,
                graph_data.batch
            )
            return self.proj(gnn_out)

    model = HybridModel(gnn, llm).to(device)

    return model, gnn_config

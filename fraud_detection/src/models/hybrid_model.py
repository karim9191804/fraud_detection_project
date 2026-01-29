"""
Hybrid Model Builder
GNN + Transformer (LLM)
Importable depuis Kaggle / GitHub
"""

def build_hybrid_model(device=None):
    import torch
    import torch.nn as nn
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType
    from torch_geometric.nn import GCNConv, global_mean_pool

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"🖥️ Device: {device}")

    gnn_config = {"input_dim": 64, "hidden_dim": 128, "output_dim": 128, "num_layers": 2}

    class GNNEncoder(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.conv1 = GCNConv(config["input_dim"], config["hidden_dim"])
            self.conv2 = GCNConv(config["hidden_dim"], config["output_dim"])

        def forward(self, x, edge_index, batch):
            x = self.conv1(x, edge_index).relu()
            x = self.conv2(x, edge_index)
            x = global_mean_pool(x, batch)
            return x

    gnn_encoder = GNNEncoder(gnn_config)

    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    llm = AutoModelForCausalLM.from_pretrained(model_name)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.05
    )
    llm = get_peft_model(llm, lora_config)

    class HybridModel(nn.Module):
        def __init__(self, gnn, llm):
            super().__init__()
            self.gnn = gnn
            self.llm = llm
            self.projection = nn.Linear(gnn_config["output_dim"], llm.config.n_embd)

        def forward(self, graph_data, input_ids, attention_mask=None):
            gnn_out = self.gnn(graph_data.x, graph_data.edge_index, graph_data.batch)
            gnn_emb = self.projection(gnn_out).unsqueeze(1)
            outputs = self.llm(input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=None)
            return outputs, gnn_emb

    model = HybridModel(gnn_encoder, llm).to(device)
    print("✅ MODÈLE HYBRIDE PRÊT POUR L'ENTRAÎNEMENT")
    return model, gnn_config

if __name__ == "__main__":
    build_hybrid_model()

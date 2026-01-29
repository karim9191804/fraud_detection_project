import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool

class GNNModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv1 = GCNConv(config["input_dim"], config["hidden_channels"])
        self.conv2 = GCNConv(config["hidden_channels"], config["hidden_channels"])
        self.dropout = nn.Dropout(config.get("dropout", 0.3))

    def get_embeddings(self, x, edge_index, batch=None):
        x = self.conv1(x, edge_index).relu()
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        if batch is not None:
            x = global_mean_pool(x, batch)
        return x

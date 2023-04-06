import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv


# IPU paradigm demands that models return the loss as second output
class LossAppender(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, y):
        if y is not None:
            return x, F.mse_loss(x, y)
        return x

    
class MLP(LossAppender):
    def __init__(self, window_size_hrs, num_hidden, hidden_size, residual):
        super().__init__()
        self.residual = residual
        self.layers = nn.ModuleList([nn.Linear(window_size_hrs, hidden_size) if i == 0 else
                                     nn.Linear(hidden_size, 1) if i == num_hidden else
                                     nn.Linear(hidden_size, hidden_size)
                                     for i in range(0, num_hidden + 1)])

    def forward(self, x, edge_index, y=None):
        x = F.relu(self.layers[0](x))
        for layer in self.layers[1:-1]:
            x = F.relu(layer(x)) + (x if self.residual else 0)
        x = self.layers[-1](x)

        return super().forward(x, y)
    

class WeightedGCN(LossAppender):
    def __init__(self, window_size_hrs, num_convs, hidden_size, residual, edge_weights):
        super().__init__()
        self.dense_in = nn.Linear(window_size_hrs, hidden_size)
        self.convs = nn.ModuleList([GCNConv(hidden_size, hidden_size, add_self_loops=False) for _ in range(num_convs)])
        self.dense_out = nn.Linear(hidden_size, 1)
        self.edge_weights = edge_weights
        self.residual = residual

    def forward(self, x, edge_index, y=None):
        num_graphs = edge_index.size(1) // len(self.edge_weights)
        edge_weights = self.edge_weights.clamp(min=1e-10).repeat(num_graphs).to(x.device)

        x = F.relu(self.dense_in(x))
        for conv in self.convs:
            h = conv(x, edge_index, edge_weights)
            x = F.relu(h) + (x if self.residual else 0)
        x = self.dense_out(x)

        return super().forward(x, y)

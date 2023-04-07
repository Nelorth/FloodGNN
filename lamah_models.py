from torch.nn import Module, ModuleList
from torch.nn.functional import relu
from torch_geometric.nn import GCNConv, Linear


# IPU paradigm demands that models return the loss as second output
def append_MSE(x, y):
    if y is not None:
        return x, F.mse_loss(x, y)
    return x


class MLP(Module):
    def __init__(self, window_size_hrs, num_hidden, hidden_size, residual):
        super().__init__()
        self.residual = residual
        self.layers = ModuleList([
            Linear(window_size_hrs, hidden_size, weight_initializer="glorot") if i == 0 else
            Linear(hidden_size, 1, weight_initializer="glorot") if i == num_hidden else
            Linear(hidden_size, hidden_size, weight_initializer="glorot")
            for i in range(0, num_hidden + 1)])

    def forward(self, x, edge_index, y=None):
        x = relu(self.layers[0](x))
        for layer in self.layers[1:-1]:
            x = relu(layer(x)) + (x if self.residual else 0)
        x = self.layers[-1](x)

        return append_mse(x, y)


class WeightedGCN(Module):
    def __init__(self, window_size_hrs, num_convs, hidden_size, residual, edge_weights):
        super().__init__()
        self.dense_in = Linear(window_size_hrs, hidden_size, weight_initializer="glorot")
        self.convs = ModuleList([GCNConv(hidden_size, hidden_size, add_self_loops=False) for _ in range(num_convs)])
        self.dense_out = Linear(hidden_size, 1, weight_initializer="glorot")
        self.edge_weights = edge_weights
        self.residual = residual

    def forward(self, x, edge_index, y=None, track_evolution=False):
        num_graphs = edge_index.size(1) // len(self.edge_weights)
        edge_weights = self.edge_weights.clamp(min=1e-10).repeat(num_graphs).to(x.device)

        x = self.dense_in(x)
        if track_evolution:
            self.evolution = [x.detach()]
        for conv in self.convs:
            x = relu(conv(x, edge_index, edge_weights)) + (x if self.residual else 0)
            if track_evolution:
                self.evolution.append(x.detach())
        x = self.dense_out(x)

        return append_mse(x, y)

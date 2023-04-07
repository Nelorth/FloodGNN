from torch.nn import Module, ModuleList
from torch.nn import ReLU, Sequential
from torch.nn.functional import mse_loss, relu
from torch_geometric.nn import GCNConv, Linear
from torch_geometric.nn import MLP
from torch_geometric.typing import OptTensor

from graff_conv import GRAFFConv


# IPU paradigm demands that models return the loss as second output
def append_mse(x, y):
    if y is not None:
        return x, mse_loss(x, y)
    return x


# functionality: encoder/decoder, evolution tracking
class BaseModel(Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.encoder = Sequential(Linear(in_channels, hidden_channels, weight_initializer="glorot"), ReLU(inplace=True))
        self.decoder = Linear(hidden_channels, 1, weight_initializer="glorot")
        self.evolution = None  # for tracking hidden layer activations

    def start_evolution(self, evo_tracking):
        self.evolution = None
        self.evo_tracking = evo_tracking

    def update_evolution(self, x):
        if self.evo_tracking:
            if self.evolution is None:
                self.evolution = [x]
            else:
                self.evolution.append(x)


class FloodMLP(BaseModel):
    def __init__(self, in_channels, hidden_channels, num_hidden, residual):
        super().__init__(in_channels, hidden_channels)
        self.residual = residual
        self.dense_layers = ModuleList([Linear(hidden_channels, hidden_channels, weight_initializer="glorot")
                                        for _ in range(num_hidden)])

    def forward(self, x, edge_index, y=None, evo_tracking=False):
        super().start_evolution(evo_tracking)

        x = self.encoder(x)
        self.update_evolution(x)
        for layer in self.dense_layers:
            x = relu(layer(x)) + (x if self.residual else 0)
            self.update_evolution(x)
        x = self.decoder(x)

        return append_mse(x, y)


class FloodGCN(BaseModel):
    def __init__(self, in_channels, hidden_channels, num_hidden, residual, edge_weights):
        super().__init__(in_channels, hidden_channels)
        self.residual = residual
        self.edge_weights = edge_weights
        self.gcn_layers = ModuleList([GCNConv(hidden_channels, hidden_channels, add_self_loops=False)
                                      for _ in range(num_hidden)])

    def forward(self, x, edge_index, y=None, evo_tracking=False):
        super().start_evolution(evo_tracking)
        num_graphs = edge_index.size(1) // len(self.edge_weights)
        edge_weights = self.edge_weights.clamp(min=1e-10).repeat(num_graphs).to(x.device)

        x = self.encoder(x)
        self.update_evolution(x)
        for conv in self.gcn_layers:
            x = relu(conv(x, edge_index, edge_weights)) + (x if self.residual else 0)
            self.update_evolution(x)
        x = self.decoder(x)

        return append_mse(x, y)


class FloodGRAFFNN(BaseModel):
    def __init__(self, in_channels, hidden_channels, num_hidden, shared_weights, step_size, edge_weights):
        super().__init__(in_channels, hidden_channels)
        self.edge_weights = edge_weights
        if shared_weights:
            self.graff_convs = ModuleList(num_hidden * [GRAFFConv(channels=hidden_channels, step_size=step_size)])
        else:
            self.graff_convs = ModuleList([GRAFFConv(channels=hidden_channels, step_size=step_size)
                                           for _ in range(num_hidden)])

    def forward(self, x, edge_index, y=None, evo_tracking=False):
        super().start_evolution(evo_tracking)
        num_graphs = edge_index.size(1) // len(self.edge_weights)
        edge_weights = self.edge_weights.clamp(min=1e-10).repeat(num_graphs).to(x.device)

        x_0 = self.encoder(x)
        x = x_0
        self.update_evolution(x)
        for graff_conv in self.graff_convs:
            x = graff_conv(x, x_0, edge_index, edge_weights)
            self.update_evolution(x)
        x = self.decoder(x)

        return append_mse(x, y)

from abc import ABC, abstractmethod
from graff_conv import GRAFFConv
from torch.nn import Module, ModuleList
from torch.nn.functional import mse_loss, relu
from torch_geometric.nn import GCNConv, GCN2Conv, Linear


# functionality: encoder/decoder, evolution tracking, IPU loss return
class BaseModel(Module, ABC):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.encoder = Linear(in_channels, hidden_channels, weight_initializer="glorot")
        self.decoder = Linear(hidden_channels, 1, weight_initializer="glorot")
        self.evolution = None  # for tracking hidden layer activations

    def forward(self, x, edge_index, y=None, evo_tracking=False):
        if self.layers is None:
            raise ValueError("self.layers is undefined")

        if self.edge_weights is not None:
            num_graphs = edge_index.size(1) // len(self.edge_weights)
            edge_weights = self.edge_weights.clamp(min=1e-10).repeat(num_graphs).to(x.device)
        else:
            edge_weights = None

        x_0 = self.encoder(x)
        self.evolution = [x_0] if evo_tracking else None

        x = x_0
        for layer in self.layers:
            x = self.apply_layer(layer, x, x_0, edge_index, edge_weights)
            if evo_tracking:
                self.evolution.append(x)
        x = self.decoder(x)

        if y is not None:  # IPU paradigm demands that models return the loss as 2nd output
            return x, mse_loss(x, y)
        return x

    @abstractmethod
    def apply_layer(self, layer, x, x_0, edge_index, edge_weights):
        pass


class FloodMLP(BaseModel):
    def __init__(self, in_channels, hidden_channels, num_hidden, residual):
        super().__init__(in_channels, hidden_channels)
        self.residual = residual
        self.edge_weights = None
        self.layers = ModuleList([Linear(hidden_channels, hidden_channels, weight_initializer="glorot")
                                  for _ in range(num_hidden)])

    def apply_layer(self, layer, x, x_0, edge_index, edge_weights):
        return relu(layer(x)) + (x if self.residual else 0)


class FloodGCN(BaseModel):
    def __init__(self, in_channels, hidden_channels, num_hidden, residual, edge_weights):
        super().__init__(in_channels, hidden_channels)
        self.residual = residual
        self.edge_weights = edge_weights
        self.layers = ModuleList([GCNConv(hidden_channels, hidden_channels, add_self_loops=False, cached=True)
                                  for _ in range(num_hidden)])

    def apply_layer(self, layer, x, x_0, edge_index, edge_weights):
        return relu(layer(x, edge_index, edge_weights)) + (x if self.residual else 0)


class FloodGCNII(BaseModel):
    def __init__(self, in_channels, hidden_channels, num_hidden, edge_weights):
        super().__init__(in_channels, hidden_channels)
        self.edge_weights = edge_weights
        self.layers = ModuleList([GCN2Conv(hidden_channels, alpha=0.5, add_self_loops=False, cached=True)
                                  for _ in range(num_hidden)])

    def apply_layer(self, layer, x, x_0, edge_index, edge_weights):
        return relu(layer(x, x_0, edge_index, edge_weights))


class FloodGRAFFNN(BaseModel):
    def __init__(self, in_channels, hidden_channels, num_hidden, shared_weights, step_size, edge_weights):
        super().__init__(in_channels, hidden_channels)
        self.edge_weights = edge_weights
        if shared_weights:
            self.layers = ModuleList(num_hidden * [GRAFFConv(channels=hidden_channels, step_size=step_size,
                                                             add_self_loops=False, cached=True)])
        else:
            self.layers = ModuleList([GRAFFConv(channels=hidden_channels, step_size=step_size,
                                                add_self_loops=False, cached=True)
                                      for _ in range(num_hidden)])

    def apply_layer(self, layer, x, x_0, edge_index, edge_weights):
        return layer(x, x_0, edge_index, edge_weights)  # GRAFFConv already includes non-linearity

from torch.nn import Module, ModuleList, ReLU, Sequential
from torch_geometric.nn import GCNConv, Linear
from torch_geometric.nn.resolver import activation_resolver


# IPU paradigm demands that models return the loss as second output
def append_MSE(x, y):
    if y is not None:
        return x, F.mse_loss(x, y)
    return x


# functionality: encoder/decoder, evolution tracking
class BaseModel(Module):
    def __init__(self, window_size_hrs, hidden_size, act, residual):
        self.encoder = Sequential(Linear(window_size_hrs, hidden_size, weight_initializer="glorot"),
                                  ReLU(inplace=True))
        self.decoder = Linear(hidden_size, 1, weight_initializer="glorot")
        self.residual = residual
        self.act = activation_resolver(act, **(act_kwargs or {}))
        self.evolution = None  # for tracking hidden layer activations

    def update_evolution(self, x):
        if self.evo_tracking:
            if self.evolution is None:
                self.evolution = [x]
            else:
                self.evolution.append(x)

    def forward(self, evo_tracking):
        self.evolution = None
        self.evo_tracking = evo_tracking


class MLP(BaseModel):
    def __init__(self, window_size_hrs, num_hidden, hidden_size, residual):
        super().__init__(residual, act)
        self.layers = ModuleList(
            [Linear(hidden_size, hidden_size, weight_initializer="glorot")] for _ in range(num_hidden))

    def forward(self, x, edge_index, y=None):
        super().forward(evo_tracking)

        x = self.encoder(x)
        self.update_evolution(x)
        for layer in self.layers:
            x = self.act(layer(x)) + (x if self.residual else 0)
            self.update_evolution(x)
        x = self.decoder(x)

        return append_mse(x, y)


class WeightedGCN(BaseModel):
    def __init__(self, window_size_hrs, num_convs, hidden_size, residual, act, edge_weights):
        super().__init__(residual, act)
        self.dense_in = Linear(window_size_hrs, hidden_size, weight_initializer="glorot")
        self.convs = ModuleList([GCNConv(hidden_size, hidden_size, add_self_loops=False) for _ in range(num_convs)])
        self.dense_out = Linear(hidden_size, 1, weight_initializer="glorot")
        self.edge_weights = edge_weights

    def forward(self, x, edge_index, y=None, evo_tracking=False):
        super().forward(evo_tracking)
        num_graphs = edge_index.size(1) // len(self.edge_weights)
        edge_weights = self.edge_weights.clamp(min=1e-10).repeat(num_graphs).to(x.device)

        x = self.encoder(x)
        self.update_evolution(x)
        for conv in self.convs:
            x = self.act(conv(x, edge_index, edge_weights)) + (x if self.residual else 0)
            self.update_evolution(x)
        x = self.decoder(x)

        return append_mse(x, y)



class SymmetricLinear(Linear):
    class Symmetric(Module):
        def forward(self, x):
            return x.triu() + x.triu(1).transpose(-1, -2)

    def __init__(self, channels):
        super().__init__(channels, channels, bias=False, weight_initializer='glorot')
        register_parametrization(self, 'weight', self.Symmetric())

class PosDefSymLinear(Linear):
    class PosDef(Module):
        def forward(self, x):
            return torch.mm(x.T, x)

    def __init__(self, channels):
        super().__init__(channels, channels, bias=False, weight_initializer='glorot')
        register_parametrization(self, 'weight', self.PosDef())

class NegDefSymLinear(Linear):
    class NegDef(Module):
        def forward(self, x):
            return -torch.mm(x.T, x)

    def __init__(self, channels):
        super().__init__(channels, channels, bias=False, weight_initializer='glorot')
        register_parametrization(self, 'weight', self.NegDef())


class GRAFFNN(Module):
    def __init__(self, num_layers: int, in_channels: int, out_channels: int, hidden_channels: int,
                 act: Union[str, Callable, None] = "relu", act_kwargs: Optional[Dict[str, Any]] = None,
                 shared_weights: bool = True, step_size: int = 1):
        super().__init__()
        self.act = activation_resolver(act, **(act_kwargs or {}))
        self.encoder = MLP([in_channels, hidden_channels])
        self.decoder = MLP([hidden_channels, out_channels])
        if shared_weights:
            self.graff_convs = ModuleList(num_layers * [GRAFFConv(channels=hidden_channels, step_size=step_size)])
        else:
            self.graff_convs = ModuleList([GRAFFConv(channels=hidden_channels, step_size=step_size)
                                           for _ in range(num_layers)])

    def forward(self, data):
        x_0 = self.encoder(data.x)
        x = x_0
        for graff_conv in self.graff_convs:
            x = graff_conv(x, x_0, data.edge_index)
            if self.act is not None:
                x = self.act(x)
        return self.decoder(x)


class GRAFFConv(MessagePassing):
    _cached_edge_index: Optional[OptPairTensor]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, channels: int, step_size: int = 1,
                 cached: bool = False, add_self_loops: bool = True, normalize: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.channels = channels
        self.step_size = step_size
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self.internal_mixer = SymmetricLinear(channels)
        self.external_mixer = SymmetricLinear(channels)
        self.initial_mixer = SymmetricLinear(channels)
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.internal_mixer.reset_parameters()
        self.external_mixer.reset_parameters()
        self.initial_mixer.reset_parameters()
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x: Tensor, x_0: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim), False,
                        self.add_self_loops, self.flow, dtype=x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim), False,
                        self.add_self_loops, self.flow, dtype=x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        internal_repr = self.propagate(edge_index, x=self.internal_mixer(x), edge_weight=edge_weight)
        return x + self.step_size * (internal_repr - self.external_mixer(x))  #- self.initial_mixer(x_0))

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.channels})'
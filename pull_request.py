from torch import Tensor
from torch.nn import Module, ModuleList
from torch.nn.utils.parametrize import register_parametrization
from torch_geometric.nn import MLP
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, SparseTensor
from torch_geometric.utils import spmm
from typing import Any, Callable, Dict, Optional, Union


class SymmetricLinear(Linear):
    """
    only for square matrices, i.e., in_channels == out_channels
    """

    class Symmetrizer(Module):
        def forward(self, x: Tensor) -> Tensor:
            return x.triu() + x.triu(1).transpose(-1, -2)

    def __init__(self, channels: int):
        super().__init__(channels, channels, bias=False, weight_initializer='glorot')
        register_parametrization(self, 'weight', self.Symmetrizer())


class GRAFFNN(Module):
    def __init__(self,
                 num_layers: int,
                 in_channels: int,
                 out_channels: int,
                 hidden_channels: int,
                 shared_weights: bool = True,
                 step_size: int = 1,
                 act: Union[str, Callable, None] = 'relu',
                 act_kwargs: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.encoder = MLP([in_channels, hidden_channels])
        self.decoder = MLP([hidden_channels, out_channels])
        if shared_weights:
            self.graff_convs = ModuleList(num_layers * [GRAFFConv(channels=hidden_channels, step_size=step_size,
                                                                  act=act, act_kwargs=act_kwargs)])
        else:
            self.graff_convs = ModuleList([GRAFFConv(channels=hidden_channels, step_size=step_size,
                                                     act=act, act_kwargs=act_kwargs)
                                           for _ in range(num_layers)])

    def forward(self, x, edge_index, edge_weight: OptTensor = None):
        x_0 = self.encoder(x)
        x = x_0
        for graff_conv in self.graff_convs:
            x = graff_conv(x, x_0, edge_index, edge_weight)
            if self.act is not None:
                x = self.act(x)
        return self.decoder(x)


class GRAFFConv(MessagePassing):
    _cached_edge_index: Optional[OptPairTensor]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self,
                 channels: int,
                 step_size: int = 1,
                 act: Union[str, Callable, None] = 'relu',
                 act_kwargs: Optional[Dict[str, Any]] = None,
                 cached: bool = False,
                 add_self_loops: bool = True,
                 normalize: bool = True,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.channels = channels
        self.step_size = step_size
        self.act = activation_resolver(act, **(act_kwargs or {}))
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

    def forward(self, x: Tensor, x_0: Tensor, edge_index: Adj, edge_weight: OptTensor = None) -> Tensor:
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
        return x + self.step_size * self.act(internal_repr - self.external_mixer(x) - self.initial_mixer(x_0))

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.channels})'

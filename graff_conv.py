from typing import Any, Callable, Dict, Optional, Union

from torch import Tensor
from torch.nn import Module
from torch.nn.utils.parametrize import register_parametrization
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, SparseTensor
from torch_geometric.utils import spmm


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
        external_repr = self.external_mixer(x)
        initial_repr = self.initial_mixer(x_0)
        return x + self.step_size * self.act(internal_repr - external_repr - initial_repr)

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.channels})'


class SymmetricLinear(Linear):

    class Symmetrizer(Module):
        def forward(self, x: Tensor) -> Tensor:
            return x.triu() + x.triu(1).transpose(-1, -2)

    def __init__(self, channels: int):
        super().__init__(channels, channels, bias=False, weight_initializer='glorot')
        register_parametrization(self, 'weight', self.Symmetrizer())

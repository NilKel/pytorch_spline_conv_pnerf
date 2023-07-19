from typing import Optional

import torch

from .basis import spline_basis
from .weighting_mod import spline_weighting
from .scatter import scatter

@torch.jit.script
def spline_conv_mod(x: torch.Tensor, edge_index: torch.Tensor,
                pseudo: torch.Tensor,
                kernel_size: torch.Tensor, is_open_spline: torch.Tensor,
                degree: int = 1, norm: bool = True,
                root_weight: Optional[torch.Tensor] = None,
                bias: Optional[torch.Tensor] = None,
                o_size: int = 0) -> torch.Tensor:
    r"""Applies the spline-based convolution operator :math:`(f \star g)(i) =
    \frac{1}{|\mathcal{N}(i)|} \sum_{l=1}^{M_{in}} \sum_{j \in \mathcal{N}(i)}
    f_l(j) \cdot g_l(u(i, j))` over several node features of an input graph.
    The kernel function :math:`g_l` is defined over the weighted B-spline
    tensor product basis for a single input feature map :math:`l`.

    Args:
        x (:class:`Tensor`): Input node features of shape
            (number_of_nodes x in_channels x grid dimensions).
        edge_index (:class:`LongTensor`): Graph edges, given by source and
            target indices, of shape (2 x number_of_edges) in the fixed
            interval [0, 1].
        pseudo (:class:`Tensor`): Edge attributes, ie. pseudo coordinates,
            of shape (number_of_edges x number_of_edge_attributes).
        weight (:class:`Tensor`): Trainable weight parameters of shape
            (kernel_size x in_channels x out_channels x number of nodes).
        kernel_size (:class:`LongTensor`): Size of the feature grid
        is_open_spline (:class:`ByteTensor`): Whether to use open or closed
            B-spline bases for each dimension.
        degree (int, optional): B-spline basis degree. (default: :obj:`1`)
        norm (bool, optional): Whether to normalize output by node degree.
            (default: :obj:`True`)
        root_weight (:class:`Tensor`, optional): Additional shared trainable
            parameters for each feature of the root node of shape
            (in_channels x out_channels). (default: :obj:`None`)
        bias (:class:`Tensor`, optional): Optional bias of shape
            (out_channels). (default: :obj:`None`)

    :rtype: :class:`Tensor`
    """

    x = x.unsqueeze(-1) if x.dim() == 1 else x
    pseudo = pseudo.unsqueeze(-1) if pseudo.dim() == 1 else pseudo

    row, col = edge_index[0], edge_index[1]

    # Weight each node. Should give E number of basis functions
    basis, weight_index = spline_basis(pseudo, kernel_size, is_open_spline,
                                       degree)

    #new spline convolution
    out = spline_weighting(x[col], basis, weight_index)
        
    finout= scatter(out,o_size,row)

    return finout

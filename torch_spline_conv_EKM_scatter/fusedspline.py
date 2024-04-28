import torch


@torch.jit.script
def fusedspline(feats: torch.Tensor,
                edge_index: torch.Tensor, scatter_index: torch.Tensor,
                pseudo: torch.Tensor, kernel_size: torch.Tensor,
                is_open_spline: torch.Tensor, size_scatter_out: int, degree: int, 
                basis: torch.Tensor, weight_index: torch.Tensor,) -> torch.Tensor:
    return torch.ops.torch_spline_conv_EKM_scatter.fusedspline(
        feats, edge_index, scatter_index, pseudo, kernel_size, is_open_spline, size_scatter_out, degree, basis, weight_index)


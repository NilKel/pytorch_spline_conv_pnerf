import torch


@torch.jit.script
def multispline_fused(feats: torch.Tensor,
                edge_index: torch.Tensor, scatter_index: torch.Tensor,
                pseudo: torch.Tensor, kernel_size: torch.Tensor,
                is_open_spline: torch.Tensor, size_scatter_out: int, 
                grad_feats: torch.Tensor, 
                basis: torch.Tensor, weight_index: torch.Tensor, xyz: torch.Tensor, resolution: torch.Tensor, log2_hashmap_size: int, cellsize: torch.Tensor) -> torch.Tensor:
    return torch.ops.torch_spline_conv_EKM.multispline_fused(
        feats, edge_index, scatter_index, pseudo, kernel_size, is_open_spline, size_scatter_out, grad_feats, basis, weight_index, xyz, resolution, log2_hashmap_size, cellsize)
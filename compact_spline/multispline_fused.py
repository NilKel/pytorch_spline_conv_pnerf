import torch


@torch.jit.script
def multispline_fused(feats: torch.Tensor,
                edge_index: torch.Tensor, scatter_index: torch.Tensor,
                pseudo: torch.Tensor, kernel_size: torch.Tensor,
                is_open_spline: torch.Tensor, size_scatter_out: int, 
                basis: torch.Tensor, weight_index: torch.Tensor, log2_hashmap_size: int,
                primes: torch.Tensor, offsets: torch.Tensor, factors: torch.Tensor) -> torch.Tensor:
    return torch.ops.compact_spline.multispline_fused(
        feats, edge_index, scatter_index, pseudo, kernel_size, is_open_spline, size_scatter_out, basis, weight_index, log2_hashmap_size, primes, offsets, factors)
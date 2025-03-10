from typing import Tuple
import torch

@torch.jit.script
def ray_intersect(
    ray_origin: torch.Tensor,
    ray_dirs: torch.Tensor, 
    inv_ray_dirs: torch.Tensor,
    candidate_ids: torch.Tensor, 
    start_indices: torch.Tensor,
    ray_counts: torch.Tensor, 
    min_bounds: torch.Tensor,
    max_bounds: torch.Tensor, 
    M: int,
    num_samples: int,
    conf_kernel_size: int,
    conf_grid: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.ops.compact_ray_intersect.ray_intersect(
        ray_origin, ray_dirs, inv_ray_dirs, candidate_ids, start_indices, ray_counts,
        min_bounds, max_bounds, M, num_samples, conf_kernel_size, conf_grid
    )

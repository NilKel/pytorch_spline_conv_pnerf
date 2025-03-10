from typing import Tuple
import torch

@torch.jit.script
def confidence_basis(
    pseudo: torch.Tensor,
    pseudo_unscaled: torch.Tensor,
    kernel_size: torch.Tensor,
    is_open_spline: torch.Tensor,
    degree: int,
    resolution: torch.Tensor,
    log2_hashmap_size: int,
    cellsize: torch.Tensor,
    xyz: torch.Tensor,
    point_index: torch.Tensor,
    primes: torch.Tensor,
    offsets: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the confidence basis and corresponding indices using a multispline formulation.
    
    Args:
        pseudo: A tensor representing the scaled pseudo-coordinates.
        pseudo_unscaled: A tensor of the unscaled pseudo-coordinates (used for gradient propagation).
        kernel_size: A tensor specifying the kernel size (e.g., 3x3x3 for each primitive).
        is_open_spline: A tensor indicating whether the spline is open.
        degree: The spline degree.
        resolution: A tensor containing the resolution per level.
        log2_hashmap_size: The log2 of the hashmap size (used for hashing, though for confidence you might set this differently).
        cellsize: A tensor representing the cell size.
        xyz: A tensor of 3D positions (sample points).
        point_index: A tensor with indices of the corresponding primitive for each sample.
        primes: A tensor of primes (for hashing; for confidence you may pass a tensor of ones).
        offsets: A tensor of offsets.
        
    Returns:
        A tuple (confidence_basis, confidence_indices). The forward pass does not modify its outputs.
    """
    return torch.ops.compact_spline.confidence_basis(
        pseudo, pseudo_unscaled, kernel_size, is_open_spline, degree,
        resolution, log2_hashmap_size, cellsize, xyz, point_index, primes, offsets
    )

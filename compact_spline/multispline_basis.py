from typing import Tuple

import torch


@torch.jit.script
def multispline_basis(pseudo: torch.Tensor, kernel_size: torch.Tensor,
                 is_open_spline: torch.Tensor,
                 degree: int, resolution: torch.Tensor, log2_hashmap_size: int,
                  cellsize: torch.Tensor, xyz: torch.Tensor, point_index: torch.Tensor, primes: torch.Tensor, offsets: torch.Tensor  ) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.compact_spline.multispline_basis(pseudo, kernel_size,
                                                    is_open_spline, degree, resolution, log2_hashmap_size, cellsize, xyz, point_index, primes, offsets)

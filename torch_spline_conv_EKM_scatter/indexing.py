import torch


@torch.jit.script
def indexing(x: torch.Tensor,
                     size: int,
                     point_inds: torch.Tensor, grad_zeros: torch.Tensor) -> torch.Tensor:
    return torch.ops.torch_spline_conv_EKM_scatter.indexing(
        x, size, point_inds, grad_zeros)

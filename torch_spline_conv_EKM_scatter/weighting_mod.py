import torch


@torch.jit.script
def spline_weighting(x: torch.Tensor,
                     basis: torch.Tensor,
                     weight_index: torch.Tensor) -> torch.Tensor:
    return torch.ops.torch_spline_conv_EKM_scatter.spline_weighting(
        x, basis, weight_index)

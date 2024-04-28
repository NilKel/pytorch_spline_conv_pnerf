import torch


@torch.jit.script
def multispline_weighting(x: torch.Tensor,
                     basis: torch.Tensor,
                     weight_index: torch.Tensor, kernel_sizes: torch.Tensor) -> torch.Tensor:
    return torch.ops.torch_spline_conv_EKM.multispline_weighting(
        x, basis, weight_index, kernel_sizes)

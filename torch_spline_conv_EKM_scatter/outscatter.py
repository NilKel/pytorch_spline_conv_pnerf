import torch


@torch.jit.script
def outscatter(input: torch.Tensor,
                     output: torch.Tensor,
                     sample_inds: torch.Tensor) -> torch.Tensor:
    return torch.ops.torch_spline_conv_EKM_scatter.outscatter(
        input, output, sample_inds)

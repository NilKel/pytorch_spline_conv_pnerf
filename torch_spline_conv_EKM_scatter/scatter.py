import torch


@torch.jit.script
def scatter(out: torch.Tensor,
                     size: int,
                     sample_inds: torch.Tensor) -> torch.Tensor:
    return torch.ops.torch_spline_conv_EKM_scatter.scatter(
        out, size, sample_inds)

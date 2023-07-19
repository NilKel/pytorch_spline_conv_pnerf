import importlib
import os.path as osp

import torch

__version__ = '1.2.2'

for library in ['_version', '_basis', '_weighting_mod', '_scatter']:
    cuda_spec = importlib.machinery.PathFinder().find_spec(
        f'{library}_cuda', [osp.dirname(__file__)])
    cpu_spec = importlib.machinery.PathFinder().find_spec(
        f'{library}_cpu', [osp.dirname(__file__)])
    spec = cuda_spec or cpu_spec
    if spec is not None:
        torch.ops.load_library(spec.origin)
    else:  # pragma: no cover
        raise ImportError(f"Could not find module '{library}_cpu' in "
                          f"{osp.dirname(__file__)}")

cuda_version = torch.ops.torch_spline_conv_EKM_scatter.cuda_version()
if torch.version.cuda is not None and cuda_version != -1:  # pragma: no cover
    if cuda_version < 10000:
        major, minor = int(str(cuda_version)[0]), int(str(cuda_version)[2])
    else:
        major, minor = int(str(cuda_version)[0:2]), int(str(cuda_version)[3])
    t_major, t_minor = [int(x) for x in torch.version.cuda.split('.')]

    if t_major != major:
        raise RuntimeError(
            f'Detected that PyTorch and torch_spline_conv_EKM_scatter were compiled with '
            f'different CUDA versions. PyTorch has CUDA version '
            f'{t_major}.{t_minor} and torch_spline_conv_EKM_scatter has CUDA version '
            f'{major}.{minor}. Please reinstall the torch_spline_conv_EKM_scatter that '
            f'matches your PyTorch install.')

from .basis import spline_basis  # noqa  # noqa
from .newconv import spline_conv_mod  # noqa
from .weighting_mod import spline_weighting  # noqa
from .scatter import scatter

__all__ = [
    'spline_basis',
    'spline_weighting',
    'spline_conv_mod',
    'scatter',
    '__version__',
]

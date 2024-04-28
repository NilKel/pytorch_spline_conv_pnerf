import importlib
import os.path as osp

import torch

__version__ = '1.2.2'

for library in ['_version' ,'_multispline_basis', '_multispline_fused' ]:
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

cuda_version = torch.ops.compact_spline.cuda_version()
if torch.version.cuda is not None and cuda_version != -1:  # pragma: no cover
    if cuda_version < 10000:
        major, minor = int(str(cuda_version)[0]), int(str(cuda_version)[2])
    else:
        major, minor = int(str(cuda_version)[0:2]), int(str(cuda_version)[3])
    t_major, t_minor = [int(x) for x in torch.version.cuda.split('.')]

    if t_major != major:
        raise RuntimeError(
            f'Detected that PyTorch and compact_spline were compiled with '
            f'different CUDA versions. PyTorch has CUDA version '
            f'{t_major}.{t_minor} and compact_spline has CUDA version '
            f'{major}.{minor}. Please reinstall the compact_spline that '
            f'matches your PyTorch install.')

from .multispline_basis import multispline_basis  # noqa
from .multispline_fused import multispline_fused  # noqa

__all__ = [
    'multispline_basis',
    'multispline_fused',
    '__version__',
]

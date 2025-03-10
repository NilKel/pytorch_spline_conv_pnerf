#pragma once
#include <torch/extension.h>

/// Forward CUDA function for confidence basis generation.
/// The extra argument "pseudo_unscaled" is provided for gradient flow (dummy input) but is ignored in the forward computation.
std::tuple<torch::Tensor, torch::Tensor>
confidence_basis_fw_cuda(torch::Tensor pseudo,
                         torch::Tensor pseudo_unscaled,  // dummy input for gradient flow
                         torch::Tensor kernel_size,
                         torch::Tensor is_open_spline,
                         int64_t degree,
                         torch::Tensor resolution,
                         int64_t log2_hashmap_size,
                         int64_t cellsize,
                         torch::Tensor xyz,
                         torch::Tensor point_index,
                         torch::Tensor primes,
                         torch::Tensor offsets);

/// Backward CUDA function for confidence basis generation.
torch::Tensor confidence_basis_bw_cuda(torch::Tensor grad_basis,
                                       torch::Tensor pseudo,
                                       torch::Tensor kernel_size,
                                       torch::Tensor is_open_spline,
                                       int64_t degree);

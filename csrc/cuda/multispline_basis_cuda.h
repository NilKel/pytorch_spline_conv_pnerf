#pragma once

#include <torch/extension.h>

std::tuple<torch::Tensor, torch::Tensor>
multispline_basis_fw_cuda(torch::Tensor pseudo, torch::Tensor kernel_size,
                     torch::Tensor is_open_spline, int64_t degree,
                     torch::Tensor resolution, int64_t log2_hashmap_size,
                     torch::Tensor cellsize, torch::Tensor xyz, torch::Tensor point_index);

torch::Tensor multispline_basis_bw_cuda(torch::Tensor grad_basis,
                                   torch::Tensor pseudo,
                                   torch::Tensor kernel_size,
                                   torch::Tensor is_open_spline,
                                   int64_t degree);

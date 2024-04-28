#pragma once

#include <torch/extension.h>

int64_t cuda_version();

std::tuple<torch::Tensor, torch::Tensor>
spline_basis(torch::Tensor pseudo, torch::Tensor kernel_size,
             torch::Tensor is_open_spline, int64_t degree);

torch::Tensor spline_weighting(torch::Tensor x,
                               torch::Tensor basis, torch::Tensor weight_index);


torch::Tensor multispline_fused(torch::Tensor feats,
                                    torch::Tensor edge_index,
                                    torch::Tensor scatter_index,
                                    torch::Tensor pseudo, torch::Tensor kernel_size,
                                    torch::Tensor is_open_spline,
                                    int64_t size_scatter_out, torch::Tensor grad_feats, torch::Tensor basis,
                                    torch::Tensor weight_index, torch::Tensor xyz, torch::Tensor resolution, int64_t log2_hashmap_size, torch::Tensor cellsize);
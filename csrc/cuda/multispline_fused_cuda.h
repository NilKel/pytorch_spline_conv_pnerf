#pragma once

#include <torch/extension.h>

torch::Tensor multispline_fused_fw_cuda(torch::Tensor feats,
                                  torch::Tensor edge_index,
                                  torch::Tensor scatter_index,
                                  torch::Tensor pseudo, torch::Tensor kernel_size,
                                  torch::Tensor is_open_spline,
                                  int64_t size_scatter_out,
                                  torch::Tensor basis, torch::Tensor weight_index, torch::Tensor xyz,
                                  torch::Tensor resolution, int64_t log2_hashmap_size, torch::Tensor cellsize);

torch::Tensor multispline_fused_bw_cuda(torch::Tensor grad_feats, torch::Tensor grad_out,
                                  torch::Tensor edge_index,torch::Tensor scatter_index,
                                  torch::Tensor pseudo, torch::Tensor kernel_size, torch::Tensor is_open_spline, torch::Tensor basis, torch::Tensor weight_index, torch::Tensor xyz,
                                  torch::Tensor resolution, int64_t log2_hashmap_size, torch::Tensor cellsize);


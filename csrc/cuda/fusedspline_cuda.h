#pragma once

#include <torch/extension.h>

torch::Tensor fusedspline_fw_cuda(torch::Tensor feats,
                                    torch::Tensor edge_index,
                                    torch::Tensor scatter_index,
                                    torch::Tensor pseudo, torch::Tensor kernel_size,
                                    torch::Tensor is_open_spline,
                                    int64_t size_scatter_out, int64_t degree,
                                    torch::Tensor basis, torch::Tensor weight_index);

torch::Tensor fusedspline_bw_cuda(torch::Tensor grad_out,
                                        torch::Tensor edge_index,torch::Tensor scatter_index,
                                        torch::Tensor pseudo, torch::Tensor kernel_size, torch::Tensor is_open_spline,
                                        int64_t degree, torch::Tensor basis, torch::Tensor weight_index, int64_t numpoints);

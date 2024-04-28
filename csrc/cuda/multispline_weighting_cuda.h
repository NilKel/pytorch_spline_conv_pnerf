#pragma once

#include <torch/extension.h>

torch::Tensor multispline_weighting_fw_cuda(torch::Tensor x,
                                       torch::Tensor basis,
                                       torch::Tensor weight_index,
                                       torch::Tensor kernel_sizes);

torch::Tensor multispline_weighting_bw_x_cuda(torch::Tensor grad_out,
                                         torch::Tensor basis,
                                         torch::Tensor weight_index,
                                         torch::Tensor kernel_sizes,
                                         int ksize);
#pragma once

#include <torch/extension.h>

torch::Tensor spline_weighting_fw_cuda(torch::Tensor x,
                                       torch::Tensor basis,
                                       torch::Tensor weight_index);

torch::Tensor spline_weighting_bw_x_cuda(torch::Tensor grad_out,
                                         torch::Tensor basis,
                                         torch::Tensor weight_index,
                                         int64_t kernel_size);


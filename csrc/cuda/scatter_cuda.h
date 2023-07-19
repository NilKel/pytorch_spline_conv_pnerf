#pragma once

#include <torch/extension.h>

torch::Tensor scatter_fw_cuda(torch::Tensor out,
                        int64_t size_scatter_out,
                        torch::Tensor sample_ind);

torch::Tensor scatter_bw_cuda(torch::Tensor grad_out,
                        int64_t size_out,
                        torch::Tensor sample_ind);

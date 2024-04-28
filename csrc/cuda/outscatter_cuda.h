#pragma once

#include <torch/extension.h>

torch::Tensor outscatter_fw_cuda(torch::Tensor input,
                        torch::Tensor output,
                        torch::Tensor sample_ind);

torch::Tensor outscatter_bw_cuda(torch::Tensor grad_out,
                        int64_t size_input,
                        int64_t size_input_feat,
                        torch::Tensor sample_ind);

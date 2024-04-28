#pragma once

#include <torch/extension.h>

torch::Tensor indexing_fw_cuda(torch::Tensor x,
                        int64_t size_index_out,
                        torch::Tensor point_ind);

torch::Tensor indexing_bw_cuda(torch::Tensor grad_out,
                        int64_t size_index_out,
                        torch::Tensor point_ind,torch::Tensor grad_zeros);

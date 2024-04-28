#pragma once

#include <torch/extension.h>

int64_t cuda_version();

std::tuple<torch::Tensor, torch::Tensor>
spline_basis(torch::Tensor pseudo, torch::Tensor kernel_size,
             torch::Tensor is_open_spline, int64_t degree);

torch::Tensor spline_weighting(torch::Tensor x,
                               torch::Tensor basis, torch::Tensor weight_index);

torch::Tensor scatter(torch::Tensor out,
                               int64_t size, torch::Tensor sample_index);
                               
torch::Tensor outscatter(torch::Tensor input,
                               torch::Tensor output, torch::Tensor sample_index);

torch::Tensor indexing(torch::Tensor x,
                               int64_t size, torch::Tensor point_index, torch::Tensor grad_zeros);

torch::Tensor fusedspline(torch::Tensor feats,
                                    torch::Tensor edge_index,
                                    torch::Tensor scatter_index,
                                    torch::Tensor pseudo, torch::Tensor kernel_size,
                                    torch::Tensor is_open_spline,
                                    int64_t size_scatter_out, int64_t degree, torch::Tensor grad_feats,
                                    torch::Tensor basis, torch::Tensor weight_index);
            
#include <Python.h>
#include <torch/script.h>


#include "cuda/multispline_fused_cuda.h"
PyMODINIT_FUNC PyInit__multispline_fused_cuda(void) { return NULL; }

torch::Tensor multispline_fused_fw(torch::Tensor feats,
                                  torch::Tensor edge_index,
                                  torch::Tensor scatter_index,
                                  torch::Tensor pseudo,
                                  torch::Tensor kernel_size,
                                  torch::Tensor is_open_spline,
                                  int64_t size_scatter_out, 
                                  torch::Tensor basis,
                                  torch::Tensor weight_index,
                                  int64_t log2_hashmap_size,
                                  torch::Tensor primes,
                                  torch::Tensor offsets) {
  if (feats.device().is_cuda()) {
#ifdef WITH_CUDA
    return multispline_fused_fw_cuda(feats, edge_index, scatter_index, pseudo, kernel_size, is_open_spline,
                                     size_scatter_out, basis, weight_index, log2_hashmap_size, primes, offsets);
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
  } 
  else {
    return multispline_fused_fw_cuda(feats, edge_index, scatter_index, pseudo, kernel_size, is_open_spline,
                                     size_scatter_out, basis, weight_index, log2_hashmap_size, primes, offsets);
  }
}

torch::Tensor multispline_fused_bw(torch::Tensor grad_out,
                                  torch::Tensor edge_index,
                                  torch::Tensor scatter_index,
                                  torch::Tensor pseudo,
                                  torch::Tensor kernel_size,
                                  torch::Tensor is_open_spline,
                                  torch::Tensor basis,
                                  torch::Tensor weight_index,
                                  int64_t log2_hashmap_size,
                                  torch::Tensor primes,
                                  torch::Tensor offsets,
                                  torch::Tensor feats) {
    return multispline_fused_bw_cuda(grad_out, edge_index, scatter_index, pseudo, kernel_size, is_open_spline,
                                     basis, weight_index, log2_hashmap_size, primes, offsets, feats);
}

// New function to compute gradients with respect to basis:
torch::Tensor multispline_fused_bw_basis(torch::Tensor grad_out,
                                         torch::Tensor f,
                                         torch::Tensor weight_index,
                                         torch::Tensor scatter_index) {
    return multispline_fused_bw_basis_cuda(grad_out, f, weight_index, scatter_index);
}

using torch::autograd::AutogradContext;
using torch::autograd::Variable;
using torch::autograd::variable_list;

class Multispline_Fused : public torch::autograd::Function<Multispline_Fused> {
public:
  static variable_list forward(AutogradContext *ctx, Variable feats,
                                Variable edge_index,
                                Variable scatter_index,
                                Variable pseudo,
                                Variable kernel_size,
                                Variable is_open_spline,
                                int64_t size_scatter_out,
                                Variable basis,
                                Variable weight_index,
                                int log2_hashmap_size,
                                Variable primes,
                                Variable offsets) {
    auto multispline_fused_out = multispline_fused_fw(feats, edge_index, scatter_index, pseudo,
                                                     kernel_size, is_open_spline, size_scatter_out,
                                                     basis, weight_index, log2_hashmap_size,
                                                     primes, offsets);
    ctx->save_for_backward({edge_index, scatter_index, pseudo, kernel_size,
                            is_open_spline, basis, weight_index, primes, offsets, feats});
    ctx->saved_data["hashsize"] = log2_hashmap_size;
    return {multispline_fused_out};
  }

  static variable_list backward(AutogradContext *ctx, variable_list grad_outs) {
    auto grad_multispline_fused_out = grad_outs[0].contiguous();
    auto saved = ctx->get_saved_variables();
    auto edge_index = saved[0];
    auto scatter_index = saved[1];
    auto pseudo = saved[2];
    auto kernel_size = saved[3];
    auto is_open_spline = saved[4];
    auto basis = saved[5];
    auto weight_index = saved[6];
    auto primes = saved[7];
    auto offsets = saved[8];
    auto feats = saved[9];
    
    auto log2_hashmap_size = ctx->saved_data["hashsize"].toInt();
    auto grad_feats = multispline_fused_bw(grad_multispline_fused_out, edge_index, scatter_index,
                                           pseudo, kernel_size, is_open_spline,
                                           basis, weight_index, log2_hashmap_size, primes, offsets, feats);
    auto grad_basis = multispline_fused_bw_basis(grad_multispline_fused_out, feats, weight_index, scatter_index);

    return {grad_feats, Variable(), Variable(), Variable(), Variable(), Variable(), Variable(), grad_basis, Variable(), Variable(), Variable(), Variable()};
  }
};

torch::Tensor multispline_fused(torch::Tensor feats,
                                torch::Tensor edge_index,
                                torch::Tensor scatter_index,
                                torch::Tensor pseudo,
                                torch::Tensor kernel_size,
                                torch::Tensor is_open_spline,
                                int64_t size_scatter_out,
                                torch::Tensor basis,
                                torch::Tensor weight_index,
                                int64_t log2_hashmap_size,
                                torch::Tensor primes,
                                torch::Tensor offsets) {
  return Multispline_Fused::apply(feats, edge_index, scatter_index, pseudo, kernel_size, is_open_spline,
                                   size_scatter_out, basis, weight_index, log2_hashmap_size, primes, offsets)[0];
}

static auto registry = torch::RegisterOperators().op("compact_spline::multispline_fused", &multispline_fused);

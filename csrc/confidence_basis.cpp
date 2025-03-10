#include <Python.h>
#include <torch/script.h>
#include <cuda_runtime.h>
#ifdef WITH_CUDA
#include "cuda/confidence_basis_cuda.h"  // This header will declare your CUDA kernels for confidence
#endif

#ifdef _WIN32
#ifdef WITH_CUDA
PyMODINIT_FUNC PyInit__confidence_cuda(void) { return NULL; }
#endif
#endif

// Forward pass: calls the CUDA kernel for confidence basis generation.
std::tuple<torch::Tensor, torch::Tensor>
confidence_basis_fw(torch::Tensor pseudo, torch::Tensor pseudo_unscaled, 
                    torch::Tensor kernel_size,
                    torch::Tensor is_open_spline, int64_t degree, 
                    torch::Tensor resolution, int64_t log2_hashmap_size, int64_t cellsize, 
                    torch::Tensor xyz, torch::Tensor point_index, 
                    torch::Tensor primes, torch::Tensor offsets) {
  if (pseudo.device().is_cuda()) {
#ifdef WITH_CUDA
    return confidence_basis_fw_cuda(pseudo, pseudo_unscaled, kernel_size, is_open_spline, degree, 
                                    resolution, log2_hashmap_size, cellsize, xyz, point_index, primes, offsets);
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
  }
}

// Backward pass: calls the CUDA kernel for computing the gradient with respect to pseudo_unscaled.
torch::Tensor confidence_basis_bw(torch::Tensor grad_basis, torch::Tensor pseudo_unscaled,
                                  torch::Tensor kernel_size,
                                  torch::Tensor is_open_spline, int64_t degree) {
  if (grad_basis.device().is_cuda()) {
#ifdef WITH_CUDA
    return confidence_basis_bw_cuda(grad_basis, pseudo_unscaled, kernel_size, is_open_spline, degree);
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
  }
}

using torch::autograd::AutogradContext;
using torch::autograd::Variable;
using torch::autograd::variable_list;

class ConfidenceBasis : public torch::autograd::Function<ConfidenceBasis> {
public:
  // The forward function takes both scaled and unscaled pseudo (the latter is used for gradient propagation).
  static variable_list forward(AutogradContext *ctx, Variable pseudo, Variable pseudo_unscaled,
                               Variable kernel_size, Variable is_open_spline,
                               int64_t degree, Variable resolution, int64_t log2_hashmap_size, int cellsize,
                               Variable xyz, Variable point_index, Variable primes, Variable offsets) {
    ctx->saved_data["degree"] = degree;
    // Call the CUDA forward function for confidence basis generation.
    auto result = confidence_basis_fw(pseudo, pseudo_unscaled, kernel_size, is_open_spline,
                                      degree, resolution, log2_hashmap_size, cellsize,
                                      xyz, point_index, primes, offsets);
    auto conf_basis = std::get<0>(result);
    auto conf_weight_index = std::get<1>(result);
    // Save inputs needed for the backward pass.
    ctx->save_for_backward({pseudo, kernel_size, is_open_spline});
    // Mark the weight index as non-differentiable.
    ctx->mark_non_differentiable({conf_weight_index});
    return {conf_basis, conf_weight_index};
  }

  static variable_list backward(AutogradContext *ctx, variable_list grad_outs) {
    // grad_outs[0] is the gradient w.r.t. the confidence basis.
    auto grad_conf_basis = grad_outs[0];
    auto saved = ctx->get_saved_variables();
    auto pseudo = saved[0];
    auto kernel_size = saved[1];
    auto is_open_spline = saved[2];
    auto degree = ctx->saved_data["degree"].toInt();
    // Compute the gradient with respect to pseudo_unscaled using your confidence-specific backward CUDA kernel.
    auto grad_pseudo_unscaled = confidence_basis_bw(grad_conf_basis, pseudo, kernel_size, is_open_spline, degree);
    // The forward inputs were: [pseudo, pseudo_unscaled, kernel_size, is_open_spline, degree, resolution, log2_hashmap_size, cellsize, xyz, point_index, primes, offsets]
    // We return gradients for pseudo as None and for pseudo_unscaled as grad_pseudo_unscaled.
    return {Variable(), grad_pseudo_unscaled, Variable(), Variable(), Variable(), Variable(), Variable(), Variable(), Variable(), Variable(), Variable(), Variable()};
  }
};

std::tuple<torch::Tensor, torch::Tensor>
confidence_basis(torch::Tensor pseudo, torch::Tensor pseudo_unscaled, torch::Tensor kernel_size,
                 torch::Tensor is_open_spline, int64_t degree, torch::Tensor resolution,
                 int64_t log2_hashmap_size, int64_t cellsize, torch::Tensor xyz, torch::Tensor point_index,
                 torch::Tensor primes, torch::Tensor offsets) {
  pseudo = pseudo.contiguous();
  pseudo_unscaled = pseudo_unscaled.contiguous();
  auto result = ConfidenceBasis::apply(pseudo, pseudo_unscaled, kernel_size, is_open_spline, degree,
                                       resolution, log2_hashmap_size, cellsize, xyz, point_index, primes, offsets);
  return std::make_tuple(result[0], result[1]);
}

static auto registry = torch::RegisterOperators().op(
    "compact_spline::confidence_basis", &confidence_basis);

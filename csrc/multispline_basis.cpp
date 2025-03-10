#include <Python.h>
#include <torch/script.h>

#include <cuda_runtime.h>
#ifdef WITH_CUDA
#include "cuda/multispline_basis_cuda.h"

#endif

#ifdef _WIN32
#ifdef WITH_CUDA
PyMODINIT_FUNC PyInit__basis_cuda(void) { return NULL; }

#endif
#endif

std::tuple<torch::Tensor, torch::Tensor>
multispline_basis_fw(torch::Tensor pseudo, torch::Tensor pseudo_unscaled, 
                     torch::Tensor kernel_size,
                     torch::Tensor is_open_spline, int64_t degree, 
                     torch::Tensor resolution, int64_t log2_hashmap_size, int64_t cellsize, 
                     torch::Tensor xyz, torch::Tensor point_index, torch::Tensor primes, torch::Tensor offsets) {
  if (pseudo.device().is_cuda()) {
#ifdef WITH_CUDA
    return multispline_basis_fw_cuda(pseudo, pseudo_unscaled, kernel_size, is_open_spline, degree, 
                                     resolution, log2_hashmap_size, cellsize, xyz, point_index, primes, offsets);
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
  }
}

torch::Tensor multispline_basis_bw(torch::Tensor grad_basis, torch::Tensor pseudo_unscaled,
                                   torch::Tensor kernel_size,
                                   torch::Tensor is_open_spline, int64_t degree) {
  if (grad_basis.device().is_cuda()) {
#ifdef WITH_CUDA
    return multispline_basis_bw_cuda(grad_basis, pseudo_unscaled, kernel_size, is_open_spline, degree);
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
  }
}


using torch::autograd::AutogradContext;
using torch::autograd::Variable;
using torch::autograd::variable_list;

class SplineBasis : public torch::autograd::Function<SplineBasis> {
public:
  // Forward now takes two pseudo inputs: pseudo (scaled) and pseudo_unscaled.
  static variable_list forward(AutogradContext *ctx, Variable pseudo, Variable pseudo_unscaled,
                               Variable kernel_size, Variable is_open_spline,
                               int64_t degree, Variable resolution, int64_t log2_hashmap_size, int cellsize,
                               Variable xyz, Variable point_index, Variable primes, Variable offsets) {
    ctx->saved_data["degree"] = degree;
    // Call the CUDA forward function with pseudo (for computation) and pseudo_unscaled (for backward)
    auto result = multispline_basis_fw(pseudo, pseudo_unscaled, kernel_size, is_open_spline,
                                            degree, resolution, log2_hashmap_size, cellsize,
                                            xyz, point_index, primes, offsets);
    auto basis = std::get<0>(result);
    auto weight_index = std::get<1>(result);
    // Save pseudo_unscaled (and any other inputs needed) for backward.
    ctx->save_for_backward({pseudo, kernel_size, is_open_spline});
    // We mark weight_index as non-differentiable.
    ctx->mark_non_differentiable({weight_index});
    return {basis, weight_index};
  }

  static variable_list backward(AutogradContext *ctx, variable_list grad_outs) {
    // grad_outs[0] is the gradient for basis.
    auto grad_basis = grad_outs[0];
    auto saved = ctx->get_saved_variables();
    // Retrieve pseudo_unscaled (the tensor for which we want gradients)
    auto pseudo = saved[0];
    auto kernel_size = saved[1];
    auto is_open_spline = saved[2];
    auto degree = ctx->saved_data["degree"].toInt();
    // Compute grad_pseudo (the gradient with respect to pseudo_unscaled)
    auto grad_pseudo_unscaled = multispline_basis_bw(grad_basis, pseudo, kernel_size, is_open_spline, degree);
    // The forward inputs were: [pseudo, pseudo_unscaled, kernel_size, is_open_spline, degree, resolution, log2_hashmap_size, cellsize, xyz, point_index, primes, offsets]
    // We return gradients for pseudo as None and for pseudo_unscaled as grad_pseudo; all other inputs get None.
    return {Variable(), grad_pseudo_unscaled, Variable(), Variable(), Variable(), Variable(), Variable(), Variable(), Variable(), Variable(), Variable(), Variable()};
  }
};

std::tuple<torch::Tensor, torch::Tensor>
multispline_basis(torch::Tensor pseudo, torch::Tensor pseudo_unscaled, torch::Tensor kernel_size,
                  torch::Tensor is_open_spline, int64_t degree, torch::Tensor resolution,
                  int64_t log2_hashmap_size, int64_t cellsize, torch::Tensor xyz, torch::Tensor point_index,
                  torch::Tensor primes, torch::Tensor offsets) {
  // Ensure inputs are contiguous.
  pseudo = pseudo.contiguous();
  pseudo_unscaled = pseudo_unscaled.contiguous();
  auto result = SplineBasis::apply(pseudo, pseudo_unscaled, kernel_size, is_open_spline, degree,
                                   resolution, log2_hashmap_size, cellsize, xyz, point_index, primes, offsets);
  return std::make_tuple(result[0], result[1]);
}

static auto registry = torch::RegisterOperators().op(
    "compact_spline::multispline_basis", &multispline_basis);
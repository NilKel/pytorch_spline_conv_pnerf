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
multispline_basis_fw(torch::Tensor pseudo, torch::Tensor kernel_size,
                torch::Tensor is_open_spline, int64_t degree, 
                torch::Tensor resolution, int64_t log2_hashmap_size, torch::Tensor cellsize, torch::Tensor xyz, torch::Tensor point_index) {
  if (pseudo.device().is_cuda()) {
    
#ifdef WITH_CUDA
    return multispline_basis_fw_cuda(pseudo, kernel_size, is_open_spline, degree, resolution, log2_hashmap_size, cellsize, xyz, point_index);
    
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
  } 
}

torch::Tensor multispline_basis_bw(torch::Tensor grad_basis, torch::Tensor pseudo,
                              torch::Tensor kernel_size,
                              torch::Tensor is_open_spline, int64_t degree) {
  if (grad_basis.device().is_cuda()) {
#ifdef WITH_CUDA
    return multispline_basis_bw_cuda(grad_basis, pseudo, kernel_size, is_open_spline,
                                degree);
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
  static variable_list forward(AutogradContext *ctx, Variable pseudo,
                               Variable kernel_size, Variable is_open_spline,
                               int64_t degree, Variable resolution, int log2_hashmap_size, Variable cellsize, Variable xyz, Variable point_index) {
    ctx->saved_data["degree"] = degree;
    auto result = multispline_basis_fw(pseudo, kernel_size, is_open_spline, degree, resolution, log2_hashmap_size, cellsize, xyz, point_index);
    auto basis = std::get<0>(result), weight_index = std::get<1>(result);
    ctx->save_for_backward({pseudo, kernel_size, is_open_spline});
    ctx->mark_non_differentiable({weight_index});
    return {basis, weight_index};
  }

  static variable_list backward(AutogradContext *ctx, variable_list grad_outs) {
    auto grad_basis = grad_outs[0];
    auto saved = ctx->get_saved_variables();
    auto pseudo = saved[0], kernel_size = saved[1], is_open_spline = saved[2];
    auto degree = ctx->saved_data["degree"].toInt();
    auto grad_pseudo = multispline_basis_bw(grad_basis, pseudo, kernel_size,
                                       is_open_spline, degree);
    return {grad_pseudo, Variable(), Variable(), Variable()};
  }
};

std::tuple<torch::Tensor, torch::Tensor>
multispline_basis(torch::Tensor pseudo, torch::Tensor kernel_size,
             torch::Tensor is_open_spline, int64_t degree, torch::Tensor resolution, int64_t log2_hashmap_size, torch::Tensor cellsize, torch::Tensor xyz, torch::Tensor point_index) {
  pseudo = pseudo.contiguous();
  auto result = SplineBasis::apply(pseudo, kernel_size, is_open_spline, degree, resolution, log2_hashmap_size, cellsize, xyz, point_index);
  return std::make_tuple(result[0], result[1]);
}

static auto registry = torch::RegisterOperators().op(
    "torch_spline_conv_EKM::multispline_basis", &multispline_basis);

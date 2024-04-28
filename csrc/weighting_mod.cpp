#include <Python.h>
#include <torch/script.h>

#include "cpu/weighting_mod_cpu.h"
#ifdef WITH_CUDA
#include "cuda/weighting_mod_cuda.h"
#endif

#ifdef _WIN32
#ifdef WITH_CUDA
PyMODINIT_FUNC PyInit__weighting_cuda(void) { return NULL; }
#else
PyMODINIT_FUNC PyInit__weighting_cpu(void) { return NULL; }
#endif
#endif

torch::Tensor spline_weighting_fw(torch::Tensor x,
                                  torch::Tensor basis,
                                  torch::Tensor weight_index) {
  if (x.device().is_cuda()) {
#ifdef WITH_CUDA
    return spline_weighting_fw_cuda(x, basis, weight_index);
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
  } else {
    return spline_weighting_fw_cpu(x, basis, weight_index);
  }
}

torch::Tensor spline_weighting_bw_x(torch::Tensor grad_out,
                                    torch::Tensor basis,
                                    torch::Tensor weight_index, int64_t kernel_size) {
  if (grad_out.device().is_cuda()) {
#ifdef WITH_CUDA
    return spline_weighting_bw_x_cuda(grad_out, basis, weight_index, kernel_size);
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
  } else {
    return spline_weighting_bw_x_cpu(grad_out, basis, weight_index, kernel_size);
  }
}



// torch::Tensor spline_weighting_bw_basis(torch::Tensor grad_out, torch::Tensor x,
//                                         torch::Tensor weight_index) {
//   if (grad_out.device().is_cuda()) {
// #ifdef WITH_CUDA
//     return spline_weighting_bw_basis_cuda(grad_out, x,  weight_index);
// #else
//     AT_ERROR("Not compiled with CUDA support");
// #endif
//   } else {
//     return spline_weighting_bw_basis_cpu(grad_out, x, weight_index);
//   }
// }

using torch::autograd::AutogradContext;
using torch::autograd::Variable;
using torch::autograd::variable_list;

class SplineWeighting : public torch::autograd::Function<SplineWeighting> {
public:
  static variable_list forward(AutogradContext *ctx, Variable x,
                               Variable basis,
                               Variable weight_index) {
    auto out = spline_weighting_fw(x, basis, weight_index);
    ctx->save_for_backward({x, basis, weight_index});
    return {out};
  }

  static variable_list backward(AutogradContext *ctx, variable_list grad_outs) {
    auto grad_out = grad_outs[0];
    auto saved = ctx->get_saved_variables();
    auto x = saved[0],  basis = saved[1],
         weight_index = saved[2];

    auto grad_x = Variable();
    if (torch::autograd::any_variable_requires_grad({x})) {
      grad_x = spline_weighting_bw_x(grad_out,  basis, weight_index, x.size(1));
    }


    // auto grad_basis = Variable();
    // if (torch::autograd::any_variable_requires_grad({basis})) {
    //   grad_basis = spline_weighting_bw_basis(grad_out, x, weight_index);
    // }

    return {grad_x, Variable(), Variable()};
  }
};

torch::Tensor spline_weighting(torch::Tensor x,
                               torch::Tensor basis,
                               torch::Tensor weight_index) {
  x = x.contiguous();
  
  return SplineWeighting::apply(x, basis, weight_index)[0];
}

static auto registry = torch::RegisterOperators().op(
    "torch_spline_conv_EKM_scatter::spline_weighting", &spline_weighting);

#include <Python.h>
#include <torch/script.h>


#ifdef WITH_CUDA
#include "cuda/multispline_weighting_cuda.h"
#endif

#ifdef _WIN32
#ifdef WITH_CUDA
PyMODINIT_FUNC PyInit__weighting_cuda(void) { return NULL; }

#endif
#endif

torch::Tensor multispline_weighting_fw(torch::Tensor x,
                                  torch::Tensor basis,
                                  torch::Tensor weight_index, torch::Tensor kernel_sizes) {
  if (x.device().is_cuda()) {
#ifdef WITH_CUDA
    return multispline_weighting_fw_cuda(x, basis, weight_index, kernel_sizes);
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
  }
}

torch::Tensor multispline_weighting_bw_x(torch::Tensor grad_out,
                                    torch::Tensor basis,
                                    torch::Tensor weight_index, torch::Tensor kernel_sizes, int ksize) {
  if (grad_out.device().is_cuda()) {
#ifdef WITH_CUDA
// I want to reshape the return value to swap the last two dimensions
    return multispline_weighting_bw_x_cuda(grad_out, basis, weight_index, kernel_sizes, ksize);
    // return multispline_weighting_bw_x_cuda(grad_out, basis, weight_index, kernel_size);
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
  }
}



// torch::Tensor multispline_weighting_bw_basis(torch::Tensor grad_out, torch::Tensor x,
//                                         torch::Tensor weight_index) {
//   if (grad_out.device().is_cuda()) {
// #ifdef WITH_CUDA
//     return multispline_weighting_bw_basis_cuda(grad_out, x,  weight_index);
// #else
//     AT_ERROR("Not compiled with CUDA support");
// #endif
//   } 
// }

using torch::autograd::AutogradContext;
using torch::autograd::Variable;
using torch::autograd::variable_list;

class MultiSplineWeighting : public torch::autograd::Function<MultiSplineWeighting> {
public:
  static variable_list forward(AutogradContext *ctx, Variable x,
                               Variable basis,
                               Variable weight_index, Variable kernel_sizes) {
    auto out = multispline_weighting_fw(x, basis, weight_index, kernel_sizes);
    ctx->save_for_backward({x, basis, weight_index, kernel_sizes});
    return {out};
  }

  static variable_list backward(AutogradContext *ctx, variable_list grad_outs) {
    auto grad_out = grad_outs[0];
    auto saved = ctx->get_saved_variables();
    auto x = saved[0],  basis = saved[1],
         weight_index = saved[2], kernel_sizes = saved[3];

    auto grad_x = Variable();
    if (torch::autograd::any_variable_requires_grad({x})) {
      grad_x = multispline_weighting_bw_x(grad_out,  basis, weight_index, kernel_sizes, x.size(1));
    }


    

    return {grad_x, Variable(), Variable(), Variable()};
  }
};

torch::Tensor multispline_weighting(torch::Tensor x,
                               torch::Tensor basis,
                               torch::Tensor weight_index, torch::Tensor kernel_sizes) {
  x = x.contiguous();
  
  return MultiSplineWeighting::apply(x, basis, weight_index, kernel_sizes)[0];
}

static auto registry = torch::RegisterOperators().op(
    "torch_spline_conv_EKM::multispline_weighting", &multispline_weighting);

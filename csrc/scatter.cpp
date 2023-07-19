#include <Python.h>
#include <torch/script.h>


#include "cuda/scatter_cuda.h"




PyMODINIT_FUNC PyInit__scatter_cuda(void) { return NULL; }

//x, basis, weight
//to out, size(int), sample_inds
torch::Tensor scatter_fw(torch::Tensor out,
                                  int64_t size,
                                  torch::Tensor sample_ind) {
  if (out.device().is_cuda()) {
#ifdef WITH_CUDA
    return scatter_fw_cuda(out, size, sample_ind);
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
  } 
  else {
    return scatter_fw_cuda(out, size, sample_ind);
  }
}

torch::Tensor scatter_bw(torch::Tensor grad_scatter_out,
                                    int64_t size,
                                    torch::Tensor sample_ind) {
  if (grad_scatter_out.device().is_cuda()) {
#ifdef WITH_CUDA
    return scatter_bw_cuda(grad_scatter_out, size, sample_ind);
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
  }
   else {
    return scatter_bw_cuda(grad_scatter_out, size, sample_ind);
  }
}




using torch::autograd::AutogradContext;
using torch::autograd::Variable;
using torch::autograd::variable_list;

class ScatterAdd : public torch::autograd::Function<ScatterAdd> {
public:
  static variable_list forward(AutogradContext *ctx, Variable out,
                               int64_t size,
                               Variable sample_ind) {
    auto scatter_out = scatter_fw(out, size, sample_ind);
    ctx->save_for_backward({out, sample_ind});
    return {scatter_out};
  }

  static variable_list backward(AutogradContext *ctx, variable_list grad_outs) {
    auto grad_scatter_out = grad_outs[0].contiguous();
    auto saved = ctx->get_saved_variables();
    auto out = saved[0], sample_ind = saved[1];

    auto grad_out = Variable();
    if (torch::autograd::any_variable_requires_grad({out})) {
      grad_out = scatter_bw(grad_scatter_out,  out.size(0), sample_ind);
    }



    return {grad_out,Variable(), Variable()};
  }
};

torch::Tensor scatter(torch::Tensor out,
                               int64_t size,
                               torch::Tensor sample_ind) {
  out = out.contiguous();
  sample_ind=sample_ind.contiguous();
  return ScatterAdd::apply(out, size, sample_ind)[0];
}

static auto registry = torch::RegisterOperators().op(
    "torch_spline_conv_EKM_scatter::scatter", &scatter);

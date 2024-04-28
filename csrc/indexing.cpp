#include <Python.h>
#include <torch/script.h>


#include "cuda/indexing_cuda.h"




PyMODINIT_FUNC PyInit__indexing_cuda(void) { return NULL; }

//x, basis, weight
//to out, size(int), sample_inds
torch::Tensor indexing_fw(torch::Tensor x,
                                  int64_t size,
                                  torch::Tensor point_ind,torch::Tensor grad_zeros) {
  if (x.device().is_cuda()) {
#ifdef WITH_CUDA
    return indexing_fw_cuda(x, size, point_ind);
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
  } 
  else {
    return indexing_fw_cuda(x, size, point_ind);
  }
}

torch::Tensor indexing_bw(torch::Tensor grad_indexing_out,
                                    int64_t size,
                                    torch::Tensor point_ind, torch::Tensor grad_zeros) {
  if (grad_indexing_out.device().is_cuda()) {
#ifdef WITH_CUDA
    return indexing_bw_cuda(grad_indexing_out, size, point_ind, grad_zeros);
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
  }
   else {
    return indexing_bw_cuda(grad_indexing_out, size, point_ind, grad_zeros);
  }
}




using torch::autograd::AutogradContext;
using torch::autograd::Variable;
using torch::autograd::variable_list;

class Indexing : public torch::autograd::Function<Indexing> {
public:
  static variable_list forward(AutogradContext *ctx, Variable x,
                               int64_t size,
                               Variable point_ind, Variable grad_zeros) {
    auto indexing_out = indexing_fw(x, size, point_ind, grad_zeros);
    ctx->save_for_backward({x, point_ind, grad_zeros});
    return {indexing_out};
  }

  static variable_list backward(AutogradContext *ctx, variable_list grad_outs) {
    auto grad_indexing_out = grad_outs[0].contiguous();
    
    auto saved = ctx->get_saved_variables();
    auto x = saved[0], point_ind = saved[1];
    auto grad_zeros = saved[2];

    auto grad_x = Variable();
    if (torch::autograd::any_variable_requires_grad({x})) {
      grad_x = indexing_bw(grad_indexing_out,  x.size(0), point_ind, grad_zeros);
    }



    return {grad_x,Variable(), Variable(), Variable()};
  }
};

torch::Tensor indexing(torch::Tensor x,
                               int64_t size,
                               torch::Tensor point_ind, torch::Tensor grad_zeros) {
  x = x.contiguous();
  point_ind=point_ind.contiguous();
  return Indexing::apply(x, size, point_ind, grad_zeros)[0];
}

static auto registry = torch::RegisterOperators().op(
    "torch_spline_conv_EKM_scatter::indexing", &indexing);

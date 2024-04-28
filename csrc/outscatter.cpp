#include <Python.h>
#include <torch/script.h>


#include "cuda/outscatter_cuda.h"




PyMODINIT_FUNC PyInit__outscatter_cuda(void) { return NULL; }

//x, basis, weight
//to out, size(int), sample_inds
torch::Tensor outscatter_fw(torch::Tensor input,
                                  torch::Tensor output,
                                  torch::Tensor sample_ind) {
  if (input.device().is_cuda()) {
#ifdef WITH_CUDA
    return outscatter_fw_cuda(input, output, sample_ind);
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
  } 
  else {
    return outscatter_fw_cuda(input, output, sample_ind);
  }
}

torch::Tensor outscatter_bw(torch::Tensor grad_outscatter_out,
                                    int64_t input_size,int64_t input_feat,
                                    torch::Tensor sample_ind) {
  if (grad_outscatter_out.device().is_cuda()) {
#ifdef WITH_CUDA
    return outscatter_bw_cuda(grad_outscatter_out, input_size,input_feat, sample_ind);
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
  }
   else {
    return outscatter_bw_cuda(grad_outscatter_out, input_size,input_feat, sample_ind);
  }
}




using torch::autograd::AutogradContext;
using torch::autograd::Variable;
using torch::autograd::variable_list;

class OutScatterAdd : public torch::autograd::Function<OutScatterAdd> {
public:
  static variable_list forward(AutogradContext *ctx, Variable input,
                               torch::Tensor output,
                               Variable sample_ind) {
    auto outscatter_out = outscatter_fw(input, output, sample_ind);
    ctx->save_for_backward({input, sample_ind});
    return {outscatter_out};
  }

  static variable_list backward(AutogradContext *ctx, variable_list grad_outs) {
    auto grad_outscatter_out = grad_outs[0].contiguous();
    auto saved = ctx->get_saved_variables();
    auto input = saved[0], sample_ind = saved[1];

    auto grad_out = Variable();
    if (torch::autograd::any_variable_requires_grad({input})) {
      grad_out = outscatter_bw(grad_outscatter_out,  input.size(0), input.size(1), sample_ind);
    }



    return {grad_out,Variable(), Variable()};
  }
};

torch::Tensor outscatter(torch::Tensor input,
                               torch::Tensor output,
                               torch::Tensor sample_ind) {
  input = input.contiguous();
  sample_ind=sample_ind.contiguous();
  return OutScatterAdd::apply(input, output, sample_ind)[0];
}

static auto registry = torch::RegisterOperators().op(
    "torch_spline_conv_EKM_scatter::outscatter", &outscatter);

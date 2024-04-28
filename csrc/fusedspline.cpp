#include <Python.h>
#include <torch/script.h>


#include "cuda/fusedspline_cuda.h"



PyMODINIT_FUNC PyInit__fusedspline_cuda(void) { return NULL; }


torch::Tensor fusedspline_fw(torch::Tensor feats,
                                    torch::Tensor edge_index,
                                    torch::Tensor scatter_index,
                                    torch::Tensor pseudo, torch::Tensor kernel_size,
                                    torch::Tensor is_open_spline,
                                    int64_t size_scatter_out, int64_t degree, torch::Tensor basis,
                                    torch::Tensor weight_index) {
  if (feats.device().is_cuda()) {
#ifdef WITH_CUDA
    return fusedspline_fw_cuda(feats, edge_index, scatter_index, pseudo, kernel_size, is_open_spline, size_scatter_out, degree, basis, weight_index);
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
  } 
  else {
    return fusedspline_fw_cuda(feats, edge_index, scatter_index, pseudo, kernel_size, is_open_spline, size_scatter_out, degree, basis, weight_index);
  }
}

torch::Tensor fusedspline_bw(torch::Tensor grad_out,
                                        torch::Tensor edge_index,torch::Tensor scatter_index,
                                        torch::Tensor pseudo, torch::Tensor kernel_size,
                                        torch::Tensor is_open_spline, int64_t degree, torch::Tensor basis,
                                        torch::Tensor weight_index, int64_t numpoints) {
  if (grad_out.device().is_cuda()) {
#ifdef WITH_CUDA
    return fusedspline_bw_cuda(grad_out, edge_index, scatter_index, pseudo, kernel_size, is_open_spline, degree, basis, weight_index, numpoints);
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
  }
   else {
    return fusedspline_bw_cuda( grad_out, edge_index, scatter_index, pseudo, kernel_size, is_open_spline, degree, basis, weight_index, numpoints);
  }
}




using torch::autograd::AutogradContext;
using torch::autograd::Variable;
using torch::autograd::variable_list;

class FusedSpline : public torch::autograd::Function<FusedSpline> {
public:
  static variable_list forward(AutogradContext *ctx, Variable feats,
                                Variable edge_index, Variable scatter_index,
                                Variable pseudo, Variable kernel_size,
                                Variable is_open_spline, int64_t size_scatter_out, int64_t degree,Variable basis,
                                Variable weight_index) {
    auto fusedspline_out = fusedspline_fw(feats, edge_index, scatter_index, pseudo, kernel_size, is_open_spline, size_scatter_out, degree, basis, weight_index);
    auto numpoints = feats.size(1);
    ctx->save_for_backward({edge_index, scatter_index, pseudo, kernel_size, is_open_spline, basis, weight_index});
    ctx->saved_data["degree"] = degree;
    ctx->saved_data["numpoints"] = numpoints;
    return {fusedspline_out};
  }

  static variable_list backward(AutogradContext *ctx, variable_list grad_outs) {
    auto grad_fusedspline_out = grad_outs[0].contiguous();
    auto saved = ctx->get_saved_variables();
    auto edge_index = saved[0], scatter_index = saved[1], pseudo = saved[2], kernel_size = saved[3], is_open_spline = saved[4],  basis = saved[5], weight_index = saved[6];
    auto numpoints = ctx->saved_data["numpoints"].toInt();
    auto degree = ctx->saved_data["degree"].toInt();
    auto grad_out = Variable();
    // numpoints is the number of points in the input point cloud or feats.size(1)

    grad_out = fusedspline_bw(grad_fusedspline_out, edge_index, scatter_index, pseudo, kernel_size, is_open_spline, degree, basis, weight_index, numpoints);
    



    return {grad_out,Variable(),Variable(),Variable(),Variable(),Variable(),Variable(),Variable(),Variable(),Variable()};
  }
};

torch::Tensor fusedspline(torch::Tensor feats,
                                    torch::Tensor edge_index,
                                    torch::Tensor scatter_index,
                                    torch::Tensor pseudo, torch::Tensor kernel_size,
                                    torch::Tensor is_open_spline,
                                    int64_t size_scatter_out, int64_t degree, torch::Tensor basis,
                                    torch::Tensor weight_index) {
  
  return FusedSpline::apply(feats, edge_index, scatter_index, pseudo, kernel_size, is_open_spline, size_scatter_out, degree, basis, weight_index)[0];
}

static auto registry = torch::RegisterOperators().op(
    "torch_spline_conv_EKM_scatter::fusedspline", &fusedspline);

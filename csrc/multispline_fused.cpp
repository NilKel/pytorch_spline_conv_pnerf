#include <Python.h>
#include <torch/script.h>


#include "cuda/multispline_fused_cuda.h"



PyMODINIT_FUNC PyInit__multispline_fused_cuda(void) { return NULL; }


torch::Tensor multispline_fused_fw(torch::Tensor feats,
                                  torch::Tensor edge_index,
                                  torch::Tensor scatter_index,
                                  torch::Tensor pseudo, torch::Tensor kernel_size,
                                  torch::Tensor is_open_spline,
                                  int64_t size_scatter_out, 
                                  torch::Tensor basis, torch::Tensor weight_index, torch::Tensor xyz,
                                  torch::Tensor resolution, int64_t log2_hashmap_size, torch::Tensor cellsize) {
  if (feats.device().is_cuda()) {
#ifdef WITH_CUDA
    return multispline_fused_fw_cuda(feats, edge_index, scatter_index, pseudo, kernel_size, is_open_spline, size_scatter_out, basis, weight_index, xyz, resolution, log2_hashmap_size, cellsize);
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
  } 
  else {
    return multispline_fused_fw_cuda(feats, edge_index, scatter_index, pseudo, kernel_size, is_open_spline, size_scatter_out, basis, weight_index, xyz, resolution, log2_hashmap_size, cellsize);
  }
}

torch::Tensor multispline_fused_bw(torch::Tensor grad_feats, torch::Tensor grad_out,
                                  torch::Tensor edge_index,torch::Tensor scatter_index,
                                  torch::Tensor pseudo, torch::Tensor kernel_size, torch::Tensor is_open_spline,
                                  torch::Tensor basis, torch::Tensor weight_index, torch::Tensor xyz, torch::Tensor resolution, int64_t log2_hashmap_size, torch::Tensor cellsize) {
  
    return multispline_fused_bw_cuda(grad_feats, grad_out, edge_index, scatter_index, pseudo, kernel_size, is_open_spline, basis, weight_index, xyz, resolution, log2_hashmap_size, cellsize);


  
}




using torch::autograd::AutogradContext;
using torch::autograd::Variable;
using torch::autograd::variable_list;

class Multispline_Fused : public torch::autograd::Function<Multispline_Fused> {
public:
  static variable_list forward(AutogradContext *ctx, Variable feats,
                                Variable edge_index, Variable scatter_index,
                                Variable pseudo, Variable kernel_size,
                                Variable is_open_spline, int64_t size_scatter_out, Variable grad_feats, Variable basis,
                                Variable weight_index, Variable xyz,  Variable resolution, int log2_hashmap_size, Variable cellsize) {
    auto multispline_fused_out = multispline_fused_fw(feats, edge_index, scatter_index, pseudo, kernel_size, is_open_spline, size_scatter_out, basis, weight_index, xyz, resolution, log2_hashmap_size, cellsize);
    ctx->save_for_backward({edge_index, scatter_index, pseudo, kernel_size, is_open_spline, grad_feats, basis, weight_index, xyz, resolution, cellsize});
    
    ctx->saved_data["hashsize"] = log2_hashmap_size;
    return {multispline_fused_out};
  }

  static variable_list backward(AutogradContext *ctx, variable_list grad_outs) {
    auto grad_multispline_fused_out = grad_outs[0].contiguous();
    auto saved = ctx->get_saved_variables();
    auto edge_index = saved[0], scatter_index = saved[1], pseudo = saved[2], kernel_size = saved[3], is_open_spline = saved[4], grad_feats = saved[5], basis = saved[6], weight_index = saved[7],
    xyz = saved[8], resolution = saved[9],  cellsize = saved[10];
    
    
    auto log2_hashmap_size = ctx->saved_data["hashsize"].toInt();
    auto grad_out = Variable();
    
    grad_out = multispline_fused_bw(grad_feats, grad_multispline_fused_out, edge_index, scatter_index, pseudo, kernel_size, is_open_spline, basis, weight_index, xyz, resolution, log2_hashmap_size, cellsize);
    



    return {grad_out,Variable(),Variable(),Variable(),Variable(),Variable(),Variable(),Variable(),Variable(),Variable(),Variable(),Variable()};
  }
};

torch::Tensor multispline_fused(torch::Tensor feats,
                                    torch::Tensor edge_index,
                                    torch::Tensor scatter_index,
                                    torch::Tensor pseudo, torch::Tensor kernel_size,
                                    torch::Tensor is_open_spline,
                                    int64_t size_scatter_out, torch::Tensor grad_feats, torch::Tensor basis,
                                    torch::Tensor weight_index, torch::Tensor xyz, torch::Tensor resolution, int64_t log2_hashmap_size, torch::Tensor cellsize) {
  
  return Multispline_Fused::apply(feats, edge_index, scatter_index, pseudo, kernel_size, is_open_spline, size_scatter_out, grad_feats, basis, weight_index, xyz, resolution, log2_hashmap_size, cellsize)[0];
}

static auto registry = torch::RegisterOperators().op(
    "torch_spline_conv_EKM::multispline_fused", &multispline_fused);

#include "fusedspline_cuda.h"

#include <ATen/cuda/CUDAContext.h>

#include "atomics.cuh"
#include "utils.cuh"

#define THREADS 1024
#define BLOCKS(N) (N + THREADS - 1) / THREADS

template <typename scalar_t, int64_t degree>
struct Basis
{
  static inline __device__ scalar_t forward(scalar_t v, int64_t k_mod)
  {
    if (degree == 1)
    {
      return 1. - v - k_mod + 2. * v * k_mod;
    }
    else if (degree == 2)
    {
      if (k_mod == 0)
        return 0.5 * v * v - v + 0.5;
      else if (k_mod == 1)
        return -v * v + v + 0.5;
      else
        return 0.5 * v * v;
    }
    else if (degree == 3)
    {
      if (k_mod == 0)
        return (1. - v) * (1. - v) * (1. - v) / 6.;
      else if (k_mod == 1)
        return (3. * v * v * v - 6. * v * v + 4.) / 6.;
      else if (k_mod == 2)
        return (-3. * v * v * v + 3. * v * v + 3. * v + 1.) / 6.;
      else
        return v * v * v / 6.;
    }
    else
    {
      return (scalar_t)-1.;
    }
  }

  static inline __device__ scalar_t backward(scalar_t v, int64_t k_mod)
  {
    if (degree == 1)
    {
      return 2 * k_mod - 1;
    }
    else if (degree == 2)
    {
      if (k_mod == 0)
        return v - 1.;
      else if (k_mod == 1)
        return -2. * v + 1.;
      else
        return v;
    }
    else if (degree == 3)
    {
      if (k_mod == 0)
        return (-v * v + 2. * v - 1.) / 2.;
      else if (k_mod == 1)
        return (3. * v * v - 4. * v) / 2.;
      else if (k_mod == 2)
        return (-3. * v * v + 2. * v + 1.) / 2.;
      else
        return v * v / 2.;
    }
    else
    {
      return (scalar_t)-1.;
    }
  }
};

template <typename scalar_t, int64_t S, int64_t degree>
__global__ void
fusedspline_fw_kernel(const scalar_t *f, const scalar_t *pseudo, const int64_t *kernel_size,scalar_t *output,
                       const uint8_t *is_open_spline, const int64_t *scatter_index, const int64_t *edge_index,
                       int64_t E, int64_t D,  int64_t M_out, int64_t K, int64_t N, const scalar_t *basis, const int64_t *weight_index)
{
  // Variables needed are S, pseudo, kernel size, open_spline, basis?
  const int64_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t e = thread_idx / M_out;
  const int64_t m_out = thread_idx % M_out;

  if (thread_idx < E*M_out) {
    #pragma unroll(S)
    for (int64_t s = 0; s < S; s++) {

      auto b = basis[e*S+s];
      auto wi = weight_index[e*S+s];
      
      // auto interm=f[edge_index[e]*M_out*K+m_out*K+wi]*b;
      auto interm=f[edge_index[e]*M_out*K+wi*M_out+m_out]*b;

      atomAdd(output,scatter_index[e]*M_out+m_out,N*M_out,interm);
    }
  }
}

// FIRST WE START WITH THE DISTANCE TENSORS FOR EACH EDGE E
// We want to compute the B-Spline basis of shape E,8 and weight of shape E,8.
// Next we want to index and multiply the basis by the feature.
// Inside a loop of size 8 we next add up all of the features. We can perform indexing here itself!!!!
torch::Tensor fusedspline_fw_cuda(torch::Tensor feats,
                                  torch::Tensor edge_index,
                                  torch::Tensor scatter_index,
                                  torch::Tensor pseudo, torch::Tensor kernel_size,
                                  torch::Tensor is_open_spline,
                                  int64_t size_scatter_out, int64_t degree,
                                  torch::Tensor basis, torch::Tensor weight_index) {
  

  cudaSetDevice(feats.get_device());

  // x has shape [E, M_out, K_size]
  // S has shape shape (degree+1)^dim, 2^3 here.
  auto E = edge_index.size(0); //num_edges
  auto M_out = feats.size(3); //2 in example
  auto D = 3;
  // auto S = (int64_t)(powf(degree + 1, D) + 0.5); //8
  auto K = 27;
  auto out = at::zeros({size_scatter_out, M_out}, feats.options());
  
  auto edge_index_data = edge_index.data_ptr<int64_t>();
  auto scatter_index_data = scatter_index.data_ptr<int64_t>();

  auto weight_index_data = weight_index.data_ptr<int64_t>();


  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(feats.scalar_type(), "fused_fw", [&] {
    auto kernel_size_data = kernel_size.data_ptr<int64_t>();
    auto is_open_spline_data = is_open_spline.data_ptr<uint8_t>();
    auto pseudo_data = pseudo.data_ptr<scalar_t>();
    auto basis_data = basis.data_ptr<scalar_t>();
    auto f_data = feats.data_ptr<scalar_t>();
    auto out_data = out.data_ptr<scalar_t>();
    AT_DISPATCH_DEGREE_TYPES(degree, [&] {
    fusedspline_fw_kernel<scalar_t, 8, DEGREE>
        <<<BLOCKS(E*M_out), THREADS, 0, stream>>>(
            f_data, pseudo_data, kernel_size_data, out_data, is_open_spline_data,
            scatter_index_data, edge_index_data,E,D,
            M_out, K,size_scatter_out, basis_data, weight_index_data
            );
    });
  });
  return out;
}


template <typename scalar_t, int64_t S, int64_t degree>
__global__ void
fusedspline_bw_kernel(scalar_t *grad_feat,const scalar_t *grad_out ,const scalar_t *pseudo, const int64_t *kernel_size,
                             const uint8_t *is_open_spline, const int64_t *sample_index, const int64_t *edge_index,
                             int64_t E, int64_t D, int64_t M_out, int64_t K, int64_t N,
                             int64_t outsize, const scalar_t *basis, const int64_t *weight_index)
{
  
  const int64_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t e = thread_idx / M_out;
  const int64_t m_out = thread_idx % M_out;

  
  if (thread_idx < E*M_out) {
    // #pragma unroll(M_out)
    // for (int64_t m_out = 0; m_out < M_out; m_out++){
      // atomAdd(grad_out,e*M_out+m_out, numel, grad_scatter_out[sample_index[e]*M_out+m_out]);
      auto g = grad_out[sample_index[e]*M_out+m_out];
      
      #pragma unroll(S)
      for (int64_t s = 0; s < S; s++) {
        auto b = basis[e*S+s];
        auto wi = weight_index[e*S+s];
        atomAdd(grad_feat,edge_index[e] * M_out * K  + wi * M_out + m_out,outsize, b*g);
        
      }
    // }
  }
}

// add an int argument for kernel size
torch::Tensor fusedspline_bw_cuda(torch::Tensor grad_out,
                                        torch::Tensor edge_index,torch::Tensor scatter_index,
                                        torch::Tensor pseudo, torch::Tensor kernel_size, torch::Tensor is_open_spline,
                                        int64_t degree, torch::Tensor basis, torch::Tensor weight_index, int64_t numpoints)
{
  
  
  cudaSetDevice(grad_out.get_device());
  auto D = 3;
  auto E = edge_index.size(0); //num_edges
  auto M_out = grad_out.size(1); //2 in example
  // auto S = (int64_t)(powf(degree + 1, D) + 0.5); //8
  auto K = 27;
  
  auto edge_index_data = edge_index.data_ptr<int64_t>();
  auto scatter_index_data = scatter_index.data_ptr<int64_t>();
  auto size_scatter_out = grad_out.size(0);
  
  auto weight_index_data = weight_index.data_ptr<int64_t>();
  
  auto grad_feats = at::zeros({1,numpoints, K, M_out}, grad_out.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_out.scalar_type(), "fused_bw", [&] {
    auto kernel_size_data = kernel_size.data_ptr<int64_t>();
    auto is_open_spline_data = is_open_spline.data_ptr<uint8_t>();
    auto pseudo_data = pseudo.data_ptr<scalar_t>();
    auto grad_out_data = grad_out.data_ptr<scalar_t>();
    auto grad_feat_data = grad_feats.data_ptr<scalar_t>();
    auto basis_data = basis.data_ptr<scalar_t>();
    AT_DISPATCH_DEGREE_TYPES(degree, [&] {
    fusedspline_bw_kernel<scalar_t, 8, DEGREE>
        <<<BLOCKS(E*M_out), THREADS, 0, stream>>>(
            grad_feat_data,grad_out_data, pseudo_data, kernel_size_data,is_open_spline_data,
            scatter_index_data, edge_index_data,E,D,
            M_out, K,size_scatter_out, grad_feats.numel(), basis_data, weight_index_data
            );
    });
  });
  return grad_feats;
}


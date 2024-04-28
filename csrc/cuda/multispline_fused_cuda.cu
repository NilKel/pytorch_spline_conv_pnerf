#include "multispline_fused_cuda.h"

#include <ATen/cuda/CUDAContext.h>

#include "atomics.cuh"
#include "utils.cuh"

using namespace at;

#include <ATen/native/cuda/KernelUtils.cuh>
#define THREADS 1024
#define BLOCKS(N) (N + THREADS - 1) / THREADS

template <typename scalar_t>
__device__ scalar_t clamp_with_gradient(scalar_t value, scalar_t min, scalar_t max) {
    scalar_t clamped = value < min ? min : (value > max ? max : value);
    return value - (clamped - value); // Clamping effect only in forward pass
} 

template <typename scalar_t>
__global__ void
multispline_fused_fw_kernel(const scalar_t *f, const scalar_t *pseudo, const int64_t *kernel_size, scalar_t *output,
                      const uint8_t *is_open_spline, const int64_t *scatter_index, const int64_t *edge_index,
                      int64_t M_out,  int64_t N, const scalar_t *basis, const int64_t *weight_index, 
                      int64_t levels,  int64_t total_size,  long numel, int64_t S, int64_t E, int64_t *primes, int64_t *offsets, 
                      const bool *factors)
{
  
  // Variables needed are S, pseudo, kernel size, open_spline, basis?
  const int64_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t e = (thread_idx / (levels*M_out));
  const int64_t m_out = (thread_idx % (M_out));
  // const int64_t level = (thread_idx %(M_out*levels))/ M_out;
  const int64_t level = (thread_idx % (levels*M_out))/M_out;
  // unsigned int primes[16] = {844154009, 878106367, 311083159, 407038081, 427944887, 431333681, 715902911, 729070343, 919548613, 449542579, 530997011, 314167211, 146148241, 458291711, 225061747, 385280261};
  // unsigned int offsets[16] = {8, 27, 64, 125, 216, 343, 512,729, 1000,1331,1728,2197,2744,3375,4096,4913 };
  if (thread_idx < numel) {
    // scalar_t conf_value = confidence[edge_index[e]*levels+level];
    // scalar_t clamped_conf = clamp_with_gradient(conf_value, static_cast<scalar_t>(0.0), static_cast<scalar_t>(1.0)); // Assuming min=0, max=1

    #pragma unroll(8)
    for (int64_t s = 0; s < S; s++) {
      

      auto b = basis[e*S*levels  + levels * s + level];
      auto wi = weight_index[e*S*levels  + levels * s + level];
      // auto interm=f[edge_index[e]*M_out*K+wi*M_out+m_out]*b;
      // auto interm=f[level * total_size * M_out + wi * M_out + m_out]*b;
      // auto interm=f[level*total_size*M_out+M_out*((edge_index[e]*offsets[level]+wi)%total_size)+m_out]*b;
      // auto interm=f[M_out* levels* (((edge_index[e]*offsets[level]+wi)*(primes[level]*factors[level]+!factors[level]))&(total_size-1)) + m_out*levels +level]*b;
      auto interm=f[M_out* levels* (wi) + m_out*levels +level]*b;
      // auto interm=f[M_out* levels* ((wi)) + m_out*levels +level]*b;
      // atomAdd(output,level* N * M_out + scatter_index[e]*M_out+m_out,N*M_out*levels,interm);
      atomAdd(output,scatter_index[e] * levels* M_out + levels*m_out+level,N*M_out*levels,interm);
      
    }
  }
}

// FIRST WE START WITH THE DISTANCE TENSORS FOR EACH EDGE E
// We want to compute the B-Spline basis of shape E,8 and weight of shape E,8.
// Next we want to index and multiply the basis by the feature.
// Inside a loop of size 8 we next add up all of the features. We can perform indexing here itself!!!!
torch::Tensor multispline_fused_fw_cuda(torch::Tensor feats,
                                  torch::Tensor edge_index,
                                  torch::Tensor scatter_index,
                                  torch::Tensor pseudo, torch::Tensor kernel_size,
                                  torch::Tensor is_open_spline,
                                  int64_t size_scatter_out, 
                                  torch::Tensor basis, torch::Tensor weight_index, int64_t total_size, torch::Tensor primes, torch::Tensor offsets, 
                                  torch::Tensor factors) {
  

  cudaSetDevice(feats.get_device());

  // x has shape [E, M_out, K_size]
  int64_t E = edge_index.size(0); //num_edges
  int64_t M_out = feats.size(1); //2 in example
  
  int64_t levels = weight_index.size(2);
  auto out = at::zeros({size_scatter_out,levels* M_out}, feats.options());
  auto S = 8;
  auto edge_index_data = edge_index.data_ptr<int64_t>();
  auto scatter_index_data = scatter_index.data_ptr<int64_t>();

  auto weight_index_data = weight_index.data_ptr<int64_t>();
  auto primes_data = primes.data_ptr<int64_t>();
  auto offsets_data = offsets.data_ptr<int64_t>();
  auto factors_data = factors.data_ptr<bool>();
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(feats.scalar_type(), "multispline_fused_fw", [&] {
    auto kernel_size_data = kernel_size.data_ptr<int64_t>();
    auto is_open_spline_data = is_open_spline.data_ptr<uint8_t>();
    auto pseudo_data = pseudo.data_ptr<scalar_t>();
    auto basis_data = basis.data_ptr<scalar_t>();
    auto f_data = feats.data_ptr<scalar_t>();
    auto out_data = out.data_ptr<scalar_t>();
    
    multispline_fused_fw_kernel<scalar_t>
        <<<BLOCKS(E*M_out*levels), THREADS, 0, stream>>>(
            f_data, pseudo_data, kernel_size_data, out_data, is_open_spline_data,
            scatter_index_data, edge_index_data,
            M_out,size_scatter_out, basis_data, weight_index_data,levels, total_size, E*M_out*levels, S, E, primes_data, offsets_data, factors_data
            );
    
  });
  return out;
}



template <typename scalar_t, int64_t S>
__global__ void
multispline_fused_bw_kernel(scalar_t *grad_feat,const scalar_t *grad_out  ,const scalar_t *pseudo, const int64_t *kernel_size,
                             const uint8_t *is_open_spline, const int64_t *scatter_index, 
                             int64_t E, int D, long M_out, int64_t N,
                             const scalar_t *basis, const int64_t *weight_index, int64_t levels, int total_size, int64_t numel, const int64_t *edge_index,
                              int64_t *primes, int64_t *offsets, const scalar_t *feats, const bool *factors)
{
  // unsigned int primes[16] = {0, 0, 0, 0, 0, 0, 715902911, 729070343, 919548613, 449542579, 530997011, 314167211, 146148241, 458291711, 225061747, 385280261};
  // unsigned int offsets[16] = {8, 27, 64, 125, 216, 343, 512,729, 1000,1331,1728,2197,2744,3375,4096,4913 };
  // const int64_t m_out = thread_idx % M_out;
  // const int64_t level = (thread_idx %(M_out*levels))/ M_out;
  const int64_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t e = (thread_idx / (levels*M_out));
  const int64_t m_out = (thread_idx % (M_out));
  // const int64_t level = (thread_idx %(M_out*levels))/ M_out;
  const int64_t level = (thread_idx % (levels*M_out))/M_out;
  
  if (thread_idx < numel) {
    // #pragma unroll(M_out)
    // for (int64_t m_out = 0; m_out < M_out; m_out++){
      // atomAdd(grad_out,e*M_out+m_out, numel, grad_scatter_out[sample_index[e]*M_out+m_out]);
      
      auto g = grad_out[scatter_index[e] * levels* M_out + levels*m_out+level];
      // auto confval = confidence[edge_index[e]*levels+level];
      // unsigned int primes[7] = {1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737};
      // int kl = kernel_size[3*level];
      // int pc[3];
      #pragma unroll(S)
      for (int64_t s = 0; s < S; s++) {

        auto b = basis[e*S*levels  + levels * s + level];
        auto wi = weight_index[e*S*levels  + levels * s + level];
        
        // atomAdd(grad_feat,level * total_size * M_out + wi * M_out + m_out,total_size*levels*M_out, b*g);
        // atomAdd(grad_feat,M_out* levels* (((edge_index[e]*offsets[level]+wi)*(primes[level]*factors[level]+!factors[level]))&(total_size-1)) + m_out*levels +level,total_size*levels*M_out, b*g);
        atomAdd(grad_feat,M_out* levels* (wi) + m_out*levels +level,total_size*levels*M_out, b*g);
        // atomAdd(grad_conf,edge_index[e]*levels+level, E*levels, b*g*feats[M_out* levels* (((edge_index[e]*offsets[level]+wi)^(primes[s]*factors[level]))&(total_size-1)) + m_out*levels +level]);
        // atomAdd(grad_feat,M_out* levels* (wi) + m_out*levels +level,total_size*levels*M_out, b*g);
      }
      
    // }
  }
}

// add an int argument for kernel size
torch::Tensor multispline_fused_bw_cuda(torch::Tensor grad_out,
                                  torch::Tensor edge_index,torch::Tensor scatter_index,
                                  torch::Tensor pseudo, torch::Tensor kernel_size, torch::Tensor is_open_spline,
                                  torch::Tensor basis, torch::Tensor weight_index, 
                                  int64_t total_size, torch::Tensor primes, torch::Tensor offsets,  torch::Tensor feats,
                                  torch::Tensor factors)
{
  
  
  cudaSetDevice(grad_out.get_device());
  int64_t levels = weight_index.size(2);
  int64_t E = edge_index.size(0);

  auto D = 3;
  int64_t M_out = grad_out.size(1)/levels; 
  // auto outsize = grad_feats.size(0);
  auto scatter_index_data = scatter_index.data_ptr<int64_t>();
  auto size_scatter_out = grad_out.size(0);
  auto primes_data = primes.data_ptr<int64_t>();
  auto offsets_data = offsets.data_ptr<int64_t>();
  auto weight_index_data = weight_index.data_ptr<int64_t>();
  auto grad_feats = at::zeros({total_size,M_out,levels}, grad_out.options());

  // auto grad_conf = at::zeros({pts,levels}, grad_out.options());
  auto edge_index_data = edge_index.data_ptr<int64_t>();
  auto factors_data = factors.data_ptr<bool>();
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_out.scalar_type(), "multispline_fused_bw", [&] {
    auto kernel_size_data = kernel_size.data_ptr<int64_t>();
    auto is_open_spline_data = is_open_spline.data_ptr<uint8_t>();
    auto pseudo_data = pseudo.data_ptr<scalar_t>();
    auto grad_out_data = grad_out.data_ptr<scalar_t>();
    auto grad_feat_data = grad_feats.data_ptr<scalar_t>();
    auto basis_data = basis.data_ptr<scalar_t>();
    auto feat_data = feats.data_ptr<scalar_t>();
    multispline_fused_bw_kernel<scalar_t, 8>
        <<<BLOCKS(E*M_out*levels), THREADS, 0, stream>>>(
            grad_feat_data,grad_out_data, pseudo_data, kernel_size_data,is_open_spline_data,
            scatter_index_data,E,D,
            M_out,size_scatter_out, basis_data, weight_index_data,levels, total_size, E*M_out*levels, edge_index_data,
            primes_data, offsets_data, feat_data, factors_data
            );
    
  });
  return grad_feats;
}



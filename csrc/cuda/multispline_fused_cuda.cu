#include "multispline_fused_cuda.h"

#include <ATen/cuda/CUDAContext.h>

#include "atomics.cuh"
#include "utils.cuh"

using namespace at;

#include <ATen/native/cuda/KernelUtils.cuh>
#define THREADS 1024
#define BLOCKS(N) (N + THREADS - 1) / THREADS



template <typename scalar_t>
__global__ void
multispline_fused_fw_kernel(const scalar_t *f, const scalar_t *pseudo, const int64_t *kernel_size, scalar_t *output,
                      const uint8_t *is_open_spline, const int64_t *scatter_index, const int64_t *edge_index,
                      int64_t M_out,  int64_t N, const scalar_t *basis, const int64_t *weight_index, const scalar_t *xyz, 
                      int64_t levels, const int64_t *resolution, int64_t log2_hashmap_size, const scalar_t *cellsize, long numel, int64_t S
                       )
{
  
  // Variables needed are S, pseudo, kernel size, open_spline, basis?
  const int64_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t e = thread_idx / (levels*M_out);
  const int64_t m_out = thread_idx % M_out;
  const int64_t level = (thread_idx %(M_out*levels))/ M_out;

  if (thread_idx < numel) {
    // unsigned int primes[7] = {1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737};
    // int kl = kernel_size[3*level];
    // int pc[3];
    #pragma unroll(8)
    for (int64_t s = 0; s < S; s++) {
      // unsigned int hashed_coords[3];
      
      // pc[0] = weight_index[e*S+s]%kl;
      // pc[1] = (weight_index[e*S+s]/kl)%kl;
      // pc[2] = weight_index[e*S+s]/(kl*kl);
      
      

      // unsigned int xor_result = 0;
      // #pragma unroll(3)
      // for (int i = 0; i < 3; i++) {
      //   auto coord = pc[i]*cellsize[level] + xyz[e * 3 + i] - 0.5*cellsize[0];
      //   hashed_coords[i] = static_cast<unsigned int>(floor(coord * resolution[level]));
      //   xor_result ^= hashed_coords[i] * primes[i];
      // }
      // unsigned int hashed_index = xor_result & ((1 << log2_hashmap_size) - 1);



      auto b = basis[e*S*levels+s*levels+level];
      auto wi = weight_index[e*S*levels+s*levels+level];
      auto interm=f[level * ((1 << log2_hashmap_size)) * M_out + wi * M_out + m_out]*b;

      atomAdd(output,scatter_index[e]*levels*M_out+level*M_out+m_out,N*M_out*levels,interm);
      
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
                                  torch::Tensor basis, torch::Tensor weight_index, torch::Tensor xyz,
                                  torch::Tensor resolution, int64_t log2_hashmap_size, torch::Tensor cellsize) {
  

  cudaSetDevice(feats.get_device());

  // x has shape [E, M_out, K_size]
  auto E = weight_index.size(0); //num_edges
  auto M_out = feats.size(2); //2 in example
  
  
  


  auto levels = weight_index.size(2);
  auto out = at::zeros({size_scatter_out, M_out*levels}, feats.options());
  auto S = 8;
  auto edge_index_data = edge_index.data_ptr<int64_t>();
  auto scatter_index_data = scatter_index.data_ptr<int64_t>();

  auto weight_index_data = weight_index.data_ptr<int64_t>();
  auto resolution_data = resolution.data_ptr<int64_t>();
  

  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(feats.scalar_type(), "multispline_fused_fw", [&] {
    auto kernel_size_data = kernel_size.data_ptr<int64_t>();
    auto is_open_spline_data = is_open_spline.data_ptr<uint8_t>();
    auto pseudo_data = pseudo.data_ptr<scalar_t>();
    auto basis_data = basis.data_ptr<scalar_t>();
    auto f_data = feats.data_ptr<scalar_t>();
    auto out_data = out.data_ptr<scalar_t>();
    auto xyz_data = xyz.data_ptr<scalar_t>();
    auto cellsize_data = cellsize.data_ptr<scalar_t>();
    multispline_fused_fw_kernel<scalar_t>
        <<<BLOCKS(E*M_out*levels), THREADS, 0, stream>>>(
            f_data, pseudo_data, kernel_size_data, out_data, is_open_spline_data,
            scatter_index_data, edge_index_data,
            M_out,size_scatter_out, basis_data, weight_index_data, xyz_data,levels, resolution_data, log2_hashmap_size, cellsize_data, E*M_out*levels, S
            );
    
  });
  return out;
}



template <typename scalar_t, int64_t S>
__global__ void
multispline_fused_bw_kernel(scalar_t *grad_feat,const scalar_t *grad_out ,const scalar_t *pseudo, const int64_t *kernel_size,
                             const uint8_t *is_open_spline, const int64_t *sample_index, 
                             int64_t E, int D, long M_out, int64_t N,
                             const scalar_t *basis, const int64_t *weight_index, const scalar_t *xyz, int64_t levels,
                             int64_t *resolution, int log2_hashmap_size, const scalar_t *cellsize, int64_t numel)
{

  const int64_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t e = thread_idx / (levels*M_out);
  const int64_t m_out = thread_idx % M_out;
  const int64_t level = (thread_idx %(M_out*levels))/ M_out;

  
  if (thread_idx < numel) {
    // #pragma unroll(M_out)
    // for (int64_t m_out = 0; m_out < M_out; m_out++){
      // atomAdd(grad_out,e*M_out+m_out, numel, grad_scatter_out[sample_index[e]*M_out+m_out]);
      auto g = grad_out[sample_index[e]*M_out+m_out];
      // unsigned int primes[7] = {1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737};
      // int kl = kernel_size[3*level];
      // int pc[3];
      #pragma unroll(S)
      for (int64_t s = 0; s < S; s++) {
        // unsigned int hashed_coords[3];
        
        // pc[0] = weight_index[e*S+s]%kl;
        // pc[1] = (weight_index[e*S+s]/kl)%kl;
        // pc[2] = weight_index[e*S+s]/(kl*kl);
        
        

        // unsigned int xor_result = 0;
        // #pragma unroll(3)
        // for (int i = 0; i < 3; i++) {
        //   auto coord = pc[i]*cellsize[level] + xyz[e * 3 + i] - 0.5*cellsize[0];
        //   hashed_coords[i] = static_cast<unsigned int>(floor(coord * resolution[level]));
        //   xor_result ^= hashed_coords[i] * primes[i];
        // }
        // unsigned int hashed_index = xor_result & ((1 << log2_hashmap_size) - 1);

        auto b = basis[e*S*levels+s*levels+level];
        auto wi = weight_index[e*S*levels+s*levels+level];
        atomAdd(grad_feat,level * (1 << log2_hashmap_size) * M_out + wi * M_out + m_out,(1 << log2_hashmap_size)*levels*M_out, b*g);
        
      }
    // }
  }
}

// add an int argument for kernel size
torch::Tensor multispline_fused_bw_cuda(torch::Tensor grad_feats, torch::Tensor grad_out,
                                  torch::Tensor edge_index,torch::Tensor scatter_index,
                                  torch::Tensor pseudo, torch::Tensor kernel_size, torch::Tensor is_open_spline,
                                  torch::Tensor basis, torch::Tensor weight_index, torch::Tensor xyz,
                                  torch::Tensor resolution, int64_t log2_hashmap_size, torch::Tensor cellsize)
{
  
  
  cudaSetDevice(grad_out.get_device());
  auto levels = weight_index.size(2);
  auto E = weight_index.size(0);

  auto D = 3;
  auto M_out = grad_out.size(1) / levels; //2 in example\
  auto outsize = grad_feats.size(0);
  
  auto scatter_index_data = scatter_index.data_ptr<int64_t>();
  auto size_scatter_out = grad_out.size(0);
  
  auto weight_index_data = weight_index.data_ptr<int64_t>();
  auto resolution_data = resolution.data_ptr<int64_t>();
  

  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_out.scalar_type(), "multispline_fused_bw", [&] {
    auto kernel_size_data = kernel_size.data_ptr<int64_t>();
    auto is_open_spline_data = is_open_spline.data_ptr<uint8_t>();
    auto pseudo_data = pseudo.data_ptr<scalar_t>();
    auto grad_out_data = grad_out.data_ptr<scalar_t>();
    auto grad_feat_data = grad_feats.data_ptr<scalar_t>();
    auto basis_data = basis.data_ptr<scalar_t>();
    auto xyz_data = xyz.data_ptr<scalar_t>();
    auto cellsize_data = cellsize.data_ptr<scalar_t>();
    multispline_fused_bw_kernel<scalar_t, 8>
        <<<BLOCKS(E*M_out*levels), THREADS, 0, stream>>>(
            grad_feat_data,grad_out_data, pseudo_data, kernel_size_data,is_open_spline_data,
            scatter_index_data,E,D,
            M_out,size_scatter_out, basis_data, weight_index_data, xyz_data,levels, resolution_data, log2_hashmap_size, cellsize_data, E*M_out*levels
            );
    
  });
  return grad_feats;
}

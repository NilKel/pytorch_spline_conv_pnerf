#include "weighting_mod_cuda.h"
using namespace at;
#include <ATen/cuda/CUDAContext.h>

#include "atomics.cuh"
#include "utils.cuh"
#include <ATen/native/cuda/KernelUtils.cuh>
#define THREADS 1024
#define BLOCKS(N) (N + THREADS - 1) / THREADS

template <typename scalar_t>
__global__ void
spline_weighting_fw_kernel(const scalar_t *x, 
                           const scalar_t *basis, const int64_t *weight_index,
                           scalar_t *out, int64_t E,
                           int64_t M_out, int64_t S, int64_t K, int64_t numel) {

  const int64_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  const int64_t e = thread_idx / M_out;
  //if distributed on all threads, e is when we partition by M_out i.e. total elements in partitions of 32. Sort 
  //of like processing all elements 32 times. e indicates which of the edges it is

  const int64_t m_out = thread_idx % M_out;
  // m_out is the remainder, so for one of the dimensions, it indicates which dimension we are considering for our element
  //print the shape of x. This is a cuda file.
 
  //print E,S,M_out
  if (thread_idx < numel) {
    //v is the output
    scalar_t v = (scalar_t)0.;
    //S is size of kernel or total number of control values for one spline grid
    for (ptrdiff_t s = 0; s < S; s++) {
      //dimensions of basis and weight index are [number of edges, number of control values]
      const scalar_t b = basis[e * S + s];

      const int64_t wi = weight_index[e * S + s];
      //b and wi iterate over all kernel values for one edge
      
      // for (int64_t m_in = 0; m_in < M_in; m_in++) {
        // Weight has shape [kernel size, M_in, M_out]
        // We want a shape of [E, M_out, kernel size]
      //print the value of wi each iteration


      
      scalar_t tmp = x[e * K * M_out  + wi * M_out + m_out];
      // scalar_t tmp = x[e * E * M_out  + e * M_out  + m_out];
      tmp *= b;
      v += tmp;
      // }
    }
    out[thread_idx] = v;
  }
}

torch::Tensor spline_weighting_fw_cuda(torch::Tensor x,
                                       torch::Tensor basis,
                                       torch::Tensor weight_index) {
  CHECK_CUDA(x);

  CHECK_CUDA(basis);
  CHECK_CUDA(weight_index);
  cudaSetDevice(x.get_device());

  // x has shape [E, M_out, K_size]
  // S has shape shape (degree+1)^dim, 2^3 here.
  auto E = x.size(0); //num_edges
  auto M_out = x.size(2); //2 in example
  auto S = basis.size(1); //8
  auto K = x.size(1);

  auto out = at::empty({E, M_out}, x.options()); //4,2

  auto weight_index_data = weight_index.data_ptr<int64_t>(); //6,8

  


  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "weighting_fw", [&] {
    auto x_data = x.data_ptr<scalar_t>();
    auto basis_data = basis.data_ptr<scalar_t>();
    auto out_data = out.data_ptr<scalar_t>();

    spline_weighting_fw_kernel<scalar_t>
        <<<BLOCKS(out.numel()), THREADS, 0, stream>>>(
            x_data, basis_data, weight_index_data, out_data, E,
            M_out, S, K, out.numel()
            );
  });
  return out;
}

template <typename scalar_t>
__global__ void
spline_weighting_bw_x_kernel(const scalar_t *grad_out,
                             const scalar_t *basis, const int64_t *weight_index,
                             scalar_t *grad_x, int64_t E,
                             int64_t M_out, int64_t S, int64_t K , int64_t numel) {
  // E,M_out, K
  const int64_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  // const int64_t e = (thread_idx % (M_out*E))/E;  KEM
  const int64_t e = thread_idx / (M_out);
  // const int64_t m_out = (thread_idx%(M_out*K))/K;
  // const int64_t e = (thread_idx % (E*M_out)) / M_out;
  const int64_t m_out = thread_idx % M_out;
  if (thread_idx < numel) {
    // scalar_t v = (scalar_t)0.;
    auto g = grad_out[e * M_out + m_out];
    for (int64_t s = 0; s < S; s++) {
      const scalar_t b = basis[e * S + s];
      const int64_t wi = weight_index[e * S + s];
      // for (int64_t m_out = 0; m_out < M_out; m_out++) {
        //In the backward pass we now originally have weight of shape [kernel size, M_in, M_out].
        //We want to replace it with grad out of shape [E, M_out, kernel size]
        // scalar_t tmp = weight[wi * M_out * M_in + m_out * M_in + m_in];
        // tmp *= b * grad_out[e * M_out + m_out];
        // auto v = b*g;
        // v+=b*g;
        // atomAdd(&grad_x[e * K * M_out + m_out * K + wi], v);
        // atomAdd(&grad_x[e * K * M_out  + wi * M_out + m_out], v);
        // atomAdd(&grad_x[e * K * M_out  + wi * M_out + m_out], b*g);
        atomAdd(grad_x,e * K * M_out  + wi * M_out + m_out,E*K*M_out, b*g);
        // grad_x[e * K * M_out  + wi * M_out + m_out]+= b*g;
        // grad_x[wi * E * M_out  + e * M_out + m_out]+= v;
      // }
    }
    // grad_x[thread_idx] = v;
  }
}
//add an int argument for kernel size
torch::Tensor spline_weighting_bw_x_cuda(torch::Tensor grad_out,
                                         torch::Tensor basis,
                                         torch::Tensor weight_index, int64_t kernel_size) {
  CHECK_CUDA(grad_out);
  CHECK_CUDA(basis);
  CHECK_CUDA(weight_index);
  cudaSetDevice(grad_out.get_device());


  auto E = grad_out.size(0);
  auto M_out = grad_out.size(1);
  auto S = basis.size(1);
  // auto W = weight_index.max().item<int64_t>().to<int>() + 1;
  auto K = kernel_size;

   auto grad_x = at::zeros({ E,K,M_out}, grad_out.options());


  auto weight_index_data = weight_index.data_ptr<int64_t>();

  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_out.scalar_type(), "weighting_bw_x", [&] {
    auto grad_out_data = grad_out.data_ptr<scalar_t>();
    auto basis_data = basis.data_ptr<scalar_t>();
    auto grad_x_data = grad_x.data_ptr<scalar_t>();

    spline_weighting_bw_x_kernel<scalar_t>
        <<<BLOCKS(grad_out.numel()), THREADS, 0, stream>>>(
            grad_out_data, basis_data, weight_index_data,
            grad_x_data, E, M_out, S, K, grad_out.numel());
    
  });
  return grad_x;
}





template <typename scalar_t>
__global__ void spline_weighting_bw_basis_kernel(
    const scalar_t *grad_out, const scalar_t *x, 
    const int64_t *weight_index, scalar_t *grad_basis, int64_t K, int64_t E,
    int64_t M_out, int64_t S,  int64_t numel) {

  const size_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t e = thread_idx / M_out;
  const int64_t m_out = thread_idx % M_out;

  if (thread_idx < numel) {
    const scalar_t g = grad_out[e * M_out + m_out];

    for (int64_t s = 0; s < S; s++) {
      
      const int64_t wi = weight_index[e * S + s];
      scalar_t v = g * x[e *K* M_out+ wi*K + m_out];
      
      // atomAdd(&grad_basis[e * S + s], v);
      atomAdd(grad_basis,e * S + s,E*S, v);
      // grad_basis[e * S + s]+=v;
    }
  }
}

torch::Tensor spline_weighting_bw_basis_cuda(torch::Tensor grad_out,
                                             torch::Tensor x,
                                             torch::Tensor weight_index) {
  CHECK_CUDA(grad_out);
  CHECK_CUDA(weight_index);
  cudaSetDevice(grad_out.get_device());


  auto E = grad_out.size(0);
  auto M_out = grad_out.size(1);
  auto S = weight_index.size(1);
  auto K = x.size(1);
  auto grad_basis = at::zeros({E, S}, grad_out.options());

  auto weight_index_data = weight_index.data_ptr<int64_t>();

  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "weighting_bw_basis", [&] {
    auto grad_out_data = grad_out.data_ptr<scalar_t>();
    auto grad_basis_data = grad_basis.data_ptr<scalar_t>();
    auto x_data = x.data_ptr<scalar_t>();
    spline_weighting_bw_basis_kernel<scalar_t>
        <<<BLOCKS(grad_out.numel()), THREADS, 0, stream>>>(
            grad_out_data, x_data, weight_index_data,
            grad_basis_data, K, E, M_out, S, grad_out.numel());
       
  });
  return grad_basis;
}

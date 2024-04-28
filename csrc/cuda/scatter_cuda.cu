#include "scatter_cuda.h"

#include <ATen/cuda/CUDAContext.h>

#include "atomics.cuh"
#include "utils.cuh"

#define THREADS 1024
#define BLOCKS(N) (N + THREADS - 1) / THREADS

template <typename scalar_t>
__global__ void
scatter_fw_kernel(const scalar_t *out, 
                           const int64_t *sample_index,
                           scalar_t *scatter_out,
                           int64_t M_out, int64_t thresh, int64_t size_scatter_out, int64_t numel) {

  const int64_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;  
  const int64_t e = thread_idx / M_out;
  const int64_t m_out = thread_idx % M_out;

  if (thread_idx < numel) {
    // if(sample_index[e]<size_scatter_out){
        // scatter_out[sample_index[e]*M_out+m_out] += out[thread_idx]; 
        atomAdd(scatter_out,sample_index[e]*M_out+m_out,size_scatter_out*M_out, out[thread_idx]);
    // }
  
  }
}

torch::Tensor scatter_fw_cuda(torch::Tensor out,
                        int64_t size_scatter_out,
                        torch::Tensor sample_ind) {
  CHECK_CUDA(out);

  CHECK_CUDA(sample_ind);
  cudaSetDevice(out.get_device());
  auto thresh = sample_ind.size(0);
  auto M_out = out.size(1); //2 in example
//   printf("M_out is %d\n",M_out);
//   printf("full size of out is %d\n",out.numel());
//   printf("number of output edges is %d\n",out.size(0));
//   printf("size_scatter_out is %d\n",size_scatter_out);
//   printf("max value element of sample index is %d\n",sample_ind.max().item<int64_t>());

  auto scatter_out = at::zeros({size_scatter_out,M_out}, out.options()); //4,2

  auto sample_ind_data = sample_ind.data_ptr<int64_t>(); //6,8

  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,out.scalar_type(), "scatter_fw", [&] {
    auto out_data = out.data_ptr<scalar_t>();
    auto scatter_out_data = scatter_out.data_ptr<scalar_t>();

    scatter_fw_kernel<scalar_t>
        <<<BLOCKS(out.numel()), THREADS, 0, stream>>>(
            out_data,sample_ind_data, scatter_out_data,
            M_out,thresh, size_scatter_out, out.numel()
            );
  });
  return scatter_out;
}

template <typename scalar_t>
__global__ void
scatter_bw_kernel(scalar_t *grad_out, 
                           const int64_t *sample_index,
                           const scalar_t *grad_scatter_out,
                           int64_t M_out, int64_t numel) {

  const int64_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;  
  const int64_t e = thread_idx / M_out;
  const int64_t m_out = thread_idx % M_out;

  if (thread_idx < numel) {
   
      // atomAdd(&grad_out[thread_idx], grad_scatter_out[sample_index[e]*M_out+m_out]);
      atomAdd(grad_out,thread_idx, numel, grad_scatter_out[sample_index[e]*M_out+m_out]);
      // grad_out[thread_idx] = grad_scatter_out[sample_index[e]*M_out+m_out];
      
  }
}
//add an int argument for kernel size
torch::Tensor scatter_bw_cuda(torch::Tensor grad_scatter_out,
                        int64_t size_out,
                        torch::Tensor sample_ind) {

  CHECK_CUDA(sample_ind);
  cudaSetDevice(grad_scatter_out.get_device());

  auto M_out = grad_scatter_out.size(1); //2 in example
  //print if there are any nan values in grad_scatter_out
//   printf("number of nan values in grad_scatter_out is %d\n",torch::isnan(grad_scatter_out).sum().item<int64_t>());

  auto grad_out = at::zeros({size_out,M_out}, grad_scatter_out.options()); //4,2

  auto sample_ind_data = sample_ind.data_ptr<int64_t>(); //6,8

  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,grad_scatter_out.scalar_type(), "scatter_bw", [&] {
    auto grad_out_data = grad_out.data_ptr<scalar_t>();
    auto grad_scatter_out_data = grad_scatter_out.data_ptr<scalar_t>();

    scatter_bw_kernel<scalar_t>
        <<<BLOCKS(grad_out.numel()), THREADS, 0, stream>>>(
            grad_out_data,sample_ind_data, grad_scatter_out_data,
            M_out, grad_out.numel()
            );
  });
  return grad_out;
}


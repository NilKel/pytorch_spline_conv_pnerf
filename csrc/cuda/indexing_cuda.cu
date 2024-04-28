#include "indexing_cuda.h"

#include <ATen/cuda/CUDAContext.h>

#include "atomics.cuh"
#include "utils.cuh"

#define THREADS 1024
#define BLOCKS(N) (N + THREADS - 1) / THREADS

template <typename scalar_t>
__global__ void
indexing_fw_kernel(const scalar_t *x, 
                           const int64_t *point_indexing,
                           scalar_t *indexing_out,int64_t K,
                           int64_t M_out,  int64_t size_indexing_out, int64_t numel) {

  const int64_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;  
  const int64_t e = thread_idx / (M_out*K);
  const int64_t k = (thread_idx %(M_out*K))/M_out;
  const int64_t m_out = thread_idx % M_out;

  if (thread_idx < numel) {
    // if(point_indexing[e]<size_indexing_out){
        // indexing_out[point_indexing[e]*M_out+m_out] += x[thread_idx]; 
        indexing_out[thread_idx] = x[point_indexing[e]*K*M_out+k*M_out +m_out];
    // }
  
  }
}

torch::Tensor indexing_fw_cuda(torch::Tensor x,
                        int64_t size_indexing_out,
                        torch::Tensor point_ind) {
  CHECK_CUDA(x);

  CHECK_CUDA(point_ind);
  cudaSetDevice(x.get_device());
  auto M_out = x.size(2);
  auto K = x.size(1); //2 in example
//   printf("M_out is %d\n",M_out);
//   printf("full size of x is %d\n",x.numel());
//   printf("number of output edges is %d\n",x.size(0));
//   printf("size_indexing_out is %d\n",size_indexing_out);
//   printf("max value element of sample indexing is %d\n",point_ind.max().item<int64_t>());

  auto indexing_out = at::empty({size_indexing_out,K, M_out}, x.options()); //4,2

  auto point_ind_data = point_ind.data_ptr<int64_t>(); //6,8

  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "indexing_fw", [&] {
    auto out_data = x.data_ptr<scalar_t>();
    auto indexing_out_data = indexing_out.data_ptr<scalar_t>();

    indexing_fw_kernel<scalar_t>
        <<<BLOCKS(indexing_out.numel()), THREADS, 0, stream>>>(
            out_data,point_ind_data, indexing_out_data,K,
            M_out, size_indexing_out, indexing_out.numel()
            );
  });
  return indexing_out;
}

template <typename scalar_t>
__global__ void
indexing_bw_kernel(scalar_t *grad_x, 
                           const int64_t *point_indexing,
                           const scalar_t *grad_indexing_out,int64_t K,
                           int64_t M_out, int64_t numel, int64_t grad_x_numel) {

  const int64_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;  
  const int64_t e = thread_idx / (M_out*K);
  const int64_t k = (thread_idx %(M_out*K))/M_out;
  const int64_t m_out = thread_idx % M_out;

  if (thread_idx < numel) {
   
      // atomAdd(&grad_x[point_indexing[e]*K*M_out+k*M_out +m_out], grad_indexing_out[thread_idx]);
      atomAdd(grad_x,point_indexing[e]*K*M_out+k*M_out +m_out,grad_x_numel,grad_indexing_out[thread_idx]);
      
  }
}
//add an int argument for kernel size
torch::Tensor indexing_bw_cuda(torch::Tensor grad_indexing_out,
                        int64_t size_x,
                        torch::Tensor point_ind, torch::Tensor grad_x) {

  CHECK_CUDA(point_ind);
  cudaSetDevice(grad_indexing_out.get_device());
  auto M_out = grad_indexing_out.size(2);
  auto K = grad_indexing_out.size(1);
   //2 in example
  //print if there are any nan values in grad_indexing_out
//   printf("number of nan values in grad_indexing_out is %d\n",torch::isnan(grad_indexing_out).sum().item<int64_t>());

//   // auto grad_x = at::zeros({size_x,K,M_out}, grad_indexing_out.options()); //4,2
// // make grad x same type as grad indexing out. Use the given grad x
  

  auto point_ind_data = point_ind.data_ptr<int64_t>(); //6,8
  //print the type of grad_indexing_out
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_indexing_out.scalar_type(), "indexing_bw", [&] {
    auto grad_x_data = grad_x.data_ptr<scalar_t>();
    auto grad_indexing_out_data = grad_indexing_out.data_ptr<scalar_t>();

    indexing_bw_kernel<scalar_t>
        <<<BLOCKS(grad_indexing_out.numel()), THREADS, 0, stream>>>(
            grad_x_data,point_ind_data, grad_indexing_out_data,K,
            M_out, grad_indexing_out.numel(),grad_x.numel()
            );
  });
  return grad_x;
}


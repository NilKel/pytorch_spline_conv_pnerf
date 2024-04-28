#include "outscatter_cuda.h"

#include <ATen/cuda/CUDAContext.h>

#include "atomics.cuh"
#include "utils.cuh"

#define THREADS 1024
#define BLOCKS(N) (N + THREADS - 1) / THREADS

template <typename scalar_t>
__global__ void
outscatter_fw_kernel(const scalar_t *input, 
                           const int64_t *sample_index,
                           scalar_t *output,
                           int64_t M_in, int64_t M_out, int64_t numel, int64_t output_size) {

  const int64_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;  
  const int64_t e = thread_idx / M_in;
  const int64_t m_out = thread_idx % M_in;

  if (thread_idx < numel) {
        // atomAdd(&output[sample_index[e]*M_out+m_out], input[thread_idx]);
        atomAdd(output, sample_index[e]*M_out+m_out, output_size, input[thread_idx]);
  }
}

torch::Tensor outscatter_fw_cuda(torch::Tensor input,
                        torch::Tensor output,
                        torch::Tensor sample_ind) {
  CHECK_CUDA(input);

  CHECK_CUDA(sample_ind);
  cudaSetDevice(input.get_device());
  
  auto M_in = input.size(1);
  auto M_out = output.size(1);
  auto sample_ind_data = sample_ind.data_ptr<int64_t>();
  auto stream = at::cuda::getCurrentCUDAStream();
  
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "scatter_fw", [&] {
    auto input_data = input.data_ptr<scalar_t>();
    auto output_data = output.data_ptr<scalar_t>();

    outscatter_fw_kernel<scalar_t>
        <<<BLOCKS(input.numel()), THREADS, 0, stream>>>(
            input_data,sample_ind_data, output_data,
            M_in,M_out,  input.numel(),output.numel()
            );
  });
  return output;
}

template <typename scalar_t>
__global__ void
outscatter_bw_kernel(scalar_t *grad_input, 
                           const int64_t *sample_index,
                           const scalar_t *grad_output,
                           int64_t M_in, int64_t M_out, int64_t numel) {

  const int64_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;  
  const int64_t e = thread_idx / M_in;
  const int64_t m_out = thread_idx % M_in;

  if (thread_idx < numel) {
   
      // atomAdd(&grad_input[thread_idx], grad_output[sample_index[e]*M_out+m_out]);
      atomAdd(grad_input, thread_idx, numel, grad_output[sample_index[e]*M_out+m_out]);
      
  }
}
//add an int argument for kernel size
torch::Tensor outscatter_bw_cuda(torch::Tensor grad_output,
                        int64_t size_input,
                        int64_t M_in,
                        torch::Tensor pixel_ind) {

  CHECK_CUDA(pixel_ind);
  cudaSetDevice(grad_output.get_device());

  auto M_out = grad_output.size(1); //2 in example
  //print if there are any nan values in grad_output
//   printf("number of nan values in grad_output is %d\n",torch::isnan(grad_output).sum().item<int64_t>());

  auto grad_input = at::zeros({size_input,M_in}, grad_output.options()); //4,2

  auto pixel_ind_data = pixel_ind.data_ptr<int64_t>(); //6,8

  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_output.scalar_type(), "scatter_bw", [&] {
    auto grad_input_data = grad_input.data_ptr<scalar_t>();
    auto grad_output_data = grad_output.data_ptr<scalar_t>();

    outscatter_bw_kernel<scalar_t>
        <<<BLOCKS(grad_input.numel()), THREADS, 0, stream>>>(
            grad_input_data,pixel_ind_data, grad_output_data,
            M_in,M_out, grad_input.numel()
            );
  });
  return grad_input;
}


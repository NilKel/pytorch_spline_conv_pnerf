#include "multispline_basis_cuda.h"

#include <ATen/cuda/CUDAContext.h>

#include "utils.cuh"

#define THREADS 1024
#define BLOCKS(N) (N + THREADS - 1) / THREADS

template <typename scalar_t, int64_t degree> struct Basis {
  static inline __device__ scalar_t forward(scalar_t v, int64_t k_mod) {
    if (degree == 1) {
      return 1. - v - k_mod + 2. * v * k_mod;
    } else if (degree == 2) {
      if (k_mod == 0)
        return 0.5 * v * v - v + 0.5;
      else if (k_mod == 1)
        return -v * v + v + 0.5;
      else
        return 0.5 * v * v;
    } else if (degree == 3) {
      if (k_mod == 0)
        return (1. - v) * (1. - v) * (1. - v) / 6.;
      else if (k_mod == 1)
        return (3. * v * v * v - 6. * v * v + 4.) / 6.;
      else if (k_mod == 2)
        return (-3. * v * v * v + 3. * v * v + 3. * v + 1.) / 6.;
      else
        return v * v * v / 6.;
    } else {
      return (scalar_t)-1.;
    }
  }

  static inline __device__ scalar_t backward(scalar_t v, int64_t k_mod) {
    if (degree == 1) {
      return 2 * k_mod - 1;
    } else if (degree == 2) {
      if (k_mod == 0)
        return v - 1.;
      else if (k_mod == 1)
        return -2. * v + 1.;
      else
        return v;
    } else if (degree == 3) {
      if (k_mod == 0)
        return (-v * v + 2. * v - 1.) / 2.;
      else if (k_mod == 1)
        return (3. * v * v - 4. * v) / 2.;
      else if (k_mod == 2)
        return (-3. * v * v + 2. * v + 1.) / 2.;
      else
        return v * v / 2.;
    } else {
      return (scalar_t)-1.;
    }
  }
};


template <typename scalar_t, int64_t degree>
__global__ void
multispline_basis_fw_kernel(const scalar_t *pseudo, const int64_t *kernel_size,
                       const uint8_t *is_open_spline, scalar_t *basis,
                       int64_t *weight_index, int64_t E, int64_t D, int64_t S,
                       int64_t numel, int64_t levels, const int64_t *resolution, int64_t log2_hashmap_size, int64_t cellsize,
                        const scalar_t *xyz, const int64_t *point_index, const int64_t *primes, const int64_t *offsets) {
  
  const int64_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t e = thread_idx / S;
  const int64_t s = thread_idx % S;
  if (thread_idx < numel) {
    
    unsigned long point_idx = static_cast<unsigned long>(point_index[e]);
    
    // unsigned int primes[16] = {0, 0, 0, 0, 0, 0, 715902911, 729070343, 919548613, 449542579, 530997011, 314167211, 146148241, 458291711, 225061747, 385280261};
    // unsigned int offsets[16] = {8, 27, 64, 125, 216, 343, 512,729, 1000,1331,1728,2197,2744,3375,4096,4913 };
    
    #pragma unroll(16)
    for(int64_t level = 0; level<levels; level++){
      int64_t k = s, wi = 0, wi_offset = 1;
      scalar_t b = (scalar_t)1.;
      #pragma unroll(3)
      for (int64_t d = 0; d < D; d++) {

        
        const int64_t k_mod = k % (degree + 1);
        k /= degree + 1;
        scalar_t v;
       
        v = pseudo[e * D * levels + d*levels + level];
        // array of wi here and array of offsets used
        // wi += (((int64_t)v + k_mod) % kernel_size[level*D+d]) * wi_offset;
        wi = wi ^ ((((int64_t)v + k_mod) % kernel_size[level*D+d])*primes[d]);
        wi_offset *= kernel_size[level*D+d];

        v -= floor(v);
        v = Basis<scalar_t, degree>::forward(v, k_mod);
        b *= v;
      }
      
      // unsigned int temp = ((wi + point_idx *offsets[level]) *(primes[level])& ((1 << log2_hashmap_size) - 1));
      unsigned int temp = (wi ^ point_idx*primes[4]) & ((1 << log2_hashmap_size) - 1);
      unsigned int temp = (wi ^ point_idx) & ((1 << log2_hashmap_size) - 1);
      // unsigned int temp = (((wi + point_idx *offsets[level])) ^ primes[level]);
      // // temp = temp % 4294967295;
      // hashed_coords = (temp ) & ((1 << log2_hashmap_size) - 1);
      
      basis[e*S*levels  + s * levels + level] = b;
      weight_index[e*S*levels  + s * levels + level] = temp;
    }
  }

}


// template <typename scalar_t, int64_t degree>
// __global__ void
// multispline_basis_fw_kernel(const scalar_t *pseudo, const int64_t *kernel_size,
//                        const uint8_t *is_open_spline, scalar_t *basis,
//                        int64_t *weight_index, int64_t E, int64_t D, int64_t S,
//                        int64_t numel, int64_t levels, const int64_t *resolution, int64_t log2_hashmap_size, int64_t cellsize,
//                         const scalar_t *xyz, const int64_t *point_index, const int64_t *primes, const int64_t *offsets) {
  
//   const int64_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
//   const int64_t e = thread_idx / S;
//   const int64_t s = thread_idx % S;
//   if (thread_idx < numel) {
    
    
   
    
//     int64_t k = s, wi_offset = 1;
//       // wi and wi offset are now arrays. so is basis.
//     #pragma unroll(3)
//     for (int64_t d = 0; d < D; d++) {

      
//       const int64_t k_mod = k % (degree + 1);
//       k /= degree + 1;
//       #pragma unroll(16)
//       for(int64_t level = 0; level<levels; level++){
//         scalar_t v;

//         v = pseudo[e * D * levels + d*levels + level];
        
//         // array of wi here and array of offsets used
//         weight_index[e*S*levels  + s * levels + level] += (((int64_t)v + k_mod) % kernel_size[level*D+d]) * wi_offset;
//         wi_offset *= kernel_size[level*D+d];

//         v -= floor(v);
//         v = Basis<scalar_t, degree>::forward(v, k_mod);
//         basis[e*S*levels  + s * levels + level] *= v;
//       }
     
//     }
//   }

// }


// template <typename scalar_t, int64_t degree>
// __global__ void
// multispline_basis_fw_kernel(const scalar_t *pseudo, const int64_t *kernel_size,
//                        const uint8_t *is_open_spline, scalar_t *basis,
//                        int64_t *weight_index, int64_t E, int64_t D, int64_t S,
//                        int64_t numel, int64_t levels, const int64_t *resolution, int64_t log2_hashmap_size, int64_t cellsize,
//                         const scalar_t *xyz, const int64_t *point_index, const int64_t *primes, const int64_t *offsets, const int64_t *kernel_power) {
  
//   const int64_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
//   const int64_t e = thread_idx / S;
//   const int64_t s = thread_idx % S;
//   if (thread_idx < numel) {
    
//     unsigned long point_idx = static_cast<unsigned long>(point_index[e]);

//     unsigned int hashed_coords = 0;
   
    
//     int64_t k = s, wi_offset_power = 0;
//       // wi and wi offset are now arrays. so is basis.
//     #pragma unroll(3)
//     for (int64_t d = 0; d < D; d++) {

      
//       const int64_t k_mod = k & (degree);
//       k = k >> 1;
//       #pragma unroll(16)
//       for(int64_t level = 0; level<levels; level++){
//         scalar_t v;

//         v = pseudo[e * D * levels + d*levels + level];
        
//         // array of wi here and array of offsets used
//         weight_index[e*S*levels  + s * levels + level] += (((int64_t)v + k_mod) & (kernel_size[level]-1)) << wi_offset_power;
//         wi_offset_power += kernel_power[level];

//         v -= floor(v);
//         v = Basis<scalar_t, degree>::forward(v, k_mod);
//         basis[e*S*levels  + s * levels + level] *= v;
//       }
     
//     }
//   }

// }


std::tuple<torch::Tensor, torch::Tensor>
multispline_basis_fw_cuda(torch::Tensor pseudo, torch::Tensor kernel_size,
                     torch::Tensor is_open_spline, int64_t degree, torch::Tensor resolution, int64_t log2_hashmap_size, int64_t cellsize,
                    torch::Tensor xyz, torch::Tensor point_index, torch::Tensor primes, torch::Tensor offsets) {
  CHECK_CUDA(pseudo);
  CHECK_CUDA(kernel_size);
  CHECK_CUDA(is_open_spline);
  cudaSetDevice(pseudo.get_device());

  // modulo condition on pseudo and kernel size
  CHECK_INPUT(is_open_spline.dim());
  CHECK_INPUT(pseudo.size(1) == is_open_spline.numel());
  
  int64_t E = pseudo.size(0);
  auto D = pseudo.size(1);
  auto levels = resolution.size(0);
  int64_t S = ((int64_t)(powf(degree + 1, D) + 0.5));
  auto primes_data = primes.data_ptr<int64_t>();
  auto offsets_data = offsets.data_ptr<int64_t>();
  auto basis = at::ones({E, S,levels}, pseudo.options());
  auto weight_index = at::zeros({E, S, levels}, kernel_size.options());
  auto resolution_data = resolution.data_ptr<int64_t>();
  auto kernel_size_data = kernel_size.data_ptr<int64_t>();
  // auto kernel_power_data = kernel_power.data_ptr<int64_t>();
  auto is_open_spline_data = is_open_spline.data_ptr<uint8_t>();
  auto weight_index_data = weight_index.data_ptr<int64_t>();
  auto point_index_data = point_index.data_ptr<int64_t>();
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(pseudo.scalar_type(), "basis_fw", [&] {
    auto pseudo_data = pseudo.data_ptr<scalar_t>();
    auto basis_data = basis.data_ptr<scalar_t>();
    auto xyz_data = xyz.data_ptr<scalar_t>();
    AT_DISPATCH_DEGREE_TYPES(degree, [&] {
      multispline_basis_fw_kernel<scalar_t, DEGREE>
          <<<BLOCKS(E*S), THREADS, 0, stream>>>(
              pseudo_data, kernel_size_data, is_open_spline_data, basis_data,
              weight_index_data, E, D, S, E*S,levels, resolution_data, log2_hashmap_size, cellsize, xyz_data, point_index_data, primes_data, offsets_data);
    });
  });

  return std::make_tuple(basis, weight_index);
}

template <typename scalar_t, int64_t degree>
__global__ void
multispline_basis_bw_kernel(const scalar_t *grad_basis, const scalar_t *pseudo,
                       const int64_t *kernel_size,
                       const uint8_t *is_open_spline, scalar_t *grad_pseudo,
                       int64_t E, int64_t D, int64_t S, int64_t numel, int64_t levels) {

  const int64_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t e = thread_idx / D;
  const int64_t d = thread_idx % D;

  if (thread_idx < numel) {
    scalar_t g = (scalar_t)0., tmp;
    for(int64_t level = 0; level < levels; level++){
        for (ptrdiff_t s = 0; s < S; s++) {
            int64_t k_mod = (s / (int64_t)(powf(degree + 1, d) + 0.5)) % (degree + 1);
            
            scalar_t v;
            v = pseudo[e * D * levels + d*levels + level];
            v -= floor(v);
            v = Basis<scalar_t, degree>::backward(v, k_mod);
            tmp = v;

            for (int64_t d_it = 1; d_it < D; d_it++) {
                const int64_t d_new = d_it - (d >= d_it);
                k_mod = (s / (int64_t)(powf(degree + 1, d_new) + 0.5)) % (degree + 1);
                v = pseudo[e * D * levels + d*levels + level];
                v -= floor(v);
                v = Basis<scalar_t, degree>::forward(v, k_mod);
                tmp *= v;
            }
            g += tmp * grad_basis[e * S * levels + s*levels +level];
            }
            g *= kernel_size[level*D+d] - degree * is_open_spline[d];
            grad_pseudo[thread_idx] = g;
        }
  }
}

torch::Tensor multispline_basis_bw_cuda(torch::Tensor grad_basis,
                                   torch::Tensor pseudo,
                                   torch::Tensor kernel_size,
                                   torch::Tensor is_open_spline,
                                   int64_t degree) {
  CHECK_CUDA(grad_basis);
  CHECK_CUDA(pseudo);
  CHECK_CUDA(kernel_size);
  CHECK_CUDA(is_open_spline);
  cudaSetDevice(grad_basis.get_device());

  CHECK_INPUT(grad_basis.size(0) == pseudo.size(0));
  CHECK_INPUT(is_open_spline.dim());
  CHECK_INPUT(pseudo.size(1) == is_open_spline.numel());


  auto E = pseudo.size(0);
  auto D = pseudo.size(1);
  auto S = grad_basis.size(1);
  auto levels = kernel_size.size(0);

  auto grad_pseudo = at::empty({E, D}, pseudo.options());

  auto kernel_size_data = kernel_size.data_ptr<int64_t>();
  auto is_open_spline_data = is_open_spline.data_ptr<uint8_t>();

  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(pseudo.scalar_type(), "basis_bw", [&] {
    auto grad_basis_data = grad_basis.data_ptr<scalar_t>();
    auto pseudo_data = pseudo.data_ptr<scalar_t>();
    auto grad_pseudo_data = grad_pseudo.data_ptr<scalar_t>();

    AT_DISPATCH_DEGREE_TYPES(degree, [&] {
      multispline_basis_bw_kernel<scalar_t, DEGREE>
          <<<BLOCKS(grad_pseudo.numel()), THREADS, 0, stream>>>(
              grad_basis_data, pseudo_data, kernel_size_data,
              is_open_spline_data, grad_pseudo_data, E, D, S,
              grad_pseudo.numel(),levels);
    });
  });

  return grad_pseudo;
}

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
                       int64_t numel, int levels, const int64_t *resolution, int64_t log2_hashmap_size, const scalar_t *cellsize, const scalar_t *xyz, const int64_t *point_index) {
  
  const int64_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  // const int64_t e = thread_idx / (S*levels);
  // const int64_t s = (thread_idx %(S*levels))/ levels;
  // const int64_t level = thread_idx % levels;
  const int64_t e = thread_idx / S;
  const int64_t s = thread_idx % S;
  if (thread_idx < numel) {
    int64_t k = s, wi1 = 0, wi2 = 0, wi3 = 0, wi4 = 0, wi_offset1 = 1, wi_offset2 = 1, wi_offset3 = 1, wi_offset4 = 1;
    scalar_t b1 = (scalar_t)1.;
    scalar_t b2 = (scalar_t)1.;
    scalar_t b3 = (scalar_t)1.;
    scalar_t b4 = (scalar_t)1.;
    unsigned int primes[7] = {1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737};



      #pragma unroll(3)
      for (int64_t d = 0; d < D; d++) {

        
        const int64_t k_mod = k % (degree + 1);
        k /= degree + 1;
        scalar_t v;
        
        auto v1 = pseudo[e * D + d]/2 + 0.5;
        auto v2 = pseudo[e * D + d];
        auto v3 = pseudo[e * D + d]*(0.75) + 0.25;
        auto v4 = pseudo[e * D + d];

        
        v1 *= kernel_size[d] - degree * is_open_spline[d];
        v2 *= kernel_size[D+d] - degree * is_open_spline[d];
        v3 *= kernel_size[2*D+d] - degree * is_open_spline[d];
        v4 *= kernel_size[3*D+d] - degree * is_open_spline[d];

        wi1 += (((int64_t)v1 + k_mod) % kernel_size[0*D+d]) * wi_offset1;
        wi2 += (((int64_t)v2 + k_mod) % kernel_size[1*D+d]) * wi_offset2;
        wi3 += (((int64_t)v3 + k_mod) % kernel_size[2*D+d]) * wi_offset3;
        wi4 += (((int64_t)v4 + k_mod) % kernel_size[3*D+d]) * wi_offset4;


        wi_offset1 *= kernel_size[0*D+d];
        wi_offset2 *= kernel_size[1*D+d];
        wi_offset3 *= kernel_size[2*D+d];
        wi_offset4 *= kernel_size[3*D+d];

        v1 -= floor(v1);
        v2 -= floor(v2);
        v3 -= floor(v3);
        v4 -= floor(v4);


        v1 = Basis<scalar_t, degree>::forward(v1, k_mod);
        v2 = Basis<scalar_t, degree>::forward(v2, k_mod);
        v3 = Basis<scalar_t, degree>::forward(v3, k_mod);
        v4 = Basis<scalar_t, degree>::forward(v4, k_mod);

        b1 *= v1;
        b2 *= v2;
        b3 *= v3;
        b4 *= v4;
      }
      
      unsigned int coord1 = wi1 + point_index[e];
      unsigned int coord2 = wi2 + 1 + point_index[e];
      unsigned int coord3 = wi3 + 2 + point_index[e];
      unsigned int coord4 = wi4 + 3 + point_index[e];
      unsigned int hashed_index1 = (coord1 ^ primes[0]) & ((1 << log2_hashmap_size) - 1);
      unsigned int hashed_index2 = (coord2 ^ primes[1]) & ((1 << log2_hashmap_size) - 1);
      unsigned int hashed_index3 = (coord3 ^ primes[2]) & ((1 << log2_hashmap_size) - 1);
      unsigned int hashed_index4 = (coord4 ^ primes[3]) & ((1 << log2_hashmap_size) - 1);

      basis[e*S*levels + s*levels + 0] = b1;
      basis[e*S*levels + s*levels + 1] = b2;
      basis[e*S*levels + s*levels + 2] = b3;
      basis[e*S*levels + s*levels + 3] = b4;
      
      weight_index[e*S*levels + s*levels + 0] = hashed_index1;
      weight_index[e*S*levels + s*levels + 1] = hashed_index2;
      weight_index[e*S*levels + s*levels + 2] = hashed_index3;
      weight_index[e*S*levels + s*levels + 3] = hashed_index4;
    }

  

}

std::tuple<torch::Tensor, torch::Tensor>
multispline_basis_fw_cuda(torch::Tensor pseudo, torch::Tensor kernel_size,
                     torch::Tensor is_open_spline, int64_t degree, torch::Tensor resolution, int64_t log2_hashmap_size, torch::Tensor cellsize, torch::Tensor xyz, torch::Tensor point_index) {
  CHECK_CUDA(pseudo);
  CHECK_CUDA(kernel_size);
  CHECK_CUDA(is_open_spline);
  cudaSetDevice(pseudo.get_device());

  // modulo condition on pseudo and kernel size
  CHECK_INPUT(is_open_spline.dim());
  CHECK_INPUT(pseudo.size(1) == is_open_spline.numel());
  
  auto E = pseudo.size(0);
  auto D = pseudo.size(1);
  int levels = kernel_size.size(0);
  auto S = ((int64_t)(powf(degree + 1, D) + 0.5));

  auto basis = at::empty({E, S,levels}, pseudo.options());
  auto weight_index = at::empty({E, S, levels}, kernel_size.options());
  auto resolution_data = resolution.data_ptr<int64_t>();
  auto kernel_size_data = kernel_size.data_ptr<int64_t>();
  auto is_open_spline_data = is_open_spline.data_ptr<uint8_t>();
  auto weight_index_data = weight_index.data_ptr<int64_t>();
  auto point_index_data = point_index.data_ptr<int64_t>();
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(pseudo.scalar_type(), "basis_fw", [&] {
    auto pseudo_data = pseudo.data_ptr<scalar_t>();
    auto basis_data = basis.data_ptr<scalar_t>();
    auto xyz_data = xyz.data_ptr<scalar_t>();
    auto cellsize_data = cellsize.data_ptr<scalar_t>();
    AT_DISPATCH_DEGREE_TYPES(degree, [&] {
      multispline_basis_fw_kernel<scalar_t, DEGREE>
          <<<BLOCKS(E*S), THREADS, 0, stream>>>(
              pseudo_data, kernel_size_data, is_open_spline_data, basis_data,
              weight_index_data, E, D, S, E*S,levels, resolution_data, log2_hashmap_size, cellsize_data, xyz_data, point_index_data);
    });
  });

  return std::make_tuple(basis, weight_index);
}

template <typename scalar_t, int64_t degree>
__global__ void
multispline_basis_bw_kernel(const scalar_t *grad_basis, const scalar_t *pseudo,
                       const int64_t *kernel_size,
                       const uint8_t *is_open_spline, scalar_t *grad_pseudo,
                       int64_t E, int64_t D, int64_t S, int64_t numel, int levels) {

  const int64_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t e = thread_idx / D;
  const int64_t d = thread_idx % D;

  if (thread_idx < numel) {
    scalar_t g = (scalar_t)0., tmp;
    for(int64_t level = 0; level < levels; level++){
        for (ptrdiff_t s = 0; s < S; s++) {
            int64_t k_mod = (s / (int64_t)(powf(degree + 1, d) + 0.5)) % (degree + 1);

            scalar_t v = pseudo[e * D + d];
            if(level%2 == 1){
              v = pseudo[e * D + d];
            }
            else if(level==0){
              v = pseudo[e * D + d]/2 + 0.5;
            }
            else{
              v = pseudo[e * D + d]*(0.75) + 0.25;
            }
            
            v *= kernel_size[level*D+d] - degree * is_open_spline[d];
            v -= floor(v);
            v = Basis<scalar_t, degree>::backward(v, k_mod);
            tmp = v;

            for (int64_t d_it = 1; d_it < D; d_it++) {
                const int64_t d_new = d_it - (d >= d_it);
                k_mod = (s / (int64_t)(powf(degree + 1, d_new) + 0.5)) % (degree + 1);
                v = pseudo[e * D + d_new];
                v *= kernel_size[level*D+d_new] - degree * is_open_spline[d_new];
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
  int levels = kernel_size.size(0);

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

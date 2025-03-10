#include <Python.h>
#include <torch/script.h>
#include <tuple>
#include <cuda_runtime.h>

#ifdef WITH_CUDA
#include "cuda/ray_intersect_cuda.h"  // Header for our CUDA kernel.
#endif

#ifdef _WIN32
#ifdef WITH_CUDA
PyMODINIT_FUNC PyInit__ray_intersect_cuda(void) { return NULL; }
#endif
#endif

// The CUDA forward function wrapper.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
ray_intersect_fw(torch::Tensor ray_origin, torch::Tensor ray_dirs, torch::Tensor inv_ray_dirs,
                 torch::Tensor candidate_ids, torch::Tensor start_indices,
                 torch::Tensor ray_counts, torch::Tensor min_bounds,
                 torch::Tensor max_bounds, int64_t M,
                 int64_t num_samples, int64_t conf_kernel_size,
                 torch::Tensor conf_grid) {
  if (ray_dirs.device().is_cuda()) {
#ifdef WITH_CUDA
    return ray_intersect_fw_cuda(ray_origin, ray_dirs, inv_ray_dirs, candidate_ids,
                                 start_indices, ray_counts, min_bounds,
                                 max_bounds, M, num_samples, conf_kernel_size, conf_grid);
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
  }
  AT_ERROR("ray_intersect only supports CUDA");
}

// Main exposed function.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
ray_intersect(torch::Tensor ray_origin, torch::Tensor ray_dirs, torch::Tensor inv_ray_dirs,
              torch::Tensor candidate_ids, torch::Tensor start_indices,
              torch::Tensor ray_counts, torch::Tensor min_bounds,
              torch::Tensor max_bounds, int64_t M,
              int64_t num_samples, int64_t conf_kernel_size,
              torch::Tensor conf_grid) {
  ray_origin = ray_origin.contiguous();
  ray_dirs = ray_dirs.contiguous();
  inv_ray_dirs = inv_ray_dirs.contiguous();
  candidate_ids = candidate_ids.contiguous();
  start_indices = start_indices.contiguous();
  ray_counts = ray_counts.contiguous();
  min_bounds = min_bounds.contiguous();
  max_bounds = max_bounds.contiguous();
  conf_grid = conf_grid.contiguous();
  return ray_intersect_fw(ray_origin, ray_dirs, inv_ray_dirs, candidate_ids,
                          start_indices, ray_counts, min_bounds, max_bounds, M,
                          num_samples, conf_kernel_size, conf_grid);
}

// Register the operator with PyTorch.
static auto registry = torch::RegisterOperators().op(
    "compact_ray_intersect::ray_intersect", &ray_intersect);
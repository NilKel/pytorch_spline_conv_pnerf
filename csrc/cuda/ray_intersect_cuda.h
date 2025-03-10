#pragma once
#include <torch/extension.h>
#include <tuple>

// Forward declaration of the CUDA forward function.
// This operator takes:
//   - ray_dirs, inv_ray_dirs: [R, 3]
//   - candidate_ids: [C] int32 (sorted indices into the bounds)
//   - start_indices: [R] int32, the starting index for each ray in candidate_ids
//   - ray_counts: [R] int32, the total candidate count per ray (not capped)
//   - min_bounds, max_bounds: [N, 3] float32 (precomputed candidate bounds)
//   - M: maximum intersections to record per ray.
//   - num_samples: number of linspace samples per valid intersection
//   - conf_kernel_size: the resolution (K) for the confidence grid (a single resolution)
//   - conf_grid: [N, K^3] float32 confidence values per candidate
//
// Returns a 5-tuple of tensors:
//   - t_mid: [R, M] float32 midpoints along the ray for valid intersections.
//   - confidence: [R, M] float32 summed confidence per candidate.
//   - pseudo: [R, M, num_samples, 3] float32 pseudo vectors (per-sample normalized offsets)
//   - conf_sample: [R, M, num_samples] float32 individual confidence values per sample.
//   - prim_ids: [R, M] int32 candidate primitive IDs.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
ray_intersect_fw_cuda(torch::Tensor ray_origin, torch::Tensor ray_dirs, torch::Tensor inv_ray_dirs,
                      torch::Tensor candidate_ids, torch::Tensor start_indices,
                      torch::Tensor ray_counts, torch::Tensor min_bounds,
                      torch::Tensor max_bounds, int64_t M,
                      int64_t num_samples, int64_t conf_kernel_size,
                      torch::Tensor conf_grid);
#include "ray_intersect_cuda.h"
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cmath>

#define THREADS 1024
#define BLOCKS(N) ((N + THREADS - 1) / THREADS)

#include "ray_intersect_cuda.h"
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cmath>

#define THREADS 1024
#define BLOCKS(N) ((N + THREADS - 1) / THREADS)

// Modified CUDA kernel for ray-primitive intersection with sampling and confidence evaluation.
// Now takes a separate ray_origin (shape [3]) for all rays.
__global__ void ray_intersect_fw_kernel_modified(
    const float* ray_origin,       // [3] global ray origin
    const float* ray_dirs,         // [R, 3]
    const float* inv_ray_dirs,     // [R, 3]
    const int* candidate_ids,      // [C]
    const int* start_indices,      // [R]
    const int* ray_counts,         // [R]
    const float* min_bounds,       // [N, 3] (already origin-subtracted)
    const float* max_bounds,       // [N, 3] (already origin-subtracted)
    int M,                         // maximum intersections per ray
    int R,                         // number of rays
    int num_samples,               // number of linspace samples per intersection
    int conf_kernel_size,          // confidence grid resolution (K)
    const float* conf_grid,        // [N, K^3] confidence values per candidate
    float* out_t_mid,              // [R, M] output: midpoint t values
    float* out_confidence,         // [R, M] output: summed confidence per candidate
    float* out_pseudo,             // [R, M, num_samples, 3] output: pseudo vectors per sample
    float* out_conf_sample,        // [R, M, num_samples] output: individual confidence values per sample
    int* out_prim_ids              // [R, M] output: candidate primitive IDs
) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= R) return;
    
    int start = start_indices[r];
    int total_count = ray_counts[r];
    int valid_intersections = 0;
    
    // Use the global ray origin (same for all rays)
    float origin_x = ray_origin[0];
    float origin_y = ray_origin[1];
    float origin_z = ray_origin[2];
    
    // Load the ray direction for ray r.
    float dir_x = ray_dirs[r * 3 + 0];
    float dir_y = ray_dirs[r * 3 + 1];
    float dir_z = ray_dirs[r * 3 + 2];
    
    float inv_dir_x = inv_ray_dirs[r * 3 + 0];
    float inv_dir_y = inv_ray_dirs[r * 3 + 1];
    float inv_dir_z = inv_ray_dirs[r * 3 + 2];
    
    // Loop over all candidate primitives for this ray.
    for (int i = start; i < start + total_count; i++) {
        if (valid_intersections >= M) break;
        
        int cid = candidate_ids[i];
        
        // Load candidate's AABB bounds.
        float min_bound_x = min_bounds[cid * 3 + 0];
        float min_bound_y = min_bounds[cid * 3 + 1];
        float min_bound_z = min_bounds[cid * 3 + 2];
        float max_bound_x = max_bounds[cid * 3 + 0];
        float max_bound_y = max_bounds[cid * 3 + 1];
        float max_bound_z = max_bounds[cid * 3 + 2];
        
        // Compute ray-AABB intersection using the slab method.
        float tx1 = (min_bound_x - origin_x) * inv_dir_x;
        float tx2 = (max_bound_x - origin_x) * inv_dir_x;
        float tmin_x = fminf(tx1, tx2);
        float tmax_x = fmaxf(tx1, tx2);
        
        float ty1 = (min_bound_y - origin_y) * inv_dir_y;
        float ty2 = (max_bound_y - origin_y) * inv_dir_y;
        float tmin_y = fminf(ty1, ty2);
        float tmax_y = fmaxf(ty1, ty2);
        
        float tz1 = (min_bound_z - origin_z) * inv_dir_z;
        float tz2 = (max_bound_z - origin_z) * inv_dir_z;
        float tmin_z = fminf(tz1, tz2);
        float tmax_z = fmaxf(tz1, tz2);
        
        float t_entry = fmaxf(fmaxf(tmin_x, tmin_y), tmin_z);
        float t_exit  = fminf(fminf(tmax_x, tmax_y), tmax_z);
        
        // Check for a valid intersection.
        if (t_entry < t_exit && t_exit > 0.0f) {
            // Compute the midpoint t value.
            float t_mid = 0.5f * (t_entry + t_exit);
            
            // Determine spacing between samples.
            float dt = (t_exit - t_entry) / num_samples;
            float confidence_sum = 0.0f;
            
            // Loop over samples.
            for (int s = 0; s < num_samples; s++) {
                float t_sample = t_entry + (s + 0.5f) * dt;
                // Compute the sample point: p = ray_origin + t_sample * ray_dir.
                float p_x = origin_x + t_sample * dir_x;
                float p_y = origin_y + t_sample * dir_y;
                float p_z = origin_z + t_sample * dir_z;
                
                // Compute the pseudo vector: normalized coordinate within candidate's AABB.
                float scale_x = max_bound_x - min_bound_x;
                float scale_y = max_bound_y - min_bound_y;
                float scale_z = max_bound_z - min_bound_z;
                float pseudo_x = (p_x - min_bound_x) / (scale_x + 1e-6f);
                float pseudo_y = (p_y - min_bound_y) / (scale_y + 1e-6f);
                float pseudo_z = (p_z - min_bound_z) / (scale_z + 1e-6f);
                
                // Save the pseudo vector (unmultiplied).
                int out_pseudo_idx = ((r * M + valid_intersections) * num_samples + s) * 3;
                out_pseudo[out_pseudo_idx + 0] = pseudo_x;
                out_pseudo[out_pseudo_idx + 1] = pseudo_y;
                out_pseudo[out_pseudo_idx + 2] = pseudo_z;
                
                // --- Confidence interpolation via trilinear interpolation ---
                // Scale pseudo by conf_kernel_size.
                int K = conf_kernel_size;
                float scaled_x = pseudo_x * K;
                float scaled_y = pseudo_y * K;
                float scaled_z = pseudo_z * K;
                float fx = scaled_x - floorf(scaled_x);
                float fy = scaled_y - floorf(scaled_y);
                float fz = scaled_z - floorf(scaled_z);
                int ix = (int)floorf(scaled_x);
                int iy = (int)floorf(scaled_y);
                int iz = (int)floorf(scaled_z);
                // Clamp lower indices so that ix+1, iy+1, iz+1 remain in bounds.
                ix = (ix < 0) ? 0 : (ix >= K ? K - 1 : ix);
                iy = (iy < 0) ? 0 : (iy >= K ? K - 1 : iy);
                iz = (iz < 0) ? 0 : (iz >= K ? K - 1 : iz);
                float wx0 = 1.0f - fx;
                float wx1 = fx;
                float wy0 = 1.0f - fy;
                float wy1 = fy;
                float wz0 = 1.0f - fz;
                float wz1 = fz;
                int K2 = K * K;
                int base_index = cid * (K * K * K);
                int idx000 = base_index + (ix + iy * K + iz * K2);
                int idx100 = base_index + ((ix+1) + iy * K + iz * K2);
                int idx010 = base_index + (ix + (iy+1) * K + iz * K2);
                int idx110 = base_index + ((ix+1) + (iy+1) * K + iz * K2);
                int idx001 = base_index + (ix + iy * K + (iz+1) * K2);
                int idx101 = base_index + ((ix+1) + iy * K + (iz+1) * K2);
                int idx011 = base_index + (ix + (iy+1) * K + (iz+1) * K2);
                int idx111 = base_index + ((ix+1) + (iy+1) * K + (iz+1) * K2);
                
                float c000 = conf_grid[idx000];
                float c100 = conf_grid[idx100];
                float c010 = conf_grid[idx010];
                float c110 = conf_grid[idx110];
                float c001 = conf_grid[idx001];
                float c101 = conf_grid[idx101];
                float c011 = conf_grid[idx011];
                float c111 = conf_grid[idx111];
                
                float conf_val = wx0 * wy0 * wz0 * c000 +
                                  wx1 * wy0 * wz0 * c100 +
                                  wx0 * wy1 * wz0 * c010 +
                                  wx1 * wy1 * wz0 * c110 +
                                  wx0 * wy0 * wz1 * c001 +
                                  wx1 * wy0 * wz1 * c101 +
                                  wx0 * wy1 * wz1 * c011 +
                                  wx1 * wy1 * wz1 * c111;
                // Save the interpolated confidence value.
                int out_conf_sample_idx = ((r * M + valid_intersections) * num_samples + s);
                out_conf_sample[out_conf_sample_idx] = conf_val;
                
                confidence_sum += conf_val;
                // --- End confidence interpolation ---
            }
            
            // Skip candidate if summed confidence is zero.
            if (confidence_sum == 0.0f) continue;
            
            // Record outputs for this candidate.
            out_t_mid[r * M + valid_intersections] = t_mid;
            out_confidence[r * M + valid_intersections] = confidence_sum;
            out_prim_ids[r * M + valid_intersections] = cid;
            valid_intersections++;
        }
    }
}

// Forward function: launches the modified CUDA kernel.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
ray_intersect_fw_cuda(torch::Tensor ray_origin, torch::Tensor ray_dirs, torch::Tensor inv_ray_dirs,
                      torch::Tensor candidate_ids, torch::Tensor start_indices,
                      torch::Tensor ray_counts, torch::Tensor min_bounds,
                      torch::Tensor max_bounds, int64_t M,
                      int64_t num_samples, int64_t conf_kernel_size,
                      torch::Tensor conf_grid) {
    int R = ray_dirs.size(0);

    auto options_float = torch::TensorOptions().dtype(torch::kFloat32).device(ray_dirs.device());
    auto options_int   = torch::TensorOptions().dtype(torch::kInt32).device(ray_dirs.device());

    // Allocate output tensors.
    auto out_t_mid = torch::empty({R, M}, options_float);
    auto out_confidence = torch::empty({R, M}, options_float);
    auto out_pseudo = torch::empty({R, M, num_samples, 3}, options_float);
    auto out_conf_sample = torch::empty({R, M, num_samples}, options_float);
    auto out_prim_ids = torch::full({R, M}, -1, options_int);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int total_rays = R;

    // Launch the kernel.
    ray_intersect_fw_kernel_modified<<<BLOCKS(total_rays), THREADS, 0, stream>>>(
        ray_origin.data_ptr<float>(),
        ray_dirs.data_ptr<float>(),
        inv_ray_dirs.data_ptr<float>(),
        candidate_ids.data_ptr<int>(),
        start_indices.data_ptr<int>(),
        ray_counts.data_ptr<int>(),
        min_bounds.data_ptr<float>(),
        max_bounds.data_ptr<float>(),
        M,
        R,
        num_samples,
        conf_kernel_size,
        conf_grid.data_ptr<float>(),
        out_t_mid.data_ptr<float>(),
        out_confidence.data_ptr<float>(),
        out_pseudo.data_ptr<float>(),
        out_conf_sample.data_ptr<float>(),
        out_prim_ids.data_ptr<int>()
    );

    return std::make_tuple(out_t_mid, out_confidence, out_pseudo, out_conf_sample, out_prim_ids);
}


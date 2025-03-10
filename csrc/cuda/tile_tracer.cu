#include "tile_tracer_cuda.h"
#include "tile_tracer_helpers.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Main rendering kernel that processes a single tile
template <typename scalar_t, int FEATURE_DIM>
__global__ void tile_tracer_kernel(
    // Image dimensions
    const int width, const int height,
    // Ray parameters
    const scalar_t* ray_origin,          // [3]
    const scalar_t* ray_dirs,            // [width*height, 3]
    const scalar_t* inv_ray_dirs,        // [width*height, 3]
    // Primitive data
    const scalar_t* positions,           // [N, 3]
    const scalar_t* scales,              // [N, 3]
    const scalar_t* densities,           // [N]
    const scalar_t* features,            // [N, F]
    // Tile-primitive mapping
    const uint2* tile_ranges,            // [tiles_width * tiles_height, 2]
    const uint32_t* primitive_indices,   // [total_associations]
    // Sampling parameters
    const int max_samples_per_primitive, // Number of samples per segment
    const scalar_t* sample_positions,    // [max_samples_per_primitive] - normalized positions for sampling
    // Output buffers
    scalar_t* out_color,                 // [width*height, 3]
    scalar_t* out_alpha                  // [width*height]
) {
    // Identify current tile and associated min/max pixel range
    auto block = cg::this_thread_block();
    uint32_t horizontal_blocks = (width + BLOCK_X - 1) / BLOCK_X;
    uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
    uint2 pix_max = { min(pix_min.x + BLOCK_X, (uint32_t)width), min(pix_min.y + BLOCK_Y, (uint32_t)height) };
    
    // Calculate local thread position within the block
    uint2 pix = { pix_min.x + block.thread_index().x % BLOCK_X, pix_min.y + block.thread_index().x / BLOCK_X };
    uint32_t pix_id = width * pix.y + pix.x;
    
    // Check if this thread is associated with a valid pixel or outside
    bool inside = pix.x < width && pix.y < height;
    
    // If outside image bounds, skip processing
    if (!inside) return;
    
    // Load ray data for this pixel
    scalar_t ray_o[3], ray_d[3], inv_ray_d[3];
    for (int i = 0; i < 3; i++) {
        ray_o[i] = ray_origin[i];
        ray_d[i] = ray_dirs[pix_id * 3 + i];
        inv_ray_d[i] = inv_ray_dirs[pix_id * 3 + i];
    }
    
    // Load start/end range of primitives to process for this tile
    int tile_idx = block.group_index().y * horizontal_blocks + block.group_index().x;
    uint2 range = tile_ranges[tile_idx];
    
    // Rendering variables
    scalar_t T = 1.0f;  // Transmittance
    scalar_t accumulated_color[3] = {0.0f, 0.0f, 0.0f};
    scalar_t accumulated_features[FEATURE_DIM] = {0.0f};
    
    // Process all primitives in this tile's range
    for (int i = range.x; i < range.y && T > TRANSMITTANCE_THRESHOLD; i++) {
        // Get primitive index
        int prim_idx = primitive_indices[i];
        
        // Extract primitive positions and bounds
        // We'll assume primitive min/max bounds are precomputed and provided
        scalar_t prim_pos[3];
        scalar_t min_bounds[3], max_bounds[3];
        
        for (int j = 0; j < 3; j++) {
            prim_pos[j] = positions[prim_idx * 3 + j];
            min_bounds[j] = positions[prim_idx * 3 + j] - scales[prim_idx * 3 + j] * 0.5f;
            max_bounds[j] = positions[prim_idx * 3 + j] + scales[prim_idx * 3 + j] * 0.5f;
        }
        
        // Compute ray-AABB intersection
        float t_near, t_far;
        bool intersects = intersectAABB(
            make_float3(ray_o[0], ray_o[1], ray_o[2]),
            make_float3(ray_d[0], ray_d[1], ray_d[2]),
            make_float3(inv_ray_d[0], inv_ray_d[1], inv_ray_d[2]),
            make_float3(min_bounds[0], min_bounds[1], min_bounds[2]),
            make_float3(max_bounds[0], max_bounds[1], max_bounds[2]),
            t_near, t_far
        );
        
        // If no intersection, skip to next primitive
        if (!intersects) continue;
        
        // Base density from primitive
        scalar_t base_density = densities[prim_idx];
        
        // Skip primitives with negligible density
        if (base_density < 1e-4f) continue;
        
        // Sample segment length (delta_t)
        scalar_t delta_t = (t_far - t_near) / max_samples_per_primitive;
        
        // Generate sample points along segment and accumulate features, alpha
        for (int s = 0; s < max_samples_per_primitive && T > TRANSMITTANCE_THRESHOLD; s++) {
            // Compute sample position along ray
            scalar_t t = t_near + (t_far - t_near) * sample_positions[s];
            float3 sample_pos = getSamplePoint(
                make_float3(ray_o[0], ray_o[1], ray_o[2]),
                make_float3(ray_d[0], ray_d[1], ray_d[2]),
                t
            );
            
            // Compute normalized position in primitive local space
            float3 norm_pos = getNormalizedLocalPos(
                sample_pos,
                make_float3(prim_pos[0], prim_pos[1], prim_pos[2]),
                make_float3(prim_scale[0], prim_scale[1], prim_scale[2])
            );
            
            // Check if sample is within primitive bounds
            if (norm_pos.x < 0 || norm_pos.x > 1 || 
                norm_pos.y < 0 || norm_pos.y > 1 || 
                norm_pos.z < 0 || norm_pos.z > 1) {
                continue; // Skip samples outside primitive
            }
            
            // TODO: In a full implementation, this is where you would:
            // 1. Interpolate features using hashgrid (mfused)
            // 2. Call MLP to get actual density at this point
            
            // For now, we'll use the primitive's base density
            scalar_t density = base_density;
            
            // Compute opacity from density and segment length
            scalar_t alpha = 1.0f - expf(-density * delta_t);
            
            // Skip samples with negligible opacity
            if (alpha < 1e-4f) continue;
            
            // Accumulate features weighted by alpha and transmittance
            for (int f = 0; f < FEATURE_DIM; f++) {
                accumulated_features[f] += features[prim_idx * FEATURE_DIM + f] * alpha * T;
            }
            
            // Update transmittance
            T *= (1.0f - alpha);
        }
    }
    
    // Store results - convert accumulated features to RGB
    // For a simple implementation, we'll assume the first 3 features are RGB
    if (FEATURE_DIM >= 3) {
        out_color[pix_id * 3 + 0] = accumulated_features[0];
        out_color[pix_id * 3 + 1] = accumulated_features[1];
        out_color[pix_id * 3 + 2] = accumulated_features[2];
    } else {
        // Fallback if not enough features
        for (int c = 0; c < 3; c++) {
            out_color[pix_id * 3 + c] = FEATURE_DIM > 0 ? accumulated_features[0] : 0.0f;
        }
    }
    
    out_alpha[pix_id] = 1.0f - T;
}

void tile_tracer_render_cuda(
    torch::Tensor ray_origin,
    torch::Tensor ray_dirs,
    torch::Tensor inv_ray_dirs,
    torch::Tensor positions,  // Center positions of primitives
    torch::Tensor scales,     // Scale/size of primitives (used to compute min/max bounds on-the-fly)
    torch::Tensor densities,
    torch::Tensor features,
    torch::Tensor tile_ranges,
    torch::Tensor primitive_indices,
    torch::Tensor sample_positions,
    int width, int height,
    torch::Tensor out_color,
    torch::Tensor out_alpha
) {
    CHECK_CUDA(ray_origin);
    CHECK_CUDA(ray_dirs);
    CHECK_CUDA(inv_ray_dirs);
    CHECK_CUDA(positions);
    CHECK_CUDA(scales);
    CHECK_CUDA(densities);
    CHECK_CUDA(features);
    CHECK_CUDA(tile_ranges);
    CHECK_CUDA(primitive_indices);
    CHECK_CUDA(sample_positions);
    CHECK_CUDA(out_color);
    CHECK_CUDA(out_alpha);
    
    // Calculate grid/block dimensions
    int tiles_width = (width + TILE_SIZE - 1) / TILE_SIZE;
    int tiles_height = (height + TILE_SIZE - 1) / TILE_SIZE;
    
    dim3 grid(tiles_width, tiles_height, 1);
    dim3 block(BLOCK_X * BLOCK_Y, 1, 1);
    
    // Get CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // Dispatch based on data type and feature dimensions
    AT_DISPATCH_FLOATING_TYPES(ray_dirs.scalar_type(), "tile_tracer_render_cuda", [&] {
        const int feature_dim = features.size(1);
        
        // Handle different feature dimensions
        switch (feature_dim) {
            case 1:
                tile_tracer_kernel<scalar_t, 1><<<grid, block, 0, stream>>>(
                    width, height,
                    ray_origin.data_ptr<scalar_t>(),
                    ray_dirs.data_ptr<scalar_t>(),
                    inv_ray_dirs.data_ptr<scalar_t>(),
                    positions.data_ptr<scalar_t>(),
                    scales.data_ptr<scalar_t>(),
                    densities.data_ptr<scalar_t>(),
                    features.data_ptr<scalar_t>(),
                    reinterpret_cast<const uint2*>(tile_ranges.data_ptr<int>()),
                    primitive_indices.data_ptr<unsigned int>(),
                    sample_positions.size(0),
                    sample_positions.data_ptr<scalar_t>(),
                    out_color.data_ptr<scalar_t>(),
                    out_alpha.data_ptr<scalar_t>()
                );
                break;
            case 3:
                tile_tracer_kernel<scalar_t, 3><<<grid, block, 0, stream>>>(
                    width, height,
                    ray_origin.data_ptr<scalar_t>(),
                    ray_dirs.data_ptr<scalar_t>(),
                    inv_ray_dirs.data_ptr<scalar_t>(),
                    positions.data_ptr<scalar_t>(),
                    scales.data_ptr<scalar_t>(),
                    densities.data_ptr<scalar_t>(),
                    features.data_ptr<scalar_t>(),
                    reinterpret_cast<const uint2*>(tile_ranges.data_ptr<int>()),
                    primitive_indices.data_ptr<unsigned int>(),
                    sample_positions.size(0),
                    sample_positions.data_ptr<scalar_t>(),
                    out_color.data_ptr<scalar_t>(),
                    out_alpha.data_ptr<scalar_t>()
                );
                break;
            case 16:
                tile_tracer_kernel<scalar_t, 16><<<grid, block, 0, stream>>>(
                    width, height,
                    ray_origin.data_ptr<scalar_t>(),
                    ray_dirs.data_ptr<scalar_t>(),
                    inv_ray_dirs.data_ptr<scalar_t>(),
                    positions.data_ptr<scalar_t>(),
                    scales.data_ptr<scalar_t>(),
                    densities.data_ptr<scalar_t>(),
                    features.data_ptr<scalar_t>(),
                    reinterpret_cast<const uint2*>(tile_ranges.data_ptr<int>()),
                    primitive_indices.data_ptr<unsigned int>(),
                    sample_positions.size(0),
                    sample_positions.data_ptr<scalar_t>(),
                    out_color.data_ptr<scalar_t>(),
                    out_alpha.data_ptr<scalar_t>()
                );
                break;
            default:
                // Default case with dynamic feature dimension - will be slightly slower
                tile_tracer_kernel<scalar_t, 32><<<grid, block, 0, stream>>>(
                    width, height,
                    ray_origin.data_ptr<scalar_t>(),
                    ray_dirs.data_ptr<scalar_t>(),
                    inv_ray_dirs.data_ptr<scalar_t>(),
                    positions.data_ptr<scalar_t>(),
                    scales.data_ptr<scalar_t>(),
                    densities.data_ptr<scalar_t>(),
                    features.data_ptr<scalar_t>(),
                    reinterpret_cast<const uint2*>(tile_ranges.data_ptr<int>()),
                    primitive_indices.data_ptr<unsigned int>(),
                    sample_positions.size(0),
                    sample_positions.data_ptr<scalar_t>(),
                    out_color.data_ptr<scalar_t>(),
                    out_alpha.data_ptr<scalar_t>()
                );
        }
    });
}
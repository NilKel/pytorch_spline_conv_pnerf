#pragma once

#include <cuda_runtime.h>

// Constants for tile-based rendering
#define TILE_SIZE 8
#define BLOCK_X TILE_SIZE
#define BLOCK_Y TILE_SIZE
#define THREADS_PER_TILE (TILE_SIZE * TILE_SIZE)
#define MAX_SAMPLES_PER_RAY 64
#define TRANSMITTANCE_THRESHOLD 0.01f

// Helper function to convert NDC coordinates to pixel coordinates
__device__ inline float ndc2Pix(float ndc, int size) {
    return ((ndc + 1.0f) * 0.5f) * size;
}

// Helper function to compute tile bounds for a given primitive
__device__ inline void getRect(
    const float2 point, 
    int radius, 
    uint2& rect_min, 
    uint2& rect_max, 
    const dim3 grid) {
    
    rect_min.x = max(0u, min((uint32_t)((point.x - radius) / TILE_SIZE), grid.x - 1));
    rect_min.y = max(0u, min((uint32_t)((point.y - radius) / TILE_SIZE), grid.y - 1));
    rect_max.x = max(0u, min((uint32_t)((point.x + radius + TILE_SIZE - 1) / TILE_SIZE), grid.x - 1));
    rect_max.y = max(0u, min((uint32_t)((point.y + radius + TILE_SIZE - 1) / TILE_SIZE), grid.y - 1));
}

// Helper function to compute ray-AABB intersection
__device__ inline bool intersectAABB(
    const float3 ray_o,
    const float3 ray_d,
    const float3 inv_ray_d,
    const float3 min_bounds,
    const float3 max_bounds,
    float& t_near,
    float& t_far) {
    
    // X slab
    float tx1 = (min_bounds.x - ray_o.x) * inv_ray_d.x;
    float tx2 = (max_bounds.x - ray_o.x) * inv_ray_d.x;
    float tmin = fminf(tx1, tx2);
    float tmax = fmaxf(tx1, tx2);
    
    // Y slab
    float ty1 = (min_bounds.y - ray_o.y) * inv_ray_d.y;
    float ty2 = (max_bounds.y - ray_o.y) * inv_ray_d.y;
    tmin = fmaxf(tmin, fminf(ty1, ty2));
    tmax = fminf(tmax, fmaxf(ty1, ty2));
    
    // Z slab
    float tz1 = (min_bounds.z - ray_o.z) * inv_ray_d.z;
    float tz2 = (max_bounds.z - ray_o.z) * inv_ray_d.z;
    tmin = fmaxf(tmin, fminf(tz1, tz2));
    tmax = fminf(tmax, fmaxf(tz1, tz2));
    
    // Check for valid intersection
    if (tmin < tmax && tmax > 0.0f) {
        t_near = fmaxf(tmin, 0.0f);  // Clamp to non-negative
        t_far = tmax;
        return true;
    }
    
    return false;
}

// Helper to compute sample point along ray
__device__ inline float3 getSamplePoint(
    const float3 ray_o,
    const float3 ray_d,
    float t) {
    
    return make_float3(
        ray_o.x + t * ray_d.x,
        ray_o.y + t * ray_d.y,
        ray_o.z + t * ray_d.z
    );
}

// Calculate normalized position in primitive local space
__device__ inline float3 getNormalizedLocalPos(
    const float3 sample_pos,
    const float3 prim_pos,
    const float3 prim_scale) {
    
    // Compute offset from primitive center
    float3 offset;
    offset.x = sample_pos.x - prim_pos.x;
    offset.y = sample_pos.y - prim_pos.y;
    offset.z = sample_pos.z - prim_pos.z;
    
    // Normalize to [0,1] range within primitive bounds
    float3 normalized;
    normalized.x = offset.x / prim_scale.x + 0.5f;
    normalized.y = offset.y / prim_scale.y + 0.5f;
    normalized.z = offset.z / prim_scale.z + 0.5f;
    
    return normalized;
}
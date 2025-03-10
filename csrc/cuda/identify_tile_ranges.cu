#include "tile_tracer_cuda.h"
#include "tile_tracer_helpers.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#define THREADS 1024
#define BLOCKS(N) ((N + THREADS - 1) / THREADS)

// Check keys to see if it is at the start/end of one tile's range in 
// the full sorted list. If yes, write start/end of this tile.
// Run once per instanced (duplicated) primitive ID.
__global__ void identify_tile_ranges_kernel(
    int total_associations,
    const uint64_t* primitive_keys,
    uint2* tile_ranges)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_associations)
        return;

    // Read tile ID from key. Update start/end of tile range if at limit.
    uint64_t key = primitive_keys[idx];
    uint32_t currtile = key >> 32;
    
    if (idx == 0)
        tile_ranges[currtile].x = 0;
    else
    {
        uint32_t prevtile = primitive_keys[idx - 1] >> 32;
        if (currtile != prevtile)
        {
            tile_ranges[prevtile].y = idx;
            tile_ranges[currtile].x = idx;
        }
    }
    
    if (idx == total_associations - 1)
        tile_ranges[currtile].y = total_associations;
}

void identify_tile_ranges_cuda(
    torch::Tensor primitive_keys,
    torch::Tensor tile_ranges)
{
    CHECK_CUDA(primitive_keys);
    CHECK_CUDA(tile_ranges);
    
    int total_associations = primitive_keys.size(0);
    
    // Skip if there are no associations
    if (total_associations == 0)
        return;
    
    // Initialize tile ranges to zero
    tile_ranges.zero_();
    
    // Set CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // Launch kernel
    identify_tile_ranges_kernel<<<BLOCKS(total_associations), THREADS, 0, stream>>>(
        total_associations,
        reinterpret_cast<const uint64_t*>(primitive_keys.data_ptr<int64_t>()),
        reinterpret_cast<uint2*>(tile_ranges.data_ptr<int>())
    );
}
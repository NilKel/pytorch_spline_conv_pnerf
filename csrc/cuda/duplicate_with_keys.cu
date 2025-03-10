#include "tile_tracer_cuda.h"
#include "tile_tracer_helpers.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#define THREADS 1024
#define BLOCKS(N) ((N + THREADS - 1) / THREADS)

// Generates one key/value pair for all primitive/tile overlaps.
// Run once per primitive (1:N mapping).
template <typename scalar_t>
__global__ void duplicate_with_keys_kernel(
    int num_primitives,
    const scalar_t* screen_positions,   // [N, 2]
    const float* depths,                // [N]
    const int* primitive_radii,         // [N]
    const uint32_t* point_offsets,      // [N]
    uint64_t* primitive_keys_unsorted,  // [total_associations]
    uint32_t* primitive_indices_unsorted, // [total_associations]
    dim3 grid)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_primitives)
        return;

    // Generate no key/value pair for primitives with zero radius
    if (primitive_radii[idx] > 0)
    {
        // Find this primitive's offset in buffer for writing keys/values
        uint32_t off = (idx == 0) ? 0 : point_offsets[idx - 1];
        
        // Screen position of primitive
        float2 point = make_float2(screen_positions[idx*2], screen_positions[idx*2+1]);
        
        // Compute tile rectangle overlapped by this primitive
        uint2 rect_min, rect_max;
        getRect(point, primitive_radii[idx], rect_min, rect_max, grid);

        // For each tile that the bounding rect overlaps, emit a key/value pair.
        // The key is |  tile ID  |      depth      |, and the value is the ID of the primitive.
        // Sorting the values with this key yields primitive IDs in a list, such that they
        // are first sorted by tile and then by depth.
        for (int y = rect_min.y; y < rect_max.y; y++)
        {
            for (int x = rect_min.x; x < rect_max.x; x++)
            {
                uint64_t key = y * grid.x + x;
                key <<= 32;
                key |= *((uint32_t*)&depths[idx]);
                primitive_keys_unsorted[off] = key;
                primitive_indices_unsorted[off] = idx;
                off++;
            }
        }
    }
}

void duplicate_with_keys_cuda(
    torch::Tensor screen_positions,
    torch::Tensor depths,
    torch::Tensor primitive_radii,
    torch::Tensor point_offsets,
    torch::Tensor primitive_keys_unsorted,
    torch::Tensor primitive_indices_unsorted,
    int width, int height)
{
    CHECK_CUDA(screen_positions);
    CHECK_CUDA(depths);
    CHECK_CUDA(primitive_radii);
    CHECK_CUDA(point_offsets);
    CHECK_CUDA(primitive_keys_unsorted);
    CHECK_CUDA(primitive_indices_unsorted);
    
    int num_primitives = screen_positions.size(0);
    
    // Calculate grid dimensions for tiles
    dim3 grid((width + TILE_SIZE - 1) / TILE_SIZE, (height + TILE_SIZE - 1) / TILE_SIZE, 1);
    
    // Set CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // Launch kernel
    AT_DISPATCH_FLOATING_TYPES(screen_positions.scalar_type(), "duplicate_with_keys_cuda", [&] {
        duplicate_with_keys_kernel<scalar_t><<<BLOCKS(num_primitives), THREADS, 0, stream>>>(
            num_primitives,
            screen_positions.data_ptr<scalar_t>(),
            depths.data_ptr<float>(),
            primitive_radii.data_ptr<int>(),
            point_offsets.data_ptr<unsigned int>(),
            reinterpret_cast<uint64_t*>(primitive_keys_unsorted.data_ptr<int64_t>()),
            primitive_indices_unsorted.data_ptr<unsigned int>(),
            grid
        );
    });
}
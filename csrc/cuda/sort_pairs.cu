#include "tile_tracer_cuda.h"
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>

// Helper function to get the size of the temporary storage needed for CUB's radix sort
size_t get_radix_sort_temp_storage_size(int num_items) {
    // Create dummy input/output tensors for size calculation
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    
    uint64_t* d_keys_in = nullptr;
    uint64_t* d_keys_out = nullptr;
    uint32_t* d_values_in = nullptr;
    uint32_t* d_values_out = nullptr;
    
    // Calculate required temp storage size for RadixSort
    cub::DeviceRadixSort::SortPairs(
        d_temp_storage, temp_storage_bytes,
        d_keys_in, d_keys_out,
        d_values_in, d_values_out,
        num_items);
    
    return temp_storage_bytes;
}

// Sort primitive key-value pairs using CUB's radix sort
void sort_primitive_pairs_cuda(
    torch::Tensor primitive_keys_unsorted,
    torch::Tensor primitive_indices_unsorted,
    torch::Tensor& primitive_keys_sorted,
    torch::Tensor& primitive_indices_sorted,
    torch::Tensor temp_storage)
{
    CHECK_CUDA(primitive_keys_unsorted);
    CHECK_CUDA(primitive_indices_unsorted);
    CHECK_CUDA(primitive_keys_sorted);
    CHECK_CUDA(primitive_indices_sorted);
    CHECK_CUDA(temp_storage);
    
    int num_items = primitive_keys_unsorted.size(0);
    if (num_items == 0) return;
    
    // Set CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // Sort pairs using CUB's DeviceRadixSort
    // We're sorting by (tile_id << 32 | depth) as the key, so first by tile and then by depth
    cub::DeviceRadixSort::SortPairs(
        temp_storage.data_ptr(),
        temp_storage.size(0),
        reinterpret_cast<uint64_t*>(primitive_keys_unsorted.data_ptr<int64_t>()),
        reinterpret_cast<uint64_t*>(primitive_keys_sorted.data_ptr<int64_t>()),
        primitive_indices_unsorted.data_ptr<unsigned int>(),
        primitive_indices_sorted.data_ptr<unsigned int>(),
        num_items,
        0,      // begin_bit (start from bit 0)
        64,     // end_bit (use all 64 bits)
        stream
    );
}

// Get the radix sort temp storage size needed for Python-side allocation
int get_sort_temp_storage_size_cuda(int num_items) {
    return get_radix_sort_temp_storage_size(num_items);
}
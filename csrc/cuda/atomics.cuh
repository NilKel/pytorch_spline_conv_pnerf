#pragma once
#include <ATen/native/cuda/KernelUtils.cuh>

template <typename scalar_t>
__device__ void atomAdd(scalar_t *address,  long index, long totalElements, scalar_t val) {
    // Use atomicAdd for non-Half types
    atomicAdd(&address[index], val);
}

// Specialization for c10::Half type using fastAtomicAdd
template <>
__device__ void atomAdd<c10::Half>(c10::Half *address, long index, long totalElements,  c10::Half val) {
    at::native::fastAtomicAdd(address, index, totalElements, val, true);
}

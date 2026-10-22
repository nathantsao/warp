// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "builtin.h"

#include "sycl_runtime.h"

#include <array>
#include <cstring>

namespace {

using Array = wp::array_t<char>;

inline char* element(Array array, size_t index)
{
    ptrdiff_t offset = 0;
    for (int d = array.ndim - 1; d >= 0; --d) {
        const size_t coordinate = index % array.shape[d];
        index /= array.shape[d];
        offset += ptrdiff_t(coordinate) * array.strides[d];
    }
    return array.data + offset;
}

}

extern "C" WP_API int wp_sycl_array_copy(
    void* device, const void* dst, const void* src, size_t dst_offset, size_t src_offset, size_t count, size_t size
)
{
    return wp::sycl_checked([&] {
        const Array dest = *static_cast<const Array*>(dst);
        const Array source = *static_cast<const Array*>(src);
        wp::sycl_queue(device).parallel_for(sycl::range<1>(count), [=](sycl::id<1> i) {
            char* d = element(dest, dst_offset + i);
            const char* s = element(source, src_offset + i);
            for (size_t byte = 0; byte < size; ++byte)
                d[byte] = s[byte];
        });
    });
}

extern "C" WP_API int
wp_sycl_array_fill(void* device, const void* descriptor, const void* value, size_t size, size_t count)
{
    return wp::sycl_checked([&] {
        if (size > 256)
            throw std::runtime_error("Non-contiguous SYCL fills support values of at most 256 bytes");
        const Array array = *static_cast<const Array*>(descriptor);
        std::array<char, 256> pattern {};
        std::memcpy(pattern.data(), value, size);
        wp::sycl_queue(device).parallel_for(sycl::range<1>(count), [=](sycl::id<1> i) {
            char* dst = element(array, i);
            for (size_t byte = 0; byte < size; ++byte)
                dst[byte] = pattern[byte];
        });
    });
}

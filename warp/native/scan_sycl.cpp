// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Array-wide scan for SYCL devices: per-tile reduction, a single work-group scan of the tile
// sums, then per-tile scans.
//
// The scan is a sequence of kernels on the current in-order stream, with temporaries taken from
// wp_sycl_scratch(). Nothing waits on the host or allocates through the Python allocator, so the
// scan is asynchronous and can be recorded into command graphs.

#include "warp.h"

#include "scan.h"
#include "sycl_runtime.h"

#include <cstdint>

namespace {

constexpr size_t kGroupSize = 256;
constexpr size_t kScanItems = 8;
constexpr size_t kScanTile = kGroupSize * kScanItems;

size_t align_up(size_t bytes) { return (bytes + 255) & ~size_t(255); }

size_t scan_tile_count(size_t n) { return (n + kScanTile - 1) / kScanTile; }

// Scans each of ``components`` interleaved scalars of ``n`` elements. Strides are in bytes and may be
// negative. The input and output may alias, since every work-item reads its elements before writing them.
template <typename T>
void scan(
    sycl::queue& queue,
    const char* in,
    char* out,
    size_t n,
    ptrdiff_t in_stride,
    ptrdiff_t out_stride,
    size_t components,
    bool inclusive,
    T* partials
)
{
    const size_t tiles = scan_tile_count(n);
    const sycl::nd_range<2> range({ components, tiles * kGroupSize }, { 1, kGroupSize });

    if (tiles > 1) {
        queue.parallel_for(range, [=](sycl::nd_item<2> item) {
            const size_t c = item.get_global_id(0);
            const size_t tile = item.get_group(1);
            const size_t first = tile * kScanTile + item.get_local_id(1) * kScanItems;
            T sum = T(0);
            for (size_t j = 0; j < kScanItems; ++j) {
                const size_t i = first + j;
                if (i < n)
                    sum += *reinterpret_cast<const T*>(in + ptrdiff_t(i) * in_stride + c * sizeof(T));
            }
            sum = sycl::reduce_over_group(item.get_group(), sum, sycl::plus<T>());
            if (item.get_local_id(1) == 0)
                partials[c * tiles + tile] = sum;
        });

        queue.parallel_for(
            sycl::nd_range<2>({ components, kGroupSize }, { 1, kGroupSize }), [=](sycl::nd_item<2> item) {
                T* sums = partials + item.get_group(0) * tiles;
                T carry = T(0);
                for (size_t first = 0; first < tiles; first += kGroupSize) {
                    const size_t i = first + item.get_local_id(1);
                    const T value = i < tiles ? sums[i] : T(0);
                    const T prefix = sycl::exclusive_scan_over_group(item.get_group(), value, sycl::plus<T>());
                    const T total = sycl::reduce_over_group(item.get_group(), value, sycl::plus<T>());
                    if (i < tiles)
                        sums[i] = carry + prefix;
                    carry += total;
                }
            }
        );
    }

    queue.parallel_for(range, [=](sycl::nd_item<2> item) {
        const size_t c = item.get_global_id(0);
        const size_t tile = item.get_group(1);
        const size_t first = tile * kScanTile + item.get_local_id(1) * kScanItems;
        T values[kScanItems];
        T sum = T(0);
        for (size_t j = 0; j < kScanItems; ++j) {
            const size_t i = first + j;
            values[j] = i < n ? *reinterpret_cast<const T*>(in + ptrdiff_t(i) * in_stride + c * sizeof(T)) : T(0);
            sum += values[j];
        }
        T prefix = sycl::exclusive_scan_over_group(item.get_group(), sum, sycl::plus<T>());
        if (tiles > 1)
            prefix += partials[c * tiles + tile];
        for (size_t j = 0; j < kScanItems; ++j) {
            const size_t i = first + j;
            if (i >= n)
                break;
            T* dst = reinterpret_cast<T*>(out + ptrdiff_t(i) * out_stride + c * sizeof(T));
            if (inclusive) {
                prefix += values[j];
                *dst = prefix;
            } else {
                *dst = prefix;
                prefix += values[j];
            }
        }
    });
}

}  // anonymous namespace

template <typename T> size_t scan_sycl_temp_size(int n, int type_length)
{
    const size_t tiles = scan_tile_count(size_t(n));
    return tiles > 1 ? align_up(tiles * size_t(type_length) * sizeof(T)) : 0;
}

template <typename T>
void scan_sycl(
    void* device,
    const T* values_in,
    T* values_out,
    int n,
    int in_byte_stride,
    int out_byte_stride,
    int type_length,
    bool inclusive,
    void* temp
)
{
    if (n <= 0)
        return;

    const size_t temp_size = scan_sycl_temp_size<T>(n, type_length);
    if (temp_size && !temp)
        temp = wp::sycl_scratch(device, temp_size);

    scan<T>(
        wp::sycl_queue(device), reinterpret_cast<const char*>(values_in), reinterpret_cast<char*>(values_out),
        size_t(n), in_byte_stride, out_byte_stride, size_t(type_length), inclusive, static_cast<T*>(temp)
    );
}

template size_t scan_sycl_temp_size<int>(int, int);
template size_t scan_sycl_temp_size<uint32_t>(int, int);
template size_t scan_sycl_temp_size<int64_t>(int, int);
template size_t scan_sycl_temp_size<float>(int, int);
template size_t scan_sycl_temp_size<double>(int, int);

template void scan_sycl(void*, const int*, int*, int, int, int, int, bool, void*);
template void scan_sycl(void*, const uint32_t*, uint32_t*, int, int, int, int, bool, void*);
template void scan_sycl(void*, const int64_t*, int64_t*, int, int, int, int, bool, void*);
template void scan_sycl(void*, const float*, float*, int, int, int, int, bool, void*);
template void scan_sycl(void*, const double*, double*, int, int, int, int, bool, void*);

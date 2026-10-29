// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Radix sort and segmented sort for SYCL devices.
//
// Every operation is a sequence of kernels and copies on the current in-order stream, with
// temporaries taken from wp_sycl_scratch(). Nothing waits on the host or allocates through the
// Python allocator, so the operations are asynchronous and can be recorded into command graphs.

#include "warp.h"

#include "scan.h"
#include "sort.h"
#include "sycl_runtime.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace {

constexpr size_t kGroupSize = 256;

size_t align_up(size_t bytes) { return (bytes + 255) & ~size_t(255); }

// ---------------------------------------------------------------------------------------------
// Radix sort: stable least-significant-digit passes over 4-bit digits. Each pass builds per-tile
// digit counts, scans them in digit-major order to get global offsets, and scatters every tile in
// its original order. Keys and values ping-pong between [0, n) and the caller's [n, 2n) scratch.

constexpr int kRadixBits = 4;
constexpr int kBuckets = 1 << kRadixBits;
constexpr int kPackedWords = kBuckets / 4;
constexpr size_t kSortItems = 4;
constexpr size_t kSortTile = kGroupSize * kSortItems;

// Work-group scans run on four 16-bit bucket counters packed into each 64-bit word.
static_assert(kSortTile < 65536, "Packed bucket counters must not overflow 16 bits");

enum class KeyOrder { Unsigned, Signed, Float };

// Map key bits to unsigned bits with the same order, as the CPU implementation does.
template <typename U, KeyOrder Order> inline U ordered_bits(U bits)
{
    constexpr U sign = U(1) << (sizeof(U) * 8 - 1);
    if constexpr (Order == KeyOrder::Signed)
        return bits ^ sign;
    else if constexpr (Order == KeyOrder::Float)
        return bits ^ ((bits & sign) ? U(~U(0)) : sign);
    else
        return bits;
}

inline void count_bucket(uint64_t (&packed)[kPackedWords], int bucket)
{
    packed[bucket >> 2] += uint64_t(1) << (16 * (bucket & 3));
}

inline uint32_t bucket_count(const uint64_t (&packed)[kPackedWords], int bucket)
{
    return uint32_t(packed[bucket >> 2] >> (16 * (bucket & 3))) & 0xffffu;
}

size_t sort_tile_count(size_t n) { return (n + kSortTile - 1) / kSortTile; }

size_t radix_sort_temp_size(size_t n)
{
    const size_t counts = kBuckets * sort_tile_count(n);
    return align_up(counts * sizeof(uint32_t)) + scan_sycl_temp_size<uint32_t>(int(counts), 1);
}

template <typename U, KeyOrder Order, typename V>
void radix_sort(void* device, U* keys, V* values, size_t n, int begin_bit, int end_bit, char* scratch)
{
    auto& queue = wp::sycl_queue(device);
    const size_t tiles = sort_tile_count(n);
    const size_t count_size = kBuckets * tiles;
    uint32_t* counts = reinterpret_cast<uint32_t*>(scratch);
    char* scan_temp = scratch + align_up(count_size * sizeof(uint32_t));
    U* key_buffers[2] = { keys, keys + n };
    V* value_buffers[2] = { values, values + n };
    const sycl::nd_range<1> range(tiles * kGroupSize, kGroupSize);

    int current = 0;
    for (int shift = begin_bit; shift < end_bit; shift += kRadixBits) {
        const int bits = std::min(kRadixBits, end_bit - shift);
        const U mask = (U(1) << bits) - 1;
        const U* keys_in = key_buffers[current];
        const V* values_in = value_buffers[current];
        U* keys_out = key_buffers[1 - current];
        V* values_out = value_buffers[1 - current];

        queue.parallel_for(range, [=](sycl::nd_item<1> item) {
            const size_t tile = item.get_group(0);
            const size_t lane = item.get_local_id(0);
            const size_t first = tile * kSortTile + lane * kSortItems;
            uint64_t packed[kPackedWords] = {};
            for (size_t j = 0; j < kSortItems; ++j) {
                const size_t i = first + j;
                if (i < n)
                    count_bucket(packed, int((ordered_bits<U, Order>(keys_in[i]) >> shift) & mask));
            }
            for (int w = 0; w < kPackedWords; ++w)
                packed[w] = sycl::reduce_over_group(item.get_group(), packed[w], sycl::plus<uint64_t>());
            if (lane < kBuckets)
                counts[lane * tiles + tile] = bucket_count(packed, int(lane));
        });

        scan_sycl<uint32_t>(
            device, counts, counts, int(count_size), sizeof(uint32_t), sizeof(uint32_t), 1, false, scan_temp
        );

        queue.parallel_for(range, [=](sycl::nd_item<1> item) {
            const size_t tile = item.get_group(0);
            const size_t first = tile * kSortTile + item.get_local_id(0) * kSortItems;
            U key[kSortItems] = {};
            V value[kSortItems] = {};
            int bucket[kSortItems] = {};
            uint64_t packed[kPackedWords] = {};
            for (size_t j = 0; j < kSortItems; ++j) {
                const size_t i = first + j;
                if (i < n) {
                    key[j] = keys_in[i];
                    value[j] = values_in[i];
                    bucket[j] = int((ordered_bits<U, Order>(key[j]) >> shift) & mask);
                    count_bucket(packed, bucket[j]);
                }
            }
            // Rank of each element among same-digit elements earlier in the tile.
            for (int w = 0; w < kPackedWords; ++w)
                packed[w] = sycl::exclusive_scan_over_group(item.get_group(), packed[w], sycl::plus<uint64_t>());
            for (size_t j = 0; j < kSortItems; ++j) {
                if (first + j >= n)
                    break;
                const size_t dst = counts[size_t(bucket[j]) * tiles + tile] + bucket_count(packed, bucket[j]);
                count_bucket(packed, bucket[j]);
                keys_out[dst] = key[j];
                values_out[dst] = value[j];
            }
        });

        current = 1 - current;
    }

    if (current) {
        queue.memcpy(keys, keys + n, n * sizeof(U));
        queue.memcpy(values, values + n, n * sizeof(V));
    }
}

// ---------------------------------------------------------------------------------------------
// Segmented sort: one stable radix sort on (segment index, key) composite keys. Elements outside
// every segment get a sentinel segment index and are left in place.

struct SegmentedLayout {
    size_t composite;
    size_t order;
    size_t lengths;
    size_t offsets;
    size_t scan;
    size_t sort;
    size_t bytes;

    SegmentedLayout(size_t n, size_t segments)
    {
        composite = 0;
        order = composite + align_up(2 * n * sizeof(uint64_t));
        lengths = order + align_up(2 * n * sizeof(uint32_t));
        offsets = lengths + align_up(segments * sizeof(int));
        scan = offsets + align_up(segments * sizeof(int));
        sort = scan + scan_sycl_temp_size<int>(int(segments), 1);
        bytes = sort + radix_sort_temp_size(n);
    }
};

inline long long clamp_index(int index, long long low, long long high)
{
    return std::min(std::max(static_cast<long long>(index), low), high);
}

template <KeyOrder Order>
void segmented_sort(
    void* device,
    uint32_t* keys,
    uint32_t* values,
    size_t n,
    const int* starts,
    const int* ends,
    size_t segments,
    char* scratch
)
{
    auto& queue = wp::sycl_queue(device);
    const SegmentedLayout layout { n, segments };
    uint64_t* composite = reinterpret_cast<uint64_t*>(scratch + layout.composite);
    uint32_t* order = reinterpret_cast<uint32_t*>(scratch + layout.order);
    int* lengths = reinterpret_cast<int*>(scratch + layout.lengths);
    int* offsets = reinterpret_cast<int*>(scratch + layout.offsets);

    // The sentinel index ``segments`` must be representable in the composite key.
    int segment_bits = 0;
    while (segment_bits < 32 && (uint64_t(1) << segment_bits) <= segments)
        ++segment_bits;

    const long long count = static_cast<long long>(n);
    queue.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
        const size_t i = id[0];
        composite[i] = (uint64_t(segments) << 32) | ordered_bits<uint32_t, Order>(keys[i]);
        order[i] = uint32_t(i);
    });

    // Out-of-range bounds are clamped so that invalid input cannot write outside the arrays.
    queue.parallel_for(sycl::range<1>(segments), [=](sycl::id<1> id) {
        const size_t s = id[0];
        const long long start = clamp_index(starts[s], 0, count);
        const long long end = clamp_index(ends[s], start, count);
        lengths[s] = int(end - start);
        offsets[s] = int(end - start);
    });

    // One row of work-items per segment, about as many in total as there are elements.
    const size_t lanes = std::max<size_t>(1, (n + segments - 1) / segments);
    queue.parallel_for(sycl::range<2>(segments, lanes), [=](sycl::item<2> item) {
        const size_t s = item.get_id(0);
        const long long start = clamp_index(starts[s], 0, count);
        const long long end = clamp_index(ends[s], start, count);
        for (long long i = start + static_cast<long long>(item.get_id(1)); i < end; i += lanes)
            composite[i] = (uint64_t(s) << 32) | ordered_bits<uint32_t, Order>(keys[i]);
    });

    scan_sycl<int>(device, offsets, offsets, int(segments), sizeof(int), sizeof(int), 1, false, scratch + layout.scan);

    radix_sort<uint64_t, KeyOrder::Unsigned, uint32_t>(
        device, composite, order, n, 0, 32 + segment_bits, scratch + layout.sort
    );

    // Gather into the caller's [n, 2n) scratch first, since the destinations overlap the sources.
    queue.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
        const size_t p = id[0];
        if ((composite[p] >> 32) < segments) {
            keys[n + p] = keys[order[p]];
            values[n + p] = values[order[p]];
        }
    });

    queue.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
        const size_t p = id[0];
        const uint64_t s = composite[p] >> 32;
        if (s >= segments)
            return;
        const long long rank = static_cast<long long>(p) - offsets[s];
        if (rank < 0 || rank >= lengths[s])
            return;
        const long long dst = clamp_index(starts[s], 0, count) + rank;
        keys[dst] = keys[n + p];
        values[dst] = values[n + p];
    });
}

// Type codes shared with warp/_src/sycl.py.
enum SortKeyType {
    SORT_INT32 = 0,
    SORT_UINT32 = 1,
    SORT_FLOAT32 = 2,
    SORT_INT64 = 3,
    SORT_UINT64 = 4,
    SORT_FLOAT64 = 5,
};

template <typename U, KeyOrder Order>
void radix_sort_pairs_sycl(void* device, void* keys, void* values, int n, int begin_bit, int end_bit, int value_size)
{
    char* scratch = static_cast<char*>(wp::sycl_scratch(device, radix_sort_temp_size(size_t(n))));
    if (value_size == 4)
        radix_sort<U, Order, uint32_t>(
            device, static_cast<U*>(keys), static_cast<uint32_t*>(values), size_t(n), begin_bit, end_bit, scratch
        );
    else
        radix_sort<U, Order, uint64_t>(
            device, static_cast<U*>(keys), static_cast<uint64_t*>(values), size_t(n), begin_bit, end_bit, scratch
        );
}

template <KeyOrder Order>
void segmented_sort_pairs_sycl(
    void* device,
    void* keys,
    void* values,
    int n,
    const void* segment_starts,
    const void* segment_ends,
    int num_segments
)
{
    const SegmentedLayout layout { size_t(n), size_t(num_segments) };
    char* scratch = static_cast<char*>(wp::sycl_scratch(device, layout.bytes));
    segmented_sort<Order>(
        device, static_cast<uint32_t*>(keys), static_cast<uint32_t*>(values), size_t(n),
        static_cast<const int*>(segment_starts), static_cast<const int*>(segment_ends), size_t(num_segments), scratch
    );
}

}  // anonymous namespace

extern "C" WP_API int wp_sycl_radix_sort_pairs(
    void* device, int key_type, void* keys, void* values, int n, int begin_bit, int end_bit, int value_size
)
{
    return wp::sycl_checked([&] {
        if (n <= 0 || begin_bit == end_bit)
            return;
        if (value_size != 4 && value_size != 8)
            throw std::runtime_error("SYCL radix sort values must be 4 or 8 bytes wide");
        const int key_width = key_type <= SORT_FLOAT32 ? 32 : 64;
        if (begin_bit < 0 || end_bit < begin_bit || end_bit > key_width)
            throw std::runtime_error("Invalid SYCL radix sort bit range");
        switch (key_type) {
        case SORT_INT32:
            radix_sort_pairs_sycl<uint32_t, KeyOrder::Signed>(device, keys, values, n, begin_bit, end_bit, value_size);
            break;
        case SORT_UINT32:
            radix_sort_pairs_sycl<uint32_t, KeyOrder::Unsigned>(
                device, keys, values, n, begin_bit, end_bit, value_size
            );
            break;
        case SORT_FLOAT32:
            radix_sort_pairs_sycl<uint32_t, KeyOrder::Float>(device, keys, values, n, begin_bit, end_bit, value_size);
            break;
        case SORT_INT64:
            radix_sort_pairs_sycl<uint64_t, KeyOrder::Signed>(device, keys, values, n, begin_bit, end_bit, value_size);
            break;
        case SORT_UINT64:
            radix_sort_pairs_sycl<uint64_t, KeyOrder::Unsigned>(
                device, keys, values, n, begin_bit, end_bit, value_size
            );
            break;
        case SORT_FLOAT64:
            radix_sort_pairs_sycl<uint64_t, KeyOrder::Float>(device, keys, values, n, begin_bit, end_bit, value_size);
            break;
        default:
            throw std::runtime_error("Unsupported SYCL radix sort key type");
        }
    });
}

extern "C" WP_API int wp_sycl_segmented_sort_pairs(
    void* device,
    int key_type,
    void* keys,
    void* values,
    int n,
    const void* segment_starts,
    const void* segment_ends,
    int num_segments
)
{
    return wp::sycl_checked([&] {
        if (n <= 0 || num_segments <= 0)
            return;
        switch (key_type) {
        case SORT_INT32:
            segmented_sort_pairs_sycl<KeyOrder::Signed>(
                device, keys, values, n, segment_starts, segment_ends, num_segments
            );
            break;
        case SORT_FLOAT32:
            segmented_sort_pairs_sycl<KeyOrder::Float>(
                device, keys, values, n, segment_starts, segment_ends, num_segments
            );
            break;
        default:
            throw std::runtime_error("Unsupported SYCL segmented sort key type");
        }
    });
}

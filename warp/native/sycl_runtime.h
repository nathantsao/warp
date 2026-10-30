// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api.h"

#include <cstddef>
#include <cstdio>
#include <cstring>
#include <memory>
#include <new>
#include <stdexcept>
#include <vector>

#include <sycl/sycl.hpp>

extern "C" {
WP_API int wp_sycl_init();
WP_API const char* wp_sycl_version();
WP_API const char* wp_sycl_get_error_string();
WP_API void wp_sycl_set_error_string(const char* message);
WP_API int wp_sycl_device_count();
WP_API void* wp_sycl_device_get(int ordinal);
WP_API const char* wp_sycl_device_name(void* device);
WP_API size_t wp_sycl_device_max_work_group_size(void* device);
WP_API size_t wp_sycl_device_local_memory_size(void* device);
WP_API int wp_sycl_device_compute_units(void* device);
WP_API size_t wp_sycl_device_resident_threads(void* device);
WP_API int wp_sycl_device_memory_info(void* device, size_t* free_memory, size_t* total_memory);
WP_API int wp_sycl_device_has_fp64(void* device);
WP_API int wp_sycl_device_has_atomic64(void* device);
WP_API void* wp_sycl_queue(void* device);
WP_API void* wp_sycl_alloc(void* device, size_t size);
WP_API int wp_sycl_free(void* device, void* ptr);
WP_API int wp_sycl_memcpy(void* device, void* dest, const void* src, size_t size, int synchronous);
WP_API int wp_sycl_memset(void* device, void* dest, int value, size_t size);
WP_API int wp_sycl_memtile(void* device, void* dest, const void* src, size_t size, size_t count);
WP_API int wp_sycl_array_copy(
    void* device, const void* dst, const void* src, size_t dst_offset, size_t src_offset, size_t count, size_t size
);
WP_API int wp_sycl_array_fill(void* device, const void* descriptor, const void* value, size_t size, size_t count);
WP_API int wp_sycl_array_scan(
    void* device,
    int type,
    const void* in,
    void* out,
    int n,
    int in_stride,
    int out_stride,
    int type_length,
    int inclusive
);
WP_API void* wp_sycl_stage_args(void* device, const void* src, size_t size);
WP_API void* wp_sycl_scratch(void* device, size_t size);
WP_API void* wp_sycl_stream_create(void* device);
WP_API void* wp_sycl_stream_default(void* device);
WP_API void* wp_sycl_stream_current(void* device);
WP_API int wp_sycl_stream_destroy(void* device, void* stream);
WP_API int wp_sycl_stream_set_current(void* device, void* stream, int sync);
WP_API int wp_sycl_stream_synchronize(void* device, void* stream);
WP_API int wp_sycl_stream_query(void* stream);
WP_API void* wp_sycl_event_create(void* device, int enable_timing);
WP_API void wp_sycl_event_destroy(void* event);
WP_API int wp_sycl_event_record(void* device, void* event, void* stream, int external);
WP_API int wp_sycl_stream_wait_event(void* device, void* stream, void* event, int external);
WP_API int wp_sycl_event_synchronize(void* event);
WP_API int wp_sycl_event_query(void* event);
WP_API int wp_sycl_event_elapsed_time(void* start, void* end, float* milliseconds);
WP_API int wp_sycl_synchronize(void* device);
WP_API void* wp_sycl_load_module(const char* path);
WP_API void* wp_sycl_get_kernel(void* module, const char* name);
WP_API int wp_sycl_unload_module(void* device, void* module);
WP_API int wp_sycl_graph_supported(void* device);
WP_API int wp_sycl_graph_event_supported(void* device);
WP_API void* wp_sycl_graph_begin(void* device);
WP_API int wp_sycl_graph_end(void* device, void* graph);
WP_API int wp_sycl_graph_launch(void* device, void* graph);
WP_API int wp_sycl_graph_destroy(void* device, void* graph);
WP_API int wp_sycl_graph_memory_info(void* device, size_t* allocations, size_t* bytes);
}

// Kernel arguments are passed by value up to this size; larger argument structs are staged in USM.
#ifndef WP_SYCL_MAX_KERNEL_ARG_BYTES
#define WP_SYCL_MAX_KERNEL_ARG_BYTES 2048
#endif

namespace wp {

void sycl_defer_resource(void* device, std::shared_ptr<void> resource);

inline void* sycl_stage_args(void* device, const void* src, size_t size)
{
    void* staged = wp_sycl_stage_args(device, src, size);
    if (!staged)
        throw std::runtime_error(wp_sycl_get_error_string());
    return staged;
}

inline sycl::queue& sycl_queue(void* device) { return *static_cast<sycl::queue*>(wp_sycl_queue(device)); }

// Device scratch ordered by the current stream; during graph capture the buffer is owned by the graph.
inline void* sycl_scratch(void* device, size_t size)
{
    void* scratch = wp_sycl_scratch(device, size);
    if (!scratch)
        throw std::runtime_error(wp_sycl_get_error_string());
    return scratch;
}

template <typename F> int sycl_checked(F&& fn)
{
    try {
        fn();
        return 0;
    } catch (const std::exception& e) {
        wp_sycl_set_error_string(e.what());
        return -1;
    }
}

// Blocking copy of ``count`` device elements into ``host``.
template <typename T> void sycl_read_array(void* device, std::vector<T>& host, const T* source, size_t count)
{
    host.resize(count);
    if (count && source)
        sycl_queue(device).memcpy(host.data(), source, sizeof(T) * count).wait_and_throw();
}

// Shared USM copies of host data, owned for the lifetime of a BVH or mesh.
struct SyclSharedStorage {
    void* device;
    std::vector<void*> allocations;

    explicit SyclSharedStorage(void* device)
        : device(device)
    {
        if (!sycl_queue(device).get_device().has(sycl::aspect::usm_shared_allocations))
            throw std::runtime_error("SYCL geometry requires shared USM support");
    }

    SyclSharedStorage(const SyclSharedStorage&) = delete;
    SyclSharedStorage& operator=(const SyclSharedStorage&) = delete;

    ~SyclSharedStorage()
    {
        for (void* ptr : allocations) {
            try {
                sycl::free(ptr, sycl_queue(device));
            } catch (const std::exception& error) {
                fprintf(stderr, "Warp SYCL geometry cleanup failed: %s\n", error.what());
            }
        }
    }

    template <typename T> T* copy(const T* source, size_t count)
    {
        if (!source || !count)
            return nullptr;
        T* result = sycl::malloc_shared<T>(count, sycl_queue(device));
        if (!result)
            throw std::bad_alloc();
        try {
            allocations.push_back(result);
        } catch (...) {
            sycl::free(result, sycl_queue(device));
            throw;
        }
        std::memcpy(result, source, sizeof(T) * count);
        return result;
    }
};

}

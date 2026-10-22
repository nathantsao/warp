// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api.h"

#include <cstddef>

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
WP_API int wp_sycl_device_has_fp64(void* device);
WP_API int wp_sycl_device_has_atomic64(void* device);
WP_API void* wp_sycl_queue(void* device);
WP_API void* wp_sycl_alloc(void* device, size_t size);
WP_API int wp_sycl_free(void* device, void* ptr);
WP_API int wp_sycl_memcpy(void* device, void* dest, const void* src, size_t size, int synchronous);
WP_API int wp_sycl_memset(void* device, void* dest, int value, size_t size);
WP_API int wp_sycl_memtile(void* device, void* dest, const void* src, size_t size, size_t count);
WP_API int wp_sycl_synchronize(void* device);
WP_API void* wp_sycl_load_module(const char* path);
WP_API void* wp_sycl_get_kernel(void* module, const char* name);
WP_API int wp_sycl_unload_module(void* device, void* module);
WP_API int wp_sycl_graph_supported(void* device);
WP_API void* wp_sycl_graph_begin(void* device);
WP_API int wp_sycl_graph_end(void* device, void* graph);
WP_API int wp_sycl_graph_launch(void* device, void* graph);
WP_API int wp_sycl_graph_destroy(void* device, void* graph);
}

namespace wp {

inline sycl::queue& sycl_queue(void* device) { return *static_cast<sycl::queue*>(wp_sycl_queue(device)); }

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

}

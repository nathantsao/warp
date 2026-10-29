// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <sycl/ext/oneapi/work_group_static.hpp>
#include <sycl/sycl.hpp>

#ifndef WP_SYCL_SHARED_BYTES
#define WP_SYCL_SHARED_BYTES 16
#endif

namespace wp {

inline auto sycl_work_item() { return sycl::ext::oneapi::this_work_item::get_nd_item<1>(); }

inline void sycl_tile_sync() { sycl::group_barrier(sycl_work_item().get_group()); }

inline int sycl_tile_active_count() { return int(sycl_work_item().get_local_range(0)); }

inline void sycl_tile_clear_flags(bool* flags)
{
    const int tid = int(sycl_work_item().get_local_linear_id());
    const int active_count = sycl_tile_active_count();
    for (int base = 0; base < WP_TILE_BLOCK_DIM; base += active_count) {
        const int i = base + tid;
        if (i < WP_TILE_BLOCK_DIM)
            flags[i] = false;
    }
}

inline unsigned int* sycl_tile_offsets()
{
    sycl::ext::oneapi::experimental::work_group_static<unsigned int[WP_TILE_BLOCK_DIM]> offsets;
    return *&offsets;
}

inline char* sycl_tile_storage()
{
    struct alignas(16) Storage {
        char bytes[WP_SYCL_SHARED_BYTES];
    };
    sycl::ext::oneapi::experimental::work_group_static<Storage> storage;
    return (&storage)->bytes;
}

}

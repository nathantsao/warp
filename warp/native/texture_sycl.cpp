// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "warp.h"

#include "sycl_runtime.h"
#include "texture.h"

#include <memory>
#include <mutex>
#include <unordered_map>

#include <sycl/ext/oneapi/bindless_images.hpp>

namespace {

namespace experimental = sycl::ext::oneapi::experimental;

struct TextureState {
    experimental::image_descriptor descriptor;
    experimental::image_mem memory;
    experimental::sampled_image_handle handle;

    TextureState(
        void* device,
        const experimental::image_descriptor& descriptor,
        const experimental::bindless_image_sampler& sampler
    )
        : descriptor(descriptor)
        , memory(descriptor, wp::sycl_queue(device))
        , handle(experimental::create_image(memory, sampler, descriptor, wp::sycl_queue(device)))
    {
    }

    ~TextureState()
    {
        try {
            experimental::destroy_image_handle(handle, memory.get_device(), memory.get_context());
        } catch (const std::exception& error) {
            fprintf(stderr, "Warp SYCL texture cleanup failed: %s\n", error.what());
        }
    }
};

std::mutex texture_mutex;
std::unordered_map<void*, std::unordered_map<uint64_t, std::shared_ptr<TextureState>>> textures;

sycl::addressing_mode address_mode(int mode, bool normalized)
{
    switch (mode) {
    case WP_TEXTURE_ADDRESS_WRAP:
    case WP_TEXTURE_ADDRESS_MIRROR:
        if (!normalized)
            throw std::runtime_error("SYCL wrap and mirror texture addressing require normalized coordinates");
        return mode == WP_TEXTURE_ADDRESS_WRAP ? sycl::addressing_mode::repeat : sycl::addressing_mode::mirrored_repeat;
    case WP_TEXTURE_ADDRESS_CLAMP:
        return sycl::addressing_mode::clamp_to_edge;
    case WP_TEXTURE_ADDRESS_BORDER:
        return sycl::addressing_mode::clamp;
    default:
        throw std::runtime_error("Invalid SYCL texture address mode");
    }
}

}  // anonymous namespace

extern "C" {

WP_API uint64_t wp_sycl_texture_create(
    void* device,
    int width,
    int height,
    int channels,
    int dtype,
    int filter,
    int address_u,
    int address_v,
    int normalized
)
{
    uint64_t id = 0;
    wp::sycl_checked([&] {
        if (width <= 0 || height <= 0 || channels != 4 || dtype != WP_TEXTURE_DTYPE_FLOAT32)
            throw std::runtime_error("SYCL textures require a positive 2D shape and float32 RGBA data");
        if (filter != WP_TEXTURE_FILTER_CLOSEST && filter != WP_TEXTURE_FILTER_LINEAR)
            throw std::runtime_error("Invalid SYCL texture filter mode");
        auto& queue = wp::sycl_queue(device);
        if (!queue.get_device().has(sycl::aspect::ext_oneapi_bindless_images))
            throw std::runtime_error("SYCL device does not support DPC++ bindless images");
        if (address_u != address_v && !queue.get_device().has(sycl::aspect::ext_oneapi_unique_addressing_per_dim))
            throw std::runtime_error("SYCL device does not support unique per-axis texture addressing modes");
        const experimental::image_descriptor descriptor(
            sycl::range<2>(width, height), channels, sycl::image_channel_type::fp32
        );
        if (!experimental::is_image_handle_supported<experimental::sampled_image_handle>(
                descriptor, experimental::image_memory_handle_type::opaque_handle, queue
            ))
            throw std::runtime_error("SYCL device does not support the requested float32 RGBA sampled image");
        sycl::addressing_mode addressing[] = { address_mode(address_u, normalized), address_mode(address_v, normalized),
                                               sycl::addressing_mode::clamp_to_edge };
        const experimental::bindless_image_sampler sampler(
            addressing,
            normalized ? sycl::coordinate_normalization_mode::normalized
                       : sycl::coordinate_normalization_mode::unnormalized,
            filter == WP_TEXTURE_FILTER_LINEAR ? sycl::filtering_mode::linear : sycl::filtering_mode::nearest
        );
        auto state = std::make_shared<TextureState>(device, descriptor, sampler);
        const auto handle = state->handle.raw_handle;
        std::lock_guard<std::mutex> lock(texture_mutex);
        if (!textures[device].emplace(handle, std::move(state)).second)
            throw std::runtime_error("Duplicate SYCL texture handle");
        id = handle;
    });
    return id;
}

WP_API int wp_sycl_texture_destroy(void* device, uint64_t id)
{
    return wp::sycl_checked([&] {
        std::lock_guard<std::mutex> lock(texture_mutex);
        auto& resources = textures.at(device);
        // Graph-held Python owners preserve sampled textures. Unrelated releases during capture
        // still need the same deferred, cross-stream reclamation as array allocations and modules.
        wp::sycl_defer_resource(device, resources.at(id));
        resources.erase(id);
    });
}

WP_API int wp_sycl_texture_copy(
    void* device, uint64_t dst_texture, void* dst_array, uint64_t src_texture, const void* src_array, int synchronous
)
{
    return wp::sycl_checked([&] {
        if (((dst_texture != 0) == (dst_array != nullptr)) || ((src_texture != 0) == (src_array != nullptr))
            || (!dst_texture && !src_texture))
            throw std::runtime_error("SYCL texture copies require one source and destination and at least one texture");
        std::lock_guard<std::mutex> lock(texture_mutex);
        auto& resources = textures.at(device);
        auto& queue = wp::sycl_queue(device);
        sycl::event completion;
        if (src_texture && dst_texture) {
            const auto& src = *resources.at(src_texture);
            const auto& dst = *resources.at(dst_texture);
            if (&src == &dst)
                return;
            if (src.descriptor.width != dst.descriptor.width || src.descriptor.height != dst.descriptor.height)
                throw std::runtime_error("Incompatible SYCL texture shapes for copy");
            completion = queue.ext_oneapi_copy(
                src.memory.get_handle(), src.descriptor, dst.memory.get_handle(), dst.descriptor
            );
        } else if (dst_texture) {
            const auto& dst = *resources.at(dst_texture);
            completion = queue.ext_oneapi_copy(src_array, dst.memory.get_handle(), dst.descriptor);
        } else {
            const auto& src = *resources.at(src_texture);
            completion = queue.ext_oneapi_copy(src.memory.get_handle(), dst_array, src.descriptor);
        }
        if (synchronous)
            completion.wait_and_throw();
    });
}

}  // extern "C"

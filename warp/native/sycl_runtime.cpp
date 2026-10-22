// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "sycl_runtime.h"
#include "version.h"

#include <array>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <dlfcn.h>
#include <sycl/ext/oneapi/experimental/graph.hpp>

namespace {

struct SyclDevice {
    sycl::queue queue;
    std::string name;
    bool capturing = false;
    size_t max_work_group_size;
    size_t local_memory_size;
    bool has_fp64;
    bool has_atomic64;
    bool has_graphs;

    explicit SyclDevice(const sycl::device& device)
        : queue(
              device,
              [](sycl::exception_list errors) { std::rethrow_exception(*errors.begin()); },
              sycl::property::queue::in_order()
          )
        , name(device.get_info<sycl::info::device::name>())
        , max_work_group_size(device.get_info<sycl::info::device::max_work_group_size>())
        , local_memory_size(device.get_info<sycl::info::device::local_mem_size>())
        , has_fp64(device.has(sycl::aspect::fp64))
        , has_atomic64(device.has(sycl::aspect::atomic64))
        , has_graphs(device.has(sycl::aspect::ext_oneapi_graph))
    {
    }
};

std::vector<std::unique_ptr<SyclDevice>> devices;
std::once_flag init_flag;
thread_local std::string last_error;

SyclDevice& get_device(void* device) { return *static_cast<SyclDevice*>(device); }

}

const char* wp_sycl_get_error_string() { return last_error.c_str(); }
void wp_sycl_set_error_string(const char* message) { last_error = message; }

const char* wp_sycl_version() { return WP_VERSION_STRING; }

int wp_sycl_init()
{
    return wp::sycl_checked([] {
        std::call_once(init_flag, [] {
            std::vector<std::unique_ptr<SyclDevice>> discovered;
            for (const auto& platform : sycl::platform::get_platforms()) {
                // Prefer Level Zero and avoid enumerating the same Intel GPU again through OpenCL.
                if (platform.get_backend() != sycl::backend::ext_oneapi_level_zero)
                    continue;
                for (const auto& device : platform.get_devices(sycl::info::device_type::gpu)) {
                    if (device.get_info<sycl::info::device::vendor_id>() == 0x8086
                        && device.has(sycl::aspect::usm_device_allocations))
                        discovered.push_back(std::make_unique<SyclDevice>(device));
                }
            }
            devices = std::move(discovered);
        });
    });
}

int wp_sycl_device_count() { return static_cast<int>(devices.size()); }

void* wp_sycl_device_get(int ordinal)
{
    if (ordinal < 0 || static_cast<size_t>(ordinal) >= devices.size()) {
        wp_sycl_set_error_string("Invalid SYCL device ordinal");
        return nullptr;
    }
    return devices[ordinal].get();
}

const char* wp_sycl_device_name(void* device) { return get_device(device).name.c_str(); }

size_t wp_sycl_device_max_work_group_size(void* device) { return get_device(device).max_work_group_size; }

size_t wp_sycl_device_local_memory_size(void* device) { return get_device(device).local_memory_size; }

int wp_sycl_device_has_fp64(void* device) { return get_device(device).has_fp64; }
int wp_sycl_device_has_atomic64(void* device) { return get_device(device).has_atomic64; }

void* wp_sycl_queue(void* device) { return &get_device(device).queue; }

void* wp_sycl_alloc(void* device, size_t size)
{
    void* ptr = nullptr;
    wp::sycl_checked([&] {
        if (size > SIZE_MAX - 3)
            throw std::bad_alloc();
        // Sub-word atomics use a containing 32-bit CAS word, including the last array element.
        ptr = sycl::malloc_device((size + 3) & ~size_t(3), get_device(device).queue);
        if (!ptr)
            throw std::bad_alloc();
    });
    return ptr;
}

int wp_sycl_free(void* device, void* ptr)
{
    return wp::sycl_checked([&] {
        if (get_device(device).capturing)
            throw std::runtime_error("Cannot free SYCL memory during graph capture");
        auto& queue = get_device(device).queue;
        queue.wait_and_throw();
        sycl::free(ptr, queue);
    });
}

int wp_sycl_memcpy(void* device, void* dest, const void* src, size_t size, int synchronous)
{
    return wp::sycl_checked([&] {
        if (synchronous && get_device(device).capturing)
            throw std::runtime_error("Host transfers are not supported during SYCL graph capture");
        auto event = get_device(device).queue.memcpy(dest, src, size);
        if (synchronous)
            event.wait_and_throw();
    });
}

int wp_sycl_memset(void* device, void* dest, int value, size_t size)
{
    return wp::sycl_checked([&] { get_device(device).queue.memset(dest, value, size); });
}

int wp_sycl_memtile(void* device, void* dest, const void* src, size_t size, size_t count)
{
    return wp::sycl_checked([&] {
        if (!size || !count)
            return;
        auto& queue = get_device(device).queue;
        if (size <= 256) {
            std::array<unsigned char, 256> pattern {};
            std::memcpy(pattern.data(), src, size);
            auto* output = static_cast<unsigned char*>(dest);
            queue.parallel_for(sycl::range<1>(size * count), [=](sycl::id<1> i) { output[i] = pattern[i % size]; });
            return;
        }
        if (get_device(device).capturing)
            throw std::runtime_error("SYCL graph fills support values of at most 256 bytes");
        auto* pattern = sycl::malloc_device<unsigned char>(size, queue);
        if (!pattern)
            throw std::bad_alloc();
        try {
            queue.memcpy(pattern, src, size);
            auto* output = static_cast<unsigned char*>(dest);
            queue.parallel_for(
                     sycl::range<1>(size * count), [=](sycl::id<1> i) { output[i] = pattern[i % size]; }
            ).wait_and_throw();
        } catch (...) {
            sycl::free(pattern, queue);
            throw;
        }
        sycl::free(pattern, queue);
    });
}

int wp_sycl_synchronize(void* device)
{
    return wp::sycl_checked([&] {
        if (get_device(device).capturing)
            throw std::runtime_error("Cannot synchronize during SYCL graph capture");
        get_device(device).queue.wait_and_throw();
    });
}

void* wp_sycl_load_module(const char* path)
{
    void* module = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!module)
        wp_sycl_set_error_string(dlerror());
    return module;
}

void* wp_sycl_get_kernel(void* module, const char* name)
{
    dlerror();
    void* kernel = dlsym(module, name);
    if (const char* error = dlerror())
        wp_sycl_set_error_string(error);
    return kernel;
}

int wp_sycl_unload_module(void* device, void* module)
{
    return wp::sycl_checked([&] {
        get_device(device).queue.wait_and_throw();
        if (dlclose(module))
            throw std::runtime_error(dlerror());
    });
}

namespace {
namespace experimental = sycl::ext::oneapi::experimental;
struct SyclGraph {
    experimental::command_graph<experimental::graph_state::modifiable> recording;
    std::unique_ptr<experimental::command_graph<experimental::graph_state::executable>> executable;
    bool active = true;

    explicit SyclGraph(sycl::queue& queue)
        : recording(queue.get_context(), queue.get_device())
    {
    }
};
}

int wp_sycl_graph_supported(void* device) { return get_device(device).has_graphs; }

void* wp_sycl_graph_begin(void* device)
{
    void* result = nullptr;
    wp::sycl_checked([&] {
        auto& state = get_device(device);
        if (!wp_sycl_graph_supported(device))
            throw std::runtime_error("SYCL command graphs are unavailable on this device");
        if (state.capturing)
            throw std::runtime_error("SYCL graph capture already active");
        auto graph = std::make_unique<SyclGraph>(state.queue);
        graph->recording.begin_recording(state.queue);
        state.capturing = true;
        result = graph.release();
    });
    return result;
}

int wp_sycl_graph_end(void* device, void* handle)
{
    return wp::sycl_checked([&] {
        auto& state = get_device(device);
        auto& graph = *static_cast<SyclGraph*>(handle);
        graph.recording.end_recording(state.queue);
        graph.active = false;
        state.capturing = false;
        graph.executable = std::make_unique<experimental::command_graph<experimental::graph_state::executable>>(
            graph.recording.finalize()
        );
    });
}

int wp_sycl_graph_launch(void* device, void* handle)
{
    return wp::sycl_checked([&] {
        auto& graph = *static_cast<SyclGraph*>(handle);
        if (!graph.executable)
            throw std::runtime_error("SYCL graph has not been finalized");
        get_device(device).queue.ext_oneapi_graph(*graph.executable);
    });
}

int wp_sycl_graph_destroy(void* device, void* handle)
{
    return wp::sycl_checked([&] {
        auto& state = get_device(device);
        auto* graph = static_cast<SyclGraph*>(handle);
        if (graph->active) {
            graph->recording.end_recording(state.queue);
            graph->active = false;
            state.capturing = false;
        }
        state.queue.wait_and_throw();
        delete graph;
    });
}

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "warp.h"

#include "scan.h"
#include "sycl_runtime.h"
#include "version.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include <dlfcn.h>
#include <level_zero/ze_api.h>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <sycl/ext/oneapi/experimental/graph.hpp>

namespace {

namespace experimental = sycl::ext::oneapi::experimental;

struct SyclEventRecord;

struct SyclEventState {
    std::shared_ptr<SyclEventRecord> record;
};

struct SyclCaptureState {
    bool active = true;
};

struct SyclGraphEvent {
    std::weak_ptr<SyclEventState> state;
    std::shared_ptr<SyclEventRecord> record;
};

struct SyclEventRecord {
    sycl::context context;
    sycl::event event;
    sycl::event completion;
    std::shared_ptr<SyclCaptureState> capture;
    uint64_t* timestamp = nullptr;
    bool external = false;
    bool launched = false;

    explicit SyclEventRecord(const sycl::context& context)
        : context(context)
    {
    }

    ~SyclEventRecord()
    {
        if (timestamp) {
            try {
                sycl::free(timestamp, context);
            } catch (const std::exception& error) {
                fprintf(stderr, "Warp SYCL event cleanup failed: %s\n", error.what());
            }
        }
    }
};

struct SyclGraph {
    using Executable = experimental::command_graph<experimental::graph_state::executable>;

    sycl::queue queue;
    ze_command_list_handle_t native_list;
    std::unique_ptr<experimental::command_graph<experimental::graph_state::modifiable>> recording;
    std::vector<std::shared_ptr<Executable>> segments;
    std::vector<void*> allocations;
    std::vector<void*> staged;
    std::vector<SyclGraphEvent> events;
    std::shared_ptr<SyclCaptureState> capture = std::make_shared<SyclCaptureState>();
    sycl::event last_event;
    bool active = false;

    SyclGraph(const sycl::queue& q, ze_command_list_handle_t native_list)
        : queue(q)
        , native_list(native_list)
    {
    }

    void begin_segment()
    {
        auto segment = std::make_unique<experimental::command_graph<experimental::graph_state::modifiable>>(
            queue.get_context(), queue.get_device(),
            native_list ? sycl::property_list { experimental::property::graph::enable_native_recording {} }
                        : sycl::property_list {}
        );
        segment->begin_recording(queue);
        recording = std::move(segment);
    }

    void finish_segment()
    {
        recording->end_recording(queue);
        auto segment = std::move(recording);
        segments.push_back(std::make_shared<Executable>(segment->finalize()));
    }

    void retain_staged(void* ptr)
    {
        try {
            staged.push_back(ptr);
        } catch (...) {
            sycl::free(ptr, queue);
            throw;
        }
    }
};

// As on CUDA, a captured allocation is released only after both the graph and user reference are gone.
struct SyclGraphAllocInfo {
    sycl::context context;
    size_t size;
    bool ref_exists = true;
    bool graph_destroyed = false;
};

struct SyclFreeInfo {
    void* ptr;
    sycl::context context;
};

// Profiling is enabled so that events can be timed.
sycl::queue make_queue(const sycl::device& device)
{
    return sycl::queue(
        device, [](sycl::exception_list errors) { std::rethrow_exception(*errors.begin()); },
        sycl::property_list { sycl::property::queue::in_order(), sycl::property::queue::enable_profiling() }
    );
}

// A stream is an in-order queue. Each one owns its staging ring so that streams cannot overwrite the
// kernel arguments staged by another stream.
struct SyclStream {
    sycl::queue queue;
    ze_command_list_handle_t native_list = nullptr;

    // Kernels whose arguments exceed the hardware limit read them from USM staging memory instead.
    // Eager launches cycle through a ring of slots that are ordered by the in-order queue. Launches
    // recorded into a graph get a private copy owned by the graph, since the graph replays later.
    static constexpr size_t stage_slot_bytes = 16384;
    static constexpr size_t stage_slot_count = 256;
    std::mutex stage_mutex;
    unsigned char* stage_host = nullptr;
    unsigned char* stage_device = nullptr;
    std::array<sycl::event, stage_slot_count> stage_events;
    size_t stage_next = 0;
    std::unique_ptr<sycl::queue> stage_queue;

    // Device scratch for native algorithms (scan, sort). The in-order queue serializes its reuse.
    std::mutex scratch_mutex;
    void* scratch = nullptr;
    size_t scratch_bytes = 0;

    explicit SyclStream(sycl::queue q)
        : queue(std::move(q))
    {
        const auto native = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(queue);
        if (const auto* list = std::get_if<ze_command_list_handle_t>(&native))
            native_list = *list;
    }

    ~SyclStream()
    {
        try {
            queue.wait_and_throw();
        } catch (...) {
        }
        if (stage_host)
            sycl::free(stage_host, queue);
        if (stage_device)
            sycl::free(stage_device, queue);
        if (scratch)
            sycl::free(scratch, queue);
    }
};

struct SyclEvent {
    void* device;
    bool timing;
    std::shared_ptr<SyclEventState> state = std::make_shared<SyclEventState>();
};

struct SyclDevice {
    std::unique_ptr<SyclStream> default_stream;
    SyclStream* current = nullptr;
    std::mutex streams_mutex;
    std::vector<SyclStream*> streams;
    std::string name;
    SyclGraph* capture = nullptr;
    std::mutex memory_mutex;
    std::unordered_map<void*, SyclGraphAllocInfo> graph_allocs;
    std::vector<SyclFreeInfo> deferred_frees;
    std::vector<std::unique_ptr<SyclGraph>> deferred_graphs;
    std::vector<void*> deferred_modules;
    std::vector<std::shared_ptr<void>> deferred_resources;
    size_t max_work_group_size;
    size_t local_memory_size;
    bool has_fp64;
    bool has_atomic64;
    bool has_graphs;
    int compute_units;
    size_t resident_threads;
    size_t total_memory;
    bool has_free_memory;
    bool has_native_graphs = false;
    bool has_graph_events = false;
    uint64_t timestamp_mask = 0;
    uint32_t timestamp_bits = 0;
    uint64_t timer_hz = 0;

    explicit SyclDevice(const sycl::device& device)
        : name(device.get_info<sycl::info::device::name>())
        , max_work_group_size(device.get_info<sycl::info::device::max_work_group_size>())
        , local_memory_size(device.get_info<sycl::info::device::local_mem_size>())
        , has_fp64(device.has(sycl::aspect::fp64))
        , has_atomic64(device.has(sycl::aspect::atomic64))
        , has_graphs(device.has(sycl::aspect::ext_oneapi_graph))
        , compute_units(static_cast<int>(device.get_info<sycl::info::device::max_compute_units>()))
        , resident_threads(device.get_info<sycl::info::device::max_work_group_size>())
        , total_memory(device.get_info<sycl::info::device::global_mem_size>())
        , has_free_memory(device.has(sycl::aspect::ext_intel_free_memory))
    {
        default_stream = std::make_unique<SyclStream>(make_queue(device));
        current = default_stream.get();

        if (has_graphs && default_stream->native_list
            && device.get_platform().get_info<sycl::info::platform::name>()
                == "Intel(R) oneAPI Unified Runtime over Level-Zero V2") {
            try {
                experimental::command_graph graph(
                    default_stream->queue,
                    sycl::property_list { experimental::property::graph::enable_native_recording {} }
                );
                has_native_graphs = true;
            } catch (const sycl::exception& error) {
                if (error.code() != sycl::errc::invalid && error.code() != sycl::errc::feature_not_supported)
                    throw;
                fprintf(stderr, "Warp SYCL native graph capture unavailable on %s: %s\n", name.c_str(), error.what());
            }
        }
        has_graph_events = has_native_graphs && device.has(sycl::aspect::usm_host_allocations);
        if (has_graph_events) {
            ze_device_properties_t properties {};
            properties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES_1_2;
            if (zeDeviceGetProperties(sycl::get_native<sycl::backend::ext_oneapi_level_zero>(device), &properties)
                != ZE_RESULT_SUCCESS)
                throw std::runtime_error("Cannot query Level Zero timestamp properties");
            timestamp_bits = properties.timestampValidBits;
            timer_hz = properties.timerResolution;
            if (!timer_hz || !timestamp_bits || timestamp_bits > 64)
                throw std::runtime_error("Invalid Level Zero timestamp properties");
            timestamp_mask
                = timestamp_bits == 64 ? std::numeric_limits<uint64_t>::max() : (uint64_t(1) << timestamp_bits) - 1;
        }

        // Express the device size in Xe cores (the analogue of CUDA SMs) and resident SIMD lanes.
        namespace intel = sycl::ext::intel::info::device;
        if (device.has(sycl::aspect::ext_intel_gpu_slices)
            && device.has(sycl::aspect::ext_intel_gpu_subslices_per_slice))
            compute_units = static_cast<int>(
                device.get_info<intel::gpu_slices>() * device.get_info<intel::gpu_subslices_per_slice>()
            );
        if (device.has(sycl::aspect::ext_intel_gpu_eu_count)
            && device.has(sycl::aspect::ext_intel_gpu_hw_threads_per_eu)) {
            // The smallest sub-group width gives the conservative lane count per hardware thread.
            const auto widths = device.get_info<sycl::info::device::sub_group_sizes>();
            const size_t lanes = widths.empty() ? 1 : *std::min_element(widths.begin(), widths.end());
            resident_threads
                = device.get_info<intel::gpu_eu_count>() * device.get_info<intel::gpu_hw_threads_per_eu>() * lanes;
        }
    }
};

std::vector<std::unique_ptr<SyclDevice>> devices;
std::once_flag init_flag;
thread_local std::string last_error;

SyclDevice& get_device(void* device) { return *static_cast<SyclDevice*>(device); }

void check_event_access(const SyclEvent& event)
{
    if (get_device(event.device).capture)
        throw std::runtime_error("Cannot query or synchronize SYCL events during graph capture");
    const auto& record = event.state->record;
    if (record && record->capture && !record->external)
        throw std::runtime_error("Graph-internal SYCL events require external=True for host access");
}

sycl::event event_completion(const SyclEventRecord& record)
{
    if (record.capture) {
        if (!record.launched)
            throw std::runtime_error("The SYCL graph must be launched before using its recorded events");
        return record.completion;
    }
    return record.event;
}

void wait_streams(SyclDevice& state)
{
    state.default_stream->queue.wait_and_throw();
    std::lock_guard<std::mutex> lock(state.streams_mutex);
    for (SyclStream* stream : state.streams)
        stream->queue.wait_and_throw();
}

void run_deferred_actions(SyclDevice& state)
{
    std::lock_guard<std::mutex> lock(state.memory_mutex);
    if (state.capture)
        return;
    const bool graph_frees = std::any_of(state.graph_allocs.begin(), state.graph_allocs.end(), [](const auto& entry) {
        return entry.second.graph_destroyed && !entry.second.ref_exists;
    });
    if (state.deferred_frees.empty() && state.deferred_graphs.empty() && state.deferred_modules.empty()
        && state.deferred_resources.empty() && !graph_frees)
        return;

    // Escaped arrays may also have been used by eager work on another stream.
    wait_streams(state);
    while (!state.deferred_graphs.empty()) {
        auto& graph = *state.deferred_graphs.back();
        graph.last_event.wait_and_throw();
        graph.segments.clear();
        graph.recording.reset();
        while (!graph.allocations.empty()) {
            state.graph_allocs.at(graph.allocations.back()).graph_destroyed = true;
            graph.allocations.pop_back();
        }
        while (!graph.staged.empty()) {
            sycl::free(graph.staged.back(), graph.queue);
            graph.staged.pop_back();
        }
        state.deferred_graphs.pop_back();
    }
    for (auto it = state.graph_allocs.begin(); it != state.graph_allocs.end();) {
        if (it->second.graph_destroyed && !it->second.ref_exists) {
            sycl::free(it->first, it->second.context);
            it = state.graph_allocs.erase(it);
        } else {
            ++it;
        }
    }
    while (!state.deferred_frees.empty()) {
        const auto& info = state.deferred_frees.back();
        sycl::free(info.ptr, info.context);
        state.deferred_frees.pop_back();
    }
    while (!state.deferred_modules.empty()) {
        if (dlclose(state.deferred_modules.back()))
            throw std::runtime_error(dlerror());
        state.deferred_modules.pop_back();
    }
    state.deferred_resources.clear();
}

}

void wp::sycl_defer_resource(void* device, std::shared_ptr<void> resource)
{
    auto& state = get_device(device);
    {
        std::lock_guard<std::mutex> lock(state.memory_mutex);
        state.deferred_resources.push_back(std::move(resource));
    }
    run_deferred_actions(state);
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

int wp_sycl_device_compute_units(void* device) { return get_device(device).compute_units; }
size_t wp_sycl_device_resident_threads(void* device) { return get_device(device).resident_threads; }

int wp_sycl_device_memory_info(void* device, size_t* free_memory, size_t* total_memory)
{
    return wp::sycl_checked([&] {
        auto& state = get_device(device);
        *total_memory = state.total_memory;
        *free_memory = state.has_free_memory
            ? state.current->queue.get_device().get_info<sycl::ext::intel::info::device::free_memory>()
            : 0;
    });
}

int wp_sycl_device_has_fp64(void* device) { return get_device(device).has_fp64; }
int wp_sycl_device_has_atomic64(void* device) { return get_device(device).has_atomic64; }

void* wp_sycl_queue(void* device) { return &get_device(device).current->queue; }

void* wp_sycl_alloc(void* device, size_t size)
{
    void* result = nullptr;
    wp::sycl_checked([&] {
        if (size > SIZE_MAX - 3)
            throw std::bad_alloc();
        auto& state = get_device(device);
        std::lock_guard<std::mutex> lock(state.memory_mutex);
        auto& queue = state.current->queue;
        // Sub-word atomics use a containing 32-bit CAS word, including the last array element.
        void* ptr = sycl::malloc_device((size + 3) & ~size_t(3), queue);
        if (!ptr)
            throw std::bad_alloc();
        if (state.capture) {
            try {
                state.graph_allocs.emplace(ptr, SyclGraphAllocInfo { queue.get_context(), size });
                state.capture->allocations.push_back(ptr);
            } catch (...) {
                state.graph_allocs.erase(ptr);
                sycl::free(ptr, queue);
                throw;
            }
        }
        result = ptr;
    });
    return result;
}

int wp_sycl_free(void* device, void* ptr)
{
    return wp::sycl_checked([&] {
        auto& state = get_device(device);
        {
            std::lock_guard<std::mutex> lock(state.memory_mutex);
            auto it = state.graph_allocs.find(ptr);
            if (it != state.graph_allocs.end()) {
                if (!it->second.ref_exists)
                    throw std::runtime_error("SYCL graph allocation has already been released");
                it->second.ref_exists = false;
                if (!it->second.graph_destroyed)
                    return;
            } else {
                state.deferred_frees.push_back(SyclFreeInfo { ptr, state.current->queue.get_context() });
            }
        }
        run_deferred_actions(state);
    });
}

int wp_sycl_graph_memory_info(void* device, size_t* allocations, size_t* bytes)
{
    return wp::sycl_checked([&] {
        auto& state = get_device(device);
        std::lock_guard<std::mutex> lock(state.memory_mutex);
        *allocations = state.graph_allocs.size();
        *bytes = 0;
        for (const auto& entry : state.graph_allocs)
            *bytes += entry.second.size;
    });
}

int wp_sycl_memcpy(void* device, void* dest, const void* src, size_t size, int synchronous)
{
    return wp::sycl_checked([&] {
        if (synchronous && get_device(device).capture)
            throw std::runtime_error("Host transfers are not supported during SYCL graph capture");
        auto event = get_device(device).current->queue.memcpy(dest, src, size);
        if (synchronous)
            event.wait_and_throw();
    });
}

int wp_sycl_memset(void* device, void* dest, int value, size_t size)
{
    return wp::sycl_checked([&] { get_device(device).current->queue.memset(dest, value, size); });
}

int wp_sycl_memtile(void* device, void* dest, const void* src, size_t size, size_t count)
{
    return wp::sycl_checked([&] {
        if (!size || !count)
            return;
        auto& queue = get_device(device).current->queue;
        if (size <= 256) {
            std::array<unsigned char, 256> pattern {};
            std::memcpy(pattern.data(), src, size);
            auto* output = static_cast<unsigned char*>(dest);
            queue.parallel_for(sycl::range<1>(size * count), [=](sycl::id<1> i) { output[i] = pattern[i % size]; });
            return;
        }
        if (get_device(device).capture)
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

void* wp_sycl_stage_args(void* device, const void* src, size_t size)
{
    void* result = nullptr;
    wp::sycl_checked([&] {
        auto& state = get_device(device);
        auto& stream = *state.current;
        auto& queue = stream.queue;
        if (size > SyclStream::stage_slot_bytes)
            throw std::runtime_error("SYCL kernel arguments exceed the staging limit of 16384 bytes");

        if (state.capture) {
            // The recording queue only defers work, so a separate queue performs the copy immediately.
            if (!stream.stage_queue)
                stream.stage_queue = std::make_unique<sycl::queue>(queue.get_context(), queue.get_device());
            auto* copy = sycl::malloc_device(size, queue);
            if (!copy)
                throw std::bad_alloc();
            try {
                stream.stage_queue->memcpy(copy, src, size).wait_and_throw();
            } catch (...) {
                sycl::free(copy, queue);
                throw;
            }
            state.capture->retain_staged(copy);
            result = copy;
            return;
        }

        std::lock_guard<std::mutex> lock(stream.stage_mutex);
        if (!stream.stage_host) {
            const size_t bytes = SyclStream::stage_slot_bytes * SyclStream::stage_slot_count;
            stream.stage_host = static_cast<unsigned char*>(sycl::malloc_host(bytes, queue));
            stream.stage_device = static_cast<unsigned char*>(sycl::malloc_device(bytes, queue));
            if (!stream.stage_host || !stream.stage_device)
                throw std::bad_alloc();
        }
        const size_t slot = stream.stage_next;
        stream.stage_next = (slot + 1) % SyclStream::stage_slot_count;
        const size_t offset = slot * SyclStream::stage_slot_bytes;
        // The host slot may be overwritten only after its previous transfer has completed.
        stream.stage_events[slot].wait_and_throw();
        std::memcpy(stream.stage_host + offset, src, size);
        stream.stage_events[slot] = queue.memcpy(stream.stage_device + offset, stream.stage_host + offset, size);
        result = stream.stage_device + offset;
    });
    return result;
}

void* wp_sycl_scratch(void* device, size_t size)
{
    void* result = nullptr;
    wp::sycl_checked([&] {
        auto& state = get_device(device);
        auto& stream = *state.current;
        auto& queue = stream.queue;
        if (size == 0)
            size = 1;

        if (state.capture) {
            // Replays run after this call returns, so recorded work needs scratch that lives as long as the graph.
            void* buffer = sycl::malloc_device(size, queue);
            if (!buffer)
                throw std::bad_alloc();
            state.capture->retain_staged(buffer);
            result = buffer;
            return;
        }

        std::lock_guard<std::mutex> lock(stream.scratch_mutex);
        if (stream.scratch_bytes < size) {
            const size_t bytes = std::max(size, stream.scratch_bytes + stream.scratch_bytes / 2);
            // Earlier work on this stream may still read the old buffer.
            queue.wait_and_throw();
            if (stream.scratch)
                sycl::free(stream.scratch, queue);
            stream.scratch = nullptr;
            stream.scratch_bytes = 0;
            stream.scratch = sycl::malloc_device(bytes, queue);
            if (!stream.scratch)
                throw std::bad_alloc();
            stream.scratch_bytes = bytes;
        }
        result = stream.scratch;
    });
    return result;
}

void* wp_sycl_stream_create(void* device)
{
    void* result = nullptr;
    wp::sycl_checked([&] {
        auto& state = get_device(device);
        auto stream = std::make_unique<SyclStream>(make_queue(state.default_stream->queue.get_device()));
        std::lock_guard<std::mutex> lock(state.streams_mutex);
        state.streams.push_back(stream.get());
        result = stream.release();
    });
    return result;
}

void* wp_sycl_stream_default(void* device) { return get_device(device).default_stream.get(); }

void* wp_sycl_stream_current(void* device) { return get_device(device).current; }

int wp_sycl_stream_destroy(void* device, void* handle)
{
    return wp::sycl_checked([&] {
        auto& state = get_device(device);
        auto* stream = static_cast<SyclStream*>(handle);
        if (stream == state.default_stream.get())
            throw std::runtime_error("Cannot destroy the default SYCL stream");
        if (state.capture && state.current == stream)
            throw std::runtime_error("Cannot destroy a SYCL stream during graph capture");
        if (state.current == stream)
            state.current = state.default_stream.get();
        {
            std::lock_guard<std::mutex> lock(state.streams_mutex);
            state.streams.erase(std::remove(state.streams.begin(), state.streams.end(), stream), state.streams.end());
        }
        delete stream;
    });
}

int wp_sycl_stream_set_current(void* device, void* handle, int sync)
{
    return wp::sycl_checked([&] {
        auto& state = get_device(device);
        auto* stream = static_cast<SyclStream*>(handle);
        if (state.capture)
            throw std::runtime_error("Cannot change the current SYCL stream during graph capture");
        if (sync && stream != state.current)
            stream->queue.ext_oneapi_submit_barrier({ state.current->queue.ext_oneapi_submit_barrier() });
        state.current = stream;
    });
}

int wp_sycl_stream_synchronize(void* device, void* handle)
{
    return wp::sycl_checked([&] {
        if (get_device(device).capture)
            throw std::runtime_error("Cannot synchronize during SYCL graph capture");
        static_cast<SyclStream*>(handle)->queue.wait_and_throw();
        run_deferred_actions(get_device(device));
    });
}

int wp_sycl_stream_query(void* handle) { return static_cast<SyclStream*>(handle)->queue.ext_oneapi_empty() ? 0 : 1; }

void* wp_sycl_event_create(void* device, int enable_timing)
{
    void* result = nullptr;
    wp::sycl_checked([&] { result = new SyclEvent { device, enable_timing != 0 }; });
    return result;
}

void wp_sycl_event_destroy(void* handle) { delete static_cast<SyclEvent*>(handle); }

int wp_sycl_event_record(void* device, void* event, void* stream, int external)
{
    return wp::sycl_checked([&] {
        auto& state = get_device(device);
        auto* e = static_cast<SyclEvent*>(event);
        auto& queue = static_cast<SyclStream*>(stream)->queue;
        auto record = std::make_shared<SyclEventRecord>(queue.get_context());
        if (state.capture) {
            auto& graph = *state.capture;
            if (queue != graph.queue)
                throw std::runtime_error("SYCL event recording during capture must use the capturing stream");
            record->capture = graph.capture;
            record->external = external != 0 || e->timing;
            if (record->external && (!state.has_graph_events || !graph.native_list))
                throw std::runtime_error(
                    "Captured SYCL events require Level Zero v2 native recording on an immediate queue; "
                    "select a supported device and set SYCL_UR_USE_LEVEL_ZERO_V2=1 before starting the process"
                );
            if (e->timing) {
                // Host USM avoids migration faults when Python has replaced the driver's SIGSEGV handler.
                record->timestamp = sycl::malloc_host<uint64_t>(1, queue);
                if (!record->timestamp)
                    throw std::bad_alloc();
                *record->timestamp = 0;
            }
            // The graph owns each recording independently of subsequent event re-recording/destruction.
            graph.events.push_back({ e->state, record });
            if (record->timestamp) {
                // Native recording captures these queue commands directly, without command-buffer callbacks.
                if (zeCommandListAppendBarrier(graph.native_list, nullptr, 0, nullptr) != ZE_RESULT_SUCCESS
                    || zeCommandListAppendWriteGlobalTimestamp(
                           graph.native_list, record->timestamp, nullptr, 0, nullptr
                       ) != ZE_RESULT_SUCCESS
                    || zeCommandListAppendBarrier(graph.native_list, nullptr, 0, nullptr) != ZE_RESULT_SUCCESS)
                    throw std::runtime_error("Cannot record a Level Zero queue timestamp");
            }
            record->event = queue.ext_oneapi_submit_barrier();
        } else {
            record->event = queue.ext_oneapi_submit_barrier();
        }
        e->state->record = std::move(record);
    });
}

int wp_sycl_stream_wait_event(void* device, void* stream, void* event, int external)
{
    return wp::sycl_checked([&] {
        auto& state = get_device(device);
        auto* e = static_cast<SyclEvent*>(event);
        auto& queue = static_cast<SyclStream*>(stream)->queue;
        const auto& record = e->state->record;
        if (!record)
            return;
        if (state.capture) {
            if (queue != state.capture->queue)
                throw std::runtime_error("SYCL event waits during capture must use the capturing stream");
            if (record->capture != state.capture->capture)
                throw std::runtime_error(
                    external ? "External event waits between SYCL graphs are not supported"
                             : "SYCL capture can only wait on events recorded in the same graph"
                );
            // Capture uses one in-order queue, including across native graph segments.
            queue.ext_oneapi_submit_barrier();
        } else {
            check_event_access(*e);
            queue.ext_oneapi_submit_barrier({ event_completion(*record) });
        }
    });
}

int wp_sycl_event_synchronize(void* event)
{
    return wp::sycl_checked([&] {
        auto* e = static_cast<SyclEvent*>(event);
        check_event_access(*e);
        if (e->state->record)
            event_completion(*e->state->record).wait_and_throw();
    });
}

int wp_sycl_event_query(void* event)
{
    int result = 0;
    const int status = wp::sycl_checked([&] {
        auto* e = static_cast<SyclEvent*>(event);
        check_event_access(*e);
        const auto& record = e->state->record;
        if (!record)
            return;
        if (record->capture && !record->launched) {
            result = 1;
            return;
        }
        result = event_completion(*record).get_info<sycl::info::event::command_execution_status>()
                == sycl::info::event_command_status::complete
            ? 0
            : 1;
    });
    return status ? status : result;
}

int wp_sycl_event_elapsed_time(void* start, void* end, float* milliseconds)
{
    return wp::sycl_checked([&] {
        auto* s = static_cast<SyclEvent*>(start);
        auto* e = static_cast<SyclEvent*>(end);
        if (s->device != e->device)
            throw std::runtime_error("SYCL timing events must belong to the same device");
        check_event_access(*s);
        check_event_access(*e);
        if (!s->timing || !e->timing)
            throw std::runtime_error("Events must be created with enable_timing=True to measure elapsed time");
        const auto& start_record = s->state->record;
        const auto& end_record = e->state->record;
        if (!start_record || !end_record)
            throw std::runtime_error("Events must be recorded before measuring elapsed time");
        for (const auto& record : { start_record, end_record }) {
            if (event_completion(*record).get_info<sycl::info::event::command_execution_status>()
                != sycl::info::event_command_status::complete)
                throw std::runtime_error("SYCL timing events have not completed");
        }
        auto& device = get_device(s->device);
        if (start_record->timestamp && end_record->timestamp) {
            const auto ticks = (*end_record->timestamp - *start_record->timestamp) & device.timestamp_mask;
            *milliseconds = static_cast<float>(static_cast<long double>(ticks) * 1000 / device.timer_hz);
        } else {
            auto nanoseconds = [&](const SyclEventRecord& record) -> long double {
                if (record.timestamp)
                    return static_cast<long double>(*record.timestamp & device.timestamp_mask) * 1e9L / device.timer_hz;
                return record.event.get_profiling_info<sycl::info::event_profiling::command_end>();
            };
            auto elapsed = nanoseconds(*end_record) - nanoseconds(*start_record);
            if (start_record->timestamp || end_record->timestamp) {
                const auto period = std::ldexp(1.0L, device.timestamp_bits) * 1e9L / device.timer_hz;
                elapsed = std::fmod(elapsed, period);
                if (elapsed < 0)
                    elapsed += period;
            }
            *milliseconds = static_cast<float>(elapsed * 1e-6L);
        }
    });
}

int wp_sycl_synchronize(void* device)
{
    return wp::sycl_checked([&] {
        auto& state = get_device(device);
        if (state.capture)
            throw std::runtime_error("Cannot synchronize during SYCL graph capture");
        wait_streams(state);
        run_deferred_actions(state);
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
        auto& state = get_device(device);
        {
            std::lock_guard<std::mutex> lock(state.memory_mutex);
            state.deferred_modules.push_back(module);
        }
        run_deferred_actions(state);
    });
}

int wp_sycl_graph_supported(void* device) { return get_device(device).has_graphs; }

int wp_sycl_graph_event_supported(void* device)
{
    auto& state = get_device(device);
    return state.has_graph_events && state.current->native_list;
}

void* wp_sycl_graph_begin(void* device)
{
    void* result = nullptr;
    wp::sycl_checked([&] {
        auto& state = get_device(device);
        if (!wp_sycl_graph_supported(device))
            throw std::runtime_error("SYCL command graphs are unavailable on this device");
        run_deferred_actions(state);
        std::lock_guard<std::mutex> lock(state.memory_mutex);
        if (state.capture)
            throw std::runtime_error("SYCL graph capture already active");
        auto graph = std::make_unique<SyclGraph>(
            state.current->queue, state.has_native_graphs ? state.current->native_list : nullptr
        );
        graph->begin_segment();
        graph->active = true;
        state.capture = graph.get();
        result = graph.release();
    });
    return result;
}

int wp_sycl_graph_end(void* device, void* handle)
{
    return wp::sycl_checked([&] {
        auto& state = get_device(device);
        {
            std::lock_guard<std::mutex> lock(state.memory_mutex);
            auto& graph = *static_cast<SyclGraph*>(handle);
            if (state.capture != &graph || !graph.active)
                throw std::runtime_error("SYCL graph capture is not active");
            graph.finish_segment();
            graph.active = false;
            graph.capture->active = false;
            state.capture = nullptr;
        }
        run_deferred_actions(state);
    });
}

int wp_sycl_graph_launch(void* device, void* handle)
{
    return wp::sycl_checked([&] {
        auto& graph = *static_cast<SyclGraph*>(handle);
        if (graph.active || graph.segments.empty())
            throw std::runtime_error("SYCL graph has not been finalized");
        auto& state = get_device(device);
        if (state.capture) {
            auto& parent = *state.capture;
            if (parent.native_list) {
                // Native graph launches are not captured by all drivers. Retain ordered GPU segments instead.
                parent.finish_segment();
                parent.segments.insert(parent.segments.end(), graph.segments.begin(), graph.segments.end());
                parent.begin_segment();
            } else {
                if (graph.native_list)
                    throw std::runtime_error("Cannot nest native SYCL graphs in a command-buffer capture");
                state.current->queue.ext_oneapi_graph(*graph.segments.front());
            }
            // Retain nested markers in execution order so their events follow the parent replay.
            parent.events.insert(parent.events.end(), graph.events.begin(), graph.events.end());
        } else {
            std::vector<sycl::event> dependencies;
            for (const auto& marker : graph.events) {
                if (marker.record->launched
                    && std::find(dependencies.begin(), dependencies.end(), marker.record->completion)
                        == dependencies.end())
                    dependencies.push_back(marker.record->completion);
            }
            for (const auto& segment : graph.segments) {
                graph.last_event = state.current->queue.submit([&](sycl::handler& handler) {
                    if (!dependencies.empty())
                        handler.depends_on(dependencies);
                    handler.ext_oneapi_graph(*segment);
                });
                dependencies.clear();
            }
            for (const auto& marker : graph.events) {
                marker.record->completion = graph.last_event;
                marker.record->launched = true;
                if (auto state = marker.state.lock())
                    state->record = marker.record;
            }
        }
    });
}

int wp_sycl_graph_destroy(void* device, void* handle)
{
    return wp::sycl_checked([&] {
        auto& state = get_device(device);
        auto* graph = static_cast<SyclGraph*>(handle);
        {
            std::lock_guard<std::mutex> lock(state.memory_mutex);
            if (graph->active) {
                if (state.capture != graph)
                    throw std::runtime_error("SYCL graph does not own the active capture");
                if (graph->recording)
                    graph->recording->end_recording(graph->queue);
                graph->active = false;
                graph->capture->active = false;
                state.capture = nullptr;
            }
            state.deferred_graphs.emplace_back(graph);
        }
        run_deferred_actions(state);
    });
}

namespace {

using Array = wp::array_t<char>;

inline char* array_element(Array array, size_t index)
{
    ptrdiff_t offset = 0;
    for (int d = array.ndim - 1; d >= 0; --d) {
        const size_t coordinate = index % array.shape[d];
        index /= array.shape[d];
        offset += ptrdiff_t(coordinate) * array.strides[d];
    }
    return array.data + offset;
}

// Type codes shared with warp/_src/sycl.py.
enum ScanType { SCAN_INT32 = 0, SCAN_INT64 = 1, SCAN_FLOAT32 = 2, SCAN_FLOAT64 = 3 };

}  // anonymous namespace

int wp_sycl_array_copy(
    void* device, const void* dst, const void* src, size_t dst_offset, size_t src_offset, size_t count, size_t size
)
{
    return wp::sycl_checked([&] {
        const Array dest = *static_cast<const Array*>(dst);
        const Array source = *static_cast<const Array*>(src);
        wp::sycl_queue(device).parallel_for(sycl::range<1>(count), [=](sycl::id<1> i) {
            char* d = array_element(dest, dst_offset + i);
            const char* s = array_element(source, src_offset + i);
            for (size_t byte = 0; byte < size; ++byte)
                d[byte] = s[byte];
        });
    });
}

int wp_sycl_array_fill(void* device, const void* descriptor, const void* value, size_t size, size_t count)
{
    return wp::sycl_checked([&] {
        if (size > 256)
            throw std::runtime_error("Non-contiguous SYCL fills support values of at most 256 bytes");
        const Array array = *static_cast<const Array*>(descriptor);
        std::array<char, 256> pattern {};
        std::memcpy(pattern.data(), value, size);
        wp::sycl_queue(device).parallel_for(sycl::range<1>(count), [=](sycl::id<1> i) {
            char* dst = array_element(array, i);
            for (size_t byte = 0; byte < size; ++byte)
                dst[byte] = pattern[byte];
        });
    });
}

int wp_sycl_array_scan(
    void* device,
    int type,
    const void* in,
    void* out,
    int n,
    int in_stride,
    int out_stride,
    int type_length,
    int inclusive
)
{
    return wp::sycl_checked([&] {
        if (n <= 0)
            return;
        if (type_length <= 0)
            throw std::runtime_error("SYCL scan requires a positive type length");
        switch (type) {
        case SCAN_INT32:
            scan_sycl(device, (const int*)in, (int*)out, n, in_stride, out_stride, type_length, inclusive != 0);
            break;
        case SCAN_INT64:
            scan_sycl(device, (const int64_t*)in, (int64_t*)out, n, in_stride, out_stride, type_length, inclusive != 0);
            break;
        case SCAN_FLOAT32:
            scan_sycl(device, (const float*)in, (float*)out, n, in_stride, out_stride, type_length, inclusive != 0);
            break;
        case SCAN_FLOAT64:
            if (!wp_sycl_device_has_fp64(device))
                throw std::runtime_error("SYCL device does not support float64 scans");
            scan_sycl(device, (const double*)in, (double*)out, n, in_stride, out_stride, type_length, inclusive != 0);
            break;
        default:
            throw std::runtime_error("Unsupported SYCL scan type");
        }
    });
}

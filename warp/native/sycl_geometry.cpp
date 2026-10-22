// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "warp.h"

#include "sycl_runtime.h"

#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace {

struct GeometryStorage {
    void* device;
    std::vector<void*> allocations;

    explicit GeometryStorage(void* device)
        : device(device)
    {
        if (!wp::sycl_queue(device).get_device().has(sycl::aspect::usm_shared_allocations))
            throw std::runtime_error("SYCL geometry requires shared USM support");
    }

    ~GeometryStorage()
    {
        for (void* ptr : allocations) {
            try {
                sycl::free(ptr, wp::sycl_queue(device));
            } catch (const std::exception& error) {
                fprintf(stderr, "Warp SYCL geometry cleanup failed: %s\n", error.what());
            }
        }
    }

    template <typename T> T* copy(const T* source, size_t count)
    {
        if (!source || !count)
            return nullptr;
        T* result = sycl::malloc_shared<T>(count, wp::sycl_queue(device));
        if (!result)
            throw std::bad_alloc();
        try {
            allocations.push_back(result);
        } catch (...) {
            sycl::free(result, wp::sycl_queue(device));
            throw;
        }
        std::memcpy(result, source, sizeof(T) * count);
        return result;
    }

    wp::BVH copy_bvh(const wp::BVH& host)
    {
        wp::BVH result = host;
        if (!host.num_items) {
            // Host builders leave empty trees without a root; queries still dereference it.
            const wp::bounds3 empty;
            const auto lower = wp::make_node(empty.lower, 0, true);
            const auto upper = wp::make_node(empty.upper, 0, true);
            const int root = 0;
            const int parent = -1;
            result.node_lowers = copy(&lower, 1);
            result.node_uppers = copy(&upper, 1);
            result.node_parents = copy(&parent, 1);
            result.root = copy(&root, 1);
            result.num_nodes = result.max_nodes = 1;
            result.context = device;
            return result;
        }
        result.node_lowers = copy(host.node_lowers, host.max_nodes);
        result.node_uppers = copy(host.node_uppers, host.max_nodes);
        result.node_parents = copy(host.node_parents, host.max_nodes);
        result.primitive_indices = copy(host.primitive_indices, host.num_items);
        result.root = copy(host.root, 1);
        result.context = device;
        return result;
    }
};

template <typename T> void read_array(void* device, std::vector<T>& host, const T* source, size_t count)
{
    host.resize(count);
    if (count && source)
        wp::sycl_queue(device).memcpy(host.data(), source, sizeof(T) * count).wait_and_throw();
}

void check_constructor(int constructor)
{
    if (constructor != BVH_CONSTRUCTOR_SAH && constructor != BVH_CONSTRUCTOR_MEDIAN)
        throw std::runtime_error("SYCL geometry supports the SAH and median host constructors only");
}

void refit_bvh(void* device, wp::BVH bvh)
{
    if (bvh.num_items == 0)
        return;
    auto& queue = wp::sycl_queue(device);
    for (int level = bvh.max_depth; level >= 0; --level) {
        queue.parallel_for(sycl::range<1>(bvh.num_nodes), [=](sycl::id<1> index) {
            const int node = index[0];
            int depth = 0;
            for (int parent = bvh.node_parents[node]; parent >= 0; parent = bvh.node_parents[parent])
                ++depth;
            if (depth != level)
                return;
            auto& lower = bvh.node_lowers[node];
            auto& upper = bvh.node_uppers[node];
            wp::bounds3 bounds;
            if (lower.b) {
                for (unsigned i = lower.i; i < upper.i; ++i) {
                    const int primitive = bvh.primitive_indices[i];
                    bounds.add_bounds(bvh.item_lowers[primitive], bvh.item_uppers[primitive]);
                }
            } else {
                const auto& ll = bvh.node_lowers[lower.i];
                const auto& lu = bvh.node_uppers[lower.i];
                const auto& rl = bvh.node_lowers[upper.i];
                const auto& ru = bvh.node_uppers[upper.i];
                bounds.add_bounds(wp::vec3(ll.x, ll.y, ll.z), wp::vec3(lu.x, lu.y, lu.z));
                bounds.add_bounds(wp::vec3(rl.x, rl.y, rl.z), wp::vec3(ru.x, ru.y, ru.z));
            }
            lower = wp::make_node(bounds.lower, lower.i, lower.b);
            upper = wp::make_node(bounds.upper, upper.i, upper.b);
        });
    }
}

struct BvhState : GeometryStorage {
    uint64_t host = 0;
    wp::BVH* gpu = nullptr;
    std::vector<wp::vec3> lowers, uppers;
    std::vector<int> groups;

    using GeometryStorage::GeometryStorage;
    ~BvhState()
    {
        if (host)
            wp_bvh_destroy_host(host);
    }
};

struct MeshState : GeometryStorage {
    uint64_t host = 0;
    wp::Mesh* gpu = nullptr;
    std::vector<wp::vec3> points, velocities;
    std::vector<int> indices, groups;

    using GeometryStorage::GeometryStorage;
    ~MeshState()
    {
        if (host)
            wp_mesh_destroy_host(host);
    }
};

std::mutex geometry_mutex;
std::unordered_map<uint64_t, std::unique_ptr<BvhState>> bvhs;
std::unordered_map<uint64_t, std::unique_ptr<MeshState>> meshes;

void update_bvh_nodes(wp::BVH& gpu, const wp::BVH& host)
{
    if (!host.num_items)
        return;
    std::memcpy(gpu.node_lowers, host.node_lowers, host.max_nodes * sizeof(wp::BVHPackedNodeHalf));
    std::memcpy(gpu.node_uppers, host.node_uppers, host.max_nodes * sizeof(wp::BVHPackedNodeHalf));
    std::memcpy(gpu.node_parents, host.node_parents, host.max_nodes * sizeof(int));
    std::memcpy(gpu.primitive_indices, host.primitive_indices, host.num_items * sizeof(int));
    *gpu.root = *host.root;
    gpu.num_nodes = host.num_nodes;
    gpu.num_leaf_nodes = host.num_leaf_nodes;
    gpu.max_depth = host.max_depth;
    gpu.constructor_type = host.constructor_type;
}

void refit_mesh(MeshState& state)
{
    auto* mesh = state.gpu;
    if (!mesh->num_tris)
        return;
    auto& queue = wp::sycl_queue(state.device);
    if (mesh->solid_angle_props) {
        if (wp_sycl_synchronize(state.device))
            throw std::runtime_error(wp_sycl_get_error_string());
        read_array(state.device, state.points, mesh->points.data, mesh->num_points);
        read_array(state.device, state.indices, mesh->indices.data, mesh->num_tris * 3);
        wp_mesh_refit_host(state.host);
        const auto* host = reinterpret_cast<const wp::Mesh*>(state.host);
        update_bvh_nodes(mesh->bvh, host->bvh);
        std::memcpy(mesh->lowers, host->lowers, mesh->num_tris * sizeof(wp::vec3));
        std::memcpy(mesh->uppers, host->uppers, mesh->num_tris * sizeof(wp::vec3));
        std::memcpy(
            mesh->solid_angle_props, host->solid_angle_props, (2 * mesh->num_tris - 1) * sizeof(wp::SolidAngleProps)
        );
        mesh->average_edge_length = host->average_edge_length;
        return;
    }
    queue.parallel_for(sycl::range<1>(mesh->num_tris), [=](sycl::id<1> i) {
        const wp::vec3 p0 = mesh->points[mesh->indices[i * 3]];
        const wp::vec3 p1 = mesh->points[mesh->indices[i * 3 + 1]];
        const wp::vec3 p2 = mesh->points[mesh->indices[i * 3 + 2]];
        wp::bounds3 bounds;
        bounds.add_point(p0);
        bounds.add_point(p1);
        bounds.add_point(p2);
        mesh->lowers[i] = bounds.lower;
        mesh->uppers[i] = bounds.upper;
    });
    queue.single_task([=] {
        float sum = 0.0f;
        for (int i = 0; i < mesh->num_tris; ++i) {
            const auto p0 = mesh->points[mesh->indices[i * 3]];
            const auto p1 = mesh->points[mesh->indices[i * 3 + 1]];
            const auto p2 = mesh->points[mesh->indices[i * 3 + 2]];
            sum += wp::length(p0 - p1) + wp::length(p0 - p2) + wp::length(p1 - p2);
        }
        mesh->average_edge_length = sum / (mesh->num_tris * 3);
    });
    refit_bvh(state.device, mesh->bvh);
}

}

extern "C" {

WP_API uint64_t wp_sycl_bvh_create(
    void* device, wp::vec3* lowers, wp::vec3* uppers, int count, int constructor, int* groups, int leaf_size
)
{
    uint64_t id = 0;
    wp::sycl_checked([&] {
        check_constructor(constructor);
        if (wp_sycl_synchronize(device))
            throw std::runtime_error(wp_sycl_get_error_string());
        auto state = std::make_unique<BvhState>(device);
        read_array(device, state->lowers, lowers, count);
        read_array(device, state->uppers, uppers, count);
        if (groups)
            read_array(device, state->groups, groups, count);
        state->host = wp_bvh_create_host(
            state->lowers.data(), state->uppers.data(), count, constructor, groups ? state->groups.data() : nullptr,
            leaf_size
        );
        if (!state->host)
            throw std::runtime_error(wp_get_error_string());
        wp::BVH tree = state->copy_bvh(*reinterpret_cast<wp::BVH*>(state->host));
        tree.item_lowers = lowers;
        tree.item_uppers = uppers;
        tree.item_groups = groups;
        state->gpu = state->copy(&tree, 1);
        const auto handle = reinterpret_cast<uint64_t>(state->gpu);
        std::lock_guard<std::mutex> lock(geometry_mutex);
        bvhs.emplace(handle, std::move(state));
        id = handle;
    });
    return id;
}

WP_API int wp_sycl_bvh_destroy(uint64_t id)
{
    return wp::sycl_checked([&] {
        std::lock_guard<std::mutex> lock(geometry_mutex);
        auto& state = *bvhs.at(id);
        if (wp_sycl_synchronize(state.device))
            throw std::runtime_error(wp_sycl_get_error_string());
        bvhs.erase(id);
    });
}

WP_API int wp_sycl_bvh_refit(uint64_t id)
{
    return wp::sycl_checked([&] {
        std::lock_guard<std::mutex> lock(geometry_mutex);
        auto& state = *bvhs.at(id);
        refit_bvh(state.device, *state.gpu);
    });
}

WP_API int wp_sycl_bvh_rebuild(uint64_t id, int constructor)
{
    return wp::sycl_checked([&] {
        check_constructor(constructor);
        std::lock_guard<std::mutex> lock(geometry_mutex);
        auto& state = *bvhs.at(id);
        if (wp_sycl_synchronize(state.device))
            throw std::runtime_error(wp_sycl_get_error_string());
        read_array(state.device, state.lowers, state.gpu->item_lowers, state.gpu->num_items);
        read_array(state.device, state.uppers, state.gpu->item_uppers, state.gpu->num_items);
        if (state.gpu->item_groups)
            read_array(state.device, state.groups, state.gpu->item_groups, state.gpu->num_items);
        wp_bvh_rebuild_host(state.host, constructor);
        update_bvh_nodes(*state.gpu, *reinterpret_cast<wp::BVH*>(state.host));
    });
}

WP_API uint64_t wp_sycl_mesh_create(
    void* device,
    wp::array_t<wp::vec3> points,
    wp::array_t<wp::vec3> velocities,
    wp::array_t<int> indices,
    int num_points,
    int num_tris,
    int winding,
    int constructor,
    int* groups,
    int leaf_size
)
{
    uint64_t id = 0;
    wp::sycl_checked([&] {
        check_constructor(constructor);
        if (winding && !num_tris)
            throw std::runtime_error("Winding-number support requires a nonempty SYCL mesh");
        if (wp_sycl_synchronize(device))
            throw std::runtime_error(wp_sycl_get_error_string());
        auto state = std::make_unique<MeshState>(device);
        read_array(device, state->points, points.data, num_points);
        read_array(device, state->indices, indices.data, num_tris * 3);
        if (velocities.data)
            read_array(device, state->velocities, velocities.data, num_points);
        if (groups)
            read_array(device, state->groups, groups, num_tris);
        auto host_points = points;
        auto host_velocities = velocities;
        auto host_indices = indices;
        host_points.data = state->points.data();
        host_velocities.data = velocities.data ? state->velocities.data() : nullptr;
        host_indices.data = state->indices.data();
        state->host = wp_mesh_create_host(
            host_points, host_velocities, host_indices, num_points, num_tris, winding, constructor,
            groups ? state->groups.data() : nullptr, leaf_size
        );
        if (!state->host)
            throw std::runtime_error(wp_get_error_string());
        auto mesh = *reinterpret_cast<wp::Mesh*>(state->host);
        mesh.points = points;
        mesh.velocities = velocities;
        mesh.indices = indices;
        mesh.lowers = state->copy(mesh.lowers, num_tris);
        mesh.uppers = state->copy(mesh.uppers, num_tris);
        mesh.solid_angle_props = state->copy(mesh.solid_angle_props, num_tris ? 2 * num_tris - 1 : 0);
        mesh.bvh = state->copy_bvh(mesh.bvh);
        mesh.bvh.item_lowers = mesh.lowers;
        mesh.bvh.item_uppers = mesh.uppers;
        mesh.bvh.item_groups = groups;
        mesh.context = device;
        state->gpu = state->copy(&mesh, 1);
        const auto handle = reinterpret_cast<uint64_t>(state->gpu);
        std::lock_guard<std::mutex> lock(geometry_mutex);
        meshes.emplace(handle, std::move(state));
        id = handle;
    });
    return id;
}

WP_API int wp_sycl_mesh_destroy(uint64_t id)
{
    return wp::sycl_checked([&] {
        std::lock_guard<std::mutex> lock(geometry_mutex);
        if (wp_sycl_synchronize(meshes.at(id)->device))
            throw std::runtime_error(wp_sycl_get_error_string());
        meshes.erase(id);
    });
}

WP_API int wp_sycl_mesh_refit(uint64_t id)
{
    return wp::sycl_checked([&] {
        std::lock_guard<std::mutex> lock(geometry_mutex);
        refit_mesh(*meshes.at(id));
    });
}

WP_API int wp_sycl_mesh_set_points(uint64_t id, wp::array_t<wp::vec3> points)
{
    return wp::sycl_checked([&] {
        std::lock_guard<std::mutex> lock(geometry_mutex);
        auto& state = *meshes.at(id);
        if (wp_sycl_synchronize(state.device))
            throw std::runtime_error(wp_sycl_get_error_string());
        const auto previous = state.gpu->points;
        state.gpu->points = points;
        try {
            refit_mesh(state);
            wp::sycl_queue(state.device).wait_and_throw();
        } catch (...) {
            state.gpu->points = previous;
            throw;
        }
    });
}

WP_API int wp_sycl_mesh_set_velocities(uint64_t id, wp::array_t<wp::vec3> velocities)
{
    return wp::sycl_checked([&] {
        std::lock_guard<std::mutex> lock(geometry_mutex);
        auto& state = *meshes.at(id);
        if (wp_sycl_synchronize(state.device))
            throw std::runtime_error(wp_sycl_get_error_string());
        state.gpu->velocities = velocities;
    });
}
}

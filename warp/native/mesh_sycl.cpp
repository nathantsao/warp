// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "warp.h"

#include "bvh.h"
#include "mesh.h"
#include "sycl_runtime.h"

#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace {

struct MeshState : wp::SyclSharedStorage {
    uint64_t host = 0;
    wp::Mesh* gpu = nullptr;
    std::vector<wp::vec3> points, velocities;
    std::vector<int> indices, groups;

    using wp::SyclSharedStorage::SyclSharedStorage;
    ~MeshState()
    {
        if (host)
            wp_mesh_destroy_host(host);
    }
};

std::mutex mesh_mutex;
std::unordered_map<uint64_t, std::unique_ptr<MeshState>> meshes;

void mesh_refit_sycl(MeshState& state)
{
    auto* mesh = state.gpu;
    if (!mesh->num_tris)
        return;
    auto& queue = wp::sycl_queue(state.device);
    if (mesh->solid_angle_props) {
        if (wp_sycl_synchronize(state.device))
            throw std::runtime_error(wp_sycl_get_error_string());
        wp::sycl_read_array(state.device, state.points, mesh->points.data, mesh->num_points);
        wp::sycl_read_array(state.device, state.indices, mesh->indices.data, mesh->num_tris * 3);
        wp_mesh_refit_host(state.host);
        const auto* host = reinterpret_cast<const wp::Mesh*>(state.host);
        wp::bvh_update_sycl(mesh->bvh, host->bvh);
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
    wp::bvh_refit_sycl(state.device, mesh->bvh);
}

}  // anonymous namespace

extern "C" {

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
        wp::bvh_check_constructor_sycl(constructor);
        if (winding && !num_tris)
            throw std::runtime_error("Winding-number support requires a nonempty SYCL mesh");
        if (wp_sycl_synchronize(device))
            throw std::runtime_error(wp_sycl_get_error_string());
        auto state = std::make_unique<MeshState>(device);
        wp::sycl_read_array(device, state->points, points.data, num_points);
        wp::sycl_read_array(device, state->indices, indices.data, num_tris * 3);
        if (velocities.data)
            wp::sycl_read_array(device, state->velocities, velocities.data, num_points);
        if (groups)
            wp::sycl_read_array(device, state->groups, groups, num_tris);
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
        mesh.bvh = wp::copy_host_tree_to_sycl(*state, mesh.bvh);
        mesh.bvh.item_lowers = mesh.lowers;
        mesh.bvh.item_uppers = mesh.uppers;
        mesh.bvh.item_groups = groups;
        mesh.context = device;
        state->gpu = state->copy(&mesh, 1);
        const auto handle = reinterpret_cast<uint64_t>(state->gpu);
        std::lock_guard<std::mutex> lock(mesh_mutex);
        meshes.emplace(handle, std::move(state));
        id = handle;
    });
    return id;
}

WP_API int wp_sycl_mesh_destroy(uint64_t id)
{
    return wp::sycl_checked([&] {
        std::lock_guard<std::mutex> lock(mesh_mutex);
        if (wp_sycl_synchronize(meshes.at(id)->device))
            throw std::runtime_error(wp_sycl_get_error_string());
        meshes.erase(id);
    });
}

WP_API int wp_sycl_mesh_refit(uint64_t id)
{
    return wp::sycl_checked([&] {
        std::lock_guard<std::mutex> lock(mesh_mutex);
        mesh_refit_sycl(*meshes.at(id));
    });
}

WP_API int wp_sycl_mesh_set_points(uint64_t id, wp::array_t<wp::vec3> points)
{
    return wp::sycl_checked([&] {
        std::lock_guard<std::mutex> lock(mesh_mutex);
        auto& state = *meshes.at(id);
        if (wp_sycl_synchronize(state.device))
            throw std::runtime_error(wp_sycl_get_error_string());
        const auto previous = state.gpu->points;
        state.gpu->points = points;
        try {
            mesh_refit_sycl(state);
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
        std::lock_guard<std::mutex> lock(mesh_mutex);
        auto& state = *meshes.at(id);
        if (wp_sycl_synchronize(state.device))
            throw std::runtime_error(wp_sycl_get_error_string());
        state.gpu->velocities = velocities;
    });
}

}  // extern "C"

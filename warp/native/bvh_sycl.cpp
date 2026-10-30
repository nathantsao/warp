// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "warp.h"

#include "bvh.h"
#include "sycl_runtime.h"

#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace {

struct BvhState : wp::SyclSharedStorage {
    uint64_t host = 0;
    wp::BVH* gpu = nullptr;
    wp::BVH descriptor {};
    std::vector<wp::vec3> lowers, uppers;
    std::vector<int> groups;

    using wp::SyclSharedStorage::SyclSharedStorage;
    ~BvhState()
    {
        if (host)
            wp_bvh_destroy_host(host);
    }
};

std::mutex bvh_mutex;
std::unordered_map<uint64_t, std::unique_ptr<BvhState>> bvhs;

}  // anonymous namespace

namespace wp {

void bvh_check_constructor_sycl(int constructor_type)
{
    if (constructor_type != BVH_CONSTRUCTOR_SAH && constructor_type != BVH_CONSTRUCTOR_MEDIAN)
        throw std::runtime_error("SYCL geometry supports the SAH and median host constructors only");
}

BVH copy_host_tree_to_sycl(SyclSharedStorage& storage, const BVH& bvh_host)
{
    BVH result = bvh_host;
    if (!bvh_host.num_items) {
        // Host builders leave empty trees without a root; queries still dereference it.
        const bounds3 empty;
        const auto lower = make_node(empty.lower, 0, true);
        const auto upper = make_node(empty.upper, 0, true);
        const int root = 0;
        const int parent = -1;
        result.node_lowers = storage.copy(&lower, 1);
        result.node_uppers = storage.copy(&upper, 1);
        result.node_parents = storage.copy(&parent, 1);
        result.root = storage.copy(&root, 1);
        result.num_nodes = result.max_nodes = 1;
        result.context = storage.device;
        return result;
    }
    result.node_lowers = storage.copy(bvh_host.node_lowers, bvh_host.max_nodes);
    result.node_uppers = storage.copy(bvh_host.node_uppers, bvh_host.max_nodes);
    result.node_parents = storage.copy(bvh_host.node_parents, bvh_host.max_nodes);
    result.primitive_indices = storage.copy(bvh_host.primitive_indices, bvh_host.num_items);
    result.root = storage.copy(bvh_host.root, 1);
    result.context = storage.device;
    return result;
}

void bvh_update_sycl(BVH& bvh_sycl, const BVH& bvh_host)
{
    if (!bvh_host.num_items)
        return;
    std::memcpy(bvh_sycl.node_lowers, bvh_host.node_lowers, bvh_host.max_nodes * sizeof(BVHPackedNodeHalf));
    std::memcpy(bvh_sycl.node_uppers, bvh_host.node_uppers, bvh_host.max_nodes * sizeof(BVHPackedNodeHalf));
    std::memcpy(bvh_sycl.node_parents, bvh_host.node_parents, bvh_host.max_nodes * sizeof(int));
    std::memcpy(bvh_sycl.primitive_indices, bvh_host.primitive_indices, bvh_host.num_items * sizeof(int));
    *bvh_sycl.root = *bvh_host.root;
    bvh_sycl.num_nodes = bvh_host.num_nodes;
    bvh_sycl.num_leaf_nodes = bvh_host.num_leaf_nodes;
    bvh_sycl.max_depth = bvh_host.max_depth;
    bvh_sycl.constructor_type = bvh_host.constructor_type;
}

void bvh_refit_sycl(void* device, BVH bvh)
{
    if (bvh.num_items == 0)
        return;
    auto& queue = sycl_queue(device);
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
            bounds3 bounds;
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
                bounds.add_bounds(vec3(ll.x, ll.y, ll.z), vec3(lu.x, lu.y, lu.z));
                bounds.add_bounds(vec3(rl.x, rl.y, rl.z), vec3(ru.x, ru.y, ru.z));
            }
            lower = make_node(bounds.lower, lower.i, lower.b);
            upper = make_node(bounds.upper, upper.i, upper.b);
        });
    }
}

}  // namespace wp

extern "C" {

WP_API uint64_t wp_sycl_bvh_create(
    void* device, wp::vec3* lowers, wp::vec3* uppers, int count, int constructor, int* groups, int leaf_size
)
{
    uint64_t id = 0;
    wp::sycl_checked([&] {
        wp::bvh_check_constructor_sycl(constructor);
        if (wp_sycl_synchronize(device))
            throw std::runtime_error(wp_sycl_get_error_string());
        auto state = std::make_unique<BvhState>(device);
        wp::sycl_read_array(device, state->lowers, lowers, count);
        wp::sycl_read_array(device, state->uppers, uppers, count);
        if (groups)
            wp::sycl_read_array(device, state->groups, groups, count);
        state->host = wp_bvh_create_host(
            state->lowers.data(), state->uppers.data(), count, constructor, groups ? state->groups.data() : nullptr,
            leaf_size
        );
        if (!state->host)
            throw std::runtime_error(wp_get_error_string());
        wp::BVH tree = wp::copy_host_tree_to_sycl(*state, *reinterpret_cast<wp::BVH*>(state->host));
        tree.item_lowers = lowers;
        tree.item_uppers = uppers;
        tree.item_groups = groups;
        state->descriptor = tree;
        state->gpu = state->copy(&tree, 1);
        const auto handle = reinterpret_cast<uint64_t>(state->gpu);
        std::lock_guard<std::mutex> lock(bvh_mutex);
        bvhs.emplace(handle, std::move(state));
        id = handle;
    });
    return id;
}

WP_API int wp_sycl_bvh_destroy(uint64_t id)
{
    return wp::sycl_checked([&] {
        std::lock_guard<std::mutex> lock(bvh_mutex);
        auto& state = *bvhs.at(id);
        if (wp_sycl_synchronize(state.device))
            throw std::runtime_error(wp_sycl_get_error_string());
        bvhs.erase(id);
    });
}

WP_API int wp_sycl_bvh_refit(uint64_t id)
{
    return wp::sycl_checked([&] {
        std::lock_guard<std::mutex> lock(bvh_mutex);
        auto& state = *bvhs.at(id);
        wp::bvh_refit_sycl(state.device, state.descriptor);
    });
}

WP_API int wp_sycl_bvh_rebuild(uint64_t id, int constructor)
{
    return wp::sycl_checked([&] {
        wp::bvh_check_constructor_sycl(constructor);
        std::lock_guard<std::mutex> lock(bvh_mutex);
        auto& state = *bvhs.at(id);
        if (wp_sycl_synchronize(state.device))
            throw std::runtime_error(wp_sycl_get_error_string());
        wp::sycl_read_array(state.device, state.lowers, state.descriptor.item_lowers, state.descriptor.num_items);
        wp::sycl_read_array(state.device, state.uppers, state.descriptor.item_uppers, state.descriptor.num_items);
        if (state.descriptor.item_groups)
            wp::sycl_read_array(state.device, state.groups, state.descriptor.item_groups, state.descriptor.num_items);
        wp_bvh_rebuild_host(state.host, constructor);
        wp::bvh_update_sycl(state.descriptor, *reinterpret_cast<wp::BVH*>(state.host));
        std::memcpy(state.gpu, &state.descriptor, sizeof(wp::BVH));
    });
}

}  // extern "C"

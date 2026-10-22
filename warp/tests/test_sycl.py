# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
import unittest
import weakref

import numpy as np

import warp as wp
from warp.tests.unittest_utils import add_function_test


@wp.kernel
def sycl_square(x: wp.array[float], y: wp.array[float]):
    i = wp.tid()
    y[i] = x[i] * x[i]


@wp.kernel
def sycl_sum(x: wp.array[float], result: wp.array[float]):
    wp.atomic_add(result, 0, x[wp.tid()])


@wp.kernel(module="unique")
def sycl_atomic_types(
    small: wp.array[wp.float16],
    half: wp.array[wp.float16],
    signed_values: wp.array[wp.int32],
    wide: wp.array[wp.int64],
):
    wp.atomic_add(small, wp.tid() % 3, wp.float16(1.0))
    wp.atomic_add(half, 0, wp.float16(1.0))
    if wp.tid() == 0:
        wp.atomic_add(signed_values, 0, wp.int32(1))
        wp.atomic_add(wide, 0, wp.int64(1))


@wp.struct
class SyclArrayStruct:
    values: wp.array[float]


@wp.kernel(module="unique")
def sycl_struct_value(value: SyclArrayStruct, output: wp.array[float]):
    output[wp.tid()] = value.values[wp.tid()]


@wp.func
def sycl_tile_total(x: wp.array[float]):
    tile = wp.tile_load(x, shape=64)
    return wp.tile_extract(wp.tile_sum(tile), 0)


@wp.kernel(module="unique")
def sycl_nested_tile(x: wp.array[float], y: wp.array[float]):
    value = sycl_tile_total(x)
    y[wp.tid()] = value


@wp.kernel(module="unique")
def sycl_tile_loss(x: wp.array[float], loss: wp.array[float]):
    tile = wp.tile_load(x, shape=64)
    wp.tile_store(loss, wp.tile_sum(wp.tile_map(wp.mul, tile, tile)))


@wp.kernel
def sycl_bvh_query(tree: wp.uint64, hits: wp.array[int]):
    query = wp.bvh_query_aabb(tree, wp.vec3(-0.1), wp.vec3(1.1))
    index = int(0)
    while wp.bvh_query_next(query, index):
        hits[index] = 1


@wp.kernel
def sycl_mesh_query(mesh: wp.uint64, distance: wp.array[float]):
    query = wp.mesh_query_ray(mesh, wp.vec3(0.2, 0.2, 5.0), wp.vec3(0.0, 0.0, -1.0), 10.0)
    distance[0] = query.t if query.result else -1.0


@wp.kernel(module="unique")
def sycl_tiles(x: wp.array[float], y: wp.array[float], total: wp.array[float]):
    tile = wp.tile_load(x, shape=64)
    squared = wp.tile_map(wp.mul, tile, tile)
    wp.tile_store(y, wp.tile_scan_inclusive(squared))
    wp.tile_store(total, wp.tile_sum(squared))


def test_sycl_tiles(test, device):
    values = np.arange(64, dtype=np.float32) / 64.0
    for block_dim in (16, 32, 64):
        x = wp.array(values, device=device)
        y = wp.zeros_like(x)
        total = wp.zeros(1, device=device)
        wp.launch_tiled(sycl_tiles, dim=1, inputs=[x, y, total], block_dim=block_dim, device=device)
        np.testing.assert_allclose(y.numpy(), np.cumsum(values * values), rtol=1e-5)
        np.testing.assert_allclose(total.numpy(), [np.sum(values * values)], rtol=1e-5)


def test_sycl_core(test, device):
    test.assertFalse(device.is_cpu)
    test.assertFalse(device.is_cuda)
    test.assertTrue(device.is_sycl)
    values = np.arange(257, dtype=np.float32) / 257.0
    x = wp.array(values, device=device, requires_grad=True)
    y = wp.zeros_like(x)
    with wp.Tape() as tape:
        wp.launch(sycl_square, len(values), inputs=[x], outputs=[y], device=device)
    np.testing.assert_allclose(y.numpy(), values * values)
    tape.backward(grads={y: wp.ones_like(y)})
    np.testing.assert_allclose(x.grad.numpy(), 2.0 * values)


def test_sycl_atomics(test, device):
    x = wp.ones(4097, device=device)
    result = wp.zeros(1, device=device)
    wp.launch(sycl_sum, x.size, inputs=[x], outputs=[result], device=device)
    np.testing.assert_array_equal(result.numpy(), [4097.0])


def test_sycl_atomic_types(test, device):
    small = wp.zeros(3, dtype=wp.float16, device=device)
    half = wp.zeros(1, dtype=wp.float16, device=device)
    signed = wp.array([2147483647], dtype=wp.int32, device=device)
    wide = wp.array([9223372036854775807], dtype=wp.int64, device=device)
    wp.launch(sycl_atomic_types, 300, inputs=[small, half, signed, wide], device=device)
    np.testing.assert_array_equal(small.numpy(), [100, 100, 100])
    np.testing.assert_array_equal(half.numpy(), [300.0])
    np.testing.assert_array_equal(signed.numpy(), [-2147483648])
    np.testing.assert_array_equal(wide.numpy(), [-9223372036854775808])


def test_sycl_tile_grad(test, device):
    values = np.arange(64, dtype=np.float32) / 64.0
    for block_dim in (16, 32, 64):
        x = wp.array(values, device=device, requires_grad=True)
        loss = wp.zeros(1, device=device, requires_grad=True)
        with wp.Tape() as tape:
            wp.launch_tiled(sycl_tile_loss, dim=1, inputs=[x, loss], block_dim=block_dim, device=device)
        tape.backward(loss)
        np.testing.assert_allclose(loss.numpy(), [np.sum(values * values)], rtol=1e-5)
        np.testing.assert_allclose(x.grad.numpy(), 2.0 * values, rtol=1e-5)

    x = wp.ones(64, device=device)
    y = wp.zeros(32, device=device)
    wp.launch(sycl_nested_tile, 32, inputs=[x, y], block_dim=32, device=device)
    np.testing.assert_array_equal(y.numpy(), np.full(32, 64.0))
    with test.assertRaisesRegex(RuntimeError, "complete work-groups"):
        wp.launch(sycl_nested_tile, 31, inputs=[x, y], block_dim=32, device=device)


def test_sycl_graph(test, device):
    x = wp.ones(257, device=device)
    y = wp.zeros_like(x)
    wp.launch(sycl_square, x.size, inputs=[x, y], device=device)
    with wp.ScopedCapture(device=device) as capture:
        wp.launch(sycl_square, x.size, inputs=[x, y], device=device)
    x.fill_(3.0)
    wp.capture_launch(capture.graph)
    np.testing.assert_array_equal(y.numpy(), np.full(257, 9.0))
    x.fill_(4.0)
    wp.capture_launch(capture.graph)
    np.testing.assert_array_equal(y.numpy(), np.full(257, 16.0))


def test_sycl_graph_lifetimes(test, device):
    x = wp.ones(32, device=device)
    y = wp.zeros_like(x)
    retained = weakref.ref(x)
    with wp.ScopedCapture(device=device) as capture:
        x.zero_()
        x.fill_(3.0)
        wp.copy(y, x)
    del x
    gc.collect()
    test.assertIsNotNone(retained())
    wp.capture_launch(capture.graph)
    np.testing.assert_array_equal(y.numpy(), np.full(32, 3.0))
    with test.assertRaisesRegex(NotImplementedError, "Allocations"):
        with wp.ScopedCapture(device=device):
            wp.empty(1, device=device)
    test.assertFalse(device.is_capturing)
    with wp.ScopedCapture(device=device) as second:
        y.fill_(4.0)
    wp.capture_launch(second.graph)
    np.testing.assert_array_equal(y.numpy(), np.full(32, 4.0))


def test_sycl_strides(test, device):
    values = np.arange(24, dtype=np.float32).reshape(4, 6)
    x = wp.array(values, device=device)
    view = x[:, ::2]
    np.testing.assert_array_equal(view.numpy(), values[:, ::2])
    view.fill_(7.0)
    values[:, ::2] = 7.0
    np.testing.assert_array_equal(x.numpy(), values)


def test_sycl_geometry(test, device):
    values = np.arange(24, dtype=np.float32).reshape(8, 3)
    values[0] = 0.0
    lowers = wp.array(values, dtype=wp.vec3, device=device)
    uppers = wp.array(values + 1.0, dtype=wp.vec3, device=device)
    tree = wp.Bvh(lowers, uppers, leaf_size=1)
    hits = wp.zeros(8, dtype=int, device=device)
    wp.launch(sycl_bvh_query, 1, inputs=[tree.id, hits], device=device)
    np.testing.assert_array_equal(hits.numpy(), [1, 0, 0, 0, 0, 0, 0, 0])
    lowers.fill_(wp.vec3(5.0))
    uppers.fill_(wp.vec3(6.0))
    tree.refit()
    hits.zero_()
    wp.launch(sycl_bvh_query, 1, inputs=[tree.id, hits], device=device)
    np.testing.assert_array_equal(hits.numpy(), np.zeros(8))
    lowers.fill_(wp.vec3(0.0))
    uppers.fill_(wp.vec3(1.0))
    tree.rebuild("median")
    wp.launch(sycl_bvh_query, 1, inputs=[tree.id, hits], device=device)
    np.testing.assert_array_equal(hits.numpy(), np.ones(8))
    with test.assertRaisesRegex(RuntimeError, "SAH and median"):
        wp.Bvh(lowers, uppers, constructor="lbvh")

    with wp.ScopedCapture(device=device) as capture:
        lowers.fill_(wp.vec3(5.0))
        uppers.fill_(wp.vec3(6.0))
        tree.refit()
        hits.zero_()
        wp.launch(sycl_bvh_query, 1, inputs=[tree.id, hits], device=device)
    wp.capture_launch(capture.graph)
    np.testing.assert_array_equal(hits.numpy(), np.zeros(8))

    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    indices = wp.array([0, 1, 2], dtype=int, device=device)
    distance = wp.zeros(1, device=device)
    for winding in (False, True):
        points = wp.array(vertices, dtype=wp.vec3, device=device)
        mesh = wp.Mesh(points, indices, support_winding_number=winding)
        wp.launch(sycl_mesh_query, 1, inputs=[mesh.id, distance], device=device)
        np.testing.assert_allclose(distance.numpy(), [5.0])
        points.assign(vertices + np.array([0, 0, 1], dtype=np.float32))
        mesh.refit()
        wp.launch(sycl_mesh_query, 1, inputs=[mesh.id, distance], device=device)
        np.testing.assert_allclose(distance.numpy(), [4.0])
        mesh.points = wp.array(vertices + np.array([0, 0, 2], dtype=np.float32), dtype=wp.vec3, device=device)
        mesh.velocities = wp.zeros_like(mesh.points)
        wp.launch(sycl_mesh_query, 1, inputs=[mesh.id, distance], device=device)
        np.testing.assert_allclose(distance.numpy(), [3.0])

    empty_points = wp.empty(0, dtype=wp.vec3, device=device)
    empty_tree = wp.Bvh(empty_points, empty_points)
    empty_tree.refit()
    empty_tree.rebuild()
    hits.zero_()
    wp.launch(sycl_bvh_query, 1, inputs=[empty_tree.id, hits], device=device)
    np.testing.assert_array_equal(hits.numpy(), np.zeros(8))
    empty_mesh = wp.Mesh(empty_points, wp.empty(0, dtype=int, device=device))
    empty_mesh.refit()
    wp.launch(sycl_mesh_query, 1, inputs=[empty_mesh.id, distance], device=device)
    np.testing.assert_array_equal(distance.numpy(), [-1.0])


def test_sycl_unsupported(test, device):
    x = wp.ones(1, device=device)
    with test.assertRaisesRegex(NotImplementedError, "DLPack"):
        wp.to_dlpack(x)
    with test.assertRaisesRegex(NotImplementedError, "indexed arrays"):
        wp.indexedarray(x)
    with test.assertRaisesRegex(NotImplementedError, "hash grids"):
        wp.HashGrid(4, 4, 4, device=device)
    with test.assertRaisesRegex(NotImplementedError, "array_scan"):
        wp.utils.array_scan(x, x)
    with test.assertRaisesRegex(NotImplementedError, "conditional graphs"):
        wp.capture_if(wp.ones(1, dtype=int, device=device), lambda: None)
    with test.assertRaisesRegex(NotImplementedError, "Raw array"):
        wp.launch(sycl_square, 1, inputs=[x.__ctype__(), x], device=device)
    value = SyclArrayStruct()
    value.values = wp.ones(1, device="cpu")
    with test.assertRaisesRegex(RuntimeError, "launch device"):
        wp.launch(sycl_struct_value, 1, inputs=[value, x], device=device)


class TestSycl(unittest.TestCase):
    pass


devices = [device for device in wp.get_devices() if device.is_sycl]
add_function_test(TestSycl, "test_core", test_sycl_core, devices=devices)
add_function_test(TestSycl, "test_atomics", test_sycl_atomics, devices=devices)
add_function_test(TestSycl, "test_tiles", test_sycl_tiles, devices=devices)
add_function_test(TestSycl, "test_graph", test_sycl_graph, devices=devices)
add_function_test(TestSycl, "test_strides", test_sycl_strides, devices=devices)
add_function_test(TestSycl, "test_geometry", test_sycl_geometry, devices=devices)
add_function_test(TestSycl, "test_atomic_types", test_sycl_atomic_types, devices=devices)
add_function_test(TestSycl, "test_tile_grad", test_sycl_tile_grad, devices=devices)
add_function_test(TestSycl, "test_graph_lifetimes", test_sycl_graph_lifetimes, devices=devices)
add_function_test(TestSycl, "test_unsupported", test_sycl_unsupported, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)

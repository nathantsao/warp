# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ctypes
import faulthandler
import gc
import hashlib
import subprocess
import tempfile
import unittest
import weakref
from pathlib import Path
from unittest import mock

import numpy as np

import warp as wp
from warp._src import build
from warp._src.context import _is_graph_capture_allocation_enabled, _is_graph_capture_allocation_supported
from warp._src.sycl import check
from warp._src.types import tile
from warp.tests import test_graph as graph_tests
from warp.tests import test_reload as reload_tests
from warp.tests import test_utils as utils_tests
from warp.tests.tile import test_tile_sort as tile_sort_tests
from warp.tests.unittest_utils import add_function_test, run_test_in_subprocess


@wp.kernel
def sycl_square(x: wp.array[float], y: wp.array[float]):
    i = wp.tid()
    y[i] = x[i] * x[i]


@wp.kernel
def sycl_sum(x: wp.array[float], result: wp.array[float]):
    wp.atomic_add(result, 0, x[wp.tid()])


@wp.kernel(module="unique", enable_backward=False)
def sycl_texture_sample(
    texture: wp.Texture2D,
    coordinates: wp.array[wp.vec2],
    scalar: wp.array[float],
    pair: wp.array[wp.vec2],
    rgba: wp.array[wp.vec4],
):
    i = wp.tid()
    uv = coordinates[i]
    scalar[i] = wp.texture_sample(texture, uv, dtype=float)
    pair[i] = wp.texture_sample(texture, uv[0], uv[1], dtype=wp.vec2)
    rgba[i] = wp.texture_sample(texture, uv, dtype=wp.vec4, lod=2.5)


@wp.kernel(module="unique", enable_backward=False)
def sycl_texture_array_sample(
    textures: wp.array[wp.Texture2D],
    selectors: wp.array[int],
    output: wp.array[wp.vec4],
):
    i = wp.tid()
    output[i] = wp.texture_sample(textures[selectors[i]], wp.vec2(0.5), dtype=wp.vec4)


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


@wp.struct
class SyclTextureStruct:
    texture: wp.Texture2D


@wp.struct
class SyclNestedTextureStruct:
    value: SyclTextureStruct


@wp.kernel(module="unique", enable_backward=False)
def sycl_texture_struct_sample(
    value: SyclNestedTextureStruct,
    entries: wp.array[SyclTextureStruct],
    output: wp.array[wp.vec4],
):
    output[0] = wp.texture_sample(value.value.texture, wp.vec2(0.5), dtype=wp.vec4)
    output[1] = wp.texture_sample(entries[0].texture, wp.vec2(0.5), dtype=wp.vec4)


SyclArgumentPadding = wp.types.vector(length=512, dtype=wp.float32)


@wp.struct
class SyclTileArguments:
    values: wp.array[float]
    padding: SyclArgumentPadding


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


@wp.func
def sycl_sort_tile_row(keys: wp.array2d[int], values: wp.array2d[int], row: int):
    key_tile = wp.tile_load(keys[row], shape=65)
    value_tile = wp.tile_load(values[row], shape=65)
    for _ in range(2):
        wp.tile_sort(key_tile, value_tile)
    wp.tile_store(keys[row], key_tile)
    wp.tile_store(values[row], value_tile)


@wp.kernel(enable_backward=False, module="unique")
def sycl_nested_tile_sort(keys: wp.array2d[int], values: wp.array2d[int]):
    sycl_sort_tile_row(keys, values, wp.tid())
    sycl_sort_tile_row(keys, values, wp.tid())


@wp.kernel(module="unique")
def sycl_tile_loss(x: wp.array[float], loss: wp.array[float]):
    tile = wp.tile_load(x, shape=64)
    wp.tile_store(loss, wp.tile_sum(wp.tile_map(wp.mul, tile, tile)))


@wp.kernel(module="unique", enable_backward=False)
def sycl_partial_tile(
    x: wp.array[float],
    totals: wp.array[float],
    scans: wp.array[wp.vec2],
    extrema: wp.array[wp.vec2],
    indices: wp.array[wp.vec2i],
    axis_totals: wp.array[wp.vec2],
    block_dims: wp.array[int],
):
    i = wp.tid()
    value = float(x[i])
    values = wp.tile(value)
    pairs = wp.tile(wp.vec2(value, 2.0 * value), preserve_type=True)
    for _ in range(2):
        total = wp.tile_sum(values)[0]
        minimum = wp.tile_min(values)[0]
        maximum = wp.tile_max(values)[0]
        imin = wp.tile_argmin(values)[0]
        imax = wp.tile_argmax(values)[0]
        inclusive = wp.untile(wp.tile_scan_inclusive(values))
        exclusive = wp.untile(wp.tile_scan_exclusive(values))
        reduced_pairs = wp.tile_sum(pairs, axis=0)
        pair_total = reduced_pairs[0]
    totals[i] = total
    scans[i] = wp.vec2(inclusive, exclusive)
    extrema[i] = wp.vec2(minimum, maximum)
    indices[i] = wp.vec2i(imin, imax)
    axis_totals[i] = pair_total
    block_dims[i] = wp.block_dim()


@wp.kernel(module="unique", enable_backward=False)
def sycl_partial_tile_staged(args: SyclTileArguments, totals: wp.array[float]):
    i = wp.tid()
    value = float(args.values[i])
    total = wp.tile_sum(wp.tile(value))[0]
    totals[i] = total


@wp.kernel(module="unique")
def sycl_partial_tile_loss(x: wp.array[float], totals: wp.array[float]):
    i = wp.tid()
    total = wp.tile_sum(wp.tile(x[i] * x[i]))[0]
    if i % wp.block_dim() == 0:
        totals[i // wp.block_dim()] = total


@wp.kernel(module="unique")
def sycl_printf(x: wp.array[float], y: wp.array[float]):
    i = wp.tid()
    wp.printf("sycl_printf\n")
    wp.printf("sycl_printf %d %f\n", i, x[i])
    y[i] = x[i]


@wp.kernel(module="unique")
def sycl_tile_linalg(
    a: wp.array3d[float],
    b: wp.array3d[float],
    c: wp.array3d[float],
    rhs: wp.array3d[float],
    product: wp.array3d[float],
    accumulated: wp.array3d[float],
    solution: wp.array3d[float],
    factor: wp.array3d[float],
):
    i = wp.tid()
    ta = wp.tile_load(a[i], shape=(16, 16), storage="shared")
    tb = wp.tile_load(b[i], shape=(16, 16), storage="shared")
    tc = wp.tile_load(c[i], shape=(16, 16), storage="shared")
    wp.tile_store(product[i], wp.tile_matmul(ta, tb))
    wp.tile_matmul(wp.tile_transpose(ta), tb, tc, alpha=-1.0)
    wp.tile_store(accumulated[i], tc)

    spd = wp.tile_load(c[i], shape=(16, 16), storage="shared")
    spd += wp.tile_matmul(wp.tile_transpose(ta), ta)
    wp.tile_store(factor[i], spd)
    wp.tile_cholesky_inplace(spd, fill_mode="upper")
    r = wp.tile_load(rhs[i], shape=(16, 1), storage="shared")
    wp.tile_store(solution[i], wp.tile_cholesky_solve(spd, r, fill_mode="upper"))


@wp.kernel(module="unique")
def sycl_many_arguments(
    a0: wp.array[float],
    a1: wp.array[float],
    a2: wp.array[float],
    a3: wp.array[float],
    a4: wp.array[float],
    a5: wp.array[float],
    a6: wp.array[float],
    a7: wp.array[float],
    a8: wp.array[float],
    a9: wp.array[float],
    a10: wp.array[float],
    a11: wp.array[float],
    a12: wp.array[float],
    a13: wp.array[float],
    a14: wp.array[float],
    a15: wp.array[float],
    a16: wp.array[float],
    a17: wp.array[float],
    a18: wp.array[float],
    a19: wp.array[float],
    a20: wp.array[float],
    a21: wp.array[float],
    a22: wp.array[float],
    a23: wp.array[float],
    a24: wp.array[float],
    a25: wp.array[float],
    a26: wp.array[float],
    a27: wp.array[float],
    a28: wp.array[float],
    a29: wp.array[float],
    a30: wp.array[float],
    a31: wp.array[float],
    a32: wp.array[float],
    a33: wp.array[float],
    a34: wp.array[float],
    a35: wp.array[float],
    a36: wp.array[float],
    a37: wp.array[float],
    a38: wp.array[float],
    a39: wp.array[float],
    out: wp.array[float],
):
    i = wp.tid()
    total = (
        a0[i]
        + a1[i]
        + a2[i]
        + a3[i]
        + a4[i]
        + a5[i]
        + a6[i]
        + a7[i]
        + a8[i]
        + a9[i]
        + a10[i]
        + a11[i]
        + a12[i]
        + a13[i]
        + a14[i]
        + a15[i]
        + a16[i]
        + a17[i]
        + a18[i]
        + a19[i]
        + a20[i]
        + a21[i]
        + a22[i]
        + a23[i]
        + a24[i]
        + a25[i]
        + a26[i]
        + a27[i]
        + a28[i]
        + a29[i]
        + a30[i]
        + a31[i]
        + a32[i]
        + a33[i]
        + a34[i]
        + a35[i]
        + a36[i]
        + a37[i]
        + a38[i]
        + a39[i]
    )
    out[i] = total


SYCL_BLOCK = 16
SYCL_MATRIX = 48


@wp.kernel(module="unique", enable_backward=False)
def sycl_blocked_cholesky(a: wp.array3d[float], b: wp.array3d[float], u: wp.array3d[float], x: wp.array3d[float]):
    # right-looking blocked Cholesky and solve over fully unrolled constant-size tiles
    w = wp.tid()
    rhs = wp.tile_load(b[w], shape=(SYCL_MATRIX, 1), storage="shared", bounds_check=False)
    for k in range(0, SYCL_MATRIX, SYCL_BLOCK):
        rhs_k = wp.tile_view(rhs, shape=(SYCL_BLOCK, 1), offset=(k, 0))
        a_kk = wp.tile_load(
            a[w], shape=(SYCL_BLOCK, SYCL_BLOCK), offset=(k, k), storage="shared", bounds_check=False, aligned=True
        )
        for j in range(0, k, SYCL_BLOCK):
            u_jk = wp.tile_load(
                u[w], shape=(SYCL_BLOCK, SYCL_BLOCK), offset=(j, k), storage="shared", bounds_check=False, aligned=True
            )
            wp.tile_matmul(wp.tile_transpose(u_jk), u_jk, a_kk, alpha=-1.0)
            y_j = wp.tile_view(rhs, shape=(SYCL_BLOCK, 1), offset=(j, 0))
            wp.tile_matmul(wp.tile_transpose(u_jk), y_j, rhs_k, alpha=-1.0)
        wp.tile_cholesky_inplace(a_kk, fill_mode="upper")
        wp.tile_store(u[w], a_kk, offset=(k, k), bounds_check=False, aligned=True)
        wp.tile_lower_solve_inplace(wp.tile_transpose(a_kk), rhs_k)
        for i in range(k + SYCL_BLOCK, SYCL_MATRIX, SYCL_BLOCK):
            a_ki = wp.tile_load(
                a[w], shape=(SYCL_BLOCK, SYCL_BLOCK), offset=(k, i), storage="shared", bounds_check=False, aligned=True
            )
            for j in range(0, k, SYCL_BLOCK):
                u_jk = wp.tile_load(
                    u[w],
                    shape=(SYCL_BLOCK, SYCL_BLOCK),
                    offset=(j, k),
                    storage="shared",
                    bounds_check=False,
                    aligned=True,
                )
                u_ji = wp.tile_load(
                    u[w],
                    shape=(SYCL_BLOCK, SYCL_BLOCK),
                    offset=(j, i),
                    storage="shared",
                    bounds_check=False,
                    aligned=True,
                )
                wp.tile_matmul(wp.tile_transpose(u_jk), u_ji, a_ki, alpha=-1.0)
            wp.tile_lower_solve_inplace(wp.tile_transpose(a_kk), a_ki)
            wp.tile_store(u[w], a_ki, offset=(k, i), bounds_check=False, aligned=True)
    for i in range(SYCL_MATRIX - SYCL_BLOCK, -1, -SYCL_BLOCK):
        tmp = wp.tile_view(rhs, shape=(SYCL_BLOCK, 1), offset=(i, 0))
        for j in range(i + SYCL_BLOCK, SYCL_MATRIX, SYCL_BLOCK):
            u_ij = wp.tile_load(
                u[w], shape=(SYCL_BLOCK, SYCL_BLOCK), offset=(i, j), storage="shared", bounds_check=False, aligned=True
            )
            x_j = wp.tile_view(rhs, shape=(SYCL_BLOCK, 1), offset=(j, 0))
            wp.tile_matmul(u_ij, x_j, tmp, alpha=-1.0)
        u_ii = wp.tile_load(
            u[w], shape=(SYCL_BLOCK, SYCL_BLOCK), offset=(i, i), storage="shared", bounds_check=False, aligned=True
        )
        wp.tile_upper_solve_inplace(u_ii, tmp)
    wp.tile_store(x[w], rhs, offset=(0, 0), bounds_check=False)


@wp.kernel(module="unique", enable_backward=False)
def sycl_math(x: wp.array[float], y: wp.array[float], out: wp.array2d[float]):
    i = wp.tid()
    out[0, i] = wp.sin(x[i])
    out[1, i] = wp.cos(x[i])
    out[2, i] = wp.tan(x[i] * 0.4)
    out[3, i] = wp.sqrt(y[i])
    out[4, i] = x[i] / y[i]


@wp.kernel(module="unique")
def sycl_increment(values: wp.array[int]):
    values[wp.tid()] += 1


@wp.kernel(module="unique")
def sycl_scale(values: wp.array[int], factor: int):
    values[wp.tid()] *= factor


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


def test_sycl_partial_tiles(test, device):
    for block_dim in (32, 64, 256):
        for length in (1, 17, 31, 32, 33, 65, 257, 513):
            for sign in (-1, 1):
                with test.subTest(block_dim=block_dim, length=length, sign=sign):
                    values = sign * np.arange(1, length + 1, dtype=np.float32)
                    totals = wp.zeros(length + 3, device=device)
                    scans = wp.zeros(length + 3, dtype=wp.vec2, device=device)
                    extrema = wp.zeros_like(scans)
                    indices = wp.zeros(length + 3, dtype=wp.vec2i, device=device)
                    axis_totals = wp.zeros_like(scans)
                    block_dims = wp.zeros(length + 3, dtype=int, device=device)
                    wp.launch(
                        sycl_partial_tile,
                        dim=length,
                        inputs=[wp.array(values, device=device)],
                        outputs=[totals, scans, extrema, indices, axis_totals, block_dims],
                        block_dim=block_dim,
                        device=device,
                    )
                    expected_totals = np.zeros(length + 3, dtype=np.float32)
                    expected_scans = np.zeros((length + 3, 2), dtype=np.float32)
                    expected_extrema = np.zeros_like(expected_scans)
                    expected_indices = np.zeros((length + 3, 2), dtype=np.int32)
                    for start in range(0, length, block_dim):
                        end = min(start + block_dim, length)
                        chunk = values[start:end]
                        expected_totals[start:end] = np.sum(chunk)
                        expected_scans[start:end, 0] = np.cumsum(chunk)
                        expected_scans[start:end, 1] = np.cumsum(chunk) - chunk
                        expected_extrema[start:end] = (np.min(chunk), np.max(chunk))
                        expected_indices[start:end] = (np.argmin(chunk), np.argmax(chunk))
                    np.testing.assert_array_equal(totals.numpy(), expected_totals)
                    np.testing.assert_array_equal(scans.numpy(), expected_scans)
                    np.testing.assert_array_equal(extrema.numpy(), expected_extrema)
                    np.testing.assert_array_equal(indices.numpy(), expected_indices)
                    np.testing.assert_array_equal(
                        axis_totals.numpy(), np.stack((expected_totals, 2 * expected_totals), axis=1)
                    )
                    np.testing.assert_array_equal(
                        block_dims.numpy(), np.concatenate((np.full(length, block_dim), np.zeros(3)))
                    )


def test_sycl_partial_tile_graph(test, device):
    block_dim = 64
    length = 2 * block_dim + 17
    values = wp.zeros(length, device=device)
    args = SyclTileArguments()
    args.values = values
    totals = wp.zeros_like(values)
    wp.launch(sycl_partial_tile_staged, length, inputs=[args, totals], block_dim=block_dim, device=device)
    with wp.ScopedCapture(device=device) as capture:
        wp.launch(sycl_partial_tile_staged, length, inputs=[args, totals], block_dim=block_dim, device=device)
    for replay in range(3):
        data = np.arange(length, dtype=np.float32) + replay
        values.assign(data)
        totals.fill_(wp.inf)
        wp.capture_launch(capture.graph)
        expected = np.empty_like(data)
        for start in range(0, length, block_dim):
            end = min(start + block_dim, length)
            expected[start:end] = np.sum(data[start:end])
        np.testing.assert_array_equal(totals.numpy(), expected)


def test_sycl_partial_tile_grad(test, device):
    block_dim = 64
    length = 2 * block_dim + 17
    values = np.arange(length, dtype=np.float32) / length
    x = wp.array(values, device=device, requires_grad=True)
    totals = wp.zeros(3, device=device, requires_grad=True)
    with wp.Tape() as tape:
        wp.launch(sycl_partial_tile_loss, length, inputs=[x, totals], block_dim=block_dim, device=device)
    tape.backward(grads={totals: wp.ones_like(totals)})
    expected = [np.sum(values[start : start + block_dim] ** 2) for start in range(0, length, block_dim)]
    np.testing.assert_allclose(totals.numpy(), expected, rtol=1e-5)
    np.testing.assert_allclose(x.grad.numpy(), 2 * values, rtol=1e-5)


def test_sycl_partial_tile_sort(test, device):
    for length in (1, 17, 31, 32, 33, 65):
        with test.subTest(length=length):
            kernel = tile_sort_tests.create_payload_sort_kernel(int, length)
            data = np.random.default_rng(length).permutation(length).astype(np.int32)
            keys = wp.array(data, device=device)
            values = wp.array(np.arange(length, dtype=np.int32), device=device)
            output_keys, output_values = wp.empty_like(keys), wp.empty_like(values)
            with test.assertRaisesRegex(RuntimeError, "tile_sort launches require complete work-groups"):
                wp.launch(
                    kernel,
                    dim=length,
                    inputs=[keys, values, output_keys, output_values],
                    block_dim=128,
                    device=device,
                )
            wp.launch(
                kernel,
                dim=128,
                inputs=[keys, values, output_keys, output_values],
                block_dim=128,
                device=device,
            )
            np.testing.assert_array_equal(output_keys.numpy(), np.sort(data))
            np.testing.assert_array_equal(data[output_values.numpy()], output_keys.numpy())


def test_sycl_debug_tiles(test, device):
    @wp.kernel(module="unique")
    def debug_total(x: wp.array[float], y: wp.array[float]):
        y[wp.tid()] = sycl_tile_total(x)

    wp.set_module_options({"mode": "debug"}, debug_total.module)
    x = wp.ones(64, device=device)
    y = wp.zeros(31, device=device)
    wp.launch(debug_total, dim=31, inputs=[x, y], block_dim=32, device=device)
    np.testing.assert_array_equal(y.numpy(), np.full(31, 62.0))

    for length in (31, 32, 65):
        kernel = tile_sort_tests.create_payload_sort_kernel(int, length)
        original_mode = kernel.module.options["mode"]
        data = np.random.default_rng(length).permutation(length).astype(np.int32)
        keys = wp.array(data, device=device)
        values = wp.array(np.arange(length, dtype=np.int32), device=device)
        output_keys, output_values = wp.empty_like(keys), wp.empty_like(values)
        try:
            for mode in ("release", "debug", "release"):
                wp.set_module_options({"mode": mode}, kernel.module)
                wp.launch(
                    kernel,
                    dim=128,
                    inputs=[keys, values, output_keys, output_values],
                    block_dim=128,
                    device=device,
                )
                np.testing.assert_array_equal(output_keys.numpy(), np.sort(data))
                np.testing.assert_array_equal(data[output_values.numpy()], output_keys.numpy())
        finally:
            wp.set_module_options({"mode": original_mode}, kernel.module)

    for kernel, test_function in (
        (sycl_tile_linalg, test_sycl_tile_linalg),
        (sycl_blocked_cholesky, test_sycl_blocked_cholesky),
    ):
        original_mode = kernel.module.options["mode"]
        try:
            wp.set_module_options({"mode": "debug"}, kernel.module)
            test_function(test, device)
        finally:
            wp.set_module_options({"mode": original_mode}, kernel.module)


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
    y.zero_()
    wp.launch(sycl_nested_tile, 31, inputs=[x, y], block_dim=32, device=device)
    np.testing.assert_array_equal(y.numpy(), np.concatenate((np.full(31, 62.0), [0.0])))


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
    with test.assertRaisesRegex(RuntimeError, "capture body failed"):
        with wp.ScopedCapture(device=device) as failed:
            wp.empty(1, device=device)
            raise RuntimeError("capture body failed")
    test.assertFalse(device.is_capturing)
    del failed
    with wp.ScopedCapture(device=device) as second:
        y.fill_(4.0)
    wp.capture_launch(second.graph)
    np.testing.assert_array_equal(y.numpy(), np.full(32, 4.0))


def _sycl_graph_memory_info(device):
    allocations = ctypes.c_size_t()
    size = ctypes.c_size_t()
    library = device.runtime.sycl
    check(
        library,
        library.wp_sycl_graph_memory_info(device.context, ctypes.byref(allocations), ctypes.byref(size)),
    )
    return allocations.value, size.value


def test_sycl_graph_allocation_capability(test, device):
    test.assertTrue(_is_graph_capture_allocation_supported(device))
    test.assertTrue(_is_graph_capture_allocation_enabled(device))
    test.assertFalse(device.is_mempool_supported)
    test.assertFalse(device.is_mempool_enabled)


def test_sycl_graph_allocation_constructors(test, device):
    n = 37
    output = wp.zeros(n, device=device)
    with wp.ScopedCapture(device, force_module_load=False) as capture:
        empty = wp.empty(0, device=device)
        values = wp.full(n, 3.0, device=device)
        clone = wp.clone(values)
        base = wp.zeros(2 * n, device=device)
        view = base[::2]
        wp.copy(view, clone)
        wp.launch(sycl_square, n, inputs=[view, output], device=device)
    test.assertIsNone(empty.ptr)
    for _ in range(2):
        values.fill_(11.0)
        base.fill_(12.0)
        wp.capture_launch(capture.graph)
        np.testing.assert_array_equal(output.numpy(), np.full(n, 9.0))
        np.testing.assert_array_equal(base.numpy()[1::2], np.zeros(n))
    del capture
    gc.collect()
    np.testing.assert_array_equal(clone.numpy(), np.full(n, 3.0))
    view.fill_(4.0)
    np.testing.assert_array_equal(view.numpy(), np.full(n, 4.0))


def test_sycl_graph_allocation_gradients(test, device):
    n = 37
    with wp.ScopedCapture(device, force_module_load=False) as capture:
        with wp.Tape() as tape:
            values = wp.full(n, 3.0, device=device, requires_grad=True)
            output = wp.empty(n, device=device, requires_grad=True)
            wp.launch(sycl_square, n, inputs=[values], outputs=[output], device=device)
        tape.backward(grads={output: wp.ones_like(output)})
    for _ in range(2):
        values.grad.fill_(99.0)
        wp.capture_launch(capture.graph)
        np.testing.assert_array_equal(output.numpy(), np.full(n, 9.0))
        np.testing.assert_array_equal(values.grad.numpy(), np.full(n, 6.0))


def test_sycl_graph_allocation_lifetimes(test, device):
    gc.collect()
    baseline = _sycl_graph_memory_info(device)
    n = 37
    retained = (baseline[0] + 1, baseline[1] + n * 4)
    output = wp.zeros(1, device=device)
    for launch in (False, True):
        for release_user_first in (False, True):
            with test.subTest(launch=launch, release_user_first=release_user_first):
                with wp.ScopedCapture(device, force_module_load=False) as capture:
                    unused = wp.empty(n, device=device)
                    output.fill_(7.0)
                graph = capture.graph
                del capture
                test.assertEqual(_sycl_graph_memory_info(device), retained)
                if release_user_first:
                    del unused
                    gc.collect()
                    test.assertEqual(_sycl_graph_memory_info(device), retained)
                if launch:
                    for _ in range(3):
                        wp.capture_launch(graph)
                    np.testing.assert_array_equal(output.numpy(), [7.0])
                    test.assertEqual(_sycl_graph_memory_info(device), retained)
                del graph
                gc.collect()
                if not release_user_first:
                    test.assertEqual(_sycl_graph_memory_info(device), retained)
                    unused.fill_(3.0)
                    np.testing.assert_array_equal(unused.numpy(), np.full(n, 3.0))
                    del unused
                gc.collect()
                test.assertEqual(_sycl_graph_memory_info(device), baseline)


def test_sycl_graph_allocation_release_during_capture(test, device):
    gc.collect()
    baseline = _sycl_graph_memory_info(device)
    unrelated = wp.zeros(37, device=device)
    output = wp.zeros(1, device=device)
    with wp.ScopedCapture(device, force_module_load=False) as capture:
        unused = wp.empty(37, device=device)
        del unused, unrelated
        gc.collect()
        output.fill_(5.0)
        test.assertEqual(_sycl_graph_memory_info(device), (baseline[0] + 1, baseline[1] + 148))
    wp.capture_launch(capture.graph)
    np.testing.assert_array_equal(output.numpy(), [5.0])
    del capture
    gc.collect()
    test.assertEqual(_sycl_graph_memory_info(device), baseline)


def test_sycl_graph_allocation_failures(test, device):
    gc.collect()
    baseline = _sycl_graph_memory_info(device)
    output = wp.zeros(1, device=device)
    with test.assertRaisesRegex(RuntimeError, "bad_alloc"):
        with wp.ScopedCapture(device, force_module_load=False) as capture:
            unused = wp.empty(37, device=device)
            del unused
            output.fill_(1.0)
            device.default_allocator.allocate(ctypes.c_size_t(-1).value)
    test.assertFalse(device.is_capturing)
    del capture
    gc.collect()
    test.assertEqual(_sycl_graph_memory_info(device), baseline)

    wp.capture_begin(device=device, force_module_load=False)
    unused = wp.empty(37, device=device)
    del unused
    output.fill_(2.0)
    library = device.runtime.sycl
    with (
        mock.patch.object(library, "wp_sycl_graph_end", return_value=-1),
        mock.patch.object(library, "wp_sycl_get_error_string", return_value=b"injected finalization failure"),
        test.assertRaisesRegex(RuntimeError, "injected finalization failure"),
    ):
        wp.capture_end(device=device)
    test.assertFalse(device.is_capturing)
    gc.collect()
    test.assertEqual(_sycl_graph_memory_info(device), baseline)
    with wp.ScopedCapture(device, force_module_load=False) as capture:
        output.fill_(3.0)
    wp.capture_launch(capture.graph)
    np.testing.assert_array_equal(output.numpy(), [3.0])


def test_sycl_graph_allocation_deferred_destruction(test, device):
    gc.collect()
    baseline = _sycl_graph_memory_info(device)
    output = wp.zeros(1, dtype=int, device=device)
    sycl_increment.module.load(device)
    test.addCleanup(sycl_increment.module.load, device)
    with wp.ScopedCapture(device, force_module_load=False) as first:
        unused = wp.empty(37, device=device)
        del unused
        wp.launch(sycl_increment, dim=1, inputs=[output], device=device)
    wp.capture_launch(first.graph)
    sycl_increment.module.unload()
    with wp.ScopedCapture(device, force_module_load=False) as second:
        unused = wp.empty(37, device=device)
        del unused, first
        gc.collect()
        test.assertEqual(_sycl_graph_memory_info(device), (baseline[0] + 2, baseline[1] + 296))
        output.fill_(4)
    test.assertEqual(_sycl_graph_memory_info(device), (baseline[0] + 1, baseline[1] + 148))
    wp.capture_launch(second.graph)
    np.testing.assert_array_equal(output.numpy(), [4])
    del second
    gc.collect()
    test.assertEqual(_sycl_graph_memory_info(device), baseline)


def test_sycl_graph_allocation_stream_lifetimes(test, device):
    gc.collect()
    baseline = _sycl_graph_memory_info(device)
    output = wp.zeros(37, device=device)
    recording_stream = wp.Stream(device)
    with wp.ScopedStream(recording_stream):
        with wp.ScopedCapture(device, force_module_load=False) as capture:
            values = wp.full(37, 3.0, device=device)
            wp.launch(sycl_square, 37, inputs=[values, output], device=device)
    graph = capture.graph
    del capture, values, recording_stream
    gc.collect()
    default_stream = device.stream
    replay_stream = wp.Stream(device)
    other_stream = wp.Stream(device)
    try:
        device.set_stream(replay_stream, sync=False)
        wp.capture_launch(graph)
        device.set_stream(other_stream, sync=False)
        del graph
        gc.collect()
        test.assertEqual(_sycl_graph_memory_info(device), baseline)
        np.testing.assert_array_equal(output.numpy(), np.full(37, 9.0))
    finally:
        device.set_stream(default_stream, sync=False)


def test_sycl_graph_subgraph_lifetimes(test, device):
    output = wp.zeros(37, device=device)
    with wp.ScopedCapture(device, force_module_load=False) as child:
        values = wp.full(37, 3.0, device=device)
        wp.launch(sycl_square, 37, inputs=[values, output], device=device)
    retained = weakref.ref(child.graph)
    with wp.ScopedCapture(device, force_module_load=False) as parent:
        wp.capture_launch(child.graph)
    del child, values
    gc.collect()
    test.assertIsNotNone(retained())
    wp.capture_launch(parent.graph)
    np.testing.assert_array_equal(output.numpy(), np.full(37, 9.0))
    del parent
    gc.collect()
    test.assertIsNone(retained())


def test_sycl_graph_utility_lifetimes(test, device):
    def record(operation):
        if operation == "scan":
            source = wp.array(np.arange(6, dtype=np.int32), device=device)
            output = wp.zeros(6, dtype=int, device=device)
            refs = [weakref.ref(source)]
            expected = np.cumsum(np.arange(6, dtype=np.int32))
            with wp.ScopedCapture(device, force_module_load=False) as capture:
                wp.utils.array_scan(source, output)
        else:
            keys = wp.array([3, 1, 2, 5, 4, 0, 0, 0, 0, 0, 0, 0], dtype=int, device=device)
            output = wp.array(np.arange(12, dtype=np.int32), device=device)
            refs = [weakref.ref(keys)]
            if operation == "radix":
                expected = [5, 1, 2, 0, 4, 3]
                with wp.ScopedCapture(device, force_module_load=False) as capture:
                    wp.utils.radix_sort_pairs(keys, output, 6)
            else:
                starts = wp.array([0, 3], dtype=int, device=device)
                ends = wp.array([3, 6], dtype=int, device=device)
                refs.extend((weakref.ref(starts), weakref.ref(ends)))
                expected = [1, 2, 0, 5, 4, 3]
                with wp.ScopedCapture(device, force_module_load=False) as capture:
                    wp.utils.segmented_sort_pairs(keys, output, 6, starts, ends)
        return capture.graph, output, refs, expected

    for operation in ("scan", "radix", "segmented"):
        with test.subTest(operation=operation):
            graph, output, refs, expected = record(operation)
            gc.collect()
            test.assertTrue(all(ref() is not None for ref in refs))
            wp.capture_launch(graph)
            np.testing.assert_array_equal(output.numpy()[:6], expected)
            del graph
            gc.collect()
            test.assertTrue(all(ref() is None for ref in refs))


def test_sycl_strides(test, device):
    values = np.arange(24, dtype=np.float32).reshape(4, 6)
    x = wp.array(values, device=device)
    view = x[:, ::2]
    np.testing.assert_array_equal(view.numpy(), values[:, ::2])
    view.fill_(7.0)
    values[:, ::2] = 7.0
    np.testing.assert_array_equal(x.numpy(), values)


def test_sycl_textures(test, device):
    pixels = np.arange(16, dtype=np.float32).reshape(2, 2, 4) / 16.0
    coordinates = np.array(
        [[0.25, 0.25], [0.75, 0.75], [0.5, 0.5], [-0.25, 0.25], [1.25, 0.75], [0.0, 0.0], [1.0, 1.0]],
        dtype=np.float32,
    )
    for normalized in (False, True):
        modes = (
            tuple(wp.TextureAddressMode) if normalized else (wp.TextureAddressMode.CLAMP, wp.TextureAddressMode.BORDER)
        )
        for filtering in wp.TextureFilterMode:
            for mode in modes:
                with test.subTest(normalized=normalized, filtering=filtering, mode=mode):
                    outputs = []
                    for target in ("cpu", device):
                        texture = wp.Texture2D(
                            pixels,
                            device=target,
                            filter_mode=filtering,
                            address_mode=mode,
                            normalized_coords=normalized,
                        )
                        uv = wp.array(coordinates if normalized else coordinates * 2.0, dtype=wp.vec2, device=target)
                        arrays = [
                            wp.empty(len(coordinates), dtype=dtype, device=target)
                            for dtype in (float, wp.vec2, wp.vec4)
                        ]
                        wp.launch(sycl_texture_sample, len(coordinates), inputs=[texture, uv, *arrays], device=target)
                        outputs.append([value.numpy().copy() for value in arrays])
                    expected = outputs[0]
                    if mode == wp.TextureAddressMode.WRAP and filtering == wp.TextureFilterMode.CLOSEST:
                        # CPU nearest-wrap clamps the rounded seam index; hardware wraps it to zero.
                        rgba = pixels[[0, 1, 1, 0, 1, 0, 0], [0, 1, 1, 1, 0, 0, 0]]
                        expected = [rgba[:, 0], rgba[:, :2], rgba]
                    for actual, reference in zip(outputs[1], expected, strict=True):
                        np.testing.assert_allclose(actual, reference, rtol=1e-5, atol=1e-5)
    arrays = [wp.empty(len(coordinates), dtype=dtype, device=device) for dtype in (float, wp.vec2, wp.vec4)]
    uv = wp.array(coordinates, dtype=wp.vec2, device=device)
    wp.launch(sycl_texture_sample, len(coordinates), inputs=[None, uv, *arrays], device=device)
    for value in arrays:
        np.testing.assert_array_equal(value.numpy(), 0.0)


def test_sycl_texture_copies(test, device):
    pixels = np.arange(48, dtype=np.float32).reshape(3, 4, 4)
    texture = wp.Texture2D(pixels, device=device)
    output = np.empty_like(pixels)
    texture.copy_to(output)
    np.testing.assert_array_equal(output, pixels)
    source = wp.array(pixels * 2.0, device=device)
    texture.copy_from(source)
    target = wp.empty_like(source)
    texture.copy_to(target)
    np.testing.assert_array_equal(target.numpy(), pixels * 2.0)
    host = wp.Texture2D(pixels, device="cpu")
    host.copy_from(texture)
    host.copy_to(output)
    np.testing.assert_array_equal(output, pixels * 2.0)
    host.copy_from(pixels * 3.0)
    host.copy_to(texture)
    texture.copy_to(output)
    np.testing.assert_array_equal(output, pixels * 3.0)
    other = wp.Texture2D(width=4, height=3, num_channels=4, dtype=wp.float32, device=device)
    other.copy_from(texture)
    texture.copy_from(pixels)
    other.copy_to(texture)
    texture.copy_from(texture)
    texture.copy_to(output)
    np.testing.assert_array_equal(output, pixels * 3.0)
    with test.assertRaisesRegex(ValueError, "contiguous"):
        texture.copy_from(wp.empty((3, 8, 4), dtype=float, device=device)[:, ::2, :])
    with test.assertRaisesRegex(ValueError, "shape"):
        texture.copy_from(np.zeros((2, 4, 4), dtype=np.float32))
    with test.assertRaisesRegex(ValueError, "data type"):
        texture.copy_from(pixels.astype(np.float16))


def test_sycl_texture_graphs(test, device):
    for nworld in (1, 2, 64, 1024, 4096):
        with test.subTest(nworld=nworld):
            textures = [wp.Texture2D(np.ones((2, 2, 4), dtype=np.float32), device=device) for _ in range(2)]
            refs = [weakref.ref(texture) for texture in textures]
            array = wp.clone(wp.array(textures, dtype=wp.Texture2D, device=device)[:])
            del textures
            gc.collect()
            test.assertTrue(all(ref() is not None for ref in refs))
            selectors = wp.zeros(nworld, dtype=int, device=device)
            output = wp.empty(nworld, dtype=wp.vec4, device=device)
            uploads = [wp.empty((2, 2, 4), dtype=float, device=device) for _ in range(2)]
            for captured in (False, True):
                if captured:
                    with wp.ScopedCapture(device=device) as capture:
                        wp.launch(sycl_texture_array_sample, nworld, inputs=[array, selectors, output], device=device)
                    del array
                    gc.collect()
                    test.assertTrue(all(ref() is not None for ref in refs))
                for frame in range(3):
                    colors = np.array(
                        [
                            [frame + 1.0, frame + 1.25, frame + 1.5, 1.0],
                            [frame + 11.0, frame + 11.25, frame + 11.5, 1.0],
                        ],
                        dtype=np.float32,
                    )
                    for index in range(2):
                        uploads[index].assign(np.tile(colors[index], (2, 2, 1)))
                        refs[index]().copy_from(uploads[index])
                    indices = (np.arange(nworld, dtype=np.int32) + frame) % 2
                    selectors.assign(indices)
                    if captured:
                        wp.capture_launch(capture.graph)
                    else:
                        wp.launch(sycl_texture_array_sample, nworld, inputs=[array, selectors, output], device=device)
                    np.testing.assert_allclose(output.numpy(), colors[indices], rtol=1e-5, atol=1e-5)
            del capture
            gc.collect()
            test.assertTrue(all(ref() is None for ref in refs))


def test_sycl_texture_structs(test, device):
    textures = [wp.Texture2D(np.full((2, 2, 4), index + 1.0, dtype=np.float32), device=device) for index in range(2)]
    refs = [weakref.ref(texture) for texture in textures]
    nested = SyclNestedTextureStruct()
    nested.value.texture = textures[0]
    entry = SyclTextureStruct()
    entry.texture = textures[1]
    entries = wp.clone(wp.array([entry], dtype=SyclTextureStruct, device=device)[:])
    output = wp.empty(2, dtype=wp.vec4, device=device)
    with wp.ScopedCapture(device=device) as capture:
        wp.launch(sycl_texture_struct_sample, 1, inputs=[nested, entries, output], device=device)
    nested.value.texture = None
    entry.texture = None
    del textures, entries
    gc.collect()
    test.assertTrue(all(ref() is not None for ref in refs))
    wp.capture_launch(capture.graph)
    np.testing.assert_array_equal(output.numpy(), [[1.0] * 4, [2.0] * 4])
    del capture
    gc.collect()
    test.assertTrue(all(ref() is None for ref in refs))
    host = wp.Texture2D(np.ones((2, 2, 4), dtype=np.float32), device="cpu")
    nested.value.texture = host
    entries = wp.empty(1, dtype=SyclTextureStruct, device=device)
    with test.assertRaisesRegex(RuntimeError, "launch device"):
        wp.launch(sycl_texture_struct_sample, 1, inputs=[nested, entries, output], device=device)
    entry.texture = host
    with test.assertRaisesRegex(ValueError, "array's device"):
        wp.array([entry], dtype=SyclTextureStruct, device=device)


def test_sycl_texture_streams(test, device):
    texture = wp.Texture2D(np.ones((2, 2, 4), dtype=np.float32), device=device)
    stream_a, stream_b = wp.Stream(device), wp.Stream(device)
    ready = wp.Event(device)
    uv = wp.array([[0.5, 0.5]], dtype=wp.vec2, device=device)
    arrays = [wp.empty(1, dtype=dtype, device=device) for dtype in (float, wp.vec2, wp.vec4)]
    with wp.ScopedStream(stream_a):
        pixels = wp.full((2, 2, 4), 3.0, dtype=float, device=device)
        texture.copy_from(pixels)
        stream_a.record_event(ready)
    with wp.ScopedStream(stream_b):
        stream_b.wait_event(ready)
        with wp.ScopedCapture(device=device) as capture:
            wp.launch(sycl_texture_sample, 1, inputs=[texture, uv, *arrays], device=device)
        wp.capture_launch(capture.graph)
        np.testing.assert_array_equal(arrays[2].numpy(), [[3.0] * 4])
        target = wp.empty_like(pixels)
        texture.copy_to(target)
        np.testing.assert_array_equal(target.numpy(), np.full((2, 2, 4), 3.0, dtype=np.float32))


def test_sycl_texture_errors(test, device):
    pixels = np.ones((2, 2, 4), dtype=np.float32)
    for kwargs in (
        {"data": pixels.astype(np.float16)},
        {"data": pixels[:, :, :2]},
        {"data": pixels, "num_mip_levels": 0},
        {"data": pixels, "surface_access": True},
    ):
        with test.subTest(kwargs=kwargs), test.assertRaises(NotImplementedError):
            wp.Texture2D(device=device, **kwargs)
    with test.assertRaises(NotImplementedError):
        wp.Texture1D(np.ones((2, 4), dtype=np.float32), device=device)
    with test.assertRaises(NotImplementedError):
        wp.Texture3D(np.ones((2, 2, 2, 4), dtype=np.float32), device=device)
    for mode in (wp.TextureAddressMode.WRAP, wp.TextureAddressMode.MIRROR):
        with test.assertRaisesRegex(RuntimeError, "normalized coordinates"):
            wp.Texture2D(pixels, device=device, normalized_coords=False, address_mode=mode)
    try:
        mixed = wp.Texture2D(
            pixels, device=device, address_mode=(wp.TextureAddressMode.BORDER, wp.TextureAddressMode.CLAMP)
        )
    except RuntimeError as error:
        test.assertIn("unique per-axis texture addressing modes", str(error))
    else:
        target = np.empty_like(pixels)
        mixed.copy_to(target)
        np.testing.assert_array_equal(target, pixels)
    host = wp.Texture2D(pixels, device="cpu")
    uv = wp.array([[0.5, 0.5]], dtype=wp.vec2, device=device)
    arrays = [wp.empty(1, dtype=dtype, device=device) for dtype in (float, wp.vec2, wp.vec4)]
    with test.assertRaisesRegex(RuntimeError, "launch device"):
        wp.launch(sycl_texture_sample, 1, inputs=[host, uv, *arrays], device=device)
    with test.assertRaisesRegex(ValueError, "array's device"):
        wp.array([host], dtype=wp.Texture2D, device=device)
    texture = wp.Texture2D(pixels, device=device)
    with test.assertRaisesRegex(RuntimeError, "launch device"):
        cpu_uv = wp.array([[0.5, 0.5]], dtype=wp.vec2, device="cpu")
        cpu_arrays = [wp.empty(1, dtype=dtype, device="cpu") for dtype in (float, wp.vec2, wp.vec4)]
        wp.launch(sycl_texture_sample, 1, inputs=[texture, cpu_uv, *cpu_arrays], device="cpu")
    unused = wp.Texture2D(pixels, device=device)
    with wp.ScopedCapture(device=device) as capture:
        with test.assertRaisesRegex(RuntimeError, "before graph capture"):
            wp.Texture2D(pixels, device=device)
        with test.assertRaisesRegex(RuntimeError, "outside graph capture"):
            texture.copy_from(pixels)
        del unused
        gc.collect()
        wp.launch(sycl_texture_sample, 1, inputs=[texture, uv, *arrays], device=device)
    wp.capture_launch(capture.graph)
    np.testing.assert_array_equal(arrays[2].numpy(), np.ones((1, 4), dtype=np.float32))


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


def test_sycl_printf(test, device):
    x = wp.array(np.arange(2, dtype=np.float32), device=device)
    y = wp.zeros(2, dtype=float, device=device)
    wp.launch(sycl_printf, 2, inputs=[x, y], device=device)
    np.testing.assert_array_equal(y.numpy(), x.numpy())


def test_sycl_tile_linalg(test, device):
    rng = np.random.default_rng(0)
    a = rng.random((3, 16, 16), dtype=np.float32)
    b = rng.random((3, 16, 16), dtype=np.float32)
    c = (16.0 * np.eye(16, dtype=np.float32))[None].repeat(3, axis=0)
    rhs = rng.random((3, 16, 1), dtype=np.float32)
    arrays = [wp.array(value, device=device) for value in (a, b, c, rhs)]
    outputs = [wp.zeros((3, 16, 16), dtype=float, device=device) for _ in range(3)]
    solution = wp.zeros((3, 16, 1), dtype=float, device=device)
    wp.launch_tiled(
        sycl_tile_linalg,
        dim=3,
        inputs=arrays,
        outputs=[outputs[0], outputs[1], solution, outputs[2]],
        block_dim=64,
        device=device,
    )
    product, accumulated, factor = (output.numpy() for output in outputs)
    np.testing.assert_allclose(product, a @ b, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(accumulated, c - a.transpose(0, 2, 1) @ b, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(solution.numpy(), np.linalg.solve(factor, rhs), rtol=1e-3, atol=1e-4)


def test_sycl_large_arguments(test, device):
    # The kernel arguments exceed the hardware kernel parameter limit, so they are staged in device memory.
    n = 40
    count = 1000
    arrays = [wp.full(count, float(k + 1), dtype=float, device=device, requires_grad=True) for k in range(n)]
    out = wp.zeros(count, dtype=float, device=device, requires_grad=True)
    expected = float(n * (n + 1) // 2)

    wp.launch(sycl_many_arguments, count, inputs=arrays, outputs=[out], device=device)
    np.testing.assert_allclose(out.numpy(), expected)

    # repeated launches reuse the staging ring and must not mix up their arguments
    for step in range(600):
        wp.launch(sycl_many_arguments, count, inputs=arrays, outputs=[out], device=device)
        arrays[0].fill_(float(step))
    np.testing.assert_allclose(out.numpy(), expected - 1.0 + 598.0)

    tape = wp.Tape()
    with tape:
        wp.launch(sycl_many_arguments, count, inputs=arrays, outputs=[out], device=device)
    tape.backward(grads={out: wp.ones(count, dtype=float, device=device)})
    np.testing.assert_allclose(arrays[n - 1].grad.numpy(), 1.0)

    # captured launches own their staged arguments
    out.zero_()
    arrays[0].fill_(1.0)
    with wp.ScopedCapture(device=device) as capture:
        wp.launch(sycl_many_arguments, count, inputs=arrays, outputs=[out], device=device)
    arrays[0].fill_(2.0)
    other = wp.zeros(count, dtype=float, device=device)
    for _ in range(300):
        wp.launch(sycl_many_arguments, count, inputs=arrays, outputs=[other], device=device)
    wp.capture_launch(capture.graph)
    np.testing.assert_allclose(out.numpy(), expected + 1.0)


def test_sycl_device_info(test, device):
    test.assertGreater(device.sm_count, 0)
    test.assertGreater(device.total_memory, 0)
    test.assertLessEqual(device.free_memory, device.total_memory)

    block_size, min_grid_size = wp.get_suggested_block_size(sycl_square, device)
    test.assertGreater(block_size, 0)
    test.assertLessEqual(block_size, device.max_work_group_size)
    test.assertGreater(min_grid_size, 0)

    # max_blocks only applies to CUDA and is ignored
    x = wp.ones(1000, device=device)
    y = wp.zeros(1000, device=device)
    wp.launch(sycl_square, x.size, inputs=[x, y], max_blocks=1, device=device)
    np.testing.assert_array_equal(y.numpy(), x.numpy())


def test_sycl_ndebug(test, device):
    snippet = """
    #ifdef NDEBUG
    out[tid] = 1;
    #else
    out[tid] = 0;
    #endif
    """

    @wp.func_native(snippet)
    def define_state(
        out: wp.array[int],
        tid: int,
    ):  # fmt: skip
        ...

    @wp.kernel(module="unique")
    def release_kernel(out: wp.array[int]):
        define_state(out, wp.tid())

    @wp.kernel(module="unique")
    def debug_kernel(out: wp.array[int]):
        define_state(out, wp.tid())

    wp.set_module_options({"mode": "debug"}, debug_kernel.module)

    # release kernels must not carry the debug-only tile bounds checks and device asserts
    release = wp.zeros(1, dtype=int, device=device)
    wp.launch(release_kernel, 1, outputs=[release], device=device)
    test.assertEqual(release.numpy()[0], 1)

    debug = wp.ones(1, dtype=int, device=device)
    wp.launch(debug_kernel, 1, outputs=[debug], device=device)
    test.assertEqual(debug.numpy()[0], 0)


def test_sycl_blocked_cholesky(test, device):
    rng = np.random.default_rng(0)
    nworld = 2
    m = rng.random((nworld, SYCL_MATRIX, SYCL_MATRIX), dtype=np.float32)
    a = m @ m.transpose(0, 2, 1) + SYCL_MATRIX * np.eye(SYCL_MATRIX, dtype=np.float32)
    b = rng.random((nworld, SYCL_MATRIX, 1), dtype=np.float32)
    u = wp.zeros(a.shape, dtype=float, device=device)
    x = wp.zeros(b.shape, dtype=float, device=device)
    wp.launch_tiled(
        sycl_blocked_cholesky,
        dim=nworld,
        inputs=[wp.array(a, device=device), wp.array(b, device=device), u, x],
        block_dim=32,
        device=device,
    )
    np.testing.assert_allclose(x.numpy(), np.linalg.solve(a, b), rtol=1e-3, atol=1e-5)
    np.testing.assert_allclose(np.triu(u.numpy()), np.linalg.cholesky(a).transpose(0, 2, 1), rtol=1e-3, atol=1e-5)


def test_sycl_math_precision(test, device):
    # the default DPC++ floating-point model uses approximate trigonometry, sqrt and division
    rng = np.random.default_rng(0)
    n = 100000
    x = rng.uniform(-3.2, 3.2, n).astype(np.float32)
    y = rng.uniform(0.01, 10.0, n).astype(np.float32)
    out = wp.zeros((5, n), dtype=float, device=device)
    wp.launch(sycl_math, n, inputs=[wp.array(x, device=device), wp.array(y, device=device), out], device=device)
    out = out.numpy().astype(np.float64)
    eps = np.finfo(np.float32).eps
    x64 = x.astype(np.float64)
    for row, ref in enumerate((np.sin(x64), np.cos(x64), np.tan(x64 * np.float32(0.4)))):
        relative = np.abs(out[row] - ref) / np.maximum(np.abs(ref), 1e-3)
        test.assertLess(relative.max() / eps, 8.0)
    np.testing.assert_array_equal(out[3].astype(np.float32), np.sqrt(y))
    np.testing.assert_array_equal(out[4].astype(np.float32), x / y)


def test_sycl_streams(test, device):
    default_stream = wp.get_stream(device)
    test.assertEqual(default_stream.device, device)
    test.assertFalse(default_stream.is_capturing)

    stream = wp.Stream(device)
    other = wp.Stream(device)
    test.assertIsNot(stream, default_stream)

    values = wp.zeros(1024, dtype=int, device=device)

    # launches on explicit streams, ordered across streams with events
    wp.launch(sycl_increment, values.size, inputs=[values], stream=stream, device=device)
    event = stream.record_event()
    other.wait_event(event)
    wp.launch(sycl_scale, values.size, inputs=[values, 7], stream=other, device=device)
    wp.synchronize_stream(other)
    np.testing.assert_array_equal(values.numpy(), np.full(1024, 7))

    # the current stream is used by default and restored on exit
    values.zero_()
    with wp.ScopedStream(stream):
        test.assertIs(wp.get_stream(device), stream)
        wp.launch(sycl_increment, values.size, inputs=[values], device=device)
    test.assertIs(wp.get_stream(device), default_stream)
    wp.synchronize_stream(stream)
    np.testing.assert_array_equal(values.numpy(), np.ones(1024))

    # wait_stream orders the default stream after work queued on another stream
    values.zero_()
    wp.launch(sycl_increment, values.size, inputs=[values], stream=stream, device=device)
    default_stream.wait_stream(stream)
    wp.launch(sycl_scale, values.size, inputs=[values, 5], device=device)
    np.testing.assert_array_equal(values.numpy(), np.full(1024, 5))

    # host synchronization on a stream and an event
    wp.launch(sycl_increment, values.size, inputs=[values], stream=stream, device=device)
    done = stream.record_event()
    wp.synchronize_event(done)
    test.assertTrue(done.is_complete)
    wp.synchronize_stream(stream)
    test.assertTrue(stream.is_complete)


def test_sycl_event_timing(test, device):
    values = wp.zeros(1 << 20, dtype=int, device=device)
    start = wp.Event(device, enable_timing=True)
    end = wp.Event(device, enable_timing=True)
    with wp.ScopedDevice(device):
        wp.record_event(start)
        for _ in range(20):
            wp.launch(sycl_increment, values.size, inputs=[values])
        wp.record_event(end)
        elapsed = wp.get_event_elapsed_time(start, end)
        test.assertGreater(elapsed, 0.0)
        test.assertLess(elapsed, 10000.0)

        untimed = wp.Event(device)
        wp.record_event(untimed)
        with test.assertRaisesRegex(RuntimeError, "enable_timing"):
            wp.get_event_elapsed_time(untimed, end)


def test_sycl_graph_event_timing(test, device):
    if not device.is_graph_capture_event_supported:
        test.skipTest("Captured event timing requires Level Zero v2 native recording")
    values = wp.zeros(1 << 18, dtype=int, device=device)
    start, inner_start, inner_end, end, outer_start, outer_end = [
        wp.Event(device, enable_timing=True) for _ in range(6)
    ]
    with wp.ScopedDevice(device):
        with wp.ScopedCapture(device, force_module_load=False) as capture:
            wp.record_event(start)
            wp.launch(sycl_increment, values.size, inputs=[values])
            wp.record_event(inner_start)
            for _ in range(3):
                wp.launch(sycl_increment, values.size, inputs=[values])
            wp.record_event(inner_end)
            wp.launch(sycl_increment, values.size, inputs=[values])
            wp.record_event(end)
            with test.assertRaisesRegex(RuntimeError, "during graph capture"):
                _ = end.is_complete
            with test.assertRaisesRegex(RuntimeError, "during graph capture"):
                wp.get_event_elapsed_time(start, end, synchronize=False)
        test.assertFalse(end.is_complete)
        with test.assertRaisesRegex(RuntimeError, "must be launched"):
            wp.get_event_elapsed_time(start, end)

        streams = [wp.get_stream(device), wp.Stream(device)]
        for replay in range(4):
            with wp.ScopedStream(streams[replay % 2]):
                wp.record_event(outer_start)
                wp.capture_launch(capture.graph)
                wp.record_event(outer_end)
                outer_time = wp.get_event_elapsed_time(outer_start, outer_end)
                elapsed = wp.get_event_elapsed_time(start, end, synchronize=False)
                inner_time = wp.get_event_elapsed_time(inner_start, inner_end, synchronize=False)
                test.assertGreater(inner_time, 0.0)
                test.assertLessEqual(inner_time, elapsed)
                test.assertLessEqual(elapsed, outer_time)
                for begin, finish in ((outer_start, end), (start, outer_end)):
                    mixed_time = wp.get_event_elapsed_time(begin, finish, synchronize=False)
                    test.assertGreater(mixed_time, 0.0)
                    test.assertLessEqual(mixed_time, outer_time)
                test.assertTrue(end.is_complete)
                np.testing.assert_array_equal(values.numpy(), np.full(values.size, 5 * (replay + 1)))

        # A graph replay records its events again, even after an ordinary recording.
        wp.record_event(start)
        wp.record_event(end)
        wp.synchronize_event(end)
        wp.capture_launch(capture.graph)
        test.assertGreater(wp.get_event_elapsed_time(start, end), 0.0)


def test_sycl_graph_event_lifetimes(test, device):
    if not device.is_graph_capture_event_supported:
        test.skipTest("Captured event timing requires Level Zero v2 native recording")
    values = wp.zeros(1 << 18, dtype=int, device=device)
    start, end = [wp.Event(device, enable_timing=True) for _ in range(2)]
    with wp.ScopedDevice(device):
        with wp.ScopedCapture(device, force_module_load=False) as capture:
            wp.record_event(start)
            temporary = wp.Event(device, enable_timing=True)
            wp.record_event(temporary)
            del temporary
            gc.collect()
            wp.launch(sycl_increment, values.size, inputs=[values])
            wp.record_event(end)
        wp.capture_launch(capture.graph)
        wp.synchronize_event(end)
        elapsed = wp.get_event_elapsed_time(start, end, synchronize=False)
        del capture
        gc.collect()
        # User events retain their completed sample after the graph has been destroyed.
        test.assertEqual(wp.get_event_elapsed_time(start, end), elapsed)
        np.testing.assert_array_equal(values.numpy(), np.ones(values.size))


def test_sycl_graph_event_faulthandler(test, device):
    """Keep captured timing usable with a Python fault handler installed after GPU initialization."""
    if not device.is_graph_capture_event_supported:
        test.skipTest("Captured event timing requires Level Zero v2 native recording")
    if run_test_in_subprocess(test):
        return

    # Reinstall even if faulthandler was enabled at interpreter startup.
    faulthandler.disable()
    faulthandler.enable()
    test_sycl_graph_event_timing(test, device)
    test_sycl_graph_event_lifetimes(test, device)


def test_sycl_graph_event_waits(test, device):
    if not device.is_graph_capture_event_supported:
        test.skipTest("Captured external events require Level Zero v2 native recording")
    values = wp.zeros(1 << 18, dtype=int, device=device)
    internal = wp.Event(device)
    external = wp.Event(device)
    stream = wp.get_stream(device)
    other = wp.Stream(device)
    with wp.ScopedDevice(device):
        with wp.ScopedCapture(device, force_module_load=False) as capture:
            wp.launch(sycl_increment, values.size, inputs=[values])
            stream.record_event(internal)
            stream.wait_event(internal)
            wp.launch(sycl_increment, values.size, inputs=[values])
            stream.record_event(external, external=True)
        wp.capture_launch(capture.graph)
        other.wait_event(external)
        wp.launch(sycl_scale, values.size, inputs=[values, 7], stream=other)
        wp.synchronize_stream(other)
        test.assertTrue(external.is_complete)
        with test.assertRaisesRegex(RuntimeError, "external=True"):
            _ = internal.is_complete
        with test.assertRaisesRegex(RuntimeError, "external=True"):
            wp.synchronize_event(internal)
        np.testing.assert_array_equal(values.numpy(), np.full(values.size, 14))
        with test.assertRaisesRegex(RuntimeError, "External event waits between SYCL graphs"):
            with wp.ScopedCapture(device, force_module_load=False):
                stream.wait_event(external, external=True)
        test.assertFalse(device.is_capturing)


def test_sycl_graph_event_subgraph(test, device):
    if not device.is_graph_capture_event_supported:
        test.skipTest("Captured event timing requires Level Zero v2 native recording")
    values = wp.zeros(1 << 18, dtype=int, device=device)
    start, end = [wp.Event(device, enable_timing=True) for _ in range(2)]
    outer_start, outer_end = [wp.Event(device, enable_timing=True) for _ in range(2)]
    internal = wp.Event(device)
    with wp.ScopedDevice(device):
        with wp.ScopedCapture(device, force_module_load=False) as child:
            wp.record_event(start)
            wp.launch(sycl_increment, values.size, inputs=[values])
            wp.record_event(end)
        np.testing.assert_array_equal(values.numpy(), np.zeros(values.size))
        with wp.ScopedCapture(device, force_module_load=False) as parent:
            wp.record_event(outer_start)
            wp.launch(sycl_scale, values.size, inputs=[values, 2])
            wp.record_event(internal)
            wp.capture_launch(child.graph)
            wp.wait_event(internal)
            wp.launch(sycl_scale, values.size, inputs=[values, 3])
            wp.record_event(outer_end)
        np.testing.assert_array_equal(values.numpy(), np.zeros(values.size))
        with wp.ScopedCapture(device, force_module_load=False) as grandparent:
            wp.capture_launch(parent.graph)
        np.testing.assert_array_equal(values.numpy(), np.zeros(values.size))
        del child, parent
        gc.collect()
        expected = 0
        for _ in range(3):
            wp.capture_launch(grandparent.graph)
            outer_time = wp.get_event_elapsed_time(outer_start, outer_end)
            elapsed = wp.get_event_elapsed_time(start, end, synchronize=False)
            test.assertGreater(elapsed, 0.0)
            test.assertLessEqual(elapsed, outer_time)
            expected = (expected * 2 + 1) * 3
            np.testing.assert_array_equal(values.numpy(), np.full(values.size, expected))


def test_sycl_graph_event_unsupported_adapter(test, device):
    if device.is_graph_capture_event_supported:
        test.skipTest("The selected adapter supports captured events")
    with test.assertRaisesRegex(RuntimeError, "SYCL_UR_USE_LEVEL_ZERO_V2=1"):
        with wp.ScopedCapture(device, force_module_load=False):
            wp.get_stream(device).record_event(wp.Event(device, enable_timing=True))
    test.assertFalse(device.is_capturing)


def test_sycl_streams_with_large_arguments(test, device):
    # each stream stages its own large kernel arguments
    n = 40
    count = 256
    arrays = [wp.full(count, float(k + 1), dtype=float, device=device) for k in range(n)]
    streams = [wp.Stream(device) for _ in range(3)]
    outputs = [wp.zeros(count, dtype=float, device=device) for _ in streams]
    for _ in range(300):
        for stream, out in zip(streams, outputs, strict=True):
            wp.launch(sycl_many_arguments, count, inputs=arrays, outputs=[out], stream=stream, device=device)
    for stream in streams:
        wp.synchronize_stream(stream)
    for out in outputs:
        np.testing.assert_allclose(out.numpy(), float(n * (n + 1) // 2))


def test_sycl_scan_sort_match_cpu(test, device):
    # results must be bit-identical to the CPU implementations, including -0.0 before +0.0
    rng = np.random.default_rng(42)
    n = 300007
    cases = (
        (wp.int32, rng.integers(-(2**31), 2**31, size=n, dtype=np.int32), 0, None),
        (wp.uint32, rng.integers(0, 1000, size=n, dtype=np.uint32), 0, None),
        (wp.float32, rng.standard_normal(n).astype(np.float32), 0, None),
        (wp.int64, rng.integers(-(2**62), 2**62, size=n, dtype=np.int64), 3, 50),
        (wp.uint64, rng.integers(0, 2**63, size=n, dtype=np.uint64), 0, None),
        (wp.float64, rng.standard_normal(n), 0, None),
    )
    for dtype, keys_np, begin_bit, end_bit in cases:
        with test.subTest(dtype=dtype):
            if dtype in (wp.float32, wp.float64):
                keys_np[:4] = (-0.0, 0.0, np.inf, -np.inf)
            storage = np.zeros(2 * n, dtype=keys_np.dtype)
            storage[:n] = keys_np
            values_np = np.zeros(2 * n, dtype=np.int64)
            values_np[:n] = np.arange(n) * 3 + 1
            results = []
            for target in ("cpu", device):
                keys = wp.array(storage, dtype=dtype, device=target)
                values = wp.array(values_np, dtype=wp.int64, device=target)
                wp.utils.radix_sort_pairs(keys, values, n, begin_bit=begin_bit, end_bit=end_bit)
                results.append((keys.numpy()[:n], values.numpy()[:n]))
            np.testing.assert_array_equal(results[1][0].view(np.uint8), results[0][0].view(np.uint8))
            np.testing.assert_array_equal(results[1][1], results[0][1])

    # unordered, gapped, empty and single-element segments with many duplicate keys
    for dtype in (wp.int32, wp.float32):
        with test.subTest(segmented=dtype):
            count = 100000
            keys_np = np.zeros(2 * count, dtype=np.int32 if dtype == wp.int32 else np.float32)
            keys_np[:count] = rng.integers(-50, 50, size=count)
            values_np = np.zeros(2 * count, dtype=np.int32)
            values_np[:count] = np.arange(count)
            cuts = np.unique(np.concatenate(([0, count], rng.integers(0, count, size=500))))
            starts = cuts[:-1].copy()
            ends = cuts[1:].copy()
            starts[::3] += np.minimum(2, ends[::3] - starts[::3])
            order = rng.permutation(starts.size)
            starts, ends = starts[order].astype(np.int32), ends[order].astype(np.int32)
            results = []
            for target in ("cpu", device):
                keys = wp.array(keys_np, dtype=dtype, device=target)
                values = wp.array(values_np, dtype=wp.int32, device=target)
                wp.utils.segmented_sort_pairs(
                    keys,
                    values,
                    count,
                    wp.array(starts, dtype=wp.int32, device=target),
                    wp.array(ends, dtype=wp.int32, device=target),
                )
                results.append((keys.numpy()[:count], values.numpy()[:count]))
            np.testing.assert_array_equal(results[1][0], results[0][0])
            np.testing.assert_array_equal(results[1][1], results[0][1])

    # integer scans are exact; vector components are scanned independently
    values_np = rng.integers(-1000, 1000, size=(250001, 3), dtype=np.int64)
    for inclusive in (True, False):
        results = []
        for target in ("cpu", device):
            values = wp.array(values_np, dtype=wp.vec3l, device=target)
            out = wp.empty_like(values)
            wp.utils.array_scan(values, out, inclusive)
            results.append(out.numpy())
        np.testing.assert_array_equal(results[1], results[0])


def test_sycl_scan_in_place(test, device):
    values_np = np.arange(10000, dtype=np.int32)
    values = wp.array(values_np, device=device)
    wp.utils.array_scan(values, values, inclusive=False)
    expected = np.zeros_like(values_np)
    expected[1:] = np.cumsum(values_np)[:-1]
    np.testing.assert_array_equal(values.numpy(), expected)


def test_sycl_scan_graph(test, device):
    src = wp.zeros(100000, dtype=wp.float32, device=device)
    dst = wp.zeros_like(src)
    with wp.ScopedCapture(device, force_module_load=False) as capture:
        wp.utils.array_scan(src, dst, inclusive=True)
    for scale in (1.0, 2.0):
        values_np = np.full(src.size, scale, dtype=np.float32)
        src.assign(values_np)
        wp.capture_launch(capture.graph)
        np.testing.assert_allclose(dst.numpy(), np.cumsum(values_np, dtype=np.float64), rtol=1e-6)


def test_sycl_sort_streams(test, device):
    # each stream owns its scratch, so concurrent sorts on different streams cannot corrupt each other
    rng = np.random.default_rng(7)
    streams = [wp.Stream(device) for _ in range(3)]
    sizes = (5000, 200000, 70000)
    inputs = []
    for n in sizes:
        keys_np = rng.integers(0, 2**31 - 1, size=n, dtype=np.int32)
        keys = wp.array(np.concatenate([keys_np, np.zeros(n, dtype=np.int32)]), device=device)
        values = wp.array(np.arange(2 * n, dtype=np.int32), device=device)
        inputs.append((keys_np, keys, values))
    wp.synchronize_device(device)
    for _ in range(3):
        for stream, (keys_np, keys, values) in zip(streams, inputs, strict=True):
            with wp.ScopedStream(stream):
                wp.utils.radix_sort_pairs(keys, values, keys_np.size)
    for stream in streams:
        wp.synchronize_stream(stream)
    for keys_np, keys, values in inputs:
        order = np.argsort(keys_np, kind="stable")
        np.testing.assert_array_equal(keys.numpy()[: keys_np.size], keys_np[order])
        np.testing.assert_array_equal(values.numpy()[: keys_np.size], order)


def test_sycl_sort_graph_stream_destruction(test, device):
    # scratch recorded into a graph belongs to the graph, not to the stream that captured it
    rng = np.random.default_rng(123)
    n = 100000
    keys_np = rng.integers(0, 2**40, size=n, dtype=np.int64)
    keys = wp.zeros(2 * n, dtype=wp.int64, device=device)
    values = wp.zeros(2 * n, dtype=wp.int32, device=device)

    def upload():
        keys.assign(np.concatenate([keys_np, np.zeros(n, dtype=np.int64)]))
        values.assign(np.arange(2 * n, dtype=np.int32))

    stream = wp.Stream(device)
    with wp.ScopedStream(stream):
        upload()
        wp.utils.radix_sort_pairs(keys, values, n)
        with wp.ScopedCapture(device, force_module_load=False) as capture:
            wp.utils.radix_sort_pairs(keys, values, n)
    graph = capture.graph
    del capture, stream
    gc.collect()

    order = np.argsort(keys_np, kind="stable")
    for _ in range(2):
        upload()
        wp.capture_launch(graph)
        np.testing.assert_array_equal(keys.numpy()[:n], keys_np[order])
        np.testing.assert_array_equal(values.numpy()[:n], order)


def test_sycl_unsupported(test, device):
    x = wp.ones(1, device=device)
    with test.assertRaisesRegex(NotImplementedError, "DLPack"):
        wp.to_dlpack(x)
    with test.assertRaisesRegex(NotImplementedError, "indexed arrays"):
        wp.indexedarray(x)
    with test.assertRaisesRegex(NotImplementedError, "hash grids"):
        wp.HashGrid(4, 4, 4, device=device)
    with test.assertRaisesRegex(NotImplementedError, "array_sum"):
        wp.utils.array_sum(x)
    with test.assertRaisesRegex(NotImplementedError, "conditional graphs"):
        wp.capture_if(wp.ones(1, dtype=int, device=device), lambda: None)
    with test.assertRaisesRegex(NotImplementedError, "Raw array"):
        wp.launch(sycl_square, 1, inputs=[x.__ctype__(), x], device=device)
    value = SyclArrayStruct()
    value.values = wp.ones(1, device="cpu")
    with test.assertRaisesRegex(RuntimeError, "launch device"):
        wp.launch(sycl_struct_value, 1, inputs=[value, x], device=device)


def test_sycl_tile_sort_extremes(test, device):
    for dtype, np_dtype in (
        (wp.int32, np.int32),
        (wp.uint32, np.uint32),
        (wp.int64, np.int64),
        (wp.uint64, np.uint64),
        (wp.float32, np.float32),
    ):
        if dtype == wp.float32:
            limit = np.finfo(np_dtype).max
            samples = np.array([np.inf, -np.inf, limit, -limit, 0.0, -0.0, 1.0], dtype=np_dtype)
        else:
            limits = np.iinfo(np_dtype)
            samples = np.array([limits.max, limits.min, 0, limits.max, 1], dtype=np_dtype)
        for length in (32, 33):
            with test.subTest(dtype=dtype, length=length):
                np_keys = np.resize(samples, length)
                input_keys = wp.array(np_keys, dtype=dtype, device=device)
                input_values = wp.array(np.arange(length, dtype=np.int32), device=device)
                output_keys = wp.empty_like(input_keys)
                output_values = wp.empty_like(input_values)
                wp.launch_tiled(
                    tile_sort_tests.create_sort_kernel(dtype, length),
                    dim=1,
                    inputs=[input_keys, input_values, output_keys, output_values],
                    block_dim=16,
                    device=device,
                )
                keys, values = output_keys.numpy(), output_values.numpy()
                np.testing.assert_array_equal(keys, np.sort(np_keys))
                np.testing.assert_array_equal(np.sort(values), np.arange(length))
                np.testing.assert_array_equal(keys.view(np.uint8), np_keys[values].view(np.uint8))

    bits = np.array([0x7FC00001, 0xFFC00002, 0x7F800000, 0xFF800000, 0, 0x80000000, 0x3F800000], dtype=np.uint32)
    for length in (32, 33):
        np_keys = np.resize(bits, length).view(np.float32)
        input_keys = wp.array(np_keys, device=device)
        input_values = wp.array(np.arange(length, dtype=np.int32), device=device)
        output_keys = wp.empty_like(input_keys)
        output_values = wp.empty_like(input_values)
        wp.launch_tiled(
            tile_sort_tests.create_sort_kernel(wp.float32, length),
            dim=1,
            inputs=[input_keys, input_values, output_keys, output_values],
            block_dim=64,
            device=device,
        )
        values = output_values.numpy()
        np.testing.assert_array_equal(np.sort(values), np.arange(length))
        output_bits = output_keys.numpy().view(np.uint32)
        input_bits = np_keys.view(np.uint32)
        masks = np.where(input_bits >> 31, np.uint32(0xFFFFFFFF), np.uint32(0x80000000))
        expected_bits = input_bits[np.argsort(input_bits ^ masks, kind="stable")]
        np.testing.assert_array_equal(output_bits, expected_bits)
        np.testing.assert_array_equal(output_bits, input_bits[values])


def test_sycl_tile_sort_workgroups(test, device):
    rng = np.random.default_rng(301)
    np_keys = rng.integers(-5, 6, size=(3, 65), dtype=np.int32)
    np_values = np.tile(np.arange(65, dtype=np.int32), (3, 1))
    for block_dim in (16, 32, 48, 64, 96, 128):
        with test.subTest(block_dim=block_dim):
            keys = wp.array(np_keys, device=device)
            values = wp.array(np_values, device=device)
            wp.launch_tiled(sycl_nested_tile_sort, dim=3, inputs=[keys, values], block_dim=block_dim, device=device)
            sorted_keys, sorted_values = keys.numpy(), values.numpy()
            np.testing.assert_array_equal(sorted_keys, np.sort(np_keys, axis=1))
            np.testing.assert_array_equal(np.sort(sorted_values, axis=1), np_values)
            np.testing.assert_array_equal(sorted_keys, np.take_along_axis(np_keys, sorted_values, axis=1))


def test_sycl_tile_sort_local_memory(test, device):
    length = 1025
    block_dim = 32
    kernel = tile_sort_tests.create_payload_sort_kernel(wp.vec4, length)
    module = kernel.module
    module.get_module_hash(block_dim)
    options = module.resolved_options[block_dim]
    _, _, meta, _, _ = module._run_codegen(options, is_cpu=False, is_sycl=True)
    aligned = tile.round_up
    frames = aligned(4 * length) + aligned(16 * length)
    scratch = aligned(4 * 2048) + aligned(16 * 2048) + aligned(2048)
    test.assertEqual(meta["sycl_shared_bytes"], 16 + frames + scratch)
    required = meta["sycl_shared_bytes"] + 4 * block_dim
    if required > device.max_shared_memory_per_block:
        test.skipTest(f"Sort workspace requires {required} bytes of local memory")

    np_keys = np.arange(length - 1, -1, -1, dtype=np.int32)
    np_values = np.arange(length * 4, dtype=np.float32).reshape(length, 4)
    keys = wp.array(np_keys, device=device)
    values = wp.array(np_values, dtype=wp.vec4, device=device)
    output_keys = wp.empty_like(keys)
    output_values = wp.empty_like(values)
    with mock.patch.object(device, "max_shared_memory_per_block", required - 1):
        with test.assertRaisesRegex(RuntimeError, f"SYCL module requires {required} bytes of local memory"):
            wp.launch_tiled(
                kernel, dim=1, inputs=[keys, values, output_keys, output_values], block_dim=block_dim, device=device
            )
    wp.launch_tiled(
        kernel, dim=1, inputs=[keys, values, output_keys, output_values], block_dim=block_dim, device=device
    )
    np.testing.assert_array_equal(output_keys.numpy(), np_keys[::-1])
    np.testing.assert_array_equal(output_values.numpy(), np_values[::-1])


def test_sycl_tile_sort_graph_streams(test, device):
    streams = [wp.Stream(device), wp.Stream(device)]
    captures = []
    rng = np.random.default_rng(155)
    for stream in streams:
        with wp.ScopedStream(stream):
            keys = wp.empty((3, 65), dtype=int, device=device)
            values = wp.empty_like(keys)
            with wp.ScopedCapture(device, force_module_load=False) as capture:
                wp.launch_tiled(sycl_nested_tile_sort, dim=3, inputs=[keys, values], block_dim=32, device=device)
            captures.append((capture.graph, keys, values))
    for _ in range(3):
        expected = []
        for stream, (graph, keys, values) in zip(streams, captures, strict=True):
            np_keys = rng.integers(-7, 8, size=(3, 65), dtype=np.int32)
            np_values = np.tile(np.arange(65, dtype=np.int32), (3, 1))
            expected.append((np_keys, np_values))
            with wp.ScopedStream(stream):
                keys.assign(np_keys)
                values.assign(np_values)
                wp.capture_launch(graph)
        for stream, (_, keys, values), (np_keys, np_values) in zip(streams, captures, expected, strict=True):
            with wp.ScopedStream(stream):
                sorted_keys, sorted_values = keys.numpy(), values.numpy()
            np.testing.assert_array_equal(sorted_keys, np.sort(np_keys, axis=1))
            np.testing.assert_array_equal(np.sort(sorted_values, axis=1), np_values)
            np.testing.assert_array_equal(sorted_keys, np.take_along_axis(np_keys, sorted_values, axis=1))


class TestSycl(unittest.TestCase):
    def test_build_sycl(self):
        precise = ["-ffp-model=precise", "-foffload-fp32-prec-sqrt", "-foffload-fp32-prec-div"]
        cases = (
            ("release", None, False, True, "-O2", ["-DNDEBUG", *precise]),
            ("debug", None, False, True, "-O0", ["-g", *precise]),
            ("release", 0, True, False, "-O0", ["-DNDEBUG", "-ffast-math", "-ffp-contract=off"]),
            ("debug", 3, True, False, "-O3", ["-g", "-ffast-math", "-ffp-contract=off"]),
        )
        native = Path(build.__file__).resolve().parents[1] / "native"
        for mode, optimization, fast_math, fuse_fp, optimization_flag, flags in cases:
            with (
                self.subTest(mode=mode, optimization=optimization, fast_math=fast_math, fuse_fp=fuse_fp),
                mock.patch.object(build, "get_sycl_compiler", return_value="/compiler path/icpx") as compiler,
                mock.patch.object(build, "run_sycl_compiler") as run,
            ):
                build.build_sycl(
                    "kernel output.so",
                    "kernel source.cpp",
                    {
                        "mode": mode,
                        "optimization_level": optimization,
                        "fast_math": fast_math,
                        "fuse_fp": fuse_fp,
                    },
                )
                compiler.assert_called_once_with()
                run.assert_called_once_with(
                    [
                        "/compiler path/icpx",
                        "-fsycl",
                        "-std=c++17",
                        "-fPIC",
                        "-shared",
                        "-fvisibility=hidden",
                        "-DWP_BUILD_DLL",
                        optimization_flag,
                        "-I",
                        str(native),
                        "kernel source.cpp",
                        "-L",
                        str(native.parent / "bin"),
                        "-l:warp-sycl.so",
                        f"-Wl,-rpath,{native.parent / 'bin'}",
                        "-o",
                        "kernel output.so",
                        *flags,
                    ]
                )

    def test_sycl_compiler_discovery(self):
        with mock.patch.object(build.sys, "platform", "linux"), mock.patch.dict(build.os.environ):
            build.os.environ.pop("WARP_SYCL_COMPILER", None)
            with mock.patch.object(build.shutil, "which", return_value="/compiler/icpx") as which:
                self.assertEqual(build.get_sycl_compiler(), "/compiler/icpx")
                which.assert_called_once_with("icpx")

            with (
                mock.patch.dict(build.os.environ, {"WARP_SYCL_COMPILER": "/custom compiler/icpx"}),
                mock.patch.object(build.shutil, "which", return_value="/custom compiler/icpx") as which,
            ):
                self.assertEqual(build.get_sycl_compiler(), "/custom compiler/icpx")
                which.assert_called_once_with("/custom compiler/icpx")

            with mock.patch.object(build.shutil, "which", return_value=None):
                with self.assertRaisesRegex(RuntimeError, "SYCL compiler not found"):
                    build.get_sycl_compiler()

        with mock.patch.object(build.sys, "platform", "win32"), mock.patch.object(build.shutil, "which") as which:
            with self.assertRaisesRegex(RuntimeError, "Warp SYCL support requires Linux"):
                build.get_sycl_compiler()
            which.assert_not_called()

    def test_run_sycl_compiler(self):
        command = ["icpx", "--version"]
        result = subprocess.CompletedProcess(command, 0, stdout="compiler output", stderr="")
        with mock.patch.object(build.subprocess, "run", return_value=result) as run:
            self.assertEqual(build.run_sycl_compiler(command), "compiler output")
            run.assert_called_once_with(command, capture_output=True, text=True, check=False)

            result.returncode = 1
            result.stderr = "compiler error"
            with self.assertRaises(RuntimeError) as error:
                build.run_sycl_compiler(command)
            self.assertEqual(
                str(error.exception), "SYCL compilation failed:\nicpx --version\ncompiler output\ncompiler error"
            )

    def test_sycl_compiler_fingerprint(self):
        build.sycl_compiler_fingerprint.cache_clear()
        self.addCleanup(build.sycl_compiler_fingerprint.cache_clear)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            sources = root / "_src"
            native = root / "native"
            binary = root / "bin"
            for path in (sources, native, binary):
                path.mkdir()
            headers = [native / "a.h", native / "b.h"]
            dependencies = [
                binary / "warp-sycl.so",
                *(sources / name for name in ("codegen.py", "build.py", "build_dll.py", "context.py")),
            ]
            for path in [*headers, *dependencies]:
                path.write_bytes(path.name.encode())
            compiler = str(root / "icpx")
            digest = hashlib.sha256()
            digest.update(compiler.encode())
            digest.update(b"compiler version")
            for header in headers:
                digest.update(header.name.encode())
                digest.update(header.read_bytes())
            for path in dependencies:
                digest.update(path.read_bytes())
            with (
                mock.patch.object(build, "__file__", str(sources / "build.py")),
                mock.patch.object(build, "run_sycl_compiler", return_value="compiler version") as run,
            ):
                fingerprint = build.sycl_compiler_fingerprint(compiler)
                self.assertEqual(fingerprint, digest.hexdigest()[:16])
                self.assertEqual(build.sycl_compiler_fingerprint(compiler), fingerprint)
                run.assert_called_once_with([compiler, "--version"])
                for path in [*headers, *dependencies]:
                    with self.subTest(dependency=path.name):
                        original = path.read_bytes()
                        path.write_bytes(original + b"changed")
                        build.sycl_compiler_fingerprint.cache_clear()
                        self.assertNotEqual(build.sycl_compiler_fingerprint(compiler), fingerprint)
                        path.write_bytes(original)
                self.assertNotEqual(build.sycl_compiler_fingerprint(compiler + "-other"), fingerprint)
                run.return_value = "new compiler version"
                build.sycl_compiler_fingerprint.cache_clear()
                self.assertNotEqual(build.sycl_compiler_fingerprint(compiler), fingerprint)


devices = [device for device in wp.get_devices() if device.is_sycl]
add_function_test(TestSycl, "test_core", test_sycl_core, devices=devices)
add_function_test(TestSycl, "test_atomics", test_sycl_atomics, devices=devices)
add_function_test(TestSycl, "test_tiles", test_sycl_tiles, devices=devices)
add_function_test(TestSycl, "test_partial_tiles", test_sycl_partial_tiles, devices=devices)
add_function_test(TestSycl, "test_partial_tile_graph", test_sycl_partial_tile_graph, devices=devices)
add_function_test(TestSycl, "test_partial_tile_grad", test_sycl_partial_tile_grad, devices=devices)
add_function_test(TestSycl, "test_partial_tile_sort", test_sycl_partial_tile_sort, devices=devices)
add_function_test(TestSycl, "test_debug_tiles", test_sycl_debug_tiles, devices=devices)
add_function_test(
    TestSycl,
    "test_tile_sort_keys_and_values",
    tile_sort_tests.test_tile_sort_keys_and_values_across_dtypes_and_sizes,
    devices=devices,
)
add_function_test(
    TestSycl,
    "test_tile_sort_key_value_permutation",
    tile_sort_tests.test_tile_sort_key_value_permutation,
    devices=devices,
)
add_function_test(TestSycl, "test_tile_sort_payloads", tile_sort_tests.test_tile_sort_payloads, devices=devices)
add_function_test(
    TestSycl, "test_tile_sort_bfloat16_payload", tile_sort_tests.test_tile_sort_bfloat16_payload, devices=devices
)
add_function_test(TestSycl, "test_tile_sort_extremes", test_sycl_tile_sort_extremes, devices=devices)
add_function_test(TestSycl, "test_tile_sort_workgroups", test_sycl_tile_sort_workgroups, devices=devices)
add_function_test(TestSycl, "test_tile_sort_local_memory", test_sycl_tile_sort_local_memory, devices=devices)
add_function_test(TestSycl, "test_tile_sort_graph_streams", test_sycl_tile_sort_graph_streams, devices=devices)
add_function_test(TestSycl, "test_graph", test_sycl_graph, devices=devices)
add_function_test(TestSycl, "test_strides", test_sycl_strides, devices=devices)
add_function_test(TestSycl, "test_geometry", test_sycl_geometry, devices=devices)
add_function_test(TestSycl, "test_textures", test_sycl_textures, devices=devices)
add_function_test(TestSycl, "test_texture_copies", test_sycl_texture_copies, devices=devices)
add_function_test(TestSycl, "test_texture_graphs", test_sycl_texture_graphs, devices=devices)
add_function_test(TestSycl, "test_texture_structs", test_sycl_texture_structs, devices=devices)
add_function_test(TestSycl, "test_texture_streams", test_sycl_texture_streams, devices=devices)
add_function_test(TestSycl, "test_texture_errors", test_sycl_texture_errors, devices=devices)
add_function_test(TestSycl, "test_atomic_types", test_sycl_atomic_types, devices=devices)
add_function_test(TestSycl, "test_tile_grad", test_sycl_tile_grad, devices=devices)
add_function_test(TestSycl, "test_graph_lifetimes", test_sycl_graph_lifetimes, devices=devices)
add_function_test(TestSycl, "test_graph_alloc", graph_tests.test_graph_alloc, devices=devices)
add_function_test(
    TestSycl,
    "test_graph_launch_after_module_reload",
    reload_tests.test_graph_launch_after_module_reload,
    devices=devices,
)
add_function_test(
    TestSycl,
    "test_graph_module_unload_during_capture",
    reload_tests.test_module_unload_during_graph_capture,
    devices=devices,
)
add_function_test(TestSycl, "test_graph_allocation_capability", test_sycl_graph_allocation_capability, devices=devices)
add_function_test(
    TestSycl, "test_graph_allocation_constructors", test_sycl_graph_allocation_constructors, devices=devices
)
add_function_test(TestSycl, "test_graph_allocation_gradients", test_sycl_graph_allocation_gradients, devices=devices)
add_function_test(TestSycl, "test_graph_allocation_lifetimes", test_sycl_graph_allocation_lifetimes, devices=devices)
add_function_test(
    TestSycl,
    "test_graph_allocation_release_during_capture",
    test_sycl_graph_allocation_release_during_capture,
    devices=devices,
)
add_function_test(TestSycl, "test_graph_allocation_failures", test_sycl_graph_allocation_failures, devices=devices)
add_function_test(
    TestSycl,
    "test_graph_allocation_deferred_destruction",
    test_sycl_graph_allocation_deferred_destruction,
    devices=devices,
)
add_function_test(
    TestSycl, "test_graph_allocation_stream_lifetimes", test_sycl_graph_allocation_stream_lifetimes, devices=devices
)
add_function_test(TestSycl, "test_graph_subgraph_lifetimes", test_sycl_graph_subgraph_lifetimes, devices=devices)
add_function_test(TestSycl, "test_graph_utility_lifetimes", test_sycl_graph_utility_lifetimes, devices=devices)
add_function_test(TestSycl, "test_printf", test_sycl_printf, devices=devices, check_output=False)
add_function_test(TestSycl, "test_tile_linalg", test_sycl_tile_linalg, devices=devices)
add_function_test(TestSycl, "test_large_arguments", test_sycl_large_arguments, devices=devices)
add_function_test(TestSycl, "test_device_info", test_sycl_device_info, devices=devices)
add_function_test(TestSycl, "test_ndebug", test_sycl_ndebug, devices=devices)
add_function_test(TestSycl, "test_blocked_cholesky", test_sycl_blocked_cholesky, devices=devices)
add_function_test(TestSycl, "test_math_precision", test_sycl_math_precision, devices=devices)
add_function_test(TestSycl, "test_streams", test_sycl_streams, devices=devices)
add_function_test(TestSycl, "test_event_timing", test_sycl_event_timing, devices=devices)
add_function_test(TestSycl, "test_graph_event_timing", test_sycl_graph_event_timing, devices=devices)
add_function_test(TestSycl, "test_graph_event_lifetimes", test_sycl_graph_event_lifetimes, devices=devices)
add_function_test(TestSycl, "test_graph_event_faulthandler", test_sycl_graph_event_faulthandler, devices=devices)
add_function_test(TestSycl, "test_graph_event_waits", test_sycl_graph_event_waits, devices=devices)
add_function_test(TestSycl, "test_graph_event_subgraph", test_sycl_graph_event_subgraph, devices=devices)
add_function_test(
    TestSycl, "test_graph_event_unsupported_adapter", test_sycl_graph_event_unsupported_adapter, devices=devices
)
add_function_test(TestSycl, "test_streams_large_arguments", test_sycl_streams_with_large_arguments, devices=devices)
add_function_test(TestSycl, "test_unsupported", test_sycl_unsupported, devices=devices)
add_function_test(TestSycl, "test_scan_sort_match_cpu", test_sycl_scan_sort_match_cpu, devices=devices)
add_function_test(TestSycl, "test_scan_in_place", test_sycl_scan_in_place, devices=devices)
add_function_test(TestSycl, "test_scan_graph", test_sycl_scan_graph, devices=devices)
add_function_test(TestSycl, "test_sort_streams", test_sycl_sort_streams, devices=devices)
add_function_test(
    TestSycl, "test_sort_graph_stream_destruction", test_sycl_sort_graph_stream_destruction, devices=devices
)

# Device-generic scan and sort tests. Conditional graphs, capture pause, and capture on an explicit
# or forked stream are not supported on SYCL, so their tests are not registered here.
for _name in (
    "test_array_scan",
    "test_array_scan_vector",
    "test_array_scan_strided_views",
    "test_array_scan_empty",
    "test_array_scan_error_unsupported_dtype",
    "test_radix_sort_pairs",
    "test_radix_sort_pairs_signed_keys",
    "test_radix_sort_pairs_bit_range",
    "test_radix_sort_pairs_value_types",
    "test_radix_sort_pairs_64_bit_keys_8_byte_values",
    "test_radix_sort_pairs_empty",
    "test_radix_sort_pairs_error_unsupported_dtype",
    "test_radix_sort_pairs_error_invalid_bit_range",
    "test_radix_sort_pairs_error_insufficient_storage",
    "test_radix_sort_pairs_error_non_contiguous",
    "test_radix_sort_pairs_graph_capture",
    "test_radix_sort_pairs_graph_capture_independent_graphs",
    "test_segmented_sort_pairs",
    "test_segmented_sort_pairs_empty",
    "test_segmented_sort_pairs_no_segments",
    "test_segmented_sort_pairs_error_unsupported_dtype",
    "test_segmented_sort_pairs_error_insufficient_storage",
    "test_segmented_sort_pairs_error_non_contiguous",
    "test_segmented_sort_pairs_graph_capture",
):
    add_function_test(TestSycl, _name, getattr(utils_tests, _name), devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)

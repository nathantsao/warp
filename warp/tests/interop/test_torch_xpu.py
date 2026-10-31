# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
import unittest
import weakref
from typing import Any
from unittest import mock

import numpy as np

import warp as wp
from warp._src.dlpack import DLManagedTensor, PyCapsule_GetPointer, _c_str_dltensor
from warp._src.sycl import check
from warp.tests.unittest_utils import add_function_test


@wp.kernel
def increment(values: wp.array[Any]):
    i = wp.tid()
    values[i] = values[i] + values.dtype(1)


@wp.kernel
def increment_2d(values: wp.array2d[float]):
    i, j = wp.tid()
    values[i, j] += 1.0


def test_mapping(test, device):
    import torch  # noqa: PLC0415

    torch_device = wp.device_to_torch(device)
    test.assertEqual(wp.device_from_torch(torch_device), device)
    test.assertEqual(wp.device_from_torch(torch.device(torch_device)), device)
    # Discovery order must not participate in physical device matching.
    from warp._src.torch import _device_from_torch_xpu  # noqa: PLC0415

    _device_from_torch_xpu.cache_clear()
    with mock.patch.object(device.runtime, "sycl_devices", list(reversed(device.runtime.sycl_devices))):
        test.assertEqual(wp.device_from_torch(torch_device), device)
    library = device.runtime.sycl
    with test.assertRaisesRegex(RuntimeError, "compiler"):
        check(library, library.wp_sycl_runtime_check(1, int(torch._C._GLIBCXX_USE_CXX11_ABI)))
    with test.assertRaisesRegex(RuntimeError, "ABIs"):
        check(library, library.wp_sycl_runtime_check(int(torch.version.xpu), 1 - int(torch._C._GLIBCXX_USE_CXX11_ABI)))
    host = np.ones(8, dtype=np.float32)
    with test.assertRaisesRegex(RuntimeError, "allocation"):
        check(library, library.wp_sycl_pointer_check(device.context, host.ctypes.data))


def test_roundtrip(test, device):
    import torch  # noqa: PLC0415

    torch_device = wp.device_to_torch(device)
    scalar_types = (
        (torch.float16, wp.float16),
        (torch.bfloat16, wp.bfloat16),
        (torch.float32, wp.float32),
        (torch.float64, wp.float64),
        (torch.int8, wp.int8),
        (torch.int16, wp.int16),
        (torch.int32, wp.int32),
        (torch.int64, wp.int64),
        (torch.uint8, wp.uint8),
    )
    for torch_type, warp_type in scalar_types:
        with test.subTest(dtype=torch_type):
            owner = torch.zeros(65, dtype=torch_type, device=torch_device)
            tensor = owner[1::2]
            tensor.fill_(3)
            array = wp.from_torch(tensor, dtype=warp_type)
            test.assertEqual(array.ptr, tensor.data_ptr())
            test.assertEqual(array.strides, (tensor.element_size() * 2,))
            wp.launch(increment, array.size, inputs=[array], device=device)
            result = wp.to_torch(array)
            test.assertEqual(result.data_ptr(), tensor.data_ptr())
            test.assertEqual(result.stride(), tensor.stride())
            test.assertEqual(result.dtype, torch_type)
            torch.testing.assert_close(result, torch.full_like(result, 4))

    for dtype, shape in ((wp.vec3, (20, 3)), (wp.mat22, (20, 2, 2))):
        with test.subTest(dtype=dtype):
            tensor = torch.full(shape, 2.0, device=torch_device)[1::2]
            array = wp.from_torch(tensor, dtype=dtype)
            wp.launch(increment, array.size, inputs=[array], device=device)
            result = wp.to_torch(array)
            test.assertEqual(result.data_ptr(), tensor.data_ptr())
            torch.testing.assert_close(result, torch.full_like(result, 3))

    tensor = torch.ones(8, dtype=torch.bool, device=torch_device)
    array = wp.from_torch(tensor)
    array.fill_(False)
    torch.testing.assert_close(wp.to_torch(array), torch.zeros_like(tensor))


def test_dlpack(test, device):
    import torch  # noqa: PLC0415

    torch_device = wp.device_to_torch(device)
    source = torch.arange(64, dtype=torch.float32, device=torch_device).view(8, 8)[1::2, ::2]
    array = wp.from_dlpack(source)
    test.assertEqual(array.ptr, source.data_ptr())
    wp.launch(increment_2d, array.shape, inputs=[array], device=device)
    consumer = torch.xpu.current_stream(torch_device)
    result = torch.from_dlpack(array.__dlpack__(stream=consumer.sycl_queue))
    test.assertEqual(result.data_ptr(), source.data_ptr())
    test.assertEqual(result.stride(), source.stride())
    torch.testing.assert_close(result, source)
    capsule = wp.to_dlpack(array, stream=device.stream.sycl_queue)
    managed = DLManagedTensor.from_address(PyCapsule_GetPointer(capsule, _c_str_dltensor))
    # oneAPI device IDs alone cannot identify an allocation across runtimes.
    managed.dl_tensor.device.device_id = 12345
    test.assertEqual(wp.from_dlpack(capsule).device, device)
    with test.assertRaisesRegex(BufferError, "zero-copy"):
        array.__dlpack__(copy=True)
    with test.assertRaisesRegex(BufferError, "different device"):
        array.__dlpack__(dl_device=(1, 0))
    with test.assertRaisesRegex(TypeError, "queue pointer"):
        array.__dlpack__(stream=0)
    with test.assertRaisesRegex(ValueError, "only supported for oneAPI"):
        wp.from_dlpack(np.ones(8, dtype=np.float32), device=device)


def test_empty(test, device):
    import torch  # noqa: PLC0415

    for dtype, shape in ((wp.float32, (0, 3)), (wp.int32, (2, 0)), (wp.vec3, (0,))):
        with test.subTest(dtype=dtype, shape=shape):
            array = wp.empty(shape, dtype=dtype, device=device)
            tensor = wp.to_torch(array)
            expected_shape = (*shape, *getattr(dtype, "_shape_", ()))
            test.assertEqual(tuple(tensor.shape), expected_shape)
            test.assertEqual(tensor.device.type, "xpu")
            restored = wp.from_dlpack(tensor, dtype=dtype)
            test.assertEqual(restored.shape, shape)
            test.assertEqual(restored.device, device)
            consumer = torch.xpu.current_stream(tensor.device)
            protocol_tensor = torch.from_dlpack(array.__dlpack__(stream=consumer.sycl_queue))
            test.assertEqual(tuple(protocol_tensor.shape), expected_shape)


def test_dlpack_queue_contract(test, device):
    import torch  # noqa: PLC0415

    array = wp.zeros(8, device=device)
    for stream in (None, -1):
        with test.subTest(stream=stream):
            with test.assertRaisesRegex(BufferError, "explicit consumer queue"):
                array.__dlpack__(stream=stream)
            with test.assertRaisesRegex(BufferError, "explicit consumer queue"):
                wp.to_dlpack(array, stream=stream)
    with test.assertRaisesRegex(BufferError, "explicit consumer queue"):
        torch.from_dlpack(array)
    for stream in (0, -2, True, False, "queue"):
        with test.subTest(stream=stream):
            with test.assertRaisesRegex(TypeError, "queue pointer"):
                array.__dlpack__(stream=stream)
            with test.assertRaisesRegex(TypeError, "queue pointer"):
                wp.to_dlpack(array, stream=stream)

    restored = wp.from_dlpack(array)
    test.assertEqual(restored.ptr, array.ptr)
    np.testing.assert_array_equal(restored.numpy(), np.zeros(8))


def test_dlpack_queue_aliases(test, device):
    import torch  # noqa: PLC0415

    array = wp.zeros(8, device=device)
    consumer = torch.xpu.Stream(device=wp.device_to_torch(device))
    alias = wp.Stream(device, sycl_queue=consumer.sycl_queue)
    test.assertNotEqual(alias.sycl_queue, consumer.sycl_queue)
    first = array.__dlpack__(stream=consumer.sycl_queue)
    second = wp.to_dlpack(array, stream=alias.sycl_queue)
    test.assertEqual(len(array._dlpack_consumers), 1)

    other = torch.xpu.Stream(device=consumer.device)
    third = wp.to_dlpack(array, stream=other.sycl_queue)
    test.assertEqual(len(array._dlpack_consumers), 2)
    del first, second, third


def test_async_export_lifetime(test, device):
    import torch  # noqa: PLC0415

    for export in ("protocol", "capsule", "torch"):
        with test.subTest(export=export):
            consumer = torch.xpu.Stream(device=wp.device_to_torch(device))
            completion = torch.xpu.Event()
            guard = wp.empty(1, device=device)
            array = wp.zeros(64, device=device)
            owner = weakref.ref(array)
            deallocate = array.deleter
            observations = []

            def retire(
                ptr,
                capacity,
                guard=guard,
                deallocate=deallocate,
                observations=observations,
                completion=completion,
                consumer=consumer,
            ):
                try:
                    # Probe retirement using unused storage, never freeing live tensor data unsafely.
                    deallocate(guard.ptr, guard.capacity)
                    guard.deleter = None
                    observations.append(completion.query())
                finally:
                    consumer.synchronize()
                    deallocate(ptr, capacity)

            try:
                with torch.xpu.stream(consumer):
                    if export == "protocol":
                        tensor = torch.from_dlpack(array.__dlpack__(stream=consumer.sycl_queue))
                    elif export == "capsule":
                        tensor = torch.from_dlpack(wp.to_dlpack(array, stream=consumer.sycl_queue))
                    else:
                        tensor = wp.to_torch(array)
                    array.deleter = retire
                    torch.xpu._sleep(100_000_000)
                    tensor.add_(1)
                    completion.record(consumer)
                    del tensor, array
                test.assertIsNone(owner())
                test.assertEqual(observations, [True], "Allocation retirement did not wait for the consumer queue")
            finally:
                consumer.synchronize()


def test_imported_queue_timing(test, device):
    import torch  # noqa: PLC0415

    imported = wp.stream_from_torch(torch.xpu.Stream(device=wp.device_to_torch(device)))
    values = wp.zeros(32, device=device)
    for stream in (device.stream, imported):
        with test.subTest(stream=stream):
            stream.wait_stream(device.stream)
            start = wp.Event(device, enable_timing=True)
            end = wp.Event(device, enable_timing=True)
            stream.record_event(start)
            wp.launch(increment, values.size, inputs=[values], stream=stream)
            stream.record_event(end)
            wp.synchronize_event(end)
            elapsed = wp.get_event_elapsed_time(start, end)
            test.assertTrue(np.isfinite(elapsed))
            test.assertGreaterEqual(elapsed, 0)

    start = wp.Event(device, enable_timing=True)
    end = wp.Event(device, enable_timing=True)
    device.stream.record_event(start)
    imported.wait_event(start)
    wp.launch(increment, values.size, inputs=[values], stream=imported)
    imported.record_event(end)
    wp.synchronize_event(end)
    elapsed = wp.get_event_elapsed_time(start, end)
    test.assertTrue(np.isfinite(elapsed))
    test.assertGreaterEqual(elapsed, 0)
    np.testing.assert_array_equal(values.numpy(), np.full(32, 3.0))


def test_dlpack_capture(test, device):
    import torch  # noqa: PLC0415

    if not device.supports_graph_capture:
        test.skipTest("Graph capture is unavailable")
    consumer = torch.xpu.Stream(device=wp.device_to_torch(device))
    other = torch.xpu.Stream(device=consumer.device)
    shared = wp.stream_from_torch(consumer)
    with torch.xpu.stream(consumer), wp.ScopedStream(shared):
        array = wp.zeros(32, device=device)
        wp.launch(increment, array.size, inputs=[array], device=device)
        with wp.ScopedCapture(device=device) as capture:
            tensor = torch.from_dlpack(array.__dlpack__(stream=consumer.sycl_queue))
            alias = torch.from_dlpack(wp.to_dlpack(array, stream=consumer.sycl_queue))
            with test.assertRaises(RuntimeError):
                array.__dlpack__(stream=other.sycl_queue)
            wp.launch(increment, array.size, inputs=[array], device=device)
        wp.capture_launch(capture.graph, stream=shared)
        torch.testing.assert_close(tensor, torch.full_like(tensor, 2))
        torch.testing.assert_close(alias, tensor)
        test.assertEqual(len(array._dlpack_consumers), 1)


def test_lifetimes(test, device):
    import torch  # noqa: PLC0415

    tensor = torch.full((64,), 3.0, device=wp.device_to_torch(device))
    owner = weakref.ref(tensor)
    array = wp.from_torch(tensor)
    del tensor
    gc.collect()
    test.assertIsNotNone(owner())
    wp.launch(increment, array.size, inputs=[array], device=device)
    exported = wp.to_torch(array)
    ref = weakref.ref(array)
    del array
    gc.collect()
    test.assertIsNotNone(ref())
    torch.testing.assert_close(exported, torch.full_like(exported, 4))
    del exported
    gc.collect()
    test.assertIsNone(ref())
    test.assertIsNone(owner())


def test_streams_and_graph(test, device):
    import torch  # noqa: PLC0415

    torch_device = wp.device_to_torch(device)
    producer = torch.xpu.Stream(device=torch_device)
    consumer = torch.xpu.Stream(device=torch_device)
    warp_stream = wp.Stream(device)
    with torch.xpu.stream(producer), wp.ScopedStream(warp_stream):
        tensor = torch.zeros(256, device=torch_device)
        tensor.fill_(10)
        array = wp.from_torch(tensor)
        wp.launch(increment, array.size, inputs=[array], device=device)
    with torch.xpu.stream(consumer), wp.ScopedStream(warp_stream):
        result = wp.to_torch(array)
        torch.testing.assert_close(result, torch.full_like(result, 11))

    shared = wp.stream_from_torch(producer)
    exported_stream = wp.stream_to_torch(shared)
    with torch.xpu.stream(exported_stream), wp.ScopedStream(shared):
        tensor = torch.zeros(256, device=torch_device)
        array = wp.from_torch(tensor)
        wp.launch(increment, array.size, inputs=[array], device=device)
        with wp.ScopedCapture(device=device) as capture:
            wp.launch(increment, array.size, inputs=[array], device=device)
        owner = weakref.ref(tensor)
        del tensor, array
        gc.collect()
        test.assertIsNotNone(owner())
        for value in (4, 17, 100):
            owner().fill_(value)
            wp.capture_launch(capture.graph, stream=shared)
            torch.testing.assert_close(owner(), torch.full_like(owner(), value + 1))
    wp.synchronize_device(device)


class TestTorchXpu(unittest.TestCase):
    pass


try:
    import torch
except ImportError:
    torch = None

devices = [device for device in wp.get_devices() if device.is_sycl]
if torch is not None and torch.xpu.is_available():
    for name, function in (
        ("mapping", test_mapping),
        ("roundtrip", test_roundtrip),
        ("dlpack", test_dlpack),
        ("empty", test_empty),
        ("dlpack_queue_contract", test_dlpack_queue_contract),
        ("dlpack_queue_aliases", test_dlpack_queue_aliases),
        ("dlpack_capture", test_dlpack_capture),
        ("async_export_lifetime", test_async_export_lifetime),
        ("imported_queue_timing", test_imported_queue_timing),
        ("lifetimes", test_lifetimes),
        ("streams_and_graph", test_streams_and_graph),
    ):
        add_function_test(TestTorchXpu, f"test_{name}", function, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)

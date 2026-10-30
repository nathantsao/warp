# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ctypes
import os

from warp._src.types import array_t, float32, float64, int32, int64, uint32, uint64

# Type codes shared with warp/native/warp_sycl.cpp (scan) and warp/native/sort_sycl.cpp (sort).
SCAN_TYPES = {int32: 0, int64: 1, float32: 2, float64: 3}
SORT_KEY_TYPES = {int32: 0, uint32: 1, float32: 2, int64: 3, uint64: 4, float64: 5}


def configure_environment() -> None:
    """Set SYCL compiler defaults without overriding the caller's environment."""
    # TODO: Minimize the IGC 2.40.13 retry-stage CCD failure and remove this default after a verified fix.
    os.environ.setdefault("IGC_DisableRecompilation", "1")


def register_functions(library) -> None:
    """Register the optional SYCL library's C ABI."""
    ptr = ctypes.c_void_p
    size = ctypes.c_size_t
    integer = ctypes.c_int
    string = ctypes.c_char_p
    identifier = ctypes.c_uint64
    signatures = {
        "init": (integer, []),
        "version": (string, []),
        "get_error_string": (string, []),
        "device_count": (integer, []),
        "device_get": (ptr, [integer]),
        "device_name": (string, [ptr]),
        "device_max_work_group_size": (size, [ptr]),
        "device_local_memory_size": (size, [ptr]),
        "device_compute_units": (integer, [ptr]),
        "device_resident_threads": (size, [ptr]),
        "device_memory_info": (integer, [ptr, ctypes.POINTER(size), ctypes.POINTER(size)]),
        "device_has_fp64": (integer, [ptr]),
        "device_has_atomic64": (integer, [ptr]),
        "alloc": (ptr, [ptr, size]),
        "free": (integer, [ptr, ptr]),
        "memcpy": (integer, [ptr, ptr, ptr, size, integer]),
        "memset": (integer, [ptr, ptr, integer, size]),
        "memtile": (integer, [ptr, ptr, ptr, size, size]),
        "stage_args": (ptr, [ptr, ptr, size]),
        "stream_create": (ptr, [ptr]),
        "stream_default": (ptr, [ptr]),
        "stream_current": (ptr, [ptr]),
        "stream_destroy": (integer, [ptr, ptr]),
        "stream_set_current": (integer, [ptr, ptr, integer]),
        "stream_synchronize": (integer, [ptr, ptr]),
        "stream_query": (integer, [ptr]),
        "stream_wait_event": (integer, [ptr, ptr, ptr, integer]),
        "event_create": (ptr, [ptr, integer]),
        "event_destroy": (None, [ptr]),
        "event_record": (integer, [ptr, ptr, ptr, integer]),
        "event_synchronize": (integer, [ptr]),
        "event_query": (integer, [ptr]),
        "event_elapsed_time": (integer, [ptr, ptr, ctypes.POINTER(ctypes.c_float)]),
        "synchronize": (integer, [ptr]),
        "load_module": (ptr, [string]),
        "get_kernel": (ptr, [ptr, string]),
        "unload_module": (integer, [ptr, ptr]),
        "graph_supported": (integer, [ptr]),
        "graph_event_supported": (integer, [ptr]),
        "graph_begin": (ptr, [ptr]),
        "graph_end": (integer, [ptr, ptr]),
        "graph_launch": (integer, [ptr, ptr]),
        "graph_destroy": (integer, [ptr, ptr]),
        "graph_memory_info": (integer, [ptr, ctypes.POINTER(size), ctypes.POINTER(size)]),
        "array_copy": (integer, [ptr, ptr, ptr, size, size, size, size]),
        "array_fill": (integer, [ptr, ptr, ptr, size, size]),
        "array_scan": (integer, [ptr, integer, ptr, ptr, integer, integer, integer, integer, integer]),
        "radix_sort_pairs": (integer, [ptr, integer, ptr, ptr, integer, integer, integer, integer]),
        "segmented_sort_pairs": (integer, [ptr, integer, ptr, ptr, integer, ptr, ptr, integer]),
        "bvh_create": (identifier, [ptr, ptr, ptr, integer, integer, ptr, integer]),
        "bvh_destroy": (integer, [identifier]),
        "bvh_refit": (integer, [identifier]),
        "bvh_rebuild": (integer, [identifier, integer]),
        "mesh_create": (identifier, [ptr, array_t, array_t, array_t, integer, integer, integer, integer, ptr, integer]),
        "mesh_destroy": (integer, [identifier]),
        "mesh_refit": (integer, [identifier]),
        "mesh_set_points": (integer, [identifier, array_t]),
        "mesh_set_velocities": (integer, [identifier, array_t]),
        "texture_create": (identifier, [ptr, integer, integer, integer, integer, integer, integer, integer, integer]),
        "texture_destroy": (integer, [ptr, identifier]),
        "texture_copy": (integer, [ptr, identifier, ptr, identifier, ptr, integer]),
    }
    for name, (restype, argtypes) in signatures.items():
        function = getattr(library, f"wp_sycl_{name}")
        function.restype = restype
        function.argtypes = argtypes


def check(library, result: int) -> None:
    if result:
        raise RuntimeError(f"Warp SYCL error: {library.wp_sycl_get_error_string().decode('utf-8')}")


def _resource_refs(value):
    from warp._src.codegen import StructInstance  # noqa: PLC0415

    if value is not None:
        yield value
        if isinstance(value, StructInstance):
            for name in value._cls.vars:
                yield from _resource_refs(getattr(value, name))


def texture_array_refs(values, device):
    from warp._src.texture import Texture  # noqa: PLC0415

    refs = tuple(dict.fromkeys(ref for value in values for ref in _resource_refs(value) if isinstance(ref, Texture)))
    if any(ref.device != device for ref in refs):
        raise ValueError("SYCL texture arrays require textures on the array's device")
    return refs


def retain_arrays(device, *arrays) -> None:
    if device.is_capturing:
        device.captures[None]._sycl_refs.extend(ref for value in arrays for ref in _resource_refs(value))


class SyclAllocator:
    deallocate_requires_context_guard = False

    def __init__(self, device):
        self.device = device

    def allocate(self, size_in_bytes):
        library = self.device.runtime.sycl
        ptr = library.wp_sycl_alloc(self.device.context, size_in_bytes)
        if ptr is None:
            check(library, -1)
        return ptr

    def deallocate(self, ptr, size_in_bytes):
        library = self.device.runtime.sycl
        check(library, library.wp_sycl_free(self.device.context, ptr))


def kernel_hook(device, module, name: str, adjoint: bool):
    library = device.runtime.sycl
    address = library.wp_sycl_get_kernel(module, name.encode("utf-8"))
    if address is None:
        check(library, -1)
    signature = [ctypes.c_void_p] * (4 if adjoint else 3)
    entry = ctypes.CFUNCTYPE(ctypes.c_int, *signature)(address)

    def invoke(*args):
        check(library, entry(device.context, *args))

    return invoke

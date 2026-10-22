# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ctypes

from warp._src.types import array_t


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
        "device_has_fp64": (integer, [ptr]),
        "device_has_atomic64": (integer, [ptr]),
        "alloc": (ptr, [ptr, size]),
        "free": (integer, [ptr, ptr]),
        "memcpy": (integer, [ptr, ptr, ptr, size, integer]),
        "memset": (integer, [ptr, ptr, integer, size]),
        "memtile": (integer, [ptr, ptr, ptr, size, size]),
        "synchronize": (integer, [ptr]),
        "load_module": (ptr, [string]),
        "get_kernel": (ptr, [ptr, string]),
        "unload_module": (integer, [ptr, ptr]),
        "graph_supported": (integer, [ptr]),
        "graph_begin": (ptr, [ptr]),
        "graph_end": (integer, [ptr, ptr]),
        "graph_launch": (integer, [ptr, ptr]),
        "graph_destroy": (integer, [ptr, ptr]),
        "array_copy": (integer, [ptr, ptr, ptr, size, size, size, size]),
        "array_fill": (integer, [ptr, ptr, ptr, size, size]),
        "bvh_create": (identifier, [ptr, ptr, ptr, integer, integer, ptr, integer]),
        "bvh_destroy": (integer, [identifier]),
        "bvh_refit": (integer, [identifier]),
        "bvh_rebuild": (integer, [identifier, integer]),
        "mesh_create": (identifier, [ptr, array_t, array_t, array_t, integer, integer, integer, integer, ptr, integer]),
        "mesh_destroy": (integer, [identifier]),
        "mesh_refit": (integer, [identifier]),
        "mesh_set_points": (integer, [identifier, array_t]),
        "mesh_set_velocities": (integer, [identifier, array_t]),
    }
    for name, (restype, argtypes) in signatures.items():
        function = getattr(library, f"wp_sycl_{name}")
        function.restype = restype
        function.argtypes = argtypes


def check(library, result: int) -> None:
    if result:
        raise RuntimeError(f"Warp SYCL error: {library.wp_sycl_get_error_string().decode('utf-8')}")


class SyclAllocator:
    deallocate_requires_context_guard = False

    def __init__(self, device):
        self.device = device

    def allocate(self, size_in_bytes):
        if self.device.is_capturing:
            raise NotImplementedError("Allocations during SYCL graph capture are not supported")
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

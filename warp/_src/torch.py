# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ctypes
from functools import cache
from typing import TYPE_CHECKING

import numpy

import warp
import warp._src.context

if TYPE_CHECKING:
    import torch


@cache
def _check_torch_xpu_runtime():
    import torch  # noqa: PLC0415

    from warp._src.sycl import check  # noqa: PLC0415

    warp.init()
    library = warp._src.context.runtime.sycl
    version = torch.version.xpu
    if library is None or version is None:
        raise RuntimeError("SYCL interoperability requires both Warp SYCL and an XPU-enabled PyTorch build")
    check(library, library.wp_sycl_runtime_check(int(version), int(torch._C._GLIBCXX_USE_CXX11_ABI)))


@cache
def _device_from_torch_xpu(index: int):
    import torch  # noqa: PLC0415

    from warp._src.sycl import check  # noqa: PLC0415

    _check_torch_xpu_runtime()
    runtime = warp._src.context.runtime
    stream = torch.xpu.current_stream(index)
    context = runtime.sycl.wp_sycl_device_from_queue(stream.sycl_queue)
    if context is None:
        check(runtime.sycl, -1)
    for device in runtime.sycl_devices:
        if device.context == context:
            return device
    raise RuntimeError(f"No Warp device is registered for Torch XPU device {index}")


# return the warp device corresponding to a torch device
def device_from_torch(torch_device: torch.device | str) -> warp.Device:
    """Return the Warp device corresponding to a Torch device.

    Args:
        torch_device: Torch device identifier

    Raises:
        RuntimeError: Torch device does not have a corresponding Warp device
    """
    warp.init()
    if type(torch_device) is str:
        if torch_device == "xpu" or torch_device.startswith("xpu:"):
            import torch  # noqa: PLC0415

            torch_device = torch.device(torch_device)
            index = torch.xpu.current_device() if torch_device.index is None else torch_device.index
            return _device_from_torch_xpu(index)
        warp_device = warp._src.context.runtime.device_map.get(torch_device)
        if warp_device is not None:
            return warp_device
        elif torch_device == "cuda":
            return warp._src.context.runtime.get_current_cuda_device()
        else:
            raise RuntimeError(f"Unsupported Torch device {torch_device}")
    else:
        try:
            if torch_device.type == "cuda":
                if torch_device.index is None:
                    import torch  # noqa: PLC0415

                    return warp._src.context.runtime.cuda_devices[torch.cuda.current_device()]
                return warp._src.context.runtime.cuda_devices[torch_device.index]
            elif torch_device.type == "cpu":
                return warp._src.context.runtime.cpu_device
            elif torch_device.type == "xpu":
                import torch  # noqa: PLC0415

                index = torch.xpu.current_device() if torch_device.index is None else torch_device.index
                return _device_from_torch_xpu(index)
            else:
                raise RuntimeError(f"Unsupported Torch device type {torch_device.type}")
        except Exception as e:
            import torch  # noqa: PLC0415

            if not isinstance(torch_device, torch.device):
                raise ValueError("Argument must be a torch.device object or a string") from e
            raise


def device_to_torch(warp_device: warp.DeviceLike) -> str:
    """Return the Torch device string corresponding to a Warp device.

    Args:
        warp_device: An identifier that can be resolved to a :class:`warp.Device`.

    Raises:
        RuntimeError: The Warp device is not compatible with PyTorch.
    """
    device = warp.get_device(warp_device)
    if device.is_sycl:
        import torch  # noqa: PLC0415

        _check_torch_xpu_runtime()
        for index in range(torch.xpu.device_count()):
            if _device_from_torch_xpu(index) == device:
                return f"xpu:{index}"
        raise RuntimeError(f"Warp device {device} has no corresponding Torch XPU device")
    if device.is_cpu or device.is_primary:
        return str(device)
    elif device.is_cuda and device.is_uva:
        # it's not a primary context, but torch can access the data ptr directly thanks to UVA
        return f"cuda:{device.ordinal}"
    raise RuntimeError(f"Warp device {device} is not compatible with torch")


def _prepare_xpu_tensor(t, device):
    from warp._src.sycl import check  # noqa: PLC0415

    producer = stream_from_torch(t.device)
    consumer = device.stream
    library = device.runtime.sycl
    check(library, library.wp_sycl_pointer_check(device.context, t.data_ptr()))
    check(library, library.wp_sycl_queue_wait_stream(device.context, consumer.sycl_queue, producer.sycl_stream))
    t.record_stream(stream_to_torch(consumer))
    return producer, consumer


def dtype_to_torch(warp_dtype):
    """Return the Torch dtype corresponding to a Warp dtype.

    Args:
        warp_dtype: A Warp data type that has a corresponding ``torch.dtype``.
            ``warp.uint16``, ``warp.uint32``, and ``warp.uint64`` are mapped
            to the signed integer ``torch.dtype`` of the same width.
    Raises:
        TypeError: Unable to find a corresponding PyTorch data type.
    """
    # initialize lookup table on first call to defer torch import
    if dtype_to_torch.type_map is None:
        import torch  # noqa: PLC0415

        dtype_to_torch.type_map = {
            warp.float16: torch.float16,
            warp.bfloat16: torch.bfloat16,
            warp.float32: torch.float32,
            warp.float64: torch.float64,
            warp.int8: torch.int8,
            warp.int16: torch.int16,
            warp.int32: torch.int32,
            warp.int64: torch.int64,
            warp.uint8: torch.uint8,
            # torch doesn't support unsigned ints bigger than 8 bits
            warp.uint16: torch.int16,
            warp.uint32: torch.int32,
            warp.uint64: torch.int64,
            warp.bool: torch.bool,
        }

    torch_dtype = dtype_to_torch.type_map.get(warp_dtype)
    if torch_dtype is not None:
        return torch_dtype
    else:
        raise TypeError(f"Cannot convert {warp_dtype} to a Torch type")


def dtype_from_torch(torch_dtype):
    """Return the Warp dtype corresponding to a Torch dtype.

    Args:
        torch_dtype: A ``torch.dtype`` that has a corresponding Warp data type.
            Currently ``torch.complex64`` and ``torch.complex128`` are not
            supported.

    Raises:
        TypeError: Unable to find a corresponding Warp data type.
    """
    # initialize lookup table on first call to defer torch import
    if dtype_from_torch.type_map is None:
        import torch  # noqa: PLC0415

        dtype_from_torch.type_map = {
            torch.float16: warp.float16,
            torch.bfloat16: warp.bfloat16,
            torch.float32: warp.float32,
            torch.float64: warp.float64,
            torch.int8: warp.int8,
            torch.int16: warp.int16,
            torch.int32: warp.int32,
            torch.int64: warp.int64,
            torch.uint8: warp.uint8,
            torch.bool: warp.bool,
            # currently unsupported by Warp
            # torch.complex64:
            # torch.complex128:
        }

    warp_dtype = dtype_from_torch.type_map.get(torch_dtype)

    if warp_dtype is not None:
        return warp_dtype
    else:
        raise TypeError(f"Cannot convert {torch_dtype} to a Warp type")


def dtype_is_compatible(torch_dtype, warp_dtype) -> bool:
    """Evaluate whether the given torch dtype is compatible with the given Warp dtype."""
    # initialize lookup table on first call to defer torch import
    if dtype_is_compatible.compatible_sets is None:
        import torch  # noqa: PLC0415

        dtype_is_compatible.compatible_sets = {
            torch.float64: {warp.float64},
            torch.float32: {warp.float32},
            torch.float16: {warp.float16},
            torch.bfloat16: {warp.bfloat16},
            # allow aliasing integer tensors as signed or unsigned integer arrays
            torch.int64: {warp.int64, warp.uint64},
            torch.int32: {warp.int32, warp.uint32},
            torch.int16: {warp.int16, warp.uint16},
            torch.int8: {warp.int8, warp.uint8},
            torch.uint8: {warp.uint8, warp.int8},
            torch.bool: {warp.bool, warp.uint8, warp.int8},
            # currently unsupported by Warp
            # torch.complex64:
            # torch.complex128:
        }

    compatible_set = dtype_is_compatible.compatible_sets.get(torch_dtype)

    if compatible_set is not None:
        if warp_dtype in compatible_set:
            return True
        # check if it's a vector or matrix type
        if hasattr(warp_dtype, "_wp_scalar_type_"):
            return warp_dtype._wp_scalar_type_ in compatible_set

    return False


# lookup tables initialized when needed
dtype_from_torch.type_map = None
dtype_to_torch.type_map = None
dtype_is_compatible.compatible_sets = None


# wrap a PyTorch tensor to a wp.array, data is not copied
def from_torch(
    t: torch.Tensor,
    dtype: type | None = None,
    requires_grad: bool | None = None,
    grad=None,
    return_ctype: bool = False,
    retain_grad: bool = False,
) -> warp.array | warp._src.types.array_t:
    """Convert a PyTorch tensor to a Warp array without copying the data.

    The resulting Warp array aliases the PyTorch tensor's storage, so mutations
    through either view are visible from the other.
    Scalar Warp dtypes preserve the PyTorch tensor's shape and strides. Vector
    and matrix Warp dtypes consume trailing component dimensions, and those
    trailing dimensions must be contiguous.

    Args:
        t: The PyTorch tensor to wrap.
        dtype: The target data type of the resulting Warp array. Defaults to the
          tensor value type mapped to a Warp array value type.
        requires_grad: Whether the resulting array should wrap the tensor's gradient,
          if it exists (the grad tensor will be allocated otherwise). Defaults to the tensor's ``requires_grad`` value.
        grad: Optional gradient array to attach to the result. Can be a Warp array or PyTorch tensor.
          If not provided and ``requires_grad`` is True, the tensor's gradient will be wrapped or allocated.
        return_ctype: Whether to return a low-level array descriptor instead of a ``wp.array`` object (faster).
          The descriptor can be passed to Warp kernels.
        retain_grad: Whether to preserve gradients during backward instead of zeroing after read.

    Returns:
        The wrapped array or array descriptor.
    """
    if dtype is None:
        dtype = dtype_from_torch(t.dtype)
    elif not dtype_is_compatible(t.dtype, dtype):
        raise RuntimeError(f"Cannot convert Torch type {t.dtype} to Warp type {dtype}")

    device = device_from_torch(t.device)
    interop_streams = _prepare_xpu_tensor(t, device) if device.is_sycl else None

    # get size of underlying data type to compute strides
    ctype_size = ctypes.sizeof(dtype._type_)

    shape = tuple(t.shape)
    strides = tuple(s * ctype_size for s in t.stride())

    # if target is a vector or matrix type
    # then check if trailing dimensions match
    # the target type and update the shape
    if hasattr(dtype, "_shape_"):
        dtype_shape = dtype._shape_
        dtype_dims = len(dtype._shape_)
        # ensure inner shape matches
        if dtype_dims > len(shape) or dtype_shape != shape[-dtype_dims:]:
            raise RuntimeError(
                f"Could not convert PyTorch tensor with shape {shape} to Warp array with dtype={dtype}, "
                f"ensure that source inner shape is {dtype_shape}"
            )
        # ensure inner strides are contiguous
        if strides[-1] != ctype_size or (dtype_dims > 1 and strides[-2] != ctype_size * dtype_shape[-1]):
            raise RuntimeError(
                f"Could not convert PyTorch tensor with shape {shape} to Warp array with dtype={dtype}, "
                "because the source inner strides are not contiguous"
            )
        # trim shape and strides
        shape = tuple(shape[:-dtype_dims]) or (1,)
        strides = tuple(strides[:-dtype_dims]) or (ctype_size,)

    # gradient
    # - if return_ctype is False, we set `grad` to a wp.array or None
    # - if return_ctype is True, we set `grad_ptr` and set `grad` as the owner (wp.array or torch.Tensor)
    requires_grad = t.requires_grad if requires_grad is None else requires_grad
    grad_ptr = 0
    if grad is not None:
        if isinstance(grad, warp.array):
            if return_ctype:
                if grad.strides != strides:
                    raise RuntimeError(
                        f"Gradient strides must match array strides, expected {strides} but got {grad.strides}"
                    )
                grad_ptr = grad.ptr
        else:
            # assume grad is a torch.Tensor
            if return_ctype:
                if t.stride() != grad.stride():
                    raise RuntimeError(
                        f"Gradient strides must match array strides, expected {t.stride()} but got {grad.stride()}"
                    )
                grad_ptr = grad.data_ptr()
            else:
                grad = from_torch(grad, dtype=dtype, requires_grad=False)
    elif requires_grad:
        # wrap the tensor gradient, allocate if necessary
        if t.grad is not None:
            if return_ctype:
                grad = t.grad
                if t.stride() != grad.stride():
                    raise RuntimeError(
                        f"Gradient strides must match array strides, expected {t.stride()} but got {grad.stride()}"
                    )
                grad_ptr = grad.data_ptr()
            else:
                grad = from_torch(t.grad, dtype=dtype, requires_grad=False)
        else:
            # allocate a zero-filled gradient if it doesn't exist
            # Note: we use Warp to allocate the shared gradient with compatible strides
            grad = warp.zeros(dtype=dtype, shape=shape, strides=strides, device=device_from_torch(t.device))
            t.grad = to_torch(grad, requires_grad=False)
            grad_ptr = grad.ptr

    if return_ctype:
        ptr = t.data_ptr()

        # create array descriptor
        array_ctype = warp._src.types.array_t(ptr, grad_ptr, len(shape), shape, strides)

        # keep data and gradient alive
        array_ctype._ref = t
        array_ctype._gradref = grad
        if interop_streams is not None:
            array_ctype._interop_streams = interop_streams

        return array_ctype

    else:
        ptr = t.data_ptr()
        capacity = 0
        if ptr:
            # A view's data pointer includes its storage offset. Warp capacity
            # is measured from ptr, so exclude the preceding storage bytes.
            storage = t.untyped_storage()
            storage_base_ptr = storage.data_ptr()
            storage_nbytes = storage.nbytes()
            capacity = storage_nbytes - (ptr - storage_base_ptr)

        a = warp.array(
            ptr=ptr,
            dtype=dtype,
            shape=shape,
            strides=strides,
            capacity=capacity,
            device=device,
            copy=False,
            grad=grad,
            requires_grad=requires_grad,
            retain_grad=retain_grad,
        )

        # save a reference to the source tensor, otherwise it may get deallocated
        a._tensor = t
        if interop_streams is not None:
            a._interop_streams = interop_streams
        if ptr:
            # APIC uses allocation-wide bounds to recognize independently
            # converted views of the same PyTorch storage as one memory region.
            a._storage_base_ptr = storage_base_ptr
            a._storage_nbytes = storage_nbytes

        return a


def to_torch(a: warp.array, requires_grad: bool | None = None):
    """Convert a Warp array to a PyTorch tensor without copying the data.

    The returned PyTorch tensor aliases the Warp array's storage. Clone the
    tensor if PyTorch must retain those values after Warp may modify or zero the
    underlying array.

    Args:
        a: The Warp array to convert.
        requires_grad: Whether the resulting tensor should convert the array's
          gradient, if it exists, to a grad tensor. Defaults to the array's
          ``requires_grad`` value.

    Returns:
        torch.Tensor: The converted PyTorch tensor.
    """
    import torch  # noqa: PLC0415

    if requires_grad is None:
        requires_grad = a.requires_grad

    # Torch does not support structured arrays
    if isinstance(a.dtype, warp._src.codegen.Struct):
        raise RuntimeError("Cannot convert structured Warp arrays to Torch.")

    if a.device.is_sycl:
        torch_device = device_to_torch(a.device)
        consumer = torch.xpu.current_stream(torch_device)
        with torch.xpu.device(torch_device):
            t = torch.from_dlpack(a.__dlpack__(stream=consumer.sycl_queue))
        t.requires_grad = requires_grad
        if requires_grad and a.requires_grad:
            t.grad = to_torch(a.grad, requires_grad=False)
        t._warp_array = a
        return t

    # bfloat16 is not representable via __array_interface__ or __cuda_array_interface__
    # (it exposes as uint16), so use DLPack for a correct dtype round-trip.
    # This also covers compound types (vectors/matrices) whose scalar type is bfloat16.
    scalar_type = getattr(a.dtype, "_wp_scalar_type_", a.dtype)
    if scalar_type is warp.bfloat16:
        t = torch.from_dlpack(warp.to_dlpack(a))
        t.requires_grad = requires_grad
        if requires_grad and a.requires_grad:
            t.grad = torch.from_dlpack(warp.to_dlpack(a.grad))
            t.grad._warp_grad_array = a.grad
        # prevent GC of the Warp array while the tensor is alive
        t._warp_array = a
        return t

    if a.device.is_cpu:
        # Torch has an issue wrapping CPU objects
        # that support the __array_interface__ protocol
        # in this case we need to workaround by going
        # to an ndarray first, see https://pearu.github.io/array_interface_pytorch.html
        t = torch.as_tensor(numpy.asarray(a))
        t.requires_grad = requires_grad
        if requires_grad and a.requires_grad:
            t.grad = torch.as_tensor(numpy.asarray(a.grad))
        return t

    elif a.device.is_cuda:
        # Torch does support the __cuda_array_interface__
        # correctly, but we must be sure to maintain a reference
        # to the owning object to prevent memory allocs going out of scope
        t = torch.as_tensor(a, device=device_to_torch(a.device))
        t.requires_grad = requires_grad
        if requires_grad and a.requires_grad:
            t.grad = torch.as_tensor(a.grad, device=device_to_torch(a.device))
        return t

    else:
        raise RuntimeError("Unsupported device")


def stream_from_torch(stream_or_device=None):
    """Convert a Torch CUDA stream or XPU queue to a Warp stream.

    Streams created using ``torch.cuda.Stream()`` are non-blocking and the returned
    ``warp.Stream`` inherits that behavior.
    When using those streams to run Warp operations, ensure that Warp-owned resources are not garbage-collected while work on the non-blocking stream is still pending.
    See :ref:`nonblocking_streams` for details and mitigation strategies.
    """
    import torch  # noqa: PLC0415

    if isinstance(stream_or_device, torch.xpu.Stream):
        stream = stream_or_device
    elif (isinstance(stream_or_device, torch.device) and stream_or_device.type == "xpu") or (
        isinstance(stream_or_device, str) and (stream_or_device == "xpu" or stream_or_device.startswith("xpu:"))
    ):
        stream = torch.xpu.current_stream(stream_or_device)
    else:
        stream = None

    if stream is not None:
        device = device_from_torch(stream.device)
        warp_stream = warp.Stream(device, sycl_queue=stream.sycl_queue)
        warp_stream._torch_stream = stream
        return warp_stream

    if isinstance(stream_or_device, torch.cuda.Stream):
        stream = stream_or_device
    else:
        # assume arg is a torch device
        stream = torch.cuda.current_stream(stream_or_device)

    device = device_from_torch(stream.device)

    warp_stream = warp.Stream(device, cuda_stream=stream.cuda_stream)

    # save a reference to the source stream, otherwise it may be destroyed
    warp_stream._torch_stream = stream

    return warp_stream


def stream_to_torch(stream_or_device=None):
    """Convert a Warp CUDA stream or SYCL queue to a Torch stream.

    Streams created using ``warp.Stream()`` are blocking and the returned PyTorch ``ExternalStream`` inherits that behavior.
    This means PyTorch operations on the returned stream will implicitly synchronize
    with the CUDA default stream, matching the default Warp behavior.
    See :ref:`nonblocking_streams` for background on blocking vs. non-blocking streams.
    """
    import torch  # noqa: PLC0415

    if isinstance(stream_or_device, warp.Stream):
        stream = stream_or_device
    else:
        # assume arg is a warp device
        stream = warp.get_device(stream_or_device).stream

    device = device_to_torch(stream.device)

    if stream.device.is_sycl:
        torch_stream = torch.xpu.get_stream_from_external(stream.sycl_queue, device=device)
    else:
        torch_stream = torch.cuda.ExternalStream(stream.cuda_stream, device=device)

    # save a reference to the source stream, otherwise it may be destroyed
    torch_stream._warp_stream = stream

    return torch_stream

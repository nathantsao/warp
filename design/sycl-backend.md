# Intel SYCL Backend

**Status**: Implemented (experimental, staged coverage)

## Motivation

Run Warp kernels on Intel GPUs without treating GPU memory as CPU memory or
forking the CPU/CUDA implementation. The older SYCL patch is a capability
reference, not a source-compatible patch for this version.

## Design

### Changes and reasons

| Change | Reason |
| --- | --- |
| Add explicit `Device.backend` / `is_sycl` and `sycl:N` devices. | Keep CPU and CUDA dispatch truthful; preserve existing default-device selection. |
| Build an optional `warp-sycl.so` with `icpx`, from both build frontends. | Keep oneAPI dependencies out of ordinary CPU/CUDA builds. |
| Enumerate Intel Level Zero GPUs and own one in-order queue per device. | Avoid duplicate OpenCL devices and preserve operation ordering without pretending queues are CUDA streams. |
| Use device USM for arrays and checked native allocation/copy/fill/synchronization APIs. | Avoid accidental host dereferences; report SYCL exceptions through Warp's Python error path. |
| Reuse Warp's module builder, argument descriptors, and generated forward/backward bodies. | Add a launch backend rather than duplicating the compiler or exposing Python replicas of native layouts. |
| Submit `nd_range` kernels with value-captured arguments and bounds guards. | Preserve asynchronous argument lifetimes and support non-multiple ordinary launch sizes. |
| Give work-group/local-storage variants distinct SYCL kernel names. | Prevent DPC++ device-image collisions between differently sized variants. |
| Give SYCL binaries and metadata their own compiler/runtime/header fingerprint. | Prevent CPU/CUDA cache collisions and stale ABI reuse. |
| Use typed atomic CAS updates, including packed half words and unsigned integer addition. | Preserve contended updates, neighboring half values, and signed wrapping semantics. |
| Use work-group barriers and local storage for basic tiles. | Reuse portable algorithms across actual GPU work-items, not a single CPU-like work-item. |
| Keep CUDA shuffle/mask paths CUDA-only and query subgroup width at runtime. | Intel subgroup width is not universally 32; work-group size is not subgroup size. |
| Track transitive tile use and reject incomplete tile work-groups. | Every participating work-item must reach collective barriers, including inside user functions. |
| Route native diagnostics through SYCL's device `printf`. | Preserve diagnostics rather than silently disabling them. |
| Build BVHs/meshes with existing native host SAH/median builders, then own shared-USM native structures. | Reuse current layouts and algorithms without copying private C++ layouts into Python. |
| Refit ordinary geometry on the GPU; stage winding-number refits explicitly on the host. | Support queries/refits while keeping the host portion visible and bounded. |
| Give empty SYCL trees a valid empty leaf/root. | Native queries dereference the root even when the host builder produced no nodes. |
| Always compile oneAPI command graphs with the SYCL backend and retain captured arrays/modules. | Match CUDA's backend-level inclusion, require a graph-capable compiler, and preserve buffer/code lifetimes without emulating CUDA graph handles. |
| Reject unsupported resources, interop, launch options, and native utilities. | An unavailable operation must not silently execute on the CPU or dispatch a SYCL pointer to CUDA. |
| Add dedicated GPU tests and run existing CPU regressions. | Exercise backend behavior while checking that shared code changes preserve CPU behavior. |

### Supported first stage

- Linux Intel Level Zero GPUs; validated with oneAPI 2026.1.1 on Arc Pro B70.
- Regular contiguous/strided Warp arrays, explicit CPU transfers, kernel launches,
  scalar/vector operations, atomics, and generated reverse-mode kernels.
- Basic tile load/store, elementwise map, reduction and scan paths; tile-map/sum
  gradients and nested tile functions.
- BVH/mesh queries, SAH/median construction, BVH rebuild, ordinary GPU refit,
  mesh point/velocity replacement, and host-backed winding-number refit.
- Basic command-graph capture/replay, always built with SYCL. The compiler must
  provide the oneAPI graph extension; capture checks device support at runtime.

### Deliberate limitations

There is **no automatic CPU fallback**. Geometry construction and winding refit
are explicit synchronous host portions of supported operations. They cannot run
during graph capture.

This stage does not implement CUDA/SYCL direct copies, peer copies, external
streams/events, allocation during capture, APIC serialization, conditional
graphs, recorded `Launch` commands, deterministic lowering, custom module
preambles/includes, floating-point verification, indexed arrays, textures,
volumes, hash grids, sparse matrices, or DLPack interoperability. Advanced tile sort, FFT,
matrix multiplication and solver operations are not enabled.
Native array-wide scan/sort/reduction utilities are separate from tile operations
and are not implemented for SYCL.

Tile local-memory accounting is conservative and module-wide; exceeding device
local memory raises an error. Tiled kernels require complete work-groups and
uniform collective control flow. No subgroup width is requested or assumed.
Large non-contiguous fills and captured fills currently support elements of at
most 256 bytes.

The queue/capture interface is single-queue and requires externally serialized
access during capture. Geometry objects passed as integer IDs must outlive
their kernels and captured graphs, just as other opaque native resources do.
Geometry topology/setter changes must not be made while a graph using that
geometry is retained; recapture after those changes.

### Alternatives rejected

- CPU device impersonation: makes host access and allocator dispatch unsafe.
- Duplicating Python ctypes layouts for BVH/mesh: silently breaks with native ABI changes.
- Replacing every CUDA name with SYCL: conflates streams, graphs, warps, and memory models.
- Universal CPU fallback: cannot preserve asynchronous aliasing, gradient, and capture semantics.

## Testing Strategy

`warp.tests.test_sycl` exercises forward/backward kernels, contended atomics,
half-word preservation, signed integer overflow, strides, geometry mutation,
graph replay/lifetimes, explicit rejection paths, and tile execution/gradients
at work-group sizes 16, 32, and 64. These work-group tests do not claim to force
every hardware subgroup mode.

Existing array, codegen, launch, gradient, and module tests run on CPU.
Both `build_lib.py --sycl --no-cuda` and the CMake `warp-sycl` target are validated.
CUDA execution requires separate hardware validation; it is not inferred from
passing CPU or SYCL tests.

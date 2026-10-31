# Intel SYCL Backend

**Status**: Implemented (experimental, staged coverage)

For a source-linked, side-by-side comparison with CUDA and the rationale for
Intel-specific choices, see [Warp SYCL vs CUDA](./sycl-cuda-comparison.md).

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
| Keep explicit companion build definitions in CMake and Python, with a build-parity regression. | Follow the existing build layout while detecting drift in source lists, release assertions, debug information and linker flags. |
| Enumerate Intel Level Zero GPUs and own one in-order queue per device. | Avoid duplicate OpenCL devices and preserve operation ordering without pretending queues are CUDA streams. |
| Use device USM for arrays and checked native allocation/copy/fill/synchronization APIs. | Avoid accidental host dereferences; report SYCL exceptions through Warp's Python error path. |
| Reuse Warp's module builder, argument descriptors, and generated forward/backward bodies. | Add a launch backend rather than duplicating the compiler or exposing Python replicas of native layouts. |
| Submit `nd_range` kernels with value-captured arguments and bounds guards. | Preserve asynchronous argument lifetimes and support non-multiple ordinary launch sizes. |
| Give each module content/options identity and work-group/local-storage variant distinct SYCL kernel names. | Prevent DPC++ device-image collisions between release/debug modules and differently sized variants. |
| Give SYCL binaries and metadata their own compiler/runtime/header fingerprint. | Prevent CPU/CUDA cache collisions and stale ABI reuse. |
| Use typed atomic CAS updates, including packed half words and unsigned integer addition. | Preserve contended updates, neighboring half values, and signed wrapping semantics. |
| Use work-group barriers and local storage for basic tiles. | Reuse portable algorithms across actual GPU work-items, not a single CPU-like work-item. |
| Keep CUDA shuffle/mask paths CUDA-only and query subgroup width at runtime. | Intel subgroup width is not universally 32; work-group size is not subgroup size. |
| Track transitive tile use and submit partial logical blocks as smaller physical tail work-groups. | Every physical work-item reaches collective barriers without changing the requested logical block dimension or register ownership, including inside user functions. |
| Instantiate tile arenas and generated kernel calls only in the SYCL device image. | Debug host closures must not retain CPU tile-thread or barrier symbols. Common argument captures remain visible to both compilation passes. |
| Route native diagnostics through SYCL's device `printf`. | Preserve diagnostics rather than silently disabling them. |
| Build BVHs/meshes with existing native host SAH/median builders, then own shared-USM native structures. | Reuse current layouts and algorithms without copying private C++ layouts into Python. |
| Refit ordinary geometry on the GPU; stage winding-number refits explicitly on the host. | Support queries/refits while keeping the host portion visible and bounded. |
| Keep a separate host BVH descriptor for refit launch metadata. | Do not read the GPU's shared-USM descriptor on the host while queued work is using it, or add a device-wide wait to every refit. |
| Give empty SYCL trees a valid empty leaf/root. | Native queries dereference the root even when the host builder produced no nodes. |
| Always compile oneAPI command graphs with the SYCL backend and retain captured arrays/modules. | Match CUDA's backend-level inclusion, require a graph-capable compiler, and preserve buffer/code lifetimes without emulating CUDA graph handles. |
| Map `wp.Stream` to an in-order queue with profiling enabled, and `wp.Event` to a `sycl::event`; keep a per-device current stream. | Match CUDA's stream and event semantics (ordering, `wait_event`, `ScopedStream`, event timing) without assuming queues are CUDA streams. Each stream owns its staging buffers so streams cannot corrupt each other's arguments. |
| Enable upstream native recording on supported v2 devices and record timing events directly on the native immediate queue, with graph-owned host-USM samples. | Native queue commands become replayed graph commands. This bypasses the v2 command-buffer callback/getter deadlock without patching Intel libraries, forcing v1, or disabling locks. CPU-resident samples avoid shared-USM migration faults and conflicts with Python's `faulthandler`. |
| Preserve oneAPI adapter selection and expose `Device.supports_graph_capture`, `Device.is_graph_capture_event_supported` and `Device.is_conditional_graph_supported`. | Ordinary graph support is independent of captured timing and conditional control flow. Query each capability instead of assuming it from a backend name; retain ordinary capture on other devices/adapters and reject unsupported timing operations explicitly. |
| Pass kernel arguments larger than the 2048-byte parameter limit through USM staging memory. | Large Warp kernels (e.g. MuJoCo Warp collision and constraint kernels) exceed the hardware limit; graph-captured launches own a private copy. |
| Compile modules with `-ffp-model=precise` and correctly-rounded `sqrt` and division unless `fast_math` is set; define `NDEBUG` in release mode. | `icpx` defaults to a fast floating-point model (about 1e-4 relative error in `sin`, `cos` and `tan`, inexact `sqrt` and division) and, without `NDEBUG`, release kernels carry the debug-only tile bounds checks and device asserts. CPU and CUDA builds use IEEE semantics and `NDEBUG`. |
| Keep `inline` as the default function hint and accept explicit backend overrides; MJWarp uses `override={"sycl": False}` for polygon clipping. | Avoid the observed IGC 2.40.13 CCD failure without changing collision math, CPU/CUDA inlining or process-wide compiler settings. The previous global retry-disabled default regressed 23 box-box cases and has been removed. |
| Compute the shared (local) memory requirement of a module from the deepest kernel plus the largest transient tile scratch, and report device compute units and memory. | Summing every function overestimated local memory and rejected valid modules. `sm_count`, `free_memory` and `total_memory` now report real values. |
| Implement `array_scan`, `radix_sort_pairs` and `segmented_sort_pairs` as native SYCL kernels on the current stream, with per-stream scratch that a graph takes ownership of during capture. | Public Warp utilities that MJWarp calls every step. Library implementations (oneDPL) block and allocate internally, which breaks asynchronous ordering and graph capture. |
| Implement `tile_sort` with work-group bitonic compare-exchanges over the existing local-memory arena and barriers. | Preserve CUDA's in-place key/value contract without assuming 32-lane subgroups, adding host work, or forking MJWarp's tiled SAP and contact-sensor kernels. |
| Track capture-time device-USM allocations with shared graph/user ownership, and defer frees and module unloading until capture ends. | Match CUDA's allocation lifetime contract without assuming that SYCL graph-owned allocation nodes can produce arrays usable outside their graph. |
| Implement RGBA32F 2D textures with DPC++ experimental bindless images and retain texture owners across arrays, copies and captured arguments. | Preserve Warp's opaque texture-handle ABI and use hardware filtering without a second renderer or a software sampling fallback. |
| Reject unsupported resources, interop, launch options, and native utilities. | An unavailable operation must not silently execute on the CPU or dispatch a SYCL pointer to CUDA. |
| Add dedicated GPU tests and run existing CPU regressions. | Exercise backend behavior while checking that shared code changes preserve CPU behavior. |

### Supported first stage

- Linux Intel Level Zero GPUs; validated with oneAPI 2026.1.1 on Arc Pro B70.
- Streams and events (`wp.Stream`, `wp.Event`, `wp.ScopedStream`, event timing), with per-stream ordering.
- Imported in-order SYCL queues and Torch XPU zero-copy arrays, with physical
  device/context/ABI validation and retained consumer queues.
- oneAPI DLPack imports and explicit-consumer-queue exports, including empty,
  strided, vector and matrix arrays. Streamless SYCL exports are rejected.
- Regular contiguous/strided Warp arrays, explicit CPU transfers, kernel launches,
  scalar/vector operations, atomics, and generated reverse-mode kernels.
- Basic tile load/store, elementwise map, reduction, scan and sort paths; tile-map/sum
  gradients and nested tile functions, including partial logical blocks.
- BVH/mesh queries, SAH/median construction, BVH rebuild, ordinary GPU refit,
  mesh point/velocity replacement, and host-backed winding-number refit.
- Array-wide `wp.utils.array_scan`, `radix_sort_pairs` and `segmented_sort_pairs`,
  asynchronous on the current stream and recordable into graphs.
- Single-level RGBA32F `wp.Texture2D` resources, hardware nearest/linear sampling,
  texture arrays, and captured sampling through DPC++ experimental bindless images.
- Basic command-graph capture/replay, including array allocations during capture,
  always built with SYCL. The compiler must provide the oneAPI graph extension;
  capture checks device support at runtime.
- Nested event timing in native-recording graphs with Level Zero v2. Timing events
  are automatically external, matching CUDA's public recording API.

### Current capability contract

This matrix is the authoritative current support summary. Detailed sections
explain the implementations; the qualification logs below record historical
checkouts and must not be interpreted as current test totals.

| Surface | Current contract / capability | Regression coverage |
| --- | --- | --- |
| Device and kernel execution | Linux Intel Level Zero GPUs, explicit `sycl:N` identity, device USM, shared codegen and autodiff. Query device limits, FP64 and atomic64 support rather than assuming them. | [SYCL tests](../warp/tests/test_sycl.py) |
| Work-group collectives | Uniform collective control flow; supported tile load/store, map, reduction and scan paths include partial logical groups. `tile_sort` still requires complete logical groups. No fixed 32-lane subgroup assumption. | Partial-tile, scan, sort and gradient cases in [SYCL tests](../warp/tests/test_sycl.py) |
| External queues / Torch | In-order queues on the same physical device and platform-default context; compatible oneAPI/C++ ABI. Imported streams own queue copies; no implicit CUDA default-stream ordering. | [XPU interop](../warp/tests/interop/test_torch_xpu.py) |
| oneAPI DLPack | Zero-copy imports; exports require a positive consumer queue pointer through `__dlpack__` or `to_dlpack(..., stream=...)`. The export retains canonical consumer queues through retirement. `None` and `-1` exports are rejected, including legacy capsule exports without an explicit queue. | Queue aliases, explicit rejection, empty/strided/compound types and asynchronous last-owner teardown in [XPU interop](../warp/tests/interop/test_torch_xpu.py) |
| Ordinary event timing | Profiling tags support imported queues when `ext_oneapi_queue_profiling_tag` is available; otherwise the queue must have `enable_profiling`. Unsupported timing fails at recording, not elapsed-time retrieval. | Owned/imported/cross-queue timing in [XPU interop](../warp/tests/interop/test_torch_xpu.py) and [SYCL events](../warp/tests/test_sycl.py) |
| Ordinary graphs | Check `Device.supports_graph_capture`; retain one recording queue and captured resources. | Graph allocation, module, stream and resource lifetimes in [SYCL tests](../warp/tests/test_sycl.py) |
| Replay-visible captured events | Check `Device.is_graph_capture_event_supported` independently; supported native-recording v2 immediate queues use graph-owned host-USM timestamp samples. Host waits complete the containing graph. | Captured/mixed timing, nested replay and unsupported-adapter cases in [SYCL tests](../warp/tests/test_sycl.py) |
| Conditional / multistream graphs | Not implemented: `Device.is_conditional_graph_supported` is false. Foreign event import, cross-graph external waits and external PyTorch graph capture are separate unsupported contracts. | Explicit rejection cases in [SYCL tests](../warp/tests/test_sycl.py) |
| Geometry | Explicit host SAH/median construction; ordinary GPU refit; winding-number refit has a synchronous host portion. No topology mutation while retained by a graph. | [Grouped BVHs](../warp/tests/geometry/test_grouped_bvh.py) and [SYCL geometry](../warp/tests/test_sycl.py) |
| Textures | Single-level RGBA32F 2D hardware sampling; query addressing capabilities. Other dimensions/formats/surfaces are rejected. | Texture copies, arrays, nested structs, graphs and lifetime cases in [SYCL tests](../warp/tests/test_sycl.py) |
| Array utilities / matrix tiles | Native asynchronous scan and sorts; portable matrix tile arithmetic, not XMX/oneMKL. Array-wide reductions, run-length encoding and tile FFT are unsupported. | Shared utility tests registered in [SYCL tests](../warp/tests/test_sycl.py) |
| MuJoCo benchmarking | Capture selection accounts for event tracing before recording. Eager warm-up restores nested data/render array contents from host snapshots outside rollout measurement/tracing; each eager step owns a fresh tracer. Captured replay retains its original event recordings. | [CLI regressions](../../mujoco_warp/mujoco_warp/_src/cli_test.py): frozen controls, heterogeneous worlds, Euler/RK4 and unchanged MuJoCo tolerances; runnable with CPU/CUDA/SYCL device selection. |
| Companion builds | Explicit definitions in [CMake](../CMakeLists.txt) and [Python](../warp/_src/build_dll.py): Release `-O2 -DNDEBUG`, Debug `-O0 -g`, identical source/link inputs. | [Build-contract test](../tools/ci/test_sycl_build.py), which builds the CPU core and companion through the root CMake project in isolated source copies for both modes, then compares actual companion arguments with the Python builder. |

### Compiler workaround lifecycle

`collision_gjk._polygon_clip` retains its SYCL-only noinline override. The
observed affected stack is Arc Pro B70, DPC++ 2026.1.1 and IGC 2.40.13; this is
an observed configuration, not an asserted range of affected compiler releases.
CPU/CUDA inlining and process-wide IGC settings remain unchanged.

The smallest checked-in end-to-end reproduction is
[`test_capsule_mesh_multiccd_shallow`](../../mujoco_warp/mujoco_warp/_src/collision_driver_test.py):
one capsule above an eight-vertex cube mesh. Keep its full caller context:
isolated clipping kernels did not reproduce the retry-IR corruption. There is
not yet a minimized standalone compiler testcase or a verified upstream fix;
the historical investigation below distinguishes observed IR from hypotheses.

Before removing the override, change only that hint in an isolated checkout,
use a fresh Warp kernel-cache directory, leave `IGC_DisableRecompilation` unset,
and disable persistent driver caches. From the MuJoCo Warp checkout, run:

```bash
SYCL_CACHE_PERSISTENT=0 NEO_CACHE_PERSISTENT=0 uv run --no-sync python -m pytest \
  mujoco_warp/_src/collision_driver_test.py --device sycl:0 \
  --kernel_cache_dir=/path/to/fresh/isolated/warp-cache \
  -k 'capsule_mesh_multiccd or cylinder_multiccd or box_box'
```

Run the same selection with the override restored, and with `--cpu` and an
available CUDA device. Record the compiler/driver versions and exact failing
case IDs. Removal requires all original contact counts, positions and distances
to pass unchanged in both eager and captured execution, no box-box regression,
and a reduced compiler reproducer or upstream fix identifying the affected
optimization. Do not mask failures with skips, looser tolerances, global retry
settings or alternate collision math. Upgrade the observed version record only
after testing the replacement compiler.

### Bindless textures

`texture_sycl.cpp` owns DPC++ `image_mem` allocations and `sampled_image_handle`
objects. The existing texture kernel descriptor carries the raw bindless handle;
device code in `texture.h` samples it directly with `sample_image`. Runtime-selected
texture-array elements do not require accessors or a different kernel ABI. This is
hardware image sampling, not interpolation over an ordinary USM array.

The supported storage is 2D, float32, four channels and one mip level. Sampling can
return a float, vec2 or vec4. Nearest and linear filtering support clamp, border,
wrap and mirror modes. Wrap/mirror require normalized coordinates; unnormalized
coordinates support clamp/border. Mixed U/V modes require the device's
`ext_oneapi_unique_addressing_per_dim` aspect. The validated B70 configuration does
not advertise that aspect, so mixed modes are rejected rather than silently using
one mode for both axes. Other dimensions, storage formats, mipmaps and surface
access also raise explicit errors.

Create and copy textures outside graph capture. Host/device/image transfers use
the current in-order queue, with explicit event dependencies for cross-stream use;
host transfers wait for completion. Captured sampling can observe texture updates
submitted before replay. Direct arguments, nested structs, and texture/struct
arrays created from Python objects retain the texture owners. Array copies and
views preserve those references, and capture snapshots the nested resources even
if a Python struct field is later reassigned. Raw handles or NumPy structured data
cannot recover Python owners automatically. Native destruction is deferred until
queued users complete, including destruction during an unrelated capture.

Sampling regressions use analytical nearest-wrap seam expectations: the existing
CPU sampler clamps a rounded index at that seam instead of wrapping it. CPU and
CUDA sampling implementations are unchanged; CPU output is not treated as an
oracle for that boundary case. No software or CPU fallback is provided when the
bindless extension or a required image capability is unavailable.

### Captured event semantics

Each event recording owns an independent sample. Graphs retain their samples even
if Python events are destroyed or re-recorded. A replay updates surviving events
to that graph's latest recording; event samples remain readable after graph
destruction. Parent graphs retain nested graph markers in execution order.

Supported v2 devices use the upstream `property::graph::enable_native_recording`
property, already shipped in oneAPI 2026.1.1. The native immediate command-list
handle is obtained before recording. `zeCommandListAppendWriteGlobalTimestamp`
commands, bounded by native barriers, are recorded directly on that queue rather
than through a command-buffer native callback. Adapter selection is unchanged.

Native graph launches are not themselves captured correctly by every driver.
When a child graph is launched during capture, Warp finalizes the preceding
native segment, retains the child's segments, and records the following segment
separately. Replay submits these GPU graphs in order on the selected stream,
without host synchronization or executing child work during capture. Ordinary
graphs remain a single submission; nested graphs add submissions at their
boundaries. Child graph resources and event samples remain retained by the parent.

Captured timing uses the device timer
frequency and valid-bit mask, including single wraparound. It is independent of
whole-graph profiling. Timing-enabled ordinary events still use SYCL queue
profiling; both forms use the device's global clock.

Host synchronization and stream waits on captured external events conservatively
wait for the containing graph's completion, not just the marker. Stream waits stay
asynchronous. Replays sharing marker storage are ordered by execution-event
dependencies, including across current streams and parent/child graphs.
Querying or synchronizing graph-internal events requires recording with
`external=True`; timing events set this automatically. Timing queries before the
first replay or before completion raise errors rather than reading placeholders.

Same-graph event dependencies are supported. Captured waits on events from another
graph or from uncaptured work remain unsupported and raise explicitly; enabling
traced capture does not enable multistream capture or conditional graphs.

### Graph allocation ownership

The native allocator tracks capture-time array allocations as CUDA tracks
`GraphAllocInfo`: storage remains valid until both the originating graph and the
array's user reference are released. Arrays can escape capture, survive graph
destruction, and be retained by other graphs. Unused arrays released while recording
are also tracked; kernel-argument retention alone cannot cover them.

Allocation uses ordinary device USM while recording. Zero/fill/copy operations
remain graph commands and execute on every replay. The oneAPI
[graph-owned allocation contract](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_oneapi_graph.asciidoc#allocation-lifetime)
does not permit using those pointers outside their owning graph, so its asynchronous
allocation API is not a replacement for Warp's escaping arrays. No custom pool or
CUDA memory-pool flag is needed.

The graph owns its recording queue and tracks replay completion. Physical frees,
graph destruction, and module unloading are deferred when another capture is active;
reclamation preserves cross-stream safety without synchronizing a recording queue.
Each live graph retains the sum of its captured allocations, not just peak-live
scratch. Replays do not allocate more storage. Escaping arrays can retain storage
after the graph is dropped.

### Deliberate limitations

There is **no automatic CPU fallback**. Geometry construction and winding refit
are explicit synchronous host portions of supported operations. They cannot run
during graph capture.

This stage does not implement CUDA/SYCL direct copies, peer copies, foreign
event import, capturing work from several streams, external event waits
between graphs, APIC serialization, conditional graphs, recorded `Launch` commands,
deterministic lowering, custom module preambles/includes, floating-point
verification, indexed arrays, volumes, hash grids, or sparse matrices.
Queue imports and explicit-queue DLPack are supported under the contract above.
Tile FFT is not enabled. Tile matrix
multiplication, Cholesky and triangular solves use Warp's portable scalar
implementations; no vendor matrix engine (XMX/oneMKL) is used yet.
The native array-wide reductions (`array_sum`, `array_inner`) and `runlength_encode`
are not implemented for SYCL. The array-wide scan and sorts produce the same results
as the CPU implementations (integer and sorted outputs are bit-identical; float sorts
order `-0.0` before `+0.0`). Float scans sum in a different order, as on CUDA. Invalid
segment bounds are clamped rather than reported, so callers must validate them as on CUDA.

Tile local-memory accounting is conservative and module-wide; exceeding device
local memory raises an error. Tiled kernels require uniform collective control
flow; supported partial logical groups use a smaller physical tail group.
`tile_sort` still requires complete logical groups. No subgroup width is requested or assumed.
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

## Historical MuJoCo Warp enablement

The following investigation logs describe their stated revisions and runtime
configurations, not a current release-support promise. Use the
[current capability contract](#current-capability-contract) for supported APIs.

[MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp) (MJWarp, 3.16.0) is the
first large external Warp codebase run on this backend. It is written against the CUDA
Warp API and defines about 370 kernels, so it exercises the language and runtime far
beyond the unit tests. Everything below was measured on an Arc Pro B70 with oneAPI
2026.1.1 and IGC 2.40.13, with MJWarp's existing tests as the oracle
(`pytest --device sycl:0`; GPU results are compared with MuJoCo C and with MJWarp on CPU).

### Why Warp had to change first

MJWarp is not ported by rewriting its kernels. The first attempt at this port (the
reference SYCL patch for MJWarp 3.8.1) replaced the Newton solver's Cholesky and
JTDAJ kernels with hand-written single-work-item kernels in a separate file and removed
the graph-conditional loop. That forks the algorithm: every upstream solver change
would need a second implementation, and it does not help any other Warp user. Instead,
each failure was classified by one rule:

- **It belongs in Warp** when MJWarp uses a Warp language or runtime feature that CPU and
  CUDA already provide, and the SYCL backend rejects it or computes it differently.
  Fixing it once makes MJWarp (and any other Warp program) run unchanged.
- **It belongs in MJWarp** when it is device policy: which launch geometry to use, whether to
  capture a graph, whether a conditional graph node exists. Warp cannot decide that.

| Blocker found | Where MJWarp hits it | Symptom on the B70 | Why the fix is in Warp |
| --- | --- | --- | --- |
| `wp.printf` emitted a variadic call | Overflow and error messages in about 9 modules (smooth, sensor, collision, solver, forward, island) | `SYCL kernel cannot call a variadic function`; one occurrence fails the whole module | A language builtin that CPU and CUDA support. Device `printf` needs a literal format string, which only the code generator can guarantee. |
| Kernel arguments over the 2048-byte parameter limit | `linesearch_iterative` (2112 B), `primitive_narrowphase` (3168 B); about 16 of 370 kernels exceed the limit by a static estimate | `Total size of kernel arguments exceeds limit` at program build | The limit is a hardware property that CUDA's much larger parameter space does not impose. Kernels carry whole `Model`/`Data` structs by design, so MJWarp cannot reasonably shrink them. The launcher must spill them. |
| `tile_matmul`, `tile_cholesky_solve` rejected by the code generator | `block_cholesky.py`, `smooth.py`, `solver.py` (blocked Cholesky, Newton solve) | `is not yet supported on SYCL` | Warp already has portable scalar implementations; the guard was conservative. With it lifted they matched NumPy to 1e-6 on the B70. |
| Local-memory estimate summed every function | Blocked Cholesky at nv = 46 | Module needs 153,744 B, device has 131,072 B | The estimate is a Warp code-generation property. The true need is the deepest kernel plus the largest transient scratch. |
| Release kernels compiled without `NDEBUG` | Every tile kernel | Debug tile bounds checks and device asserts remained in release code; blocked Cholesky returned NaN for nv > 32 | CPU and CUDA release builds define `NDEBUG`. Only the module builder knows the build mode. |
| `icpx` default floating-point model | Kinematics and dynamics everywhere | `sin`/`cos`/`tan` relative error about 1e-4 (roughly 1000 float32 epsilons), inexact `sqrt` and division; `qfrc_bias` off by 8e-3 against MuJoCo | CPU and CUDA kernels are IEEE-754 unless `fast_math` is set. The module builder owns the compiler flags. |
| Device queries returned stubs (`sm_count` 0, block size (1, 1), memory 0) | Launch sizing in `collision_convex.py`, `constraint.py`, `solver.py`; memory report in `testspeed.py` | Zero-sized or serial launch widths | Properties of the device that only the runtime can query. |
| No `wp.Stream` or `wp.Event` | `cli.unroll` (`ScopedStream`), `event_scope` (`wp.Event`) | `Device sycl:0 is not a CUDA device` | Core Warp API used by MJWarp's CLI and profiling. Maps directly to in-order queues and `sycl::event`. |
| `max_blocks` rejected | NXN broadphase launch in `collision_driver.py` | `NotImplementedError` | A launch hint that other backends ignore or honour; it must not be an error on SYCL. |

### What stays in MuJoCo Warp

These are device policy and are small, upstream-style changes:

- `graph_capture_supported()` and `graph_conditional_supported()` in `warp_util.py` are
  independent and consume Warp's per-device capabilities. Ordinary capture first requires
  `device.supports_graph_capture`; GPU event tracing additionally requires
  `device.is_graph_capture_event_supported`. CPU retains its existing APIC capture policy.
  `device.is_conditional_graph_supported` is false for SYCL, so `solver._solve`, the flex
  FPS loop and the incremental broadphase take the existing fixed-iteration branch.
- Branches written as `is_cuda` for "has a GPU" now read "is not CPU" (contact Jacobian wave
  size, sparse solve block size, JTCJ launch width), otherwise SYCL silently takes the slow
  CPU paths.
- `cli.unroll` captures ordinary SYCL steps and allows event-traced capture when
  `device.is_graph_capture_event_supported` is true. Unsupported adapters keep the
  existing uncaptured tracing path. It scopes a stream only on devices that have one.
- `--device` in `conftest.py`, and a README note.

No MJWarp kernel was rewritten. The existing CPU suites still pass unchanged
(705 tests across solver, smooth, constraint, forward, collision driver).

### Current status

The bullets below describe the initial rigid-body baseline, not whole-port
sign-off. See [latest qualification](#latest-qualification) for the current
3.17 source checks and remaining failures.

- Unmodified `mjw.forward` and `mjw.step` match MuJoCo for Newton and CG, pyramidal and
  elliptic cones, dense and sparse Jacobians, and Euler, implicit-fast and RK4 integrators.
  The nv = 8 models match within the test tolerance, and nv = 46 (blocked Cholesky) agrees with
  MuJoCo to 1.3e-5 in `qacc` (CPU: 3.4e-5).
- `mjwarp-testspeed` on `humanoid.xml`, 1024 worlds, eager steps: about 72,000 steps/s,
  1024/1024 worlds converged. This is a first number with scalar tile math and no early
  solver exit. There is no CUDA machine to compare against.
- The GPU baseline before array-wide sort and capture-allocation enablement was
  1515 passed, 42 skipped, 582 failed. These are historical failure groups, not results
  after the changes below. Attribution is by final error message, so the rows are
  approximate and do not sum exactly to 582:

| Approximate failures | Missing capability | Next step |
| --- | --- | --- |
| 278 | Array-wide `radix_sort_pairs`, `segmented_sort_pairs` | 1 |
| 146 | Allocation during graph capture | 2 |
| 40 | `tile_sort` | 3 |
| 50 | `texture_sample`, textures (render megakernel and skybox) | 4 |
| 40 | BVH constructor: MJWarp requests `cubql` (CUDA-oriented) for meshes and height fields; SYCL supports only SAH and median | 4 (MJWarp-side) |
| 4 | Multi-contact CCD gives wrong results at default optimization | 5 |
| about 8 | Mixed (bvh, sensor, flex); not yet triaged | 6 |

### Next steps

They are ordered by how many MJWarp tests each unblocks and by risk. Correctness comes
before performance.

1. **Array-wide scan and sort** (`wp.utils.array_scan`, `radix_sort_pairs`,
   `segmented_sort_pairs`). *Implemented.*
   - *Needed by:* the sweep-and-prune broadphase (`collision_driver.py:755` and `:767`) and flex
     collision (`collision_flex.py:3114`, `:3148`, `:3407`, `:3428`). They are called every
     step on device arrays, so a host fallback would break the no-CPU-fallback rule and graph
     capture.
   - *Why Warp:* these are public Warp utilities with CUDA and CPU implementations.
   - *Approach taken:* native kernels in `scan_sycl.cpp` and `sort_sycl.cpp`, not oneDPL. oneDPL's
     `exclusive_scan` and `sort_by_key` wait on the host and allocate their own temporaries,
     so they cannot be recorded into a graph or stay asynchronous, and they do not take a
     bit range. Every operation is a sequence of kernels and copies on the current in-order
     stream:
     - *Scan:* per-tile reduction, one work-group scan of the tile sums per vector component,
       then a per-tile scan. Strided, vector and in-place arrays are supported.
     - *Radix sort:* stable least-significant-digit passes over 4-bit digits, using the
       caller's `[count, 2 * count)` region as the ping-pong buffer, like CPU and CUDA. A
       tile of 1024 keys is ranked with four work-group scans over 16-bit bucket counters
       packed into 64-bit words; no subgroup width is assumed. Keys are mapped to ordered
       unsigned bits exactly as the CPU sort does, so `begin_bit`/`end_bit` behave identically.
     - *Segmented sort:* one stable radix sort over (segment index, key) composite keys and an
       index payload, limited to `32 + bits(num_segments)` bits. Elements outside every
       segment get a sentinel index and are not moved. Segments may be unordered or have gaps.
     - *Scratch:* `wp_sycl_scratch()` returns a per-stream device buffer that grows on demand
       (waiting only for that stream). During capture it allocates a buffer owned by the graph,
       like staged kernel arguments, so replays never share scratch with eager work. This is
       native algorithm scratch, separate from the public array allocations in step 2.
   - *Done when:* Warp's scan and sort tests pass on `sycl:0`, and MJWarp's broadphase and flex
     tests pass.
2. **Allocation during graph capture.** *Implemented and validated on Arc Pro B70.*
   - *Needed by:* MJWarp allocates scratch inside the step (`create_collision_context` in
     `collision_core.py:525`, called from `collision_driver.py:1095`, and many others). CUDA permits
     this because the stream-ordered allocator is capture-aware. These allocations previously
     prevented SYCL steps from being captured and required eager host dispatch for every kernel.
   - *Why Warp:* the allocator is the only layer that can know a pointer's lifetime relative to a
     graph. Moving scratch into `Data` instead would touch many call sites in an upstream
     library and still leave other Warp programs unable to allocate during capture.
   - *Approach taken:* native shared graph/user ownership of ordinary device-USM allocations.
     Freeing an array during capture only releases its user reference. The graph keeps its
     storage valid for replay; a surviving array keeps it valid after graph destruction.
     Native utility arguments, modules, and child graphs are retained as well. Cleanup covers
     failed captures, never-launched graphs, destroyed recording streams, and graph destruction
     during another capture.
   - *Why not async allocation nodes:* oneAPI exposes the allocation API, but the graph-owned
     pointer contract forbids use outside the owning graph. That does not preserve Warp's
     escaping-array semantics. A graph-local scratch optimization would be separate work.
   - *Risk:* storage grows with the sum of scratch recorded into each retained graph, not with
     replay count. Escaping arrays can keep storage after their graph is dropped.
   - *Acceptance:* `mjw.step` captures and replays, and the historical capture-related failures
     are re-run. Report remaining unrelated blockers separately rather than assuming the
     historical count of 146 is unchanged.
   - *Validation:* all 62 dedicated SYCL tests pass. The selected MJWarp forward, collision,
     solver, and flex modules have no allocation-during-capture rejections; their remaining
     failure groups are recorded below.
3. **`tile_sort`.** *Implemented and validated in release mode on Arc Pro B70.*
   - *Needed by:* the contact-sort sensor (`sensor.py:2425`) and tiled SAP broadphase
     (`collision_driver.py:655`). Both keep their existing kernels; no MJWarp production change
     or CPU fallback is needed.
   - *Contract:* the same five key types as CUDA (`float32`, `int32`, `uint32`, `int64`,
     `uint64`), generic value payloads, and matching-shape tiles sorted ascending in place.
     Shared storage promotion and the non-differentiable public API are unchanged. Stable
     duplicate ordering and a particular NaN order are not promised.
   - *Approach taken:* a SYCL device branch in `tile_radix_sort.h`, using `WP_TILE_THREAD_IDX`,
     `WP_TILE_BLOCK_DIM`, `WP_TILE_SYNC`, and the existing LIFO local-memory arena. Work-items
     own disjoint compare-exchange pairs, use uniform chunk loop bounds, and synchronize
     between bitonic stages. No subgroup size is assumed.
   - *Storage:* power-of-two lengths sort directly in their shared tiles. Other lengths use
     next-power-of-two key/value scratch and validity flags so padding cannot replace real
     maximum keys or floating-point special values. Scratch is released after the final
     collective barrier. Codegen accounts for both element types, padding, flags, and each
     allocation's alignment, combined by maximum with other transient tile workspace.
   - *CUDA difference:* CUDA uses bitonic sorting up to 2048 elements and for 64-bit keys,
     then radix sorting for larger 32-bit tiles. SYCL initially uses bitonic sorting for all
     supported lengths within the device's local-memory budget. The threshold is a tuning
     choice, not a public size limit. CPU and CUDA implementations are unchanged.
4. **Textures and rendering.** *Implemented and qualified for the bindless subset above.*
   MJWarp selects `"sah"` for non-CUDA mesh/heightfield construction and keeps its
   existing renderer. A separate host BVH launch descriptor fixes a shared-USM host
   read during asynchronous refit without synchronizing every refit. Existing
   rendering regressions and changing-input forward/refit/render capture pass
   through 4096 worlds. This does not resolve the physics issues in steps 5 and 6.
5. **Optimized multi-contact CCD correctness.** *Mitigated by a targeted SYCL
   noinline hint; the permanent compiler fix remains open.* Without the hint,
   normal IGC retry fails seven cases:
   four capsule-mesh and three cylinder contact tests. The capsule-mesh path
   (`test_capsule_mesh_multiccd_*`) gives wrong results at default optimization. Earlier probes passed with
   `icpx -O0`, with `-fno-inline-functions`, with the device compiler's optimizer disabled
   (`-cl-opt-disable`), and when the same SPIR-V runs on the CPU OpenCL device, which points at the
   GPU compiler (IGC 2.40.13), but does not establish one cause for all seven cases.
   A follow-up scratch-buffer comparison of the shallow capsule-mesh case matches CPU
   through EPA vertices, candidate normals, and recovered faces. Device checkpoints
   then observe only copy index 0 during polygon initialization even though its side
   count is 2; CPU copies indices 0 and 1. Optimized SPIR-V contains the expected
   two-iteration loop. IGC's first optimized IR contains two correct, distinct vertex
   copies, but its retry/recompile IR replaces them with a two-iteration loop using
   vertex-0 addresses and a constant-zero index marker. Device execution matches that
   incorrect retry IR, leaving vertex 1 unwritten. Standalone clipping still passes
   with runtime bounds, 2D array views, a divergent outer grid-stride loop, and an
   out-of-line function, so the failing compiler context and responsible pass have
   not yet been minimized. This establishes a compiler-stage error on the traced
   path, not a common root cause for every cylinder case or proof that
   Warp-generated C++ is free of undefined behavior. The initial process-wide
   `IGC_DisableRecompilation=1` default passed all 11 selected CCD cases but
   introduced 23 box-box failures. Fresh-compilation controls confirm the
   tradeoff, so Warp no longer sets that variable. Instead,
   [MJWarp polygon clipping](../../mujoco_warp/mujoco_warp/_src/collision_gjk.py)
   uses `@wp.func(override={"sycl": False})`. All 66 selected CCD and box-box
   cases pass with normal IGC retry and persistent caches disabled. CPU and CUDA
   retain compiler-chosen inlining. Frontend `-O3` and the large-register-file
   backend option did not resolve the original CCD failures.
   The hint changes an optimization boundary rather than fixing the compiler;
   its performance impact is unmeasured. A TODO beside the helper tracks
   minimization and removal after a verified fix. Keep the existing tests enabled and do not relax
   numerical checks or rewrite collision algorithms. Debug blocked Cholesky now runs
   successfully after the separate module-loading fixes.
6. **Large-world line-search nondeterminism.** One of three identical 1024-world
   benchmark runs aborted on a line-search overflow in a single world. This remains
   separate from broadphase record ordering: CUDA `atomicAdd` and SYCL atomic CAS both
   reserve unique append positions without ordering world IDs. The two-world broadphase
   assertion now compares exact `(world, geom1, geom2)` record multisets, retaining
   pair orientation and duplicate counts. No production sort or atomic change is needed.
   Investigate the line-search overflow before adding more concurrency.
7. **Performance, after the items above.** Nothing so far is tuned for Intel hardware, and the
   estimates below have not been measured:
   - *Early solver exit.* CUDA stops at convergence with a conditional graph node; SYCL runs all
     iterations. Options: check convergence on the host every few iterations, or an in-kernel
     loop.
   - *Matrix engines.* Use XMX through oneMKL or `joint_matrix` for `tile_matmul` and the
     Cholesky tiles instead of the scalar fallback.
   - *Kernel arguments.* Staging costs a copy per launch for the roughly 16 large kernels; pack
     `Model` and `Data` arrays into one device struct.
   - *Launch geometry.* Tune `block_dim` and sub-group size (16 or 32) per kernel; MJWarp's
     defaults assume 32-wide warps.
   - Measure per-kernel time with unitrace against the device's memory and compute limits rather
     than against CUDA numbers.
   - *Array-wide scan and sort.* The first implementation favors simplicity: 4-bit radix digits (eight
     passes for 32-bit keys), uncoalesced scatters, and segmented sorts that sort every
     element. Wider digits, a one-sweep scan, and per-segment small-sort kernels are options.

## Historical testing strategy and evidence

`warp.tests.test_sycl` exercises forward/backward kernels, contended atomics,
half-word preservation, signed integer overflow, strides, geometry mutation,
graph replay/lifetimes, explicit rejection paths, and tile execution/gradients
at work-group sizes 16, 32, and 64. These work-group tests do not claim to force
every hardware subgroup mode.

Regression tests for the MJWarp-driven fixes cover `printf`, kernel arguments over the
parameter limit (eager, taped, graph-captured and across streams), tile matmul, Cholesky and
triangular solves, a blocked Cholesky over constant-size tiles, `NDEBUG` in release and not in
debug, and math accuracy and rounding. Streams and events have ordering, scoping and timing tests.
Scan and sort run the device-generic `test_utils` cases on SYCL (including graph capture,
scratch growth mid-capture, and stream destruction), compare bit-for-bit against the CPU
implementations for every key type and for unordered, gapped segments, and check per-stream
scratch isolation and in-place scans.
Capture-allocation tests cover constructors, strided views, gradients, both user/graph
destruction orders, unused temporaries, capture failures, cross-stream teardown, child
graphs, module unloading, and native utility argument retention. Native allocation counts
and bytes must return to baseline after final release; GPU-wide free-memory deltas are not
the oracle. CUDA's existing graph-allocation and module-lifetime tests are reused on SYCL.
MJWarp compares captured collision steps against eager steps and MuJoCo C for one and two
heterogeneous worlds, and checks that ordinary and conditional capture policies stay separate.

Action-2 validation uses Intel Arc Pro B70, oneAPI 2026.1.1, and the compatible
MuJoCo `3.16.1.dev1003740167` package:

| Suite | Passed | Skipped | Failed |
|---|---:|---:|---:|
| Warp dedicated SYCL tests | 62 | 0 | 0 |
| Warp CPU graph/allocator and scan/sort regressions | 50 | 53 | 0 |
| MJWarp CPU capture policy and one/two-world parity | 3 | 0 | 0 |
| MJWarp SYCL existing and new graph-capture tests | 16 | 0 | 0 |
| MJWarp SYCL forward, collision driver, solver, and flex modules | 761 | 22 | 23 |

The 23 integration failures at the action-2 baseline were 14 unsupported `tile_sort` cases, two tile
launches with incomplete work-groups, and seven eager collision/contact numerical
mismatches. None is an allocation-during-capture rejection. These are targeted results,
not a replacement whole-suite total for the historical baseline above. Newer MuJoCo 3.17
nightlies have a different equality-data layout from this MJWarp checkout; that mismatch
also reproduces on CPU before capture, so the results above use the compatible 3.16 build.

Tile-sort regressions reuse the existing device-generic sort tests, including all five key
types, and cover lengths around work-group boundaries and 2048/2049, duplicate pair
permutations, scalar/vector/matrix/struct payloads, and default-storage promotion.
SYCL-specific cases preserve maximum keys, infinities, signed zeros and NaN payloads,
exercise nested repeated sorts at work-group sizes 16, 32, 48, 64, 96 and 128, check exact
local-memory sizing and explicit resource-limit errors, and replay independent graphs
on separate streams. These work-group sizes do not force particular hardware subgroup modes.
MJWarp captures tiled SAP and contact-sensor sorting for one and two heterogeneous worlds,
changes inputs between replays, and compares with eager execution and MuJoCo C.

Action-3 release-mode validation uses the same B70, compiler, and MuJoCo package:

| Suite | Passed | Skipped | Failed |
|---|---:|---:|---:|
| Warp dedicated SYCL tests, including eight tile-sort groups | 70 | 0 | 0 |
| Warp CPU tile-sort and block-codegen regressions | 12 | 0 | 0 |
| MJWarp CPU new tiled SAP/contact-sensor capture tests | 4 | 0 | 0 |
| MJWarp SYCL new tiled SAP/contact-sensor capture tests | 4 | 0 | 0 |
| MJWarp SYCL previously rejected tile-sort cases | 14 | 0 | 0 |
| MJWarp SYCL forward, collision driver, solver, flex, broadphase, and sensor modules | 867 | 22 | 11 |

Focused rows are subsets of broader runs, not additive totals. All 14 tile-sort
rejections from the action-2 baseline now pass. At that baseline, the six-module
parallel run had seven existing multi-contact CCD numerical failures, two incomplete
tile work-group launches, and two NXN broadphase world-order assertions. The NXN
failures also reproduce in isolation; their scheduling-dependent ordering is not
a tile-sort failure or evidence by itself of a data race.

The follow-up launch fix submits complete groups and a smaller physical tail on the
same queue, preserving launch bounds, the requested `wp.block_dim()`, and register
layout. Reduction, scan, and arg-reduction flags explicitly exclude absent logical
owners; sort work is distributed among actual participants. This does not support
arbitrary early returns inside collectives: all physical participants must follow
uniform collective control flow. The original two solver tests are unchanged.

Debug tile modules previously retained unresolved CPU tile-thread/barrier symbols
in their host closures. Device-only arena construction and invocation remove those
imports. A second defect gave release and debug modules identical SYCL kernel tags,
crashing the SYCL runtime during image lookup when modes were mixed. Kernel tags now
include the complete module content/options hash. Regressions exercise release,
debug, then release sorting in one process, nested partial tiles, and debug tile
matrix math and blocked Cholesky without changing the optimization policy.

Follow-up validation uses the same B70, compiler, and compatible MuJoCo package:

| Suite | Passed | Skipped | Failed |
|---|---:|---:|---:|
| Warp dedicated SYCL tests, including five new partial/debug groups | 75 | 0 | 0 |
| Warp CPU tile-sort and block-codegen regressions | 12 | 0 | 0 |
| MJWarp CPU broadphase and the two unchanged solver regressions | 37 | 0 | 0 |
| MJWarp SYCL broadphase and the two unchanged solver regressions | 37 | 0 | 0 |
| MJWarp SYCL forward, collision driver, solver, flex, broadphase, and sensor modules | 871 | 22 | 7 |

The six-module rerun used four pytest workers and also reported two passing
subtests. All four launch/order failures from the action-3 baseline are resolved.
The seven remaining failures are the four capsule-mesh and three cylinder
multi-contact cases described in next step 5. Their assertions, numerical
tolerances, and enabled status are unchanged. This is not a fully passing MJWarp
run; resolving the compiler-stage error and tracing the remaining cylinder paths
is still required. At that baseline, compiler work was deferred and the fixes
left the toolchain, driver configuration, and optimization policy unchanged.

Historical bindless/rendering validation uses Arc Pro B70, oneAPI 2026.1.1, the unmodified
Level Zero v2 runtime, Warp 1.18.0, MJWarp 3.16.0 and MuJoCo
`3.17.1.dev1008153370`. Headless reference rendering uses `MUJOCO_GL=egl`.

| Suite | Result |
|---|---|
| Warp dedicated SYCL tests | 90 passed, 1 adapter-specific negative test skipped |
| Warp existing CPU texture tests | 61 passed |
| MJWarp BVH, rays, render utilities and rendering on SYCL | 144 passed, 1 known flex-layout case deselected |
| MJWarp new one/two-world forward/refit/render capture on CPU | 2 passed |

The SYCL run includes seven geometry/texture groups covering addressing/filtering,
copies, runtime handle arrays, nested struct/struct-array ownership, field mutation
after capture, cross-stream ordering, self-copy and destruction. The MJWarp camera
regression exercises 1, 2, 64, 1024 and 4096 worlds over three changing-input frames,
compares eager and captured RGB/depth/segmentation outputs, checks their exact
shapes/dtypes, and compares ray depths with MuJoCo CPU. These are rendering and
forward-state checks, not a long physics integration or performance qualification.

At that baseline, the deselected `test_get_segmentation_preserves_flex_ids` was
run separately on both CPU and SYCL and failed before GPU execution: the installed
MuJoCo supplies two 12-value equality records where MJWarp expected vec11 records.
No compatibility patch or permanent skip had been added. Those historical results
do not include the equality compatibility fix or targeted CCD workaround below.

### Equality compatibility and targeted CCD compiler workaround

MJWarp now sizes equality records from `mujoco.mjNEQDATA` and preserves modern
automatic weld-relative-pose semantics. The full BVH/ray/render-utility/render
selection includes the previously failing flex segmentation case without
deselection.

The initial process-wide retry-disabled default was not a safe general
workaround. The six-module physics run passed 842 cases, skipped 22 and failed
51: 23 box-box cases plus 28 flex cases. The 23 box-box cases pass on CPU and
with normal IGC retry, while disabling retry fails them even with persistent
caches disabled. The same 28 flex case IDs fail in source CPU and unmodified
installed CPU controls; two installed-package cases stop earlier at the old
equality-record reshape instead of reaching the later source force assertions.
These baseline distinctions must not be hidden with skips or relaxed tolerances.

Warp now leaves `IGC_DisableRecompilation` untouched. MJWarp requests noinline
only for its SYCL polygon-clipping helper with
`@wp.func(override={"sycl": False})`.

`inline` remains the default Boolean/`None` hint. An explicit backend override
takes precedence; unlisted backends use `inline`. Thus this helper remains
compiler-chosen on CPU and CUDA. Overrides are validated, copied, included in
stable cache hashes, and propagated to generic functions, adjoints and custom
gradient/replay functions. See [function configuration](../docs/user_guide/configuration.rst).

The following runs, recorded before the MJWarp 3.17 upstream update, leave
`IGC_DisableRecompilation` unset. The combined Warp and
focused GPU collision runs disable persistent caches to exercise compilation;
the CPU and broad SYCL runs use normal cache settings:

| Suite | Result |
|---|---|
| Combined Warp dedicated SYCL and function-inline suites | 107 passed, 3 skipped: one adapter-specific negative case and two CUDA-only PTX checks |
| MJWarp capsule-mesh/cylinder CCD and box-box cases | 66 passed |
| Full source CPU collision-driver suite | 201 passed, 2 skipped |
| Nine-module SYCL physics/ray/render rerun with normal caches | 1001 passed, 22 skipped, 28 failed, plus 16 passed subtests |
| Full SYCL collision-driver suite, included in the broad rerun | 201 passed, 2 skipped |
| SYCL broadphase, ray, render-utility and render suites, included in the broad rerun | 171 passed, no exclusions |
| Short eager/captured physics/reference cases at 1, 2, 64, 1024 and 4096 heterogeneous worlds, tracing on/off, included in the broad rerun | 10 passed |

The broad rerun's 28 failing case IDs exactly match the source CPU flex control;
there are no additional failures. All 23 retry-disabled box-box regressions and
the original seven CCD failures are absent. The render refit/capture cases also
pass at 1, 2, 64, 1024 and 4096 worlds. Subset counts in this table overlap and
must not be added together.

Fresh-process checks verify that real Warp initialization does not set the IGC
variable and preserves explicit `0`, `1` and empty-string settings. An explicit
`1` can still reproduce the known box-box failure; leave it unset for normal
retry. No compiler, driver or system package was changed, and the native Warp
runtime was not rebuilt for those focused runs. Normal release/debug, math and optimization options
remain intact. Performance of the noinline hint has not yet been measured.

Those captured physics/reference checks cover three steps, not long integration.
In the separate historical 1000-step
humanoid probe under the previous retry-disabled default, both 1024-world
repeats had finite state and zero raw overflow flags, but the second exceeded
the existing `atol=rtol=5e-4` repeat comparison: maximum absolute differences
were about `1.54e-3` in `qpos`, `1.32e-3` in `qvel` and `1.02e-2` in `qacc`.
Captured and eager traces first differ at step 1 despite equal input states
and contact multisets. Constraint-row ordering and constraint forces differ;
both first-differing-world outputs pass the unchanged one-step CPU tolerance.
The later all-world ordering and reference checks are recorded below.

One separate historical 4096-world, 1000-step captured rollout was finite with
all raw flags zero. It was one repeat, not ten, used the retry-disabled setting,
and ran alongside other tests, so it is neither current-default long-run
qualification nor a performance measurement. The historical line-search
exhaustion remains a separate investigation. The underlying compiler issue
still needs minimization and a verified fix before removing the noinline hint.

### Latest qualification

These checks use MJWarp `f34c6dd` (3.17.0), MuJoCo
`3.17.1.dev1008153370`, and the local Warp 1.18.0 source build on Arc Pro B70
with DPC++ 2026.1.1. `IGC_DisableRecompilation` is unset; headless references
use `MUJOCO_GL=egl`. The older nine-module 1001-pass/28-failure total above is
not a fresh result for this upstream update. The refreshed focused source CPU
compatibility selection has 67 passes and 24 failures; four formerly failing
interpolated elastic2d cases now pass.

**Grouped BVH fix.** The G1 integration check exposed a real SYCL refit defect.
For the same saved 64-world geometry, CPU and SYCL primitive bounds, topology
and group roots were identical, but the tree reported maximum depth 8 while
its parent chains reached 14. Group expansion restarted the depth at zero.
SYCL's level-wise refit therefore skipped the mesh leaves; brute-force rays
hit the robot while BVH rays and both eager/captured rendering hit the floor.

The shared SAH/median builder now preserves the group-leaf depth when expanding
each subtree. The qualified G1 topology and refitted node bounds match the
original correct CPU tree bit for bit; the existing query-stack depth guard
also now uses the full grouped-tree depth. Physics and intersection math are
unchanged. The native core and SYCL runtime were rebuilt. A generic translated-bounds regression
fails before the fix and passes after it for leaf sizes 1/4, CPU/SYCL, eager
refit and captured refit after rebuilding. The final rebuild also includes the
concurrently added queue/interop ABI changes, with user approval; those features
are not qualified by these MJWarp checks.

| Check before the shifted-cost fix | Result |
|---|---|
| Warp BVH, grouped BVH and dedicated SYCL geometry selection | 29 passed, 3 no-CUDA skips; clean resource destruction |
| Source MJWarp BVH, ray, render-utility and render selection | 146 passed, 1 CUDA-only graph test skipped |
| G1 replay/render at 1, 2, 64, 1024 and 4096 worlds | Each completed 1000 controls and 100 captured refit/render frames with default backface culling; all layouts, finite-state, changing-image and 24 sampled CPU center-ray checks pass |

The G1 ray maximum absolute depth error is `4.483e-7`, below the unchanged
`atol=rtol=5e-4` threshold. Rendering success is not a clean physics sign-off:
with the benchmark's unchanged 10 solver / 20 line-search limits, the per-step
raw-flag monitor records the following counts of worlds that hit each limit:

| Worlds | `ITERATIONS` | `LS_ITERATIONS` |
|---|---|---|
| 1 | 0 | 1 |
| 2 | 0 | 0 |
| 64 | 1 | 34 |
| 1024 | 15 | 588 |
| 4096 | 69 | 2314 |

All other raw flags are zero and every state remains finite. A source CPU
1000-control, one-world G1 replay also records `LS_ITERATIONS`. The probe does
not inherit the benchmark's warning masks: it exits with failure for the four
sizes that do not meet the stronger zero-raw-flags criterion.

**Long humanoid reference/stability check, before the shifted-cost fix.** Ten captured 1000-step rollouts
completed at each of 1024 and 4096 worlds. Every per-step finite-state check
passes and every raw flag, including solver and line-search limits, is zero.
The reference gate compares a MuJoCo one-step result from every world's saved
pre-step inputs at steps 1, 10, 100, 500 and 1000, not an independent long CPU
trajectory. Its absolute and relative tolerances both remain `5e-4`.

| Worlds | Completed rollouts | Step-1 reference checks | Later reference checks | Maximum absolute `qacc` error |
|---|---|---|---|---|
| 1024 | 10 x 1000 steps | 10 pass | 40 fail in `qacc` only | `0.007485` |
| 4096 | 10 x 1000 steps | 10 pass | 40 fail in `qacc` only | `0.008475` |

`qpos`, `qvel` and time pass every reference checkpoint. Exact saved-state
MJWarp CPU controls also fail the acceleration gate: at the 1024-world step-500
and step-1000 snapshots, all worlds have at least one failing `qacc` element,
with maximum errors `0.004662` and `0.004541`. This excludes an exclusively
Intel or capture-only explanation, but does not turn the failed reference gate
into a pass. The collecting probe completes
all rollouts and deliberately exits nonzero for its 80 failed checkpoints.

First-step diagnostics confirm identical reset inputs, contact multisets,
mass matrices/factors and active mathematical constraint rows up to permutation
in all 1024/4096 worlds, with all-world first-step reference agreement.
Constraint ordering differs, but this is not proof that later drift is harmless.
Strict long-horizon comparisons remain failed separately: across the repeats,
maximum differences are about `0.002989` in `qpos`, `0.002557` in `qvel` and
`0.011139` in `qacc`. Constraint-order determinism is not a remedy yet:
MJWarp's `_sort_constraints` is still a stub. No tolerance, solver limit,
determinism policy or physics algorithm was changed for acceptance.

**Shifted-cost false-convergence repair.** Exact replay of the original
4096-world worst case (world 2598, step 500, right-ankle DOF 14) exposed a
separate SYCL fast-math problem. A redundant `quad0 - cost0` offset for an
initially active unilateral row allowed the large baseline cost to contaminate
the small shifted cost. A minimal device probe returned `5.3481344e-6` at
alpha zero, where the shifted cost must be exactly zero; CPU returned zero.
An improving Newton trial was consequently rejected and alpha zero caused the
outer solver to stop without an overflow flag. The initial slope/noise guard
was not responsible.

MJWarp now selects the active row's exact zero offset directly in both cone
paths and both single/triple trial evaluators. Fast math, stationary-point
safeguards, stopping tolerances and iteration limits remain unchanged. The
new backend-independent tests cover cost, gradient, Hessian, activation
transitions and useful-step acceptance at 1, 2, 64, 1024 and 4096 worlds.
The full solver suite passes **131 tests, one skip and two passed subtests
on each of CPU and SYCL**; the focused SYCL numerical selection passes 29 tests.
Ruff and the repository kernel analyzer pass.

In the original full-batch replay, alpha changes from zero to `0.999935`.
The selected acceleration changes from `0.00765125` to `0.00260908` rad/s²,
versus the MuJoCo reference `-0.000823323`: error drops from `0.00847457`
to `0.00343240`, but still fails the unchanged reference allowance. The
FP64 diagnostic optimum of MJWarp's own rounded inputs is `0.00260307`;
the repaired solver is within `2.05e-5` of that optimum across all selected
world DOFs. This is a solver repair, not full reference-accuracy sign-off.

**Remaining constraint-input precision.** CPU controls identify an independent
input-precision limitation. The selected knee-limit distance error
`-8.1236581e-8` rad comes entirely from converting the model's limit to
float32; the subsequent subtraction adds no error in that example. Rounding
only joint limits in a diagnostic double-precision MuJoCo model changes the
selected acceleration by `0.000917194` rad/s², already greater than its
approximately `0.000500412` allowance. That diagnostic does not change the
qualification reference. Contact positions/transforms also differ at roughly
`1e-7` m scale. Solving rounded constraints more accurately cannot recover
information lost before the solver. A higher-precision input policy remains
an explicit architectural decision. Per user direction, the upstream float32
policy is retained: no higher-precision model, state, constraint or solver
changes are included. FP64 and rounded-model experiments remain host-only
session diagnostics, not changes to the qualification reference or runtime.
The case decompositions do not establish the composition of every failed
checkpoint.

**Post-fix float32 requalification (2026-10-30).** The long humanoid and G1
checks have now been repeated after the solver repair. Runtime dtype assertions
confirm representative model, state, mass, constraint and solver arrays remain
float32. Thirteen recorded source/model/native-library fingerprints remain
unchanged across the qualification. No production code, compiler/runtime
configuration, tolerance, iteration limit or dependency changes were made
during this testing phase.

Fresh forward/integrator/capture tests pass **29 tests and 14 subtests on
SYCL**, including eager/captured and event-traced comparisons at 1, 2, 64,
1024 and 4096 worlds. CPU controls pass **24 tests, five expected captured-event
skips and 14 subtests**. These complement the full solver results above; their
counts are not additive with earlier selections.

All 20 post-fix captured humanoid rollouts complete 1000 steps, with finite
state and zero raw flags at every step. The same saved-input MuJoCo reference
gate, at unchanged `atol=rtol=5e-4`, still fails:

| Worlds | Completed rollouts | Step-1 reference checks | Later reference checks | Maximum absolute `qacc` error |
|---|---|---|---|---|
| 1024 | 10 x 1000 steps | 10 pass | 40 fail in `qacc` only | `0.005184` |
| 4096 | 10 x 1000 steps | 10 pass | 40 fail in `qacc` only | `0.004939` |

`qpos`, `qvel` and time pass every reference checkpoint. No world in any of
the 80 saved failed checkpoints returns its entire input `qacc_warmstart`
unchanged. Strict long-horizon repeat comparisons fail separately, with
maximum differences `0.002968` in `qpos`, `0.002509` in `qvel` and `0.011147`
in `qacc`; time matches. The collecting probe completes all requested runs
and deliberately exits nonzero rather than hiding these accuracy failures.

The new worst component is the left ankle (`ankle_x_left`, DOF 20), at
1024 worlds, repeat 1, step 500, world 416. Its error `0.005183999` exceeds
the `0.000502050` allowance by `10.3257x`. A full-batch eager replay's
host-only FP64 refinement attributes `0.005183446` to its rounded mathematical
inputs; the selected acceleration needs only a `2.55e-6` solver correction.
The maximum same-input correction across that world's DOFs is `2.11e-5`.
Substituting aligned reference `J`, `D` and `aref` reduces maximum error to
`8.54e-6`. This supports the input-precision diagnosis for this case without
promoting the diagnostic's precision to production.

Fresh exact-input CPU replays also fail the reference `qacc` gate: maximum
error is `0.004857` for the original 4096-world worst-case snapshot and
`0.005075` for the new 1024-world worst-case snapshot. Both have zero raw
flags, one solver iteration per world and no unchanged whole-world warmstarts.
A fresh captured SYCL replay of the original 4096-world snapshot likewise
takes a useful step in every world and has zero raw flags, while its reference
acceleration gate still fails. These controls exclude an exclusively Intel or
capture-only explanation; they do not turn the failed reference gate into a pass.

The post-fix G1 replay/render check completes 1000 controls and 100 captured
RGB/depth/segmentation frames at every requested size, with default backface
culling. All layouts and finite-state checks pass, 99 frames per size change
from the first image, and all 24 sampled CPU center-ray depth/segmentation
checks pass. Maximum depth error is `6.183e-7` m. All resource/capacity flags
remain zero. With the unchanged 10 solver / 20 line-search limits, the
stronger zero-raw-flags criterion now fails at all five sizes:

| Worlds | `ITERATIONS` | `LS_ITERATIONS` |
|---|---|---|
| 1 | 0 | 1 |
| 2 | 0 | 1 |
| 64 | 1 | 29 |
| 1024 | 8 | 593 |
| 4096 | 57 | 2302 |

CPU G1 controls also complete 1000 controls and 100 rendered frames at 1, 64
and 1024 worlds, with finite state, correct sampled rays and no resource
flags. They record line-search-limit flags in 1, 41 and 641 worlds,
respectively, and no solver-limit flags. Limits were not raised and flags
were not masked. Rendering and stability coverage pass, but the strict
reference, repeatability and zero-raw-flags gates remain explicitly failed;
this is not full physics sign-off.

MJWarp itself is the integration test: `pytest --device sycl:0 -n 24` in `mujoco_warp`.

Existing array, codegen, launch, gradient, and module tests run on CPU.
Both `build_lib.py --sycl --no-cuda` and the CMake `warp-sycl` target are validated.
CUDA execution requires separate hardware validation; it is not inferred from
passing CPU or SYCL tests.

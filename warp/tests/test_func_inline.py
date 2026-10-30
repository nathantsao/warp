# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np

import warp as wp
from warp._src.context import ModuleBuilder
from warp.tests.unittest_utils import add_function_test, assert_np_equal, get_test_devices, run_test_in_subprocess


def module_source(kernel, device: str) -> str:
    """Generate the full module source, including @wp.func bodies, for a kernel's module."""
    options = kernel.module.resolve_options(wp.config) | kernel.options
    if device == "sycl":
        return kernel.module._run_codegen(options, is_cpu=False, is_sycl=True)[0]
    return ModuleBuilder(kernel.module, options).codegen(device)


@wp.func(inline=False)
def scale_out_of_line(x: float) -> float:
    return x * 2.0


@wp.kernel(enable_backward=False, module="unique")
def apply_scale(a: wp.array[float]):
    tid = wp.tid()
    a[tid] = scale_out_of_line(float(tid))


def test_noinline_func_runs(test, device):
    """Run a kernel calling a function marked noinline."""
    values = wp.empty(128, dtype=float, device=device)
    wp.launch(apply_scale, dim=values.shape, inputs=[values], device=device)
    assert_np_equal(values.numpy(), np.arange(128, dtype=np.float32) * 2.0)


@wp.func(override={"sycl": False})
def square_backend_hint(x: float) -> float:
    return x * x


@wp.kernel(module="unique")
def apply_square_backend_hint(values: wp.array[float], output: wp.array[float]):
    tid = wp.tid()
    output[tid] = square_backend_hint(values[tid])


def test_inline_override_func_runs(test, device):
    """Run forward and backward kernels with a backend-specific inline hint."""
    expected = np.arange(16, dtype=np.float32)
    values = wp.array(expected, dtype=float, device=device, requires_grad=True)
    output = wp.empty_like(values)
    with wp.Tape() as tape:
        wp.launch(apply_square_backend_hint, dim=values.shape, inputs=[values, output], device=device)
    tape.backward(grads={output: wp.ones_like(output)})
    assert_np_equal(output.numpy(), expected * expected)
    assert_np_equal(values.grad.numpy(), expected * 2.0)


class TestFuncInline(unittest.TestCase):
    def test_invalid_inline_hint_rejected(self):
        """Reject non-bool values for the inline hint."""
        for value in (1, 1.5, "yes"):
            with self.subTest(value=value), self.assertRaisesRegex(TypeError, "inline must be a bool"):

                @wp.func(inline=value, module="unique")
                def invalid(x: float) -> float:
                    return x

    def test_invalid_inline_override_rejected(self):
        """Reject invalid backend override containers, keys and values."""
        for override in (False, [], "sycl"):
            with self.subTest(override=override), self.assertRaisesRegex(TypeError, "override must be a dict"):

                @wp.func(override=override, module="unique")
                def invalid_container(x: float) -> float:
                    return x

        for backend in ("sycl:0", "vulkan", None):
            with self.subTest(backend=backend), self.assertRaisesRegex(ValueError, "Unknown inline override backend"):

                @wp.func(override={backend: False}, module="unique")
                def invalid_backend(x: float) -> float:
                    return x

        for value in (None, 1, 1.5, "yes"):
            with self.subTest(value=value), self.assertRaisesRegex(TypeError, "must be a bool"):

                @wp.func(override={"sycl": value}, module="unique")
                def invalid_value(x: float) -> float:
                    return x

    def test_inline_override_copied(self):
        """Copy the override mapping so caller mutations do not change a registered function."""
        override = {"sycl": False}

        @wp.func(override=override, module="unique")
        def helper(x: float) -> float:
            return x

        override["sycl"] = True
        self.assertIsNone(helper.inline_hint)
        self.assertEqual(helper.inline_hint_overrides, {"sycl": "noinline"})

    def test_codegen_emits_inline_attrs(self):
        """Emit the requested inline attribute for each backend, and nothing when none is requested."""
        for inline, macro in ((None, None), (False, "WP_NOINLINE"), (True, "WP_FORCEINLINE")):

            @wp.func(inline=inline, module="unique")
            def helper(x: float) -> float:
                return x + 1.0

            @wp.kernel(enable_backward=False, module=helper.module)
            def uses_helper(a: wp.array[float]):
                a[wp.tid()] = helper(1.0)

            for device in ("cpu", "cuda", "sycl"):
                with self.subTest(device=device, inline=inline):
                    # NOTE: match on each macro's use, not its name -- the module header
                    # #defines both regardless of whether any function requested them.
                    source = module_source(uses_helper, device)
                    emitted = [m for m in ("WP_NOINLINE", "WP_FORCEINLINE") if f"static {m} " in source]
                    self.assertEqual(emitted, [macro] if macro else [])

    def test_codegen_emits_inline_overrides(self):
        """Apply backend overrides before the default hint without affecting unlisted backends."""
        cases = (
            (None, {"sycl": False}),
            (None, {"cpu": True}),
            (False, {"sycl": True}),
            (True, {"cuda": False, "sycl": False}),
            (None, {}),
        )
        for inline, override in cases:

            @wp.func(inline=inline, override=override, module="unique")
            def helper(x: float) -> float:
                return x + 1.0

            @wp.kernel(enable_backward=False, module=helper.module)
            def uses_helper(a: wp.array[float]):
                a[wp.tid()] = helper(1.0)

            for device in ("cpu", "cuda", "sycl"):
                with self.subTest(device=device, inline=inline, override=override):
                    source = module_source(uses_helper, device)
                    hint = override.get(device, inline)
                    macro = None if hint is None else ("WP_FORCEINLINE" if hint else "WP_NOINLINE")
                    emitted = [m for m in ("WP_NOINLINE", "WP_FORCEINLINE") if f"static {m} " in source]
                    self.assertEqual(emitted, [macro] if macro else [])

    def test_inline_override_cache_hash(self):
        """Hash override values and backends while ignoring mapping insertion order."""

        def function_hash(inline=None, override=None):
            @wp.func(inline=inline, override=override, module="unique")
            def helper(x: float) -> float:
                return x + 1.0

            @wp.kernel(enable_backward=False, module=helper.module)
            def uses_helper(a: wp.array[float]):
                a[wp.tid()] = helper(1.0)

            return uses_helper.module.get_module_hash()

        self.assertEqual(function_hash(), function_hash(override={}))
        hashes = (
            function_hash(),
            function_hash(override={"sycl": False}),
            function_hash(override={"sycl": True}),
            function_hash(override={"cuda": False}),
            function_hash(inline=True, override={"sycl": False}),
        )
        self.assertEqual(len(set(hashes)), len(hashes))
        self.assertEqual(
            function_hash(override={"sycl": False, "cpu": True}),
            function_hash(override={"cpu": True, "sycl": False}),
        )

    def test_inline_override_applied_to_generic(self):
        """Preserve overrides when a generic function is instantiated."""

        @wp.func(override={"sycl": False}, module="unique")
        def generic(x: Any):
            return x * x

        @wp.kernel(enable_backward=False, module="unique")
        def uses_generic(a: wp.array[float]):
            a[wp.tid()] = generic(2.0)

        for device in ("cpu", "cuda", "sycl"):
            with self.subTest(device=device):
                source = module_source(uses_generic, device)
                self.assertEqual("static WP_NOINLINE " in source, device == "sycl")

    def test_inline_override_applied_to_adjoint(self):
        """Resolve the default and override for both forward and adjoint definitions."""

        @wp.func(inline=True, override={"sycl": False}, module="unique")
        def differentiable(x: float) -> float:
            return x * x

        @wp.kernel(enable_backward=True, module="unique")
        def uses_differentiable(a: wp.array[float]):
            a[wp.tid()] = differentiable(2.0)

        for device in ("cpu", "cuda", "sycl"):
            with self.subTest(device=device):
                source = module_source(uses_differentiable, device)
                macro = "WP_NOINLINE" if device == "sycl" else "WP_FORCEINLINE"
                for name in (differentiable.native_func, f"adj_{differentiable.native_func}"):
                    definitions = [
                        line for line in source.splitlines() if line.startswith("static") and f" {name}(" in line
                    ]
                    self.assertEqual(len(definitions), 1)
                    self.assertIn(macro, definitions[0])

    def test_inline_override_applied_to_custom_grad_and_replay(self):
        """Propagate backend overrides to custom gradient and replay functions."""

        @wp.func(override={"sycl": False}, module="unique")
        def squared(x: float) -> float:
            return x * x

        @wp.func_grad(squared)
        def adj_squared(x: float, adj_ret: float):
            wp.adjoint[x] += 2.0 * x * adj_ret

        @wp.func_replay(squared)
        def replay_squared(x: float):
            return x * x

        @wp.kernel(enable_backward=True, module="unique")
        def uses_squared(a: wp.array[float]):
            a[wp.tid()] = squared(3.0)

        self.assertEqual(squared.custom_grad_func.inline_hint_overrides, {"sycl": "noinline"})
        self.assertEqual(squared.custom_replay_func.inline_hint_overrides, {"sycl": "noinline"})
        for device in ("cpu", "cuda", "sycl"):
            with self.subTest(device=device):
                source = module_source(uses_squared, device)
                names = (
                    squared.native_func,
                    f"adj_{squared.native_func}",
                    squared.custom_replay_func.native_func,
                )
                for name in names:
                    definitions = [
                        line for line in source.splitlines() if line.startswith("static") and f" {name}(" in line
                    ]
                    self.assertEqual(len(definitions), 1)
                    self.assertEqual("WP_NOINLINE" in definitions[0], device == "sycl", definitions[0])

    def test_inline_attr_applied_to_adjoint(self):
        """Apply the inline attribute to the generated adjoint as well as the forward function."""

        @wp.func(inline=False)
        def differentiable(x: float) -> float:
            return x * x

        @wp.kernel(enable_backward=True, module="unique")
        def uses_differentiable(a: wp.array[float]):
            a[wp.tid()] = differentiable(2.0)

        source = module_source(uses_differentiable, "cuda")
        self.assertIn("static WP_NOINLINE CUDA_CALLABLE ", source)
        self.assertIn("static WP_NOINLINE CUDA_CALLABLE void adj_", source)

    def test_inline_attr_applied_to_custom_grad_and_replay(self):
        """Apply the inline attribute to custom gradient and replay hooks, which are separate Functions."""

        @wp.func(inline=False)
        def squared(x: float) -> float:
            return x * x

        @wp.func_grad(squared)
        def adj_squared(x: float, adj_ret: float):
            wp.adjoint[x] += 2.0 * x * adj_ret

        @wp.func_replay(squared)
        def replay_squared(x: float):  # no return annotation: it is matched against the forward args
            return x * x

        @wp.kernel(enable_backward=True, module="unique")
        def uses_squared(a: wp.array[float]):
            a[wp.tid()] = squared(3.0)

        self.assertEqual(squared.custom_grad_func.inline_hint, "noinline")
        self.assertEqual(squared.custom_replay_func.inline_hint, "noinline")

        source = module_source(uses_squared, "cuda")
        # Every definition generated for this function, custom hooks included, carries the hint.
        defs = [line for line in source.splitlines() if line.startswith("static") and "squared" in line]
        self.assertTrue(defs, "expected generated definitions for the hinted function")
        for line in defs:
            self.assertIn("WP_NOINLINE", line, f"missing inline attribute: {line}")

    def test_native_snippet_codegen_unaffected(self):
        """Keep generating native snippets, which share the function templates but take no hint."""
        snippet = "out[tid] = a * x[tid];"

        @wp.func_native(snippet)
        def scale(a: wp.float32, x: wp.array[wp.float32], out: wp.array[wp.float32], tid: int):  # fmt: skip
            ...

        @wp.kernel(module="unique")
        def uses_snippet(a: wp.float32, x: wp.array[wp.float32], out: wp.array[wp.float32]):
            tid = wp.tid()
            scale(a, x, out, tid)

        # Regression: the {inline_attr} slot must be filled here too, or codegen raises KeyError.
        self.assertIn(snippet, module_source(uses_snippet, "cuda"))

    @staticmethod
    def _compile_ptx(mode):
        """Compile a kernel calling one hinted-out-of-line and one hinted-inline helper, return its PTX."""

        @wp.func(inline=False)
        def kept_out_of_line(x: float, y: float) -> float:
            a = x * y + wp.sin(x)
            return wp.sqrt(wp.abs(a)) + wp.cos(y)

        @wp.func(inline=True)
        def forced_inline(x: float, y: float) -> float:
            a = x * y + wp.cos(x)
            return wp.sqrt(wp.abs(a)) + wp.sin(y)

        @wp.kernel(enable_backward=False, module="unique", module_options={"mode": mode})
        def uses_both(out: wp.array[float]):
            tid = wp.tid()
            out[tid] = kept_out_of_line(float(tid), 2.0) + forced_inline(float(tid), 3.0)

        # Compiling into a temporary directory keeps this self-contained: module headers are not
        # part of the module hash, so a shared kernel cache could serve PTX built with a
        # different macro definition.
        with tempfile.TemporaryDirectory() as module_dir:
            artifacts = wp.compile_aot_module(
                uses_both.module, device=None, arch=80, module_dir=module_dir, use_ptx=True
            )
            ptx_files = [p for p in artifacts if str(p).endswith(".ptx")]
            if not ptx_files:
                raise AssertionError(f"no PTX artifact produced: {artifacts}")
            with open(ptx_files[0]) as f:
                return f.read()

    @staticmethod
    def _out_of_line(ptx, name):
        """Whether ``name`` survives in the PTX as a real function rather than being inlined away."""
        defined = any(line.strip().startswith(".func") and name in line for line in ptx.splitlines())
        # A PTX call spans several lines: `call.uni (retval), <mangled name>, (params);`
        called = re.search(rf"call\.uni[^;]*?{name}", ptx, re.S) is not None
        return defined, called

    @unittest.skipUnless(wp.is_cuda_available(), "requires CUDA support")
    def test_noinline_honored_in_ptx(self):
        """Check ``inline=False`` keeps a helper out of line in an optimized build.

        Release mode is what makes this discriminating: the compiler inlines a small helper by
        choice, so an ignored attribute would leave no ``.func`` behind. In debug mode nothing is
        inlined anyway and the assertion would hold with or without the attribute.
        """
        ptx = self._compile_ptx("release")
        defined, called = self._out_of_line(ptx, "kept_out_of_line")
        self.assertTrue(defined, "inline=False helper was inlined away")
        self.assertTrue(called, "inline=False helper has no call site")

    @unittest.skipUnless(wp.is_cuda_available(), "requires CUDA support")
    def test_forceinline_honored_in_ptx(self):
        """Check ``inline=True`` inlines a helper in a debug build.

        Debug mode is what makes this discriminating: it disables heuristic inlining, so the
        compiler would leave the helper out of line unless the attribute is honored. In release
        mode it inlines small helpers anyway and the assertion would hold either way.
        """
        if run_test_in_subprocess(self):
            return

        ptx = self._compile_ptx("debug")
        defined, called = self._out_of_line(ptx, "forced_inline")
        self.assertFalse(defined, "inline=True helper was left out of line")
        self.assertFalse(called, "inline=True helper is still called out of line")

    def test_inline_macros_defined_in_builtin_header(self):
        """Define the inline macros for every backend in the shared native header."""
        header = (Path(wp.__file__).parent / "native" / "builtin.h").read_text()

        self.assertIn("#define WP_NOINLINE __declspec(noinline)", header)
        # newline-anchored: __forceinline is a prefix of __forceinline__, so a bare substring passes
        self.assertIn("#define WP_FORCEINLINE __forceinline\n", header)
        self.assertIn("#define WP_NOINLINE __attribute__((noinline))", header)
        self.assertIn("#define WP_FORCEINLINE inline __attribute__((always_inline))", header)


devices = get_test_devices()
add_function_test(TestFuncInline, "test_noinline_func_runs", test_noinline_func_runs, devices=devices)
add_function_test(TestFuncInline, "test_inline_override_func_runs", test_inline_override_func_runs, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)

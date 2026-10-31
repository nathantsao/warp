# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the SYCL companion with CMake and verify both frontends' actual arguments."""

import ctypes
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

from warp._src.build import get_sycl_compiler
from warp._src.build_dll import build_sycl_runtime


class TestSyclBuild(unittest.TestCase):
    def test_build_frontends(self):
        source = Path(__file__).resolve().parents[2]
        compiler = get_sycl_compiler()
        for mode in ("debug", "release"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory(prefix="warp-sycl-build-") as directory:
                root = Path(directory)
                checkout = root / "source"
                checkout.mkdir()
                shutil.copy2(source / "CMakeLists.txt", checkout)
                shutil.copytree(source / "tools", checkout / "tools", ignore=shutil.ignore_patterns("__pycache__"))
                shutil.copytree(
                    source / "warp",
                    checkout / "warp",
                    ignore=shutil.ignore_patterns("bin", "tests", "examples", "__pycache__"),
                )
                wrapper = root / "icpx"
                wrapper.write_text(
                    f"#!{sys.executable}\n"
                    + textwrap.dedent("""\
                        import json
                        import os
                        import sys
                        from pathlib import Path

                        Path(os.environ["WARP_TEST_SYCL_ARGS"]).write_text(json.dumps(sys.argv[1:]), encoding="utf-8")
                        compiler = os.environ["WARP_TEST_SYCL_COMPILER"]
                        os.execv(compiler, [compiler, *sys.argv[1:]])
                        """),
                    encoding="utf-8",
                )
                wrapper.chmod(0o755)
                arguments = root / "arguments.json"
                env = {
                    **os.environ,
                    "WARP_TEST_SYCL_ARGS": str(arguments),
                    "WARP_TEST_SYCL_COMPILER": compiler,
                }
                build = root / "build"
                commands = (
                    [
                        "cmake",
                        "-S",
                        str(checkout),
                        "-B",
                        str(build),
                        "-DWARP_ENABLE_CUDA=OFF",
                        "-DWARP_BUILD_CLANG=OFF",
                        "-DWARP_ENABLE_SYCL=ON",
                        f"-DPython3_EXECUTABLE={sys.executable}",
                        f"-DWARP_SYCL_COMPILER={wrapper}",
                        f"-DCMAKE_BUILD_TYPE={mode.capitalize()}",
                    ],
                    ["cmake", "--build", str(build), "--target", "warp-sycl", "--parallel", "2"],
                )
                for command in commands:
                    result = subprocess.run(
                        command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False
                    )
                    self.assertEqual(result.returncode, 0, result.stdout)
                library = checkout / "warp/bin/warp-sycl.so"
                self.assertTrue(library.is_file())
                actual = json.loads(arguments.read_text(encoding="utf-8"))
                with mock.patch("warp._src.build_dll.run_sycl_compiler") as run_compiler:
                    build_sycl_runtime(str(library), mode)
                expected = [arg.replace(str(source), str(checkout)) for arg in run_compiler.call_args.args[0][1:]]
                self.assertEqual(actual, expected)
                self.assertIn("-O0" if mode == "debug" else "-O2", actual)
                if mode == "debug":
                    self.assertIn("-g", actual)
                    self.assertNotIn("-DNDEBUG", actual)
                else:
                    self.assertIn("-DNDEBUG", actual)
                native = ctypes.CDLL(str(library))
                native.wp_sycl_version.restype = ctypes.c_char_p
                self.assertIsNotNone(native.wp_sycl_version())
                self.assertIsNotNone(native.wp_sycl_queue_matches_stream)


if __name__ == "__main__":
    unittest.main(verbosity=2)

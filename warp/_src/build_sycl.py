# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path


def get_compiler() -> str:
    """Find the Intel oneAPI compiler in the configured Linux environment."""
    if sys.platform != "linux":
        raise RuntimeError("Warp SYCL support requires Linux")
    compiler = shutil.which(os.environ.get("WARP_SYCL_COMPILER", "icpx"))
    if compiler is None:
        raise RuntimeError("SYCL compiler not found; source oneAPI setvars.sh or set WARP_SYCL_COMPILER")
    return compiler


def _run(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(f"SYCL compilation failed:\n{' '.join(command)}\n{result.stdout}\n{result.stderr}")
    return result.stdout


@functools.cache
def compiler_fingerprint(compiler: str) -> str:
    digest = hashlib.sha256()
    digest.update(os.path.realpath(compiler).encode())
    digest.update(_run([compiler, "--version"]).encode())
    native = Path(__file__).resolve().parents[1] / "native"
    for header in sorted(native.glob("*.h")):
        digest.update(header.name.encode())
        digest.update(header.read_bytes())
    runtime = native.parent / "bin" / "warp-sycl.so"
    digest.update(runtime.read_bytes())
    for name in ("codegen.py", "build_sycl.py", "context.py"):
        digest.update((Path(__file__).parent / name).read_bytes())
    return digest.hexdigest()[:16]


def build_runtime(output: str, mode: str = "release") -> None:
    native = Path(__file__).resolve().parents[1] / "native"
    _run(
        [
            get_compiler(),
            "-fsycl",
            "-std=c++17",
            "-fPIC",
            "-shared",
            "-fvisibility=hidden",
            "-DWP_BUILD_DLL",
            "-O0" if mode == "debug" else "-O2",
            str(native / "sycl_runtime.cpp"),
            str(native / "sycl_array.cpp"),
            str(native / "sycl_geometry.cpp"),
            "-L",
            str(native.parent / "bin"),
            "-l:warp.so",
            "-Wl,-rpath,$ORIGIN",
            "-o",
            output,
        ]
    )


def build_module(output: str, source: str, options: dict) -> None:
    native = Path(__file__).resolve().parents[1] / "native"
    optimization = options["optimization_level"]
    if optimization is None:
        optimization = 2 if options["mode"] == "release" else 0
    command = [
        get_compiler(),
        "-fsycl",
        "-std=c++17",
        "-fPIC",
        "-shared",
        "-fvisibility=hidden",
        "-DWP_BUILD_DLL",
        f"-O{optimization}",
        "-I",
        str(native),
        source,
        "-L",
        str(native.parent / "bin"),
        "-l:warp-sycl.so",
        f"-Wl,-rpath,{native.parent / 'bin'}",
        "-o",
        output,
    ]
    if options["mode"] == "debug":
        command.append("-g")
    if options["fast_math"]:
        command.append("-ffast-math")
    if not options["fuse_fp"]:
        command.append("-ffp-contract=off")
    _run(command)

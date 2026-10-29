// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stddef.h>

template <typename T> void scan_host(const T* values_in, T* values_out, int n, bool inclusive = true);
template <typename T>
void scan_host(
    const T* values_in,
    T* values_out,
    int n,
    int in_byte_stride,
    int out_byte_stride,
    int type_length,
    bool inclusive = true
);
template <typename T> bool scan_device(const T* values_in, T* values_out, int n, bool inclusive = true);
template <typename T>
bool scan_device(
    const T* values_in,
    T* values_out,
    int n,
    int in_byte_stride,
    int out_byte_stride,
    int type_length,
    bool inclusive = true
);

// SYCL scans (scan_sycl.cpp) run on the current stream of ``device``. Without ``temp`` the
// temporary storage is taken from the stream's scratch buffer.
template <typename T> size_t scan_sycl_temp_size(int n, int type_length);
template <typename T>
void scan_sycl(
    void* device,
    const T* values_in,
    T* values_out,
    int n,
    int in_byte_stride,
    int out_byte_stride,
    int type_length,
    bool inclusive = true,
    void* temp = nullptr
);

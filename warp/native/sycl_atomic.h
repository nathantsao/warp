// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Included inside namespace wp after the scalar types and min/max helpers.
template <typename T, typename Op> inline T sycl_atomic_update(T* address, Op op)
{
    static_assert(sizeof(T) <= 8, "SYCL atomics require scalar values of at most 64 bits");
    using Word = std::conditional_t<sizeof(T) == 8, uint64_t, uint32_t>;
    using Bits = std::conditional_t<
        sizeof(T) == 8, uint64_t,
        std::conditional_t<sizeof(T) == 4, uint32_t, std::conditional_t<sizeof(T) == 2, uint16_t, uint8_t>>>;
    const uintptr_t ptr = reinterpret_cast<uintptr_t>(address);
    auto* aligned = reinterpret_cast<Word*>(ptr & ~(uintptr_t(sizeof(Word)) - 1));
    const unsigned shift = (ptr % sizeof(Word)) * 8;
    const Word mask = Word(Bits(~Bits(0))) << shift;
    sycl::atomic_ref<
        Word, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::generic_space>
        ref(*aligned);
    Word old = ref.load();
    for (;;) {
        T value = sycl::bit_cast<T>(Bits((old & mask) >> shift));
        Word desired = (old & ~mask) | (Word(sycl::bit_cast<Bits>(op(value))) << shift);
        if (ref.compare_exchange_weak(old, desired))
            return value;
    }
}

template <typename T> inline T atomic_add(T* address, T value)
{
    return sycl_atomic_update(address, [=](T old) {
        if constexpr (std::is_integral_v<T>) {
            using Unsigned = std::make_unsigned_t<T>;
            return sycl::bit_cast<T>(Unsigned(Unsigned(old) + Unsigned(value)));
        } else {
            return T(old + value);
        }
    });
}

template <typename T> inline T atomic_min(T* address, T value)
{
    return sycl_atomic_update(address, [=](T old) { return min(old, value); });
}

template <typename T> inline T atomic_max(T* address, T value)
{
    return sycl_atomic_update(address, [=](T old) { return max(old, value); });
}

template <typename T> inline T atomic_cas(T* address, T compare, T value)
{
    return sycl_atomic_update(address, [=](T old) {
        using Bits = std::conditional_t<
            sizeof(T) == 8, uint64_t,
            std::conditional_t<sizeof(T) == 4, uint32_t, std::conditional_t<sizeof(T) == 2, uint16_t, uint8_t>>>;
        return sycl::bit_cast<Bits>(old) == sycl::bit_cast<Bits>(compare) ? value : old;
    });
}

template <typename T> inline T atomic_exch(T* address, T value)
{
    return sycl_atomic_update(address, [=](T) { return value; });
}

template <typename T> inline T atomic_and(T* address, T value)
{
    return sycl_atomic_update(address, [=](T old) { return T(old & value); });
}

template <typename T> inline T atomic_or(T* address, T value)
{
    return sycl_atomic_update(address, [=](T old) { return T(old | value); });
}

template <typename T> inline T atomic_xor(T* address, T value)
{
    return sycl_atomic_update(address, [=](T old) { return T(old ^ value); });
}

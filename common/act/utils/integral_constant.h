/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file integral_constant.h
 * \brief
 */
#ifndef UTILS_INTEGRAL_CONSTANT_H
#define UTILS_INTEGRAL_CONSTANT_H

namespace AscendC {
namespace Std {
template <typename...>
struct always_false : public false_type {};

template <typename... Tp>
constexpr bool always_false_v = always_false<Tp...>::value;
} // namespace Std
} // namespace AscendC

namespace Act {
namespace Gemm {
template <int32_t t>
using Int = AscendC::Std::integral_constant<int32_t, t>;

using _0 = Int<0>;
using _1 = Int<1>;
using _2 = Int<2>; // 2:存储的位数或精度
using _4 = Int<4>; // 4:存储的位数或精度
using _8 = Int<8>; // 8:存储的位数或精度
using _16 = Int<16>; // 16:存储的位数或精度
using _32 = Int<32>; // 32:存储的位数或精度
using _64 = Int<64>; // 64:存储的位数或精度
using _128 = Int<128>; // 128:存储的位数或精度
using _256 = Int<256>; // 256:存储的位数或精度
using _512 = Int<512>; // 512:存储的位数或精度
using _1024 = Int<1024>; // 1024:存储的位数或精度
using _2048 = Int<2048>; // 2048:存储的位数或精度
using _4096 = Int<4096>; // 4096:存储的位数或精度
using _8192 = Int<8192>; // 8192:存储的位数或精度
using _16384 = Int<16384>; // 16384:存储的位数或精度
using _32768 = Int<32768>; // 32768:存储的位数或精度
using _65536 = Int<65536>; // 65536:存储的位数或精度

// Unary operator
template <auto t>
__host_aicore__ inline constexpr Int<+t> operator+(Int<t>)
{
    return {};
}

template <auto t>
__host_aicore__ inline constexpr Int<-t> operator-(Int<t>)
{
    return {};
}

template <auto t>
__host_aicore__ inline constexpr Int<~t> operator~(Int<t>)
{
    return {};
}

template <auto t>
__host_aicore__ inline constexpr Int<!t> operator!(Int<t>)
{
    return {};
}

template <auto t>
__host_aicore__ inline constexpr Int<*t> operator*(Int<t>)
{
    return {};
}

// Binary operator
template <auto t, auto u>
__host_aicore__ inline constexpr Int<(t + u)> operator+(Int<t>, Int<u>)
{
    return {};
}

template <auto t, auto u>
__host_aicore__ inline constexpr Int<(t - u)> operator-(Int<t>, Int<u>)
{
    return {};
}

template <auto t, auto u>
__host_aicore__ inline constexpr Int<(t * u)> operator*(Int<t>, Int<u>)
{
    return {};
}

template <auto t, auto u>
__host_aicore__ inline constexpr Int<(t / u)> operator/(Int<t>, Int<u>)
{
    return {};
}

template <auto t, auto u>
__host_aicore__ inline constexpr Int<(t % u)> operator%(Int<t>, Int<u>)
{
    return {};
}

template <auto t, auto u>
__host_aicore__ inline constexpr Int<(t & u)> operator&(Int<t>, Int<u>)
{
    return {};
}

template <auto t, auto u>
__host_aicore__ inline constexpr Int<(t | u)> operator|(Int<t>, Int<u>)
{
    return {};
}

template <auto t, auto u>
__host_aicore__ inline constexpr Int<(t ^ u)> operator^(Int<t>, Int<u>)
{
    return {};
}

template <auto t, auto u>
__host_aicore__ inline constexpr Int<(t << u)> operator<<(Int<t>, Int<u>)
{
    return {};
}

template <auto t, auto u>
__host_aicore__ inline constexpr Int<(t >> u)> operator>>(Int<t>, Int<u>)
{
    return {};
}

template <auto t, auto u>
__host_aicore__ inline constexpr Int<(t && u)> operator&&(Int<t>, Int<u>)
{
    return {};
}

template <auto t, auto u>
__host_aicore__ inline constexpr Int<(t || u)> operator||(Int<t>, Int<u>)
{
    return {};
}

template <auto t, auto u>
__host_aicore__ inline constexpr Int<(t == u)> operator==(Int<t>, Int<u>)
{
    return {};
}

template <auto t, auto u>
__host_aicore__ inline constexpr Int<(t != u)> operator!=(Int<t>, Int<u>)
{
    return {};
}

template <auto t, auto u>
__host_aicore__ inline constexpr Int<(t > u)> operator>(Int<t>, Int<u>)
{
    return {};
}

template <auto t, auto u>
__host_aicore__ inline constexpr Int<(t < u)> operator<(Int<t>, Int<u>)
{
    return {};
}

template <auto t, auto u>
__host_aicore__ inline constexpr Int<(t >= u)> operator>=(Int<t>, Int<u>)
{
    return {};
}

template <auto t, auto u>
__host_aicore__ inline constexpr Int<(t <= u)> operator<=(Int<t>, Int<u>)
{
    return {};
}
} // namespace Gemm
} // namespace Act
#endif
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ARCH35_ACT_CONSTANT_H
#define ARCH35_ACT_CONSTANT_H

#include "act/utils/integral_constant.h"
#include "utils/tensor_traits.h"

namespace WeightQuantBatchMatmulV2::Arch35::Act {
enum class QuantType
{
    NONE = 0,
    PER_TENSOR = 1,
    PER_CHANNEL = 2,
    PER_GROUP = 3,
};

namespace detail {
// per-channel
//   n,k
//     layout (n,_1):(_1,_1) 由于无法知道stride是多大，所以不能用n,k表达
//   k,n
//     layout (n,_1):(_1,n)
// per-tensor
//   layout (_1,):(_1,)
// per-group
//   n,k
//     layout (n,gn):(gn,_1)
//   k,n
//     layout (n,gn):(_1, n)
template <typename T>
struct QuantTypeFrom;

template <>
struct QuantTypeFrom<AscendC::Std::tuple<::Act::Gemm::_1>> {
    static constexpr QuantType VALUE = QuantType::PER_TENSOR;
};

template <>
struct QuantTypeFrom<AscendC::Std::tuple<uint64_t, ::Act::Gemm::_1>> {
    static constexpr QuantType VALUE = QuantType::PER_CHANNEL;
};

template <>
struct QuantTypeFrom<AscendC::Std::tuple<uint64_t, uint64_t>> {
    static constexpr QuantType VALUE = QuantType::PER_GROUP;
};
} // namespace detail

template <typename Shape>
constexpr QuantType QUANT_TYPE = detail::QuantTypeFrom<typename AscendC::Std::remove_cvref_t<Shape>>::VALUE;

constexpr uint32_t BYTE_PER_BLK = 32;
static constexpr uint64_t SYNC_MODE4 = 4;
static constexpr uint64_t SYNC_AIV_AIC_FLAG = 8;
static constexpr uint64_t SYNC_AIC_AIV_FLAG = 9;
static constexpr uint64_t FLAG_ID_MAX = 16;

namespace detail {
// (1, n) layout (n,_1):(_1,n)
// (n, 1) layout (n,_1):(_1,_1)
template <typename Layout>
struct is_2d_dim1_1_impl : AscendC::Std::false_type {};

template <typename S0, typename S1, typename T0, typename T1>
struct is_2d_dim1_1_impl<AscendC::Layout<AscendC::Std::tuple<S0, S1>, AscendC::Std::tuple<T0, T1>>>
    : AscendC::Std::bool_constant<
          AscendC::Std::is_same_v<S1, ::Act::Gemm::_1> && AscendC::Std::is_same_v<T0, ::Act::Gemm::_1>> {};

template <typename Layout>
struct IsRowMajor2D_impl : AscendC::Std::false_type {};

template <typename Shape, typename T0>
struct IsRowMajor2D_impl<AscendC::Layout<Shape, AscendC::Std::tuple<T0, ::Act::Gemm::_1>>>
    : AscendC::Std::bool_constant<!AscendC::Std::is_same_v<T0, ::Act::Gemm::_1>> {};

template <typename Layout>
struct IsColumnMajor2D_impl : AscendC::Std::false_type {};

// shape(t0, t1) stride(t0, t1)
template <typename T0, typename T1, typename U1>
struct IsColumnMajor2D_impl<AscendC::Layout<AscendC::Std::tuple<T0, T1>, AscendC::Std::tuple<::Act::Gemm::_1, U1>>>
    : AscendC::Std::bool_constant<!AscendC::Std::is_same_v<T1, ::Act::Gemm::_1> && !AscendC::Std::is_tuple_v<T1>> {};

template <typename Layout>
struct is_2d_nz_impl : AscendC::Std::false_type {};
template <typename Shape, typename T0, typename T1, typename U0, typename U1>
struct is_2d_nz_impl<
    AscendC::Layout<Shape, AscendC::Std::tuple<AscendC::Std::tuple<T0, T1>, AscendC::Std::tuple<U0, U1>>>>
    : AscendC::Std::bool_constant<
          AscendC::Std::is_same_v<T0, ::Act::Gemm::_16> && AscendC::Std::is_same_v<T1, ::Act::Gemm::_256> &&
          AscendC::Std::is_same_v<U0, ::Act::Gemm::_1>> {};

// (a1,b1,b0,a0)
// layout ((a0,a1),(b0,b1)):((_1,u32),(_16,_256))
template <typename Layout>
struct is_2d_zn_impl : AscendC::Std::false_type {};
template <typename Shape, typename T1>
struct is_2d_zn_impl<AscendC::Layout<
    Shape, AscendC::Std::tuple<
               AscendC::Std::tuple<::Act::Gemm::_1, T1>, AscendC::Std::tuple<::Act::Gemm::_16, ::Act::Gemm::_256>>>>
    : AscendC::Std::bool_constant<!AscendC::Std::is_same_v<T1, ::Act::Gemm::_1>> {};

template <typename Layout>
struct IsRowMajor2DContiguousImpl : AscendC::Std::false_type {};
template <typename S0, typename S1>
struct IsRowMajor2DContiguousImpl<
    AscendC::Layout<AscendC::Std::tuple<S0, S1>, AscendC::Std::tuple<S1, ::Act::Gemm::_1>>>
    : AscendC::Std::bool_constant<is_static_v<S1>> {};
} // namespace detail

// is_2d_dim1_1
//   (,_1):(_1,)
// is_2d_nd_no_trans
//   :(!=_1,_1)
// is_2d_nd_trans
//   (,!=_1):(_1,)
// is_2d_nz
//   :((_16,_256),(_1,))
// is_2d_zn
//   :((_1,!=_1),(_16,_256))
// IsRowMajor2DContiguous
//   (,_x):(_x,_1)
template <typename Layout>
struct is_2d_dim1_1 : detail::is_2d_dim1_1_impl<typename AscendC::Std::remove_cvref_t<Layout>> {};

template <typename Layout>
struct IsColumnMajor2D : detail::IsColumnMajor2D_impl<typename AscendC::Std::remove_cvref_t<Layout>> {};

template <typename Layout>
struct IsRowMajor2D : detail::IsRowMajor2D_impl<typename AscendC::Std::remove_cvref_t<Layout>> {};

template <typename Layout>
struct is_2d_nz : detail::is_2d_nz_impl<typename AscendC::Std::remove_cvref_t<Layout>> {};

template <typename Layout>
struct is_2d_zn : detail::is_2d_zn_impl<typename AscendC::Std::remove_cvref_t<Layout>> {};

template <typename Layout>
struct IsRowMajor2DContiguous : detail::IsRowMajor2DContiguousImpl<typename AscendC::Std::remove_cvref_t<Layout>> {};

namespace detail {
template <typename T>
struct KbElemImpl;

template <>
struct KbElemImpl<int4b_t> {
    static constexpr uint32_t VALUE = 2048;
};

template <>
struct KbElemImpl<int8_t> {
    static constexpr uint32_t VALUE = 1024;
};

template <>
struct KbElemImpl<float> {
    static constexpr uint32_t VALUE = 256;
};

template <>
struct KbElemImpl<half> {
    static constexpr uint32_t VALUE = 512;
};

template <>
struct KbElemImpl<bfloat16_t> {
    static constexpr uint32_t VALUE = 512;
};
} // namespace detail

template <typename T>
constexpr uint32_t KB_ELEM = detail::KbElemImpl<typename AscendC::Std::remove_cvref_t<T>>::VALUE;

constexpr int32_t DOUBLE_BUFFER_NUM = 2;
constexpr int32_t QUADRUPLE_BUFFER_NUM = 4;

template <typename T>
constexpr uint64_t BLK_ELEM = 32 / sizeof(T);

template <typename T>
inline constexpr uint32_t C0_SIZE = 32 / sizeof(T);
} // namespace WeightQuantBatchMatmulV2::Arch35::Act
#endif
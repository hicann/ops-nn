/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ARCH35_ACT_UTILS_TENSOR_TRAITS_H
#define ARCH35_ACT_UTILS_TENSOR_TRAITS_H
#include "kernel_operator.h"
#include "act/utils/integral_constant.h"

namespace WeightQuantBatchMatmulV2::Arch35::Act {

template <class T>
struct is_static : AscendC::Std::bool_constant<std::is_empty<T>::value> {};

template <class T>
constexpr bool is_static_v = is_static<AscendC::Std::remove_cvref_t<T>>::value;

template <class T>
struct all_static : AscendC::Std::bool_constant<is_static_v<T>> {};

template <class... Ts>
struct all_static<AscendC::Std::tuple<Ts...>> : AscendC::Std::bool_constant<(all_static<Ts>::value && ...)> {};

template <class T>
constexpr bool all_static_v = all_static<AscendC::Std::remove_cvref_t<T>>::value;

template <typename TensorTrait>
constexpr bool in_gm = TensorTrait::tPos == AscendC::TPosition::GM;

template <typename TensorTrait>
constexpr bool in_l1 = TensorTrait::tPos == AscendC::TPosition::TSCM || TensorTrait::tPos == AscendC::TPosition::A1 ||
                       TensorTrait::tPos == AscendC::TPosition::B1 || TensorTrait::tPos == AscendC::TPosition::C1;

template <typename TensorTrait>
constexpr bool in_l0a = TensorTrait::tPos == AscendC::TPosition::A2;

template <typename TensorTrait>
constexpr bool in_l0b = TensorTrait::tPos == AscendC::TPosition::B2;

template <typename TensorTrait>
constexpr bool in_bt = TensorTrait::tPos == AscendC::TPosition::C2;

template <typename TensorTrait>
constexpr bool in_l0c = TensorTrait::tPos == AscendC::TPosition::CO1;

template <typename TensorTrait>
constexpr bool in_ub =
    TensorTrait::tPos == AscendC::TPosition::VECIN || TensorTrait::tPos == AscendC::TPosition::VECOUT ||
    TensorTrait::tPos == AscendC::TPosition::VECCALC;

template <bool isNz, typename T, AscendC::TPosition TPos>
struct TensorTraitL1 {};

template <typename T, AscendC::TPosition TPos>
struct TensorTraitL1<true, T, TPos> {
    // nz kn n1,k1,k0,n0
    using Type = AscendC::TensorTrait<
        T, TPos,
        AscendC::Layout<
            AscendC::Shape<AscendC::Shape<::Act::Gemm::_16, uint64_t>, AscendC::Shape<::Act::Gemm::_16, uint64_t>>,
            AscendC::Stride<
                AscendC::Stride<::Act::Gemm::_1, uint64_t>, AscendC::Stride<::Act::Gemm::_16, ::Act::Gemm::_256>>>>;
};

template <typename T, AscendC::TPosition TPos>
struct TensorTraitL1<false, T, TPos> {
    // zn nk k1,n1,n0,k0
    using Type = AscendC::TensorTrait<
        T, TPos,
        AscendC::Layout<
            AscendC::Shape<AscendC::Shape<::Act::Gemm::_16, uint64_t>, AscendC::Shape<::Act::Gemm::_16, uint64_t>>,
            AscendC::Stride<
                AscendC::Stride<::Act::Gemm::_16, ::Act::Gemm::_256>, AscendC::Stride<::Act::Gemm::_1, uint64_t>>>>;
};
} // namespace WeightQuantBatchMatmulV2::Arch35::Act
#endif
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef PROLOGUE_CONSTANT_H
#define PROLOGUE_CONSTANT_H

#include "kernel_operator_intf.h"
#include "../utils/tensor_traits.h"

namespace WeightQuantBatchMatmulV2::Arch35::Act::Prologue {
namespace detail {
template <typename Dst, typename Src>
struct VectorRegSize {
    static_assert(AscendC::Std::always_false_v<Src>, "not impl");
};

template <typename Dst>
struct VectorRegSize<Dst, Dst> {
    static_assert(AscendC::SupportType<Dst, uint8_t, int8_t, half, bfloat16_t>(), "not support this dtype");
    static constexpr uint32_t VALUE = AscendC::VECTOR_REG_WIDTH / sizeof(Dst);
};
template <>
struct VectorRegSize<bfloat16_t, uint8_t> {
    static constexpr uint32_t VALUE = AscendC::VECTOR_REG_WIDTH >> 1;
};

template <>
struct VectorRegSize<bfloat16_t, int8_t> {
    static constexpr uint32_t VALUE = AscendC::VECTOR_REG_WIDTH >> 1;
};

template <>
struct VectorRegSize<bfloat16_t, fp8_e5m2_t> {
    static constexpr uint32_t VALUE = AscendC::VECTOR_REG_WIDTH >> 1;
};

template <>
struct VectorRegSize<bfloat16_t, hifloat8_t> {
    static constexpr uint32_t VALUE = AscendC::VECTOR_REG_WIDTH >> 1;
};

// 对 half 做同样的处理
template <>
struct VectorRegSize<half, uint8_t> {
    static constexpr uint32_t VALUE = AscendC::VECTOR_REG_WIDTH >> 1;
};

template <>
struct VectorRegSize<half, int8_t> {
    static constexpr uint32_t VALUE = AscendC::VECTOR_REG_WIDTH >> 1;
};

template <>
struct VectorRegSize<half, fp8_e5m2_t> {
    static constexpr uint32_t VALUE = AscendC::VECTOR_REG_WIDTH >> 1;
};

template <>
struct VectorRegSize<half, hifloat8_t> {
    static constexpr uint32_t VALUE = AscendC::VECTOR_REG_WIDTH >> 1;
};

template <>
struct VectorRegSize<int4x2_t, int4x2_t> {
    static constexpr uint32_t VALUE = AscendC::VECTOR_REG_WIDTH << 1;
};

template <>
struct VectorRegSize<bfloat16_t, int4x2_t> {
    static constexpr uint32_t VALUE = AscendC::VECTOR_REG_WIDTH >> 1;
};

template <>
struct VectorRegSize<half, int4x2_t> {
    static constexpr uint32_t VALUE = AscendC::VECTOR_REG_WIDTH >> 1;
};
} // namespace detail

template <typename Dst, typename Src = Dst>
constexpr uint32_t VECTOR_REG_SIZE = detail::VectorRegSize<
    typename AscendC::Std::remove_cvref_t<Dst>, typename AscendC::Std::remove_cvref_t<Src>>::VALUE;
} // namespace WeightQuantBatchMatmulV2::Arch35::Act::Prologue

// template <typename T>
// inline constexpr uint32_t VECTOR_REG_SIZE = AscendC::VECTOR_REG_WIDTH / sizeof(T);

// template <>
// inline constexpr uint32_t VECTOR_REG_SIZE<AscendC::int4b_t> = AscendC::VECTOR_REG_WIDTH << 1;
// namespace WeightQuantBatchMatmulV2::Arch35::Act::Prologue

#endif
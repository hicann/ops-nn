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
 * \file catlass_convertor.h
 * \brief
 */

#ifndef QUANT_BATCH_MATMUL_V4_ARCH35_CATLASS_CONVERTOR_H
#define QUANT_BATCH_MATMUL_V4_ARCH35_CATLASS_CONVERTOR_H
#include "catlass/block/block_mmad_mx_weight_from_ub.h"
#include "catlass/block/block_scheduler_swizzle_in_mn_core_nn.h"
#include "catlass/kernel/kernel_matmul_mix_with_weight_prologue_nn.h"
#include "catlass/prologue/block_prologue_b_cast_scsc_nn.h"
#include "act/matmul/policy/dispatch_policy.h"
#include "act/utils/gemm_type.h"
#include "act/utils/integral_constant.h"
#include "kernel_operator.h"
#include "lib/std/type_traits.h"
#include "quant_batch_matmul_v4_tiling_data.h"

namespace QuantBatchMatmulV4 {
namespace Arch35 {
namespace Catlass {
using AscendC::fp8_e8m0_t;
using ProblemShape = AscendC::Std::tuple<uint64_t, uint64_t, uint64_t>;           // m, n, k
using TileShapeL1 = AscendC::Std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>;  // m, n, ka, kb
using TileShapeL0 = AscendC::Std::tuple<uint32_t, uint32_t, uint32_t>;            // m, n, k
using LayoutA = AscendC::Layout<AscendC::Std::tuple<uint64_t, uint64_t>, AscendC::Std::tuple<uint64_t, Act::Gemm::_1>>;
using LayoutC = AscendC::Layout<AscendC::Std::tuple<uint64_t, uint64_t>, AscendC::Std::tuple<uint64_t, Act::Gemm::_1>>;
using LayoutBias = AscendC::Layout<AscendC::Std::tuple<uint64_t>, AscendC::Std::tuple<Act::Gemm::_1>>;
using LayoutScale =
    AscendC::Layout<AscendC::Std::tuple<uint64_t, uint64_t>, AscendC::Std::tuple<uint64_t, Act::Gemm::_1>>;
using AType = Act::Gemm::GemmType<DTYPE_X1, LayoutA>;
using CType = Act::Gemm::GemmType<DTYPE_Y, LayoutC>;
using BiasType = Act::Gemm::GemmType<DTYPE_Y, LayoutBias>;
using ScaleType = Act::Gemm::GemmType<fp8_e8m0_t, LayoutScale>;

template <bool weightNz>
struct StrideWeight {
    static_assert(AscendC::Std::always_false_v<decltype(weightNz)>,
                  "StrideWeight should be specialized by values (true or false)");
};

template <>
struct StrideWeight<true> {
    using type = AscendC::Std::tuple<AscendC::Std::tuple<Act::Gemm::_32, Act::Gemm::_512>,
                                     AscendC::Std::tuple<Act::Gemm::_1, uint64_t>>;
};

template <>
struct StrideWeight<false> {
    using type = AscendC::Std::tuple<uint64_t, Act::Gemm::_1>;
};

template <bool weightNz>
struct ShapeWeight {
    static_assert(AscendC::Std::always_false_v<decltype(weightNz)>,
                  "ShapeWeight should be specialized by values (true or false)");
};

template <>
struct ShapeWeight<true> {
    using type = AscendC::Std::tuple<AscendC::Std::tuple<Act::Gemm::_16, uint64_t>,
                                     AscendC::Std::tuple<Act::Gemm::_32, uint64_t>>;
};

template <>
struct ShapeWeight<false> {
    using type = AscendC::Std::tuple<uint64_t, uint64_t>;
};

template <bool isNz>
struct CreateLayoutB {};

template <>
struct CreateLayoutB<false> {
    __aicore__ inline decltype(auto) operator()(uint64_t n, uint64_t k)
    {
        return AscendC::MakeLayout(AscendC::MakeShape(n, k), AscendC::MakeStride(k, Act::Gemm::_1{}));
    }
};

template <>
struct CreateLayoutB<true> {
    __aicore__ inline decltype(auto) operator()(uint64_t n, uint64_t k)
    {
        return AscendC::MakeLayout(
            AscendC::MakeShape(
                AscendC::MakeShape(Act::Gemm::_16{}, static_cast<uint64_t>(AscendC::CeilDiv(n, Act::Gemm::_16{}))),
                AscendC::MakeShape(Act::Gemm::_32{}, static_cast<uint64_t>(AscendC::CeilDiv(k, Act::Gemm::_32{})))),
            AscendC::MakeStride(
                AscendC::MakeStride(Act::Gemm::_32{}, Act::Gemm::_512{}),
                AscendC::MakeStride(Act::Gemm::_1{}, static_cast<uint64_t>(AscendC::CeilAlign(n, Act::Gemm::_16{}) *
                                                                           Act::Gemm::_32{}))));
    }
};

template <bool IS_WEIGHT_NZ>
__aicore__ inline void InvokeActKernel(GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR x1_scale, GM_ADDR x2_scale,
                                       [[maybe_unused]] GM_ADDR y_scale, [[maybe_unused]] GM_ADDR x1_offset,
                                       [[maybe_unused]] GM_ADDR x2_offset, [[maybe_unused]] GM_ADDR y_offset,
                                       [[maybe_unused]] GM_ADDR x2_table, GM_ADDR y, [[maybe_unused]] GM_ADDR workspace,
                                       const GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GET_TILING_DATA_WITH_STRUCT(qbmmv4_tiling::QuantBatchMatmulV4TilingDataParams, tilingDataIn, tiling);
    using LayoutB = AscendC::Layout<typename ShapeWeight<IS_WEIGHT_NZ>::type, typename StrideWeight<IS_WEIGHT_NZ>::type>;
    using BType = Act::Gemm::GemmType<DTYPE_X2, LayoutB>;
    using DispatchPolicy = Act::Gemm::UbAntiquantWithScSc;
    using BlockMmad =
        Act::Gemm::Block::BlockMmadNN<DispatchPolicy, TileShapeL1, TileShapeL0, AscendC::Std::tuple<AType, ScaleType>,
                                      BType, CType, BiasType, void, void>;
    using BlockPrologue = Act::Prologue::BlockPrologueNN<Act::Prologue::BCastScsc, BType, AType, BiasType, TileShapeL1>;
    using BlockScheduler =
        Act::Gemm::Block::BlockSchedulerSwizzleInMnCoreNN<ProblemShape, AscendC::Std::tuple<uint32_t, uint32_t>,
                                                          AscendC::Std::tuple<uint8_t, uint8_t>>;
    using KernelMmad =
        Act::Gemm::Kernel::KernelMatmulMixWeightPrologueNN<ProblemShape, BlockMmad, BlockScheduler, BlockPrologue>;
    auto problemShape = AscendC::MakeShape(tilingDataIn.mSize, tilingDataIn.nSize, tilingDataIn.kSize);
    typename BlockMmad::Arguments mmad{
        .ptrA = x1,
        .ptrC = y,
        .ptrAScale = x1_scale,
        .ptrBScale = x2_scale,
        .layoutA = AscendC::MakeLayout(AscendC::MakeShape(tilingDataIn.mSize, tilingDataIn.kSize),
                                       AscendC::MakeStride(tilingDataIn.kSize, Act::Gemm::_1{})),
        .layoutC = AscendC::MakeLayout(AscendC::MakeShape(tilingDataIn.mSize, tilingDataIn.nSize),
                                       AscendC::MakeStride(tilingDataIn.nSize, Act::Gemm::_1{})),
        .layoutScale = AscendC::MakeLayout(
            AscendC::MakeShape(tilingDataIn.nSize, static_cast<uint64_t>(AscendC::CeilDiv(tilingDataIn.kSize, 32UL))),
            AscendC::MakeStride(static_cast<uint64_t>(AscendC::CeilDiv(tilingDataIn.kSize, 32UL)), Act::Gemm::_1{}))};
    typename BlockPrologue::Arguments prologue{
        .ptrB = x2,
        .ptrBias = bias,
        .layoutB = CreateLayoutB<IS_WEIGHT_NZ>{}(tilingDataIn.nSize, tilingDataIn.kSize),
        .layoutBias =
            AscendC::MakeLayout(AscendC::MakeShape(tilingDataIn.nSize), AscendC::MakeStride(Act::Gemm::_1{}))};
    typename BlockScheduler::Arguments scheduler{};
    typename KernelMmad::Arguments args{
        .problemShape = problemShape, .mmad = mmad, .prologue = prologue, .scheduler = scheduler};
    auto params = KernelMmad::ToUnderlyingArguments(args, &tilingDataIn);
    KernelMmad op;
    op(params);
}
}  // namespace Catlass
}  // namespace Arch35
}  // namespace QuantBatchMatmulV4

#define KERNEL_PARAMS \
    x1, x2, bias, x1_scale, x2_scale, y_scale, x1_offset, x2_offset, y_offset, x2_table, y, workspace, tiling

#endif
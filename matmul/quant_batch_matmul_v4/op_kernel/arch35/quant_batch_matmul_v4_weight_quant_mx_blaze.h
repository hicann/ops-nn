/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_batch_matmul_v4_weight_quant_mx_blaze.h
 * \brief
 */

#pragma once

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "blaze/gemm/kernel/kernel_matmul_mix_weight_prologue.h"
#include "quant_batch_matmul_v4_tiling_data_apt.h"

namespace QuantBatchMatmulV4 {
namespace Arch35 {

template <bool IS_WEIGHT_NZ>
__aicore__ inline void RunWeightQuantMxBlazeSwat(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR x1Scale, GM_ADDR x2Scale, GM_ADDR y,
    const qbmmv4_tiling::QuantBatchMatmulV4WeightQuantMxSwatTilingData& tilingData)
{
    using AType = DTYPE_X1;
    using BType = DTYPE_X2;
    using ScaleAType = AscendC::fp8_e8m0_t;
    using ScaleBType = AscendC::fp8_e8m0_t;
    using CType = DTYPE_Y;
    using BiasType = DTYPE_BIAS;

    using LayoutA = AscendC::Te::NDExtLayoutPtn;
    using LayoutB = AscendC::Std::conditional_t<IS_WEIGHT_NZ, AscendC::Te::ZNLayoutPtn, AscendC::Te::DNExtLayoutPtn>;
    using LayoutC = AscendC::Te::NDExtLayoutPtn;
    using LayoutScaleA = AscendC::Te::ScaleANDLayoutPtn;
    using LayoutScaleB = AscendC::Te::ScaleBDNLayoutPtn;
    using LayoutBias = LayoutC;

    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t>;
    using DispatchPolicy = Blaze::Gemm::MatmulWithWeightQuantMx;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<
        DispatchPolicy, AscendC::Std::tuple<AType, ScaleAType>, AscendC::Std::tuple<LayoutA, LayoutScaleA>,
        AscendC::Std::tuple<BType, ScaleBType>, AscendC::Std::tuple<LayoutB, LayoutScaleB>, CType, LayoutC, BiasType,
        LayoutBias>;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerMatmulSwatWithTailSplit<ProblemShape>;
    using KernelImpl = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, void, BlockScheduler>;

    typename KernelImpl::Params params{
        AscendC::Te::MakeShape(static_cast<int64_t>(tilingData.m), static_cast<int64_t>(tilingData.n),
                               static_cast<int64_t>(tilingData.k)),
        {x1, x1Scale, x2Scale, y,
         AscendC::Te::MakeShape(static_cast<int64_t>(tilingData.baseM), static_cast<int64_t>(tilingData.baseN),
                                static_cast<int64_t>(tilingData.tileShapeKL1),
                                static_cast<int64_t>(tilingData.tileShapeScaleKL1)),
         AscendC::Te::MakeShape(static_cast<int64_t>(tilingData.baseM), static_cast<int64_t>(tilingData.baseN),
                                static_cast<int64_t>(tilingData.baseK)),
         tilingData.l1BufferNum, tilingData.hasBias != 0U},
        {x2, bias, tilingData.kBubSize, tilingData.nBubSize},
        {tilingData.baseM, tilingData.baseN, tilingData.mTailTile, tilingData.nTailTile, tilingData.mBaseTailSplitCnt,
         tilingData.nBaseTailSplitCnt, tilingData.mTailMain, tilingData.nTailMain}};
    KernelImpl kernel;
    kernel(params);
}

template <bool IS_WEIGHT_NZ>
__aicore__ inline void InvokeWeightQuantMxBlazeSwat(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR x1_scale, GM_ADDR x2_scale, [[maybe_unused]] GM_ADDR y_scale,
    [[maybe_unused]] GM_ADDR x1_offset, [[maybe_unused]] GM_ADDR x2_offset, [[maybe_unused]] GM_ADDR y_offset,
    [[maybe_unused]] GM_ADDR x2_table, GM_ADDR y, [[maybe_unused]] GM_ADDR workspace, const GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GET_TILING_DATA_WITH_STRUCT(qbmmv4_tiling::QuantBatchMatmulV4WeightQuantMxSwatTilingData, tilingDataIn, tiling);
    RunWeightQuantMxBlazeSwat<IS_WEIGHT_NZ>(x1, x2, bias, x1_scale, x2_scale, y, tilingDataIn);
}

} // namespace Arch35
} // namespace QuantBatchMatmulV4

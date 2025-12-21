/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QBMM_MIX_PERBLOCK_ACT_H
#define QBMM_MIX_PERBLOCK_ACT_H

#include "act/epilogue/block_epilogue_pertile.h"
#include "act/matmul/block/block_mmad_pertile.h"
#include "act/matmul/block/block_scheduler_gmm_aswt_with_tail_split.h"
#include "act/matmul/block/block_scheduler_policy.h"
#include "act/matmul/policy/dispatch_policy.h"
#include "act/matmul/kernel/kernel_qbmm_pertile.h"
#include "quant_batch_matmul_v3_tiling_data.h"

template <
    class xType, class wType, class biasType, class scaleType, class ptScaleType, class yType, class xLayout,
    class wLayout, class yLayout, class l0cType>

__aicore__ inline void QbmmActPerTileKernel(
    GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR scale, GM_ADDR perTokenScale, GM_ADDR y, GM_ADDR workSpace,
    const DequantBmm::QuantBatchMatmulV3TilingDataParams* tilingParamsIn, AscendC::TPipe* que)
{
    using L1Tileshape = AscendC::Shape<Act::Gemm::_0, Act::Gemm::_0, Act::Gemm::_0>;
    using L0Tileshape = AscendC::Shape<Act::Gemm::_0, Act::Gemm::_0, Act::Gemm::_0>;

    using AType = xType;
    using BType = wType;
    using CType = l0cType;
    using BiasType = biasType;
    using ScaleType = scaleType;
    using PtScaleType = ptScaleType;
    using YType = yType;

    using LayoutA = xLayout;
    using LayoutB = wLayout;
    using LayoutC = yLayout;
    using LayoutY = yLayout;
    using LayoutBias = yLayout;

    using ProblemShape = Act::Gemm::MatmulShape;
    using BlockScheduler = Act::Gemm::GroupedMatmulAswtWithTailSplitScheduler;
    using BlockMmadPolicy = Act::Gemm::GMMPerTile<>;
    using BlockMmad = Act::Gemm::Block::BlockMmadGmm<
        BlockMmadPolicy, AType, LayoutA, BType, LayoutB, CType, LayoutC, biasType, LayoutBias, L1Tileshape,
        L0Tileshape>;
    using BlockEpilogue = Act::Gemm::Block::BlockEpiloguePerTile<
        L0Tileshape, YType, CType, BiasType, ScaleType, LayoutA, ptScaleType, LayoutB>;

    using QbmmKernel = Act::Gemm::Kernel::QuantMmBatchPertile<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    using Params = typename QbmmKernel::Params;
    using QbmmTiling = typename QbmmKernel::QBMMTiling;

    QbmmTiling qbmmParams{

        tilingParamsIn->params.batchA,
        tilingParamsIn->params.batchB,
        tilingParamsIn->params.batchC,
        tilingParamsIn->params.batchA1,
        tilingParamsIn->params.batchA2,
        tilingParamsIn->params.batchA3,
        tilingParamsIn->params.batchA4,
        tilingParamsIn->params.batchB1,
        tilingParamsIn->params.batchB2,
        tilingParamsIn->params.batchB3,
        tilingParamsIn->params.batchB4,
        tilingParamsIn->params.batchC1,
        tilingParamsIn->params.batchC2,
        tilingParamsIn->params.batchC3,
        tilingParamsIn->params.batchC4,

        tilingParamsIn->matmulTiling.M,
        tilingParamsIn->matmulTiling.N,
        tilingParamsIn->matmulTiling.Ka,
        tilingParamsIn->matmulTiling.baseM,
        tilingParamsIn->matmulTiling.baseN,
        tilingParamsIn->matmulTiling.baseK,
        tilingParamsIn->matmulTiling.stepM,
        tilingParamsIn->matmulTiling.stepN,
        tilingParamsIn->matmulTiling.stepKa,
        tilingParamsIn->matmulTiling.stepKb,

        tilingParamsIn->adaptiveSlidingWin.mTailTile,
        tilingParamsIn->adaptiveSlidingWin.nTailTile,
        tilingParamsIn->params.groupSizeM,
        tilingParamsIn->params.groupSizeN,
        tilingParamsIn->params.groupSizeK,
    };

    Params params = {
        {1, 1, 1, 1},
        {x, weight, y, bias},
        {y, scale, perTokenScale, nullptr, static_cast<uint32_t>(tilingParamsIn->matmulTiling.baseM),
         static_cast<uint32_t>(tilingParamsIn->matmulTiling.baseN),
         static_cast<uint32_t>(tilingParamsIn->matmulTiling.baseK), tilingParamsIn->params.groupSizeM,
         tilingParamsIn->params.groupSizeN, tilingParamsIn->params.groupSizeK},
        qbmmParams};
    QbmmKernel qbmm;
    qbmm(params);
}
#endif
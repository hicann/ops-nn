/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. 
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QBMM_MIX_PERTILE_CMCT_H
#define QBMM_MIX_PERTILE_CMCT_H

#include "cmct/epilogue/block_epilogue_pertile.h"
#include "cmct/block/block_mmad_pertile.h"
#include "cmct/block/block_scheduler_qbmm.h"
#include "cmct/block/block_scheduler_policy.h"
#include "cmct/policy/dispatch_policy.h"
#include "cmct/kernel/kernel_qbmm_pertile.h"
#include "quant_batch_matmul_v3_tiling_data.h"

template <
    class xType, class wType, class biasType, class scaleType, class ptScaleType, class yType, class xLayout,
    class wLayout, class yLayout, class l0cType>

__aicore__ inline void QbmmCmctPertileKernel(
    GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR scale, GM_ADDR perTokenScale, GM_ADDR y, GM_ADDR workSpace,
    const void* tilingParamsIn, AscendC::TPipe* que)
{
    using L1Tileshape = AscendC::Shape<Cmct::Gemm::_0, Cmct::Gemm::_0, Cmct::Gemm::_0>;
    using L0Tileshape = AscendC::Shape<Cmct::Gemm::_0, Cmct::Gemm::_0, Cmct::Gemm::_0>;

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

    using ProblemShape = Cmct::Gemm::MatmulShape;
    using BlockScheduler = Cmct::Gemm::QuantBatchMatmulV3Scheduler;
    using BlockMmadPolicy = Cmct::Gemm::MmadCAccOnUb<>;
    using BlockMmad = Cmct::Gemm::Block::BlockMmadPertile<
        BlockMmadPolicy, AType, LayoutA, BType, LayoutB, CType, LayoutC, biasType, LayoutBias, L1Tileshape,
        L0Tileshape>;
    using BlockEpilogue = Cmct::Gemm::Block::BlockEpiloguePertile<
        L0Tileshape, YType, CType, BiasType, ptScaleType, ScaleType, LayoutA, LayoutB>;

    using QbmmKernel = Cmct::Gemm::Kernel::QuantMmBatchPertile<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    using Params = typename QbmmKernel::Params;
    using QbmmTiling = typename QbmmKernel::QBMMTiling;
    const DequantBmm::QuantBatchMatmulV3BasicAPITilingData* tilingData;
    tilingData = static_cast<const DequantBmm::QuantBatchMatmulV3BasicAPITilingData*>(tilingParamsIn);
    const DequantBmm::QuantBatchMatmulV3BasicAPIDataParams& dataParams = tilingData->params;
    const DequantBmm::BasicAPICubeTiling& matmulTiling = tilingData->matmulTiling;
    const DequantBmm::SlidingWindowParams& slidingWindowParams = tilingData->adaptiveSlidingWin;

    QbmmTiling qbmmParams{
        dataParams.batchA1,
        dataParams.batchA2,
        dataParams.batchA3,
        dataParams.batchA4,
        dataParams.batchB1,
        dataParams.batchB2,
        dataParams.batchB3,
        dataParams.batchB4,
        dataParams.batchC1,
        dataParams.batchC2,
        dataParams.batchC3,
        dataParams.batchC4,

        matmulTiling.stepKa * matmulTiling.baseK,
        matmulTiling.stepKb * matmulTiling.baseK,
        matmulTiling.nBufferNum,
        dataParams.biasThreeDim,
        matmulTiling.isBias
    };

    Params params = {
        {matmulTiling.m, matmulTiling.n, matmulTiling.k, dataParams.batchC},
        {x, weight, y, bias},
        {y, scale, perTokenScale, bias, matmulTiling.baseM, matmulTiling.baseN, matmulTiling.baseK,
         dataParams.groupSizeM, dataParams.groupSizeN, dataParams.groupSizeK, matmulTiling.isBias},
        {matmulTiling.baseM, matmulTiling.baseN, slidingWindowParams.mTailTile, slidingWindowParams.nTailTile,
         slidingWindowParams.mBaseTailSplitCnt, 1, slidingWindowParams.mTailMain, matmulTiling.isBias},
        qbmmParams};
    QbmmKernel qbmm;
    qbmm(params);
}
#endif
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file qbmm_mx_basic_api_act.h
 * \brief
 */
#ifndef QBMM_MX_BAISC_API_ACT_H
#define QBMM_MX_BAISC_API_ACT_H
#include "act/matmul/block/block_scheduler_policy.h"
#include "act/matmul/block/block_scheduler_utils.h"
#include "act/epilogue/block_epilogue_empty.h"
#include "act/matmul/block/mmad_mx_quant.h"
#include "act/matmul/kernel/kernel_qbmm_mx.h"
namespace QuantBatchMatmulV3Advanced {
using namespace Act;
using namespace Act::Gemm;
template <
    class A_TYPE, class B_TYPE, class PERTOKEN_SCALE_TYPE, class SCALE_TYPE, class BIAS_TYPE, class C_TYPE,
    class aLayout, class bLayout, class cLayout, bool aTrans, bool bTrans>
__aicore__ inline void QbmmMxBaiscApiKernel(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR scaleGM, GM_ADDR biasGM, GM_ADDR pertokenScaleGM, GM_ADDR cGM,
    const void* tilingData, AscendC::TPipe* que)
{
    // 定义L1和L0的TileShape
    using L1TileShape = AscendC::Shape<_0, _0, _0>;
    using L0TileShape = AscendC::Shape<_0, _0, _0>;

    // 定义矩阵的类型和布局
    using AType = A_TYPE;
    using BType = B_TYPE;
    using PertokenScaleType = PERTOKEN_SCALE_TYPE;
    using ScaleType = SCALE_TYPE;
    using BiasType = BIAS_TYPE;
    using OutType = C_TYPE;
    constexpr bool tranA = aTrans;
    constexpr bool tranB = bTrans;

    // 定义BlockEpilogue类型
    using BlockEpilogue = Block::BlockEpilogueEmpty;

    // 定义shape的形状，tuple保存 m n k batch
    using ProblemShape = MatmulShape;

    // 定义scheduler类型 来自block_scheduler_policy.h
    using BlockScheduler = Act::Gemm::QuantBatchMatmulV3Scheduler;

    // 定义MMAD类型
    using DispatchPolicy = MatmulWithScale<AscendC::Shape<_0, _0, _0, _0>, A_FULL_LOAD_MODE>;
    using BlockMmad = Block::BlockMmadMx<
        DispatchPolicy, L1TileShape, L0TileShape, AType, aLayout, BType, bLayout, OutType, cLayout, BiasType, cLayout,
        void>;

    // 定义Kernel类型
    using MatmulKernel = Act::Gemm::Kernel::QuantMmBatchMX<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    using Params = typename MatmulKernel::Params;
    const DequantBmm::QuantBatchMatmulV3BasicAPITilingData* quantBmmTilingData_;
    quantBmmTilingData_ = static_cast<const DequantBmm::QuantBatchMatmulV3BasicAPITilingData*>(tilingData);
    DequantBmm::BasicAPICubeTiling matmulTiling = quantBmmTilingData_->matmulTiling;
    DequantBmm::SlidingWindowParams slidingWindowParams = quantBmmTilingData_->adaptiveSlidingWin;
    using QBMMTiling = typename MatmulKernel::QBMMTiling;
    QBMMTiling qbmmParams{
        quantBmmTilingData_->params.batchA1,
        quantBmmTilingData_->params.batchA2,
        quantBmmTilingData_->params.batchA3,
        quantBmmTilingData_->params.batchA4,
        quantBmmTilingData_->params.batchB1,
        quantBmmTilingData_->params.batchB2,
        quantBmmTilingData_->params.batchB3,
        quantBmmTilingData_->params.batchB4,
        quantBmmTilingData_->params.batchC1,
        quantBmmTilingData_->params.batchC2,
        quantBmmTilingData_->params.batchC3,
        quantBmmTilingData_->params.batchC4,
        matmulTiling.m,
        matmulTiling.n,
        matmulTiling.k,
        matmulTiling.baseM,
        matmulTiling.baseN,
        matmulTiling.baseK,
    };
    Params params = {
        {matmulTiling.m, matmulTiling.n, matmulTiling.k, quantBmmTilingData_->params.batchC},
        {aGM, bGM, cGM, biasGM, pertokenScaleGM, scaleGM}, // gm addr
        {matmulTiling.m, matmulTiling.n, matmulTiling.k, matmulTiling.kL1, matmulTiling.scaleKL1, matmulTiling.baseM,
         matmulTiling.baseN, matmulTiling.baseK, slidingWindowParams.mTailTile, slidingWindowParams.nTailTile,
         matmulTiling.isBias, matmulTiling.nBufferNum, matmulTiling.dbL0C},
        {matmulTiling.usedCoreNum, matmulTiling.baseM, matmulTiling.baseN, slidingWindowParams.mTailTile,
         slidingWindowParams.nTailTile, slidingWindowParams.mBaseTailSplitCnt, slidingWindowParams.nBaseTailSplitCnt,
         slidingWindowParams.mTailMain, slidingWindowParams.nTailMain},
        qbmmParams};
    AscendC::TPipe tPipe;
    MatmulKernel qbmm;
    qbmm(params);
}
} // namespace QuantBatchMatmulV3Advanced
#endif
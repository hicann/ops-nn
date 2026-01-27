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
 * \file qbmm_mx_basic_api_cmct.h
 * \brief
 */
#ifndef QBMM_MX_BAISC_API_CMCT_H
#define QBMM_MX_BAISC_API_CMCT_H
#include "cmct/block/block_scheduler_policy.h"
#include "cmct/block/block_scheduler_utils.h"
#include "cmct/epilogue/block_epilogue_empty.h"
#include "cmct/block/block_mmad_mx.h"
#include "cmct/kernel/kernel_qbmm_mx.h"
using namespace Cmct;
using namespace Cmct::Gemm;
template <class A_TYPE, class B_TYPE, class C_TYPE, class aLayout, class bLayout, class cLayout,
          uint64_t FULL_LOAD_MODE = 0>
__aicore__ inline void QbmmMxBasicApiKernel(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR scale, GM_ADDR bias,
                                            GM_ADDR perTokenScale, GM_ADDR cGM, const void *tilingData)
{
    // 定义L1和L0的TileShape
    using L1TileShape = AscendC::Shape<_0, _0, _0>;
    using L0TileShape = AscendC::Shape<_0, _0, _0>;

    // 定义矩阵的类型和布局
    using AType = A_TYPE;
    using BType = B_TYPE;
    using BiasType = float;
    using OutType = C_TYPE;

    // 定义BlockEpilogue类型
    using BlockEpilogue = Block::BlockEpilogueEmpty;

    // 定义shape的形状，tuple保存 m n k batch
    using ProblemShape = MatmulShape;

    // 定义scheduler类型 来自block_scheduler_policy.h
    using BlockScheduler = Cmct::Gemm::QuantBatchMatmulV3Scheduler;

    // 定义MMAD类型
    using DispatchPolicy = MatmulWithScale<AscendC::Shape<_0, _0, _0, _0>, FULL_LOAD_MODE>;
    using BlockMmad = Block::BlockMmadMx<DispatchPolicy, L1TileShape, L0TileShape, AType, aLayout, BType, bLayout,
                                         OutType, cLayout, BiasType, cLayout, void>;

    // 定义Kernel类型
    using MatmulKernel = Cmct::Gemm::Kernel::QuantMmBatchMX<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    using Params = typename MatmulKernel::Params;
    const DequantBmm::QuantBatchMatmulV3BasicAPITilingData* quantBmmTilingData_;
    quantBmmTilingData_ = static_cast<const DequantBmm::QuantBatchMatmulV3BasicAPITilingData*>(tilingData);
    DequantBmm::BasicAPICubeTiling matmulTiling = quantBmmTilingData_->matmulTiling;
    DequantBmm::SlidingWindowParams slidingWindowParams = quantBmmTilingData_->adaptiveSlidingWin;
    using QBMMTiling = typename MatmulKernel::QBMMTiling;
    QBMMTiling qbmmParams{quantBmmTilingData_->params.batchA1,
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
                          quantBmmTilingData_->params.biasThreeDim,
                          matmulTiling.baseM, matmulTiling.baseN, matmulTiling.baseK,
                          static_cast<uint32_t>(matmulTiling.isBias), static_cast<uint32_t>(matmulTiling.dbL0C)};
    Params params = {
        {matmulTiling.m, matmulTiling.n, matmulTiling.k, quantBmmTilingData_->params.batchC},
        {aGM, bGM, cGM, bias, perTokenScale, scale}, // gm addr
        {matmulTiling.stepKb * matmulTiling.baseK, matmulTiling.scaleKL1, matmulTiling.nBufferNum},
        {matmulTiling.baseM, matmulTiling.baseN, slidingWindowParams.mTailTile, slidingWindowParams.nTailTile,
         slidingWindowParams.mBaseTailSplitCnt, slidingWindowParams.nBaseTailSplitCnt, slidingWindowParams.mTailMain,
         slidingWindowParams.nTailMain},
        qbmmParams};
    MatmulKernel qbmm;
    qbmm(params);
}
#endif
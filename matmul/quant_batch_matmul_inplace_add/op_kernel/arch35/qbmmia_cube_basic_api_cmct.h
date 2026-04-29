/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file qbmmia_cube_basic_api_cmct.h
 * \brief HiFloat8 T-T kernel for QuantBatchMatmulInplaceAdd.
 *        Uses BlockMmadA8W8FixpipeQuant for cube-based MMAD with FixPipe dequant.
 */
#ifndef QBMMIA_CUBE_BASIC_API_CMCT_H
#define QBMMIA_CUBE_BASIC_API_CMCT_H
#include "cmct/block/block_mmad_a8w8_fixpipe_quant.h"
#include "cmct/block/block_scheduler_policy.h"
#include "cmct/block/block_scheduler_utils.h"
#include "cmct/epilogue/block_epilogue_empty.h"
#include "cmct/kernel/kernel_qbmm_cube.h"
using namespace Cmct;
using namespace Cmct::Gemm;

template <
    class A_TYPE, class B_TYPE, class SCALE_TYPE, class C_TYPE, class BIAS_TYPE,
    class aLayout, class bLayout, class cLayout, uint64_t FULL_LOAD_MODE = 0>
__aicore__ inline void QbmmiaCubeBasicApiKernel(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR scale, GM_ADDR perTokenScale, GM_ADDR cGM, const void* tilingData)
{
    using L1TileShape = AscendC::Shape<_0, _0, _0>;
    using L0TileShape = AscendC::Shape<_0, _0, _0>;

    using AType = A_TYPE;
    using BType = B_TYPE;
    using BiasType = BIAS_TYPE;
    using X2ScaleType = SCALE_TYPE;
    using OutType = C_TYPE;

    using BlockEpilogue = Block::BlockEpilogueEmpty;
    using ProblemShape = MatmulShape;
    using BlockScheduler = Cmct::Gemm::QuantBatchMatmulV3Scheduler<FULL_LOAD_MODE>;

    using DispatchPolicy = MatmulWithScale<AscendC::Shape<_0, _0, _0, _0>, FULL_LOAD_MODE>;
    using BlockMmad = Block::BlockMmadA8W8FixpipeQuant<
        DispatchPolicy, L1TileShape, L0TileShape, AType, aLayout, BType, bLayout, OutType, cLayout, BiasType, cLayout,
        X2ScaleType>;

    using MatmulKernel =
        Cmct::Gemm::Kernel::QuantMmBatchCube<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler, true>;
    using Params = typename MatmulKernel::Params;

    const QMMIA::QuantBatchMatmulInplaceAddTilingData* quantBmmInplaceAddTilingData_;
    quantBmmInplaceAddTilingData_ = static_cast<const QMMIA::QuantBatchMatmulInplaceAddTilingData*>(tilingData);
    const QMMIA::BasicAPICubeTiling& matmulTiling = quantBmmInplaceAddTilingData_->matmulTiling;
    const QMMIA::SlidingWindowParams& slidingWindowParams = quantBmmInplaceAddTilingData_->adaptiveSlidingWin;

    using QBMMTiling = typename MatmulKernel::QBMMTiling;
    QBMMTiling qbmmParams{quantBmmInplaceAddTilingData_->params.batchA1,
                          quantBmmInplaceAddTilingData_->params.batchA2,
                          quantBmmInplaceAddTilingData_->params.batchA3,
                          quantBmmInplaceAddTilingData_->params.batchA4,
                          quantBmmInplaceAddTilingData_->params.batchB1,
                          quantBmmInplaceAddTilingData_->params.batchB2,
                          quantBmmInplaceAddTilingData_->params.batchB3,
                          quantBmmInplaceAddTilingData_->params.batchB4,
                          quantBmmInplaceAddTilingData_->params.batchC1,
                          quantBmmInplaceAddTilingData_->params.batchC2,
                          quantBmmInplaceAddTilingData_->params.batchC3,
                          quantBmmInplaceAddTilingData_->params.batchC4,
                          quantBmmInplaceAddTilingData_->params.biasThreeDim,
                          quantBmmInplaceAddTilingData_->params.x1QuantMode,
                          quantBmmInplaceAddTilingData_->params.x2QuantMode,
                          matmulTiling.stepKa * matmulTiling.baseK,
                          matmulTiling.stepKb * matmulTiling.baseK,
                          matmulTiling.nBufferNum,
                          matmulTiling.baseM, matmulTiling.baseN, matmulTiling.baseK,
                          static_cast<uint32_t>(matmulTiling.isBias),
                          static_cast<uint32_t>(matmulTiling.dbL0C)};
    Params params = {
        {matmulTiling.m, matmulTiling.n, matmulTiling.k, quantBmmInplaceAddTilingData_->params.batchC},
        {aGM, bGM, cGM, nullptr, perTokenScale, scale},
        {matmulTiling.baseM, matmulTiling.baseN, slidingWindowParams.mTailTile, slidingWindowParams.nTailTile,
         slidingWindowParams.mBaseTailSplitCnt, slidingWindowParams.nBaseTailSplitCnt, slidingWindowParams.mTailMain,
         slidingWindowParams.nTailMain},
        qbmmParams};
    MatmulKernel qbmm;
    qbmm(params);
}
#endif

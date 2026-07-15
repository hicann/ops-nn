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
 * \file qbmm_cube_tensor_api_blaze.h
 * \brief Blaze tensor API entry for quant batch matmul cube kernels (ND / WeightNz)
 */
#pragma once
#include "quant_batch_matmul_v3_tiling_data.h"
#include "blaze/gemm/block/block_scheduler_qbmm.h"
#include "blaze/epilogue/block/block_epilogue_empty.h"
#include "blaze/gemm/block/block_mmad_a8w8_fixpipe_quant.h"
#include "blaze/gemm/kernel/kernel_qbmm_cube.h"

template <
    class A_TYPE, class B_TYPE, class SCALE_TYPE, class C_TYPE, class BIAS_TYPE, class aLayout, class bLayout,
    class cLayout, uint64_t FULL_LOAD_MODE = 0>
__aicore__ inline void QbmmCubeTensorApiKernel(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR scale, GM_ADDR bias, GM_ADDR perTokenScale, GM_ADDR cGM, const void* tilingData)
{
    using AType = A_TYPE;
    using BType = B_TYPE;
    using BiasType = BIAS_TYPE;
    using X2ScaleType = SCALE_TYPE;
    using OutType = C_TYPE;

    using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueEmpty;
    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<
        ProblemShape, FULL_LOAD_MODE, aLayout, bLayout, AType>;

    using BlockMmad = Blaze::Gemm::Block::BlockMmad<
        Blaze::Gemm::MatmulWithScaleFixpipeQuant<FULL_LOAD_MODE>, AType, aLayout, AscendC::Std::tuple<BType, X2ScaleType>,
        bLayout, OutType, cLayout, BiasType, cLayout>;

    using MatmulKernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    using Params = typename MatmulKernel::Params;
    const DequantBmm::QuantBatchMatmulV3BasicAPITilingData* quantBmmTilingData_;
    quantBmmTilingData_ = static_cast<const DequantBmm::QuantBatchMatmulV3BasicAPITilingData*>(tilingData);
    const DequantBmm::BasicAPICubeTiling& matmulTiling = quantBmmTilingData_->matmulTiling;
    const DequantBmm::SlidingWindowParams& slidingWindowParams = quantBmmTilingData_->adaptiveSlidingWin;
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
        quantBmmTilingData_->params.biasThreeDim,
        quantBmmTilingData_->params.x1QuantMode,
        quantBmmTilingData_->params.x2QuantMode,
        static_cast<uint32_t>(matmulTiling.stepKa * matmulTiling.baseK),
        static_cast<uint32_t>(matmulTiling.stepKb * matmulTiling.baseK),
        matmulTiling.nBufferNum,
        static_cast<uint32_t>(matmulTiling.baseM),
        static_cast<uint32_t>(matmulTiling.baseN),
        static_cast<uint32_t>(matmulTiling.baseK),
        static_cast<uint32_t>(matmulTiling.isBias),
        static_cast<uint32_t>(matmulTiling.dbL0C)};
    Params params = {
        {matmulTiling.m, matmulTiling.n, matmulTiling.k, quantBmmTilingData_->params.batchC},
        {aGM, bGM, cGM, bias, perTokenScale, scale},
        {matmulTiling.baseM, matmulTiling.baseN, slidingWindowParams.mTailTile, slidingWindowParams.nTailTile,
         slidingWindowParams.mBaseTailSplitCnt, slidingWindowParams.nBaseTailSplitCnt, slidingWindowParams.mTailMain,
         slidingWindowParams.nTailMain},
        qbmmParams};
    MatmulKernel qbmm;
    qbmm(params);
}

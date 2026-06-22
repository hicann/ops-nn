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
 * \file qbmmia_mx_tensor_api_blaze.h
 * \brief
 */
#pragma once
#include "quant_batch_matmul_inplace_add_tiling_data.h"
#include "blaze/gemm/block/block_scheduler_qbmm.h"
#include "blaze/epilogue/block/block_epilogue_empty.h"
#include "blaze/gemm/block/block_mmad_qbmm_mx.h"
#include "blaze/gemm/kernel/kernel_qbmm_mx.h"

template <
    class A_TYPE, class B_TYPE, class C_TYPE, class aLayout, class bLayout, class cLayout, uint64_t FULL_LOAD_MODE = 0>
__aicore__ inline void QbmmiaMxTensorApiKernel(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR scale, GM_ADDR perTokenScale, GM_ADDR cGM, const void* tilingData)
{
    using AType = A_TYPE;
    using BType = B_TYPE;
    using BiasType = float;
    using OutType = C_TYPE;

    using BlockEpilogue = Blaze::Gemm::Block::BlockEpilogueEmpty;
    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockScheduler =
        Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<ProblemShape, FULL_LOAD_MODE, aLayout, bLayout, AType>;

    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMx<FULL_LOAD_MODE, true>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<
        DispatchPolicy, AType, aLayout, BType, bLayout, OutType, cLayout, BiasType, cLayout>;

    using MatmulKernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    using Params = typename MatmulKernel::Params;
    const QMMIA::QuantBatchMatmulInplaceAddTilingData& qbmmiaTilingData =
        *static_cast<const QMMIA::QuantBatchMatmulInplaceAddTilingData*>(tilingData);
    const QMMIA::QuantBatchMatmulV3BasicAPIDataParams& dataParams = qbmmiaTilingData.params;
    const QMMIA::BasicAPICubeTiling& matmulTiling = qbmmiaTilingData.matmulTiling;
    const QMMIA::SlidingWindowParams& slidingWindowParams = qbmmiaTilingData.adaptiveSlidingWin;

    MatmulKernel{}(Params{
        {matmulTiling.m, matmulTiling.n, matmulTiling.k, dataParams.batchC},
        {aGM, bGM, cGM, nullptr, perTokenScale, scale},
        {matmulTiling.stepKb * matmulTiling.baseK, matmulTiling.scaleKL1, matmulTiling.nBufferNum},
        {matmulTiling.baseM, matmulTiling.baseN, slidingWindowParams.mTailTile, slidingWindowParams.nTailTile,
         slidingWindowParams.mBaseTailSplitCnt, slidingWindowParams.nBaseTailSplitCnt, slidingWindowParams.mTailMain,
         slidingWindowParams.nTailMain},
        {dataParams.batchA1,
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
         dataParams.biasThreeDim,
         matmulTiling.baseM,
         matmulTiling.baseN,
         matmulTiling.baseK,
         matmulTiling.isBias,
         matmulTiling.dbL0C}});
}

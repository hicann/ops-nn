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
 * \file qbmm_mix_without_batch_tensor_api_blaze.h
 * \brief Blaze tensor API entry for quant matmul MIX kernels without batch (WeightNz).
 */
#pragma once

#include "quant_batch_matmul_v3_tiling_data.h"
#include "blaze/gemm/block/block_scheduler_qbmm.h"
#include "blaze/epilogue/block/block_epilogue_dequant.h"
#include "blaze/gemm/block/block_mmad_a8w8_mix.h"
#include "blaze/gemm/kernel/kernel_qbmm_mix_without_batch.h"

template <
    class A_TYPE, class B_TYPE, class SCALE_TYPE, class C_TYPE, class BIAS_TYPE, class aLayout, class bLayout,
    class cLayout, uint64_t FULL_LOAD_MODE = 0>
__aicore__ inline void QbmmMixWithoutBatchTensorApiKernel(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR scale, GM_ADDR bias, GM_ADDR perTokenScale, GM_ADDR cGM,
    const void* tilingData)
{
    using AType = A_TYPE;
    using BType = B_TYPE;
    using BiasType = BIAS_TYPE;
    using X2ScaleType = SCALE_TYPE;
    using OutType = C_TYPE;
    using L0CType = typename AscendC::GetMmDstType<AType>::Type;

    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<
        ProblemShape, FULL_LOAD_MODE, aLayout, bLayout, AType>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<
        Blaze::Gemm::MatmulWithScaleMix<FULL_LOAD_MODE>, AType, aLayout,
        AscendC::Std::tuple<BType, X2ScaleType>, bLayout, OutType, cLayout, BiasType, cLayout>;
    using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueDequant<OutType, BiasType, X2ScaleType, float, L0CType>;
    using MatmulKernel = Blaze::Gemm::Kernel::QbmmMixWithoutBatch<
        ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    using Params = typename MatmulKernel::Params;
    using EpilogueParams = typename BlockEpilogue::Params;

    const auto* quantBmmTilingData =
        static_cast<const DequantBmm::QuantBatchMatmulV3TensorAPIWithoutBatchTilingData*>(tilingData);

    EpilogueParams epilogueParams{
        scale,
        perTokenScale,
        bias,
        cGM,
        static_cast<int64_t>(quantBmmTilingData->m),
        static_cast<int64_t>(quantBmmTilingData->n),
        static_cast<int64_t>(quantBmmTilingData->baseM),
        static_cast<int64_t>(quantBmmTilingData->baseN),
        quantBmmTilingData->x1QuantMode,
        quantBmmTilingData->x2QuantMode,
        static_cast<bool>(quantBmmTilingData->isBias),
        quantBmmTilingData->biasDtype};

    const ProblemShape problemShape{
        static_cast<int64_t>(quantBmmTilingData->m), static_cast<int64_t>(quantBmmTilingData->n),
        static_cast<int64_t>(quantBmmTilingData->k), 1};
    const typename BlockMmad::BlockShape l0TileShape{
        static_cast<int64_t>(quantBmmTilingData->baseM), static_cast<int64_t>(quantBmmTilingData->baseN),
        static_cast<int64_t>(quantBmmTilingData->baseK), 0};
    Params params{
        problemShape,
        {aGM, bGM, problemShape, l0TileShape,
         static_cast<uint64_t>(quantBmmTilingData->stepKa * quantBmmTilingData->baseK),
         static_cast<uint64_t>(quantBmmTilingData->stepKb * quantBmmTilingData->baseK),
         static_cast<uint64_t>(quantBmmTilingData->nBufferNum), quantBmmTilingData->dbL0C > 1},
        {quantBmmTilingData->baseM, quantBmmTilingData->baseN, quantBmmTilingData->mTailTile,
         quantBmmTilingData->nTailTile, quantBmmTilingData->mBaseTailSplitCnt,
         quantBmmTilingData->nBaseTailSplitCnt, quantBmmTilingData->mTailMain,
         quantBmmTilingData->nTailMain},
        {quantBmmTilingData->groupSizeM, quantBmmTilingData->groupSizeN, quantBmmTilingData->groupSizeK,
         static_cast<uint32_t>(quantBmmTilingData->stepKa * quantBmmTilingData->baseK),
         static_cast<uint32_t>(quantBmmTilingData->stepKb * quantBmmTilingData->baseK),
         quantBmmTilingData->nBufferNum, quantBmmTilingData->baseM, quantBmmTilingData->baseN,
         quantBmmTilingData->baseK, quantBmmTilingData->isBias, quantBmmTilingData->dbL0C},
        epilogueParams};
    MatmulKernel qbmm;
    qbmm(params);
}

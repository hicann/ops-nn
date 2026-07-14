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
 * \file qbmm_mx_tensor_api_blaze.h
 * \brief
 */
#pragma once
#include "blaze/gemm/block/block_scheduler_qbmm.h"
#include "blaze/epilogue/block/block_epilogue_empty.h"
#include "blaze/gemm/block/block_mmad_qbmm_mx.h"
#include "blaze/gemm/kernel/kernel_qbmm_mx.h"

template <class A_TYPE, class B_TYPE, class C_TYPE, class aLayout, class bLayout, class cLayout,
          uint64_t FULL_LOAD_MODE = 0>
__aicore__ inline void QbmmMxTensorApiKernel(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR scale, GM_ADDR bias,
                                             GM_ADDR perTokenScale, GM_ADDR cGM, const void* tilingData)
{
    // 定义矩阵的类型和布局
    using AType = A_TYPE;
    using BType = B_TYPE;
    using BiasType = float;
    using OutType = C_TYPE;

    // 定义BlockEpilogue类型
    using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueEmpty;

    // 定义shape的形状，tuple保存 m n k batch
    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;

    // 定义scheduler类型 来自block_scheduler_policy.h
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<ProblemShape, FULL_LOAD_MODE, aLayout,
                                                                                bLayout, AType>;

    // 定义MMAD类型
    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMx<FULL_LOAD_MODE, false>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, AType, aLayout, BType, bLayout, OutType, cLayout,
                                                    BiasType, cLayout>;

    // 定义Kernel类型
    using MatmulKernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    using Params = typename MatmulKernel::Params;
    const DequantBmm::QuantBatchMatmulV3BasicAPITilingData&
        quantBmmTilingData = *static_cast<const DequantBmm::QuantBatchMatmulV3BasicAPITilingData*>(tilingData);
    const DequantBmm::QuantBatchMatmulV3BasicAPIDataParams& dataParams = quantBmmTilingData.params;
    const DequantBmm::BasicAPICubeTiling& matmulTiling = quantBmmTilingData.matmulTiling;
    const DequantBmm::SlidingWindowParams& slidingWindowParams = quantBmmTilingData.adaptiveSlidingWin;
    MatmulKernel{}(
        Params{{matmulTiling.m, matmulTiling.n, matmulTiling.k, dataParams.batchC},
               {aGM, bGM, cGM, bias, perTokenScale, scale}, // gm addr
               {matmulTiling.kBL1, matmulTiling.scaleKL1, matmulTiling.nBufferNum},
               {matmulTiling.baseM, matmulTiling.baseN, slidingWindowParams.mTailTile, slidingWindowParams.nTailTile,
                slidingWindowParams.mBaseTailSplitCnt, slidingWindowParams.nBaseTailSplitCnt,
                slidingWindowParams.mTailMain, slidingWindowParams.nTailMain},
               {dataParams.batchA1, dataParams.batchA2, dataParams.batchA3, dataParams.batchA4, dataParams.batchB1,
                dataParams.batchB2, dataParams.batchB3, dataParams.batchB4, dataParams.batchC1, dataParams.batchC2,
                dataParams.batchC3, dataParams.batchC4, dataParams.biasThreeDim, matmulTiling.baseM, matmulTiling.baseN,
                matmulTiling.baseK, matmulTiling.isBias, matmulTiling.dbL0C}});
}

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
 * \file qbmm_mx_without_batch_tensor_api_blaze.h
 * \brief
 */
#pragma once
#include "quant_batch_matmul_v3_tiling_data.h"
#include "blaze/gemm/block/block_scheduler_qbmm.h"
#include "blaze/epilogue/block/block_epilogue_empty.h"
#include "blaze/gemm/block/block_mmad_qbmm_mx.h"
#include "blaze/gemm/kernel/kernel_qbmm_mx_without_batch.h"

template <class A_TYPE, class B_TYPE, class C_TYPE, class aLayout, class bLayout, class cLayout,
          uint64_t FULL_LOAD_MODE = 0>
__aicore__ inline void QbmmMxWithoutBatchTensorApiKernel(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR scale, GM_ADDR bias,
                                                         GM_ADDR perTokenScale, GM_ADDR cGM, const void* tilingData)
{
    using AType = A_TYPE;
    using BType = B_TYPE;
    using BiasType = float;
    using OutType = C_TYPE;

    using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueEmpty;
    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<ProblemShape, FULL_LOAD_MODE, aLayout,
                                                                                bLayout, AType>;

    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMx<FULL_LOAD_MODE, false,
                                                          Blaze::Gemm::KernelMmadWithScaleMxWithoutBatch>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, AType, aLayout, BType, bLayout, OutType, cLayout,
                                                    BiasType, cLayout>;

    using MatmulKernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    using Params = typename MatmulKernel::Params;
    const DequantBmm::QuantBatchMatmulV3TensorAPIWithoutBatchTilingData&
        quantBmmTilingData = *static_cast<const DequantBmm::QuantBatchMatmulV3TensorAPIWithoutBatchTilingData*>(
            tilingData);

    MatmulKernel{}(
        Params{{quantBmmTilingData.m, quantBmmTilingData.n, quantBmmTilingData.k, 1L},
               {aGM, bGM, cGM, bias, perTokenScale, scale},
               {quantBmmTilingData.kBL1, quantBmmTilingData.scaleKL1, quantBmmTilingData.nBufferNum},
               {quantBmmTilingData.baseM, quantBmmTilingData.baseN, quantBmmTilingData.mTailTile,
                quantBmmTilingData.nTailTile, quantBmmTilingData.mBaseTailSplitCnt,
                quantBmmTilingData.nBaseTailSplitCnt, quantBmmTilingData.mTailMain, quantBmmTilingData.nTailMain},
               {quantBmmTilingData.baseM, quantBmmTilingData.baseN, quantBmmTilingData.baseK, quantBmmTilingData.isBias,
                quantBmmTilingData.dbL0C}});
}

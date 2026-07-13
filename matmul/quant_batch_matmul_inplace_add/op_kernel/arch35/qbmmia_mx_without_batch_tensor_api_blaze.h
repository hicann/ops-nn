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
 * \file qbmmia_mx_without_batch_tensor_api_blaze.h
 * \brief
 */
#pragma once
#include "quant_batch_matmul_inplace_add_tiling_data.h"
#include "blaze/gemm/block/block_scheduler_qbmm.h"
#include "blaze/epilogue/block/block_epilogue_empty.h"
#include "blaze/gemm/block/block_mmad_qbmm_mx.h"
#include "blaze/gemm/kernel/kernel_qbmm_mx_without_batch.h"

using namespace Blaze::Gemm;
template <class A_TYPE, class B_TYPE, class C_TYPE, class aLayout, class bLayout, class cLayout,
          uint64_t FULL_LOAD_MODE = 0>
__aicore__ inline void QbmmiaMxWithoutBatchTensorApiKernel(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR scale,
                                                           GM_ADDR perTokenScale, GM_ADDR cGM, const void* tilingData)
{
    using AType = A_TYPE;
    using BType = B_TYPE;
    using BiasType = float;
    using OutType = C_TYPE;

    using BlockEpilogue = Block::BlockEpilogueEmpty;
    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockScheduler = Block::BlockSchedulerQuantBatchMatmulV3<ProblemShape, FULL_LOAD_MODE, aLayout, bLayout,
                                                                   AType>;

    using DispatchPolicy = MatmulWithScaleMx<FULL_LOAD_MODE, true, KernelMmadWithScaleMxWithoutBatch>;
    using BlockMmad = Block::BlockMmad<DispatchPolicy, AType, aLayout, BType, bLayout, OutType, cLayout, BiasType,
                                       cLayout>;

    using MatmulKernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    using Params = typename MatmulKernel::Params;
    const QMMIA::QuantBatchMatmulInplaceAddTensorAPIWithoutBatchTilingData&
        qbmmiaTilingData = *static_cast<const QMMIA::QuantBatchMatmulInplaceAddTensorAPIWithoutBatchTilingData*>(
            tilingData);

    MatmulKernel{}(Params{{qbmmiaTilingData.m, qbmmiaTilingData.n, qbmmiaTilingData.k, 1L},
                          {aGM, bGM, cGM, nullptr, perTokenScale, scale},
                          {qbmmiaTilingData.kBL1, qbmmiaTilingData.scaleKL1, qbmmiaTilingData.nBufferNum},
                          {qbmmiaTilingData.baseM, qbmmiaTilingData.baseN, qbmmiaTilingData.mTailTile,
                           qbmmiaTilingData.nTailTile, qbmmiaTilingData.mBaseTailSplitCnt,
                           qbmmiaTilingData.nBaseTailSplitCnt, qbmmiaTilingData.mTailMain, qbmmiaTilingData.nTailMain},
                          {qbmmiaTilingData.baseM, qbmmiaTilingData.baseN, qbmmiaTilingData.baseK,
                           qbmmiaTilingData.isBias, qbmmiaTilingData.dbL0C}});
}

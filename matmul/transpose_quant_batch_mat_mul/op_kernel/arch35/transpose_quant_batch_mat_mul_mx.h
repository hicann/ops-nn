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
 * \file transpose_quant_batch_mat_mul_mx.h
 * \brief
 */
#pragma once
#include "blaze/gemm/block/block_scheduler_qbmm.h"
#include "blaze/epilogue/block/block_epilogue_empty.h"
#include "blaze/gemm/block/block_mmad_qbmm_mx.h"
#include "blaze/gemm/kernel/kernel_tqbmm_mx.h"

template <class A_TYPE, class B_TYPE, class SCALE_TYPE, class C_TYPE, class BIAS_TYPE, class aLayout, class bLayout,
          class cLayout, uint64_t FULL_LOAD_MODE = 0, uint64_t PERM_X1 = 0, uint64_t NON_CONTIGUOUS_TYPE = 0>
__aicore__ inline void TqbmmMxTensorApiKernel(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR scale, GM_ADDR bias,
                                              GM_ADDR perTokenScale, GM_ADDR cGM,
                                              const BatchMatMulV3TilingData& tilingData)
{
    using AType = A_TYPE;
    using BType = B_TYPE;
    using BiasType = BIAS_TYPE;
    using X2ScaleType = SCALE_TYPE;
    using OutType = C_TYPE;
    using L0CType = typename AscendC::GetMmDstType<AType>::Type;

    using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueEmpty;

    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;

    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<ProblemShape, FULL_LOAD_MODE, aLayout,
                                                                                bLayout, AType>;

    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMx<
        FULL_LOAD_MODE, false, Blaze::Gemm::KernelMmadMultiBlockTQBMM, Blaze::Gemm::L0C2UB_MODE_NONE,
        (PERM_X1 == 1 ? static_cast<uint64_t>(Blaze::Gemm::NoContiguousType::NON_CONTIGUOUS_TYPE_PERM_X1) : 0)>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, AType, aLayout, BType, bLayout, OutType, cLayout,
                                                    BiasType, cLayout>;

    using MatmulKernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    using Params = typename MatmulKernel::Params;
    const auto& tCubeTiling = tilingData.matMulTilingData.tCubeTiling;
    Params params = {{tCubeTiling.M, tCubeTiling.N, tCubeTiling.Ka, tilingData.cBatchDimAll},
                     {aGM, bGM, cGM, bias, perTokenScale, scale},
                     {tilingData.kL1, tilingData.kL1, tilingData.l1BufferNum},
                     {static_cast<int64_t>(tCubeTiling.baseM), static_cast<int64_t>(tCubeTiling.baseN),
                      tilingData.matMulTilingData.mTailCnt, tilingData.matMulTilingData.nTailCnt,
                      tilingData.matMulTilingData.mBaseTailSplitCnt, tilingData.matMulTilingData.nBaseTailSplitCnt,
                      tilingData.matMulTilingData.mTailMain, tilingData.matMulTilingData.nTailMain},
                     {tilingData.aBatchDim0, tilingData.aBatchDim1, tilingData.aBatchDim2, tilingData.aBatchDim3,
                      tilingData.bBatchDim0, tilingData.bBatchDim1, tilingData.bBatchDim2, tilingData.bBatchDim3,
                      tilingData.cBatchDim0, tilingData.cBatchDim1, tilingData.cBatchDim2, tilingData.cBatchDim3, 0,
                      static_cast<uint32_t>(tCubeTiling.baseM), static_cast<uint32_t>(tCubeTiling.baseN),
                      static_cast<uint32_t>(tCubeTiling.baseK), static_cast<uint32_t>(tCubeTiling.isBias),
                      static_cast<uint32_t>(tCubeTiling.dbL0C), 1}};
    MatmulKernel mm;
    mm(params);
}

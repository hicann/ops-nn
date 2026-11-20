/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file transpose_batch_mat_mul_iterbatch_kernel.h
 * \brief TBMM TensorAPI(Blaze) iterbatch wrapper, 对应batch_mat_mul_v3_iterbatch.h:
 *        等batch场景L1装载iterBatchL1个batch + L0按iterBatchL0分批mmad,
 *        分块计算沿用BMM，适配ND的MBK A和非连续B；输出支持BMN/MBN
 */

#ifndef TRANSPOSE_BATCH_MAT_MUL_ITERBATCH_KERNEL_H
#define TRANSPOSE_BATCH_MAT_MUL_ITERBATCH_KERNEL_H

#include "transpose_batch_mat_mul_tiling_key_public.h"
#include "transpose_batch_mat_mul_tiling_data.h"
#include "blaze/gemm/kernel/kernel_universal.h"
#include "blaze/gemm/block/block_mmad.h"
#include "blaze/gemm/block/block_scheduler_matmul_iterbatch.h"
#include "blaze/epilogue/block/block_epilogue_iterbatch.h"
#include "blaze/epilogue/block/block_epilogue_empty.h"
#include "blaze/gemm/policy/dispatch_policy.h"

namespace TransposeBatchMatMulAdvanced {

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class A_LAYOUT, class B_LAYOUT, class C_LAYOUT,
          uint64_t NON_CONTIGUOUS_TYPE = TRANSPOSE_BATCH_MAT_MUL_CONTIGUOUS,
          Blaze::Gemm::MatMulL0C2Out L0C2OUT_MODEL = Blaze::Gemm::MatMulL0C2Out::ON_THE_FLY>
__aicore__ inline void TBMMIterBatchTensorKernel(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR cGM,
                                                 GM_ADDR workspaceGM,
                                                 const BatchMatMulV3IterBatchBasicTilingData& tilingData)
{
    // 定义矩阵的类型和布局
    using AType = A_TYPE;
    using BType = B_TYPE;
    using BiasType = BIAS_TYPE;
    using OutType = C_TYPE;

    using LayoutA = A_LAYOUT;
    using LayoutB = B_LAYOUT;
    using LayoutC = C_LAYOUT;
    using LayoutBias = LayoutC;

    // m, n, k, batch；计算流水与BMM共用。
    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;

    // 定义scheduler类型
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerMatmulIterBatch<ProblemShape>;

    using DispatchPolicy = Blaze::Gemm::MatmulIterBatch<L0C2OUT_MODEL, 0, NON_CONTIGUOUS_TYPE>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, AType, LayoutA, BType, LayoutB, OutType, LayoutC,
                                                    BiasType, LayoutBias>;

    // MIX 场景复用 BMM 的 AIV ND 搬出，ON_THE_FLY 场景由 AIC 直出 GM。
    using BlockEpilogue = AscendC::Std::conditional_t<L0C2OUT_MODEL != Blaze::Gemm::MatMulL0C2Out::ON_THE_FLY,
                                                      Blaze::Epilogue::Block::BlockEpilogueIterbatch<OutType, OutType>,
                                                      Blaze::Gemm::Block::BlockEpilogueEmpty>;

    // 定义Kernel类型
    using MatmulKernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    using Params = typename MatmulKernel::Params;
    using SchedulerParams = typename BlockScheduler::Params;

    SchedulerParams schParams = {tilingData.baseM,
                                 tilingData.baseN,
                                 tilingData.baseK,
                                 tilingData.iterBatchL1,
                                 tilingData.iterBatchL0,
                                 static_cast<uint8_t>(tilingData.mmadParam),
                                 static_cast<uint32_t>(tilingData.l2CacheDisable)};

    Params params = {
        {tilingData.m, tilingData.n, tilingData.k, tilingData.b}, // shape
        {aGM, bGM, cGM, biasGM, tilingData.m, tilingData.n, tilingData.k, tilingData.baseM, tilingData.baseN,
         tilingData.baseK, tilingData.iterBatchL1, tilingData.iterBatchL0}, // mmad params
        {},                                                                 // epilogue params
        schParams};

    if constexpr (L0C2OUT_MODEL != Blaze::Gemm::MatMulL0C2Out::ON_THE_FLY) {
        params.epilogueParams = {cGM};
    }

    MatmulKernel mm;
    mm(params);
}
} // namespace TransposeBatchMatMulAdvanced
#endif // TRANSPOSE_BATCH_MAT_MUL_ITERBATCH_KERNEL_H

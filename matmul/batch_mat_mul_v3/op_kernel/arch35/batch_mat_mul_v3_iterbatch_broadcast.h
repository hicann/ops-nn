/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include "blaze/gemm/kernel/kernel_batch_matmul_iterbatch_broadcast.h"
#include "blaze/gemm/block/block_mmad.h"
#include "blaze/gemm/block/block_mmad_iterbatch_broadcast.h"
#include "blaze/gemm/block/block_scheduler_iterbatch_broadcast.h"
#include "blaze/gemm/policy/dispatch_policy.h"

template <
    class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE,
    class A_LAYOUT, class B_LAYOUT, class C_LAYOUT,
    bool A_BC, bool B_BC>
__aicore__ inline void BatchMatMulIterBatchBroadcastKernel(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR cGM, GM_ADDR workspaceGM,
    const BatchMatMulV3TilingData& tilingData)
{
    using AType = A_TYPE;
    using BType = B_TYPE;
    using BiasType = BIAS_TYPE;
    using OutType = C_TYPE;

    using LayoutA = A_LAYOUT;
    using LayoutB = B_LAYOUT;
    using LayoutC = C_LAYOUT;
    using LayoutBias = LayoutC;

    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;

    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerIterBatchBroadcast<ProblemShape>;

    using DispatchPolicy = Blaze::Gemm::MatmulIterBatchBroadcast<A_BC, B_BC>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<
        DispatchPolicy, AType, LayoutA, BType, LayoutB, OutType, LayoutC, BiasType, LayoutBias>;

    using BlockEpilogue = Blaze::Gemm::Block::BlockEpilogueEmpty;

    using MatmulKernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    using Params = typename MatmulKernel::Params;
    using SchedulerParams = typename BlockScheduler::Params;

    SchedulerParams schParams = {
        static_cast<uint32_t>(tilingData.matMulTilingData.tCubeTiling.baseM),
        static_cast<uint32_t>(tilingData.matMulTilingData.tCubeTiling.baseN),
        static_cast<uint32_t>(tilingData.matMulTilingData.tCubeTiling.baseK),
        static_cast<uint32_t>(tilingData.iterBatchL1),
        static_cast<uint32_t>(tilingData.iterBatchL0),
        static_cast<uint32_t>(tilingData.broadcastAxisA),
        static_cast<uint32_t>(tilingData.broadcastAxisB),
        static_cast<uint32_t>(tilingData.aBatchDim0),
        static_cast<uint32_t>(tilingData.aBatchDim1),
        static_cast<uint32_t>(tilingData.aBatchDim2),
        static_cast<uint32_t>(tilingData.aBatchDim3),
        static_cast<uint32_t>(tilingData.bBatchDim0),
        static_cast<uint32_t>(tilingData.bBatchDim1),
        static_cast<uint32_t>(tilingData.bBatchDim2),
        static_cast<uint32_t>(tilingData.bBatchDim3),
        static_cast<uint32_t>(tilingData.cBatchDim0),
        static_cast<uint32_t>(tilingData.cBatchDim1),
        static_cast<uint32_t>(tilingData.cBatchDim2),
        static_cast<uint32_t>(tilingData.cBatchDim3),
        static_cast<uint8_t>(tilingData.isHf32)};

    Params params = {
        {tilingData.matMulTilingData.tCubeTiling.M, tilingData.matMulTilingData.tCubeTiling.N,
         tilingData.matMulTilingData.tCubeTiling.Ka, tilingData.cBatchDimAll},
        {aGM, bGM, cGM, biasGM},
        {},
        schParams};

    MatmulKernel mm;
    mm(params);
}

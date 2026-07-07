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
 * \file batch_mat_mul_v3_mergebatch_basicapi_cmct.h
 * \brief
 */
#pragma once
#include "cmct/kernel/kernel_matmul_merge_batch.h"
#include "cmct/epilogue/block_epilogue_mergebatch.h"
#include "cmct/epilogue/fusion/merge_batch_fusion.h"
#include "batch_mat_mul_v3_mergebatch_basicapi_block_scheduler.h"

using namespace Cmct;
using namespace Cmct::Gemm;

constexpr uint64_t MERGE_BATCH_FIXPIPE_1V2 = 2UL;

template <uint64_t OpType, class OutType>
struct MergeBatchFusionSelector;

template <class OutType>
struct MergeBatchFusionSelector<OP_TYPE_ADD, OutType> {
    using type = Block::MergeBatchFusionAdd<OutType, OutType>;
};

template <class OutType>
struct MergeBatchFusionSelector<OP_TYPE_MUL, OutType> {
    using type = Block::MergeBatchFusionMul<OutType, OutType>;
};

template <class BlockMmad, class DispatchPolicy, class OutType, uint64_t FUSED_OPTYPE>
__aicore__ inline void RunMergeBatchFusionKernel(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR cGM,
    const BatchMatMulV3MergeBatchBasicTilingData& tilingDataGM, GM_ADDR x3GM)
{
    using ProblemShape = MatmulShape;
    using BlockScheduler = BuiltInMergeBatchScheduler;
    using FusionOp = typename MergeBatchFusionSelector<FUSED_OPTYPE, OutType>::type;
    using BlockEpilogue = Block::BlockEpilogueMergeBatch<OutType, OutType, FusionOp>;
    using MatmulKernel = Kernel::KernelMatMulMergeBatch<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    using Params = typename MatmulKernel::Params;
    using EpilogueParams = typename BlockEpilogue::Params;

    EpilogueParams epilogueParams;
    epilogueParams.cGmAddr = cGM;
    epilogueParams.fusionParams.inputGmAddr = x3GM;
    epilogueParams.fusionParams.x3BatchBroadcast = (x3GM != nullptr && tilingDataGM.b > 1 && tilingDataGM.batchX3 == 1);
    epilogueParams.fusionParams.x3M = tilingDataGM.m;
    Params params = {
        {tilingDataGM.m, tilingDataGM.n, tilingDataGM.k, tilingDataGM.b},
        {aGM, bGM, cGM, biasGM},
        epilogueParams,
        {&tilingDataGM}
    };
    MatmulKernel mm;
    mm(params);
}

template <class BlockMmad>
__aicore__ inline void RunMergeBatchBasicKernel(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR cGM,
    const BatchMatMulV3MergeBatchBasicTilingData& tilingDataGM)
{
    using ProblemShape = MatmulShape;
    using BlockScheduler = BuiltInMergeBatchScheduler;
    using BlockEpilogue = Block::BlockEpilogueEmpty;
    using MatmulKernel = Kernel::KernelMatMulMergeBatch<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    using Params = typename MatmulKernel::Params;

    Params params = {
        {tilingDataGM.m, tilingDataGM.n, tilingDataGM.k, tilingDataGM.b},
        {aGM, bGM, cGM, biasGM},
        {},
        {&tilingDataGM}
    };
    MatmulKernel mm;
    mm(params);
}

template <
    class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class A_LAYOUT, class B_LAYOUT, class C_LAYOUT,
    uint64_t FUSED_OPTYPE = 0, uint64_t FIXPIPE_OPT = 0>
__aicore__ inline void BatchMatMulActMergeBatchKernel(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM,
    GM_ADDR cGM, GM_ADDR workspaceGM, const BatchMatMulV3MergeBatchBasicTilingData& tilingDataGM,
    GM_ADDR x3GM = nullptr)
{
    using L1TileShape = AscendC::Shape<_0, _0, _0>;
    using L0TileShape = AscendC::Shape<_0, _0, _0>;
    using AType = A_TYPE;
    using BType = B_TYPE;
    using BiasType = BIAS_TYPE;
    using OutType = C_TYPE;
    using LayoutA = A_LAYOUT;
    using LayoutB = B_LAYOUT;
    using LayoutC = C_LAYOUT;
    using BlockScheduler = BuiltInMergeBatchScheduler;
    using DispatchPolicy = MatmulMergeBatch<AscendC::Shape<_0, _0, _0, _0>, FUSED_OPTYPE>;
    using BlockMmad = Block::BlockMmadBuilder<
        AType, LayoutA, BType, LayoutB, OutType, LayoutC, BiasType, LayoutC,
        L1TileShape, L0TileShape, BlockScheduler, DispatchPolicy>;

    (void)workspaceGM;
    if (biasGM != nullptr) {
        return;
    }

    if constexpr ((FUSED_OPTYPE == OP_TYPE_ADD || FUSED_OPTYPE == OP_TYPE_MUL) &&
                  FIXPIPE_OPT == MERGE_BATCH_FIXPIPE_1V2) {
        RunMergeBatchFusionKernel<BlockMmad, DispatchPolicy, OutType, FUSED_OPTYPE>(
            aGM, bGM, biasGM, cGM, tilingDataGM, x3GM);
    } else {
        RunMergeBatchBasicKernel<BlockMmad>(aGM, bGM, biasGM, cGM, tilingDataGM);
    }
}

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
 * \file batch_mat_mul_v3_asw_broadcast.h
 * \brief
 */
#ifndef BATCH_MAT_MUL_ASW_BROADCAST_TENSORAPI_H
#define BATCH_MAT_MUL_ASW_BROADCAST_TENSORAPI_H

#include "blaze/gemm/kernel/kernel_batch_matmul_broadcast.h"
#include "blaze/gemm/block/block_mmad.h"
#include "blaze/gemm/block/block_mmad_matmul_basic.h"
#include "blaze/gemm/block/block_scheduler_matmul_basic.h"
#include "blaze/gemm/policy/dispatch_policy.h"

namespace BatchMatMulV3Advanced {

template <
    class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class A_LAYOUT, class B_LAYOUT, class C_LAYOUT,
    uint64_t FULL_LOAD_MODE = 0, uint64_t FUSED_OP_TYPE = 0>
__aicore__ inline void BatchMatMulBroadcastKernel(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR cGM, GM_ADDR workspaceGM,
    const BatchMatMulV3TilingData& tilingData)
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

    // 定义shape的形状，tuple保存 m n k batch
    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;

    // 定义scheduler类型
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerMatmulBasic<ProblemShape, FULL_LOAD_MODE>;

    // 定义MMAD类型
    using DispatchPolicy = Blaze::Gemm::MatmulMultiBlockBasic<
        FULL_LOAD_MODE, FUSED_OP_TYPE, Blaze::Gemm::KernelMmadMultiBlockBmmBroadcast>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<
        DispatchPolicy, AType, LayoutA, BType, LayoutB, OutType, LayoutC, BiasType, LayoutBias>;

    // 定义BlockEpilogue类型
    using BlockEpilogue = Blaze::Gemm::Block::BlockEpilogueEmpty;

    // 定义Kernel类型
    using MatmulKernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    using Params = typename MatmulKernel::Params;
    Params params = {
        {tilingData.matMulTilingData.tCubeTiling.M, tilingData.matMulTilingData.tCubeTiling.N,
         tilingData.matMulTilingData.tCubeTiling.Ka, tilingData.cBatchDimAll}, // shape
        {aGM, bGM, cGM, biasGM},                                               // gm addr
        {},                                                                    // epilogue args
        {tilingData.mL1,                                                       // block scheduler args
         tilingData.nL1,
         tilingData.kL1,
         static_cast<uint32_t>(tilingData.matMulTilingData.tCubeTiling.baseM),
         static_cast<uint32_t>(tilingData.matMulTilingData.tCubeTiling.baseN),
         static_cast<uint32_t>(tilingData.matMulTilingData.tCubeTiling.baseK),
         tilingData.matMulTilingData.mTailCnt,
         tilingData.matMulTilingData.nTailCnt,
         tilingData.matMulTilingData.mBaseTailSplitCnt,
         tilingData.matMulTilingData.nBaseTailSplitCnt,
         tilingData.matMulTilingData.mTailMain,
         tilingData.matMulTilingData.nTailMain,
         static_cast<uint8_t>(tilingData.matMulTilingData.isHf32),
         static_cast<uint8_t>(tilingData.l1BufferNum),
         static_cast<uint8_t>(tilingData.matMulTilingData.tCubeTiling.dbL0C),
         static_cast<uint8_t>(tilingData.ubDB),
         tilingData.matMulTilingData.l2CacheDisable,
         tilingData.sliceM,
         tilingData.srcNdStride,
         tilingData.innerBatch},
        {tilingData.aBatchDim0, tilingData.bBatchDim0, tilingData.aBatchDim1, tilingData.bBatchDim1,
         tilingData.cBatchDim1, tilingData.aBatchDim2, tilingData.bBatchDim2, tilingData.cBatchDim2,
         tilingData.aBatchDim3, tilingData.bBatchDim3, tilingData.cBatchDim3, tilingData.biasBatchDimAll}};
    MatmulKernel mm;
    mm(params);
}
} // namespace BatchMatMulV3Advanced
#endif
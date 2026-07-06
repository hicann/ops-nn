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
 * \file mat_mul_streamk.h
 * \brief
 */
#pragma once

#include "blaze/gemm/kernel/kernel_matmul_streamk.h"
#include "blaze/gemm/block/block_mmad_matmul_streamk.h"
#include "blaze/gemm/block/block_scheduler_matmul_streamk.h"
#include "blaze/gemm/policy/dispatch_policy.h"

namespace MatmulV3Advanced {
template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class A_LAYOUT, class B_LAYOUT,
          Blaze::Gemm::MatMulL0C2Out MATMUL_L0C2OUT, uint64_t FUSED_OP_TYPE = 0>
__aicore__ inline void MatMulStreamKKernel(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR cGM, GM_ADDR workspaceGM,
                                           const MatMulV3BasicTilingData& tilingData, int64_t batch = 0)
{
    // 定义矩阵的类型和布局
    using AType = A_TYPE;
    using BType = B_TYPE;
    using BiasType = BIAS_TYPE;
    using OutType = C_TYPE;

    using LayoutA = A_LAYOUT;
    using LayoutB = B_LAYOUT;
    using LayoutC = AscendC::Te::NDExtLayoutPtn;

    if (batch > 1) {
        return;
    }
    // 定义shape的形状，tuple保存 m n k batch
    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;

    // 定义scheduler类型 来自block_scheduler_policy.h
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerMatmulStreamK<ProblemShape>;

    // 定义MMAD类型
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<
        Blaze::Gemm::MatmulMultiBlockWithStreamK<MATMUL_L0C2OUT, FUSED_OP_TYPE>, AType, LayoutA, BType, LayoutB,
        OutType, LayoutC, BiasType, LayoutC>;
    // 定义Fusion类型
    using FusionOp = Blaze::Gemm::Block::DefaultFusion<OutType, OutType>;

    // 定义BlockEpilogue类型
    using BlockEpilogue = Blaze::Gemm::Block::BlockEpilogueMatmulStreamK<
        float, OutType, Blaze::Gemm::MatmulMultiBlockWithStreamK<MATMUL_L0C2OUT, FUSED_OP_TYPE>>;

    // 定义Kernel类型
    using MatmulKernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    using Params = typename MatmulKernel::Params;
    Params params = {
        {tilingData.m, tilingData.n, tilingData.k, batch}, // shape
        {aGM, bGM, cGM, biasGM, nullptr, workspaceGM, tilingData.mL1, tilingData.nL1, tilingData.kL1, tilingData.baseM,
         tilingData.baseN, tilingData.baseK, tilingData.l1BufferNum, tilingData.l0cDB}, // gm addr
        {cGM, workspaceGM},                                                             // epilogue args
        {tilingData.usedCoreNum, tilingData.baseM, tilingData.baseN, tilingData.baseK, tilingData.skSingleCoreK,
         tilingData.kL1, tilingData.isHf32, static_cast<uint32_t>(tilingData.l2CacheDisable)}}; // schedule params
    MatmulKernel mm;
    mm(params);
}
} // namespace MatmulV3Advanced
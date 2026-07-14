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
 * \file mat_mul_al1_full_load.h
 * \brief
 */
#ifndef MAT_MUL_FULL_LOAD_TENSORAPI_H
#define MAT_MUL_FULL_LOAD_TENSORAPI_H

#include "blaze/gemm/kernel/kernel_matmul_al1_full_load.h"
#include "blaze/gemm/block/block_mmad.h"
#include "blaze/gemm/block/block_mmad_matmul_al1_full_load.h"
#include "blaze/gemm/block/block_scheduler_matmul_basic.h"
#include "blaze/gemm/policy/dispatch_policy.h"

namespace MatmulV3Advanced {

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class A_LAYOUT, class B_LAYOUT, class C_LAYOUT,
          uint64_t FULL_LOAD_MODE = 0, uint64_t FUSED_OP_TYPE = 0>
__aicore__ inline void MatMulAL1FullLoadKernel(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR cGM,
                                               GM_ADDR workspaceGM, const MatMulV3BasicTilingData& tilingData,
                                               int64_t batch = 0)
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

    static constexpr bool IS_FP32 = (AscendC::Std::is_same_v<BType, float>);
    static constexpr bool IS_ND_FORMAT = !(Blaze::Gemm::IsWeightNz<LayoutB>::value);
    // 定义scheduler类型
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerMatmulBasic<ProblemShape, FULL_LOAD_MODE, IS_FP32,
                                                                         IS_ND_FORMAT>;

    // 定义MMAD类型
    using DispatchPolicy = Blaze::Gemm::MatmulMultiBlockAFullLoad<FULL_LOAD_MODE, FUSED_OP_TYPE,
                                                                  Blaze::Gemm::KernelMmadMultiBlockAFullLoad>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, AType, LayoutA, BType, LayoutB, OutType, LayoutC,
                                                    BiasType, LayoutBias>;

    // 定义BlockEpilogue类型
    using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueEmpty;

    // 定义Kernel类型
    using MatmulKernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    using Params = typename MatmulKernel::Params;
    Params params = {{tilingData.m, tilingData.n, tilingData.k, batch}, // shape
                     {aGM, bGM, cGM, biasGM, tilingData.k, tilingData.mL1, tilingData.kL1, tilingData.baseM,
                      tilingData.baseN, tilingData.baseK, tilingData.l1BufferNum, tilingData.l0cDB},
                     {}, // epilogue args
                     {tilingData.mL1, tilingData.nL1, tilingData.kL1, tilingData.baseM, tilingData.baseN,
                      tilingData.baseK, tilingData.mTailCnt, tilingData.nTailCnt, tilingData.mBaseTailSplitCnt,
                      tilingData.nBaseTailSplitCnt, tilingData.mTailMain, tilingData.nTailMain, tilingData.isHf32,
                      static_cast<uint32_t>(tilingData.l2CacheDisable), tilingData.sliceM, tilingData.srcNdStride,
                      tilingData.innerBatch}}; // scheduler params
    MatmulKernel mm;
    mm(params);
}
} // namespace MatmulV3Advanced
#endif

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
 * \file transpose_batch_mat_mul_asw_kernel.h
 * \brief
 */
#ifndef TRANSPOSE_BATCH_MAT_MUL_ASW_SPLIT_K_KERNEL_H
#define TRANSPOSE_BATCH_MAT_MUL_ASW_SPLIT_K_KERNEL_H

#include "transpose_batch_mat_mul_tiling_data.h"
#include "blaze/gemm/kernel/kernel_universal.h"
#include "blaze/gemm/block/block_mmad.h"
#include "blaze/gemm/block/block_mmad_matmul_basic.h"
#include "blaze/gemm/block/block_scheduler_matmul_basic.h"
#include "blaze/epilogue/block/block_epilogue_empty.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "blaze/gemm/utils/layout_utils.h"
#include "blaze/gemm/utils/common_utils.h"

namespace TransposeBatchMatMulAdvanced {

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class A_LAYOUT, class B_LAYOUT, class C_LAYOUT,
          uint64_t NON_CONTIGUOUS_TYPE = TRANSPOSE_BATCH_MAT_MUL_C_TRANSPOSE, uint64_t FULL_LOAD_MODE = 0,
          uint64_t FUSED_OP_TYPE = 0, bool IS_SPLIT_SINGLE_CORE_K = true>
__aicore__ inline void TBMMSplitKTensorKernel(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR scalesGM, GM_ADDR cGM,
                                              GM_ADDR workspaceGM,
                                              const TransposeBatchMatMulBasicTilingData& tilingData, int64_t batch = 0)
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

    // 定义shape的形状，tuple保存 m n k batch splitBatch
    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t, int64_t>;

    static constexpr bool isFp32 = (AscendC::Std::is_same_v<BType, float>);
    static constexpr bool isNDFormat = !(Blaze::Gemm::IsWeightNz<LayoutB>::value);
    // 定义scheduler类型
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerMatmulBasic<ProblemShape, FULL_LOAD_MODE, isFp32,
                                                                         isNDFormat>;
    // 定义MMAD类型
    using DispatchPolicy = Blaze::Gemm::MatmulMultiBlockBasicSplitK<
        FULL_LOAD_MODE, IS_SPLIT_SINGLE_CORE_K, Blaze::Gemm::KernelMmadMultiBlockBasic, NON_CONTIGUOUS_TYPE>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, AType, LayoutA, BType, LayoutB, OutType, LayoutC,
                                                    BiasType, LayoutBias>;

    // 定义BlockEpilogue类型
    using BlockEpilogue = Blaze::Gemm::Block::BlockEpilogueEmpty;

    // 定义Kernel类型
    using MatmulKernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    using Params = typename MatmulKernel::Params;
    const auto& bmmTiling = tilingData.matMulTilingData;
    const auto& matmulTiling = bmmTiling.matMulTilingData;
    Params params = {
        {matmulTiling.m, matmulTiling.n, matmulTiling.k, bmmTiling.batchDimAll, tilingData.batchSplitFactor},
        {aGM, bGM, cGM, biasGM, nullptr, nullptr, matmulTiling.mL1, matmulTiling.nL1, matmulTiling.kL1,
         matmulTiling.baseM, matmulTiling.baseN, matmulTiling.baseK, matmulTiling.l1BufferNum, matmulTiling.l0cDB,
         matmulTiling.k, 0, matmulTiling.splitSingleCoreKBase}, // blockMmad args
        {},                                                     // blockEpilogue args
        {matmulTiling.mL1, matmulTiling.nL1, matmulTiling.kL1, matmulTiling.baseM, matmulTiling.baseN,
         matmulTiling.baseK, matmulTiling.mTailCnt, matmulTiling.nTailCnt, matmulTiling.mBaseTailSplitCnt,
         matmulTiling.nBaseTailSplitCnt, matmulTiling.mTailMain, matmulTiling.nTailMain, matmulTiling.mmadParam,
         static_cast<uint32_t>(matmulTiling.l2CacheDisable), matmulTiling.sliceM, matmulTiling.srcNdStride,
         matmulTiling.innerBatch, matmulTiling.aswWindowLen}};
    MatmulKernel mm;
    mm(params);
}
} // namespace TransposeBatchMatMulAdvanced
#endif

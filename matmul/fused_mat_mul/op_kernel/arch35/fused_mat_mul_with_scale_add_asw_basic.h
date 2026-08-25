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
 * \file fused_mat_mul_with_scale_add_asw_basic.h
 * \brief Blaze assembly for the BF16/FP16 FusedMatMul scale/add template on DAV_3510.
 */

#pragma once

#include "blaze/epilogue/block/block_epilogue_fmm_with_scale_add.h"
#include "blaze/gemm/block/block_scheduler_matmul_basic.h"
#include "blaze/gemm/kernel/kernel_matmul_with_scale_add.h"
#include "fused_mat_mul_tiling_data.h"

namespace FusedMatMulAdvanced {

template <typename ElementType>
__aicore__ inline void FusedMatMulWithScaleAddAswBasicKernel(GM_ADDR x1Gm, GM_ADDR x2Gm, GM_ADDR x3Gm, GM_ADDR yGm,
                                                             GM_ADDR workspaceGm,
                                                             const FusedMatMulTilingData& tilingData)
{
    using AccType = float;
    static constexpr bool enable2UB = AscendC::IsSameType<AccType, float>::value;
    // x3/output reuse the UB space after the FP32 accumulator, so accumulator UB ping-pong stays disabled.
    static constexpr uint8_t singleUbBuffer = 1U;
    using Layout = AscendC::Te::NDExtLayoutPtn;
    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    using DispatchPolicy = Blaze::Gemm::MatmulMultiBlockFixpipeOpti<Blaze::Gemm::ND_ALIG_1V2_FIXPIPE, 0,
                                                                    Blaze::Gemm::KernelMmadFmmWithScaleAdd>;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerMatmulBasic<ProblemShape, Blaze::Gemm::NONE_FULL_LOAD_MODE,
                                                                         false, true>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, ElementType, Layout, ElementType, Layout, AccType,
                                                    Layout, ElementType, Layout>;
    using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueFmmWithScaleAdd<DispatchPolicy, ElementType>;
    using MatmulKernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    using KernelParams = typename MatmulKernel::Params;

    const auto& matmulTiling = tilingData.matMulTilingData;
    const auto& matmulBaseTiling = matmulTiling.matMulTilingData;

    KernelParams params = {
        {matmulBaseTiling.m, matmulBaseTiling.n, matmulBaseTiling.k, matmulTiling.batchDimAll},
        {x1Gm, x2Gm, nullptr, nullptr, nullptr, workspaceGm, matmulBaseTiling.k, matmulBaseTiling.mL1,
         matmulBaseTiling.nL1, matmulBaseTiling.kL1, matmulBaseTiling.baseM, matmulBaseTiling.baseN,
         matmulBaseTiling.baseK, matmulBaseTiling.l1BufferNum, matmulBaseTiling.l0cDB, enable2UB, singleUbBuffer},
        {x3Gm, yGm, tilingData.alpha, tilingData.beta},
        {matmulBaseTiling.mL1, matmulBaseTiling.nL1, matmulBaseTiling.kL1, matmulBaseTiling.baseM,
         matmulBaseTiling.baseN, matmulBaseTiling.baseK, matmulBaseTiling.mTailCnt, matmulBaseTiling.nTailCnt,
         matmulBaseTiling.mBaseTailSplitCnt, matmulBaseTiling.nBaseTailSplitCnt, matmulBaseTiling.mTailMain,
         matmulBaseTiling.nTailMain, matmulBaseTiling.mmadParam, static_cast<uint32_t>(matmulBaseTiling.l2CacheDisable),
         matmulBaseTiling.sliceM, matmulBaseTiling.srcNdStride, matmulBaseTiling.innerBatch}};

    MatmulKernel kernel;
    kernel(params);
}

} // namespace FusedMatMulAdvanced

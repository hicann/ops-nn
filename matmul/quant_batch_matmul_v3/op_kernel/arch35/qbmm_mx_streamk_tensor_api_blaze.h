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
 * \file qbmm_mx_streamk_tensor_api_blaze.h
 * \brief Quantized batch matrix multiplication using StreamK pipeline.
 */
#pragma once

#include "blaze/gemm/kernel/kernel_qbmm_streamk.h"
#include "blaze/epilogue/block/block_epilogue_matmul_streamk.h"
#include "blaze/gemm/block/block_scheduler_matmul_streamk.h"
#include "blaze/gemm/block/block_mmad_qbmm_mx.h"

template <
    class A_TYPE, class B_TYPE, class C_TYPE, class aLayout, class bLayout, class cLayout, uint64_t FULL_LOAD_MODE = 0>
__aicore__ inline void QbmmMxStreamKBasicApiKernel(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR scale, GM_ADDR bias, GM_ADDR perTokenScale, GM_ADDR cGM,
    GM_ADDR workspaceGm, const void* tilingData)
{
    using AType = A_TYPE;
    using BType = B_TYPE;
    using BiasType = float;
    using OutType = C_TYPE;
    using WorkspaceType = float;

    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;

    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMx<
        FULL_LOAD_MODE, false, Blaze::Gemm::KernelQbmmMultiBlockStreamK>;
    using EpilogueDispatchPolicy = Blaze::Gemm::MatmulMultiBlockWithStreamK<>;
    using BlockEpilogue =
        Blaze::Gemm::Block::BlockEpilogueMatmulStreamK<WorkspaceType, OutType, EpilogueDispatchPolicy>;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerMatmulStreamK<ProblemShape>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<
        DispatchPolicy, AType, aLayout, BType, bLayout, OutType, cLayout, BiasType, cLayout>;
    using MatmulKernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    using Params = typename MatmulKernel::Params;

    const DequantBmm::QuantBatchMatmulV3StreamKBasicAPITilingData* quantBmmTilingData_;
    quantBmmTilingData_ = static_cast<const DequantBmm::QuantBatchMatmulV3StreamKBasicAPITilingData*>(tilingData);
    DequantBmm::BasicAPICubeTiling matmulTiling = quantBmmTilingData_->matmulTiling;

    typename MatmulKernel::QBMMStreamKParams qbmmParams{
        matmulTiling.scaleKL1,
        static_cast<uint32_t>(matmulTiling.dbL0C)};
    GM_ADDR biasGm = matmulTiling.isBias == 0U ? nullptr : bias;

    Params params{
        {matmulTiling.m, matmulTiling.n, matmulTiling.k, quantBmmTilingData_->params.batchC},
        {aGM, bGM, cGM, biasGm, perTokenScale, scale},
        {cGM, workspaceGm},
        {AscendC::GetBlockNum(), matmulTiling.baseM, matmulTiling.baseN, matmulTiling.baseK,
         quantBmmTilingData_->streamKTiling.singleCoreK, quantBmmTilingData_->streamKTiling.kL1},
        qbmmParams};

    MatmulKernel qbmm;
    qbmm(params);
}

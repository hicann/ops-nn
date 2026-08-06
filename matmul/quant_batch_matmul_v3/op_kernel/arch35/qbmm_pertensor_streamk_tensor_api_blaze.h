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
 * \file qbmm_pertensor_streamk_tensor_api_blaze.h
 * \brief Quantized batch matrix multiplication using StreamK and AIV vector dequant.
 */
#pragma once

#include "blaze/gemm/kernel/kernel_qbmm_pertensor_streamk.h"
#include "blaze/epilogue/block/block_epilogue_qbmm_pertensor_streamk.h"
#include "blaze/gemm/block/block_scheduler_matmul_streamk.h"
#include "blaze/gemm/block/block_mmad_a8w8_fixpipe_quant.h"

template <class A_TYPE, class B_TYPE, class SCALE_TYPE, class C_TYPE, class BIAS_TYPE, class aLayout, class bLayout,
          class cLayout, uint64_t FULL_LOAD_MODE = 0>
__aicore__ inline void QbmmPertensorStreamKTensorApiKernel(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR scale, GM_ADDR bias,
                                                           GM_ADDR perTokenScale, GM_ADDR cGM, GM_ADDR workspaceGm,
                                                           const void* tilingData)
{
    using AType = A_TYPE;
    using BType = B_TYPE;
    using X2ScaleType = SCALE_TYPE;
    using BiasType = BIAS_TYPE;
    using OutType = C_TYPE;

    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;

    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleFixpipeQuant<FULL_LOAD_MODE, false,
                                                                    Blaze::Gemm::KernelQbmmPertensorMultiBlockStreamK>;
    using EpilogueDispatchPolicy = DispatchPolicy;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerMatmulStreamK<ProblemShape>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, AType, aLayout,
                                                    AscendC::Std::tuple<BType, X2ScaleType>, bLayout, OutType, cLayout,
                                                    BiasType, cLayout>;
    using WorkspaceType = typename BlockMmad::WorkspaceType;
    using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueQbmmPertensorStreamK<
        WorkspaceType, OutType, EpilogueDispatchPolicy, X2ScaleType, float>;
    using MatmulKernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    using Params = typename MatmulKernel::Params;

    const DequantBmm::QuantBatchMatmulV3StreamKBasicAPITilingData*
        quantBmmTilingData = static_cast<const DequantBmm::QuantBatchMatmulV3StreamKBasicAPITilingData*>(tilingData);
    DequantBmm::BasicAPICubeTiling matmulTiling = quantBmmTilingData->matmulTiling;
    bool hasBias = matmulTiling.isBias != 0U;
    constexpr bool isIntScale = AscendC::IsSameType<X2ScaleType, uint64_t>::value ||
                                AscendC::IsSameType<X2ScaleType, int64_t>::value;
    constexpr bool isFp8Input = !AscendC::IsSameType<AType, int8_t>::value;
    bool biasInMmad = hasBias && ((quantBmmTilingData->params.biasDtype == DT_INT32 &&
                                   AscendC::IsSameType<BiasType, int32_t>::value) ||
                                  (isFp8Input && isIntScale && quantBmmTilingData->params.biasDtype == DT_FLOAT &&
                                   AscendC::IsSameType<BiasType, float>::value));
    GM_ADDR biasMmadGm = biasInMmad ? bias : nullptr;
    GM_ADDR biasEpilogueGm = hasBias && !biasInMmad ? bias : nullptr;
    bool isBiasEpilogue = biasEpilogueGm != nullptr;

    uint64_t kernelKL1 = quantBmmTilingData->streamKTiling.kL1;

    Params params{
        {matmulTiling.m, matmulTiling.n, matmulTiling.k, quantBmmTilingData->params.batchC},
        {aGM, bGM, cGM, biasMmadGm, perTokenScale, scale},
        {cGM, workspaceGm, scale, perTokenScale, biasEpilogueGm, isBiasEpilogue, quantBmmTilingData->params.biasDtype},
        {AscendC::GetBlockNum(), matmulTiling.baseM, matmulTiling.baseN, matmulTiling.baseK,
         quantBmmTilingData->streamKTiling.singleCoreK, static_cast<int64_t>(kernelKL1)}};

    MatmulKernel qbmm;
    qbmm(params);
}

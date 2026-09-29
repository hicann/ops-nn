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
 * \file quant_matmul_activation_mx_quant_without_batch.h
 * \brief Kernel assembly for quantized matmul and activation epilogues, without batch broadcasting.
 */
#pragma once
#include "quant_matmul_activation_quant_tiling_data.h"
#include "blaze/gemm/block/block_scheduler_qbmm.h"
#include "blaze/epilogue/block/block_epilogue_gelu_mx_quant.h"
#include "blaze/epilogue/block/block_epilogue_swiglu_mx_quant.h"
#include "blaze/gemm/block/block_mmad_qbmm_mx.h"
#include "blaze/gemm/kernel/kernel_qbmm_mx_activation_quant.h"

template <class AType, class BType, class OutType, class LayoutA, class LayoutB, class LayoutC,
          uint64_t FULL_LOAD_MODE = 0>
__aicore__ inline void QuantMatmulGeluMxQuantWithoutBatchKernel(GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR x1Scale,
                                                                GM_ADDR x2Scale, GM_ADDR y, GM_ADDR yScale,
                                                                GM_ADDR workspace, const void* tilingData)
{
    using MatmulOutType = float;
    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<ProblemShape, FULL_LOAD_MODE, LayoutA,
                                                                                LayoutB, AType>;
    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMx<FULL_LOAD_MODE, false,
                                                          Blaze::Gemm::KernelMmadWithScaleMxActivationQuant,
                                                          Blaze::Gemm::L0C2UB_MODE_DUAL_DST_SPLIT_M>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, AType, LayoutA, BType, LayoutB, MatmulOutType,
                                                    LayoutC, float, LayoutC>;
    using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueGeluMxQuant<OutType, MatmulOutType>;
    using MatmulKernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    using Params = typename MatmulKernel::Params;

    const auto& data = *static_cast<const QMMAQ::QMMAQWithoutBatchTilingData*>(tilingData);

    MatmulKernel{}(Params{
        {data.m, data.n, data.k, 1L},
        {x1, x2, y, bias, x1Scale, x2Scale},
        {y, yScale, static_cast<uint32_t>(data.baseM), static_cast<uint32_t>(data.baseN),
         static_cast<Blaze::Epilogue::Block::GeluAlg>(static_cast<uint8_t>(data.activationType)),
         static_cast<Blaze::Epilogue::Block::QuantAlg>(static_cast<uint8_t>(data.scaleAlg)),
         static_cast<Blaze::Epilogue::Block::ROUND_MODE_FP4>(static_cast<uint8_t>(data.roundMode)), data.dstTypeMax},
        {data.kL1, data.scaleKL1, static_cast<uint64_t>(data.nBufferNum)},
        {data.baseM, data.baseN, data.mTailTile, data.nTailTile, data.mBaseTailSplitCnt, data.nBaseTailSplitCnt,
         data.mTailMain, data.nTailMain},
        {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, static_cast<uint32_t>(data.baseM), static_cast<uint32_t>(data.baseN),
         static_cast<uint32_t>(data.baseK), static_cast<uint32_t>(data.isBias), static_cast<uint32_t>(data.dbL0C),
         static_cast<uint32_t>(data.weightMustHitL2)}});
}

template <class AType, class BType, class OutType, class LayoutA, class LayoutB, class LayoutC,
          uint64_t FULL_LOAD_MODE = 0>
__aicore__ inline void QuantMatmulSwiGluQuantMxWithoutBatchKernel(GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR x1Scale,
                                                                  GM_ADDR x2Scale, GM_ADDR y, GM_ADDR yScale,
                                                                  GM_ADDR workspace, const void* tilingData)
{
    using MatmulOutType = float;
    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<ProblemShape, FULL_LOAD_MODE, LayoutA,
                                                                                LayoutB, AType>;
    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMx<FULL_LOAD_MODE, false,
                                                          Blaze::Gemm::KernelMmadWithScaleMxActivationQuant,
                                                          Blaze::Gemm::L0C2UB_MODE_DUAL_DST_SPLIT_M, 0, true>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, AType, LayoutA, BType, LayoutB, MatmulOutType,
                                                    LayoutC, float, LayoutC>;
    using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueSwigluMxQuant<OutType, MatmulOutType,
                                                                             AscendC::fp8_e8m0_t>;
    using MatmulKernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    using Params = typename MatmulKernel::Params;

    const auto& data = *static_cast<const QMMAQ::QMMAQWithoutBatchTilingData*>(tilingData);

    MatmulKernel{}(Params{
        {data.m, data.n, data.k, 1L},
        {x1, x2, y, bias, x1Scale, x2Scale},
        {y, yScale, static_cast<uint32_t>(data.baseM), static_cast<uint32_t>(data.baseN) >> 1,
         static_cast<uint8_t>(data.scaleAlg)},
        {data.kL1, data.scaleKL1, static_cast<uint64_t>(data.nBufferNum)},
        {data.baseM, data.baseN, data.mTailTile, data.nTailTile, data.mBaseTailSplitCnt, data.nBaseTailSplitCnt,
         data.mTailMain, data.nTailMain},
        {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, static_cast<uint32_t>(data.baseM), static_cast<uint32_t>(data.baseN),
         static_cast<uint32_t>(data.baseK), static_cast<uint32_t>(data.isBias), static_cast<uint32_t>(data.dbL0C),
         static_cast<uint32_t>(data.weightMustHitL2)}});
}

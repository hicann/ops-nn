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
 * \file quant_matmul_activation_mx_quant.h
 * \brief Shared kernel assembly for activation and MX quantization epilogues.
 */
#pragma once
#include "quant_matmul_activation_quant_tiling_data.h"
#include "quant_matmul_activation_quant_tiling_key.h"
#include "blaze/gemm/block/block_scheduler_qbmm.h"
#include "blaze/epilogue/block/block_epilogue_gelu_mx_quant.h"
#include "blaze/epilogue/block/block_epilogue_swiglu_mx_quant.h"
#include "blaze/epilogue/block/block_epilogue_swish_mx_quant.h"
#include "blaze/epilogue/block/block_epilogue_sigmoid_mul_mx_quant.h"
#include "blaze/gemm/block/block_mmad_qbmm_mx.h"
#include "blaze/gemm/kernel/kernel_qbmm_mx_activation_quant.h"

namespace QMMAQ {

template <int EPILOGUE_TYPE, class OutType>
struct MxQuantEpilogueTraits;

template <class OutType>
struct MxQuantEpilogueTraits<TPL_EPILOGUE_GELU, OutType> {
    using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueGeluMxQuant<OutType, float>;
    static constexpr bool CONCAT_N = false;

    template <class TilingData>
    __aicore__ static inline typename BlockEpilogue::Params MakeParams(GM_ADDR y, GM_ADDR yScale, GM_ADDR,
                                                                       const TilingData& data)
    {
        return {y,
                yScale,
                static_cast<uint32_t>(data.baseM),
                static_cast<uint32_t>(data.baseN),
                static_cast<Blaze::Epilogue::Block::GeluAlg>(static_cast<uint8_t>(data.activationType)),
                static_cast<Blaze::Epilogue::Block::QuantAlg>(static_cast<uint8_t>(data.scaleAlg)),
                static_cast<Blaze::Epilogue::Block::ROUND_MODE_FP4>(static_cast<uint8_t>(data.roundMode)),
                data.dstTypeMax};
    }
};

template <class OutType>
struct MxQuantEpilogueTraits<TPL_EPILOGUE_SWIGLU, OutType> {
    using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueSwigluMxQuant<OutType, float, AscendC::fp8_e8m0_t>;
    // The MMAD produces [gate|linear]; the epilogue consumes N/2 output columns.
    static constexpr bool CONCAT_N = true;

    template <class TilingData>
    __aicore__ static inline typename BlockEpilogue::Params MakeParams(GM_ADDR y, GM_ADDR yScale, GM_ADDR,
                                                                       const TilingData& data)
    {
        return {y, yScale, static_cast<uint32_t>(data.baseM), static_cast<uint32_t>(data.baseN) >> 1,
                static_cast<uint8_t>(data.scaleAlg)};
    }
};

template <class OutType>
struct MxQuantEpilogueTraits<TPL_EPILOGUE_SILU, OutType> {
    using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueSwishMxQuant<OutType, float>;
    static constexpr bool CONCAT_N = false;

    template <class TilingData>
    __aicore__ static inline typename BlockEpilogue::Params MakeParams(GM_ADDR y, GM_ADDR yScale, GM_ADDR,
                                                                       const TilingData& data)
    {
        // SiLU uses scalar-beta Swish with beta fixed to 1.
        return {y,
                yScale,
                static_cast<uint32_t>(data.baseM),
                static_cast<uint32_t>(data.baseN),
                static_cast<uint32_t>(data.scaleAlg),
                1.0F};
    }
};

template <class OutType>
struct MxQuantEpilogueTraits<TPL_EPILOGUE_SIGMOID_MUL, OutType> {
    using BlockEpilogue = Blaze::Epilogue::Block::BlockEpilogueSigmoidMulMxQuant<OutType, float>;
    static constexpr bool CONCAT_N = false;

    template <class TilingData>
    __aicore__ static inline typename BlockEpilogue::Params MakeParams(GM_ADDR y, GM_ADDR yScale, GM_ADDR x3,
                                                                       const TilingData& data)
    {
        return {y, yScale, x3, static_cast<uint32_t>(data.scaleAlg)};
    }
};

__aicore__ inline int64_t GetMxQuantBatchCount(const QMMAQTilingData& data) { return data.batchCount; }

__aicore__ inline int64_t GetMxQuantBatchCount(const QMMAQWithoutBatchTilingData&) { return 1; }

template <class BatchParams>
__aicore__ inline BatchParams MakeQbmmTilingParams(const QMMAQTilingData& data)
{
    return {data.batchA1,
            data.batchA2,
            data.batchA3,
            data.batchA4,
            data.batchB1,
            data.batchB2,
            data.batchB3,
            data.batchB4,
            data.batchC1,
            data.batchC2,
            data.batchC3,
            data.batchC4,
            static_cast<uint32_t>(data.biasThreeDim),
            static_cast<uint32_t>(data.baseM),
            static_cast<uint32_t>(data.baseN),
            static_cast<uint32_t>(data.baseK),
            static_cast<uint32_t>(data.isBias),
            static_cast<uint32_t>(data.dbL0C),
            static_cast<uint32_t>(data.weightMustHitL2)};
}

template <class BatchParams>
__aicore__ inline BatchParams MakeQbmmTilingParams(const QMMAQWithoutBatchTilingData& data)
{
    return {1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            static_cast<uint32_t>(data.baseM),
            static_cast<uint32_t>(data.baseN),
            static_cast<uint32_t>(data.baseK),
            static_cast<uint32_t>(data.isBias),
            static_cast<uint32_t>(data.dbL0C),
            static_cast<uint32_t>(data.weightMustHitL2)};
}

template <int EPILOGUE_TYPE, uint64_t FULL_LOAD_MODE, class AType, class BType, class OutType, class LayoutA,
          class LayoutB, class LayoutC, class TilingData>
__aicore__ inline void QuantMatmulActivationMxQuantKernel(GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR x1Scale,
                                                          GM_ADDR x2Scale, GM_ADDR x3, GM_ADDR y, GM_ADDR yScale,
                                                          const TilingData& data)
{
    static_assert(AscendC::IsSameType<TilingData, QMMAQTilingData>::value ||
                      AscendC::IsSameType<TilingData, QMMAQWithoutBatchTilingData>::value,
                  "Unsupported QuantMatmulActivationQuant tiling data type.");

    using EpilogueTraits = MxQuantEpilogueTraits<EPILOGUE_TYPE, OutType>;
    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<ProblemShape, FULL_LOAD_MODE, LayoutA,
                                                                                LayoutB, AType>;
    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMx<
        FULL_LOAD_MODE, false, Blaze::Gemm::KernelMmadWithScaleMxActivationQuant,
        Blaze::Gemm::L0C2UB_MODE_DUAL_DST_SPLIT_M, 0, EpilogueTraits::CONCAT_N>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, AType, LayoutA, BType, LayoutB, float, LayoutC,
                                                    float, LayoutC>;
    using MatmulKernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad,
                                                            typename EpilogueTraits::BlockEpilogue, BlockScheduler>;

    MatmulKernel{}(
        typename MatmulKernel::Params{{data.m, data.n, data.k, GetMxQuantBatchCount(data)},
                                      {x1, x2, y, bias, x1Scale, x2Scale},
                                      EpilogueTraits::MakeParams(y, yScale, x3, data),
                                      {data.kL1, data.scaleKL1, static_cast<uint64_t>(data.nBufferNum)},
                                      {data.baseM, data.baseN, data.mTailTile, data.nTailTile, data.mBaseTailSplitCnt,
                                       data.nBaseTailSplitCnt, data.mTailMain, data.nTailMain},
                                      MakeQbmmTilingParams<typename MatmulKernel::QBMMTiling>(data)});
}

} // namespace QMMAQ

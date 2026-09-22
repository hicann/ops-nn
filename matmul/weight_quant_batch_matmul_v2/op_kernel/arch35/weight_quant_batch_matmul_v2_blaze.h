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
 * \file weight_quant_batch_matmul_v2_blaze.h
 * \brief ops-nn adapter that maps WQMMV2 B8 tiling data to assembled Blaze components.
 */
#pragma once

#include "weight_quant_batch_matmul_v2_arch35_tiling_key.h"
#include "blaze/gemm/kernel/kernel_wqmm_mix_antiquant.h"
#include "blaze/gemm/block/block_scheduler_wqmm.h"

namespace WeightQuantBatchMatmulV2 {

template <int TemplateCustom, bool TransA, bool TransB, int AntiquantType, bool HasAntiquantOffset, bool IsBiasFp32,
          bool IsWeightNz>
__aicore__ inline void InvokeBlazeB8Kernel(GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR antiquantOffset,
                                           GM_ADDR quantScale, GM_ADDR quantOffset, GM_ADDR bias, GM_ADDR y,
                                           GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    namespace Te = AscendC::Te;

    static constexpr bool X_SUPPORTED = AscendC::IsSameType<DTYPE_X, half>::value ||
                                        AscendC::IsSameType<DTYPE_X, bfloat16_t>::value;
    static constexpr bool Y_SUPPORTED = AscendC::IsSameType<DTYPE_Y, half>::value ||
                                        AscendC::IsSameType<DTYPE_Y, bfloat16_t>::value;
    static constexpr bool B8_SUPPORTED = AscendC::IsSameType<DTYPE_WEIGHT, int8_t>::value ||
                                         AscendC::IsSameType<DTYPE_WEIGHT, float8_e4m3_t>::value ||
                                         AscendC::IsSameType<DTYPE_WEIGHT, hifloat8_t>::value;
    static_assert(
        AntiquantType == WQBMMV2_ANTIQUANT_TYPE_PER_TENSOR || AntiquantType == WQBMMV2_ANTIQUANT_TYPE_PER_CHANNEL,
        "Blaze B8 supports only per-tensor/per-channel antiquantization");
    static constexpr Blaze::Gemm::QuantMode ANTIQUANT_TYPE = AntiquantType == WQBMMV2_ANTIQUANT_TYPE_PER_TENSOR ?
                                                                 Blaze::Gemm::QuantMode::PERTENSOR_MODE :
                                                                 Blaze::Gemm::QuantMode::PERCHANNEL_MODE;

    static_assert(X_SUPPORTED && Y_SUPPORTED, "Blaze B8 supports only FP16/BF16 X and Y");
    static_assert(AscendC::IsSameType<DTYPE_X, DTYPE_Y>::value, "Blaze B8 requires X and Y to have the same type");
    static_assert(B8_SUPPORTED, "Blaze B8 supports only int8, float8_e4m3 and hifloat8 weight");
    static_assert(AscendC::IsSameType<DTYPE_ANTIQUANT_SCALE, DTYPE_X>::value,
                  "Blaze B8 requires antiquant scale/offset type to match X");
    static_assert(!TransA, "Blaze B8 supports only TransA=false");
    static_assert(!IsWeightNz, "Blaze B8 supports only ND weight");

    static constexpr uint64_t AIV_PER_AIC = 2;
    static constexpr uint32_t DOUBLE_BUFFER_COUNT = 2;
    static constexpr uint32_t QUADRUPLE_BUFFER_COUNT = 4;
    // Physical UB row pitches in bytes for 8-bit weight inputs.
    static constexpr uint32_t UB_ROW_BYTES_256 = 256;
    static constexpr uint32_t UB_ROW_BYTES_512 = 512;
    static constexpr uint32_t UB_ROW_BYTES_1024 = 1024;

    static constexpr bool USE_512_BYTE_ROW = TemplateCustom == WQBMMV2_TEMPLATE_MTE2_INNER_SIZE_512_BUF_NUM_2 ||
                                             TemplateCustom == WQBMMV2_TEMPLATE_MTE2_INNER_SIZE_512_BUF_NUM_4 ||
                                             TemplateCustom == WQBMMV2_TEMPLATE_MTE2_INNER_SIZE_512_BUF_NUM_DEFAULT;
    static constexpr bool USE_1024_BYTE_ROW = TemplateCustom == WQBMMV2_TEMPLATE_MTE2_INNER_SIZE_1024_BUF_NUM_2 ||
                                              TemplateCustom == WQBMMV2_TEMPLATE_MTE2_INNER_SIZE_1024_BUF_NUM_4;
    static constexpr bool USE_DOUBLE_BUFFER = TemplateCustom == WQBMMV2_TEMPLATE_MTE2_INNER_SIZE_512_BUF_NUM_2 ||
                                              TemplateCustom == WQBMMV2_TEMPLATE_MTE2_INNER_SIZE_1024_BUF_NUM_2 ||
                                              (TemplateCustom == WQBMMV2_TEMPLATE_MTE2_INNER_SIZE_512_BUF_NUM_DEFAULT &&
                                               B8_SUPPORTED) ||
                                              TemplateCustom == WQBMMV2_TEMPLATE_MTE2_INNER_SIZE_256_BUF_NUM_2;
    static constexpr uint32_t UB_MTE2_INNER_SIZE = USE_512_BYTE_ROW  ? UB_ROW_BYTES_512 :
                                                   USE_1024_BYTE_ROW ? UB_ROW_BYTES_1024 :
                                                                       UB_ROW_BYTES_256;
    static constexpr uint32_t UB_MTE2_BUF_NUM = USE_DOUBLE_BUFFER ? DOUBLE_BUFFER_COUNT : QUADRUPLE_BUFFER_COUNT;

    using DispatchPolicy = Blaze::Gemm::MatmulWithWeightAntiquant<AIV_PER_AIC, UB_MTE2_INNER_SIZE, UB_MTE2_BUF_NUM,
                                                                  ANTIQUANT_TYPE, HasAntiquantOffset>;
    using ProblemShape = Te::Shape<int64_t, int64_t, int64_t>;
    using LayoutA = Te::NDExtLayoutPtn;
    using LayoutB = AscendC::Std::conditional_t<TransB, Te::NDExtLayoutPtn, Te::DNExtLayoutPtn>;
    using LayoutScale = Te::NDExtLayoutPtn;
    using LayoutC = Te::NDExtLayoutPtn;
    using LayoutBias = Te::NDExtLayoutPtn;
    using BTypes = AscendC::Std::tuple<DTYPE_WEIGHT, DTYPE_ANTIQUANT_SCALE>;
    using BLayouts = AscendC::Std::tuple<LayoutB, LayoutScale>;
    using BiasType = AscendC::Std::conditional_t<IsBiasFp32, float, DTYPE_X>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, DTYPE_X, LayoutA, BTypes, BLayouts, DTYPE_Y,
                                                    LayoutC, BiasType, LayoutBias>;
    using Scheduler = Blaze::Gemm::Block::BlockSchedulerWqmmTailResplit<ProblemShape>;
    using Kernel = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, void, Scheduler>;

    GET_TILING_DATA_WITH_STRUCT(wqbmmv2_tiling::WeightQuantBatchMatmulV2ASTilingDataParams, tilingDataIn, tiling);
    const int64_t m = static_cast<int64_t>(tilingDataIn.mSize);
    const int64_t n = static_cast<int64_t>(tilingDataIn.nSize);
    const int64_t k = static_cast<int64_t>(tilingDataIn.kSize);
    const int64_t baseM = static_cast<int64_t>(tilingDataIn.matmulTiling.baseM);
    const int64_t baseN = static_cast<int64_t>(tilingDataIn.matmulTiling.baseN);
    const int64_t baseK = static_cast<int64_t>(tilingDataIn.matmulTiling.baseK);
    const int64_t stepKa = static_cast<int64_t>(tilingDataIn.matmulTiling.stepKa);
    const int64_t stepKb = static_cast<int64_t>(tilingDataIn.matmulTiling.stepKb);

    ProblemShape problemShape = Te::MakeShape(m, n, k);

    typename BlockMmad::Params mmadParams{};
    mmadParams.aGmAddr = x;
    mmadParams.cGmAddr = y;
    mmadParams.biasGmAddr = bias;
    mmadParams.l1TileShape = Te::MakeShape(baseM, baseN, stepKa * baseK, stepKb * baseK);
    mmadParams.l0TileShape = Te::MakeShape(baseM, baseN, baseK);
    mmadParams.hasBias = static_cast<bool>(tilingDataIn.hasBias);
    mmadParams.kSize = tilingDataIn.kSize;

    typename Kernel::PrologueParams prologueParams{};
    prologueParams.bGmAddr = weight;
    prologueParams.scaleGmAddr = antiquantScale;
    prologueParams.offsetGmAddr = antiquantOffset;
    prologueParams.weightL2Cacheable = tilingDataIn.weightL2Cacheable;

    typename Scheduler::Params schedulerParams{};
    schedulerParams.mL1Tile = tilingDataIn.matmulTiling.baseM;
    schedulerParams.mainBlockCount = tilingDataIn.mainBlockCount;
    schedulerParams.firstTailBlockCount = tilingDataIn.firstTailBlockCount;
    schedulerParams.secondTailBlockCount = tilingDataIn.secondTailBlockCount;
    schedulerParams.mainBlockSize = tilingDataIn.mainBlockL1Size;
    schedulerParams.firstTailBlockSize = tilingDataIn.firstTailBlockL1Size;
    schedulerParams.secondTailBlockSize = tilingDataIn.secondTailBlockL1Size;
    schedulerParams.cubeNumBlocksM = tilingDataIn.cubeNumBlocksM;
    schedulerParams.cubeNumBlocksN = tilingDataIn.cubeNumBlocksN;

    typename Kernel::Params params{};
    params.problemShape = problemShape;
    params.mmadParams = mmadParams;
    params.aPreloadSize = tilingDataIn.aPreloadSize;
    params.aElementCount = static_cast<uint64_t>(tilingDataIn.mSize) * tilingDataIn.kSize;
    params.prologueParams = prologueParams;
    params.schedulerParams = schedulerParams;

    (void)quantScale;  // FP16/BF16 output does not require output quantization.
    (void)quantOffset; // FP16/BF16 output does not require output quantization.
    (void)workspace;   // No additional GM workspace is required.
    Kernel kernel;
    kernel(params);
}

} // namespace WeightQuantBatchMatmulV2

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
 * \file quant_batch_matmul_v4_weight_quant_pergroup_blaze.h
 * \brief T-CG per-group weight dequant kernel entry via blaze GemmUniversal.
 */

#pragma once

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "blaze/gemm/kernel/kernel_wqmm_mix_pergroup.h"
#include "blaze/gemm/utils/layout_utils.h"
#include "quant_batch_matmul_v4_tiling_data_apt.h"

namespace QuantBatchMatmulV4 {
namespace Arch35 {

template <bool IS_WEIGHT_NZ>
__aicore__ inline void RunWeightQuantPergroupBlaze(GM_ADDR x1, GM_ADDR x2, GM_ADDR x2Scale, GM_ADDR yScale, GM_ADDR y,
                                                   const qbmmv4_tiling::QuantBatchMatmulV4TilingDataParams& tilingData)
{
    using LayoutA = AscendC::Te::NDExtLayoutPtn;
    using LayoutC = AscendC::Te::NDExtLayoutPtn;
    using LayoutB = AscendC::Std::conditional_t<IS_WEIGHT_NZ, AscendC::Te::NZLayoutPtn, AscendC::Te::DNExtLayoutPtn>;
    // The B side carries the packed-FP4 weight plus the per-group scale (x2Scale, consumed
    // by the AIV dequant in UB). TCG has no bias input: the trailing BiasType/LayoutBias
    // slots only satisfy the BlockMmad template signature.
    using BTypeTuple = AscendC::Std::tuple<DTYPE_X2, DTYPE_X2_SCALE>;
    using BlockMmadType = Blaze::Gemm::Block::BlockMmad<Blaze::Gemm::MatmulWithWeightQuantPergroup, DTYPE_X1, LayoutA,
                                                        BTypeTuple, LayoutB, DTYPE_Y, LayoutC, void, void>;
    using ProblemShape = AscendC::Te::Shape<int64_t, int64_t, int64_t>;
    using BlockSchedulerType = Blaze::Gemm::Block::BlockSchedulerWqmmBlockSplit<ProblemShape, LayoutB, DTYPE_X1>;
    using KernelImpl = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmadType, void, BlockSchedulerType>;

    const auto& matmulTiling = tilingData.matmulTiling;
    // BlockMmad::Params carries both the quantized-fixpipe fields and the per-group
    // weight-prologue fields; fill the per-group subset by name (the GM-B streaming fields
    // keep their defaults).
    typename BlockMmadType::Params mmadParams{};
    mmadParams.aGmAddr = x1;
    mmadParams.cGmAddr = y;
    mmadParams.yScaleGmAddr = yScale;
    mmadParams.l1TileShape = asc::te::make_shape(static_cast<int64_t>(matmulTiling.stepM) * matmulTiling.baseM,
                                                 static_cast<int64_t>(matmulTiling.stepN) * matmulTiling.baseN,
                                                 static_cast<int64_t>(matmulTiling.stepKa) * matmulTiling.baseK,
                                                 static_cast<int64_t>(matmulTiling.stepKb) * matmulTiling.baseK);
    mmadParams.l0TileShape = asc::te::make_shape(static_cast<int64_t>(matmulTiling.baseM),
                                                 static_cast<int64_t>(matmulTiling.baseN),
                                                 static_cast<int64_t>(matmulTiling.baseK));
    mmadParams.vecCoreParallel = tilingData.vecCoreParallel;
    mmadParams.AL1Pingpong = tilingData.AL1Pingpong;
    mmadParams.BL1Pingpong = tilingData.BL1Pingpong;
    mmadParams.dbL0C = static_cast<uint32_t>(matmulTiling.dbL0C);
    typename KernelImpl::Params params{
        asc::te::make_shape(static_cast<int64_t>(tilingData.mSize), static_cast<int64_t>(tilingData.nSize),
                            static_cast<int64_t>(tilingData.kSize)),
        mmadParams,
        {x2, x2Scale, tilingData.groupSize, tilingData.nBubSize, tilingData.kBubSize},
        {tilingData.cubeNumBlocksM, tilingData.cubeNumBlocksN, static_cast<uint32_t>(matmulTiling.baseM),
         static_cast<uint32_t>(matmulTiling.baseN), static_cast<uint32_t>(matmulTiling.iterateOrder)}};
    KernelImpl kernel;
    kernel(params);
}

template <bool IS_WEIGHT_NZ>
__aicore__ inline void InvokeWeightQuantPergroupBlaze(GM_ADDR x1, GM_ADDR x2, [[maybe_unused]] GM_ADDR bias,
                                                      [[maybe_unused]] GM_ADDR x1_scale, GM_ADDR x2_scale,
                                                      GM_ADDR y_scale, [[maybe_unused]] GM_ADDR x1_offset,
                                                      [[maybe_unused]] GM_ADDR x2_offset,
                                                      [[maybe_unused]] GM_ADDR y_offset, GM_ADDR y,
                                                      [[maybe_unused]] GM_ADDR workspace, const GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GET_TILING_DATA_WITH_STRUCT(qbmmv4_tiling::QuantBatchMatmulV4TilingDataParams, tilingDataIn, tiling);
    RunWeightQuantPergroupBlaze<IS_WEIGHT_NZ>(x1, x2, x2_scale, y_scale, y, tilingDataIn);
}

} // namespace Arch35
} // namespace QuantBatchMatmulV4

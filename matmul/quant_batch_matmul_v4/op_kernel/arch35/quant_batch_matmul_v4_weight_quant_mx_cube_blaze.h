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
 * \file quant_batch_matmul_v4_weight_quant_mx_cube_blaze.h
 * \brief AIC-only MxA8W4 kernels
 */

#pragma once

#include "blaze/gemm/block/block_mmad_qbmm_mx.h"
#include "blaze/gemm/block/block_scheduler_qbmm.h"
#include "blaze/gemm/kernel/kernel_qbmm_mx_without_batch.h"
#include "blaze/gemm/policy/dispatch_policy.h"
#include "../../quant_batch_matmul_v3/arch35/quant_batch_matmul_v3_tiling_data.h"

namespace QuantBatchMatmulV4 {
namespace Arch35 {

template <uint64_t FULL_LOAD_MODE>
__aicore__ inline void RunWeightQuantMxCubeBlaze(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR x1Scale, GM_ADDR x2Scale, GM_ADDR y,
    const DequantBmm::QuantBatchMatmulV3TensorAPIWithoutBatchTilingData& tilingData)
{
    using AType = DTYPE_X1;
    using BType = DTYPE_X2;
    using CType = DTYPE_Y;
    using BiasType = DTYPE_Y;
    using LayoutA = asc::te::nd_ext_layout_ptn;
    using LayoutB = asc::te::zn_layout_ptn;
    using LayoutScaleB = asc::te::scaleb_dn_layout_ptn;
    using WeightAndScaleLayout = AscendC::Std::tuple<LayoutB, LayoutScaleB>;
    using LayoutC = asc::te::nd_ext_layout_ptn;
    using LayoutBias = LayoutC;

    using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMx<FULL_LOAD_MODE, false,
                                                          Blaze::Gemm::KernelMmadWithScaleMxWithoutBatch,
                                                          Blaze::Gemm::L0C2UB_MODE_NONE, 0U, false>;
    using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, AType, LayoutA, BType, WeightAndScaleLayout, CType,
                                                    LayoutC, BiasType, LayoutBias>;

    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<ProblemShape, FULL_LOAD_MODE, LayoutA,
                                                                                LayoutB, AType>;
    using KernelImpl = Blaze::Gemm::Kernel::GemmUniversal<ProblemShape, BlockMmad, void, BlockScheduler>;

    typename KernelImpl::Params params{
        asc::te::make_shape(static_cast<int64_t>(tilingData.m), static_cast<int64_t>(tilingData.n),
                            static_cast<int64_t>(tilingData.k), int64_t{1}),
        {x1, x2, y, bias, x1Scale, x2Scale},
        {tilingData.kBL1, tilingData.scaleKL1, tilingData.nBufferNum},
        {static_cast<int64_t>(tilingData.baseM), static_cast<int64_t>(tilingData.baseN),
         static_cast<int64_t>(tilingData.mTailTile), static_cast<int64_t>(tilingData.nTailTile),
         static_cast<int64_t>(tilingData.mBaseTailSplitCnt), static_cast<int64_t>(tilingData.nBaseTailSplitCnt),
         static_cast<int64_t>(tilingData.mTailMain), static_cast<int64_t>(tilingData.nTailMain),
         static_cast<int64_t>(tilingData.windowLen)},
        {tilingData.baseM, tilingData.baseN, tilingData.baseK, tilingData.isBias, tilingData.dbL0C,
         tilingData.weightMustHitL2}};
    KernelImpl kernel;
    kernel(params);
}

template <uint64_t FULL_LOAD_MODE>
__aicore__ inline void InvokeWeightQuantMxCubeBlaze(GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR x1Scale,
                                                    GM_ADDR x2Scale, GM_ADDR y, const GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    GET_TILING_DATA_WITH_STRUCT(DequantBmm::QuantBatchMatmulV3TensorAPIWithoutBatchTilingData, tilingData, tiling);
    RunWeightQuantMxCubeBlaze<FULL_LOAD_MODE>(x1, x2, bias, x1Scale, x2Scale, y, tilingData);
}

} // namespace Arch35
} // namespace QuantBatchMatmulV4

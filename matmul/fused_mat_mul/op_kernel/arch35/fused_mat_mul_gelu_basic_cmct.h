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
 * \file fused_mat_mul_gelu_basic_cmct.h
 * \brief
 */
#pragma once

#include "cmct/block/block_scheduler_policy.h"
#include "cmct/block/block_scheduler_utils.h"
#include "cmct/epilogue/block_epilogue_elementwise.h"
#include "cmct/kernel/kernel_matmul_mix_without_que.h"
#include "cmct/tile/tile_copy.h"
#include "fused_mat_mul_tiling_data.h"
#include "../../mat_mul_v3/arch35/block_scheduler_aswt.h"

namespace MatmulV3Advanced {
using namespace Cmct;
using namespace Cmct::Gemm;
constexpr uint64_t GELU_WORKSPACE_SMALL_N_LIMIT = 16UL;

template <uint64_t OpType, class OutType>
struct GeluFusionOpSelector;

template <class OutType>
struct GeluFusionOpSelector<OP_TYPE_GELU_ERF, OutType> {
    using type = Block::FusionGelu<OutType, OutType, Block::GeluApproxiMate::ERF>;
};

template <class OutType>
struct GeluFusionOpSelector<OP_TYPE_GELU_TANH, OutType> {
    using type = Block::FusionGelu<OutType, OutType, Block::GeluApproxiMate::TANH>;
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class A_LAYOUT, class B_LAYOUT, class C_LAYOUT,
          uint64_t GELU_OP_TYPE>
__aicore__ inline void MatMulGeluMixWithoutQueActKernel(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR yGM,
                                                        GM_ADDR workspaceGM,
                                                        const MatMulV3BasicTilingData& matMulTilingData,
                                                        int64_t batch = 0)
{
    static_assert(GELU_OP_TYPE == OP_TYPE_GELU_ERF || GELU_OP_TYPE == OP_TYPE_GELU_TANH, "unsupported gelu op type");
    using L1TileShape = AscendC::Shape<_0, _0, _0>;
    using L0TileShape = AscendC::Shape<_0, _0, _0>;
    using AType = A_TYPE;
    using BType = B_TYPE;
    using OutType = C_TYPE;
    using MmadOutType = float;
    using BiasType = BIAS_TYPE;
    using LayoutA = A_LAYOUT;
    using LayoutB = B_LAYOUT;
    using LayoutC = C_LAYOUT;
    using BlockScheduler = BuiltInAswtScheduler<MAT_MUL_NO_FULL_LOAD>;
    using DispatchPolicy = MatmulMultiBlockWithOutQue<AscendC::Shape<_0, _0, _0, _0>, MAT_MUL_NO_FULL_LOAD,
                                                      GELU_OP_TYPE>;
    using BlockMmad = Block::BlockMmadBuilder<AType, LayoutA, BType, LayoutB, MmadOutType, LayoutC, BiasType, LayoutC,
                                              L1TileShape, L0TileShape, BlockScheduler, DispatchPolicy>;
    using FusionOp = typename GeluFusionOpSelector<GELU_OP_TYPE, MmadOutType>::type;
    using BlockEpilogue = Block::BlockEpilogueElementwise<L0TileShape, OutType, MmadOutType, FusionOp>;
    using ProblemShape = MatmulShape;
    using MatmulKernel = Kernel::KernelMatmulMixWithoutQue<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    using Params = typename MatmulKernel::Params;
    Params params = {{matMulTilingData.m, matMulTilingData.n, matMulTilingData.k, batch},
                     {aGM, bGM, yGM, biasGM},
                     {yGM, {nullptr}},
                     {&matMulTilingData},
                     workspaceGM,
                     matMulTilingData.n < GELU_WORKSPACE_SMALL_N_LIMIT};
    MatmulKernel mm;
    mm(params);
}
} // namespace MatmulV3Advanced

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
 * \file mat_mul_mix_basic_cmct.h
 * \brief
 */
#pragma once

#include "cmct/block/block_scheduler_policy.h"
#include "cmct/block/block_scheduler_utils.h"
#include "cmct/kernel/kernel_matmul_mix_without_que.h"
#include "block_scheduler_aswt.h"

namespace MatmulV3Advanced {
using namespace Cmct;
using namespace Cmct::Gemm;

template <uint64_t OpType, class MmOutType, class X3Type = MmOutType>
struct FusionOpSelector;

template <class MmOutType, class X3Type>
struct FusionOpSelector<OP_TYPE_MUL, MmOutType, X3Type> {
    using type = Block::FusionMul<MmOutType, MmOutType, X3Type>;
};

template <class MmOutType, class X3Type>
struct FusionOpSelector<OP_TYPE_ADD, MmOutType, X3Type> {
    using type = Block::FusionAdd<MmOutType, MmOutType, X3Type>;
};

template <class MmOutType, class X3Type>
struct FusionOpSelector<0, MmOutType, X3Type> {
    using type = Block::DefaultFusion<MmOutType, MmOutType>;
};

template <int8_t InnerPrecise, uint64_t FusedOpType, class C_TYPE>
struct MmOutTypeSelector {
    static constexpr bool useFloat = (InnerPrecise == 0) && (FusedOpType == OP_TYPE_ADD || FusedOpType == OP_TYPE_MUL);
    using type = AscendC::Std::conditional_t<useFloat, float, C_TYPE>;
};

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class A_LAYOUT, class B_LAYOUT, class C_LAYOUT,
          uint64_t FULL_LOAD_MODE = 0, uint64_t FUSED_OP_TYPE = 0, int8_t INNER_PRECISE = 0>
__aicore__ inline void MatMulMixWithoutQueActKernel(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR yGM, GM_ADDR x3GM,
                                                    const MatMulV3BasicTilingData& tilingData, int64_t batch = 0,
                                                    int64_t batchX3 = 1)
{
    using L1TileShape = AscendC::Shape<_0, _0, _0>;
    using L0TileShape = AscendC::Shape<_0, _0, _0>;

    using AType = A_TYPE;
    using BType = B_TYPE;
    using BiasType = BIAS_TYPE;
    using OutType = C_TYPE;
    using MmOutType = typename MmOutTypeSelector<INNER_PRECISE, FUSED_OP_TYPE, C_TYPE>::type;

    using LayoutA = A_LAYOUT;
    using LayoutB = B_LAYOUT;
    using LayoutC = C_LAYOUT;

    using BlockScheduler = BuiltInAswtScheduler<FULL_LOAD_MODE>;
    using DispatchPolicy = MatmulMultiBlockWithOutQue<AscendC::Shape<_0, _0, _0, _0>, FULL_LOAD_MODE, FUSED_OP_TYPE,
                                                      INNER_PRECISE>;
    using BlockMmad = Block::BlockMmadBuilder<AType, LayoutA, BType, LayoutB, MmOutType, LayoutC, BiasType, LayoutC,
                                              L1TileShape, L0TileShape, BlockScheduler, DispatchPolicy>;
    using FusionOp = typename FusionOpSelector<FUSED_OP_TYPE, MmOutType, C_TYPE>::type;
    using BlockEpilogue = Block::BlockEpilogueElementwise<L0TileShape, OutType, MmOutType, FusionOp>;
    using ProblemShape = MatmulShape;
    using MatmulKernel = Kernel::KernelMatmulMixWithoutQue<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    using Params = typename MatmulKernel::Params;
    using EpilogueParams = typename BlockEpilogue::Params;

    EpilogueParams epilogueParams;
    epilogueParams.outGmAddr = yGM;
    epilogueParams.fusionParams.inputGmAddr = x3GM;
    epilogueParams.fusionParams.x3BatchBroadcast = (x3GM != nullptr && batch > 1 && batchX3 == 1);
    epilogueParams.fusionParams.x3M = tilingData.m;
    Params params = {
        {tilingData.m, tilingData.n, tilingData.k, batch}, {aGM, bGM, yGM, biasGM}, epilogueParams, {&tilingData}};

    MatmulKernel mm;
    mm(params);
}

} // namespace MatmulV3Advanced

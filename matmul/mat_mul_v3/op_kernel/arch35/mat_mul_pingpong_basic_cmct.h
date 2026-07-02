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
 * \file mat_mul_pingpong_basic_cmct.h
 * \brief
 */
#pragma once

#include "cmct/block/block_scheduler_policy.h"
#include "cmct/block/block_scheduler_utils.h"
#include "cmct/epilogue/block_epilogue_elementwise.h"
#include "cmct/epilogue/fusion/fusion_add.h"
#include "cmct/epilogue/fusion/fusion_mul.h"
#include "cmct/kernel/kernel_matmul_mix_without_que.h"
#include "cmct/kernel/kernel_matmul_without_que.h"
#include "block_scheduler_aswt.h"

namespace MatmulV3Advanced {
using namespace Cmct;
using namespace Cmct::Gemm;

namespace MatMulActInternal {
template <uint64_t OpType, class OutType>
struct FusionOpSelector {
    using type = Block::DefaultFusion<OutType, OutType>;
};

template <class OutType>
struct FusionOpSelector<OP_TYPE_ADD, OutType> {
    using type = Block::FusionAdd<OutType, OutType>;
};

template <class OutType>
struct FusionOpSelector<OP_TYPE_MUL, OutType> {
    using type = Block::FusionMul<OutType, OutType>;
};

template <uint64_t OpType, class L0TileShape, class OutType, class FusionOp>
struct BlockEpilogueSelector {
    using type = Block::BlockEpilogueEmpty;
};

template <class L0TileShape, class OutType, class FusionOp>
struct BlockEpilogueSelector<OP_TYPE_ADD, L0TileShape, OutType, FusionOp> {
    using type = Block::BlockEpilogueElementwise<L0TileShape, OutType, OutType, FusionOp>;
};

template <class L0TileShape, class OutType, class FusionOp>
struct BlockEpilogueSelector<OP_TYPE_MUL, L0TileShape, OutType, FusionOp> {
    using type = Block::BlockEpilogueElementwise<L0TileShape, OutType, OutType, FusionOp>;
};
} // namespace MatMulActInternal

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE, class A_LAYOUT,
          class B_LAYOUT, class C_LAYOUT, uint64_t FULL_LOAD_MODE = 0, uint64_t FUSED_OP_TYPE = 0>
__aicore__ inline void MatMulActKernel(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM,
    GM_ADDR cGM, GM_ADDR workspaceGM, const MatMulV3BasicTilingData& tilingData, int64_t batch = 0,
    GM_ADDR x3GM = nullptr, int64_t batchX3 = 1)
{
    // 定义L1和L0的TileShape
    using L1TileShape = AscendC::Shape<_0, _0, _0>;
    using L0TileShape = AscendC::Shape<_0, _0, _0>;

    // 定义矩阵的类型和布局
    using AType = A_TYPE;
    using BType = B_TYPE;
    using BiasType = BIAS_TYPE;
    using OutType = C_TYPE;

    using LayoutA = A_LAYOUT;
    using LayoutB = B_LAYOUT;
    using LayoutC = C_LAYOUT;

    // 定义scheduler类型 来自block_scheduler_policy.h
    using BlockScheduler = BuiltInAswtScheduler<FULL_LOAD_MODE>;

    // 定义MMAD类型
    using DispatchPolicy = MatmulMultiBlockWithOutQue<AscendC::Shape<_0, _0, _0, _0>, FULL_LOAD_MODE, FUSED_OP_TYPE>;
    using BlockMmad = Block::BlockMmadBuilder<
        AType, LayoutA, BType, LayoutB, OutType, LayoutC, BiasType, LayoutC, L1TileShape, L0TileShape, BlockScheduler,
        DispatchPolicy>;

    // 定义Fusion类型
    using FusionOp = typename MatMulActInternal::FusionOpSelector<FUSED_OP_TYPE, OutType>::type;

    // 定义BlockEpilogue类型
    using BlockEpilogue =
        typename MatMulActInternal::BlockEpilogueSelector<FUSED_OP_TYPE, L0TileShape, OutType, FusionOp>::type;

    // 定义shape的形状，tuple保存 m n k batch
    using ProblemShape = MatmulShape;

    // 定义Kernel类型
    if constexpr (FUSED_OP_TYPE == OP_TYPE_ADD || FUSED_OP_TYPE == OP_TYPE_MUL) {
        using MatmulKernel = Kernel::KernelMatmulMixWithoutQue<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
        using Params = typename MatmulKernel::Params;
        using EpilogueParams = typename BlockEpilogue::Params;
        EpilogueParams epilogueParams;
        epilogueParams.outGmAddr = cGM;
        epilogueParams.fusionParams.inputGmAddr = x3GM;
        epilogueParams.fusionParams.x3BatchBroadcast = (x3GM != nullptr && batch > 1 && batchX3 == 1);
        epilogueParams.fusionParams.x3M = tilingData.m;
        Params params = {
            {tilingData.m, tilingData.n, tilingData.k, batch}, // shape
            {aGM, bGM, cGM, biasGM},                           // gm addr
            epilogueParams,                                    // epilogue args
            {&tilingData}};
        MatmulKernel mm;
        mm(params);
    } else {
        using MatmulKernel = Kernel::KernelMatmulWithoutQue<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
        using Params = typename MatmulKernel::Params;
        Params params = {
            {tilingData.m, tilingData.n, tilingData.k, batch}, // shape
            {aGM, bGM, cGM, biasGM},                           // gm addr
            {},                                                // epilogue args
            {&tilingData}};
        MatmulKernel mm;
        mm(params);
    }
}
}

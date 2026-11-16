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
 * \file qbmm_fixpipe_quant_streamk.h
 * \brief DAV_9201/DAV_9202 per-tensor fixpipe quant StreamK kernel, CMCT BasicAPI implementation.
 */
#ifndef QBMM_FIXP_QUANT_STREAMK_H
#define QBMM_FIXP_QUANT_STREAMK_H

#include "cmct/block/block_mmad_qbmm_fixipipe_quant_streamk.h"
#include "cmct/epilogue/block_epilogue_streamk.h"
#include "cmct/kernel/kernel_qbmm_streamk.h"
#include "cmct/policy/dispatch_policy.h"
#include "cmct/utils/common_utils.h"
#include "cmct/utils/tuple_utils.h"
#include "../quant_batch_matmul_v3_base.h"

using namespace Cmct;
using namespace Cmct::Gemm;

template <class inputX1Type, class inputX2Type, class inputScaleType, class biasType, class yType, class formatX1,
          class formatX2, class formatY>
__aicore__ inline void QBMMStreamKFixpipeQuantKernel(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR cGM,
                                                     GM_ADDR scaleGM, GM_ADDR pertokenGM, GM_ADDR workspaceGM,
                                                     const void* tilingData, int64_t batch = 0)
{
    // 定义L1和L0的TileShape
    using L1TileShape = AscendC::Shape<_0, _0, _0>;
    using L0TileShape = AscendC::Shape<_0, _0, _0>;

    // 定义矩阵的类型和布局
    using AType = inputX1Type;
    using BType = inputX2Type;
    using ScaleType = inputScaleType;
    using BiasType = biasType;
    using OutType = yType;

    using WorkspaceDtype = typename AscendC::Conditional<AscendC::IsSameType<AType, int8_t>::value, int32_t,
                                                         float>::type;

    using LayoutA = formatX1;
    using LayoutB = formatX2;
    using LayoutC = formatY;

    if (batch > 1) {
        return;
    }

    // 定义scheduler类型
    using BlockScheduler = BuiltInStreamKSchedulerQbmm;

    // 定义MMAD类型
    using BlockMmad = Block::BlockMmadA8W8FixpipeQuantStreamk<
        MatmulWithScale<AscendC::Shape<_0, _0, _0, _0>, 0>, L1TileShape, L0TileShape, AType, LayoutA, BType, LayoutB,
        OutType, LayoutC, WorkspaceDtype, BiasType, LayoutC, ScaleType, void>;

    // 定义shape的形状，tuple保存 m n k batch
    using ProblemShape = MatmulShape;

    // 定义BlockEpilogue类型，float 是 worspace的数据类型
    using BlockEpilogue = Block::BlockEpilogueStreamK<WorkspaceDtype, OutType,
                                                      MatmulMultiBlockWithStreamK<MatMulL0C2Out::ON_THE_FLY>>;

    // 定义Kernel类型
    using MatmulKernel = Kernel::KernelQBMMStreamk<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    using Params = typename MatmulKernel::Params;
    const DequantBmm::QuantBatchMatmulV3StreamKBasicAPITilingData*
        tilingDataAll = static_cast<const DequantBmm::QuantBatchMatmulV3StreamKBasicAPITilingData*>(tilingData);
    DequantBmm::BasicAPICubeTiling matmulTiling = tilingDataAll->matmulTiling;
    using QBMMTiling = typename MatmulKernel::QBMMTiling;
    QBMMTiling qbmmParams{tilingDataAll->params.batchA1,
                          tilingDataAll->params.batchA2,
                          tilingDataAll->params.batchA3,
                          tilingDataAll->params.batchA4,
                          tilingDataAll->params.batchB1,
                          tilingDataAll->params.batchB2,
                          tilingDataAll->params.batchB3,
                          tilingDataAll->params.batchB4,
                          tilingDataAll->params.batchC1,
                          tilingDataAll->params.batchC2,
                          tilingDataAll->params.batchC3,
                          tilingDataAll->params.batchC4,
                          matmulTiling.baseM,
                          matmulTiling.baseN,
                          matmulTiling.baseK,
                          static_cast<uint32_t>(matmulTiling.isBias),
                          static_cast<uint32_t>(matmulTiling.dbL0C)};

    Params params = {{matmulTiling.m, matmulTiling.n, matmulTiling.k, 1},       // problemshape
                     {aGM, bGM, cGM, workspaceGM, biasGM, pertokenGM, scaleGM}, // mmadParams
                     {tilingDataAll->streamKTiling.kL1, matmulTiling.scaleKL1, matmulTiling.nBufferNum,
                      tilingDataAll->params.x1QuantMode, tilingDataAll->params.x2QuantMode}, // l1Params
                     {cGM, workspaceGM},                                                     // epilogueParams
                     {tilingDataAll->streamKTiling.usedCoreNum, matmulTiling.baseM, matmulTiling.baseN,
                      matmulTiling.baseK, tilingDataAll->streamKTiling.singleCoreK, tilingDataAll->streamKTiling.kL1},
                     qbmmParams};

    MatmulKernel mm;

    mm(params);
}

#endif // QBMM_CUBE_FIXP_QUANT_STREAMK_CMCT_H

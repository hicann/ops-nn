/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file rotate_quant_basic_cmct.h
 * \brief
 */

#ifndef ROTATE_QUANT_BASIC_CMCT_H
#define ROTATE_QUANT_BASIC_CMCT_H

#include "cmct/block/block_scheduler_policy.h"
#include "cmct/block/block_scheduler_utils.h"
#include "cmct/kernel/kernel_rotate_quant.h"
#include "cmct/epilogue/block_epilogue_rotate_quant.h"
#include "block_scheduler_rotate_quant.h"
#include "rotate_quant_tiling_data.h"

namespace RotateQuantNS {
using namespace Cmct;
using namespace Cmct::Gemm;
template <class X_TYPE, class Y_TYPE, class SCALE_TYPE, class C_LAYOUT>
__aicore__ inline void RotateQuantCmctKernel(GM_ADDR xGM, GM_ADDR rotGM, GM_ADDR alphaGM, GM_ADDR yGM, GM_ADDR scaleGM,
                                             GM_ADDR workspaceGM,
                                             const RotateQuantAptOpt::RotateQuantAptTilingData& tilingData)
{
    using L1TileShape = AscendC::Shape<_0, _0, _0>;
    using L0TileShape = AscendC::Shape<_0, _0, _0>;

    using AType = X_TYPE;
    using BType = X_TYPE;
    using CType = bfloat16_t;
    using BiasType = X_TYPE;
    using OutType = Y_TYPE;
    using ScaleType = SCALE_TYPE;

    using LayoutA = C_LAYOUT;
    using LayoutB = C_LAYOUT;
    using LayoutC = C_LAYOUT;

    using BlockScheduler = BuiltInRotateQuantScheduler;

    using DispatchPolicy = MatmulRotateQuant<>;
    using BlockMmad = Block::BlockMmadBuilder<AType, LayoutA, BType, LayoutB, CType, LayoutC, BiasType, LayoutC,
                                              L1TileShape, L0TileShape, BlockScheduler, DispatchPolicy>;

    using FusionOp = Block::DefaultFusion<OutType, CType>;

    using BlockEpilogue = Block::BlockEpilogueRotateQuant<CType, OutType, ScaleType, FusionOp>;

    using ProblemShape = MatmulShape;

    using MatmulKernel = Kernel::KernelRotateQuant<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
    using Params = typename MatmulKernel::Params;
    Params params = {
        {tilingData.M, tilingData.N, tilingData.K, tilingData.B}, {xGM, rotGM, alphaGM}, {yGM, scaleGM}, {&tilingData}};

    MatmulKernel mm;
    mm(params);
}
} // namespace RotateQuantNS
#endif // ROTATE_QUANT_BASIC_CMCT_H
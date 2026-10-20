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

#include <type_traits>
#include "cmct/block/block_scheduler_policy.h"
#include "cmct/block/block_scheduler_utils.h"
#include "cmct/kernel/kernel_rotate_quant.h"
#include "cmct/epilogue/block_epilogue_rotate_quant.h"
#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 9201) || (__NPU_ARCH__ == 9202))
#include "cmct/epilogue/block_epilogue_rotate_quant_hif4.h"
#endif
#include "block_scheduler_rotate_quant.h"
#include "rotate_quant_tiling_data.h"

namespace RotateQuantNS {
using namespace Cmct;
using namespace Cmct::Gemm;

// HIFLOAT4输出分发: v121(9201/9202)下按输出dtype选择epilogue——binary.json的hif4条目
// (y=hifloat4)编译时DTYPE_Y=hifloat4x2_t, 使用hif4 epilogue; 其余走主线原epilogue。
// 3510不支持HIFLOAT4输出, 直接使用主线原epilogue(与master一致)。
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

#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 9201) || (__NPU_ARCH__ == 9202))
    // v121: HIFLOAT4输出(y=hifloat4x2_t)使用hif4 epilogue, 其余使用主线原epilogue(dtype驱动分发)
    // 注: hifloat4x2_t为CCE编译器内置类型(同fp4x2_e2m1_t, 不属于AscendC命名空间, 须裸用)
    using BlockEpilogue = typename std::conditional<
        std::is_same<Y_TYPE, hifloat4x2_t>::value,
        Block::BlockEpilogueRotateQuantHif4<CType, OutType, ScaleType, FusionOp>,
        Block::BlockEpilogueRotateQuant<CType, OutType, ScaleType, FusionOp>>::type;
#else
    using BlockEpilogue = Block::BlockEpilogueRotateQuant<CType, OutType, ScaleType, FusionOp>;
#endif

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

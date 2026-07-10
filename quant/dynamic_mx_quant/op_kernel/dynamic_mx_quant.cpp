/**
 * Copyright (c) 2026-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dynamic_mx_quant.cpp
 * \brief
 */

#include "arch35/dynamic_mx_quant_not_tail_axis.h"
#include "arch35/dynamic_mx_quant_not_tail_axis_optimize.h"
#include "arch35/dynamic_mx_quant_tail_axis.h"
#include "arch35/dynamic_mx_quant_not_tail_axis_fp8.h"
#include "arch35/dynamic_mx_quant_not_tail_axis_optimize_fp8.h"
#include "arch35/dynamic_mx_quant_tail_axis_fp8.h"
#include "arch35/dynamic_mx_quant_post.h"
#include "arch35/dynamic_mx_quant_not_tail_axis_optimize_high_perf_large_tail.h"
#include "arch35/dynamic_mx_quant_not_tail_axis_optimize_high_perf_small_tail.h"
#include "arch35/dynamic_mx_quant_struct.h"

#define FLOAT_OVERFLOW_MODE_CTRL 60

using namespace DynamicMxQuant;

namespace {
template <typename TY>
__aicore__ inline constexpr bool IsFp8Type()
{
    return IsSame<TY, fp8_e4m3fn_t>::value || IsSame<TY, fp8_e5m2_t>::value;
}

template <typename TY>
__aicore__ inline constexpr bool UseInterleavedScalePath()
{
    return IsSame<TY, fp4x2_e2m1_t>::value || IsSame<TY, fp8_e4m3fn_t>::value;
}

template <typename TX, typename TY, bool isTailAxis, uint64_t isOddScale>
__aicore__ inline void RunMainQuantPath(GM_ADDR x, GM_ADDR y, GM_ADDR mxScale, GM_ADDR userWS, GM_ADDR tiling)
{
    GET_TILING_DATA_WITH_STRUCT(DynamicMxQuantTilingData, tilingData, tiling);
    GM_ADDR scaleOut = (isOddScale == TPL_ODD_SCALE) ? userWS : mxScale;
    if constexpr (IsFp8Type<TY>()) {
        DynamicMxQuant::DynamicMxQuantNotTailAxisFP8<TX, TY, isTailAxis> op;
        op.Init(x, y, mxScale, scaleOut, &tilingData);
        op.Process();
    } else {
        DynamicMxQuant::DynamicMxQuantNotTailAxis<TX, TY, isTailAxis> op;
        op.Init(x, y, mxScale, scaleOut, &tilingData);
        op.Process();
    }
    if constexpr (isOddScale == TPL_ODD_SCALE) {
        DynamicMxQuantPost postOp;
        postOp.Init(mxScale, userWS, &tilingData);
        postOp.Process();
    }
}

template <typename TX, typename TY, uint64_t isOddScale>
__aicore__ inline void RunSmallTailQuantPath(GM_ADDR x, GM_ADDR y, GM_ADDR mxScale, GM_ADDR userWS, GM_ADDR tiling)
{
    GET_TILING_DATA_WITH_STRUCT(DynamicMxQuantTilingData, tilingData, tiling);
    constexpr bool useInterleavedScalePath = UseInterleavedScalePath<TY>();
    GM_ADDR scaleOut = (isOddScale == TPL_ODD_SCALE) ? userWS : mxScale;
    if constexpr (IsFp8Type<TY>()) {
        DynamicMxQuant::DynamicMxQuantNotTailAxisOptimizeFP8<TX, TY, useInterleavedScalePath> op;
        op.Init(x, y, mxScale, scaleOut, &tilingData);
        op.Process();
    } else {
        DynamicMxQuant::DynamicMxQuantNotTailAxisOptimize<TX, TY, useInterleavedScalePath> op;
        op.Init(x, y, mxScale, scaleOut, &tilingData);
        op.Process();
    }
    if constexpr (isOddScale == TPL_ODD_SCALE) {
        DynamicMxQuantPost postOp;
        postOp.Init(mxScale, userWS, &tilingData);
        postOp.Process();
    }
}

template <typename TX, typename TY, uint64_t scaleAlg>
__aicore__ inline void RunTailAxisBlock32Path(GM_ADDR x, GM_ADDR y, GM_ADDR mxScale, GM_ADDR tiling)
{
    GET_TILING_DATA_WITH_STRUCT(DynamicMxQuantTailAxisTilingData, tilingData, tiling);
    if constexpr (IsFp8Type<TY>()) {
        DynamicMxQuant::DynamicMxQuantTailAxisFP8<TX, TY, scaleAlg> op;
        op.Init(x, y, mxScale, &tilingData);
        op.Process();
    } else {
        DynamicMxQuant::DynamicMxQuantTailAxis<TX, TY, scaleAlg> op;
        op.Init(x, y, mxScale, &tilingData);
        op.Process();
    }
}

template <uint64_t roundMode>
__aicore__ inline constexpr AscendC::RoundMode getRoundMode()
{
    if (roundMode == TPL_RINT)
        return AscendC::RoundMode::CAST_RINT;
    else if (roundMode == TPL_ROUND)
        return AscendC::RoundMode::CAST_ROUND;
    else if (roundMode == TPL_FLOOR)
        return AscendC::RoundMode::CAST_FLOOR;
    else
        return AscendC::RoundMode::CAST_RINT;
}

template <typename TX, typename TY, uint64_t roundMode, uint64_t scaleAlg>
__aicore__ inline void RunNotTailAxisSmallTailOptiPath(GM_ADDR x, GM_ADDR y, GM_ADDR mxScale, GM_ADDR tiling)
{
    GET_TILING_DATA_WITH_STRUCT(DynamicMxQuant4OptimizeTilingData, tilingData, tiling);
    TPipe pipe;
    DynamicMxQuantNotTailAxisOptimizeSmallTail<DTYPE_X, DTYPE_Y, getRoundMode<roundMode>(), scaleAlg> op;
    op.Init(&pipe, x, y, mxScale, &tilingData);
    op.Process();
}

template <typename TX, typename TY, uint64_t roundMode, uint64_t scaleAlg>
__aicore__ inline void RunNotTailAxisLargeTailOptiPath(GM_ADDR x, GM_ADDR y, GM_ADDR mxScale, GM_ADDR tiling)
{
    GET_TILING_DATA_WITH_STRUCT(DynamicMxQuant4OptimizeTilingData, tilingData, tiling);
    TPipe pipe;
    DynamicMxQuantNotTailAxisOptimizeLargeTail<DTYPE_X, DTYPE_Y, getRoundMode<roundMode>(), scaleAlg> op(&tilingData,
                                                                                                         &pipe);
    op.Init(x, y, mxScale);
    op.Process();
}

} // namespace

template <uint64_t optiMode, uint64_t scaleAlg, uint64_t roundMode, uint64_t isOddScale>
__global__ __aicore__ void dynamic_mx_quant(GM_ADDR x, GM_ADDR y, GM_ADDR mxScale, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(DynamicMxQuant4OptimizeTilingData);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);

#if (__NPU_ARCH__ == 3510)
    int64_t oriOverflowMode = AscendC::GetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>();
#endif

    if constexpr (optiMode == TPL_TAIL_AXIS_QUANT_OPTI) {
        RunTailAxisBlock32Path<DTYPE_X, DTYPE_Y, scaleAlg>(x, y, mxScale, tiling);
    } else if constexpr (optiMode == TPL_NOT_TAIL_AXIS_QUANT_SMALL_OPTI) {
        RunNotTailAxisSmallTailOptiPath<DTYPE_X, DTYPE_Y, roundMode, scaleAlg>(x, y, mxScale, tiling);
    } else if constexpr (optiMode == TPL_NOT_TAIL_AXIS_QUANT_LARGE_OPTI) {
        RunNotTailAxisLargeTailOptiPath<DTYPE_X, DTYPE_Y, roundMode, scaleAlg>(x, y, mxScale, tiling);
    } else if constexpr (optiMode == TPL_TAIL_AXIS_QUANT_NORMAL) {
        RunMainQuantPath<DTYPE_X, DTYPE_Y, true, isOddScale>(x, y, mxScale, workspace, tiling);
    } else if constexpr (optiMode == TPL_NOT_TAIL_AXIS_QUANT_OPTI) {
        RunSmallTailQuantPath<DTYPE_X, DTYPE_Y, isOddScale>(x, y, mxScale, workspace, tiling);
    } else {
        RunMainQuantPath<DTYPE_X, DTYPE_Y, false, isOddScale>(x, y, mxScale, workspace, tiling);
    }

#if (__NPU_ARCH__ == 3510)
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(oriOverflowMode);
#endif
}

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

#define TILING_KEY_QUANT_TAIL_AXIS 100
#define TILING_KEY_QUANT_TAIL_AXIS_ODD_SCALE 101
#define TILING_KEY_QUANT_OTHER_AXIS 110
#define TILING_KEY_QUANT_OTHER_AXIS_ODD_SCALE 111
#define TILING_KEY_QUANT_SMALL_TAIL_AXIS 120
#define TILING_KEY_QUANT_SMALL_TAIL_AXIS_ODD_SCALE 121

// 主模板 tilingKey 与 host 侧保持一一对应，只描述算法路径：
// 100/101: 尾轴量化，101 需要借助 workspace 做 scale post 处理。
// 110/111: 非尾轴大尾场景，111 需要做 scale post 处理。
// 120/121: 非尾轴小尾场景，121 需要做 scale post 处理。
// 输入/输出 dtype 不再展开进 tilingKey，而是由当前 binary 的 DTYPE_X/DTYPE_Y 宏决定。

/*
 * 非尾轴优化模板
 * 万位数为1、2，分别代表融合尾轴大于128和融合尾轴小于等于128
 * 十位数表示舍入模式，0,1,4分别代表round，floor，rint
 * 个位数表示计算模式，可取0,1,2,3
 */
#define TILING_KEY_OPT_FOR_NOT_LAST_QUANT_AXIS_SMALL_ROUND_MODEZERO 10000
#define TILING_KEY_OPT_FOR_NOT_LAST_QUANT_AXIS_SMALL_FLOOR_MODEZERO 10010
#define TILING_KEY_OPT_FOR_NOT_LAST_QUANT_AXIS_SMALL_RINT_MODEZERO 10040
#define TILING_KEY_OPT_FOR_NOT_LAST_QUANT_AXIS_SMALL_ROUND_MODEONE 10001
#define TILING_KEY_OPT_FOR_NOT_LAST_QUANT_AXIS_SMALL_FLOOR_MODEONE 10011
#define TILING_KEY_OPT_FOR_NOT_LAST_QUANT_AXIS_SMALL_RINT_MODEONE 10041
#define TILING_KEY_OPT_FOR_NOT_LAST_QUANT_AXIS_SMALL_ROUND_MODETWO 10002
#define TILING_KEY_OPT_FOR_NOT_LAST_QUANT_AXIS_SMALL_FLOOR_MODETWO 10012
#define TILING_KEY_OPT_FOR_NOT_LAST_QUANT_AXIS_SMALL_RINT_MODETWO 10042
#define TILING_KEY_OPT_FOR_NOT_LAST_QUANT_AXIS_SMALL_ROUND_MODETHREE 10003
#define TILING_KEY_OPT_FOR_NOT_LAST_QUANT_AXIS_SMALL_FLOOR_MODETHREE 10013
#define TILING_KEY_OPT_FOR_NOT_LAST_QUANT_AXIS_SMALL_RINT_MODETHREE 10043
#define TILING_KEY_OPT_FOR_NOT_LAST_QUANT_AXIS_LARGE_ROUND_MODEZERO 20000
#define TILING_KEY_OPT_FOR_NOT_LAST_QUANT_AXIS_LARGE_FLOOR_MODEZERO 20010
#define TILING_KEY_OPT_FOR_NOT_LAST_QUANT_AXIS_LARGE_RINT_MODEZERO 20040
#define TILING_KEY_OPT_FOR_NOT_LAST_QUANT_AXIS_LARGE_ROUND_MODEONE 20001
#define TILING_KEY_OPT_FOR_NOT_LAST_QUANT_AXIS_LARGE_FLOOR_MODEONE 20011
#define TILING_KEY_OPT_FOR_NOT_LAST_QUANT_AXIS_LARGE_RINT_MODEONE 20041
#define TILING_KEY_OPT_FOR_NOT_LAST_QUANT_AXIS_LARGE_ROUND_MODETWO 20002
#define TILING_KEY_OPT_FOR_NOT_LAST_QUANT_AXIS_LARGE_FLOOR_MODETWO 20012
#define TILING_KEY_OPT_FOR_NOT_LAST_QUANT_AXIS_LARGE_RINT_MODETWO 20042
#define TILING_KEY_OPT_FOR_NOT_LAST_QUANT_AXIS_LARGE_ROUND_MODETHREE 20003
#define TILING_KEY_OPT_FOR_NOT_LAST_QUANT_AXIS_LARGE_FLOOR_MODETHREE 20013
#define TILING_KEY_OPT_FOR_NOT_LAST_QUANT_AXIS_LARGE_RINT_MODETHREE 20043

// 量化轴为尾轴且 BlockSize 为 32 时，仅由 scaleAlg 决定路径，dtype 由 DTYPE_X/DTYPE_Y 编译宏确定
#define TILING_KEY_TAIL_AXIS_SCALE_ALG_ZERO 10
#define TILING_KEY_TAIL_AXIS_SCALE_ALG_ONE 11
#define TILING_KEY_TAIL_AXIS_SCALE_ALG_TWO 12

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

template <typename TX, typename TY, bool isTailAxis, bool needPost>
__aicore__ inline void RunMainQuantPath(GM_ADDR x, GM_ADDR y, GM_ADDR mxScale, GM_ADDR userWS, GM_ADDR tiling)
{
    GET_TILING_DATA_WITH_STRUCT(DynamicMxQuantTilingData, tilingData, tiling);
    GM_ADDR scaleOut = needPost ? userWS : mxScale;
    if constexpr (IsFp8Type<TY>()) {
        DynamicMxQuant::DynamicMxQuantNotTailAxisFP8<TX, TY, isTailAxis> op;
        op.Init(x, y, mxScale, scaleOut, &tilingData);
        op.Process();
    } else {
        DynamicMxQuant::DynamicMxQuantNotTailAxis<TX, TY, isTailAxis> op;
        op.Init(x, y, mxScale, scaleOut, &tilingData);
        op.Process();
    }
    if constexpr (needPost) {
        DynamicMxQuantPost postOp;
        postOp.Init(mxScale, userWS, &tilingData);
        postOp.Process();
    }
}

template <typename TX, typename TY, bool needPost>
__aicore__ inline void RunSmallTailQuantPath(GM_ADDR x, GM_ADDR y, GM_ADDR mxScale, GM_ADDR userWS, GM_ADDR tiling)
{
    GET_TILING_DATA_WITH_STRUCT(DynamicMxQuantTilingData, tilingData, tiling);
    constexpr bool useInterleavedScalePath = UseInterleavedScalePath<TY>();
    GM_ADDR scaleOut = needPost ? userWS : mxScale;
    if constexpr (IsFp8Type<TY>()) {
        DynamicMxQuant::DynamicMxQuantNotTailAxisOptimizeFP8<TX, TY, useInterleavedScalePath> op;
        op.Init(x, y, mxScale, scaleOut, &tilingData);
        op.Process();
    } else {
        DynamicMxQuant::DynamicMxQuantNotTailAxisOptimize<TX, TY, useInterleavedScalePath> op;
        op.Init(x, y, mxScale, scaleOut, &tilingData);
        op.Process();
    }
    if constexpr (needPost) {
        DynamicMxQuantPost postOp;
        postOp.Init(mxScale, userWS, &tilingData);
        postOp.Process();
    }
}

template <typename TX, typename TY, int scaleAlg>
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
} // namespace

extern "C" __global__ __aicore__ void dynamic_mx_quant(
    GM_ADDR x, GM_ADDR y, GM_ADDR mxScale, GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }

    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }
    REGISTER_TILING_DEFAULT(DynamicMxQuant4OptimizeTilingData);
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR >= 100 && TILING_KEY_VAR <= 199", DynamicMxQuantTilingData);
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR >= 10 && TILING_KEY_VAR <= 12", DynamicMxQuantTailAxisTilingData);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);

#if (__NPU_ARCH__ == 3510)
    int64_t oriOverflowMode = AscendC::GetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>();
#endif

    // 这里的分支只消费 host 计算出的算法路径；具体 dtype 组合已经在编译期实例化完成。
    if (TILING_KEY_IS(TILING_KEY_QUANT_TAIL_AXIS)) {
        RunMainQuantPath<DTYPE_X, DTYPE_Y, true, false>(x, y, mxScale, userWS, tiling);
    } else if (TILING_KEY_IS(TILING_KEY_QUANT_TAIL_AXIS_ODD_SCALE)) {
        RunMainQuantPath<DTYPE_X, DTYPE_Y, true, true>(x, y, mxScale, userWS, tiling);
    } else if (TILING_KEY_IS(TILING_KEY_QUANT_OTHER_AXIS)) {
        RunMainQuantPath<DTYPE_X, DTYPE_Y, false, false>(x, y, mxScale, userWS, tiling);
    } else if (TILING_KEY_IS(TILING_KEY_QUANT_OTHER_AXIS_ODD_SCALE)) {
        RunMainQuantPath<DTYPE_X, DTYPE_Y, false, true>(x, y, mxScale, userWS, tiling);
    } else if (TILING_KEY_IS(TILING_KEY_QUANT_SMALL_TAIL_AXIS)) {
        RunSmallTailQuantPath<DTYPE_X, DTYPE_Y, false>(x, y, mxScale, userWS, tiling);
    } else if (TILING_KEY_IS(TILING_KEY_QUANT_SMALL_TAIL_AXIS_ODD_SCALE)) {
        RunSmallTailQuantPath<DTYPE_X, DTYPE_Y, true>(x, y, mxScale, userWS, tiling);
    } else if (TILING_KEY_IS(TILING_KEY_OPT_FOR_NOT_LAST_QUANT_AXIS_SMALL_ROUND_MODEZERO)) {
        GET_TILING_DATA_WITH_STRUCT(DynamicMxQuant4OptimizeTilingData, tilingData, tiling);
        TPipe pipe;
        DynamicMxQuantNotTailAxisOptimizeHighPerf<DTYPE_X, DTYPE_Y, AscendC::RoundMode::CAST_ROUND, 0> op;
        op.Init(&pipe, x, y, mxScale, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_OPT_FOR_NOT_LAST_QUANT_AXIS_SMALL_FLOOR_MODEZERO)) {
        GET_TILING_DATA_WITH_STRUCT(DynamicMxQuant4OptimizeTilingData, tilingData, tiling);
        TPipe pipe;
        DynamicMxQuantNotTailAxisOptimizeHighPerf<DTYPE_X, DTYPE_Y, AscendC::RoundMode::CAST_FLOOR, 0> op;
        op.Init(&pipe, x, y, mxScale, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_OPT_FOR_NOT_LAST_QUANT_AXIS_SMALL_RINT_MODEZERO)) {
        GET_TILING_DATA_WITH_STRUCT(DynamicMxQuant4OptimizeTilingData, tilingData, tiling);
        TPipe pipe;
        DynamicMxQuantNotTailAxisOptimizeHighPerf<DTYPE_X, DTYPE_Y, AscendC::RoundMode::CAST_RINT, 0> op;
        op.Init(&pipe, x, y, mxScale, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_OPT_FOR_NOT_LAST_QUANT_AXIS_SMALL_RINT_MODEONE)) {
        GET_TILING_DATA_WITH_STRUCT(DynamicMxQuant4OptimizeTilingData, tilingData, tiling);
        TPipe pipe;
        DynamicMxQuantNotTailAxisOptimizeHighPerf<DTYPE_X, DTYPE_Y, AscendC::RoundMode::CAST_RINT, 1> op;
        op.Init(&pipe, x, y, mxScale, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_OPT_FOR_NOT_LAST_QUANT_AXIS_SMALL_ROUND_MODETWO)) {
        GET_TILING_DATA_WITH_STRUCT(DynamicMxQuant4OptimizeTilingData, tilingData, tiling);
        TPipe pipe;
        DynamicMxQuantNotTailAxisOptimizeHighPerf<DTYPE_X, DTYPE_Y, AscendC::RoundMode::CAST_ROUND, 2> op;
        op.Init(&pipe, x, y, mxScale, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_OPT_FOR_NOT_LAST_QUANT_AXIS_SMALL_FLOOR_MODETWO)) {
        GET_TILING_DATA_WITH_STRUCT(DynamicMxQuant4OptimizeTilingData, tilingData, tiling);
        TPipe pipe;
        DynamicMxQuantNotTailAxisOptimizeHighPerf<DTYPE_X, DTYPE_Y, AscendC::RoundMode::CAST_FLOOR, 2> op;
        op.Init(&pipe, x, y, mxScale, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_OPT_FOR_NOT_LAST_QUANT_AXIS_SMALL_RINT_MODETWO)) {
        GET_TILING_DATA_WITH_STRUCT(DynamicMxQuant4OptimizeTilingData, tilingData, tiling);
        TPipe pipe;
        DynamicMxQuantNotTailAxisOptimizeHighPerf<DTYPE_X, DTYPE_Y, AscendC::RoundMode::CAST_RINT, 2> op;
        op.Init(&pipe, x, y, mxScale, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_OPT_FOR_NOT_LAST_QUANT_AXIS_SMALL_ROUND_MODETHREE)) {
        GET_TILING_DATA_WITH_STRUCT(DynamicMxQuant4OptimizeTilingData, tilingData, tiling);
        TPipe pipe;
        DynamicMxQuantNotTailAxisOptimizeHighPerf<DTYPE_X, DTYPE_Y, AscendC::RoundMode::CAST_ROUND, 3> op;
        op.Init(&pipe, x, y, mxScale, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_OPT_FOR_NOT_LAST_QUANT_AXIS_SMALL_FLOOR_MODETHREE)) {
        GET_TILING_DATA_WITH_STRUCT(DynamicMxQuant4OptimizeTilingData, tilingData, tiling);
        TPipe pipe;
        DynamicMxQuantNotTailAxisOptimizeHighPerf<DTYPE_X, DTYPE_Y, AscendC::RoundMode::CAST_FLOOR, 3> op;
        op.Init(&pipe, x, y, mxScale, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_OPT_FOR_NOT_LAST_QUANT_AXIS_SMALL_RINT_MODETHREE)) {
        GET_TILING_DATA_WITH_STRUCT(DynamicMxQuant4OptimizeTilingData, tilingData, tiling);
        TPipe pipe;
        DynamicMxQuantNotTailAxisOptimizeHighPerf<DTYPE_X, DTYPE_Y, AscendC::RoundMode::CAST_RINT, 3> op;
        op.Init(&pipe, x, y, mxScale, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_OPT_FOR_NOT_LAST_QUANT_AXIS_LARGE_ROUND_MODEZERO)) {
        GET_TILING_DATA_WITH_STRUCT(DynamicMxQuant4OptimizeTilingData, tilingData, tiling);
        TPipe pipe;
        DynamicMxQuant::DynamicMxQuantHP2000<DTYPE_X, DTYPE_Y, AscendC::RoundMode::CAST_ROUND, 0> op(
            &tilingData, &pipe);
        op.Init(x, y, mxScale);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_OPT_FOR_NOT_LAST_QUANT_AXIS_LARGE_FLOOR_MODEZERO)) {
        GET_TILING_DATA_WITH_STRUCT(DynamicMxQuant4OptimizeTilingData, tilingData, tiling);
        TPipe pipe;
        DynamicMxQuant::DynamicMxQuantHP2000<DTYPE_X, DTYPE_Y, AscendC::RoundMode::CAST_FLOOR, 0> op(
            &tilingData, &pipe);
        op.Init(x, y, mxScale);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_OPT_FOR_NOT_LAST_QUANT_AXIS_LARGE_RINT_MODEZERO)) {
        GET_TILING_DATA_WITH_STRUCT(DynamicMxQuant4OptimizeTilingData, tilingData, tiling);
        TPipe pipe;
        DynamicMxQuant::DynamicMxQuantHP2000<DTYPE_X, DTYPE_Y, AscendC::RoundMode::CAST_RINT, 0> op(&tilingData, &pipe);
        op.Init(x, y, mxScale);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_OPT_FOR_NOT_LAST_QUANT_AXIS_LARGE_RINT_MODEONE)) {
        GET_TILING_DATA_WITH_STRUCT(DynamicMxQuant4OptimizeTilingData, tilingData, tiling);
        TPipe pipe;
        DynamicMxQuant::DynamicMxQuantHP2000<DTYPE_X, DTYPE_Y, AscendC::RoundMode::CAST_RINT, 1> op(&tilingData, &pipe);
        op.Init(x, y, mxScale);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_OPT_FOR_NOT_LAST_QUANT_AXIS_LARGE_ROUND_MODETWO)) {
        GET_TILING_DATA_WITH_STRUCT(DynamicMxQuant4OptimizeTilingData, tilingData, tiling);
        TPipe pipe;
        DynamicMxQuant::DynamicMxQuantHP2000<DTYPE_X, DTYPE_Y, AscendC::RoundMode::CAST_ROUND, 2> op(
            &tilingData, &pipe);
        op.Init(x, y, mxScale);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_OPT_FOR_NOT_LAST_QUANT_AXIS_LARGE_FLOOR_MODETWO)) {
        GET_TILING_DATA_WITH_STRUCT(DynamicMxQuant4OptimizeTilingData, tilingData, tiling);
        TPipe pipe;
        DynamicMxQuant::DynamicMxQuantHP2000<DTYPE_X, DTYPE_Y, AscendC::RoundMode::CAST_FLOOR, 2> op(
            &tilingData, &pipe);
        op.Init(x, y, mxScale);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_OPT_FOR_NOT_LAST_QUANT_AXIS_LARGE_RINT_MODETWO)) {
        GET_TILING_DATA_WITH_STRUCT(DynamicMxQuant4OptimizeTilingData, tilingData, tiling);
        TPipe pipe;
        DynamicMxQuant::DynamicMxQuantHP2000<DTYPE_X, DTYPE_Y, AscendC::RoundMode::CAST_RINT, 2> op(&tilingData, &pipe);
        op.Init(x, y, mxScale);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_OPT_FOR_NOT_LAST_QUANT_AXIS_LARGE_ROUND_MODETHREE)) {
        GET_TILING_DATA_WITH_STRUCT(DynamicMxQuant4OptimizeTilingData, tilingData, tiling);
        TPipe pipe;
        DynamicMxQuant::DynamicMxQuantHP2000<DTYPE_X, DTYPE_Y, AscendC::RoundMode::CAST_ROUND, 3> op(
            &tilingData, &pipe);
        op.Init(x, y, mxScale);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_OPT_FOR_NOT_LAST_QUANT_AXIS_LARGE_FLOOR_MODETHREE)) {
        GET_TILING_DATA_WITH_STRUCT(DynamicMxQuant4OptimizeTilingData, tilingData, tiling);
        TPipe pipe;
        DynamicMxQuant::DynamicMxQuantHP2000<DTYPE_X, DTYPE_Y, AscendC::RoundMode::CAST_FLOOR, 3> op(
            &tilingData, &pipe);
        op.Init(x, y, mxScale);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_OPT_FOR_NOT_LAST_QUANT_AXIS_LARGE_RINT_MODETHREE)) {
        GET_TILING_DATA_WITH_STRUCT(DynamicMxQuant4OptimizeTilingData, tilingData, tiling);
        TPipe pipe;
        DynamicMxQuant::DynamicMxQuantHP2000<DTYPE_X, DTYPE_Y, AscendC::RoundMode::CAST_RINT, 3> op(&tilingData, &pipe);
        op.Init(x, y, mxScale);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_TAIL_AXIS_SCALE_ALG_ZERO)) {
        RunTailAxisBlock32Path<DTYPE_X, DTYPE_Y, 0>(x, y, mxScale, tiling);
    } else if (TILING_KEY_IS(TILING_KEY_TAIL_AXIS_SCALE_ALG_ONE)) {
        RunTailAxisBlock32Path<DTYPE_X, DTYPE_Y, 1>(x, y, mxScale, tiling);
    } else if (TILING_KEY_IS(TILING_KEY_TAIL_AXIS_SCALE_ALG_TWO)) {
        RunTailAxisBlock32Path<DTYPE_X, DTYPE_Y, 2>(x, y, mxScale, tiling);
    }
#if (__NPU_ARCH__ == 3510)
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(oriOverflowMode);
#endif
}

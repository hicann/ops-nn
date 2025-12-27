/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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

#define TILING_KEY_FP16_FP4E2M1_QUANT_TAIL_AXIS 1000
#define TILING_KEY_BF16_FP4E2M1_QUANT_TAIL_AXIS 2000
#define TILING_KEY_FP16_FP4E2M1_QUANT_OTHER_AXIS 1010
#define TILING_KEY_BF16_FP4E2M1_QUANT_OTHER_AXIS 2010
#define TILING_KEY_FP16_FP4E1M2_QUANT_TAIL_AXIS 1100
#define TILING_KEY_BF16_FP4E1M2_QUANT_TAIL_AXIS 2100
#define TILING_KEY_FP16_FP4E1M2_QUANT_OTHER_AXIS 1110
#define TILING_KEY_BF16_FP4E1M2_QUANT_OTHER_AXIS 2110
#define TILING_KEY_FP16_FP4E2M1_QUANT_SMALL_TAIL_AXIS 1020
#define TILING_KEY_BF16_FP4E2M1_QUANT_SMALL_TAIL_AXIS 2020
#define TILING_KEY_FP16_FP4E1M2_QUANT_SMALL_TAIL_AXIS 1120
#define TILING_KEY_BF16_FP4E1M2_QUANT_SMALL_TAIL_AXIS 2120
#define TILING_KEY_FP16_FP4E2M1_QUANT_TAIL_AXIS_PERF 1040
#define TILING_KEY_BF16_FP4E2M1_QUANT_TAIL_AXIS_PERF 2040
#define TILING_KEY_FP16_FP4E1M2_QUANT_TAIL_AXIS_PERF 1140
#define TILING_KEY_BF16_FP4E1M2_QUANT_TAIL_AXIS_PERF 2140

#define TILING_KEY_FP16_FP8E4M3FN_QUANT_TAIL_AXIS 1200
#define TILING_KEY_BF16_FP8E4M3FN_QUANT_TAIL_AXIS 2200
#define TILING_KEY_FP16_FP8E4M3FN_QUANT_OTHER_AXIS 1210
#define TILING_KEY_BF16_FP8E4M3FN_QUANT_OTHER_AXIS 2210
#define TILING_KEY_FP16_FP8E5M2_QUANT_TAIL_AXIS 1300
#define TILING_KEY_BF16_FP8E5M2_QUANT_TAIL_AXIS 2300
#define TILING_KEY_FP16_FP8E5M2_QUANT_OTHER_AXIS 1310
#define TILING_KEY_BF16_FP8E5M2_QUANT_OTHER_AXIS 2310
#define TILING_KEY_FP16_FP8E4M3FN_QUANT_SMALL_TAIL_AXIS 1220
#define TILING_KEY_BF16_FP8E4M3FN_QUANT_SMALL_TAIL_AXIS 2220
#define TILING_KEY_FP16_FP8E5M2_QUANT_SMALL_TAIL_AXIS 1320
#define TILING_KEY_BF16_FP8E5M2_QUANT_SMALL_TAIL_AXIS 2320

#define TILING_KEY_FP16_FP8E4M3FN_QUANT_TAIL_AXIS_PERF 1240
#define TILING_KEY_BF16_FP8E4M3FN_QUANT_TAIL_AXIS_PERF 2240
#define TILING_KEY_FP16_FP8E5M2_QUANT_TAIL_AXIS_PERF 1340
#define TILING_KEY_BF16_FP8E5M2_QUANT_TAIL_AXIS_PERF 2340

#define TILING_KEY_FP16_FP4E2M1_QUANT_TAIL_AXIS_ODD_SCALE 1001
#define TILING_KEY_BF16_FP4E2M1_QUANT_TAIL_AXIS_ODD_SCALE 2001
#define TILING_KEY_FP16_FP4E2M1_QUANT_OTHER_AXIS_ODD_SCALE 1011
#define TILING_KEY_BF16_FP4E2M1_QUANT_OTHER_AXIS_ODD_SCALE 2011
#define TILING_KEY_FP16_FP4E1M2_QUANT_TAIL_AXIS_ODD_SCALE 1101
#define TILING_KEY_BF16_FP4E1M2_QUANT_TAIL_AXIS_ODD_SCALE 2101
#define TILING_KEY_FP16_FP4E1M2_QUANT_OTHER_AXIS_ODD_SCALE 1111
#define TILING_KEY_BF16_FP4E1M2_QUANT_OTHER_AXIS_ODD_SCALE 2111
#define TILING_KEY_FP16_FP4E2M1_QUANT_SMALL_TAIL_AXIS_ODD_SCALE 1021
#define TILING_KEY_BF16_FP4E2M1_QUANT_SMALL_TAIL_AXIS_ODD_SCALE 2021
#define TILING_KEY_FP16_FP4E1M2_QUANT_SMALL_TAIL_AXIS_ODD_SCALE 1121
#define TILING_KEY_BF16_FP4E1M2_QUANT_SMALL_TAIL_AXIS_ODD_SCALE 2121
#define TILING_KEY_FP16_FP4E2M1_QUANT_TAIL_AXIS_PERF_ODD_SCALE 1041
#define TILING_KEY_BF16_FP4E2M1_QUANT_TAIL_AXIS_PERF_ODD_SCALE 2041
#define TILING_KEY_FP16_FP4E1M2_QUANT_TAIL_AXIS_PERF_ODD_SCALE 1141
#define TILING_KEY_BF16_FP4E1M2_QUANT_TAIL_AXIS_PERF_ODD_SCALE 2141

#define TILING_KEY_FP16_FP8E4M3FN_QUANT_TAIL_AXIS_ODD_SCALE 1201
#define TILING_KEY_BF16_FP8E4M3FN_QUANT_TAIL_AXIS_ODD_SCALE 2201
#define TILING_KEY_FP16_FP8E4M3FN_QUANT_OTHER_AXIS_ODD_SCALE 1211
#define TILING_KEY_BF16_FP8E4M3FN_QUANT_OTHER_AXIS_ODD_SCALE 2211
#define TILING_KEY_FP16_FP8E5M2_QUANT_TAIL_AXIS_ODD_SCALE 1301
#define TILING_KEY_BF16_FP8E5M2_QUANT_TAIL_AXIS_ODD_SCALE 2301
#define TILING_KEY_FP16_FP8E5M2_QUANT_OTHER_AXIS_ODD_SCALE 1311
#define TILING_KEY_BF16_FP8E5M2_QUANT_OTHER_AXIS_ODD_SCALE 2311
#define TILING_KEY_FP16_FP8E4M3FN_QUANT_SMALL_TAIL_AXIS_ODD_SCALE 1221
#define TILING_KEY_BF16_FP8E4M3FN_QUANT_SMALL_TAIL_AXIS_ODD_SCALE 2221
#define TILING_KEY_FP16_FP8E5M2_QUANT_SMALL_TAIL_AXIS_ODD_SCALE 1321
#define TILING_KEY_BF16_FP8E5M2_QUANT_SMALL_TAIL_AXIS_ODD_SCALE 2321

#define TILING_KEY_FP16_FP8E4M3FN_QUANT_TAIL_AXIS_PERF_ODD_SCALE 1241
#define TILING_KEY_BF16_FP8E4M3FN_QUANT_TAIL_AXIS_PERF_ODD_SCALE 2241
#define TILING_KEY_FP16_FP8E5M2_QUANT_TAIL_AXIS_PERF_ODD_SCALE 1341
#define TILING_KEY_BF16_FP8E5M2_QUANT_TAIL_AXIS_PERF_ODD_SCALE 2341

#define TILING_KEY_OPT_FOR_NOT_LAST_QUANT_AXIS_SMALL 10000
#define TILING_KEY_OPT_FOR_NOT_LAST_QUANT_AXIS_LARGE 20000

#define FLOAT_OVERFLOW_MODE_CTRL 60

// 千位数为1、2，分别表示输入类型是float16、bfloat16;
// 百位数为0、1、2、3，分别表示输出类型是float4_e2m1、float4_e1m2、float8_e4m3fn、float8_e5m2
// 十位数为0、1、2、4，分别表示axis尾轴 + 非优化（blocksize != 32)
// 个位数为0、1，分别表示尾轴偶数场景（该场景无需post操作）和其他场景
// axis非尾轴 + 非优化（axis后的轴合轴后元素个数大于VL/2）、
// axis非尾轴 + 优化（axis后的轴合轴后元素个数小于等于VL/2）
// axis尾轴 + 优化（blocksize == 32)四种场景。

using namespace DynamicMxQuant;

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

    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);

#if (__NPU_ARCH__ == 3101)
    int64_t oriOverflowMode = AscendC::GetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>();
#endif

    if (TILING_KEY_IS(TILING_KEY_FP16_FP4E2M1_QUANT_TAIL_AXIS)) {
        GET_TILING_DATA_WITH_STRUCT(DynamicMxQuantTilingData, tilingData, tiling);
        DynamicMxQuant::DynamicMxQuantNotTailAxis<half, fp4x2_e2m1_t, true> op;
        op.Init(x, y, mxScale, mxScale, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_BF16_FP4E2M1_QUANT_TAIL_AXIS)) {
        GET_TILING_DATA_WITH_STRUCT(DynamicMxQuantTilingData, tilingData, tiling);
        DynamicMxQuant::DynamicMxQuantNotTailAxis<bfloat16_t, fp4x2_e2m1_t, true> op;
        op.Init(x, y, mxScale, mxScale, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_FP16_FP4E2M1_QUANT_OTHER_AXIS)) {
        GET_TILING_DATA_WITH_STRUCT(DynamicMxQuantTilingData, tilingData, tiling);
        DynamicMxQuant::DynamicMxQuantNotTailAxis<half, fp4x2_e2m1_t, false> op;
        op.Init(x, y, mxScale, mxScale, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_BF16_FP4E2M1_QUANT_OTHER_AXIS)) {
        GET_TILING_DATA_WITH_STRUCT(DynamicMxQuantTilingData, tilingData, tiling);
        DynamicMxQuant::DynamicMxQuantNotTailAxis<bfloat16_t, fp4x2_e2m1_t, false> op;
        op.Init(x, y, mxScale, mxScale, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_FP16_FP4E1M2_QUANT_TAIL_AXIS)) {
        GET_TILING_DATA_WITH_STRUCT(DynamicMxQuantTilingData, tilingData, tiling);
        DynamicMxQuant::DynamicMxQuantNotTailAxis<half, fp4x2_e1m2_t, true> op;
        op.Init(x, y, mxScale, mxScale, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_BF16_FP4E1M2_QUANT_TAIL_AXIS)) {
        GET_TILING_DATA_WITH_STRUCT(DynamicMxQuantTilingData, tilingData, tiling);
        DynamicMxQuant::DynamicMxQuantNotTailAxis<bfloat16_t, fp4x2_e1m2_t, true> op;
        op.Init(x, y, mxScale, mxScale, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_FP16_FP4E1M2_QUANT_OTHER_AXIS)) {
        GET_TILING_DATA_WITH_STRUCT(DynamicMxQuantTilingData, tilingData, tiling);
        DynamicMxQuant::DynamicMxQuantNotTailAxis<half, fp4x2_e1m2_t, false> op;
        op.Init(x, y, mxScale, mxScale, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_BF16_FP4E1M2_QUANT_OTHER_AXIS)) {
        GET_TILING_DATA_WITH_STRUCT(DynamicMxQuantTilingData, tilingData, tiling);
        DynamicMxQuant::DynamicMxQuantNotTailAxis<bfloat16_t, fp4x2_e1m2_t, false> op;
        op.Init(x, y, mxScale, mxScale, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_FP16_FP4E2M1_QUANT_SMALL_TAIL_AXIS)) {
        GET_TILING_DATA_WITH_STRUCT(DynamicMxQuantTilingData, tilingData, tiling);
        DynamicMxQuant::DynamicMxQuantNotTailAxisOptimize<half, fp4x2_e2m1_t, true> op;
        op.Init(x, y, mxScale, mxScale, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_BF16_FP4E2M1_QUANT_SMALL_TAIL_AXIS)) {
        GET_TILING_DATA_WITH_STRUCT(DynamicMxQuantTilingData, tilingData, tiling);
        DynamicMxQuant::DynamicMxQuantNotTailAxisOptimize<bfloat16_t, fp4x2_e2m1_t, true> op;
        op.Init(x, y, mxScale, mxScale, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_FP16_FP4E1M2_QUANT_SMALL_TAIL_AXIS)) {
        GET_TILING_DATA_WITH_STRUCT(DynamicMxQuantTilingData, tilingData, tiling);
        DynamicMxQuant::DynamicMxQuantNotTailAxisOptimize<half, fp4x2_e1m2_t, false> op;
        op.Init(x, y, mxScale, mxScale, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_BF16_FP4E1M2_QUANT_SMALL_TAIL_AXIS)) {
        GET_TILING_DATA_WITH_STRUCT(DynamicMxQuantTilingData, tilingData, tiling);
        DynamicMxQuant::DynamicMxQuantNotTailAxisOptimize<bfloat16_t, fp4x2_e1m2_t, false> op;
        op.Init(x, y, mxScale, mxScale, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_FP16_FP4E2M1_QUANT_TAIL_AXIS_PERF)) {
        GET_TILING_DATA_WITH_STRUCT(DynamicMxQuantTilingData, tilingData, tiling);
        DynamicMxQuant::DynamicMxQuantTailAxis<half, fp4x2_e2m1_t> op;
        op.Init(x, y, mxScale, mxScale, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_BF16_FP4E2M1_QUANT_TAIL_AXIS_PERF)) {
        GET_TILING_DATA_WITH_STRUCT(DynamicMxQuantTilingData, tilingData, tiling);
        DynamicMxQuant::DynamicMxQuantTailAxis<bfloat16_t, fp4x2_e2m1_t> op;
        op.Init(x, y, mxScale, mxScale, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_FP16_FP4E1M2_QUANT_TAIL_AXIS_PERF)) {
        GET_TILING_DATA_WITH_STRUCT(DynamicMxQuantTilingData, tilingData, tiling);
        DynamicMxQuant::DynamicMxQuantTailAxis<half, fp4x2_e1m2_t> op;
        op.Init(x, y, mxScale, mxScale, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_BF16_FP4E1M2_QUANT_TAIL_AXIS_PERF)) {
        GET_TILING_DATA_WITH_STRUCT(DynamicMxQuantTilingData, tilingData, tiling);
        DynamicMxQuant::DynamicMxQuantTailAxis<bfloat16_t, fp4x2_e1m2_t> op;
        op.Init(x, y, mxScale, mxScale, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_FP16_FP8E4M3FN_QUANT_TAIL_AXIS)) {
        GET_TILING_DATA_WITH_STRUCT(DynamicMxQuantTilingData, tilingData, tiling);
        DynamicMxQuant::DynamicMxQuantNotTailAxisFP8<half, fp8_e4m3fn_t, true> op;
        op.Init(x, y, mxScale, mxScale, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_BF16_FP8E4M3FN_QUANT_TAIL_AXIS)) {
        GET_TILING_DATA_WITH_STRUCT(DynamicMxQuantTilingData, tilingData, tiling);
        DynamicMxQuant::DynamicMxQuantNotTailAxisFP8<bfloat16_t, fp8_e4m3fn_t, true> op;
        op.Init(x, y, mxScale, mxScale, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_FP16_FP8E4M3FN_QUANT_OTHER_AXIS)) {
        GET_TILING_DATA_WITH_STRUCT(DynamicMxQuantTilingData, tilingData, tiling);
        DynamicMxQuant::DynamicMxQuantNotTailAxisFP8<half, fp8_e4m3fn_t, false> op;
        op.Init(x, y, mxScale, mxScale, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_BF16_FP8E4M3FN_QUANT_OTHER_AXIS)) {
        GET_TILING_DATA_WITH_STRUCT(DynamicMxQuantTilingData, tilingData, tiling);
        DynamicMxQuant::DynamicMxQuantNotTailAxisFP8<bfloat16_t, fp8_e4m3fn_t, false> op;
        op.Init(x, y, mxScale, mxScale, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_FP16_FP8E5M2_QUANT_TAIL_AXIS)) {
        GET_TILING_DATA_WITH_STRUCT(DynamicMxQuantTilingData, tilingData, tiling);
        DynamicMxQuant::DynamicMxQuantNotTailAxisFP8<half, fp8_e5m2_t, true> op;
        op.Init(x, y, mxScale, mxScale, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_BF16_FP8E5M2_QUANT_TAIL_AXIS)) {
        GET_TILING_DATA_WITH_STRUCT(DynamicMxQuantTilingData, tilingData, tiling);
        DynamicMxQuant::DynamicMxQuantNotTailAxisFP8<bfloat16_t, fp8_e5m2_t, true> op;
        op.Init(x, y, mxScale, mxScale, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_FP16_FP8E5M2_QUANT_OTHER_AXIS)) {
        GET_TILING_DATA_WITH_STRUCT(DynamicMxQuantTilingData, tilingData, tiling);
        DynamicMxQuant::DynamicMxQuantNotTailAxisFP8<half, fp8_e5m2_t, false> op;
        op.Init(x, y, mxScale, mxScale, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_BF16_FP8E5M2_QUANT_OTHER_AXIS)) {
        GET_TILING_DATA_WITH_STRUCT(DynamicMxQuantTilingData, tilingData, tiling);
        DynamicMxQuant::DynamicMxQuantNotTailAxisFP8<bfloat16_t, fp8_e5m2_t, false> op;
        op.Init(x, y, mxScale, mxScale, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_FP16_FP8E4M3FN_QUANT_SMALL_TAIL_AXIS)) {
        GET_TILING_DATA_WITH_STRUCT(DynamicMxQuantTilingData, tilingData, tiling);
        DynamicMxQuant::DynamicMxQuantNotTailAxisOptimizeFP8<half, fp8_e4m3fn_t, true> op;
        op.Init(x, y, mxScale, mxScale, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_BF16_FP8E4M3FN_QUANT_SMALL_TAIL_AXIS)) {
        GET_TILING_DATA_WITH_STRUCT(DynamicMxQuantTilingData, tilingData, tiling);
        DynamicMxQuant::DynamicMxQuantNotTailAxisOptimizeFP8<bfloat16_t, fp8_e4m3fn_t, true> op;
        op.Init(x, y, mxScale, mxScale, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_FP16_FP8E5M2_QUANT_SMALL_TAIL_AXIS)) {
        GET_TILING_DATA_WITH_STRUCT(DynamicMxQuantTilingData, tilingData, tiling);
        DynamicMxQuant::DynamicMxQuantNotTailAxisOptimizeFP8<half, fp8_e5m2_t, false> op;
        op.Init(x, y, mxScale, mxScale, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_BF16_FP8E5M2_QUANT_SMALL_TAIL_AXIS)) {
        GET_TILING_DATA_WITH_STRUCT(DynamicMxQuantTilingData, tilingData, tiling);
        DynamicMxQuant::DynamicMxQuantNotTailAxisOptimizeFP8<bfloat16_t, fp8_e5m2_t, false> op;
        op.Init(x, y, mxScale, mxScale, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_FP16_FP8E4M3FN_QUANT_TAIL_AXIS_PERF)) {
        GET_TILING_DATA_WITH_STRUCT(DynamicMxQuantTilingData, tilingData, tiling);
        DynamicMxQuant::DynamicMxQuantTailAxisFP8<half, fp8_e4m3fn_t> op;
        op.Init(x, y, mxScale, mxScale, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_BF16_FP8E4M3FN_QUANT_TAIL_AXIS_PERF)) {
        GET_TILING_DATA_WITH_STRUCT(DynamicMxQuantTilingData, tilingData, tiling);
        DynamicMxQuant::DynamicMxQuantTailAxisFP8<bfloat16_t, fp8_e4m3fn_t> op;
        op.Init(x, y, mxScale, mxScale, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_FP16_FP8E5M2_QUANT_TAIL_AXIS_PERF)) {
        GET_TILING_DATA_WITH_STRUCT(DynamicMxQuantTilingData, tilingData, tiling);
        DynamicMxQuant::DynamicMxQuantTailAxisFP8<half, fp8_e5m2_t> op;
        op.Init(x, y, mxScale, mxScale, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_BF16_FP8E5M2_QUANT_TAIL_AXIS_PERF)) {
        GET_TILING_DATA_WITH_STRUCT(DynamicMxQuantTilingData, tilingData, tiling);
        DynamicMxQuant::DynamicMxQuantTailAxisFP8<bfloat16_t, fp8_e5m2_t> op;
        op.Init(x, y, mxScale, mxScale, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_FP16_FP4E2M1_QUANT_TAIL_AXIS_ODD_SCALE)) {
        GET_TILING_DATA_WITH_STRUCT(DynamicMxQuantTilingData, tilingData, tiling);
        DynamicMxQuant::DynamicMxQuantNotTailAxis<half, fp4x2_e2m1_t, true> op;
        op.Init(x, y, mxScale, userWS, &tilingData);
        op.Process();
        DynamicMxQuantPost postOp;
        postOp.Init(mxScale, userWS, &tilingData);
        postOp.Process();
    } else if (TILING_KEY_IS(TILING_KEY_BF16_FP4E2M1_QUANT_TAIL_AXIS_ODD_SCALE)) {
        GET_TILING_DATA_WITH_STRUCT(DynamicMxQuantTilingData, tilingData, tiling);
        DynamicMxQuant::DynamicMxQuantNotTailAxis<bfloat16_t, fp4x2_e2m1_t, true> op;
        op.Init(x, y, mxScale, userWS, &tilingData);
        op.Process();
        DynamicMxQuantPost postOp;
        postOp.Init(mxScale, userWS, &tilingData);
        postOp.Process();
    } else if (TILING_KEY_IS(TILING_KEY_FP16_FP4E2M1_QUANT_OTHER_AXIS_ODD_SCALE)) {
        GET_TILING_DATA_WITH_STRUCT(DynamicMxQuantTilingData, tilingData, tiling);
        DynamicMxQuant::DynamicMxQuantNotTailAxis<half, fp4x2_e2m1_t, false> op;
        op.Init(x, y, mxScale, userWS, &tilingData);
        op.Process();
        DynamicMxQuantPost postOp;
        postOp.Init(mxScale, userWS, &tilingData);
        postOp.Process();
    } else if (TILING_KEY_IS(TILING_KEY_BF16_FP4E2M1_QUANT_OTHER_AXIS_ODD_SCALE)) {
        GET_TILING_DATA_WITH_STRUCT(DynamicMxQuantTilingData, tilingData, tiling);
        DynamicMxQuant::DynamicMxQuantNotTailAxis<bfloat16_t, fp4x2_e2m1_t, false> op;
        op.Init(x, y, mxScale, userWS, &tilingData);
        op.Process();
        DynamicMxQuantPost postOp;
        postOp.Init(mxScale, userWS, &tilingData);
        postOp.Process();
    } else if (TILING_KEY_IS(TILING_KEY_FP16_FP4E1M2_QUANT_TAIL_AXIS_ODD_SCALE)) {
        GET_TILING_DATA_WITH_STRUCT(DynamicMxQuantTilingData, tilingData, tiling);
        DynamicMxQuant::DynamicMxQuantNotTailAxis<half, fp4x2_e1m2_t, true> op;
        op.Init(x, y, mxScale, userWS, &tilingData);
        op.Process();
        DynamicMxQuantPost postOp;
        postOp.Init(mxScale, userWS, &tilingData);
        postOp.Process();
    } else if (TILING_KEY_IS(TILING_KEY_BF16_FP4E1M2_QUANT_TAIL_AXIS_ODD_SCALE)) {
        GET_TILING_DATA_WITH_STRUCT(DynamicMxQuantTilingData, tilingData, tiling);
        DynamicMxQuant::DynamicMxQuantNotTailAxis<bfloat16_t, fp4x2_e1m2_t, true> op;
        op.Init(x, y, mxScale, userWS, &tilingData);
        op.Process();
        DynamicMxQuantPost postOp;
        postOp.Init(mxScale, userWS, &tilingData);
        postOp.Process();
    } else if (TILING_KEY_IS(TILING_KEY_FP16_FP4E1M2_QUANT_OTHER_AXIS_ODD_SCALE)) {
        GET_TILING_DATA_WITH_STRUCT(DynamicMxQuantTilingData, tilingData, tiling);
        DynamicMxQuant::DynamicMxQuantNotTailAxis<half, fp4x2_e1m2_t, false> op;
        op.Init(x, y, mxScale, userWS, &tilingData);
        op.Process();
        DynamicMxQuantPost postOp;
        postOp.Init(mxScale, userWS, &tilingData);
        postOp.Process();
    } else if (TILING_KEY_IS(TILING_KEY_BF16_FP4E1M2_QUANT_OTHER_AXIS_ODD_SCALE)) {
        GET_TILING_DATA_WITH_STRUCT(DynamicMxQuantTilingData, tilingData, tiling);
        DynamicMxQuant::DynamicMxQuantNotTailAxis<bfloat16_t, fp4x2_e1m2_t, false> op;
        op.Init(x, y, mxScale, userWS, &tilingData);
        op.Process();
        DynamicMxQuantPost postOp;
        postOp.Init(mxScale, userWS, &tilingData);
        postOp.Process();
    } else if (TILING_KEY_IS(TILING_KEY_FP16_FP4E2M1_QUANT_SMALL_TAIL_AXIS_ODD_SCALE)) {
        GET_TILING_DATA_WITH_STRUCT(DynamicMxQuantTilingData, tilingData, tiling);
        DynamicMxQuant::DynamicMxQuantNotTailAxisOptimize<half, fp4x2_e2m1_t, true> op;
        op.Init(x, y, mxScale, userWS, &tilingData);
        op.Process();
        DynamicMxQuantPost postOp;
        postOp.Init(mxScale, userWS, &tilingData);
        postOp.Process();
    } else if (TILING_KEY_IS(TILING_KEY_BF16_FP4E2M1_QUANT_SMALL_TAIL_AXIS_ODD_SCALE)) {
        GET_TILING_DATA_WITH_STRUCT(DynamicMxQuantTilingData, tilingData, tiling);
        DynamicMxQuant::DynamicMxQuantNotTailAxisOptimize<bfloat16_t, fp4x2_e2m1_t, true> op;
        op.Init(x, y, mxScale, userWS, &tilingData);
        op.Process();
        DynamicMxQuantPost postOp;
        postOp.Init(mxScale, userWS, &tilingData);
        postOp.Process();
    } else if (TILING_KEY_IS(TILING_KEY_FP16_FP4E1M2_QUANT_SMALL_TAIL_AXIS_ODD_SCALE)) {
        GET_TILING_DATA_WITH_STRUCT(DynamicMxQuantTilingData, tilingData, tiling);
        DynamicMxQuant::DynamicMxQuantNotTailAxisOptimize<half, fp4x2_e1m2_t, false> op;
        op.Init(x, y, mxScale, userWS, &tilingData);
        op.Process();
        DynamicMxQuantPost postOp;
        postOp.Init(mxScale, userWS, &tilingData);
        postOp.Process();
    } else if (TILING_KEY_IS(TILING_KEY_BF16_FP4E1M2_QUANT_SMALL_TAIL_AXIS_ODD_SCALE)) {
        GET_TILING_DATA_WITH_STRUCT(DynamicMxQuantTilingData, tilingData, tiling);
        DynamicMxQuant::DynamicMxQuantNotTailAxisOptimize<bfloat16_t, fp4x2_e1m2_t, false> op;
        op.Init(x, y, mxScale, userWS, &tilingData);
        op.Process();
        DynamicMxQuantPost postOp;
        postOp.Init(mxScale, userWS, &tilingData);
        postOp.Process();
    } else if (TILING_KEY_IS(TILING_KEY_FP16_FP4E2M1_QUANT_TAIL_AXIS_PERF_ODD_SCALE)) {
        GET_TILING_DATA_WITH_STRUCT(DynamicMxQuantTilingData, tilingData, tiling);
        DynamicMxQuant::DynamicMxQuantTailAxis<half, fp4x2_e2m1_t> op;
        op.Init(x, y, mxScale, userWS, &tilingData);
        op.Process();
        DynamicMxQuantPost postOp;
        postOp.Init(mxScale, userWS, &tilingData);
        postOp.Process();
    } else if (TILING_KEY_IS(TILING_KEY_BF16_FP4E2M1_QUANT_TAIL_AXIS_PERF_ODD_SCALE)) {
        GET_TILING_DATA_WITH_STRUCT(DynamicMxQuantTilingData, tilingData, tiling);
        DynamicMxQuant::DynamicMxQuantTailAxis<bfloat16_t, fp4x2_e2m1_t> op;
        op.Init(x, y, mxScale, userWS, &tilingData);
        op.Process();
        DynamicMxQuantPost postOp;
        postOp.Init(mxScale, userWS, &tilingData);
        postOp.Process();
    } else if (TILING_KEY_IS(TILING_KEY_FP16_FP4E1M2_QUANT_TAIL_AXIS_PERF_ODD_SCALE)) {
        GET_TILING_DATA_WITH_STRUCT(DynamicMxQuantTilingData, tilingData, tiling);
        DynamicMxQuant::DynamicMxQuantTailAxis<half, fp4x2_e1m2_t> op;
        op.Init(x, y, mxScale, userWS, &tilingData);
        op.Process();
        DynamicMxQuantPost postOp;
        postOp.Init(mxScale, userWS, &tilingData);
        postOp.Process();
    } else if (TILING_KEY_IS(TILING_KEY_BF16_FP4E1M2_QUANT_TAIL_AXIS_PERF_ODD_SCALE)) {
        GET_TILING_DATA_WITH_STRUCT(DynamicMxQuantTilingData, tilingData, tiling);
        DynamicMxQuant::DynamicMxQuantTailAxis<bfloat16_t, fp4x2_e1m2_t> op;
        op.Init(x, y, mxScale, userWS, &tilingData);
        op.Process();
        DynamicMxQuantPost postOp;
        postOp.Init(mxScale, userWS, &tilingData);
        postOp.Process();
    } else if (TILING_KEY_IS(TILING_KEY_FP16_FP8E4M3FN_QUANT_TAIL_AXIS_ODD_SCALE)) {
        GET_TILING_DATA_WITH_STRUCT(DynamicMxQuantTilingData, tilingData, tiling);
        DynamicMxQuant::DynamicMxQuantNotTailAxisFP8<half, fp8_e4m3fn_t, true> op;
        op.Init(x, y, mxScale, userWS, &tilingData);
        op.Process();
        DynamicMxQuantPost postOp;
        postOp.Init(mxScale, userWS, &tilingData);
        postOp.Process();
    } else if (TILING_KEY_IS(TILING_KEY_BF16_FP8E4M3FN_QUANT_TAIL_AXIS_ODD_SCALE)) {
        GET_TILING_DATA_WITH_STRUCT(DynamicMxQuantTilingData, tilingData, tiling);
        DynamicMxQuant::DynamicMxQuantNotTailAxisFP8<bfloat16_t, fp8_e4m3fn_t, true> op;
        op.Init(x, y, mxScale, userWS, &tilingData);
        op.Process();
        DynamicMxQuantPost postOp;
        postOp.Init(mxScale, userWS, &tilingData);
        postOp.Process();
    } else if (TILING_KEY_IS(TILING_KEY_FP16_FP8E4M3FN_QUANT_OTHER_AXIS_ODD_SCALE)) {
        GET_TILING_DATA_WITH_STRUCT(DynamicMxQuantTilingData, tilingData, tiling);
        DynamicMxQuant::DynamicMxQuantNotTailAxisFP8<half, fp8_e4m3fn_t, false> op;
        op.Init(x, y, mxScale, userWS, &tilingData);
        op.Process();
        DynamicMxQuantPost postOp;
        postOp.Init(mxScale, userWS, &tilingData);
        postOp.Process();
    } else if (TILING_KEY_IS(TILING_KEY_BF16_FP8E4M3FN_QUANT_OTHER_AXIS_ODD_SCALE)) {
        GET_TILING_DATA_WITH_STRUCT(DynamicMxQuantTilingData, tilingData, tiling);
        DynamicMxQuant::DynamicMxQuantNotTailAxisFP8<bfloat16_t, fp8_e4m3fn_t, false> op;
        op.Init(x, y, mxScale, userWS, &tilingData);
        op.Process();
        DynamicMxQuantPost postOp;
        postOp.Init(mxScale, userWS, &tilingData);
        postOp.Process();
    } else if (TILING_KEY_IS(TILING_KEY_FP16_FP8E5M2_QUANT_TAIL_AXIS_ODD_SCALE)) {
        GET_TILING_DATA_WITH_STRUCT(DynamicMxQuantTilingData, tilingData, tiling);
        DynamicMxQuant::DynamicMxQuantNotTailAxisFP8<half, fp8_e5m2_t, true> op;
        op.Init(x, y, mxScale, userWS, &tilingData);
        op.Process();
        DynamicMxQuantPost postOp;
        postOp.Init(mxScale, userWS, &tilingData);
        postOp.Process();
    } else if (TILING_KEY_IS(TILING_KEY_BF16_FP8E5M2_QUANT_TAIL_AXIS_ODD_SCALE)) {
        GET_TILING_DATA_WITH_STRUCT(DynamicMxQuantTilingData, tilingData, tiling);
        DynamicMxQuant::DynamicMxQuantNotTailAxisFP8<bfloat16_t, fp8_e5m2_t, true> op;
        op.Init(x, y, mxScale, userWS, &tilingData);
        op.Process();
        DynamicMxQuantPost postOp;
        postOp.Init(mxScale, userWS, &tilingData);
        postOp.Process();
    } else if (TILING_KEY_IS(TILING_KEY_FP16_FP8E5M2_QUANT_OTHER_AXIS_ODD_SCALE)) {
        GET_TILING_DATA_WITH_STRUCT(DynamicMxQuantTilingData, tilingData, tiling);
        DynamicMxQuant::DynamicMxQuantNotTailAxisFP8<half, fp8_e5m2_t, false> op;
        op.Init(x, y, mxScale, userWS, &tilingData);
        op.Process();
        DynamicMxQuantPost postOp;
        postOp.Init(mxScale, userWS, &tilingData);
        postOp.Process();
    } else if (TILING_KEY_IS(TILING_KEY_BF16_FP8E5M2_QUANT_OTHER_AXIS_ODD_SCALE)) {
        GET_TILING_DATA_WITH_STRUCT(DynamicMxQuantTilingData, tilingData, tiling);
        DynamicMxQuant::DynamicMxQuantNotTailAxisFP8<bfloat16_t, fp8_e5m2_t, false> op;
        op.Init(x, y, mxScale, userWS, &tilingData);
        op.Process();
        DynamicMxQuantPost postOp;
        postOp.Init(mxScale, userWS, &tilingData);
        postOp.Process();
    } else if (TILING_KEY_IS(TILING_KEY_FP16_FP8E4M3FN_QUANT_SMALL_TAIL_AXIS_ODD_SCALE)) {
        GET_TILING_DATA_WITH_STRUCT(DynamicMxQuantTilingData, tilingData, tiling);
        DynamicMxQuant::DynamicMxQuantNotTailAxisOptimizeFP8<half, fp8_e4m3fn_t, true> op;
        op.Init(x, y, mxScale, userWS, &tilingData);
        op.Process();
        DynamicMxQuantPost postOp;
        postOp.Init(mxScale, userWS, &tilingData);
        postOp.Process();
    } else if (TILING_KEY_IS(TILING_KEY_BF16_FP8E4M3FN_QUANT_SMALL_TAIL_AXIS_ODD_SCALE)) {
        GET_TILING_DATA_WITH_STRUCT(DynamicMxQuantTilingData, tilingData, tiling);
        DynamicMxQuant::DynamicMxQuantNotTailAxisOptimizeFP8<bfloat16_t, fp8_e4m3fn_t, true> op;
        op.Init(x, y, mxScale, userWS, &tilingData);
        op.Process();
        DynamicMxQuantPost postOp;
        postOp.Init(mxScale, userWS, &tilingData);
        postOp.Process();
    } else if (TILING_KEY_IS(TILING_KEY_FP16_FP8E5M2_QUANT_SMALL_TAIL_AXIS_ODD_SCALE)) {
        GET_TILING_DATA_WITH_STRUCT(DynamicMxQuantTilingData, tilingData, tiling);
        DynamicMxQuant::DynamicMxQuantNotTailAxisOptimizeFP8<half, fp8_e5m2_t, false> op;
        op.Init(x, y, mxScale, userWS, &tilingData);
        op.Process();
        DynamicMxQuantPost postOp;
        postOp.Init(mxScale, userWS, &tilingData);
        postOp.Process();
    } else if (TILING_KEY_IS(TILING_KEY_BF16_FP8E5M2_QUANT_SMALL_TAIL_AXIS_ODD_SCALE)) {
        GET_TILING_DATA_WITH_STRUCT(DynamicMxQuantTilingData, tilingData, tiling);
        DynamicMxQuant::DynamicMxQuantNotTailAxisOptimizeFP8<bfloat16_t, fp8_e5m2_t, false> op;
        op.Init(x, y, mxScale, userWS, &tilingData);
        op.Process();
        DynamicMxQuantPost postOp;
        postOp.Init(mxScale, userWS, &tilingData);
        postOp.Process();
    } else if (TILING_KEY_IS(TILING_KEY_FP16_FP8E4M3FN_QUANT_TAIL_AXIS_PERF_ODD_SCALE)) {
        GET_TILING_DATA_WITH_STRUCT(DynamicMxQuantTilingData, tilingData, tiling);
        DynamicMxQuant::DynamicMxQuantTailAxisFP8<half, fp8_e4m3fn_t> op;
        op.Init(x, y, mxScale, userWS, &tilingData);
        op.Process();
        DynamicMxQuantPost postOp;
        postOp.Init(mxScale, userWS, &tilingData);
        postOp.Process();
    } else if (TILING_KEY_IS(TILING_KEY_BF16_FP8E4M3FN_QUANT_TAIL_AXIS_PERF_ODD_SCALE)) {
        GET_TILING_DATA_WITH_STRUCT(DynamicMxQuantTilingData, tilingData, tiling);
        DynamicMxQuant::DynamicMxQuantTailAxisFP8<bfloat16_t, fp8_e4m3fn_t> op;
        op.Init(x, y, mxScale, userWS, &tilingData);
        op.Process();
        DynamicMxQuantPost postOp;
        postOp.Init(mxScale, userWS, &tilingData);
        postOp.Process();
    } else if (TILING_KEY_IS(TILING_KEY_FP16_FP8E5M2_QUANT_TAIL_AXIS_PERF_ODD_SCALE)) {
        GET_TILING_DATA_WITH_STRUCT(DynamicMxQuantTilingData, tilingData, tiling);
        DynamicMxQuant::DynamicMxQuantTailAxisFP8<half, fp8_e5m2_t> op;
        op.Init(x, y, mxScale, userWS, &tilingData);
        op.Process();
        DynamicMxQuantPost postOp;
        postOp.Init(mxScale, userWS, &tilingData);
        postOp.Process();
    } else if (TILING_KEY_IS(TILING_KEY_BF16_FP8E5M2_QUANT_TAIL_AXIS_PERF_ODD_SCALE)) {
        GET_TILING_DATA_WITH_STRUCT(DynamicMxQuantTilingData, tilingData, tiling);
        DynamicMxQuant::DynamicMxQuantTailAxisFP8<bfloat16_t, fp8_e5m2_t> op;
        op.Init(x, y, mxScale, userWS, &tilingData);
        op.Process();
        DynamicMxQuantPost postOp;
        postOp.Init(mxScale, userWS, &tilingData);
        postOp.Process();
    } else if (TILING_KEY_IS(TILING_KEY_OPT_FOR_NOT_LAST_QUANT_AXIS_SMALL)) {
        GET_TILING_DATA_WITH_STRUCT(DynamicMxQuant4OptimizeTilingData, tilingData, tiling);
        TPipe pipe;
        DynamicMxQuantNotTailAxisOptimizeHighPerf op;
        op.Init(&pipe, x, y, mxScale, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_OPT_FOR_NOT_LAST_QUANT_AXIS_LARGE)) {
        GET_TILING_DATA_WITH_STRUCT(DynamicMxQuant4OptimizeTilingData, tilingData, tiling);
        TPipe pipe;
        DynamicMxQuant::DynamicMxQuantHP2000<DTYPE_X, DTYPE_Y> op(&tilingData, &pipe);
        op.Init(x, y, mxScale);
        op.Process();
    }

#if (__NPU_ARCH__ == 3101)
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(oriOverflowMode);
#endif
}
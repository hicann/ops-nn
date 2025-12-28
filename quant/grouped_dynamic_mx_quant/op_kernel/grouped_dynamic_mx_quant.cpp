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
 * \file grouped_dynamic_mx_quant.cpp
 * \brief
 */

#include "arch35/grouped_dynamic_mx_quant_not_tail_axis_fp8.h"

#define TILING_KEY_FP16_FP8E4M3FN_QUANT_OTHER_AXIS 11
#define TILING_KEY_BF16_FP8E4M3FN_QUANT_OTHER_AXIS 21
#define TILING_KEY_FP16_FP8E5M2_QUANT_OTHER_AXIS 12
#define TILING_KEY_BF16_FP8E5M2_QUANT_OTHER_AXIS 22

#define FLOAT_OVERFLOW_MODE_CTRL 60

// 十位数为1、2，分别表示输入类型是float16、bfloat16;
// 个位数为1、2，分别表示输出类型是float8_e4m3fn、float8_e5m2

using namespace GroupedDynamicMxQuant;

extern "C" __global__ __aicore__ void grouped_dynamic_mx_quant(GM_ADDR x, GM_ADDR groupIndex, GM_ADDR y, GM_ADDR mxScale,
    GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    #if (__NPU_ARCH__ == 3101)
    int64_t oriOverflowMode = AscendC::GetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL,FLOAT_OVERFLOW_MODE_CTRL>();
    #endif

    if (TILING_KEY_IS(TILING_KEY_FP16_FP8E4M3FN_QUANT_OTHER_AXIS)) {
        GroupedDynamicMxQuant::GroupedDynamicMxQuantBaseFP8<half, fp8_e4m3fn_t> op;
        op.Init(x, groupIndex, y, mxScale, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_BF16_FP8E4M3FN_QUANT_OTHER_AXIS)) {
        GroupedDynamicMxQuant::GroupedDynamicMxQuantBaseFP8<bfloat16_t, fp8_e4m3fn_t> op;
        op.Init(x, groupIndex, y, mxScale, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_FP16_FP8E5M2_QUANT_OTHER_AXIS)) {
        GroupedDynamicMxQuant::GroupedDynamicMxQuantBaseFP8<half, fp8_e5m2_t> op;
        op.Init(x, groupIndex, y, mxScale, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_BF16_FP8E5M2_QUANT_OTHER_AXIS)) {
        GroupedDynamicMxQuant::GroupedDynamicMxQuantBaseFP8<bfloat16_t, fp8_e5m2_t> op;
        op.Init(x, groupIndex, y, mxScale, tilingData);
        op.Process();
    }

    #if (__NPU_ARCH__ == 3101)
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL,FLOAT_OVERFLOW_MODE_CTRL>(oriOverflowMode);
    #endif
}
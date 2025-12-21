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
 * \file dynamic_block_quant_apt.cpp
 * \brief
 */
#include "arch35/dynamic_block_quant_b8_kernel.h"
#include "arch35/dynamic_block_quant_bf16_b8_kernel.h"
#include "arch35/dynamic_block_quant_single_row_kernel.h"

// 千分位表示 RoundMode
// 百位数为1、2，分别表示输入类型是float16、bfloat16;
// 十位数为0、1、2、3，分别表示输出类型是float8_e5m2、float8_e4m3fn、hifloat8
// 个位数为0、1，分别基础模板和特化模板

using namespace DynamicBlockQuant;

extern "C" __global__ __aicore__ void dynamic_block_quant(
    GM_ADDR x, GM_ADDR y, GM_ADDR scale, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(1100)) {
        DynamicBlockQuant::DynamicBlockQuantB8<half, fp8_e5m2_t, 1> op(&pipe);
        op.Init(x, y, scale, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(1200)) {
        DynamicBlockQuant::DynamicBlockQuantBF16B8<bfloat16_t, fp8_e5m2_t, 1> op(&pipe);
        op.Init(x, y, scale, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(1110)) {
        DynamicBlockQuant::DynamicBlockQuantB8<half, fp8_e4m3fn_t, 1> op(&pipe);
        op.Init(x, y, scale, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(1210)) {
        DynamicBlockQuant::DynamicBlockQuantBF16B8<bfloat16_t, fp8_e4m3fn_t, 1> op(&pipe);
        op.Init(x, y, scale, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(4120)) {
        DynamicBlockQuant::DynamicBlockQuantB8<half, hifloat8_t, 4> op(&pipe);
        op.Init(x, y, scale, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(4220)) {
        DynamicBlockQuant::DynamicBlockQuantBF16B8<bfloat16_t, hifloat8_t, 4> op(&pipe);
        op.Init(x, y, scale, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(7120)) {
        DynamicBlockQuant::DynamicBlockQuantB8<half, hifloat8_t, 7> op(&pipe);
        op.Init(x, y, scale, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(7220)) {
        DynamicBlockQuant::DynamicBlockQuantBF16B8<bfloat16_t, hifloat8_t, 7> op(&pipe);
        op.Init(x, y, scale, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(1101)) {
        DynamicBlockQuant::DynamicBlockQuantSingleRow<half, fp8_e5m2_t, 1> op(&pipe);
        op.Init(x, y, scale, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(1201)) {
        DynamicBlockQuant::DynamicBlockQuantSingleRow<bfloat16_t, fp8_e5m2_t, 1> op(&pipe);
        op.Init(x, y, scale, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(1111)) {
        DynamicBlockQuant::DynamicBlockQuantSingleRow<half, fp8_e4m3fn_t, 1> op(&pipe);
        op.Init(x, y, scale, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(1211)) {
        DynamicBlockQuant::DynamicBlockQuantSingleRow<bfloat16_t, fp8_e4m3fn_t, 1> op(&pipe);
        op.Init(x, y, scale, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(4121)) {
        DynamicBlockQuant::DynamicBlockQuantSingleRow<half, hifloat8_t, 4> op(&pipe);
        op.Init(x, y, scale, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(4221)) {
        DynamicBlockQuant::DynamicBlockQuantSingleRow<bfloat16_t, hifloat8_t, 4> op(&pipe);
        op.Init(x, y, scale, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(7121)) {
        DynamicBlockQuant::DynamicBlockQuantSingleRow<half, hifloat8_t, 7> op(&pipe);
        op.Init(x, y, scale, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(7221)) {
        DynamicBlockQuant::DynamicBlockQuantSingleRow<bfloat16_t, hifloat8_t, 7> op(&pipe);
        op.Init(x, y, scale, &tilingData);
        op.Process();
    }
}
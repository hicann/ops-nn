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
 * \file grouped_dynamic_block_quant.cpp
 * \brief
 */

#include "arch35/grouped_dynamic_block_quant_small_block.h"
#include "arch35/grouped_dynamic_block_quant_large_block.h"

#define TILING_KEY_SMALL_BLOCK_FP16_FP8E4M3FN_QUANT 1111
#define TILING_KEY_SMALL_BLOCK_BF16_FP8E4M3FN_QUANT 1211
#define TILING_KEY_SMALL_BLOCK_FP16_FP8E5M2_QUANT 1121
#define TILING_KEY_SMALL_BLOCK_BF16_FP8E5M2_QUANT 1221
#define TILING_KEY_SMALL_BLOCK_FP16_HIFP8_QUANT_ROUND 1134
#define TILING_KEY_SMALL_BLOCK_BF16_HIFP8_QUANT_ROUND 1234
#define TILING_KEY_SMALL_BLOCK_FP16_HIFP8_QUANT_HYBRID 1137
#define TILING_KEY_SMALL_BLOCK_BF16_HIFP8_QUANT_HYBRID 1237
#define TILING_KEY_LARGE_BLOCK_FP16_FP8E4M3FN_QUANT 2111
#define TILING_KEY_LARGE_BLOCK_BF16_FP8E4M3FN_QUANT 2211
#define TILING_KEY_LARGE_BLOCK_FP16_FP8E5M2_QUANT 2121
#define TILING_KEY_LARGE_BLOCK_BF16_FP8E5M2_QUANT 2221
#define TILING_KEY_LARGE_BLOCK_FP16_HIFP8_QUANT_ROUND 2134
#define TILING_KEY_LARGE_BLOCK_BF16_HIFP8_QUANT_ROUND 2234
#define TILING_KEY_LARGE_BLOCK_FP16_HIFP8_QUANT_HYBRID 2137
#define TILING_KEY_LARGE_BLOCK_BF16_HIFP8_QUANT_HYBRID 2237

#define FLOAT_OVERFLOW_MODE_CTRL 60

// 千位数为1、2，分别表示Block放得下UB、Block放不下UB
// 百位数为1、2，分别表示输入类型是float16、bfloat16;
// 十位数为1、2、3，分别表示输出类型是float8_e4m3、float8_e5m2、hifloat8
// 个位数为1、4、7，分别表示RoundMode是rint、round、hybrid

using namespace GroupedDynamicBlockQuant;

extern "C" __global__ __aicore__ void grouped_dynamic_block_quant(
    GM_ADDR x, GM_ADDR groupList, GM_ADDR yOut, GM_ADDR scaleOut, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA(tilingData, tiling);

#if (__NPU_ARCH__ == 3101)
    int64_t oriOverflowMode = AscendC::GetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>();
#endif

    if (TILING_KEY_IS(TILING_KEY_SMALL_BLOCK_FP16_FP8E4M3FN_QUANT)) {
        GroupedDynamicBlockQuant::GroupedDynamicBlockQuantSmallBlock<half, fp8_e4m3fn_t, 1> op;
        op.Init(x, groupList, yOut, scaleOut, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_SMALL_BLOCK_BF16_FP8E4M3FN_QUANT)) {
        GroupedDynamicBlockQuant::GroupedDynamicBlockQuantSmallBlock<bfloat16_t, fp8_e4m3fn_t, 1> op;
        op.Init(x, groupList, yOut, scaleOut, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_SMALL_BLOCK_FP16_FP8E5M2_QUANT)) {
        GroupedDynamicBlockQuant::GroupedDynamicBlockQuantSmallBlock<half, fp8_e5m2_t, 1> op;
        op.Init(x, groupList, yOut, scaleOut, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_SMALL_BLOCK_BF16_FP8E5M2_QUANT)) {
        GroupedDynamicBlockQuant::GroupedDynamicBlockQuantSmallBlock<bfloat16_t, fp8_e5m2_t, 1> op;
        op.Init(x, groupList, yOut, scaleOut, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_SMALL_BLOCK_FP16_HIFP8_QUANT_ROUND)) {
        GroupedDynamicBlockQuant::GroupedDynamicBlockQuantSmallBlock<half, hifloat8_t, 4> op;
        op.Init(x, groupList, yOut, scaleOut, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_SMALL_BLOCK_BF16_HIFP8_QUANT_ROUND)) {
        GroupedDynamicBlockQuant::GroupedDynamicBlockQuantSmallBlock<bfloat16_t, hifloat8_t, 4> op;
        op.Init(x, groupList, yOut, scaleOut, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_SMALL_BLOCK_FP16_HIFP8_QUANT_HYBRID)) {
        GroupedDynamicBlockQuant::GroupedDynamicBlockQuantSmallBlock<half, hifloat8_t, 7> op;
        op.Init(x, groupList, yOut, scaleOut, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_SMALL_BLOCK_BF16_HIFP8_QUANT_HYBRID)) {
        GroupedDynamicBlockQuant::GroupedDynamicBlockQuantSmallBlock<bfloat16_t, hifloat8_t, 7> op;
        op.Init(x, groupList, yOut, scaleOut, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_LARGE_BLOCK_FP16_FP8E4M3FN_QUANT)) {
        GroupedDynamicBlockQuant::GroupedDynamicBlockQuantLargeBlock<half, fp8_e4m3fn_t, 1> op;
        op.Init(x, groupList, yOut, scaleOut, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_LARGE_BLOCK_BF16_FP8E4M3FN_QUANT)) {
        GroupedDynamicBlockQuant::GroupedDynamicBlockQuantLargeBlock<bfloat16_t, fp8_e4m3fn_t, 1> op;
        op.Init(x, groupList, yOut, scaleOut, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_LARGE_BLOCK_FP16_FP8E5M2_QUANT)) {
        GroupedDynamicBlockQuant::GroupedDynamicBlockQuantLargeBlock<half, fp8_e5m2_t, 1> op;
        op.Init(x, groupList, yOut, scaleOut, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_LARGE_BLOCK_BF16_FP8E5M2_QUANT)) {
        GroupedDynamicBlockQuant::GroupedDynamicBlockQuantLargeBlock<bfloat16_t, fp8_e5m2_t, 1> op;
        op.Init(x, groupList, yOut, scaleOut, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_LARGE_BLOCK_FP16_HIFP8_QUANT_ROUND)) {
        GroupedDynamicBlockQuant::GroupedDynamicBlockQuantLargeBlock<half, hifloat8_t, 4> op;
        op.Init(x, groupList, yOut, scaleOut, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_LARGE_BLOCK_BF16_HIFP8_QUANT_ROUND)) {
        GroupedDynamicBlockQuant::GroupedDynamicBlockQuantLargeBlock<bfloat16_t, hifloat8_t, 4> op;
        op.Init(x, groupList, yOut, scaleOut, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_LARGE_BLOCK_FP16_HIFP8_QUANT_HYBRID)) {
        GroupedDynamicBlockQuant::GroupedDynamicBlockQuantLargeBlock<half, hifloat8_t, 7> op;
        op.Init(x, groupList, yOut, scaleOut, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_LARGE_BLOCK_BF16_HIFP8_QUANT_HYBRID)) {
        GroupedDynamicBlockQuant::GroupedDynamicBlockQuantLargeBlock<bfloat16_t, hifloat8_t, 7> op;
        op.Init(x, groupList, yOut, scaleOut, tilingData);
        op.Process();
    }

#if (__NPU_ARCH__ == 3101)
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(oriOverflowMode);
#endif
}
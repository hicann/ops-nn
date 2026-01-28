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
#include "arch35/dynamic_block_quant_large_blocksize_kernel.h"

#define FLOAT_OVERFLOW_MODE_CTRL 60

#define TILING_KEY_RINT_FP16_FP8E5M2_NORMAL 1100
#define TILING_KEY_RINT_BF16_FP8E5M2_NORMAL 1200
#define TILING_KEY_RINT_FP16_FP8E4M3_NORMAL 1110
#define TILING_KEY_RINT_BF16_FP8E4M3_NORMAL 1210
#define TILING_KEY_ROUND_FP16_FIFLOAT8_NORMAL 4120
#define TILING_KEY_ROUND_BF16_FIFLOAT8_NORMAL 4220
#define TILING_KEY_HYBRID_FP16_FIFLOAT8_NORMAL 7120
#define TILING_KEY_HYBRID_BF16_FIFLOAT8_NORMAL 7220

#define TILING_KEY_RINT_FP16_FP8E5M2_SINGLE 1101
#define TILING_KEY_RINT_BF16_FP8E5M2_SINGLE 1201
#define TILING_KEY_RINT_FP16_FP8E4M3_SINGLE 1111
#define TILING_KEY_RINT_BF16_FP8E4M3_SINGLE 1211
#define TILING_KEY_ROUND_FP16_FIFLOAT8_SINGLE 4121
#define TILING_KEY_ROUND_BF16_FIFLOAT8_SINGLE 4221
#define TILING_KEY_HYBRID_FP16_FIFLOAT8_SINGLE 7121
#define TILING_KEY_HYBRID_BF16_FIFLOAT8_SINGLE 7221

#define TILING_KEY_RINT_FP16_FP8E5M2_LARGE 1102
#define TILING_KEY_RINT_BF16_FP8E5M2_LARGE 1202
#define TILING_KEY_RINT_FP16_FP8E4M3_LARGE 1112
#define TILING_KEY_RINT_BF16_FP8E4M3_LARGE 1212
#define TILING_KEY_ROUND_FP16_FIFLOAT8_LARGE 4122
#define TILING_KEY_ROUND_BF16_FIFLOAT8_LARGE 4222
#define TILING_KEY_HYBRID_FP16_FIFLOAT8_LARGE 7122
#define TILING_KEY_HYBRID_BF16_FIFLOAT8_LARGE 7222

#define MODE_RINT 1
#define MODE_ROUND 4
#define MODE_HYBRID 7
// 千分位表示 RoundMode 1,4,7 分别是MODE_RINT、MODE_ROUND、MODE_HYBRID
// 百位数为1、2，分别表示输入类型是float16、bfloat16;
// 十位数为0、1、2、3，分别表示输出类型是float8_e5m2、float8_e4m3fn、hifloat8
// 个位数为0、1，2 分别基础模板和特化模板和大blocksize模板

using namespace DynamicBlockQuant;

__aicore__ inline void SingleUB(
    GM_ADDR x, GM_ADDR y, GM_ADDR scale, const DynamicBlockQuantTilingData& tilingData, TPipe& pipe)
{
    if (TILING_KEY_IS(TILING_KEY_RINT_FP16_FP8E5M2_NORMAL)) {
        DynamicBlockQuant::DynamicBlockQuantB8<half, fp8_e5m2_t, MODE_RINT> op(&pipe);
        op.Init(x, y, scale, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_RINT_BF16_FP8E5M2_NORMAL)) {
        DynamicBlockQuant::DynamicBlockQuantBF16B8<bfloat16_t, fp8_e5m2_t, MODE_RINT> op(&pipe);
        op.Init(x, y, scale, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_RINT_FP16_FP8E4M3_NORMAL)) {
        DynamicBlockQuant::DynamicBlockQuantB8<half, fp8_e4m3fn_t, MODE_RINT> op(&pipe);
        op.Init(x, y, scale, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_RINT_BF16_FP8E4M3_NORMAL)) {
        DynamicBlockQuant::DynamicBlockQuantBF16B8<bfloat16_t, fp8_e4m3fn_t, MODE_RINT> op(&pipe);
        op.Init(x, y, scale, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_ROUND_FP16_FIFLOAT8_NORMAL)) {
        DynamicBlockQuant::DynamicBlockQuantB8<half, hifloat8_t, MODE_ROUND> op(&pipe);
        op.Init(x, y, scale, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_ROUND_BF16_FIFLOAT8_NORMAL)) {
        DynamicBlockQuant::DynamicBlockQuantBF16B8<bfloat16_t, hifloat8_t, MODE_ROUND> op(&pipe);
        op.Init(x, y, scale, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_HYBRID_FP16_FIFLOAT8_NORMAL)) {
        DynamicBlockQuant::DynamicBlockQuantB8<half, hifloat8_t, MODE_HYBRID> op(&pipe);
        op.Init(x, y, scale, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_HYBRID_BF16_FIFLOAT8_NORMAL)) {
        DynamicBlockQuant::DynamicBlockQuantBF16B8<bfloat16_t, hifloat8_t, MODE_HYBRID> op(&pipe);
        op.Init(x, y, scale, tilingData);
        op.Process();
    }
}

__aicore__ inline void SingleRow(
    GM_ADDR x, GM_ADDR y, GM_ADDR scale, const DynamicBlockQuantTilingData& tilingData, TPipe& pipe)
{
    if (TILING_KEY_IS(TILING_KEY_RINT_FP16_FP8E5M2_SINGLE)) {
        DynamicBlockQuant::DynamicBlockQuantSingleRow<half, fp8_e5m2_t, MODE_RINT> op(&pipe);
        op.Init(x, y, scale, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_RINT_BF16_FP8E5M2_SINGLE)) {
        DynamicBlockQuant::DynamicBlockQuantSingleRow<bfloat16_t, fp8_e5m2_t, MODE_RINT> op(&pipe);
        op.Init(x, y, scale, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_RINT_FP16_FP8E4M3_SINGLE)) {
        DynamicBlockQuant::DynamicBlockQuantSingleRow<half, fp8_e4m3fn_t, MODE_RINT> op(&pipe);
        op.Init(x, y, scale, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_RINT_BF16_FP8E4M3_SINGLE)) {
        DynamicBlockQuant::DynamicBlockQuantSingleRow<bfloat16_t, fp8_e4m3fn_t, MODE_RINT> op(&pipe);
        op.Init(x, y, scale, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_ROUND_FP16_FIFLOAT8_SINGLE)) {
        DynamicBlockQuant::DynamicBlockQuantSingleRow<half, hifloat8_t, MODE_ROUND> op(&pipe);
        op.Init(x, y, scale, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_ROUND_BF16_FIFLOAT8_SINGLE)) {
        DynamicBlockQuant::DynamicBlockQuantSingleRow<bfloat16_t, hifloat8_t, MODE_ROUND> op(&pipe);
        op.Init(x, y, scale, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_HYBRID_FP16_FIFLOAT8_SINGLE)) {
        DynamicBlockQuant::DynamicBlockQuantSingleRow<half, hifloat8_t, MODE_HYBRID> op(&pipe);
        op.Init(x, y, scale, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_HYBRID_BF16_FIFLOAT8_SINGLE)) {
        DynamicBlockQuant::DynamicBlockQuantSingleRow<bfloat16_t, hifloat8_t, MODE_HYBRID> op(&pipe);
        op.Init(x, y, scale, &tilingData);
        op.Process();
    }
}

__aicore__ inline void LargeBlockSize(
    GM_ADDR x, GM_ADDR y, GM_ADDR scale, const DynamicBlockQuantTilingData& tilingData, TPipe& pipe)
{
    if (TILING_KEY_IS(TILING_KEY_RINT_FP16_FP8E5M2_LARGE)) {
        DynamicBlockQuant::DynamicBlockQuantLargeBlockSize<half, fp8_e5m2_t, MODE_RINT> op(&pipe);
        op.Init(x, y, scale, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_RINT_BF16_FP8E5M2_LARGE)) {
        DynamicBlockQuant::DynamicBlockQuantLargeBlockSize<bfloat16_t, fp8_e5m2_t, MODE_RINT> op(&pipe);
        op.Init(x, y, scale, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_RINT_FP16_FP8E4M3_LARGE)) {
        DynamicBlockQuant::DynamicBlockQuantLargeBlockSize<half, fp8_e4m3fn_t, MODE_RINT> op(&pipe);
        op.Init(x, y, scale, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_RINT_BF16_FP8E4M3_LARGE)) {
        DynamicBlockQuant::DynamicBlockQuantLargeBlockSize<bfloat16_t, fp8_e4m3fn_t, MODE_RINT> op(&pipe);
        op.Init(x, y, scale, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_ROUND_FP16_FIFLOAT8_LARGE)) {
        DynamicBlockQuant::DynamicBlockQuantLargeBlockSize<half, hifloat8_t, MODE_ROUND> op(&pipe);
        op.Init(x, y, scale, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_ROUND_BF16_FIFLOAT8_LARGE)) {
        DynamicBlockQuant::DynamicBlockQuantLargeBlockSize<bfloat16_t, hifloat8_t, MODE_ROUND> op(&pipe);
        op.Init(x, y, scale, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_HYBRID_FP16_FIFLOAT8_LARGE)) {
        DynamicBlockQuant::DynamicBlockQuantLargeBlockSize<half, hifloat8_t, MODE_HYBRID> op(&pipe);
        op.Init(x, y, scale, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_HYBRID_BF16_FIFLOAT8_LARGE)) {
        DynamicBlockQuant::DynamicBlockQuantLargeBlockSize<bfloat16_t, hifloat8_t, MODE_HYBRID> op(&pipe);
        op.Init(x, y, scale, tilingData);
        op.Process();
    }
}

extern "C" __global__ __aicore__ void dynamic_block_quant(
    GM_ADDR x, GM_ADDR y, GM_ADDR scale, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);

#if (__NPU_ARCH__ == 3101)
    int64_t oriOverflowMode = AscendC::GetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>();
#endif

    SingleUB(x, y, scale, tilingData, pipe);
    SingleRow(x, y, scale, tilingData, pipe);
    LargeBlockSize(x, y, scale, tilingData, pipe);

#if (__NPU_ARCH__ == 3101)
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(oriOverflowMode);
#endif
}
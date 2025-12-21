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
 * \file dequant_swiglu_quant_apt.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "arch35/dequant_swiglu_quant.h"
#include "arch35/dequant_swiglu_quant_nlast.h"

#define ACT_TRUE_QUANT_TRUE_GROUPINDEX_TRUE     111
#define ACT_TRUE_QUANT_TRUE_GROUPINDEX_FALSE    110
#define ACT_TRUE_QUANT_FALSE_GROUPINDEX_TRUE    101
#define ACT_TRUE_QUANT_FALSE_GROUPINDEX_FALSE   100
#define ACT_FALSE_QUANT_TRUE_GROUPINDEX_TRUE    11
#define ACT_FALSE_QUANT_TRUE_GROUPINDEX_FALSE   10
#define ACT_FALSE_QUANT_FALSE_GROUPINDEX_TRUE   1
#define ACT_FALSE_QUANT_FALSE_GROUPINDEX_FALSE  0
#define ACTDIM_FALSE_X_FALSE_BIAS_TRUE_ACT_TRUE_QUANT_TRUE_GROUPINDEX_TRUE  1111 // bias = int32
#define ACTDIM_FALSE_X_FALSE_BIAS_FP32_ACT_TRUE_QUANT_TRUE_GROUPINDEX_TRUE  2111 // bias = float32
#define ACTDIM_FALSE_X_FALSE_BIAS_FP16_ACT_TRUE_QUANT_TRUE_GROUPINDEX_TRUE  3111 // bias = float16
#define ACTDIM_FALSE_X_FALSE_BIAS_BF16_ACT_TRUE_QUANT_TRUE_GROUPINDEX_TRUE  4111 // bias = bfloat16
#define ACTDIM_FALSE_X_FALSE_BIAS_TRUE_ACT_TRUE_QUANT_TRUE_GROUPINDEX_FALSE  1110 // bias = int32
#define ACTDIM_FALSE_X_FALSE_BIAS_FP32_ACT_TRUE_QUANT_TRUE_GROUPINDEX_FALSE  2110 // bias = float32
#define ACTDIM_FALSE_X_FALSE_BIAS_FP16_ACT_TRUE_QUANT_TRUE_GROUPINDEX_FALSE  3110 // bias = float16
#define ACTDIM_FALSE_X_FALSE_BIAS_BF16_ACT_TRUE_QUANT_TRUE_GROUPINDEX_FALSE  4110 // bias = bfloat16
#define ACTDIM_FALSE_X_FALSE_BIAS_TRUE_ACT_TRUE_QUANT_FALSE_GROUPINDEX_TRUE  1101 // bias = int32
#define ACTDIM_FALSE_X_FALSE_BIAS_FP32_ACT_TRUE_QUANT_FALSE_GROUPINDEX_TRUE  2101 // bias = float32
#define ACTDIM_FALSE_X_FALSE_BIAS_FP16_ACT_TRUE_QUANT_FALSE_GROUPINDEX_TRUE  3101 // bias = float16
#define ACTDIM_FALSE_X_FALSE_BIAS_BF16_ACT_TRUE_QUANT_FALSE_GROUPINDEX_TRUE  4101 // bias = bfloat16
#define ACTDIM_FALSE_X_FALSE_BIAS_TRUE_ACT_TRUE_QUANT_FALSE_GROUPINDEX_FALSE  1100 // bias = int32
#define ACTDIM_FALSE_X_FALSE_BIAS_FP32_ACT_TRUE_QUANT_FALSE_GROUPINDEX_FALSE  2100 // bias = float32
#define ACTDIM_FALSE_X_FALSE_BIAS_FP16_ACT_TRUE_QUANT_FALSE_GROUPINDEX_FALSE  3100 // bias = float16
#define ACTDIM_FALSE_X_FALSE_BIAS_BF16_ACT_TRUE_QUANT_FALSE_GROUPINDEX_FALSE  4100 // bias = bfloat16
#define ACTDIM_FALSE_X_FALSE_BIAS_TRUE_ACT_FALSE_QUANT_TRUE_GROUPINDEX_TRUE  1011 // bias = int32
#define ACTDIM_FALSE_X_FALSE_BIAS_FP32_ACT_FALSE_QUANT_TRUE_GROUPINDEX_TRUE  2011 // bias = float32
#define ACTDIM_FALSE_X_FALSE_BIAS_FP16_ACT_FALSE_QUANT_TRUE_GROUPINDEX_TRUE  3011 // bias = float16
#define ACTDIM_FALSE_X_FALSE_BIAS_BF16_ACT_FALSE_QUANT_TRUE_GROUPINDEX_TRUE  4011 // bias = bfloat16
#define ACTDIM_FALSE_X_FALSE_BIAS_TRUE_ACT_FALSE_QUANT_TRUE_GROUPINDEX_FALSE  1010 // bias = int32
#define ACTDIM_FALSE_X_FALSE_BIAS_FP32_ACT_FALSE_QUANT_TRUE_GROUPINDEX_FALSE  2010 // bias = float32
#define ACTDIM_FALSE_X_FALSE_BIAS_FP16_ACT_FALSE_QUANT_TRUE_GROUPINDEX_FALSE  3010 // bias = float16
#define ACTDIM_FALSE_X_FALSE_BIAS_BF16_ACT_FALSE_QUANT_TRUE_GROUPINDEX_FALSE  4010 // bias = bfloat16
#define ACTDIM_FALSE_X_FALSE_BIAS_TRUE_ACT_FALSE_QUANT_FALSE_GROUPINDEX_TRUE  1001 // bias = int32
#define ACTDIM_FALSE_X_FALSE_BIAS_FP32_ACT_FALSE_QUANT_FALSE_GROUPINDEX_TRUE  2001 // bias = float32
#define ACTDIM_FALSE_X_FALSE_BIAS_FP16_ACT_FALSE_QUANT_FALSE_GROUPINDEX_TRUE  3001 // bias = float16
#define ACTDIM_FALSE_X_FALSE_BIAS_BF16_ACT_FALSE_QUANT_FALSE_GROUPINDEX_TRUE  4001 // bias = bfloat16
#define ACTDIM_FALSE_X_FALSE_BIAS_TRUE_ACT_FALSE_QUANT_FALSE_GROUPINDEX_FALSE  1000 // bias = int32
#define ACTDIM_FALSE_X_FALSE_BIAS_FP32_ACT_FALSE_QUANT_FALSE_GROUPINDEX_FALSE  2000 // bias = float32
#define ACTDIM_FALSE_X_FALSE_BIAS_FP16_ACT_FALSE_QUANT_FALSE_GROUPINDEX_FALSE  3000 // bias = float16
#define ACTDIM_FALSE_X_FALSE_BIAS_BF16_ACT_FALSE_QUANT_FALSE_GROUPINDEX_FALSE  4000 // bias = bfloat16
#define ACTDIM_TRUE_X_FALSE_BIAS_FALSE_ACT_TRUE_QUANT_TRUE_GROUPINDEX_FALSE  100110
#define ACTDIM_TRUE_X_FALSE_BIAS_FALSE_ACT_TRUE_QUANT_FALSE_GROUPINDEX_FALSE  100100
#define ACTDIM_TRUE_X_FALSE_BIAS_FALSE_ACT_FALSE_QUANT_TRUE_GROUPINDEX_FALSE  100010
#define ACTDIM_TRUE_X_FALSE_BIAS_FALSE_ACT_FALSE_QUANT_FALSE_GROUPINDEX_FALSE  100000
#define ACTDIM_TRUE_X_FALSE_BIAS_TRUE_ACT_TRUE_QUANT_TRUE_GROUPINDEX_FALSE  101110 // bias = int32
#define ACTDIM_TRUE_X_FALSE_BIAS_FP32_ACT_TRUE_QUANT_TRUE_GROUPINDEX_FALSE  102110 // bias = float32
#define ACTDIM_TRUE_X_FALSE_BIAS_FP16_ACT_TRUE_QUANT_TRUE_GROUPINDEX_FALSE  103110 // bias = float16
#define ACTDIM_TRUE_X_FALSE_BIAS_BF16_ACT_TRUE_QUANT_TRUE_GROUPINDEX_FALSE  104110 // bias = bfloat16
#define ACTDIM_TRUE_X_FALSE_BIAS_TRUE_ACT_TRUE_QUANT_FALSE_GROUPINDEX_FALSE  101100 // bias = int32
#define ACTDIM_TRUE_X_FALSE_BIAS_FP32_ACT_TRUE_QUANT_FALSE_GROUPINDEX_FALSE  102100 // bias = float32
#define ACTDIM_TRUE_X_FALSE_BIAS_FP16_ACT_TRUE_QUANT_FALSE_GROUPINDEX_FALSE  103100 // bias = float16
#define ACTDIM_TRUE_X_FALSE_BIAS_BF16_ACT_TRUE_QUANT_FALSE_GROUPINDEX_FALSE  104100 // bias = bfloat16
#define ACTDIM_TRUE_X_FALSE_BIAS_TRUE_ACT_FALSE_QUANT_TRUE_GROUPINDEX_FALSE  101010 // bias = int32
#define ACTDIM_TRUE_X_FALSE_BIAS_FP32_ACT_FALSE_QUANT_TRUE_GROUPINDEX_FALSE  102010 // bias = float32
#define ACTDIM_TRUE_X_FALSE_BIAS_FP16_ACT_FALSE_QUANT_TRUE_GROUPINDEX_FALSE  103010 // bias = float16
#define ACTDIM_TRUE_X_FALSE_BIAS_BF16_ACT_FALSE_QUANT_TRUE_GROUPINDEX_FALSE  104010 // bias = bfloat16
#define ACTDIM_TRUE_X_FALSE_BIAS_TRUE_ACT_FALSE_QUANT_FALSE_GROUPINDEX_FALSE  101000 // bias = int32
#define ACTDIM_TRUE_X_FALSE_BIAS_FP32_ACT_FALSE_QUANT_FALSE_GROUPINDEX_FALSE  102000 // bias = float32
#define ACTDIM_TRUE_X_FALSE_BIAS_FP16_ACT_FALSE_QUANT_FALSE_GROUPINDEX_FALSE  103000 // bias = float16
#define ACTDIM_TRUE_X_FALSE_BIAS_BF16_ACT_FALSE_QUANT_FALSE_GROUPINDEX_FALSE  104000 // bias = bfloat16

using namespace AscendC;

extern "C" __global__ __aicore__ void dequant_swiglu_quant(GM_ADDR x, GM_ADDR weightScale, GM_ADDR activationScale,
                                                           GM_ADDR bias, GM_ADDR quantScale, GM_ADDR quantOffset,
                                                           GM_ADDR groupIndex, GM_ADDR y, GM_ADDR scale,
                                                           GM_ADDR workspace, GM_ADDR tiling) {
  if (g_coreType != AIV) {
    return;
  }

  TPipe pipe;
  if (TILING_KEY_IS(ACT_TRUE_QUANT_TRUE_GROUPINDEX_TRUE)) {
    // activate_dim=-1, x=int32/fp16/bf6, bias无值, activate_scale有值, quant_scale有值, group_index有值, 000111
    GET_TILING_DATA_WITH_STRUCT(DequantSwigluQuantV35BaseTilingData, tilingDataIn, tiling);
    const DequantSwigluQuantV35BaseTilingData* __restrict tilingData = &tilingDataIn;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, float, int64_t, bool, DTYPE_X, DTYPE_Y> op(&pipe);
    op.Init(x, weightScale, activationScale, nullptr, quantScale, nullptr, groupIndex, y, scale, tilingData);
    op.Process();
  } else if (TILING_KEY_IS(ACT_TRUE_QUANT_TRUE_GROUPINDEX_FALSE)) {
    // activate_dim=-1, x=int32/fp16/bf6, bias无值, activate_scale有值, quant_scale有值, group_index无值, 000110
    GET_TILING_DATA_WITH_STRUCT(DequantSwigluQuantV35BaseTilingData, tilingDataIn, tiling);
    const DequantSwigluQuantV35BaseTilingData* __restrict tilingData = &tilingDataIn;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, float, bool, bool, DTYPE_X, DTYPE_Y> op(&pipe);
    op.Init(x, weightScale, activationScale, nullptr, quantScale, nullptr, groupIndex, y, scale, tilingData);
    op.Process();
  } else if (TILING_KEY_IS(ACT_TRUE_QUANT_FALSE_GROUPINDEX_TRUE)) {
    // activate_dim=-1, x=int32/fp16/bf6, bias无值, activate_scale有值, quant_scale无值, group_index有值, 000101
    GET_TILING_DATA_WITH_STRUCT(DequantSwigluQuantV35BaseTilingData, tilingDataIn, tiling);
    const DequantSwigluQuantV35BaseTilingData* __restrict tilingData = &tilingDataIn;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, bool, int64_t, bool, DTYPE_X, DTYPE_Y> op(&pipe);
    op.Init(x, weightScale, activationScale, nullptr, quantScale, nullptr, groupIndex, y, scale, tilingData);
    op.Process();
  } else if (TILING_KEY_IS(ACT_TRUE_QUANT_FALSE_GROUPINDEX_FALSE)) {
    // activate_dim=-1, x=int32/fp16/bf6, bias无值, activate_scale有值, quant_scale无值, group_index无值, 000100
    GET_TILING_DATA_WITH_STRUCT(DequantSwigluQuantV35BaseTilingData, tilingDataIn, tiling);
    const DequantSwigluQuantV35BaseTilingData* __restrict tilingData = &tilingDataIn;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, bool, bool, bool, DTYPE_X, DTYPE_Y> op(&pipe);
    op.Init(x, weightScale, activationScale, nullptr, quantScale, nullptr, groupIndex, y, scale, tilingData);
    op.Process();
  } else if (TILING_KEY_IS(ACT_FALSE_QUANT_TRUE_GROUPINDEX_TRUE)) {
    // activate_dim=-1, x=int32/fp16/bf6, bias无值, activate_scale无值, quant_scale有值, group_index有值, 000011
    GET_TILING_DATA_WITH_STRUCT(DequantSwigluQuantV35BaseTilingData, tilingDataIn, tiling);
    const DequantSwigluQuantV35BaseTilingData* __restrict tilingData = &tilingDataIn;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, float, int64_t, bool, DTYPE_X, DTYPE_Y> op(&pipe);
    op.Init(x, weightScale, activationScale, nullptr, quantScale, nullptr, groupIndex, y, scale, tilingData);
    op.Process();
  } else if (TILING_KEY_IS(ACT_FALSE_QUANT_TRUE_GROUPINDEX_FALSE)) {
    // activate_dim=-1, x=int32/fp16/bf6, bias无值, activate_scale无值, quant_scale有值, group_index无值, 000010
    GET_TILING_DATA_WITH_STRUCT(DequantSwigluQuantV35BaseTilingData, tilingDataIn, tiling);
    const DequantSwigluQuantV35BaseTilingData* __restrict tilingData = &tilingDataIn;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, float, bool, bool, DTYPE_X, DTYPE_Y> op(&pipe);
    op.Init(x, weightScale, activationScale, nullptr, quantScale, nullptr, groupIndex, y, scale, tilingData);
    op.Process();
  } else if (TILING_KEY_IS(ACT_FALSE_QUANT_FALSE_GROUPINDEX_TRUE)) {
    // activate_dim=-1, x=int32/fp16/bf6, bias无值, activate_scale无值, quant_scale无值, group_index有值, 000001
    GET_TILING_DATA_WITH_STRUCT(DequantSwigluQuantV35BaseTilingData, tilingDataIn, tiling);
    const DequantSwigluQuantV35BaseTilingData* __restrict tilingData = &tilingDataIn;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, bool, int64_t, bool, DTYPE_X, DTYPE_Y> op(&pipe);
    op.Init(x, weightScale, activationScale, nullptr, quantScale, nullptr, groupIndex, y, scale, tilingData);
    op.Process();
  } else if (TILING_KEY_IS(ACT_FALSE_QUANT_FALSE_GROUPINDEX_FALSE)) {
    // activate_dim=-1, x=int32/fp16/bf6, bias无值, activate_scale无值, quant_scale无值, group_index无值, 000000
    GET_TILING_DATA_WITH_STRUCT(DequantSwigluQuantV35BaseTilingData, tilingDataIn, tiling);
    const DequantSwigluQuantV35BaseTilingData* __restrict tilingData = &tilingDataIn;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, bool, bool, bool, DTYPE_X, DTYPE_Y> op(&pipe);
    op.Init(x, weightScale, activationScale, nullptr, quantScale, nullptr, groupIndex, y, scale, tilingData);
    op.Process();
  } else if (TILING_KEY_IS(ACTDIM_FALSE_X_FALSE_BIAS_TRUE_ACT_TRUE_QUANT_TRUE_GROUPINDEX_TRUE)) {
    // activate_dim=-1, x=int32/fp16/bf6, bias=int32, activate_scale有值, quant_scale有值, group_index有值, 001111
    GET_TILING_DATA_WITH_STRUCT(DequantSwigluQuantV35BaseTilingData, tilingDataIn, tiling);
    const DequantSwigluQuantV35BaseTilingData* __restrict tilingData = &tilingDataIn;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, float, int64_t, int32_t, DTYPE_X, DTYPE_Y> op(&pipe);
    op.Init(x, weightScale, activationScale, bias, quantScale, nullptr, groupIndex, y, scale, tilingData);
    op.Process();
  } else if (TILING_KEY_IS(ACTDIM_FALSE_X_FALSE_BIAS_FP32_ACT_TRUE_QUANT_TRUE_GROUPINDEX_TRUE)) {
    // activate_dim=-1, x=int32/fp16/bf6, bias=float32, activate_scale有值, quant_scale有值, group_index有值, 002111
    GET_TILING_DATA_WITH_STRUCT(DequantSwigluQuantV35BaseTilingData, tilingDataIn, tiling);
    const DequantSwigluQuantV35BaseTilingData* __restrict tilingData = &tilingDataIn;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, float, int64_t, float, DTYPE_X, DTYPE_Y> op(&pipe);
    op.Init(x, weightScale, activationScale, bias, quantScale, nullptr, groupIndex, y, scale, tilingData);
    op.Process();
  } else if (TILING_KEY_IS(ACTDIM_FALSE_X_FALSE_BIAS_FP16_ACT_TRUE_QUANT_TRUE_GROUPINDEX_TRUE)) {
    // activate_dim=-1, x=int32/fp16/bf6, bias=float16, activate_scale有值, quant_scale有值, group_index有值, 003111
    GET_TILING_DATA_WITH_STRUCT(DequantSwigluQuantV35BaseTilingData, tilingDataIn, tiling);
    const DequantSwigluQuantV35BaseTilingData* __restrict tilingData = &tilingDataIn;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, float, int64_t, half, DTYPE_X, DTYPE_Y> op(&pipe);
    op.Init(x, weightScale, activationScale, bias, quantScale, nullptr, groupIndex, y, scale, tilingData);
    op.Process();
  } else if (TILING_KEY_IS(ACTDIM_FALSE_X_FALSE_BIAS_BF16_ACT_TRUE_QUANT_TRUE_GROUPINDEX_TRUE)) {
    // activate_dim=-1, x=int32/fp16/bf6, bias=bfloat16, activate_scale有值, quant_scale有值, group_index有值, 004111
    GET_TILING_DATA_WITH_STRUCT(DequantSwigluQuantV35BaseTilingData, tilingDataIn, tiling);
    const DequantSwigluQuantV35BaseTilingData* __restrict tilingData = &tilingDataIn;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, float, int64_t, bfloat16_t, DTYPE_X, DTYPE_Y> op(&pipe);
    op.Init(x, weightScale, activationScale, bias, quantScale, nullptr, groupIndex, y, scale, tilingData);
    op.Process();
  } else if (TILING_KEY_IS(ACTDIM_FALSE_X_FALSE_BIAS_TRUE_ACT_TRUE_QUANT_TRUE_GROUPINDEX_FALSE)) {
    // activate_dim=-1, x=int32/fp16/bf6, bias=int32, activate_scale有值, quant_scale有值, group_index无值, 001110
    GET_TILING_DATA_WITH_STRUCT(DequantSwigluQuantV35BaseTilingData, tilingDataIn, tiling);
    const DequantSwigluQuantV35BaseTilingData* __restrict tilingData = &tilingDataIn;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, float, bool, int32_t, DTYPE_X, DTYPE_Y> op(&pipe);
    op.Init(x, weightScale, activationScale, bias, quantScale, nullptr, groupIndex, y, scale, tilingData);
    op.Process();
  } else if (TILING_KEY_IS(ACTDIM_FALSE_X_FALSE_BIAS_FP32_ACT_TRUE_QUANT_TRUE_GROUPINDEX_FALSE)) {
    // activate_dim=-1, x=int32/fp16/bf6, bias=float32, activate_scale有值, quant_scale有值, group_index无值, 002110
    GET_TILING_DATA_WITH_STRUCT(DequantSwigluQuantV35BaseTilingData, tilingDataIn, tiling);
    const DequantSwigluQuantV35BaseTilingData* __restrict tilingData = &tilingDataIn;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, float, bool, float, DTYPE_X, DTYPE_Y> op(&pipe);
    op.Init(x, weightScale, activationScale, bias, quantScale, nullptr, groupIndex, y, scale, tilingData);
    op.Process();
  } else if (TILING_KEY_IS(ACTDIM_FALSE_X_FALSE_BIAS_FP16_ACT_TRUE_QUANT_TRUE_GROUPINDEX_FALSE)) {
    // activate_dim=-1, x=int32/fp16/bf6, bias=float16, activate_scale有值, quant_scale有值, group_index无值, 003110
    GET_TILING_DATA_WITH_STRUCT(DequantSwigluQuantV35BaseTilingData, tilingDataIn, tiling);
    const DequantSwigluQuantV35BaseTilingData* __restrict tilingData = &tilingDataIn;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, float, bool, half, DTYPE_X, DTYPE_Y> op(&pipe);
    op.Init(x, weightScale, activationScale, bias, quantScale, nullptr, groupIndex, y, scale, tilingData);
    op.Process();
  } else if (TILING_KEY_IS(ACTDIM_FALSE_X_FALSE_BIAS_BF16_ACT_TRUE_QUANT_TRUE_GROUPINDEX_FALSE)) {
    // activate_dim=-1, x=int32/fp16/bf6, bias=bfloat16, activate_scale有值, quant_scale有值, group_index无值, 004110
    GET_TILING_DATA_WITH_STRUCT(DequantSwigluQuantV35BaseTilingData, tilingDataIn, tiling);
    const DequantSwigluQuantV35BaseTilingData* __restrict tilingData = &tilingDataIn;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, float, bool, bfloat16_t, DTYPE_X, DTYPE_Y> op(&pipe);
    op.Init(x, weightScale, activationScale, bias, quantScale, nullptr, groupIndex, y, scale, tilingData);
    op.Process();
  } else if (TILING_KEY_IS(ACTDIM_FALSE_X_FALSE_BIAS_TRUE_ACT_TRUE_QUANT_FALSE_GROUPINDEX_TRUE)) {
    // activate_dim=-1, x=int32/fp16/bf6, bias=int32, activate_scale有值, quant_scale无值, group_index有值, 001101
    GET_TILING_DATA_WITH_STRUCT(DequantSwigluQuantV35BaseTilingData, tilingDataIn, tiling);
    const DequantSwigluQuantV35BaseTilingData* __restrict tilingData = &tilingDataIn;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, bool, int64_t, int32_t, DTYPE_X, DTYPE_Y> op(&pipe);
    op.Init(x, weightScale, activationScale, bias, quantScale, nullptr, groupIndex, y, scale, tilingData);
    op.Process();
  } else if (TILING_KEY_IS(ACTDIM_FALSE_X_FALSE_BIAS_FP32_ACT_TRUE_QUANT_FALSE_GROUPINDEX_TRUE)) {
    // activate_dim=-1, x=int32/fp16/bf6, bias=float32, activate_scale有值, quant_scale无值, group_index有值, 002101
    GET_TILING_DATA_WITH_STRUCT(DequantSwigluQuantV35BaseTilingData, tilingDataIn, tiling);
    const DequantSwigluQuantV35BaseTilingData* __restrict tilingData = &tilingDataIn;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, bool, int64_t, float, DTYPE_X, DTYPE_Y> op(&pipe);
    op.Init(x, weightScale, activationScale, bias, quantScale, nullptr, groupIndex, y, scale, tilingData);
    op.Process();
  } else if (TILING_KEY_IS(ACTDIM_FALSE_X_FALSE_BIAS_FP16_ACT_TRUE_QUANT_FALSE_GROUPINDEX_TRUE)) {
    // activate_dim=-1, x=int32/fp16/bf6, bias=float16, activate_scale有值, quant_scale无值, group_index有值, 003101
    GET_TILING_DATA_WITH_STRUCT(DequantSwigluQuantV35BaseTilingData, tilingDataIn, tiling);
    const DequantSwigluQuantV35BaseTilingData* __restrict tilingData = &tilingDataIn;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, bool, int64_t, half, DTYPE_X, DTYPE_Y> op(&pipe);
    op.Init(x, weightScale, activationScale, bias, quantScale, nullptr, groupIndex, y, scale, tilingData);
    op.Process();
  } else if (TILING_KEY_IS(ACTDIM_FALSE_X_FALSE_BIAS_BF16_ACT_TRUE_QUANT_FALSE_GROUPINDEX_TRUE)) {
    // activate_dim=-1, x=int32/fp16/bf6, bias=bfloat16, activate_scale有值, quant_scale无值, group_index有值, 004101
    GET_TILING_DATA_WITH_STRUCT(DequantSwigluQuantV35BaseTilingData, tilingDataIn, tiling);
    const DequantSwigluQuantV35BaseTilingData* __restrict tilingData = &tilingDataIn;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, bool, int64_t, bfloat16_t, DTYPE_X, DTYPE_Y> op(&pipe);
    op.Init(x, weightScale, activationScale, bias, quantScale, nullptr, groupIndex, y, scale, tilingData);
    op.Process();
  } else if (TILING_KEY_IS(ACTDIM_FALSE_X_FALSE_BIAS_TRUE_ACT_TRUE_QUANT_FALSE_GROUPINDEX_FALSE)) {
    // activate_dim=-1, x=int32/fp16/bf6, bias=int32, activate_scale有值, quant_scale无值, group_index无值, 001100
    GET_TILING_DATA_WITH_STRUCT(DequantSwigluQuantV35BaseTilingData, tilingDataIn, tiling);
    const DequantSwigluQuantV35BaseTilingData* __restrict tilingData = &tilingDataIn;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, bool, bool, int32_t, DTYPE_X, DTYPE_Y> op(&pipe);
    op.Init(x, weightScale, activationScale, bias, quantScale, nullptr, groupIndex, y, scale, tilingData);
    op.Process();
  } else if (TILING_KEY_IS(ACTDIM_FALSE_X_FALSE_BIAS_FP32_ACT_TRUE_QUANT_FALSE_GROUPINDEX_FALSE)) {
    // activate_dim=-1, x=int32/fp16/bf6, bias=float32, activate_scale有值, quant_scale无值, group_index无值, 002100
    GET_TILING_DATA_WITH_STRUCT(DequantSwigluQuantV35BaseTilingData, tilingDataIn, tiling);
    const DequantSwigluQuantV35BaseTilingData* __restrict tilingData = &tilingDataIn;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, bool, bool, float, DTYPE_X, DTYPE_Y> op(&pipe);
    op.Init(x, weightScale, activationScale, bias, quantScale, nullptr, groupIndex, y, scale, tilingData);
    op.Process();
  } else if (TILING_KEY_IS(ACTDIM_FALSE_X_FALSE_BIAS_FP16_ACT_TRUE_QUANT_FALSE_GROUPINDEX_FALSE)) {
    // activate_dim=-1, x=int32/fp16/bf6, bias=float16, activate_scale有值, quant_scale无值, group_index无值, 003100
    GET_TILING_DATA_WITH_STRUCT(DequantSwigluQuantV35BaseTilingData, tilingDataIn, tiling);
    const DequantSwigluQuantV35BaseTilingData* __restrict tilingData = &tilingDataIn;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, bool, bool, half, DTYPE_X, DTYPE_Y> op(&pipe);
    op.Init(x, weightScale, activationScale, bias, quantScale, nullptr, groupIndex, y, scale, tilingData);
    op.Process();
  } else if (TILING_KEY_IS(ACTDIM_FALSE_X_FALSE_BIAS_BF16_ACT_TRUE_QUANT_FALSE_GROUPINDEX_FALSE)) {
    // activate_dim=-1, x=int32/fp16/bf6, bias=bfloat16, activate_scale有值, quant_scale无值, group_index无值, 004100
    GET_TILING_DATA_WITH_STRUCT(DequantSwigluQuantV35BaseTilingData, tilingDataIn, tiling);
    const DequantSwigluQuantV35BaseTilingData* __restrict tilingData = &tilingDataIn;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<float, bool, bool, bfloat16_t, DTYPE_X, DTYPE_Y> op(&pipe);
    op.Init(x, weightScale, activationScale, bias, quantScale, nullptr, groupIndex, y, scale, tilingData);
    op.Process();
  } else if (TILING_KEY_IS(ACTDIM_FALSE_X_FALSE_BIAS_TRUE_ACT_FALSE_QUANT_TRUE_GROUPINDEX_TRUE)) {
    // activate_dim=-1, x=int32/fp16/bf6, bias=int32, activate_scale无值, quant_scale有值, group_index有值, 001011
    GET_TILING_DATA_WITH_STRUCT(DequantSwigluQuantV35BaseTilingData, tilingDataIn, tiling);
    const DequantSwigluQuantV35BaseTilingData* __restrict tilingData = &tilingDataIn;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, float, int64_t, int32_t, DTYPE_X, DTYPE_Y> op(&pipe);
    op.Init(x, weightScale, activationScale, bias, quantScale, nullptr, groupIndex, y, scale, tilingData);
    op.Process();
  } else if (TILING_KEY_IS(ACTDIM_FALSE_X_FALSE_BIAS_FP32_ACT_FALSE_QUANT_TRUE_GROUPINDEX_TRUE)) {
    // activate_dim=-1, x=int32/fp16/bf6, bias=float32, activate_scale无值, quant_scale有值, group_index有值, 002011
    GET_TILING_DATA_WITH_STRUCT(DequantSwigluQuantV35BaseTilingData, tilingDataIn, tiling);
    const DequantSwigluQuantV35BaseTilingData* __restrict tilingData = &tilingDataIn;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, float, int64_t, float, DTYPE_X, DTYPE_Y> op(&pipe);
    op.Init(x, weightScale, activationScale, bias, quantScale, nullptr, groupIndex, y, scale, tilingData);
    op.Process();
  } else if (TILING_KEY_IS(ACTDIM_FALSE_X_FALSE_BIAS_FP16_ACT_FALSE_QUANT_TRUE_GROUPINDEX_TRUE)) {
    // activate_dim=-1, x=int32/fp16/bf6, bias=float16, activate_scale无值, quant_scale有值, group_index有值, 003011
    GET_TILING_DATA_WITH_STRUCT(DequantSwigluQuantV35BaseTilingData, tilingDataIn, tiling);
    const DequantSwigluQuantV35BaseTilingData* __restrict tilingData = &tilingDataIn;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, float, int64_t, half, DTYPE_X, DTYPE_Y> op(&pipe);
    op.Init(x, weightScale, activationScale, bias, quantScale, nullptr, groupIndex, y, scale, tilingData);
    op.Process();
  } else if (TILING_KEY_IS(ACTDIM_FALSE_X_FALSE_BIAS_BF16_ACT_FALSE_QUANT_TRUE_GROUPINDEX_TRUE)) {
    // activate_dim=-1, x=int32/fp16/bf6, bias=bfloat16, activate_scale无值, quant_scale有值, group_index有值, 004011
    GET_TILING_DATA_WITH_STRUCT(DequantSwigluQuantV35BaseTilingData, tilingDataIn, tiling);
    const DequantSwigluQuantV35BaseTilingData* __restrict tilingData = &tilingDataIn;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, float, int64_t, bfloat16_t, DTYPE_X, DTYPE_Y> op(&pipe);
    op.Init(x, weightScale, activationScale, bias, quantScale, nullptr, groupIndex, y, scale, tilingData);
    op.Process();
  } else if (TILING_KEY_IS(ACTDIM_FALSE_X_FALSE_BIAS_TRUE_ACT_FALSE_QUANT_TRUE_GROUPINDEX_FALSE)) {
    // activate_dim=-1, x=int32/fp16/bf6, bias=int32, activate_scale无值, quant_scale有值, group_index无值, 001010
    GET_TILING_DATA_WITH_STRUCT(DequantSwigluQuantV35BaseTilingData, tilingDataIn, tiling);
    const DequantSwigluQuantV35BaseTilingData* __restrict tilingData = &tilingDataIn;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, float, bool, int32_t, DTYPE_X, DTYPE_Y> op(&pipe);
    op.Init(x, weightScale, activationScale, bias, quantScale, nullptr, groupIndex, y, scale, tilingData);
    op.Process();
  } else if (TILING_KEY_IS(ACTDIM_FALSE_X_FALSE_BIAS_FP32_ACT_FALSE_QUANT_TRUE_GROUPINDEX_FALSE)) {
    // activate_dim=-1, x=int32/fp16/bf6, bias=float32, activate_scale无值, quant_scale有值, group_index无值, 002010
    GET_TILING_DATA_WITH_STRUCT(DequantSwigluQuantV35BaseTilingData, tilingDataIn, tiling);
    const DequantSwigluQuantV35BaseTilingData* __restrict tilingData = &tilingDataIn;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, float, bool, float, DTYPE_X, DTYPE_Y> op(&pipe);
    op.Init(x, weightScale, activationScale, bias, quantScale, nullptr, groupIndex, y, scale, tilingData);
    op.Process();
  } else if (TILING_KEY_IS(ACTDIM_FALSE_X_FALSE_BIAS_FP16_ACT_FALSE_QUANT_TRUE_GROUPINDEX_FALSE)) {
    // activate_dim=-1, x=int32/fp16/bf6, bias=float16, activate_scale无值, quant_scale有值, group_index无值, 003010
    GET_TILING_DATA_WITH_STRUCT(DequantSwigluQuantV35BaseTilingData, tilingDataIn, tiling);
    const DequantSwigluQuantV35BaseTilingData* __restrict tilingData = &tilingDataIn;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, float, bool, half, DTYPE_X, DTYPE_Y> op(&pipe);
    op.Init(x, weightScale, activationScale, bias, quantScale, nullptr, groupIndex, y, scale, tilingData);
    op.Process();
  } else if (TILING_KEY_IS(ACTDIM_FALSE_X_FALSE_BIAS_BF16_ACT_FALSE_QUANT_TRUE_GROUPINDEX_FALSE)) {
    // activate_dim=-1, x=int32/fp16/bf6, bias=bfloat16, activate_scale无值, quant_scale有值, group_index无值, 004010
    GET_TILING_DATA_WITH_STRUCT(DequantSwigluQuantV35BaseTilingData, tilingDataIn, tiling);
    const DequantSwigluQuantV35BaseTilingData* __restrict tilingData = &tilingDataIn;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, float, bool, bfloat16_t, DTYPE_X, DTYPE_Y> op(&pipe);
    op.Init(x, weightScale, activationScale, bias, quantScale, nullptr, groupIndex, y, scale, tilingData);
    op.Process();
  } else if (TILING_KEY_IS(ACTDIM_FALSE_X_FALSE_BIAS_TRUE_ACT_FALSE_QUANT_FALSE_GROUPINDEX_TRUE)) {
    // activate_dim=-1, x=int32/fp16/bf6, bias=int32, activate_scale无值, quant_scale无值, group_index有值, 001001
    GET_TILING_DATA_WITH_STRUCT(DequantSwigluQuantV35BaseTilingData, tilingDataIn, tiling);
    const DequantSwigluQuantV35BaseTilingData* __restrict tilingData = &tilingDataIn;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, bool, int64_t, int32_t, DTYPE_X, DTYPE_Y> op(&pipe);
    op.Init(x, weightScale, activationScale, bias, quantScale, nullptr, groupIndex, y, scale, tilingData);
    op.Process();
  } else if (TILING_KEY_IS(ACTDIM_FALSE_X_FALSE_BIAS_FP32_ACT_FALSE_QUANT_FALSE_GROUPINDEX_TRUE)) {
    // activate_dim=-1, x=int32/fp16/bf6, bias=float32, activate_scale无值, quant_scale无值, group_index有值, 002001
    GET_TILING_DATA_WITH_STRUCT(DequantSwigluQuantV35BaseTilingData, tilingDataIn, tiling);
    const DequantSwigluQuantV35BaseTilingData* __restrict tilingData = &tilingDataIn;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, bool, int64_t, float, DTYPE_X, DTYPE_Y> op(&pipe);
    op.Init(x, weightScale, activationScale, bias, quantScale, nullptr, groupIndex, y, scale, tilingData);
    op.Process();
  } else if (TILING_KEY_IS(ACTDIM_FALSE_X_FALSE_BIAS_FP16_ACT_FALSE_QUANT_FALSE_GROUPINDEX_TRUE)) {
    // activate_dim=-1, x=int32/fp16/bf6, bias=float16, activate_scale无值, quant_scale无值, group_index有值, 003001
    GET_TILING_DATA_WITH_STRUCT(DequantSwigluQuantV35BaseTilingData, tilingDataIn, tiling);
    const DequantSwigluQuantV35BaseTilingData* __restrict tilingData = &tilingDataIn;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, bool, int64_t, half, DTYPE_X, DTYPE_Y> op(&pipe);
    op.Init(x, weightScale, activationScale, bias, quantScale, nullptr, groupIndex, y, scale, tilingData);
    op.Process();
  } else if (TILING_KEY_IS(ACTDIM_FALSE_X_FALSE_BIAS_BF16_ACT_FALSE_QUANT_FALSE_GROUPINDEX_TRUE)) {
    // activate_dim=-1, x=int32/fp16/bf6, bias=bfloat16, activate_scale无值, quant_scale无值, group_index有值, 004001
    GET_TILING_DATA_WITH_STRUCT(DequantSwigluQuantV35BaseTilingData, tilingDataIn, tiling);
    const DequantSwigluQuantV35BaseTilingData* __restrict tilingData = &tilingDataIn;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, bool, int64_t, bfloat16_t, DTYPE_X, DTYPE_Y> op(&pipe);
    op.Init(x, weightScale, activationScale, bias, quantScale, nullptr, groupIndex, y, scale, tilingData);
    op.Process();
  } else if (TILING_KEY_IS(ACTDIM_FALSE_X_FALSE_BIAS_TRUE_ACT_FALSE_QUANT_FALSE_GROUPINDEX_FALSE)) {
    // activate_dim=-1, x=int32/fp16/bf6, bias=int32, activate_scale无值, quant_scale无值, group_index无值, 001000
    GET_TILING_DATA_WITH_STRUCT(DequantSwigluQuantV35BaseTilingData, tilingDataIn, tiling);
    const DequantSwigluQuantV35BaseTilingData* __restrict tilingData = &tilingDataIn;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, bool, bool, int32_t, DTYPE_X, DTYPE_Y> op(&pipe);
    op.Init(x, weightScale, activationScale, bias, quantScale, nullptr, groupIndex, y, scale, tilingData);
    op.Process();
  } else if (TILING_KEY_IS(ACTDIM_FALSE_X_FALSE_BIAS_FP32_ACT_FALSE_QUANT_FALSE_GROUPINDEX_FALSE)) {
    // activate_dim=-1, x=int32/fp16/bf6, bias=float32, activate_scale无值, quant_scale无值, group_index无值, 002000
    GET_TILING_DATA_WITH_STRUCT(DequantSwigluQuantV35BaseTilingData, tilingDataIn, tiling);
    const DequantSwigluQuantV35BaseTilingData* __restrict tilingData = &tilingDataIn;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, bool, bool, float, DTYPE_X, DTYPE_Y> op(&pipe);
    op.Init(x, weightScale, activationScale, bias, quantScale, nullptr, groupIndex, y, scale, tilingData);
    op.Process();
  } else if (TILING_KEY_IS(ACTDIM_FALSE_X_FALSE_BIAS_FP16_ACT_FALSE_QUANT_FALSE_GROUPINDEX_FALSE)) {
    // activate_dim=-1, x=int32/fp16/bf6, bias=float16, activate_scale无值, quant_scale无值, group_index无值, 003000
    GET_TILING_DATA_WITH_STRUCT(DequantSwigluQuantV35BaseTilingData, tilingDataIn, tiling);
    const DequantSwigluQuantV35BaseTilingData* __restrict tilingData = &tilingDataIn;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, bool, bool, half, DTYPE_X, DTYPE_Y> op(&pipe);
    op.Init(x, weightScale, activationScale, bias, quantScale, nullptr, groupIndex, y, scale, tilingData);
    op.Process();
  } else if (TILING_KEY_IS(ACTDIM_FALSE_X_FALSE_BIAS_BF16_ACT_FALSE_QUANT_FALSE_GROUPINDEX_FALSE)) {
    // activate_dim=-1, x=int32/fp16/bf6, bias=bfloat16, activate_scale无值, quant_scale无值, group_index无值, 004000
    GET_TILING_DATA_WITH_STRUCT(DequantSwigluQuantV35BaseTilingData, tilingDataIn, tiling);
    const DequantSwigluQuantV35BaseTilingData* __restrict tilingData = &tilingDataIn;
    DequantSwigluQuantV35Ops::DequantSwigluQuantBase<bool, bool, bool, bfloat16_t, DTYPE_X, DTYPE_Y> op(&pipe);
    op.Init(x, weightScale, activationScale, bias, quantScale, nullptr, groupIndex, y, scale, tilingData);
    op.Process();
  } else if (TILING_KEY_IS(ACTDIM_TRUE_X_FALSE_BIAS_TRUE_ACT_TRUE_QUANT_TRUE_GROUPINDEX_FALSE)) {
    // activate_dim!=-1, x=int32/fp16/bf6, bias=int32, activate_scale有值, quant_scale有值, group_index无值, 101110
    GET_TILING_DATA_WITH_STRUCT(DequantSwigluQuantV35NlastTilingData, tilingDataIn, tiling);
    const DequantSwigluQuantV35NlastTilingData* __restrict tilingData = &tilingDataIn;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<float, float, bool, int32_t, DTYPE_X, DTYPE_Y> op(&pipe);
    op.Init(x, weightScale, activationScale, bias, quantScale, nullptr, groupIndex, y, scale, tilingData);
    op.Process();
  } else if (TILING_KEY_IS(ACTDIM_TRUE_X_FALSE_BIAS_FP32_ACT_TRUE_QUANT_TRUE_GROUPINDEX_FALSE)) {
    // activate_dim!=-1, x=int32/fp16/bf6, bias=float32, activate_scale有值, quant_scale有值, group_index无值, 102110
    GET_TILING_DATA_WITH_STRUCT(DequantSwigluQuantV35NlastTilingData, tilingDataIn, tiling);
    const DequantSwigluQuantV35NlastTilingData* __restrict tilingData = &tilingDataIn;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<float, float, bool, float, DTYPE_X, DTYPE_Y> op(&pipe);
    op.Init(x, weightScale, activationScale, bias, quantScale, nullptr, groupIndex, y, scale, tilingData);
    op.Process();
  } else if (TILING_KEY_IS(ACTDIM_TRUE_X_FALSE_BIAS_FP16_ACT_TRUE_QUANT_TRUE_GROUPINDEX_FALSE)) {
    // activate_dim!=-1, x=int32/fp16/bf6, bias=float16, activate_scale有值, quant_scale有值, group_index无值, 103110
    GET_TILING_DATA_WITH_STRUCT(DequantSwigluQuantV35NlastTilingData, tilingDataIn, tiling);
    const DequantSwigluQuantV35NlastTilingData* __restrict tilingData = &tilingDataIn;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<float, float, bool, half, DTYPE_X, DTYPE_Y> op(&pipe);
    op.Init(x, weightScale, activationScale, bias, quantScale, nullptr, groupIndex, y, scale, tilingData);
    op.Process();
  } else if (TILING_KEY_IS(ACTDIM_TRUE_X_FALSE_BIAS_BF16_ACT_TRUE_QUANT_TRUE_GROUPINDEX_FALSE)) {
    // activate_dim!=-1, x=int32/fp16/bf6, bias=bfloat16, activate_scale有值, quant_scale有值, group_index无值, 104110
    GET_TILING_DATA_WITH_STRUCT(DequantSwigluQuantV35NlastTilingData, tilingDataIn, tiling);
    const DequantSwigluQuantV35NlastTilingData* __restrict tilingData = &tilingDataIn;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<float, float, bool, bfloat16_t, DTYPE_X, DTYPE_Y> op(&pipe);
    op.Init(x, weightScale, activationScale, bias, quantScale, nullptr, groupIndex, y, scale, tilingData);
    op.Process();
  } else if (TILING_KEY_IS(ACTDIM_TRUE_X_FALSE_BIAS_TRUE_ACT_TRUE_QUANT_FALSE_GROUPINDEX_FALSE)) {
    // activate_dim!=-1, x=int32/fp16/bf6, bias=int32, activate_scale有值, quant_scale无值, group_index无值, 101100
    GET_TILING_DATA_WITH_STRUCT(DequantSwigluQuantV35NlastTilingData, tilingDataIn, tiling);
    const DequantSwigluQuantV35NlastTilingData* __restrict tilingData = &tilingDataIn;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<float, bool, bool, int32_t, DTYPE_X, DTYPE_Y> op(&pipe);
    op.Init(x, weightScale, activationScale, bias, quantScale, nullptr, groupIndex, y, scale, tilingData);
    op.Process();
  } else if (TILING_KEY_IS(ACTDIM_TRUE_X_FALSE_BIAS_FP32_ACT_TRUE_QUANT_FALSE_GROUPINDEX_FALSE)) {
    // activate_dim!=-1, x=int32/fp16/bf6, bias=float32, activate_scale有值, quant_scale无值, group_index无值, 102100
    GET_TILING_DATA_WITH_STRUCT(DequantSwigluQuantV35NlastTilingData, tilingDataIn, tiling);
    const DequantSwigluQuantV35NlastTilingData* __restrict tilingData = &tilingDataIn;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<float, bool, bool, float, DTYPE_X, DTYPE_Y> op(&pipe);
    op.Init(x, weightScale, activationScale, bias, quantScale, nullptr, groupIndex, y, scale, tilingData);
    op.Process();
  } else if (TILING_KEY_IS(ACTDIM_TRUE_X_FALSE_BIAS_FP16_ACT_TRUE_QUANT_FALSE_GROUPINDEX_FALSE)) {
    // activate_dim!=-1, x=int32/fp16/bf6, bias=flaot16, activate_scale有值, quant_scale无值, group_index无值, 103100
    GET_TILING_DATA_WITH_STRUCT(DequantSwigluQuantV35NlastTilingData, tilingDataIn, tiling);
    const DequantSwigluQuantV35NlastTilingData* __restrict tilingData = &tilingDataIn;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<float, bool, bool, half, DTYPE_X, DTYPE_Y> op(&pipe);
    op.Init(x, weightScale, activationScale, bias, quantScale, nullptr, groupIndex, y, scale, tilingData);
    op.Process();
  } else if (TILING_KEY_IS(ACTDIM_TRUE_X_FALSE_BIAS_BF16_ACT_TRUE_QUANT_FALSE_GROUPINDEX_FALSE)) {
    // activate_dim!=-1, x=int32/fp16/bf6, bias=bfloat16, activate_scale有值, quant_scale无值, group_index无值, 104100
    GET_TILING_DATA_WITH_STRUCT(DequantSwigluQuantV35NlastTilingData, tilingDataIn, tiling);
    const DequantSwigluQuantV35NlastTilingData* __restrict tilingData = &tilingDataIn;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<float, bool, bool, bfloat16_t, DTYPE_X, DTYPE_Y> op(&pipe);
    op.Init(x, weightScale, activationScale, bias, quantScale, nullptr, groupIndex, y, scale, tilingData);
    op.Process();
  } else if (TILING_KEY_IS(ACTDIM_TRUE_X_FALSE_BIAS_TRUE_ACT_FALSE_QUANT_TRUE_GROUPINDEX_FALSE)) {
    // activate_dim!=-1, x=int32/fp16/bf6, bias=int32, activate_scale无值, quant_scale有值, group_index无值, 101010
    GET_TILING_DATA_WITH_STRUCT(DequantSwigluQuantV35NlastTilingData, tilingDataIn, tiling);
    const DequantSwigluQuantV35NlastTilingData* __restrict tilingData = &tilingDataIn;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<bool, float, bool, int32_t, DTYPE_X, DTYPE_Y> op(&pipe);
    op.Init(x, weightScale, activationScale, bias, quantScale, nullptr, groupIndex, y, scale, tilingData);
    op.Process();
  } else if (TILING_KEY_IS(ACTDIM_TRUE_X_FALSE_BIAS_FP32_ACT_FALSE_QUANT_TRUE_GROUPINDEX_FALSE)) {
    // activate_dim!=-1, x=int32/fp16/bf6, bias=float32, activate_scale无值, quant_scale有值, group_index无值, 102010
    GET_TILING_DATA_WITH_STRUCT(DequantSwigluQuantV35NlastTilingData, tilingDataIn, tiling);
    const DequantSwigluQuantV35NlastTilingData* __restrict tilingData = &tilingDataIn;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<bool, float, bool, float, DTYPE_X, DTYPE_Y> op(&pipe);
    op.Init(x, weightScale, activationScale, bias, quantScale, nullptr, groupIndex, y, scale, tilingData);
    op.Process();
  } else if (TILING_KEY_IS(ACTDIM_TRUE_X_FALSE_BIAS_FP16_ACT_FALSE_QUANT_TRUE_GROUPINDEX_FALSE)) {
    // activate_dim!=-1, x=int32/fp16/bf6, bias=float16, activate_scale无值, quant_scale有值, group_index无值, 103010
    GET_TILING_DATA_WITH_STRUCT(DequantSwigluQuantV35NlastTilingData, tilingDataIn, tiling);
    const DequantSwigluQuantV35NlastTilingData* __restrict tilingData = &tilingDataIn;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<bool, float, bool, half, DTYPE_X, DTYPE_Y> op(&pipe);
    op.Init(x, weightScale, activationScale, bias, quantScale, nullptr, groupIndex, y, scale, tilingData);
    op.Process();
  } else if (TILING_KEY_IS(ACTDIM_TRUE_X_FALSE_BIAS_BF16_ACT_FALSE_QUANT_TRUE_GROUPINDEX_FALSE)) {
    // activate_dim!=-1, x=int32/fp16/bf6, bias=bfloat16, activate_scale无值, quant_scale有值, group_index无值, 104010
    GET_TILING_DATA_WITH_STRUCT(DequantSwigluQuantV35NlastTilingData, tilingDataIn, tiling);
    const DequantSwigluQuantV35NlastTilingData* __restrict tilingData = &tilingDataIn;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<bool, float, bool, bfloat16_t, DTYPE_X, DTYPE_Y> op(&pipe);
    op.Init(x, weightScale, activationScale, bias, quantScale, nullptr, groupIndex, y, scale, tilingData);
    op.Process();
  } else if (TILING_KEY_IS(ACTDIM_TRUE_X_FALSE_BIAS_TRUE_ACT_FALSE_QUANT_FALSE_GROUPINDEX_FALSE)) {
    // activate_dim!=-1, x=int32/fp16/bf6, bias=int32, activate_scale无值, quant_scale无值, group_index无值, 101000
    GET_TILING_DATA_WITH_STRUCT(DequantSwigluQuantV35NlastTilingData, tilingDataIn, tiling);
    const DequantSwigluQuantV35NlastTilingData* __restrict tilingData = &tilingDataIn;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<bool, bool, bool, int32_t, DTYPE_X, DTYPE_Y> op(&pipe);
    op.Init(x, weightScale, activationScale, bias, quantScale, nullptr, groupIndex, y, scale, tilingData);
    op.Process();
  } else if (TILING_KEY_IS(ACTDIM_TRUE_X_FALSE_BIAS_FP32_ACT_FALSE_QUANT_FALSE_GROUPINDEX_FALSE)) {
    // activate_dim!=-1, x=int32/fp16/bf6, bias=flaot32, activate_scale无值, quant_scale无值, group_index无值, 102000
    GET_TILING_DATA_WITH_STRUCT(DequantSwigluQuantV35NlastTilingData, tilingDataIn, tiling);
    const DequantSwigluQuantV35NlastTilingData* __restrict tilingData = &tilingDataIn;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<bool, bool, bool, float, DTYPE_X, DTYPE_Y> op(&pipe);
    op.Init(x, weightScale, activationScale, bias, quantScale, nullptr, groupIndex, y, scale, tilingData);
    op.Process();
  } else if (TILING_KEY_IS(ACTDIM_TRUE_X_FALSE_BIAS_FP16_ACT_FALSE_QUANT_FALSE_GROUPINDEX_FALSE)) {
    // activate_dim!=-1, x=int32/fp16/bf6, bias=float16, activate_scale无值, quant_scale无值, group_index无值, 103000
    GET_TILING_DATA_WITH_STRUCT(DequantSwigluQuantV35NlastTilingData, tilingDataIn, tiling);
    const DequantSwigluQuantV35NlastTilingData* __restrict tilingData = &tilingDataIn;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<bool, bool, bool, half, DTYPE_X, DTYPE_Y> op(&pipe);
    op.Init(x, weightScale, activationScale, bias, quantScale, nullptr, groupIndex, y, scale, tilingData);
    op.Process();
  } else if (TILING_KEY_IS(ACTDIM_TRUE_X_FALSE_BIAS_BF16_ACT_FALSE_QUANT_FALSE_GROUPINDEX_FALSE)) {
    // activate_dim!=-1, x=int32/fp16/bf6, bias=bfloat16, activate_scale无值, quant_scale无值, group_index无值, 104000
    GET_TILING_DATA_WITH_STRUCT(DequantSwigluQuantV35NlastTilingData, tilingDataIn, tiling);
    const DequantSwigluQuantV35NlastTilingData* __restrict tilingData = &tilingDataIn;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<bool, bool, bool, bfloat16_t, DTYPE_X, DTYPE_Y> op(&pipe);
    op.Init(x, weightScale, activationScale, bias, quantScale, nullptr, groupIndex, y, scale, tilingData);
    op.Process();
  } else if (TILING_KEY_IS(ACTDIM_TRUE_X_FALSE_BIAS_FALSE_ACT_FALSE_QUANT_FALSE_GROUPINDEX_FALSE)) {
    // activate_dim!=-1, x=int32/fp16/bf6, bias无值, activate_scale无值, quant_scale无值, group_index无值, 100000
    GET_TILING_DATA_WITH_STRUCT(DequantSwigluQuantV35NlastTilingData, tilingDataIn, tiling);
    const DequantSwigluQuantV35NlastTilingData* __restrict tilingData = &tilingDataIn;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<bool, bool, bool, bool, DTYPE_X, DTYPE_Y> op(&pipe);
    op.Init(x, weightScale, activationScale, nullptr, quantScale, nullptr, groupIndex, y, scale, tilingData);
    op.Process();
  } else if (TILING_KEY_IS(ACTDIM_TRUE_X_FALSE_BIAS_FALSE_ACT_FALSE_QUANT_TRUE_GROUPINDEX_FALSE)) {
    // activate_dim!=-1, x=int32/fp16/bf6, bias无值, activate_scale无值, quant_scale有值, group_index无值, 100010
    GET_TILING_DATA_WITH_STRUCT(DequantSwigluQuantV35NlastTilingData, tilingDataIn, tiling);
    const DequantSwigluQuantV35NlastTilingData* __restrict tilingData = &tilingDataIn;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<bool, float, bool, bool, DTYPE_X, DTYPE_Y> op(&pipe);
    op.Init(x, weightScale, activationScale, nullptr, quantScale, nullptr, groupIndex, y, scale, tilingData);
    op.Process();
  } else if (TILING_KEY_IS(ACTDIM_TRUE_X_FALSE_BIAS_FALSE_ACT_TRUE_QUANT_FALSE_GROUPINDEX_FALSE)) {
    // activate_dim!=-1, x=int32/fp16/bf6, bias无值, activate_scale有值, quant_scale无值, group_index无值, 100100
    GET_TILING_DATA_WITH_STRUCT(DequantSwigluQuantV35NlastTilingData, tilingDataIn, tiling);
    const DequantSwigluQuantV35NlastTilingData* __restrict tilingData = &tilingDataIn;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<float, bool, bool, bool, DTYPE_X, DTYPE_Y> op(&pipe);
    op.Init(x, weightScale, activationScale, nullptr, quantScale, nullptr, groupIndex, y, scale, tilingData);
    op.Process();
  } else if (TILING_KEY_IS(ACTDIM_TRUE_X_FALSE_BIAS_FALSE_ACT_TRUE_QUANT_TRUE_GROUPINDEX_FALSE)) {
    // activate_dim!=-1, x=int32/fp16/bf6, bias无值, activate_scale有值, quant_scale有值, group_index无值, 100110
    GET_TILING_DATA_WITH_STRUCT(DequantSwigluQuantV35NlastTilingData, tilingDataIn, tiling);
    const DequantSwigluQuantV35NlastTilingData* __restrict tilingData = &tilingDataIn;
    DequantSwigluQuantV35Ops::DequantSwigluQuantNlast<float, float, bool, bool, DTYPE_X, DTYPE_Y> op(&pipe);
    op.Init(x, weightScale, activationScale, nullptr, quantScale, nullptr, groupIndex, y, scale, tilingData);
    op.Process();
  }
}

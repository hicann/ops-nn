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
 * \file quantize_add_layer_norm.cpp
 * \brief ascend950 (arch35/regbase) kernel entry for QuantizeAddLayerNorm.
 *        Tiling key layout: 8000(prefix) + 100(welford) + bias(1 elewise / 2 brc)
 *                           + quantmode(0 mul / 10 per_channel div / 20 per_tensor scalar mul)
 */
#include "quantize_add_layer_norm_static_quant_regbase_full_load_kernel.h"
#include "quantize_add_layer_norm_static_quant_regbase_welford_kernel.h"

#define TILING_FULL_LOAD_BIAS_ELEWISE_MUL_SCALE 8001
#define TILING_FULL_LOAD_BIAS_BRC_MUL_SCALE 8002
#define TILING_WELFORD_BIAS_ELEWISE_MUL_SCALE 8101
#define TILING_WELFORD_BIAS_BRC_MUL_SCALE 8102

#define TILING_FULL_LOAD_BIAS_ELEWISE_DIV_SCALE 8011
#define TILING_FULL_LOAD_BIAS_BRC_DIV_SCALE 8012
#define TILING_WELFORD_BIAS_ELEWISE_DIV_SCALE 8111
#define TILING_WELFORD_BIAS_BRC_DIV_SCALE 8112

#define TILING_FULL_LOAD_BIAS_ELEWISE_PER_TENSOR_SCALE 8021
#define TILING_FULL_LOAD_BIAS_BRC_PER_TENSOR_SCALE 8022
#define TILING_WELFORD_BIAS_ELEWISE_PER_TENSOR_SCALE 8121
#define TILING_WELFORD_BIAS_BRC_PER_TENSOR_SCALE 8122

#define SUCCESSOR_NUMBER_OF_ONE 2

// scales is a required input, DTYPE_SCALES is always injected by the build script
#ifdef DTYPE_SCALES
#else
#define DTYPE_SCALES DTYPE_X1
#endif

// zero_points is an optional input
#ifdef DTYPE_ZERO_POINTS
#define ZERO_POINTS_STATUS OFFSET_CODE
#else
#define DTYPE_ZERO_POINTS DTYPE_X1
#define ZERO_POINTS_STATUS 0
#endif

#define OPT_STATUS (ZERO_POINTS_STATUS)

#define CREATE_STATIC_QUANT_KRENEL(tilingKeyNum, KernelClass)                                             \
    do {                                                                                                  \
        KernelClass<DTYPE_X1, DTYPE_SCALES, tilingKeyNum, OPT_STATUS, SUCCESSOR_NUMBER_OF_ONE> op(&pipe); \
        op.Init(x1, x2, gamma, beta, bias, scales, zeroPoints, y, x, usrWorkspace, tilingData);           \
        op.Process();                                                                                     \
    } while (0)

extern "C" __global__ __aicore__ void quantize_add_layer_norm(GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR beta,
                                                              GM_ADDR bias, GM_ADDR scales, GM_ADDR zeroPoints,
                                                              GM_ADDR y, GM_ADDR x, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    AscendC::TPipe pipe;
    GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);
    GET_TILING_DATA_WITH_STRUCT(QuantizeAddLayerNormRegbaseTilingData, tilingDataIn, tiling);
    const QuantizeAddLayerNormRegbaseTilingData* __restrict tilingData = &tilingDataIn;

    if (TILING_KEY_IS(0)) {
        // 0 Tiling, Do Nothing.
    } else if (TILING_KEY_IS(TILING_FULL_LOAD_BIAS_ELEWISE_MUL_SCALE)) {
        CREATE_STATIC_QUANT_KRENEL(TILING_FULL_LOAD_BIAS_ELEWISE_MUL_SCALE,
                                   QuantizeAddLayerNormRegbase::KernelQuantizeAddLayerNormStaticQuantRegbaseFullLoad);
    } else if (TILING_KEY_IS(TILING_FULL_LOAD_BIAS_BRC_MUL_SCALE)) {
        CREATE_STATIC_QUANT_KRENEL(TILING_FULL_LOAD_BIAS_BRC_MUL_SCALE,
                                   QuantizeAddLayerNormRegbase::KernelQuantizeAddLayerNormStaticQuantRegbaseFullLoad);
    } else if (TILING_KEY_IS(TILING_WELFORD_BIAS_ELEWISE_MUL_SCALE)) {
        CREATE_STATIC_QUANT_KRENEL(TILING_WELFORD_BIAS_ELEWISE_MUL_SCALE,
                                   QuantizeAddLayerNormRegbase::KernelQuantizeAddLayerNormStaticQuantRegbaseWelford);
    } else if (TILING_KEY_IS(TILING_WELFORD_BIAS_BRC_MUL_SCALE)) {
        CREATE_STATIC_QUANT_KRENEL(TILING_WELFORD_BIAS_BRC_MUL_SCALE,
                                   QuantizeAddLayerNormRegbase::KernelQuantizeAddLayerNormStaticQuantRegbaseWelford);
    }

    else if (TILING_KEY_IS(TILING_FULL_LOAD_BIAS_ELEWISE_DIV_SCALE)) {
        CREATE_STATIC_QUANT_KRENEL(TILING_FULL_LOAD_BIAS_ELEWISE_DIV_SCALE,
                                   QuantizeAddLayerNormRegbase::KernelQuantizeAddLayerNormStaticQuantRegbaseFullLoad);
    } else if (TILING_KEY_IS(TILING_FULL_LOAD_BIAS_BRC_DIV_SCALE)) {
        CREATE_STATIC_QUANT_KRENEL(TILING_FULL_LOAD_BIAS_BRC_DIV_SCALE,
                                   QuantizeAddLayerNormRegbase::KernelQuantizeAddLayerNormStaticQuantRegbaseFullLoad);
    } else if (TILING_KEY_IS(TILING_WELFORD_BIAS_ELEWISE_DIV_SCALE)) {
        CREATE_STATIC_QUANT_KRENEL(TILING_WELFORD_BIAS_ELEWISE_DIV_SCALE,
                                   QuantizeAddLayerNormRegbase::KernelQuantizeAddLayerNormStaticQuantRegbaseWelford);
    } else if (TILING_KEY_IS(TILING_WELFORD_BIAS_BRC_DIV_SCALE)) {
        CREATE_STATIC_QUANT_KRENEL(TILING_WELFORD_BIAS_BRC_DIV_SCALE,
                                   QuantizeAddLayerNormRegbase::KernelQuantizeAddLayerNormStaticQuantRegbaseWelford);
    }

    else if (TILING_KEY_IS(TILING_FULL_LOAD_BIAS_ELEWISE_PER_TENSOR_SCALE)) {
        CREATE_STATIC_QUANT_KRENEL(TILING_FULL_LOAD_BIAS_ELEWISE_PER_TENSOR_SCALE,
                                   QuantizeAddLayerNormRegbase::KernelQuantizeAddLayerNormStaticQuantRegbaseFullLoad);
    } else if (TILING_KEY_IS(TILING_FULL_LOAD_BIAS_BRC_PER_TENSOR_SCALE)) {
        CREATE_STATIC_QUANT_KRENEL(TILING_FULL_LOAD_BIAS_BRC_PER_TENSOR_SCALE,
                                   QuantizeAddLayerNormRegbase::KernelQuantizeAddLayerNormStaticQuantRegbaseFullLoad);
    } else if (TILING_KEY_IS(TILING_WELFORD_BIAS_ELEWISE_PER_TENSOR_SCALE)) {
        CREATE_STATIC_QUANT_KRENEL(TILING_WELFORD_BIAS_ELEWISE_PER_TENSOR_SCALE,
                                   QuantizeAddLayerNormRegbase::KernelQuantizeAddLayerNormStaticQuantRegbaseWelford);
    } else if (TILING_KEY_IS(TILING_WELFORD_BIAS_BRC_PER_TENSOR_SCALE)) {
        CREATE_STATIC_QUANT_KRENEL(TILING_WELFORD_BIAS_BRC_PER_TENSOR_SCALE,
                                   QuantizeAddLayerNormRegbase::KernelQuantizeAddLayerNormStaticQuantRegbaseWelford);
    }
}

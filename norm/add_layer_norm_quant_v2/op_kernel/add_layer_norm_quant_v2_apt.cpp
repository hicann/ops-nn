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
 * \file add_layer_norm_quant_v2_apt.cpp
 * \brief AddLayerNormQuantV2 arch35 (Ascend950 / regbase) entry.
 */
#include "arch35/add_layer_norm_quant_v2_static_quant_regbase_full_load_kernel.h"
#include "arch35/add_layer_norm_quant_v2_static_quant_regbase_welford_kernel.h"
#include "arch35/add_layer_norm_quant_v2_dynamic_quant_regbase_full_load_kernel.h"
#include "arch35/add_layer_norm_quant_v2_dynamic_quant_regbase_welford_kernel.h"
#include "arch35/add_layer_norm_quant_v2_empty_input_kernel.h"

#define TILING_FULL_LOAD_BIAS_NONE_MUL_SCALE 8000
#define TILING_FULL_LOAD_BIAS_ELEWISE_MUL_SCALE 8001
#define TILING_FULL_LOAD_BIAS_BRC_MUL_SCALE 8002
#define TILING_WELFORD_BIAS_NONE_MUL_SCALE 8100
#define TILING_WELFORD_BIAS_ELEWISE_MUL_SCALE 8101
#define TILING_WELFORD_BIAS_BRC_MUL_SCALE 8102
#define TILING_EMPTY_INPUT 9

#define TILING_FULL_LOAD_BIAS_NONE_DIV_SCALE 8010
#define TILING_FULL_LOAD_BIAS_ELEWISE_DIV_SCALE 8011
#define TILING_FULL_LOAD_BIAS_BRC_DIV_SCALE 8012
#define TILING_WELFORD_BIAS_NONE_DIV_SCALE 8110
#define TILING_WELFORD_BIAS_ELEWISE_DIV_SCALE 8111
#define TILING_WELFORD_BIAS_BRC_DIV_SCALE 8112

#define TILING_FULL_LOAD_BIAS_NONE_DYN_QUANT 8020
#define TILING_FULL_LOAD_BIAS_ELEWISE_DYN_QUANT 8021
#define TILING_FULL_LOAD_BIAS_BRC_DYN_QUANT 8022
#define TILING_WELFORD_BIAS_NONE_DYN_QUANT 8120
#define TILING_WELFORD_BIAS_ELEWISE_DYN_QUANT 8121
#define TILING_WELFORD_BIAS_BRC_DYN_QUANT 8122

#define SUCCESSOR_NUMBER_OF_ONE 2

#ifdef DTYPE_SCALES1
#define SCALES1_STATUS SCALE1_CODE
#else
#define DTYPE_SCALES1 DTYPE_X1
#define SCALES1_STATUS 0
#endif

#ifdef DTYPE_SCALES2
#define SCALES2_STATUS SCALE2_CODE
#else
#define DTYPE_SCALES2 DTYPE_X1
#define SCALES2_STATUS 0
#endif

#ifdef DTYPE_ZERO_POINTS1
#define ZERO_OFFSET1_STATUS OFFSET1_CODE
#else
#define DTYPE_ZERO_POINTS1 DTYPE_X1
#define ZERO_OFFSET1_STATUS 0
#endif

#ifdef DTYPE_ZERO_POINTS2
#define ZERO_OFFSET2_STATUS OFFSET2_CODE
#else
#define DTYPE_ZERO_POINTS2 DTYPE_X1
#define ZERO_OFFSET2_STATUS 0
#endif

// 置位 LAYERNORM_RES_CODE：启用 layernorm_res 输出路径
#define OPT_STATUS \
    ((SCALES1_STATUS) | (SCALES2_STATUS) | (ZERO_OFFSET1_STATUS) | (ZERO_OFFSET2_STATUS) | (LAYERNORM_RES_CODE))

#define CREATE_EMPTY_INPUT_KRENEL(tilingKeyNum, KernelClass)        \
    do {                                                            \
        KernelClass<SUCCESSOR_NUMBER_OF_ONE> op(&pipe, tilingData); \
        op.Init(outScale1, outScale2);                              \
        op.Process();                                               \
    } while (0)

#define CREATE_STATIC_QUANT_KRENEL(tilingKeyNum, KernelClass)                                                   \
    do {                                                                                                        \
        KernelClass<DTYPE_X1, DTYPE_SCALES1, tilingKeyNum, OPT_STATUS, SUCCESSOR_NUMBER_OF_ONE> op(&pipe);      \
        op.Init(x1, x2, gamma, beta, bias, scales1, scales2, zeroOffset1, zeroOffset2, y1, y2, x, layernormRes, \
                usrWorkspace, tilingData);                                                                      \
        op.Process();                                                                                           \
    } while (0)

#define CREATE_DYNAMIC_QUANT_KRENEL(tilingKeyNum, KernelClass)                                              \
    do {                                                                                                    \
        KernelClass<DTYPE_X1, DTYPE_SCALES1, tilingKeyNum, OPT_STATUS, SUCCESSOR_NUMBER_OF_ONE> op(&pipe);  \
        op.Init(x1, x2, gamma, beta, bias, scales1, scales2, y1, y2, x, outScale1, outScale2, layernormRes, \
                usrWorkspace, tilingData);                                                                  \
        op.Process();                                                                                       \
    } while (0)

extern "C" __global__ __aicore__ void add_layer_norm_quant_v2(GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR beta,
                                                              GM_ADDR bias, GM_ADDR scales1, GM_ADDR scales2,
                                                              GM_ADDR zeroOffset1, GM_ADDR zeroOffset2, GM_ADDR y1,
                                                              GM_ADDR y2, GM_ADDR x, GM_ADDR layernormRes,
                                                              GM_ADDR outScale1, GM_ADDR outScale2, GM_ADDR workspace,
                                                              GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    AscendC::TPipe pipe;
    GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);
    GET_TILING_DATA_WITH_STRUCT(AddLayerNormQuantRegbaseTilingData, tilingDataIn, tiling);
    const AddLayerNormQuantRegbaseTilingData* __restrict tilingData = &tilingDataIn;

    if (TILING_KEY_IS(0)) {
        // 0 Tiling, Do Nothing.
    } else if (TILING_KEY_IS(TILING_FULL_LOAD_BIAS_NONE_DIV_SCALE)) {
        CREATE_STATIC_QUANT_KRENEL(TILING_FULL_LOAD_BIAS_NONE_DIV_SCALE,
                                   AddLayerNormQuantRegbase::KernelAddLayerNormV2StaticQuantRegbaseFullLoad);
    } else if (TILING_KEY_IS(TILING_FULL_LOAD_BIAS_ELEWISE_DIV_SCALE)) {
        CREATE_STATIC_QUANT_KRENEL(TILING_FULL_LOAD_BIAS_ELEWISE_DIV_SCALE,
                                   AddLayerNormQuantRegbase::KernelAddLayerNormV2StaticQuantRegbaseFullLoad);
    } else if (TILING_KEY_IS(TILING_FULL_LOAD_BIAS_BRC_DIV_SCALE)) {
        CREATE_STATIC_QUANT_KRENEL(TILING_FULL_LOAD_BIAS_BRC_DIV_SCALE,
                                   AddLayerNormQuantRegbase::KernelAddLayerNormV2StaticQuantRegbaseFullLoad);
    } else if (TILING_KEY_IS(TILING_WELFORD_BIAS_NONE_DIV_SCALE)) {
        CREATE_STATIC_QUANT_KRENEL(TILING_WELFORD_BIAS_NONE_DIV_SCALE,
                                   AddLayerNormQuantRegbase::KernelAddLayerNormV2StaticQuantRegbaseWelford);
    } else if (TILING_KEY_IS(TILING_WELFORD_BIAS_ELEWISE_DIV_SCALE)) {
        CREATE_STATIC_QUANT_KRENEL(TILING_WELFORD_BIAS_ELEWISE_DIV_SCALE,
                                   AddLayerNormQuantRegbase::KernelAddLayerNormV2StaticQuantRegbaseWelford);
    } else if (TILING_KEY_IS(TILING_WELFORD_BIAS_BRC_DIV_SCALE)) {
        CREATE_STATIC_QUANT_KRENEL(TILING_WELFORD_BIAS_BRC_DIV_SCALE,
                                   AddLayerNormQuantRegbase::KernelAddLayerNormV2StaticQuantRegbaseWelford);
    }

    else if (TILING_KEY_IS(TILING_FULL_LOAD_BIAS_NONE_MUL_SCALE)) {
        CREATE_STATIC_QUANT_KRENEL(TILING_FULL_LOAD_BIAS_NONE_MUL_SCALE,
                                   AddLayerNormQuantRegbase::KernelAddLayerNormV2StaticQuantRegbaseFullLoad);
    } else if (TILING_KEY_IS(TILING_FULL_LOAD_BIAS_ELEWISE_MUL_SCALE)) {
        CREATE_STATIC_QUANT_KRENEL(TILING_FULL_LOAD_BIAS_ELEWISE_MUL_SCALE,
                                   AddLayerNormQuantRegbase::KernelAddLayerNormV2StaticQuantRegbaseFullLoad);
    } else if (TILING_KEY_IS(TILING_FULL_LOAD_BIAS_BRC_MUL_SCALE)) {
        CREATE_STATIC_QUANT_KRENEL(TILING_FULL_LOAD_BIAS_BRC_MUL_SCALE,
                                   AddLayerNormQuantRegbase::KernelAddLayerNormV2StaticQuantRegbaseFullLoad);
    } else if (TILING_KEY_IS(TILING_WELFORD_BIAS_NONE_MUL_SCALE)) {
        CREATE_STATIC_QUANT_KRENEL(TILING_WELFORD_BIAS_NONE_MUL_SCALE,
                                   AddLayerNormQuantRegbase::KernelAddLayerNormV2StaticQuantRegbaseWelford);
    } else if (TILING_KEY_IS(TILING_WELFORD_BIAS_ELEWISE_MUL_SCALE)) {
        CREATE_STATIC_QUANT_KRENEL(TILING_WELFORD_BIAS_ELEWISE_MUL_SCALE,
                                   AddLayerNormQuantRegbase::KernelAddLayerNormV2StaticQuantRegbaseWelford);
    } else if (TILING_KEY_IS(TILING_WELFORD_BIAS_BRC_MUL_SCALE)) {
        CREATE_STATIC_QUANT_KRENEL(TILING_WELFORD_BIAS_BRC_MUL_SCALE,
                                   AddLayerNormQuantRegbase::KernelAddLayerNormV2StaticQuantRegbaseWelford);
    }

    else if (TILING_KEY_IS(TILING_FULL_LOAD_BIAS_NONE_DYN_QUANT)) {
        CREATE_DYNAMIC_QUANT_KRENEL(TILING_FULL_LOAD_BIAS_NONE_DYN_QUANT,
                                    AddLayerNormQuantRegbase::KernelAddLayerNormV2DynamicQuantRegbaseFullLoad);
    } else if (TILING_KEY_IS(TILING_FULL_LOAD_BIAS_ELEWISE_DYN_QUANT)) {
        CREATE_DYNAMIC_QUANT_KRENEL(TILING_FULL_LOAD_BIAS_ELEWISE_DYN_QUANT,
                                    AddLayerNormQuantRegbase::KernelAddLayerNormV2DynamicQuantRegbaseFullLoad);
    } else if (TILING_KEY_IS(TILING_FULL_LOAD_BIAS_BRC_DYN_QUANT)) {
        CREATE_DYNAMIC_QUANT_KRENEL(TILING_FULL_LOAD_BIAS_BRC_DYN_QUANT,
                                    AddLayerNormQuantRegbase::KernelAddLayerNormV2DynamicQuantRegbaseFullLoad);
    } else if (TILING_KEY_IS(TILING_WELFORD_BIAS_NONE_DYN_QUANT)) {
        CREATE_DYNAMIC_QUANT_KRENEL(TILING_WELFORD_BIAS_NONE_DYN_QUANT,
                                    AddLayerNormQuantRegbase::KernelAddLayerNormV2DynamicQuantRegbaseWelford);
    } else if (TILING_KEY_IS(TILING_WELFORD_BIAS_ELEWISE_DYN_QUANT)) {
        CREATE_DYNAMIC_QUANT_KRENEL(TILING_WELFORD_BIAS_ELEWISE_DYN_QUANT,
                                    AddLayerNormQuantRegbase::KernelAddLayerNormV2DynamicQuantRegbaseWelford);
    } else if (TILING_KEY_IS(TILING_WELFORD_BIAS_BRC_DYN_QUANT)) {
        CREATE_DYNAMIC_QUANT_KRENEL(TILING_WELFORD_BIAS_BRC_DYN_QUANT,
                                    AddLayerNormQuantRegbase::KernelAddLayerNormV2DynamicQuantRegbaseWelford);
    } else if (TILING_KEY_IS(TILING_EMPTY_INPUT)) {
        GET_TILING_DATA_WITH_STRUCT(AddLayerNormQuantEmptyTilingData, tilingDataIn, tiling);
        const AddLayerNormQuantEmptyTilingData* __restrict tilingData = &tilingDataIn;
        CREATE_EMPTY_INPUT_KRENEL(TILING_EMPTY_INPUT, AddLayerNormQuantEmptyInput::KernelAddLayerNormQuantEmptyInput);
    }
}

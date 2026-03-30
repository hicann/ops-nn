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
 * \file add_rms_norm_dynamic_mx_quant_apt.cpp
 * \brief
 */
#include "arch35/add_rms_norm_dynamic_mx_quant_fp8_r_full_load.h"
#include "arch35/add_rms_norm_dynamic_mx_quant_fp4_r_full_load.h"
#include "arch35/add_rms_norm_dynamic_mx_quant_tiling_data.h"

using namespace AscendC;
using namespace AddRmsNormDynamicMxQuant;

#define TILING_KEY_FP8_R_FULL_LOAD 100
#define TILING_KEY_FP4_R_FULL_LOAD 101

extern "C" __global__ __aicore__ void add_rms_norm_dynamic_mx_quant(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR beta,
    GM_ADDR y, GM_ADDR x, GM_ADDR mxscale, GM_ADDR rstd,
    GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }
    
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(AddRmsNormDynamicMxQuantTilingData);
    TPipe pipe;

#if (__NPU_ARCH__ == 3510)
    int64_t oriOverflowMode = AscendC::GetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>();
#endif

    if (TILING_KEY_IS(TILING_KEY_FP8_R_FULL_LOAD)) {
        GET_TILING_DATA_WITH_STRUCT(AddRmsNormDynamicMxQuantTilingData, tilingDataIn, tiling);
        AddRmsNormDynamicMxQuantFP8RFullLoad<DTYPE_X1, DTYPE_GAMMA, DTYPE_Y> op(&pipe);
        op.Init(x1, x2, gamma, beta, y, x, mxscale, mxscale, rstd, &tilingDataIn);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_FP4_R_FULL_LOAD)) {
        GET_TILING_DATA_WITH_STRUCT(AddRmsNormDynamicMxQuantTilingData, tilingDataIn, tiling);
        AddRmsNormDynamicMxQuantFP4RFullLoad<DTYPE_X1, DTYPE_GAMMA, DTYPE_Y> op(&pipe);
        op.Init(x1, x2, gamma, beta, y, x, mxscale, mxscale, rstd, &tilingDataIn);
        op.Process();
    }

#if (__NPU_ARCH__ == 3510)
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(oriOverflowMode);
#endif
}

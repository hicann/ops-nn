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
 * \file rms_norm_dynamic_mx_quant_apt.cpp
 * \brief rms_norm_dynamic_mx_quant kernel entry
 */

#include "arch35/rms_norm_dynamic_mx_quant_full_load_general.h"

using namespace AscendC;
using namespace RmsNormDynamicMxQuantNs;

#define TILING_KEY_FULL_LOAD_GENERAL 1000

#define FLOAT_OVERFLOW_MODE_CTRL 60

extern "C" __global__ __aicore__ void rms_norm_dynamic_mx_quant(
    GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mxscale, GM_ADDR rstd, GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }

    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }

    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);

#if (__NPU_ARCH__ == 3510)
    int64_t oriOverflowMode = AscendC::GetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>();
#endif

    if (TILING_KEY_IS(TILING_KEY_FULL_LOAD_GENERAL)) {
        if constexpr (std::is_same<DTYPE_Y, fp8_e4m3fn_t>::value || std::is_same<DTYPE_Y, fp8_e5m2_t>::value) {
            GET_TILING_DATA_WITH_STRUCT(RmsNormDynamicMxQuantFullLoadGeneralTilingData, tilingDataIn, tiling);
            const RmsNormDynamicMxQuantFullLoadGeneralTilingData* __restrict tilingData = &tilingDataIn;
            TPipe pipe;
            KernelRmsNormDynamicMxQuantFullLoadGeneral<DTYPE_X, DTYPE_GAMMA, DTYPE_Y> op(&pipe);
            op.Init(x, gamma, beta, y, mxscale, rstd, userWS, tilingData);
            op.Process();
        }
    }

#if (__NPU_ARCH__ == 3510)
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(oriOverflowMode);
#endif
}

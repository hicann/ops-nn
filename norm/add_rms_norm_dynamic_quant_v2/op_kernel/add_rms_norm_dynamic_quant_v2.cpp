/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file add_rms_norm_dynamic_quant_v2.cpp
 * \brief
 */
#include "add_rms_norm_dynamic_quant_v2_normal_kernel.h"
#include "add_rms_norm_dynamic_quant_v2_single_row_kernel.h"
#include "add_rms_norm_dynamic_quant_v2_cut_d_kernel.h"

#define INIT_AND_PROCESS                                                                                              \
    do {                                                                                                              \
        op.Init(x1, x2, gamma, smooth1, smooth2, y1, y2, y3, y4, x, outScale1, outScale2, usrWorkspace, &tilingData); \
        op.Process();                                                                                                 \
    } while (0)

extern "C" __global__ __aicore__ void add_rms_norm_dynamic_quant_v2(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR smooth1, GM_ADDR smooth2, GM_ADDR y1, GM_ADDR y2, GM_ADDR y3,
    GM_ADDR y4, GM_ADDR x, GM_ADDR outScale1, GM_ADDR outScale2, GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);
    GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);

    if (TILING_KEY_IS(0)) {
        // 0 Tiling, Do Nothing.
    } else if (TILING_KEY_IS(1)) {
        KernelAddRmsNormDynamicQuantV2Normal<DTYPE_X1, 1> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(2)) {
        KernelAddRmsNormDynamicQuantV2SingleRow<DTYPE_X1, 2> op(&pipe);
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(3)) {
        KernelAddRmsNormDynamicQuantV2SliceD<DTYPE_X1, 3> op(&pipe);
        INIT_AND_PROCESS;
    }
}
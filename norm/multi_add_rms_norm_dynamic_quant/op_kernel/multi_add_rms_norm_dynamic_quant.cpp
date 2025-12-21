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
 * \file multi_add_rms_norm_dynamic_quant.cpp
 * \brief
 */
#include "multi_add_rms_norm_dynamic_quant_normal_kernel.h"
#include "multi_add_rms_norm_dynamic_quant_single_row_kernel.h"
#include "multi_add_rms_norm_dynamic_quant_cut_d_kernel.h"
#include "kernel_operator.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void multi_add_rms_norm_dynamic_quant(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR smooth1, GM_ADDR smooth2, GM_ADDR y1, GM_ADDR y2, GM_ADDR x,
    GM_ADDR y, GM_ADDR outScale1, GM_ADDR outScale2, GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);
    GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);

    if (TILING_KEY_IS(1)) {
        KernelMultiAddRmsNormDynamicQuantNormal<DTYPE_X1, 1, 1> op(&pipe);
        op.Init(x1, x2, gamma, smooth1, smooth2, y1, y2, x, y, outScale1, outScale2, workspace, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        KernelMultiAddRmsNormDynamicQuantSingleRow<DTYPE_X1, 2, 1> op(&pipe);
        op.Init(x1, x2, gamma, smooth1, smooth2, y1, y2, x, y, outScale1, outScale2, workspace, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(3)) {
        KernelMultiAddRmsNormDynamicQuantSliceD<DTYPE_X1, 3, 1> op(&pipe);
        op.Init(x1, x2, gamma, smooth1, smooth2, y1, y2, x, y, outScale1, outScale2, workspace, &tilingData);
        op.Process();
    }
}
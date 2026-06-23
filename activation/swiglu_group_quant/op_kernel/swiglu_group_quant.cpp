/**
 * Copyright (c) 2026 Huawei Technologies
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file swiglu_group_quant.cpp
 * \brief Kernel entry point for SwiGLU Group Quant operator
 *
 * Unified kernel: isGroup/outputOrigin branching is handled inside SwigluGroupQuantKernel::Process.
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "swiglu_group_quant.h"

using namespace AscendC;
using namespace SwigluGroupQuantOps;

extern "C" __global__ __aicore__ void swiglu_group_quant(
    GM_ADDR xGM, GM_ADDR weightGM, GM_ADDR groupIndexGM, GM_ADDR scaleGM,
    GM_ADDR yGM, GM_ADDR yScaleGM, GM_ADDR yOriginGM,
    GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }
    GM_ADDR userspace = GetUserWorkspace(workspace);

    GET_TILING_DATA(tilingData, tiling);

    SwigluGroupQuantKernel<DTYPE_X> op;
    op.Init(xGM, weightGM, groupIndexGM, yGM, yScaleGM, yOriginGM, userspace, &tilingData);
    op.Process();
}

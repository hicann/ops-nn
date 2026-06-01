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
 * \file clipped_swiglu_apt.cpp
 * \brief Kernel entry for ClippedSwiglu Arch35 (Ascend 950)
 */

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "arch35/clipped_swiglu_tiling_key.h"
#include "arch35/clipped_swiglu_tiling_data.h"
#include "arch35/clipped_swiglu_kernel.h"

using namespace AscendC;
using namespace ClippedSwigluOp;

template <uint64_t isInterleaved, uint64_t isGroup>
__global__ __aicore__ void clipped_swiglu(GM_ADDR x, GM_ADDR groupIndex, GM_ADDR y,
    GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(ClippedSwigluArch35TilingData);
    GET_TILING_DATA_WITH_STRUCT(ClippedSwigluArch35TilingData, tilingData, tiling);
    GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);
    TPipe pipe;

    if constexpr (isInterleaved == 1) {
        if constexpr (isGroup == 1) {
            ClippedSwigluKernel<DTYPE_X, true, true> op(&tilingData, &pipe);
            op.Init(x, groupIndex, y);
            op.Process();
        } else {
            ClippedSwigluKernel<DTYPE_X, true, false> op(&tilingData, &pipe);
            op.Init(x, groupIndex, y);
            op.Process();
        }
    } else {
        if constexpr (isGroup == 1) {
            ClippedSwigluKernel<DTYPE_X, false, true> op(&tilingData, &pipe);
            op.Init(x, groupIndex, y);
            op.Process();
        } else {
            ClippedSwigluKernel<DTYPE_X, false, false> op(&tilingData, &pipe);
            op.Init(x, groupIndex, y);
            op.Process();
        }
    }
}

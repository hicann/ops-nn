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
 * \file linear_index_v2_apt.cpp
 * \brief
 */
#include "arch35/linear_index_v2_simt.h"

namespace AscendC {
extern "C" __global__ __aicore__ void linear_index_v2(
    GM_ADDR indexList, GM_ADDR stride, GM_ADDR valueSize, GM_ADDR output, GM_ADDR workSpace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    if (workSpace == nullptr) {
        return;
    }
    GM_ADDR user = AscendC::GetUserWorkspace(workSpace);
    if (user == nullptr) {
        return;
    }
    GET_TILING_DATA(tilingData, tiling);
    AscendC::TPipe pipe;
    if (TILING_KEY_IS(0)) {
        LinearIndexSimtKernelV2<int64_t> op(indexList, stride, valueSize, output, workSpace, tilingData, pipe);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        LinearIndexSimtKernelV2<int32_t> op(indexList, stride, valueSize, output, workSpace, tilingData, pipe);
        op.Process();
    }
}
} // namespace AscendC
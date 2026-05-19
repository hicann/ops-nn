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
 * \file index_check.cpp
 * \brief
 */
#include "index_check.h"

namespace AscendC {
extern "C" __global__ __aicore__ void index_check(
    GM_ADDR bounds, GM_ADDR indexList, GM_ADDR workSpace, GM_ADDR tiling)
{
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
        IndexCheckKernel<int64_t> op(bounds, indexList, workSpace, tilingData, pipe);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        IndexCheckKernel<int32_t> op(bounds, indexList, workSpace, tilingData, pipe);
        op.Process();
    }
}
} // namespace AscendC

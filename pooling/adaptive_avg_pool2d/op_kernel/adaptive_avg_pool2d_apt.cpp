/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file adaptive_avg_pool2d_apt.cpp
 * \brief
 */

#include "arch35/adaptive_avg_pool2d_simt.h"
#include "arch35/adaptive_avg_pool2d_struct.h"

using namespace AdaptiveAvgPool2dOp;
template <uint64_t TEMPLATE_MODE = TPL_SIMT_KERNEL, uint64_t DTYPE_MODE = TPL_INT32_UINT32>
__global__ __aicore__ void adaptive_avg_pool2d(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr || GetUserWorkspace(workspace) == nullptr || g_coreType == AIC) {
        return;
    }
    TPipe pipe;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(AdaptivePool2DSimtTilingData);
    if constexpr (TEMPLATE_MODE == TPL_SIMT_KERNEL) {
        REGISTER_TILING_FOR_TILINGKEY("TEMPLATE_MODE == TPL_SIMT_KERNEL", AdaptivePool2DSimtTilingData);
        GET_TILING_DATA_WITH_STRUCT(AdaptivePool2DSimtTilingData, tilingData, tiling);
        if constexpr (DTYPE_MODE == TPL_INT32_UINT32) {
            AdaptiveAvgPool2dSimt<DTYPE_X, uint32_t> op(&pipe, &tilingData);
            op.Init(x, y);
            op.Process();
        }else {
            AdaptiveAvgPool2dSimt<DTYPE_X, uint64_t> op(&pipe, &tilingData);
            op.Init(x, y);
            op.Process();
        }
    }
}

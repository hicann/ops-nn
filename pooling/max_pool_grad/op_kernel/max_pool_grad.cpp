/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "arch35/max_pool_grad_struct.h"
#include "arch35/max_pool_grad_simt.h"

using namespace AscendC;
using MaxPoolGradWithArgmaxNHWCNameSpace::MaxPoolGradWithArgmaxNCHWTilingCommonData;
using MaxPoolGradWithArgmaxNHWCNameSpace::MaxPoolGradWithArgmaxSimtTilingCommonData;
using namespace PoolGradNameSpace;

template <
    uint64_t KERNEL_MODE = TPL_SIMT_KERNEL, uint64_t FORMAT = TPL_NCHW_FORMAT, uint64_t INDICES_DTYPE = TPL_INT32,
    uint64_t IS_CHECK_RANGE = TPL_NO_CHECK_RANGE>
__global__ __aicore__ void max_pool_grad(
    GM_ADDR orig_x, GM_ADDR orig_y, GM_ADDR grads, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr || GetUserWorkspace(workspace) == nullptr || g_coreType == AIC) {
        return;
    }
    TPipe pipe;
    REGISTER_TILING_DEFAULT(MaxPoolGradWithArgmaxSimtTilingCommonData);

    if constexpr (KERNEL_MODE == TPL_SIMT_KERNEL) {
        GET_TILING_DATA_WITH_STRUCT(MaxPoolGradWithArgmaxSimtTilingCommonData, tilingData, tiling);
        if constexpr (INDICES_DTYPE == TPL_INT32) {
            MaxPoolGrad::MaxPoolGradSIMT<DTYPE_X1, int32_t, FORMAT> op(&pipe, &tilingData);
            op.Init(orig_x, orig_y, grads, y, workspace);
            op.Process();
        } else {
            MaxPoolGrad::MaxPoolGradSIMT<DTYPE_X1, int64_t, FORMAT> op(&pipe, &tilingData);
            op.Init(orig_x, orig_y, grads, y, workspace);
            op.Process();
        }
    }
}
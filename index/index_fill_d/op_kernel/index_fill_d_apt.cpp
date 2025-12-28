/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file index_fill_d.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "arch35/index_fill_d.h"

using namespace AscendC;

#define TILING_KEY_BOOL      101
#define TILING_KEY_COMMON    200


extern "C" __global__ __aicore__ void index_fill_d(GM_ADDR x, GM_ADDR assist1, GM_ADDR assist2,
                                                    GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }
    TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    if (TILING_KEY_IS(TILING_KEY_COMMON)) {
        IndexFillD<DTYPE_X> op(tilingData, pipe);
        op.Init(x, assist1, assist2, y, userWS);
        op.Process();
    }
    return;
}
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "centralization_tiling_data.h"
#include "centralization_empty.h"
#include "centralization_fast.h"
#include "centralization_generic.h"
#include "centralization_large_trailing.h"

#define CENTRALIZATION_EMPTY_TILING_KEY 8000
#define CENTRALIZATION_FAST_TILING_KEY 7000
#define CENTRALIZATION_LARGE_TILING_KEY 7020
#define CENTRALIZATION_LARGE_TRAILING_TILING_KEY 7030
#define CENTRALIZATION_GENERIC_TILING_KEY 0

extern "C" __global__ __aicore__ void centralization(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(CentralizationTilingData);
    GET_TILING_DATA_WITH_STRUCT(CentralizationTilingData, tilingDataIn, tiling);
    const CentralizationTilingData* tilingData = &tilingDataIn;
    GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);
    AscendC::TPipe pipe;
    if (TILING_KEY_IS(CENTRALIZATION_EMPTY_TILING_KEY)) {
        Centralization::Empty<DTYPE_X> op;
        op.Init(y);
        op.Process();
    } else if (TILING_KEY_IS(CENTRALIZATION_FAST_TILING_KEY)) {
        Centralization::Fast<DTYPE_X> op(&pipe, tilingData);
        op.Init(x, y);
        op.Process();
    } else if (TILING_KEY_IS(CENTRALIZATION_LARGE_TILING_KEY)) {
        Centralization::Generic<DTYPE_X> op(&pipe, tilingData);
        op.Init(x, y, userWorkspace);
        op.Process();
    } else if (TILING_KEY_IS(CENTRALIZATION_LARGE_TRAILING_TILING_KEY)) {
        Centralization::LargeTrailing<DTYPE_X> op(&pipe, tilingData);
        op.Init(x, y, userWorkspace);
        op.Process();
    } else if (TILING_KEY_IS(CENTRALIZATION_GENERIC_TILING_KEY)) {
        Centralization::Generic<DTYPE_X> op(&pipe, tilingData);
        op.Init(x, y, userWorkspace);
        op.Process();
    } else {
        Centralization::Generic<DTYPE_X> op(&pipe, tilingData);
        op.Init(x, y, userWorkspace);
        op.Process();
    }
}

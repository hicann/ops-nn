/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "index_kernel.h"

using optiling::IndexTilingData;

static __aicore__ inline void RunIndex(GM_ADDR x, GM_ADDR indexedSizes, GM_ADDR indexedStrides, GM_ADDR indices,
                                       GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace != nullptr) {
        AscendC::SetSysWorkspace(workspace);
    }
    REGISTER_TILING_DEFAULT(IndexTilingData);
    GET_TILING_DATA_WITH_STRUCT(IndexTilingData, tilingData, tiling);
    IndexExperimental::IndexKernel<DTYPE_X, DTYPE_INDICES> op;
    op.Init(x, indexedSizes, indexedStrides, indices, y, &tilingData);
    op.Process();
}

extern "C" __global__ __aicore__ void index(GM_ADDR x, GM_ADDR indexedSizes, GM_ADDR indexedStrides, GM_ADDR indices,
                                            GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    RunIndex(x, indexedSizes, indexedStrides, indices, y, workspace, tiling);
}

extern "C" __global__ __aicore__ void index_ai_core(GM_ADDR x, GM_ADDR indexedSizes, GM_ADDR indexedStrides,
                                                    GM_ADDR indices, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    RunIndex(x, indexedSizes, indexedStrides, indices, y, workspace, tiling);
}

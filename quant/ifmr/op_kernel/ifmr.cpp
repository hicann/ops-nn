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
 * \file ifmr.cpp
 * \brief
 */

#include "ifmr.h"

using namespace AscendC;
 
extern "C" __global__ __aicore__ void ifmr(
    GM_ADDR data,
    GM_ADDR data_min,
    GM_ADDR data_max,
    GM_ADDR cumsum,
    GM_ADDR scale,
    GM_ADDR offset,
    GM_ADDR workspace,
    GM_ADDR tiling) {
    GM_ADDR user_workspace = AscendC::GetUserWorkspace(workspace);
    REGISTER_TILING_DEFAULT(IfmrTilingData);
    GET_TILING_DATA_WITH_STRUCT(IfmrTilingData, tilingData, tiling);
 
#if defined(DTYPE_DATA)
    if (TILING_KEY_IS(0)) {
        Ifmr::KernelIfmr<DTYPE_DATA> op;
        op.Init(data, data_min, data_max, cumsum, scale, offset, workspace, tilingData);
        op.Process();
    }
#endif
 
    return;
}

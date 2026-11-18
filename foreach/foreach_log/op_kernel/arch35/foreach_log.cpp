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
 * \file foreach_log.cpp
 * \brief Flat RegBase kernel entry for ForeachLog (arch35 / Ascend950).
 */

#include "foreach_log_regbase.h"

using namespace ForeachLog;

extern "C" __global__ __aicore__ void foreach_log(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ForeachLogTilingData);
    GET_TILING_DATA_WITH_STRUCT(ForeachLogTilingData, tilingDataIn, tiling);
    (void)workspace;

    TPipe pipe;
    ForeachLogRegbase<DTYPE_X, ForeachLogTilingData> op;
    op.Init(&tilingDataIn, &pipe);
    op.Process(x, y);
    pipe.Destroy();
}

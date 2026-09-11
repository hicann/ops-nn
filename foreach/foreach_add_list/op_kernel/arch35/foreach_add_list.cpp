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
 * \file foreach_add_list.cpp
 * \brief Flat RegBase kernel entry for ForeachAddList.
 */
#include "foreach_add_list_regbase.h"
#include "../../foreach_utils/arch35/foreach_flat_tiling_data.h"

using namespace ForeachAddList;

extern "C" __global__ __aicore__ void foreach_add_list(GM_ADDR x1, GM_ADDR x2, GM_ADDR alpha, GM_ADDR y,
                                                       GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ForeachFlatTilingData);
    GET_TILING_DATA_WITH_STRUCT(ForeachFlatTilingData, tilingDataIn, tiling);
    (void)workspace;

    TPipe pipe;
    ForeachAddListRegbase<DTYPE_X1, DTYPE_ALPHA, ForeachFlatTilingData> op;
    op.Init(&tilingDataIn, &pipe);
    op.InitScalar(alpha);
    op.Process(x1, x2, y);
    pipe.Destroy();
}

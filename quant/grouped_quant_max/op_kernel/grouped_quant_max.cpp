/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "arch35/grouped_quant_max_regbase.h"
#include "arch35/grouped_quant_max_tiling_data.h"
#include "arch35/grouped_quant_max_struct.h"

#define FLOAT_OVERFLOW_MODE_CTRL 60

using namespace AscendC;
using namespace GroupedQuantMax;

template <uint64_t RoundMode>
__global__ __aicore__ void grouped_quant_max(GM_ADDR x, GM_ADDR scale, GM_ADDR groupList, GM_ADDR y, GM_ADDR amax,
                                             GM_ADDR workspace, GM_ADDR tiling)
{
    int64_t oriOverflowMode = GetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>();
    REGISTER_TILING_DEFAULT(GroupedQuantMaxTilingData);
    GET_TILING_DATA_WITH_STRUCT(GroupedQuantMaxTilingData, tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    GroupedQuantMaxRegbase<DTYPE_X, DTYPE_Y, RoundMode> op(&tilingData);
    GM_ADDR usrWorkspace = GetUserWorkspace(workspace);
    op.Init(x, scale, groupList, y, amax, usrWorkspace);
    op.Process();
    SyncAll();
    op.MergeAmax();
    SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(oriOverflowMode);
}

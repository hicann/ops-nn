/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "arch35/quant_max_v2_regbase.h"
#include "arch35/quant_max_v2_tiling_data.h"
#include "arch35/quant_max_v2_struct.h"

using namespace AscendC;
using namespace QuantMaxV2;

// 通过 QuantMode 和 HasGroup 在编译期选择量化场景，所有场景都输出 amax。
template <uint64_t RoundMode, uint64_t QuantMode, uint64_t HasGroup>
__global__ __aicore__ void quant_max_v2(GM_ADDR x, GM_ADDR scale, GM_ADDR groupList, GM_ADDR y, GM_ADDR amax,
                                        GM_ADDR yScale, GM_ADDR workspace, GM_ADDR tiling)
{
    int64_t oriOverflowMode = GetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>();
    REGISTER_TILING_DEFAULT(QuantMaxV2TilingData);
    GET_TILING_DATA_WITH_STRUCT(QuantMaxV2TilingData, tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    GM_ADDR usrWorkspace = GetUserWorkspace(workspace);
    static_assert(QuantMode == TPL_QUANT_MODE_STATIC || QuantMode == TPL_QUANT_MODE_DYNAMIC, "不支持的量化模式");
    static_assert(HasGroup == TPL_GROUP_MODE_TENSOR || HasGroup == TPL_GROUP_MODE_GROUP, "不支持的分组模式");
    QuantMaxV2Kernel<DTYPE_X, DTYPE_Y, RoundMode, QuantMode, HasGroup> op(&tilingData);
    op.Init(x, scale, groupList, y, amax, yScale, usrWorkspace);
    op.Process();
    SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(oriOverflowMode);
}

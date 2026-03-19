/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file concat_offset_apt.cpp
 * \brief
 */

#include "arch35/concat_offset_simt.h"

using namespace AscendC;
using namespace ConcatOffset;

#define CONCAT_OFFSET_SIMT_TILING_KEY 1000


extern "C" __global__ __aicore__ void concat_offset(GM_ADDR concat_dim, GM_ADDR x, GM_ADDR y,
                                                    GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ConcatOffsetTilingData);
    GET_TILING_DATA(tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY); 
    if (TILING_KEY_IS(CONCAT_OFFSET_SIMT_TILING_KEY)) {
        ConcatOffsetSimt<DTYPE_X> op;
        op.Init(&tilingData);
        op.Process(x, y);
    }
}
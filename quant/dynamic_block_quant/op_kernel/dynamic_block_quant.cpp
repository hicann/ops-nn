/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dynamic_block_quant.cpp
 * \brief
 */
#include "dynamic_block_quant.h"

using namespace DynamicBlockQuant;
using namespace AscendC;

extern "C" __global__ __aicore__ void dynamic_block_quant(
    GM_ADDR x, GM_ADDR y, GM_ADDR scale, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);

    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }

#if defined(ORIG_DTYPE_X) && (ORIG_DTYPE_X == DT_FLOAT16)
    if (TILING_KEY_IS(0)) {
        if ASCEND_IS_AIV {
            DynamicBlockQuantND<half> op;
            op.Init(x, y, scale, userWS, &tilingData);
            op.Process();
        }
    }
#endif

#if defined(ORIG_DTYPE_X) && (ORIG_DTYPE_X == DT_BF16)
    if (TILING_KEY_IS(10)) {
        if ASCEND_IS_AIV {
            DynamicBlockQuantND<bfloat16_t> op;
            op.Init(x, y, scale, userWS, &tilingData);
            op.Process();
        }
    }
#endif
}
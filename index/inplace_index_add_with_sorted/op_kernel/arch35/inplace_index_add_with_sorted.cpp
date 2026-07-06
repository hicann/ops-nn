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
 * \file inplace_index_add_with_sorted_apt.cpp
 * \brief A5 (ascend950) kernel entry — TILING_KEY_IS runtime dispatch
 */
#include "inplace_index_add_with_sorted_struct.h"
#include "inplace_index_add_with_sorted_fix.h"

#define SORTED_TILING_KEY 10000

extern "C" __global__ __aicore__ void inplace_index_add_with_sorted(
    GM_ADDR var, GM_ADDR value, GM_ADDR sorted_indices, GM_ADDR pos,
    GM_ADDR alpha, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    AscendC::TPipe pipe;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_NONE_TILING;

    if (TILING_KEY_IS(SORTED_TILING_KEY)) {
        GET_TILING_DATA_WITH_STRUCT(InplaceIndexAddWithSortedTilingData, tilingData, tiling);
        InplaceIndexAddWithSortedFix<DTYPE_VAR> op(&pipe, &tilingData);
        op.Init(var, value, sorted_indices, pos, alpha);
        op.Process();
    }
}

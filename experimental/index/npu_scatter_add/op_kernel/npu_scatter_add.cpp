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
 * \file npu_scatter_add.cpp
 * \brief NpuScatterAdd kernel 入口：tilingKey 0~7 分发 dtype/缩放/精度模板分支
 */
#include "kernel_operator.h"
#include "npu_scatter_add.h"

extern "C" __global__ __aicore__ void npu_scatter_add(GM_ADDR x, GM_ADDR y, GM_ADDR s, GM_ADDR indices,
                                                      GM_ADDR sort_idx, GM_ADDR valid_token_num, GM_ADDR y_ref,
                                                      GM_ADDR workspace, GM_ADDR tiling)
{
    if ASCEND_IS_AIC {
        return;
    }

    GET_TILING_DATA(tiling_data, tiling);
    using namespace NpuScatterAddKernel;

    SetSysWorkspace(workspace);
    __gm__ uint8_t* user = GetUserWorkspace(workspace);

    if (TILING_KEY_IS(0)) {
        using DataType = bfloat16_t;
        NpuScatterAdd<DataType> npuScatterAdd(tiling_data, x, y, s, indices, sort_idx, valid_token_num, user);
        npuScatterAdd.Process();
    } else if (TILING_KEY_IS(1)) {
        using DataType = half;
        NpuScatterAdd<DataType> npuScatterAdd(tiling_data, x, y, s, indices, sort_idx, valid_token_num, user);
        npuScatterAdd.Process();
    } else if (TILING_KEY_IS(2)) {
        using DataType = bfloat16_t;
        constexpr bool WithScale = false;
        NpuScatterAdd<DataType, WithScale> npuScatterAdd(tiling_data, x, y, s, indices, sort_idx, valid_token_num,
                                                         user);
        npuScatterAdd.Process();
    } else if (TILING_KEY_IS(3)) {
        using DataType = half;
        constexpr bool WithScale = false;
        NpuScatterAdd<DataType, WithScale> npuScatterAdd(tiling_data, x, y, s, indices, sort_idx, valid_token_num,
                                                         user);
        npuScatterAdd.Process();
    }
    // high precision
    else if (TILING_KEY_IS(4)) {
        using DataType = bfloat16_t;
        constexpr bool WithScale = true;
        constexpr bool UseHighPrecision = true;
        NpuScatterAdd<DataType, WithScale, UseHighPrecision> npuScatterAdd(tiling_data, x, y, s, indices, sort_idx,
                                                                           valid_token_num, user);
        npuScatterAdd.Process();
    } else if (TILING_KEY_IS(5)) {
        using DataType = half;
        constexpr bool WithScale = true;
        constexpr bool UseHighPrecision = true;
        NpuScatterAdd<DataType, WithScale, UseHighPrecision> npuScatterAdd(tiling_data, x, y, s, indices, sort_idx,
                                                                           valid_token_num, user);
        npuScatterAdd.Process();
    } else if (TILING_KEY_IS(6)) {
        using DataType = bfloat16_t;
        constexpr bool WithScale = false;
        constexpr bool UseHighPrecision = true;
        NpuScatterAdd<DataType, WithScale, UseHighPrecision> npuScatterAdd(tiling_data, x, y, s, indices, sort_idx,
                                                                           valid_token_num, user);
        npuScatterAdd.Process();
    } else if (TILING_KEY_IS(7)) {
        using DataType = half;
        constexpr bool WithScale = false;
        constexpr bool UseHighPrecision = true;
        NpuScatterAdd<DataType, WithScale, UseHighPrecision> npuScatterAdd(tiling_data, x, y, s, indices, sort_idx,
                                                                           valid_token_num, user);
        npuScatterAdd.Process();
    }
}

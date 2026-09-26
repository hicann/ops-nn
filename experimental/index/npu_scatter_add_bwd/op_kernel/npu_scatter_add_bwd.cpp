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
 * \file npu_scatter_add_bwd.cpp
 * \brief NpuScatterAddBwd kernel 入口：tilingKey 0/1 分发 bf16/fp16 模板分支
 */
#include "kernel_operator.h"
#include "npu_scatter_add_bwd.h"

extern "C" __global__ __aicore__ void npu_scatter_add_bwd(GM_ADDR y_grad, GM_ADDR x, GM_ADDR s, GM_ADDR indices,
                                                          GM_ADDR x_grad, GM_ADDR s_grad, GM_ADDR workspace,
                                                          GM_ADDR tiling)
{
    if ASCEND_IS_AIC {
        return;
    }

    GET_TILING_DATA(tiling_data, tiling);
    using namespace NpuScatterAddBwdKernel;

    SetSysWorkspace(workspace);
    __gm__ uint8_t* user = GetUserWorkspace(workspace);

    if (TILING_KEY_IS(0)) {
        using DataType = bfloat16_t;
        NpuScatterAddBwd<DataType> npuScatterAddBwd(tiling_data, y_grad, x, s, indices, x_grad, s_grad, user);
        npuScatterAddBwd.Process();
    } else if (TILING_KEY_IS(1)) {
        using DataType = half;
        NpuScatterAddBwd<DataType> npuScatterAddBwd(tiling_data, y_grad, x, s, indices, x_grad, s_grad, user);
        npuScatterAddBwd.Process();
    }
}

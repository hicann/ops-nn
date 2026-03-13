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
 * \file broadcast_gradient_args_apt.cpp
 * \brief
 */

#include "./arch35/broadcast_gradient_args_perf.h"
#include "./arch35/broadcast_gradient_args_scalar.h"

#define PERF_TEMPLATE_TILING_KEY 0
#define SCALAR_TEMPLATE_TILING_KEY 1

extern "C" __global__ __aicore__ void broadcast_gradient_args(GM_ADDR x1, GM_ADDR x2, GM_ADDR y1, GM_ADDR y2, 
    GM_ADDR out_shape, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(PERF_TEMPLATE_TILING_KEY)) {
        AscendC::InitSocState();
        BroadcastGradientArgs::BroadcastGradientArgsPerf<DTYPE_X1> op;
        op.Init(x1, x2, y1, y2, out_shape, &tilingData);
        op.Process();
        return;
    } else if (TILING_KEY_IS(SCALAR_TEMPLATE_TILING_KEY)) {
        AscendC::InitSocState();
        BroadcastGradientArgs::BroadcastGradientArgsScalar<DTYPE_X1> op;
        op.Init(x1, x2, y1, y2, out_shape, &tilingData);
        op.Process();
        return;
    }
}

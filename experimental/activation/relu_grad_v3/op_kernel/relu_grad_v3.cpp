/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Shi Xiangyang <@shi-xiangyang225>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file relu_grad_v3.cpp
 * \brief ReluGradV3算子的kernel入口函数
 */

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "relu_grad_v3_broadcast.h"
#include "relu_grad_v3_normal.h"
#include "relu_grad_v3_tiling_data.h"
#include "relu_grad_v3_tiling_key.h"

template <uint32_t schMode>
__global__ __aicore__ void relu_grad_v3(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ReluGradV3TilingData);
    GET_TILING_DATA_WITH_STRUCT(ReluGradV3TilingData, tilingData, tiling);

    AscendC::TPipe pipe;
    if constexpr (schMode == RELU_GRAD_V3_TPL_SCH_MODE_BROADCAST) {
        NsReluGradV3::ReluGradV3Broadcast<DTYPE_X> op;
        op.Init(x, y, z, &tilingData, pipe);
        op.Process();
    } else {
        NsReluGradV3::ReluGradV3Normal<DTYPE_X> op;
        op.Init(x, y, z, &tilingData, pipe);
        op.Process();
    }
}

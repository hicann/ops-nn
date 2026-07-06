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
 * \file softmax_grad.cpp
 * \brief SoftmaxGrad算子的kernel入口函数
 */

#include "softmax_grad.h"

__global__ __aicore__ void softmax_grad(GM_ADDR y, GM_ADDR dy, GM_ADDR dx,
                                        GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SoftmaxGradTilingData);
    GET_TILING_DATA_WITH_STRUCT(SoftmaxGradTilingData, tilingData, tiling);

    NsSoftmaxGrad::SoftmaxGrad<DTYPE_Y> op;
    op.Init(y, dy, dx, &tilingData);
    op.Process();
}

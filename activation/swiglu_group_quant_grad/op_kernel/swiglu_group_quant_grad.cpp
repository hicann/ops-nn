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
 * \file swiglu_group_quant_grad.cpp
 * \brief SwiGLU Group Dynamic Quant Backward kernel entry for Ascend 950 (A5)
 */

#include "swiglu_group_quant_grad.h"

using namespace SwigluGroupQuantGradOp;

extern "C" __global__ __aicore__ void swiglu_group_quant_grad(GM_ADDR gradY, GM_ADDR x, GM_ADDR weight,
                                                          GM_ADDR yOrigin, GM_ADDR groupIndex, GM_ADDR gradX,
                                                          GM_ADDR gradWeight, GM_ADDR workspace, GM_ADDR tiling)
{
    SwigluGroupQuantGrad<DTYPE_X> op;
    op.Init(gradY, x, weight, yOrigin, groupIndex, gradX, gradWeight, workspace, tiling);
    op.Process();
}
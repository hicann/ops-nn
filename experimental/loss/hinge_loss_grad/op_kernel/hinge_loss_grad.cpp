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
 * \file hinge_loss_grad.cpp
 * \brief Kernel entry for HingeLossGrad
 */

#include "hinge_loss_grad.h"

#ifndef DTYPE_PREDICT
#define DTYPE_PREDICT half
#endif

extern "C" __global__ __aicore__ void hinge_loss_grad(GM_ADDR predict, GM_ADDR target, GM_ADDR gradOutput,
                                                       GM_ADDR gradInput, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(HingeLossGradTilingData);
    GET_TILING_DATA_WITH_STRUCT(HingeLossGradTilingData, tilingData, tiling);

    NsHingeLossGrad::HingeLossGrad<DTYPE_PREDICT> op;
    op.Init(predict, target, gradOutput, gradInput, &tilingData);
    op.Process();
}

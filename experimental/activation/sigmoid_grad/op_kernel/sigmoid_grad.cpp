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
 * \file sigmoid_grad.cpp
 * \brief
*/

#include "sigmoid_grad.h"


template <uint32_t schMode>
__global__ __aicore__ void sigmoid_grad(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
     REGISTER_TILING_DEFAULT(SigmoidGradTilingData);
     GET_TILING_DATA_WITH_STRUCT(SigmoidGradTilingData, tilingData, tiling);
     NsSigmoidGrad::KernelSigmoidGrad<DTYPE_X> op; 
     op.Init(x, y, z, &tilingData);      // 算子kernel实例初始化
     op.Process();                       
}

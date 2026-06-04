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
 * \file nll_loss_grad.cpp
 * \brief nll_loss_grad kernel
 */

 #include "arch35/nll_loss_grad.h"

 using namespace NLLLossGrad;

 #define COMMON_OP_IMPL(templateClass, ...)                  \
 do {                                                    \
     templateClass<__VA_ARGS__> op;               \
     op.Init(x, y_grad, target, weight, total_weight, x_grad, tilingData);  \
     op.Process();                                       \
 } while (0)

 extern "C" __global__ __aicore__ void nll_loss_grad(GM_ADDR x, GM_ADDR y_grad, GM_ADDR target, GM_ADDR weight,
     GM_ADDR total_weight, GM_ADDR x_grad, GM_ADDR workspace, GM_ADDR tiling)
 {
     if (workspace == nullptr) {
         return;
     }
     SetSysWorkspace(workspace);
     GM_ADDR userWS = GetUserWorkspace(workspace);
     if (userWS == nullptr) {
         return;
     }
     GET_TILING_DATA(tilingData, tiling);
     KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
     COMMON_OP_IMPL(NLLLossGrad::KernelNLLLossGrad, DTYPE_X, DTYPE_TARGET);
 }
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
 * \file l1_loss_grad.cpp
 * \brief l1_loss_grad source file
 */
#include "kernel_operator.h"
#include "arch35/l1_loss_grad_dag.h"
#include "arch35/l1_loss_grad_tiling_key.h"
#include "atvoss/broadcast/broadcast_sch.h"

using namespace AscendC;

 template <uint64_t schMode, uint32_t inputGradsIsScalar>
 __global__ __aicore__ void l1_loss_grad(GM_ADDR grads, GM_ADDR predict, GM_ADDR label, GM_ADDR y, 
                                         GM_ADDR workspace, GM_ADDR tiling) {
     if (workspace == nullptr) {
        return;
    }

    SetSysWorkspace(workspace);
    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }
    
    if constexpr (inputGradsIsScalar == static_cast<uint32_t>(ATTR_IS_TRUE)) {
        using OpDag = L1LossGradKernel::L1LossGradScalarCast<DTYPE_PREDICT, float>::OpDag;
        BroadcastSch<schMode, OpDag> sch(tiling);
        sch.Process(grads, predict, label, y);
     } else {
        using OpDag = L1LossGradKernel::L1LossGradCast<DTYPE_PREDICT, float>::OpDag;
        BroadcastSch<schMode, OpDag> sch(tiling);
        sch.Process(grads, predict, label, y);
     }
 }

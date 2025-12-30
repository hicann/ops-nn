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
 * \file kl_div_loss_grad_apt.cpp
 * \brief kl_div_loss_grad_apt
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "arch35/kl_div_loss_grad_dag.h"
#include "arch35/kl_div_loss_grad_tiling_key.h"
#include "atvoss/broadcast/broadcast_sch.h"

using namespace AscendC;
using namespace KlDivLossGrad;

template <uint64_t schMode, uint32_t logTarget>
__global__ __aicore__ void kl_div_loss_grad(GM_ADDR grad, GM_ADDR input, GM_ADDR target,  
        GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    if constexpr (static_cast<uint32_t>(logTarget) == static_cast<uint32_t>(KDLG_TRUE)) {
        using OpDag = KlDivLossGrad::KDLGLogTargetTrue<DTYPE_GRAD>::OpDag;
        BroadcastSch<schMode, OpDag> sch(tiling);
        sch.Process(grad, input, target, y);
    } else {
        using OpDag = KlDivLossGrad::KDLGLogTargetFalse<DTYPE_GRAD>::OpDag;
        BroadcastSch<schMode, OpDag> sch(tiling);
        sch.Process(grad, input, target, y);
    }
}
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
 * \file logsigmoid_grad.cpp
 * \brief logsigmoid_grad
 */

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "atvoss/broadcast/broadcast_sch.h"
#include "atvoss/util/dfx.h"
#include "arch35/logsigmoid_grad_dag.h"
#include "arch35/logsigmoid_grad_struct.h"

using namespace AscendC;
using namespace LogSigmoidGradOp;

template <uint64_t schMode>
__global__ __aicore__ void log_sigmoid_grad(
    GM_ADDR grads, GM_ADDR features, GM_ADDR backprops, GM_ADDR workspace, GM_ADDR tiling)
{
    using OpDag = LogSigmoidGradDag<DTYPE_GRADS>::OpDag;
    BroadcastSch<schMode, OpDag> sch(tiling);
    sch.Process(grads, features, backprops);
    return;
}

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
 * \file adam_apply_one_with_decay.cpp
 * \brief adam_apply_one_with_decay.cpp
 */

#include "kernel_operator.h"
#include "arch35/adam_apply_one_with_decay_dag.h"
#include "arch35/adam_apply_one_with_decay_tiling_key.h"
#include "atvoss/broadcast/broadcast_sch.h"

using namespace AscendC;
using namespace Ops::Base;

template <uint64_t schMode>
__global__ __aicore__ void adam_apply_one_with_decay(
    GM_ADDR input0, GM_ADDR input1, GM_ADDR input2, GM_ADDR input3, GM_ADDR input4, GM_ADDR mul0_x, GM_ADDR mul1_x,
    GM_ADDR mul2_x, GM_ADDR mul3_x, GM_ADDR mul4_x, GM_ADDR add2_y, GM_ADDR output0, GM_ADDR output1, GM_ADDR output2,
    GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    using OpDag = AdamApplyOneWithDecayOp::AdamApplyOneWithDecayCompute<DTYPE_INPUT0, float>::OpDag;
    BroadcastSch<schMode, OpDag> sch(tiling);
    sch.Process(
        input0, input1, input2, input3, input4, mul0_x, mul1_x, mul2_x, mul3_x, mul4_x, add2_y, output0, output1,
        output2);
}
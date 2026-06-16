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
 * \file lamb_next_right.cpp
 * \brief lamb_next_right.cpp
 */

#include "kernel_operator.h"
#include "arch35/lamb_next_right_dag.h"
#include "arch35/lamb_next_right_tiling_key.h"
#include "atvoss/broadcast/broadcast_sch.h"

using namespace AscendC;
using namespace Ops::Base;

template <uint64_t schMode>
__global__ __aicore__ void lamb_next_right(
    GM_ADDR input_square, GM_ADDR input_mul2, GM_ADDR mul2_x, GM_ADDR mul3_x, GM_ADDR truediv1_recip, GM_ADDR add2_y,
    GM_ADDR y1, GM_ADDR y2, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if constexpr (std::is_same<DTYPE_INPUT_SQUARE, half>::value) {
        using OpDag = LambNextRightOp::LambNextRightCompute<half, float>::OpDag;
        BroadcastSch<schMode, OpDag> sch(tiling);
        sch.Process(input_square, input_mul2, mul2_x, mul3_x, truediv1_recip, add2_y, y1, y2);
    } else {
        using OpDag = LambNextRightOp::LambNextRightCompute<float, float>::OpDag;
        BroadcastSch<schMode, OpDag> sch(tiling);
        sch.Process(input_square, input_mul2, mul2_x, mul3_x, truediv1_recip, add2_y, y1, y2);
    }
}

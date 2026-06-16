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
 * \file lamb_next_m_v_with_decay.cpp
 * \brief lamb_next_m_v_with_decay.cpp
 */

#include "kernel_operator.h"
#include "arch35/lamb_next_m_v_with_decay_dag.h"
#include "arch35/lamb_next_m_v_with_decay_tiling_key.h"
#include "atvoss/broadcast/broadcast_sch.h"

using namespace AscendC;
using namespace Ops::Base;

template <uint64_t schMode>
__global__ __aicore__ void lamb_next_mv_with_decay(
    GM_ADDR input_mul3, GM_ADDR input_mul2, GM_ADDR input_realdiv1, GM_ADDR input_mul1, GM_ADDR input_mul0,
    GM_ADDR input_realdiv0, GM_ADDR input_mul4, GM_ADDR mul0_x, GM_ADDR mul1_sub, GM_ADDR mul2_x, GM_ADDR mul3_sub1,
    GM_ADDR mul4_x, GM_ADDR add2_y, GM_ADDR y1, GM_ADDR y2, GM_ADDR y3, GM_ADDR y4, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if constexpr (std::is_same<DTYPE_INPUT_MUL3, half>::value) {
        using OpDag = LambNextMVWithDecayOp::LambNextMVWithDecayCompute<half, float>::OpDag;
        BroadcastSch<schMode, OpDag> sch(tiling);
        sch.Process(input_mul3, input_mul2, input_realdiv1, input_mul1, input_mul0, input_realdiv0, input_mul4,
                    mul0_x, mul1_sub, mul2_x, mul3_sub1, mul4_x, add2_y, y1, y2, y3, y4);
    } else {
        using OpDag = LambNextMVWithDecayOp::LambNextMVWithDecayCompute<float, float>::OpDag;
        BroadcastSch<schMode, OpDag> sch(tiling);
        sch.Process(input_mul3, input_mul2, input_realdiv1, input_mul1, input_mul0, input_realdiv0, input_mul4,
                    mul0_x, mul1_sub, mul2_x, mul3_sub1, mul4_x, add2_y, y1, y2, y3, y4);
    }
}

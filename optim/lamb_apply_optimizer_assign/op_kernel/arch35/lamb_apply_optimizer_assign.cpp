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
 * \file lamb_apply_optimizer_assign.cpp
 * \brief LambApplyOptimizerAssign arch35 (Ascend950) kernel entry
 *
 * 手写 regbase 实现, 不走 ATVOSS: 12 入 3 出在 DAGSch 的 32 buffer 预算上编不过,
 * 且 ATVOSS 的 Vec::Brc 未实现导致 UB 广播档不可用(铺不平尾轴 1->n 与低秩补维)。
 * 搬运/广播/分核骨架见 lamb_apply_common/op_kernel/arch35/lamb_brc_kernel.h。
 * TilingKey: fp32=100 / fp16=200。
 */

#include "kernel_operator.h"
#include "lamb_apply_optimizer_assign_vf.h"

// 宏展开处需要一个文件作用域可见的具体类型名, 不能用函数内的局部别名。
using LambApplyOptimizerAssignTilingType = LambBrcTilingData<12, 3>;

extern "C" __global__ __aicore__ void lamb_apply_optimizer_assign(
    GM_ADDR grad, GM_ADDR inputv, GM_ADDR inputm, GM_ADDR input4, GM_ADDR mul0_x, GM_ADDR mul1_x, GM_ADDR mul2_x,
    GM_ADDR mul3_x, GM_ADDR add2_y, GM_ADDR steps, GM_ADDR do_use_weight, GM_ADDR weight_decay_rate, GM_ADDR output0,
    GM_ADDR inputv_ref, GM_ADDR inputm_ref, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(LambApplyOptimizerAssignTilingType);
    GET_TILING_DATA_WITH_STRUCT(LambApplyOptimizerAssignTilingType, tilingData, tiling);
    GM_ADDR inAddr[12] = {grad,   inputv, inputm, input4, mul0_x,        mul1_x,
                          mul2_x, mul3_x, add2_y, steps,  do_use_weight, weight_decay_rate};
    GM_ADDR outAddr[3] = {output0, inputv_ref, inputm_ref};
    AscendC::TPipe pipe;
    if (TILING_KEY_IS(100)) {
        LambBrc::BrcElementwiseKernel<float, 12, 3, LambApplyOptimizerAssignOp::LambApplyOptimizerAssignVf> op;
        op.Init(inAddr, outAddr, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(200)) {
        LambBrc::BrcElementwiseKernel<half, 12, 3, LambApplyOptimizerAssignOp::LambApplyOptimizerAssignVf> op;
        op.Init(inAddr, outAddr, &tilingData, &pipe);
        op.Process();
    }
}

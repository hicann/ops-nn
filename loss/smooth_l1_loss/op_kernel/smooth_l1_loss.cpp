/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file smooth_l1_loss.cpp
 * \brief SmoothL1Loss kernel (element-wise, no reduction)
 *
 * 一芯片一算子一入口：arch35 的唯一 __global__ 入口。核函数参数顺序固定：
 *   输入 predict → label → 输出 loss → workspace → tiling（workspaceSize=0）。
 *
 * 输入 dtype 由编译系统自动注入的 DTYPE_PREDICT 宏指定，按 dtype 各编译一份 kernel。
 * KernelSmoothL1Loss<T> 按 T 参数化，单一模板实例覆盖全 3 dtype。
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "arch35/smooth_l1_loss.h"

using namespace AscendC;
using namespace SmoothL1Loss;

extern "C" __global__ __aicore__ void smooth_l1_loss(GM_ADDR predict, GM_ADDR label, GM_ADDR loss, GM_ADDR workspace,
                                                     GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SmoothL1LossTilingData);
    GET_TILING_DATA_WITH_STRUCT(SmoothL1LossTilingData, tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    SmoothL1Loss::KernelSmoothL1Loss<DTYPE_PREDICT> op;
    op.Init(predict, label, loss, &tilingData);
    op.Process();
}

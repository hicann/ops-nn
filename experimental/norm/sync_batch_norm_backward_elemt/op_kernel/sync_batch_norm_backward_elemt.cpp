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
 * \file sync_batch_norm_backward_elemt.cpp
 * \brief
 */

#include "sync_batch_norm_backward_elemt.h"

template <uint32_t schMode>
__global__ __aicore__ void sync_batch_norm_backward_elemt(GM_ADDR grad_output, GM_ADDR save_input, GM_ADDR mean,
                                                          GM_ADDR invstd, GM_ADDR weight, GM_ADDR mean_dy,
                                                          GM_ADDR mean_dy_xmu, GM_ADDR grad_input, GM_ADDR workspace,
                                                          GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(SyncBatchNormBackwardElemtTilingData);
    GET_TILING_DATA_WITH_STRUCT(SyncBatchNormBackwardElemtTilingData, tilingData, tiling);
    NsSyncBatchNormBackwardElemt::KernelSyncBatchNormBackwardElemt<DTYPE_GRAD_OUTPUT, DTYPE_MEAN>
        op; // 算子kernel实例获取
    op.Init(grad_output, save_input, mean, invstd, weight, mean_dy, mean_dy_xmu, grad_input, &tilingData);
    op.Process();
}

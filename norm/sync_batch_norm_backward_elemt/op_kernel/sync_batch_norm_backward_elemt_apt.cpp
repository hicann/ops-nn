/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file sync_batch_norm_backward_elemt.cpp
 * \brief
 */
#include "arch35/sync_batch_norm_backward_elemt_tilingdata.h"
#include "arch35/sync_batch_norm_backward_elemt_dag.h"
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "atvoss/elewise/elewise_sch.h"
#include "atvoss/util/dfx.h"
#include "../inc/platform.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void sync_batch_norm_backward_elemt(
    GM_ADDR grad_output, GM_ADDR save_input, GM_ADDR mean, GM_ADDR invstd, GM_ADDR weight, GM_ADDR mean_dy,
    GM_ADDR mean_dy_xmu, GM_ADDR grad_input, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SyncBatchNormBackwardElemtTilingData);
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    if (TILING_KEY_IS(0)) {
        ElementwiseSch<0UL, SyncBatchNormBackwardElemtDag<DTYPE_GRAD_OUTPUT, DTYPE_MEAN>::OpDag> sch(&(tilingData.baseTiling), &pipe);
        sch.Init(grad_output, save_input, mean, invstd, weight, mean_dy, mean_dy_xmu, grad_input);
        sch.Process();
    }
}
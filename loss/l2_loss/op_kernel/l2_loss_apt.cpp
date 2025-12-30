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
 * \file l2_loss.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "./arch35/l2_loss_dag.h"
#include "./arch35/l2_loss_tiling_key.h"
#include "atvoss/reduce/reduce_sch.h"

using namespace ReduceOpTmpl;
using namespace AscendC;

template <REDUCE_TPL_PARAM>
__global__ __aicore__ void l2_loss(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    if (g_coreType == AIC) {
        return;
    }
    if (workspace == nullptr) {
        return;
    }
    SetSysWorkspace(workspace);
    GM_ADDR userWS = GetUserWorkspace(workspace);
    if (userWS == nullptr) {
        return;
    }
    REGISTER_TILING_DEFAULT(ReduceOpTilingData);
    GET_TILING_DATA_WITH_STRUCT(ReduceOpTilingData, tilingData, tiling);
    TPipe pipe;
    using PromoteType = __reduceType::GetPromoteType<DTYPE_X>::T;
    using Op = ReduceSch<REDUCE_TPL_VALUE, L2Loss::L2LossDag<DTYPE_X, PromoteType>::OpDag>;
    Op op(&tilingData);
    op.Init(&pipe, x, y, userWS);
    op.Process();
}
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
 * \file sync_batch_norm_backward_reduce_apt.cpp
 * \brief
 */
#include "arch35/sync_batch_norm_backward_reduce_tilingdata.h"
#include "arch35/sync_batch_norm_backward_reduce_dag.h"
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "atvoss/elewise/elewise_sch.h"
#include "atvoss/util/dfx.h"
#include "../inc/platform.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void sync_batch_norm_backward_reduce(GM_ADDR sumDy, GM_ADDR sumDyDxPad, GM_ADDR mean, 
                                                                      GM_ADDR invertStd, GM_ADDR sumDyXmu, GM_ADDR y,
                                                                      GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SyncBatchNormBackwardReduceTilingData);
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    if (TILING_KEY_IS(0UL)) {
        ElementwiseSch<0UL, SyncBatchNormBackwardReduceDag<DTYPE_SUM_DY>::OpDag> sch(&(tilingData.baseTiling), &pipe);
        sch.Init(sumDy, sumDyDxPad, mean, invertStd, sumDyXmu, y);
        sch.Process();
    }
    return;
}
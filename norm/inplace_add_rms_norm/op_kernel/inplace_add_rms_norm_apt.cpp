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
 * \file inplace_add_rms_norm_apt.cpp
 * \brief
 */
#include "../add_rms_norm/arch35/add_rms_norm_regbase.h"
#include "../add_rms_norm/arch35/add_rms_norm_regbase_split_d.h"

using namespace AscendC;
using namespace AddRmsNorm;

extern "C" __global__ __aicore__ void inplace_add_rms_norm(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR y, GM_ADDR rstd, GM_ADDR x, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    if (TILING_KEY_IS(1000)) {
        GET_TILING_DATA_WITH_STRUCT(AddRMSNormRegbaseRFullLoadTilingData, tilingDataIn, tiling);
        KernelAddRmsNormRegBase<DTYPE_X1> op(&pipe);
        op.Init(x1, x2, gamma, y, rstd, x, &tilingDataIn);
        op.Process();
    } else if (TILING_KEY_IS(2000)) {
        GET_TILING_DATA_WITH_STRUCT(AddRMSNormRegbaseTilingData, tilingDataIn, tiling);
        KernelAddRmsNormRegBaseSplitD<DTYPE_X1> op(&pipe);
        op.Init(x1, x2, gamma, y, rstd, x, &tilingDataIn);
        op.Process();
    }
}
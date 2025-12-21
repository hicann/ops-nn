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
 * \file add_rms_norm_cast_apt.cpp
 * \brief
 */

#include "arch35/add_rms_norm_cast_regbase.h"
#include "arch35/add_rms_norm_cast_regbase_spilt_reduce.h"
#include "arch35/add_rms_norm_cast_regbase_single_n.h"

using namespace AscendC;
using namespace AddRmsNormCast;

extern "C" __global__ __aicore__ void add_rms_norm_cast(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR y1, GM_ADDR y2, GM_ADDR rstd, GM_ADDR x, GM_ADDR workspace,
    GM_ADDR tiling)
{
    TPipe pipe;

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    GET_TILING_DATA_WITH_STRUCT(AddRmsNormCastRegbaseTilingData, tilingDataIn, tiling);
    const AddRmsNormCastRegbaseTilingData* __restrict tilingData = &tilingDataIn;
    if (TILING_KEY_IS(100)) {
        KernelAddRmsNormCastRegBaseNormal<DTYPE_X1> op(&pipe);
        op.Init(x1, x2, gamma, y1, y2, rstd, x, workspace, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(101)) {
        KernelAddRmsNormCastRegBaseSpiltReduce<DTYPE_X1> op(&pipe);
        op.Init(x1, x2, gamma, y1, y2, rstd, x, workspace, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(102)) {
        KernelAddRmsNormCastRegBaseSingleN<DTYPE_X1> op(&pipe);
        op.Init(x1, x2, gamma, y1, y2, rstd, x, workspace, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(199)) {
        // Empty kernel
    }
#endif
}
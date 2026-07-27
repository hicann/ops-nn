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
 * \file sync_batch_norm_backward_reduce.cpp
 * \brief SyncBatchNormBackwardReduce kernel entry. schMode selects the compute dtype.
 */

#include "sync_batch_norm_backward_reduce.h"

#if defined(ASCENDC_CPU_DEBUG)
#include "securec.h"
#ifndef GET_TILING_DATA_WITH_STRUCT
#define GET_TILING_DATA_WITH_STRUCT(structType, name, tilingArg) \
    structType name;                                             \
    (void)memcpy_s(&(name), sizeof(structType), (tilingArg), sizeof(structType))
#endif
#else
#include "sync_batch_norm_backward_reduce_tiling_key.h"
#endif

template <uint32_t schMode>
__global__ __aicore__ void sync_batch_norm_backward_reduce(GM_ADDR sum_dy, GM_ADDR sum_dy_dx_pad, GM_ADDR mean,
                                                           GM_ADDR invert_std, GM_ADDR sum_dy_xmu, GM_ADDR y,
                                                           GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SyncBatchNormBackwardReduceTilingData);
    GET_TILING_DATA_WITH_STRUCT(SyncBatchNormBackwardReduceTilingData, tilingData, tiling);

    GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);

    if constexpr (schMode == SYNCBNBR_TPL_SCH_MODE_0) {
        NsSyncBatchNormBackwardReduce::Run<half>(sum_dy, sum_dy_dx_pad, mean, invert_std, sum_dy_xmu, y, usrWorkspace,
                                                 &tilingData);
    } else if constexpr (schMode == SYNCBNBR_TPL_SCH_MODE_1) {
        NsSyncBatchNormBackwardReduce::Run<float>(sum_dy, sum_dy_dx_pad, mean, invert_std, sum_dy_xmu, y, usrWorkspace,
                                                  &tilingData);
    }
#if __CCE_AICORE__ != 200
    else if constexpr (schMode == SYNCBNBR_TPL_SCH_MODE_2) {
        NsSyncBatchNormBackwardReduce::Run<bfloat16_t>(sum_dy, sum_dy_dx_pad, mean, invert_std, sum_dy_xmu, y,
                                                       usrWorkspace, &tilingData);
    }
#endif
}

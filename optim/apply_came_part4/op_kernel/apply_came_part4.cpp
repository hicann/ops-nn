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
 * \file apply_came_part4.cpp
 * \brief ApplyCamePart4 kernel entry (arch35), aligned to canndev ApplyCamePart4
 *
 * GE IR signature (canndev REG_OP(ApplyCamePart4)):
 *   Inputs (12): param, m, r, c, weight_decay, lr, beta3, sum_r(optional),
 *                sum_u_r, sum_u_c, sum_u_rc, global_shape(optional)
 *   Outputs (3): param, r, c
 *
 * Template parameters (matching apply_came_part4_tiling_key.h):
 *   - D_T_X: Data type of param/m/r/c, from ASCENDC_TPL_DATATYPE_DECL
 */

#include "arch35/apply_came_part4.h"

using namespace AscendC;

template <typename D_T_X>
__global__ __aicore__ void apply_came_part4(GM_ADDR paramIn, GM_ADDR m, GM_ADDR rIn, GM_ADDR cIn, GM_ADDR weightDecay,
                                            GM_ADDR lr, GM_ADDR beta3, GM_ADDR sumUR, GM_ADDR sumUC, GM_ADDR sumURC,
                                            GM_ADDR sumR, GM_ADDR globalShape, GM_ADDR paramOut, GM_ADDR rOut,
                                            GM_ADDR cOut, GM_ADDR workspace, GM_ADDR tiling)
{
    if (g_coreType == AIC) {
        return;
    }
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    REGISTER_TILING_DEFAULT(ApplyCamePart4TilingData);
    GET_TILING_DATA_WITH_STRUCT(ApplyCamePart4TilingData, tilingData, tiling);
    NsApplyCamePart4::ApplyCamePart4<D_T_X> op;
    op.Init(paramIn, m, rIn, cIn, weightDecay, lr, beta3, sumUR, sumUC, sumURC, sumR, globalShape, paramOut, rOut, cOut,
            workspace, &tilingData);
    op.Process();
}

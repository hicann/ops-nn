/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file swiglu_group_grad.cpp
 * \brief SwigluGroupGrad kernel entry (arch35, Ascend950/DAV_3510)
 */

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "arch35/swiglu_group_grad_tiling_key.h"
#include "arch35/swiglu_group_grad_regbase.h"
#ifndef __CCE_KT_TEST__
#include "arch35/swiglu_group_grad_simt.h"
#endif

using namespace AscendC;
using namespace SwigluGroupGradOps;

template <uint32_t SCHMODE = TPL_REGBASE_KERNEL, uint32_t HAS_CLAMP = 0, uint32_t IS_WEIGHT = 0,
          uint32_t IS_Y_ORIGIN = 0, uint32_t IS_GROUP_INDEX = 0>
__global__ __aicore__ void swiglu_group_grad(GM_ADDR grad_y, GM_ADDR x, GM_ADDR weight, GM_ADDR y_origin,
                                             GM_ADDR group_index, GM_ADDR grad_x, GM_ADDR grad_weight,
                                             GM_ADDR workspace, GM_ADDR tiling)
{
    AscendC::TPipe pipe;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(SwigluGroupGradTilingData);
#ifndef __CCE_KT_TEST__
    if constexpr (SCHMODE == TPL_SIMT_KERNEL) {
        GET_TILING_DATA_WITH_STRUCT(SwigluGroupGradSimtTilingData, tilingData, tiling);
        SwigluGroupGradSimt<DTYPE_GRAD_Y, HAS_CLAMP, IS_WEIGHT, IS_Y_ORIGIN, IS_GROUP_INDEX> op;
        op.Init(grad_y, x, weight, y_origin, group_index, grad_x, grad_weight, workspace, &tilingData);
        op.Process();
    } else
#endif
        if constexpr (SCHMODE == TPL_REGBASE_KERNEL) {
        GET_TILING_DATA_WITH_STRUCT(SwigluGroupGradTilingData, tilingData, tiling);
        SwigluGroupGradBase<DTYPE_GRAD_Y, HAS_CLAMP, IS_WEIGHT, IS_Y_ORIGIN, IS_GROUP_INDEX> op;
        op.Init(grad_y, x, weight, y_origin, group_index, grad_x, grad_weight, workspace, &tilingData);
        op.Process();
    }
}

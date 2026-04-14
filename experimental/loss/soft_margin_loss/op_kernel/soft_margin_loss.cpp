/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/**
 * \file soft_margin_loss_arch32.cpp
 * \brief SoftMarginLoss kernel entry point (arch32 architecture - Ascend910B)
 *
 * Dispatches to template instantiations:
 *   SoftMarginLossNone<float/half>   - elementwise output
 *   SoftMarginLossReduce<float/half> - scalar reduction output
 */

#include "soft_margin_loss.h"

template <uint32_t schMode>
__global__ __aicore__ void soft_margin_loss(GM_ADDR selfInput, GM_ADDR targetInput,
                                             GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SoftMarginLossTilingData);
    GET_TILING_DATA_WITH_STRUCT(SoftMarginLossTilingData, tilingData, tiling);

    if constexpr (schMode == SML_TPL_SCH_MODE_0) {
        NsSoftMarginLoss::SoftMarginLossNone<float> op;
        op.Init(selfInput, targetInput, output, &tilingData);
        op.Process();
    }
    if constexpr (schMode == SML_TPL_SCH_MODE_1) {
        NsSoftMarginLoss::SoftMarginLossReduce<float> op;
        op.Init(selfInput, targetInput, output, workspace, &tilingData);
        op.Process();
    }
    if constexpr (schMode == SML_TPL_SCH_MODE_2) {
        NsSoftMarginLoss::SoftMarginLossNone<half> op;
        op.Init(selfInput, targetInput, output, &tilingData);
        op.Process();
    }
    if constexpr (schMode == SML_TPL_SCH_MODE_3) {
        NsSoftMarginLoss::SoftMarginLossReduce<half> op;
        op.Init(selfInput, targetInput, output, workspace, &tilingData);
        op.Process();
    }
}

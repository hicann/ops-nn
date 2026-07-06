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
 * \file fake_quant_with_min_max_vars_gradient.cpp
 * \brief fake_quant_with_min_max_vars_gradient kernel entry (regbase / arch35)
 *
 * Key differences from args version:
 *  - 4 inputs (gradients, x, min, max) vs 2 inputs (gradients, x)
 *  - 3 outputs (backprops_wrt_x, backprop_wrt_min, backprop_wrt_max) vs 1 output (y)
 *  - No workspace: multi-core reduce via SetAtomicAdd direct to output GM
 */

#include "kernel_operator.h"
#include "fake_quant_with_min_max_vars_gradient_struct.h"
#include "fake_quant_with_min_max_vars_gradient_tilingdata.h"
#include "fake_quant_with_min_max_vars_gradient_regbase.h"

using namespace AscendC;
using namespace FakeQuantWithMinMaxVarsGradient;
using namespace FakeQuantWithMinMaxVarsGradientOp;

template <uint64_t schMode>
__global__ __aicore__ void fake_quant_with_min_max_vars_gradient(GM_ADDR gradients, GM_ADDR x, GM_ADDR min_gm,
                                                                 GM_ADDR max_gm, GM_ADDR backprops_wrt_x,
                                                                 GM_ADDR backprop_wrt_min, GM_ADDR backprop_wrt_max,
                                                                 GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(FakeQuantWithMinMaxVarsGradientTilingData);
    GET_TILING_DATA(tilingData, tiling);

    FakeQuantWithMinMaxVarsGradientRegbase<schMode> op(&tilingData);
    op.Init(gradients, x, min_gm, max_gm, backprops_wrt_x, backprop_wrt_min, backprop_wrt_max);
    op.Process();
}

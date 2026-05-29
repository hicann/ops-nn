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
 * \file fake_quant_with_min_max_args_gradient.cpp
 * \brief fake_quant_with_min_max_args_gradient kernel entry (regbase / arch35)
 *
 * Self-check (design §3.2 + §5):
 *  5)  grad * mask01 multiplicative gating (NaN grad on out-of-range -> NaN)  [regbase.h]
 *  6)  clamp via >=/<= + Select (NaN-safe), not Min/Max                       [regbase.h]
 *  11) ±0 sign-bit OR re-injection (Ascend Mul drops -0 sign)                 [regbase.h]
 *  Forward NaN passthrough (item 4) is NOT used here -- gradient must let
 *  Mul naturally produce 0 for "NaN x + finite grad" (TF behaviour).
 *  Host Nudge H1-H4 (re-divide / div zp / round-half-away / closed clip) live in tiling.cpp.
 */

#include "kernel_operator.h"
#include "fake_quant_with_min_max_args_gradient_struct.h"
#include "fake_quant_with_min_max_args_gradient_tilingdata.h"
#include "fake_quant_with_min_max_args_gradient_regbase.h"

using namespace AscendC;
using namespace FakeQuantWithMinMaxArgsGradient;
using namespace FakeQuantWithMinMaxArgsGradientOp;

template <uint64_t schMode>
__global__ __aicore__ void fake_quant_with_min_max_args_gradient(
    GM_ADDR gradients, GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(FakeQuantWithMinMaxArgsGradientTilingData);
    GET_TILING_DATA(tilingData, tiling);

    FakeQuantWithMinMaxArgsGradientRegbase<schMode> op(&tilingData);
    op.Init(gradients, x, y);
    op.Process();
}

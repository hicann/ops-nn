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
 * \file fake_quant_with_min_max_args.cpp
 * \brief fake_quant_with_min_max_args kernel entry (regbase / arch35)
 *
 * Self-check (design §5, 11 items):
 *  1) round = floor(scaled + roundBias), roundBias = 0.5 - quantZero    [regbase.h]
 *  2) scheme A: shifted=clamped+(-nudgedMin); scaled*scaleInv (no merge)[regbase.h]
 *  3) dequant y = q_offset * scale (no large-add)                       [regbase.h]
 *  4) NaN: x!=x mask + final Select(vX,vY,nanMask) passthrough          [regbase.h]
 *  5) N/A (gradient-only: backward mask)
 *  6) clamp via >/< + Select (NaN-safe), not Min/Max                    [regbase.h]
 *  7-10) host Nudge H1-H5 (scaleInv re-div / zp div / round / closed)   [tiling.cpp]
 *  11) N/A (gradient-only)
 */

#include "kernel_operator.h"
#include "fake_quant_with_min_max_args_struct.h"
#include "fake_quant_with_min_max_args_tilingdata.h"
#include "fake_quant_with_min_max_args_regbase.h"

using namespace AscendC;
using namespace FakeQuantWithMinMaxArgs;
using namespace FakeQuantWithMinMaxArgsOp;

template <uint64_t schMode>
__global__ __aicore__ void fake_quant_with_min_max_args(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(FakeQuantWithMinMaxArgsTilingData);
    GET_TILING_DATA(tilingData, tiling);

    FakeQuantWithMinMaxArgsRegbase<schMode> op(&tilingData);
    op.Init(x, y);
    op.Process();
}

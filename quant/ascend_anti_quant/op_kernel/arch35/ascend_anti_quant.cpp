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
 * \file ascend_anti_quant.cpp
 * \brief AscendAntiQuant kernel entry (pure regbase / MicroAPI mode).
 *
 *   y = TOut((cast<float>(x) + offset) * effective_scale)
 *
 *   Modeled after ascend_anti_quant_v2/op_kernel/ascend_anti_quant_v2_apt.cpp.
 *   All input dtypes (int8 / hifloat8 / fp8_e5m2 / fp8_e4m3fn) dispatch into a
 *   single self-written regbase kernel `AscendAntiQuantRegbase<DTYPE_X,
 *   DTYPE_Y>`. No atvoss DAG / ElementwiseSch / dfx headers are involved.
 *
 *   The kernel applies the exact formula:
 *     - sqrtModeKey == 0 : y = TOut((x + offset) * scale)
 *     - sqrtModeKey != 0 : y = TOut((x + offset) * scale * scale)
 *   The second `* scale` is performed inside the kernel (NOT collapsed on host)
 *   so the float32 intermediate of `(x + offset) * scale` is preserved, avoiding
 *   the ~1ULP precision loss caused by host-side `scale*scale` rounding.
 *   `sqrtModeKey` is encoded as the tiling-key template parameter so the branch
 *   is resolved at compile time via `if constexpr` inside the kernel.
 */
#include "kernel_operator.h"
#include "ascend_anti_quant_struct.h"
#include "ascend_anti_quant_tilingdata.h"
#include "ascend_anti_quant_regbase.h"

using namespace AscendC;
using namespace AscendAntiQuantOp;

template <uint64_t sqrtModeKey>
__global__ __aicore__ void ascend_anti_quant(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(AscendAntiQuantTilingData);
    GET_TILING_DATA(tilingData, tiling);

    const float scaleV = static_cast<float>(tilingData.scale);
    const float offsetV = static_cast<float>(tilingData.offset);

    AscendAntiQuantRegbase<DTYPE_X, DTYPE_Y, (sqrtModeKey != 0)> op(&tilingData, scaleV, offsetV);
    op.Init(x, y);
    op.Process();
}

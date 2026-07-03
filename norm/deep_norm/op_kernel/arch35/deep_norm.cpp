/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/*!
 * \file deep_norm_apt.cpp
 * \brief DeepNorm kernel entry (arch35 / Ascend950).
 *
 * The framework defines DTYPE_X for the x/gx/beta/gamma/y dtype and compiles one
 * binary per dtype variant; the kernel is templated on that macro.
 */

#include "deep_norm.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void deep_norm(
    GM_ADDR x, GM_ADDR gx, GM_ADDR beta, GM_ADDR gamma, GM_ADDR mean, GM_ADDR rstd, GM_ADDR y,
    GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(DeepNormTilingData);
    GET_TILING_DATA_WITH_STRUCT(DeepNormTilingData, tilingDataIn, tiling);
    const DeepNormTilingData* __restrict tilingData = &tilingDataIn;

    if (TILING_KEY_IS(0)) {
        NsDeepNorm::DeepNorm<DTYPE_X, 0> op;
        op.Init(x, gx, beta, gamma, mean, rstd, y, tilingData);
        op.Process();
    }
}

/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * NOTE: Portions of this code were AI-generated and have been technically reviewed for functional accuracy.
 */

/*!
 * \file l2_normalize_grad.cpp
 * \brief L2NormalizeGrad kernel entry (arch35 / Ascend950).
 *
 * The framework defines DTYPE_X for the x/y/dy/dx dtype and compiles one binary per dtype
 * variant (fp16 / fp32); the templates are specialized on that macro. The runtime TilingKey only
 * distinguishes load form / reduction shape / empty:
 *   8000 empty | 7000 full-load (inner==1, D fits UB) | 7010 split-D (inner==1, D large)
 *   7020 strided (inner>1)
 */
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "arch35/l2_normalize_grad_regbase_dx_full_load.h"
#include "arch35/l2_normalize_grad_regbase_dx_split_d.h"
#include "arch35/l2_normalize_grad_regbase_dx_strided.h"
#include "arch35/l2_normalize_grad_empty.h"

extern "C" __global__ __aicore__ void l2_normalize_grad(GM_ADDR x, GM_ADDR y, GM_ADDR dy, GM_ADDR dx, GM_ADDR workspace,
                                                        GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(L2NormalizeGradTilingData);
    GET_TILING_DATA_WITH_STRUCT(L2NormalizeGradTilingData, tilingDataIn, tiling);
    const L2NormalizeGradTilingData* __restrict tilingData = &tilingDataIn;
    AscendC::TPipe pipe;

    if (TILING_KEY_IS(8000)) {
        L2NormalizeGrad::L2NormalizeGradEmpty op(&pipe, tilingData);
        op.Init(dx);
        op.Process();
    } else if (TILING_KEY_IS(7000)) {
        L2NormalizeGrad::RegbaseDxFullLoad<DTYPE_X> op(&pipe, tilingData);
        op.Init(x, y, dy, dx);
        op.Process();
    } else if (TILING_KEY_IS(7010)) {
        L2NormalizeGrad::RegbaseDxSplitD<DTYPE_X> op(&pipe, tilingData);
        op.Init(x, y, dy, dx);
        op.Process();
    } else if (TILING_KEY_IS(7020)) {
        L2NormalizeGrad::RegbaseDxStrided<DTYPE_X> op(&pipe, tilingData);
        op.Init(x, y, dy, dx);
        op.Process();
    }
}

/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "kernel_operator.h"
#include "global_lp_pool_tiling_data.h"
#include "global_lp_pool_tiling_key.h"
#include "global_lp_pool_kernel.h"

using namespace AscendC;

// ═══════════════════════════════════════════════════════════════════════════
// Templated kernel entry point — framework dispatches based on tiling key
// (ASCENDC_TPL_SEL in global_lp_pool_tiling_key.h).
// DTYPE_X is auto-injected per dtype by the framework (not in tiling key).
// ═══════════════════════════════════════════════════════════════════════════
template <bool isEmptyTensor, bool isTailR>
__global__ __aicore__ void global_lp_pool(GM_ADDR input_x, GM_ADDR output_y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(GlobalLpPoolTilingData);
    GET_TILING_DATA_WITH_STRUCT(GlobalLpPoolTilingData, tilingData, tiling);

    GlobalLpPoolKernel<DTYPE_X, isEmptyTensor, isTailR> op;
    op.Init(input_x, output_y, workspace, tiling, &tilingData);
    op.Process();
}

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
 * \file scatter_mul.cpp
 * \brief ScatterMul kernel entry: var[indices[m]] = *= [indices[m]], updates[m]) (in place).
 */
#include "kernel_operator.h"
#include "../scatter_reduce_common/arch35/scatter_reduce_common_simt.h"

using namespace AscendC;
using namespace ScatterReduceCommon;

extern "C" __global__ __aicore__ void scatter_mul(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR y,
                                                  GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(ScatterReduceSimtTilingData);
    GET_TILING_DATA_WITH_STRUCT(ScatterReduceSimtTilingData, tilingData, tiling);
    ScatterReduceSimt<DTYPE_VAR, DTYPE_INDICES, int64_t, MODE_MUL> op(tilingData, pipe);
    op.Init(var, indices, updates, workspace);
    op.Process();
}

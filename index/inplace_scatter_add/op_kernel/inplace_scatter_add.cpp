/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file inplace_scatter_add.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "inplace_scatter_add_big_tail.h"
#include "inplace_scatter_add_small_tail.h"

extern "C" __global__ __aicore__ void inplace_scatter_add(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR var_ref, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    AscendC::TPipe tpipe;

#if (defined(DTYPE_VAR) && defined(DTYPE_INDICES))
    if (TILING_KEY_IS(0)) {
        InplaceScatterAddNS::SmallTailOp<DTYPE_VAR, DTYPE_INDICES> op;
        op.Init(var, indices, updates, &tiling_data, &tpipe);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        InplaceScatterAddNS::BigTailOp<DTYPE_VAR, DTYPE_INDICES> op;
        op.Init(var, indices, updates, &tiling_data, &tpipe);
        op.Process();
    }
#endif
}
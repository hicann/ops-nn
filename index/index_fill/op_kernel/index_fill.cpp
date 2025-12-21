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
 * \file index_fill.cpp
 * \brief
 */
#include "index_fill_front_q.h"
#include "index_fill_front_p.h"
#include "index_fill_front_n.h"
#include "index_fill_tail_n.h"
#include "index_fill_tail_p.h"
using namespace IndexFillNS;

extern "C" __global__ __aicore__ void index_fill(GM_ADDR x, GM_ADDR indices, GM_ADDR value, GM_ADDR y,
                                                 GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);
    AscendC::TPipe tpipe;

#if (defined(DTYPE_X))
    if (TILING_KEY_IS(0)) {
        index_fill_front_q<DTYPE_X, DTYPE_INDICES>(x, indices, value, y, workspace, tiling, &tilingData, &tpipe);
    } else if (TILING_KEY_IS(1)) {
        index_fill_front_n<DTYPE_X, DTYPE_INDICES>(x, indices, value, y, workspace, tiling, &tilingData, &tpipe);
    } else if (TILING_KEY_IS(2)) {
        index_fill_front_p<DTYPE_X, DTYPE_INDICES>(x, indices, value, y, workspace, tiling, &tilingData, &tpipe);
    } else if (TILING_KEY_IS(3)) {
        index_fill_tail_n<DTYPE_X, DTYPE_INDICES>(x, indices, value, y, workspace, tiling, &tilingData, &tpipe);
    } else if (TILING_KEY_IS(4)) {
        index_fill_tail_p<DTYPE_X, DTYPE_INDICES>(x, indices, value, y, workspace, tiling, &tilingData, &tpipe);
    }
#endif
}
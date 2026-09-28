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
 * \file wts_arq.cpp
 * \brief WtsARQ kernel entry for arch35 (Ascend 950).
 *
 * Template parameter RANK (4/8) is dispatched at compile time by the
 * ASCENDC_TPL template argument mechanism (wts_arq_tiling_key.h); dtype is
 * derived from binary.json via the DTYPE_W macro.
 */
#include "kernel_operator.h"
#include "arch35/wts_arq_kernel.h"

template <uint32_t RANK>
__global__ __aicore__ void wts_arq(GM_ADDR w, GM_ADDR w_min, GM_ADDR w_max, GM_ADDR y, GM_ADDR workspace,
                                   GM_ADDR tiling)
{
    REGISTER_NONE_TILING;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    if constexpr (RANK == WTS_ARQ_RANK_4) {
        GET_TILING_DATA_WITH_STRUCT(WtsArqTilingData<WTS_ARQ_RANK_4>, tilingData, tiling);
        WtsArq::WtsArqKernel<DTYPE_W, WTS_ARQ_RANK_4> op;
        op.Init(w, w_min, w_max, y, &tilingData);
        op.Process();
    } else {
        GET_TILING_DATA_WITH_STRUCT(WtsArqTilingData<WTS_ARQ_RANK_8>, tilingData, tiling);
        WtsArq::WtsArqKernel<DTYPE_W, WTS_ARQ_RANK_8> op;
        op.Init(w, w_min, w_max, y, &tilingData);
        op.Process();
    }
}

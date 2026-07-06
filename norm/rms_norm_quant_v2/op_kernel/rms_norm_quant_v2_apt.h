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
 * \file rms_norm_quant_v2_apt.h
 * \brief
 */
#ifndef RMS_NORM_QUANT_V2_APT_H
#define RMS_NORM_QUANT_V2_APT_H

#include "arch35/rms_norm_quant_v2_regbase_recompute.h"
#include "arch35/rms_norm_quant_v2_regbase_full_load.h"
#include "arch35/rms_norm_quant_v2_tiling_key.h"
using namespace AscendC;
using namespace RmsNormQuantV2;

#ifndef DTYPE_ZERO_POINTS2
#define DTYPE_ZERO_POINTS2 float
#endif

#ifndef DTYPE_ZERO_POINTS1
#define DTYPE_ZERO_POINTS1 DTYPE_ZERO_POINTS2
#endif

REGISTER_TILING_DEFAULT(RmsNormQuantV2RegbaseFullLoadTilingData);

template <int8_t COMPUTE_MODE>
__aicore__ inline void rms_norm_quant_v2_impl(GM_ADDR x, GM_ADDR gamma, GM_ADDR scales1, GM_ADDR scales2,
                                              GM_ADDR zero_points1, GM_ADDR zero_points2, GM_ADDR beta, GM_ADDR y1,
                                              GM_ADDR y2, GM_ADDR rstd, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    if constexpr (COMPUTE_MODE == COMPUTE_MODE_FULL_LOAD) {
        GET_TILING_DATA_WITH_STRUCT(RmsNormQuantV2RegbaseFullLoadTilingData, tilingData, tiling);
        RmsNormQuantV2RegbaseFullLoad<DTYPE_X, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1> op(&pipe);
        op.Init(x, gamma, scales1, scales2, zero_points1, zero_points2, beta, y1, y2, rstd, &tilingData);
        op.Process();
    } else if constexpr (COMPUTE_MODE == COMPUTE_MODE_RECOMPUTE) {
        GET_TILING_DATA_WITH_STRUCT(RmsNormQuantV2RegbaseRecomputeTilingData, tilingData, tiling);
        RmsNormQuantV2RegbaseRecompute<DTYPE_X, DTYPE_Y1, DTYPE_SCALES1, DTYPE_ZERO_POINTS1> op(&pipe);
        op.Init(x, gamma, scales1, scales2, zero_points1, zero_points2, beta, y1, y2, rstd, &tilingData);
        op.Process();
    }
}

#endif // RMS_NORM_QUANT_V2_APT_H

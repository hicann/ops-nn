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
 * \file rms_norm_dynamic_mx_quant_apt.cpp
 * \brief rms_norm_dynamic_mx_quant kernel entry
 */

#include "arch35/rms_norm_dynamic_mx_quant_tiling_data.h"
#include "arch35/rms_norm_dynamic_mx_quant_full_load.h"
#include "arch35/rms_norm_dynamic_mx_quant_reduce_empty.h"
#include "arch35/rms_norm_dynamic_mx_quant_recompute.h"
#include "arch35/rms_norm_dynamic_mx_quant_tiling_key.h"

using namespace AscendC;
using namespace RmsNormDynamicMxQuantNs;

#define FLOAT_OVERFLOW_MODE_CTRL 60

REGISTER_TILING_DEFAULT(RmsNormDynamicMxQuantFullLoadTilingData);

template <int8_t COMPUTE_MODE, int8_t OPTIMIZE_MODE>
__global__ __aicore__ void rms_norm_dynamic_mx_quant(
    GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mxscale, GM_ADDR rstd, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

#if (__NPU_ARCH__ == 3510)
    int64_t oriOverflowMode = AscendC::GetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>();
#endif

    constexpr bool isOptimize = (OPTIMIZE_MODE == OPTIMIZE_MODE_OPTIMIZE);

    if constexpr (COMPUTE_MODE == COMPUTE_MODE_REDUCE_EMPTY) {
        GET_TILING_DATA_WITH_STRUCT(RmsNormDynamicMxQuantReduceEmptyTilingData, tilingDataIn, tiling);
        RmsNormDynamicMxQuantReduceEmpty op(&tilingDataIn);
        op.Init(rstd);
        op.Process();
    } else if constexpr (COMPUTE_MODE == COMPUTE_MODE_FULL_LOAD) {
        GET_TILING_DATA_WITH_STRUCT(RmsNormDynamicMxQuantFullLoadTilingData, tilingDataIn, tiling);
        const RmsNormDynamicMxQuantFullLoadTilingData* __restrict tilingData = &tilingDataIn;
        TPipe pipe;
        RmsNormDynamicMxQuantFullLoad<DTYPE_X, DTYPE_GAMMA, DTYPE_Y, isOptimize> op(&pipe);
        op.Init(x, gamma, beta, y, mxscale, rstd, tilingData);
        op.Process();
    } else if constexpr (COMPUTE_MODE == COMPUTE_MODE_RECOMPUTE) {
        GET_TILING_DATA_WITH_STRUCT(RmsNormDynamicMxQuantRecomputeTilingData, tilingDataIn, tiling);
        const RmsNormDynamicMxQuantRecomputeTilingData* __restrict tilingData = &tilingDataIn;
        TPipe pipe;
        RmsNormDynamicMxQuantRecompute<DTYPE_X, DTYPE_GAMMA, DTYPE_Y, isOptimize> op(&pipe);
        op.Init(x, gamma, beta, y, mxscale, rstd, tilingData);
        op.Process();
    }

#if (__NPU_ARCH__ == 3510)
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(oriOverflowMode);
#endif
}
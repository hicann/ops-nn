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
 * \file add_rms_norm_dynamic_mx_quant_apt.cpp
 * \brief
 */
#include "arch35/add_rms_norm_dynamic_mx_quant_r_full_load.h"
#include "arch35/add_rms_norm_dynamic_mx_quant_split_r.h"
#include "arch35/add_rms_norm_dynamic_mx_quant_reduce_empty.h"
#include "arch35/add_rms_norm_dynamic_mx_quant_tiling_data.h"
#include "arch35/add_rms_norm_dynamic_mx_quant_tiling_key.h"

using namespace AscendC;
using namespace AddRmsNormDynamicMxQuant;

REGISTER_TILING_DEFAULT(AddRmsNormDynamicMxQuantTilingData);

template <int8_t COMPUTE_MODE, int8_t Y_DATA_TYPE, int8_t HAS_X3>
__global__ __aicore__ void add_rms_norm_dynamic_mx_quant(GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR beta,
                                                         GM_ADDR x3, GM_ADDR y, GM_ADDR x, GM_ADDR mxscale,
                                                         GM_ADDR rstd, GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }

    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;

#if (__NPU_ARCH__ == 3510)
    int64_t oriOverflowMode = AscendC::GetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>();
#endif

    if constexpr (COMPUTE_MODE == COMPUTE_MODE_REDUCE_EMPTY) {
        GET_TILING_DATA_WITH_STRUCT(AddRmsNormDynamicMxQuantReduceEmptyTilingData, tilingDataIn, tiling);
        AddRmsNormDynamicMxQuantReduceEmpty op(&tilingDataIn);
        op.Init(rstd);
        op.Process();
    } else if constexpr (COMPUTE_MODE == COMPUTE_MODE_FULL_LOAD) {
        GET_TILING_DATA_WITH_STRUCT(AddRmsNormDynamicMxQuantTilingData, tilingDataIn, tiling);
        AddRmsNormDynamicMxQuantRFullLoad<DTYPE_X1, DTYPE_GAMMA, DTYPE_Y, HAS_X3 != 0> op(&pipe);
        op.Init(x1, x2, gamma, beta, x3, y, x, mxscale, workspace, rstd, &tilingDataIn);
        op.Process();
    } else if constexpr (COMPUTE_MODE == COMPUTE_MODE_SPLIT_R) {
        GET_TILING_DATA_WITH_STRUCT(AddRmsNormDynamicMxQuantSplitRTilingData, tilingDataIn, tiling);
        AddRmsNormDynamicMxQuantSplitR<DTYPE_X1, DTYPE_GAMMA, DTYPE_Y, HAS_X3 != 0> op(&pipe);
        op.Init(x1, x2, gamma, beta, x3, y, x, mxscale, workspace, rstd, &tilingDataIn);
        op.Process();
    }

#if (__NPU_ARCH__ == 3510)
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(oriOverflowMode);
#endif
}

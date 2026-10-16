/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!\\file add_rms_norm_hi_f4_quant_apt.cpp
 * \\brief
 */
#include "arch92/add_rms_norm_hi_f4_quant_hif4_r_full_load.h"
#include "arch92/add_rms_norm_hi_f4_quant_hif4_split_d.h"
#include "arch92/add_rms_norm_hi_f4_quant_reduce_empty.h"
#include "arch92/add_rms_norm_hi_f4_quant_tiling_data.h"

using namespace AscendC;
using namespace AddRmsNormHiF4Quant;

#define TILING_KEY_HIF4_R_FULL_LOAD 101
#define TILING_KEY_HIF4_SPLIT_D 201
#define TILING_KEY_REDUCE_EMPTY 301

#if defined(__DAV_VEC__)
extern "C" __global__ __aicore__ void add_rms_norm_hi_f4_quant(GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR beta,
                                                               GM_ADDR y, GM_ADDR x, GM_ADDR scale, GM_ADDR rstd,
                                                               GM_ADDR workspace, GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }

    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(AddRmsNormHiF4QuantTilingData);
    TPipe pipe;

#if (__NPU_ARCH__ == 3510 || __NPU_ARCH__ == 9201 || __CCE_AICORE__ == 920)
    int64_t oriOverflowMode = AscendC::GetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>();
#endif

    if (TILING_KEY_IS(TILING_KEY_REDUCE_EMPTY)) {
        GET_TILING_DATA_WITH_STRUCT(AddRmsNormHiF4QuantReduceEmptyTilingData, tilingDataIn, tiling);
        AddRmsNormHiF4QuantReduceEmpty op(&tilingDataIn);
        op.Init(rstd);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_HIF4_R_FULL_LOAD)) {
        GET_TILING_DATA_WITH_STRUCT(AddRmsNormHiF4QuantTilingData, tilingDataIn, tiling);
        AddRmsNormHiF4QuantHif4RFullLoad<DTYPE_X1, DTYPE_GAMMA> op(&pipe);
        op.Init(x1, x2, gamma, beta, y, x, scale, rstd, workspace, &tilingDataIn);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_HIF4_SPLIT_D)) {
        GET_TILING_DATA_WITH_STRUCT(AddRmsNormHiF4QuantSplitDTilingData, tilingDataIn, tiling);
        AddRmsNormHiF4QuantHif4SplitD<DTYPE_X1, DTYPE_GAMMA> op(&pipe);
        op.Init(x1, x2, gamma, beta, y, x, scale, rstd, workspace, &tilingDataIn);
        op.Process();
    }

#if (__NPU_ARCH__ == 3510 || __NPU_ARCH__ == 9201 || __CCE_AICORE__ == 920)
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(oriOverflowMode);
#endif
}
#endif

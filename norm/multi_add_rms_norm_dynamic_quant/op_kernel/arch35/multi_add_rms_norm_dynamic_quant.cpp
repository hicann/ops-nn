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
 * \file multi_add_rms_norm_dynamic_quant.cpp
 * \brief
 */
#include "multi_add_rms_norm_dynamic_quant_regbase.h"
#include "multi_add_rms_norm_dynamic_quant_regbase_perf.h"
#include "multi_add_rms_norm_dynamic_quant_regbase_split_reduce.h"
#include "multi_add_rms_norm_dynamic_quant_regbase_single_row.h"

using namespace MultiAddRmsNormDynamicQuant;
using namespace AscendC;

#define TILING_KEY_UNRUN 199

// 相对参照 add_rms_norm_dynamic_quant:去 beta 输入、增 y 输出(RmsNorm 结果),x1 为 TensorList。
#define INIT_AND_PROCESS_WORKSPACE                                                                                 \
    do {                                                                                                           \
        op.Init(x1, x2, gamma, smooathScale1, smooathScale2, y1, y2, x, y, scale1, scale2, workspace, tilingData); \
        op.Process();                                                                                              \
    } while (0)

#define INIT_AND_PROCESS                                                                                \
    do {                                                                                                \
        op.Init(x1, x2, gamma, smooathScale1, smooathScale2, y1, y2, x, y, scale1, scale2, tilingData); \
        op.Process();                                                                                   \
    } while (0)

extern "C" __global__ __aicore__ void multi_add_rms_norm_dynamic_quant(GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma,
                                                                       GM_ADDR smooathScale1, GM_ADDR smooathScale2,
                                                                       GM_ADDR y1, GM_ADDR y2, GM_ADDR x, GM_ADDR y,
                                                                       GM_ADDR scale1, GM_ADDR scale2,
                                                                       GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    if (TILING_KEY_IS(TILING_KEY_UNRUN)) {
        // Do nothing
    } else {
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
        GET_TILING_DATA_WITH_STRUCT(MultiAddRmsNormDynamicQuantRegbaseTilingData, tilingDataIn, tiling);
        const MultiAddRmsNormDynamicQuantRegbaseTilingData* __restrict tilingData = &tilingDataIn;
        if (TILING_KEY_IS(100)) {
            KernelMultiAddRmsNormDynamicQuantRegbasePerf<DTYPE_X1, DTYPE_Y1> op(&pipe);
            INIT_AND_PROCESS;
        } else if (TILING_KEY_IS(101)) {
            KernelMultiAddRmsNormDynamicQuantRegbase<DTYPE_X1, DTYPE_Y1> op(&pipe);
            INIT_AND_PROCESS;
        } else if (TILING_KEY_IS(102)) {
            KernelMultiAddRmsNormDynamicQuantRegbaseSingleRow<DTYPE_X1, DTYPE_Y1> op(&pipe);
            INIT_AND_PROCESS;
        } else if (TILING_KEY_IS(103)) {
            KernelMultiAddRmsNormDynamicQuantRegbaseSpiltReduce<DTYPE_X1, DTYPE_Y1> op(&pipe);
            INIT_AND_PROCESS_WORKSPACE;
        }
    }
}

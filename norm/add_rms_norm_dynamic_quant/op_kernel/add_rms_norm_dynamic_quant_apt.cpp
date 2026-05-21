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
 * \file add_rms_norm_dynamic_quant_apt.cpp
 * \brief
 */
#include "arch35/add_rms_norm_dynamic_quant_regbase.h"
#include "arch35/add_rms_norm_dynamic_quant_regbase_perf.h"
#include "arch35/add_rms_norm_dynamic_quant_regbase_split_reduce.h"
#include "arch35/add_rms_norm_dynamic_quant_empty.h"

using namespace AddRmsNormDynamicQuant;
using namespace AscendC;

#define TILING_KEY_UNRUN 199

#define INIT_AND_PROCESS_WORKSPACE                                                                                    \
    do {                                                                                                              \
        op.Init(x1, x2, gamma, smooathScale1, smooathScale2, beta, y1, y2, x, scale1, scale2, workspace, tilingData); \
        op.Process();                                                                                                 \
    } while (0)

#define INIT_AND_PROCESS                                                                             \
    do {                                                                                             \
        op.Init(x1, x2, gamma, smooathScale1, smooathScale2, beta, y1, y2, x, scale1, scale2, tilingData); \
        op.Process();                                                                                \
    } while (0)

#define INIT_AND_PROCESS_EMPTY   \
    do {                         \
        op.Init(scale1, scale2); \
        op.Process();            \
    } while (0)

extern "C" __global__ __aicore__ void add_rms_norm_dynamic_quant(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR smooathScale1, GM_ADDR smooathScale2, GM_ADDR beta, GM_ADDR y1,
    GM_ADDR y2, GM_ADDR x, GM_ADDR scale1, GM_ADDR scale2, GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    if (TILING_KEY_IS(500)) {
        GET_TILING_DATA_WITH_STRUCT(AddRmsNormDynamicQuantEmptyTilingData, tilingDataIn, tiling);
        const AddRmsNormDynamicQuantEmptyTilingData* __restrict tilingData = &tilingDataIn;
        KernelAddRmsNormDynamicQuantEmpty<2> op(&pipe, tilingData);
        INIT_AND_PROCESS_EMPTY;
    } else if (TILING_KEY_IS(TILING_KEY_UNRUN)) {
        // Do nothing
    } else {
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
        GET_TILING_DATA_WITH_STRUCT(AddRmsNormDynamicQuantRegbaseTilingData, tilingDataIn, tiling);
        const AddRmsNormDynamicQuantRegbaseTilingData* __restrict tilingData = &tilingDataIn;
        if (TILING_KEY_IS(100)) {
            KernelAddRmsNormDynamicQuantRegbase<DTYPE_X1, DTYPE_Y1> op(&pipe);
            INIT_AND_PROCESS;
        } else if (TILING_KEY_IS(101)) {
            KernelAddRmsNormDynamicQuantRegbaseSpiltReduce<DTYPE_X1, DTYPE_Y1> op(&pipe);
            INIT_AND_PROCESS_WORKSPACE;
        } else if (TILING_KEY_IS(102)) {
            KernelAddRmsNormDynamicQuantRegbasePerf<DTYPE_X1, DTYPE_Y1> op(&pipe);
            INIT_AND_PROCESS;
        }
    }
}

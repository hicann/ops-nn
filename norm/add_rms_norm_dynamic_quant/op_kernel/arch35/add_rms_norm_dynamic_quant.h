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
 * \file add_rms_norm_dynamic_quant.h
 * \brief arch35 kernel implementation header for AddRmsNormDynamicQuant.
 */
#include "add_rms_norm_dynamic_quant_regbase.h"
#include "add_rms_norm_dynamic_quant_regbase_perf.h"
#include "add_rms_norm_dynamic_quant_regbase_split_reduce.h"
#include "add_rms_norm_dynamic_quant_regbase_single_row.h"
#include "add_rms_norm_dynamic_quant_empty.h"
#include "add_rms_norm_dynamic_quant_tiling_key.h"

using namespace AddRmsNormDynamicQuant;
using namespace AscendC;

#define INIT_AND_PROCESS_WORKSPACE                                                                             \
    do {                                                                                                       \
        op.Init(x1, x2, gamma, smoothScale1, smoothScale2, beta, y1, y2, y3, y4, x, scale1, scale2, workspace, \
                tilingData);                                                                                   \
        op.Process();                                                                                          \
    } while (0)

#define INIT_AND_PROCESS                                                                                         \
    do {                                                                                                         \
        op.Init(x1, x2, gamma, smoothScale1, smoothScale2, beta, y1, y2, y3, y4, x, scale1, scale2, tilingData); \
        op.Process();                                                                                            \
    } while (0)

REGISTER_TILING_DEFAULT(AddRmsNormDynamicQuantRegbaseTilingData);

template <int8_t COMPUTE_MODE, bool Y3_MODE, bool Y4_MODE>
__aicore__ void add_rms_norm_dynamic_quant_impl(GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR smoothScale1,
                                                GM_ADDR smoothScale2, GM_ADDR beta, GM_ADDR y1, GM_ADDR y2, GM_ADDR y3,
                                                GM_ADDR y4, GM_ADDR x, GM_ADDR scale1, GM_ADDR scale2,
                                                GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    if constexpr (COMPUTE_MODE == COMPUTE_MODE_REDUCE_EMPTY) {
        GET_TILING_DATA_WITH_STRUCT(AddRmsNormDynamicQuantEmptyTilingData, tilingDataIn, tiling);
        const AddRmsNormDynamicQuantEmptyTilingData* __restrict tilingData = &tilingDataIn;
        KernelAddRmsNormDynamicQuantEmpty<2> op(&pipe, tilingData);
        op.Init(scale1, scale2);
        op.Process();
    } else {
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
        GET_TILING_DATA_WITH_STRUCT(AddRmsNormDynamicQuantRegbaseTilingData, tilingDataIn, tiling);
        const AddRmsNormDynamicQuantRegbaseTilingData* __restrict tilingData = &tilingDataIn;
        if (tilingData->numM == 0) {
            return;
        }
        if constexpr (COMPUTE_MODE == COMPUTE_MODE_PERF) {
            KernelAddRmsNormDynamicQuantRegbasePerf<DTYPE_X1, DTYPE_Y1, Y3_MODE, Y4_MODE> op(&pipe);
            INIT_AND_PROCESS;
        } else if constexpr (COMPUTE_MODE == COMPUTE_MODE_NORMAL) {
            KernelAddRmsNormDynamicQuantRegbase<DTYPE_X1, DTYPE_Y1, Y3_MODE, Y4_MODE> op(&pipe);
            INIT_AND_PROCESS;
        } else if constexpr (COMPUTE_MODE == COMPUTE_MODE_SINGLE_ROW) {
            KernelAddRmsNormDynamicQuantRegbaseSingleRow<DTYPE_X1, DTYPE_Y1, Y3_MODE, Y4_MODE> op(&pipe);
            INIT_AND_PROCESS;
        } else if constexpr (COMPUTE_MODE == COMPUTE_MODE_SPLIT) {
            KernelAddRmsNormDynamicQuantRegbaseSpiltReduce<DTYPE_X1, DTYPE_Y1, Y3_MODE, Y4_MODE> op(&pipe);
            INIT_AND_PROCESS_WORKSPACE;
        }
    }
}

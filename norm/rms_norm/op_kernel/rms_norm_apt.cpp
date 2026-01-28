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
 * \file rms_norm_apt.cpp
 * \brief rmsnorm kernel file
 */
#include "arch35/rms_norm_regbase_perf.h"
#include "arch35/rms_norm_regbase_split_d.h"
using namespace AscendC;
using namespace RmsNorm;

#define RMSNORM_REGBASE_NORMAL 5000
#define RMSNORM_REGBASE_SPLIT 2001

#define GENERAL_OP_IMPL(templateClass, ...)      \
    do {                                         \
        templateClass<__VA_ARGS__> op;           \
        op.Init(x, gamma, y, rstd, &tilingData); \
        op.Process();                            \
    } while (0)

extern "C" __global__ __aicore__ void rms_norm(
    GM_ADDR x, GM_ADDR gamma, GM_ADDR y, GM_ADDR rstd, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (TILING_KEY_IS(RMSNORM_REGBASE_NORMAL)) {
        GET_TILING_DATA_WITH_STRUCT(RMSNormArch35TilingData, tiling_data_in, tiling);
        const RMSNormArch35TilingData* __restrict tilingData = &tiling_data_in;
        KernelRmsNormRegBasePerf<DTYPE_X, DTYPE_GAMMA> op;
        op.Init(x, gamma, y, rstd, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(RMSNORM_REGBASE_SPLIT)) {
        GET_TILING_DATA_WITH_STRUCT(RMSNormTilingData, tiling_data_in, tiling);
        const RMSNormTilingData* __restrict tilingData = &tiling_data_in;
        KernelRmsNormRegBaseSplitD<DTYPE_X, DTYPE_GAMMA> op;
        op.Init(x, gamma, y, rstd, tilingData);
        op.Process();
    }
}

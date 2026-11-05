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
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
#include "arch35/rms_norm_regbase_split_ar.h"
#include "arch35/rms_norm_regbase_trans.h"
#include "arch35/rms_norm_regbase_reduce_empty.h"
#endif
using namespace AscendC;
using namespace RmsNorm;

#define RMSNORM_REGBASE_NORMAL 5000
#define RMSNORM_REGBASE_SPLIT 2001
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
#define RMSNORM_REGBASE_SPLIT_AR 3000
#define RMSNORM_REGBASE_TRANSPOSE 4000
#define RMSNORM_REGBASE_REDUCE_EMPTY 6000
#endif

#define GENERAL_OP_IMPL(templateClass, ...)      \
    do {                                         \
        templateClass<__VA_ARGS__> op;           \
        op.Init(x, gamma, y, rstd, &tilingData); \
        op.Process();                            \
    } while (0)

extern "C" __global__ __aicore__ void rms_norm(GM_ADDR x, GM_ADDR gamma, GM_ADDR y, GM_ADDR rstd, GM_ADDR workspace,
                                               GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (TILING_KEY_IS(RMSNORM_REGBASE_NORMAL)) {
        GET_TILING_DATA_WITH_STRUCT(RMSNormArch35TilingData, tiling_data_in, tiling);
        const RMSNormArch35TilingData* __restrict tilingData = &tiling_data_in;
        KernelRmsNormRegBasePerf<DTYPE_X, DTYPE_GAMMA, false> op;
        op.Init(x, gamma, y, rstd, tilingData);
        op.Process();
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
    } else if (TILING_KEY_IS(RMSNORM_REGBASE_SPLIT_AR)) {
        KERNEL_TASK_TYPE(RMSNORM_REGBASE_SPLIT_AR, KERNEL_TYPE_MIX_AIV_1_0);
        GET_TILING_DATA_WITH_STRUCT(RMSNormRegbaseSplitARTilingData, tilingData, tiling);
        TPipe pipe;
        KernelRmsNormRegBaseSplitAR<DTYPE_X, DTYPE_GAMMA> op(&pipe);
        op.Init(x, gamma, y, rstd, GetUserWorkspace(workspace), &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(RMSNORM_REGBASE_TRANSPOSE)) {
        GET_TILING_DATA_WITH_STRUCT(RMSNormRegbaseTransTilingData, tilingData, tiling);
        TPipe pipe;
        KernelRmsNormRegBaseTrans<DTYPE_X, DTYPE_GAMMA> op(&pipe);
        op.Init(x, gamma, y, rstd, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(RMSNORM_REGBASE_REDUCE_EMPTY)) {
        GET_TILING_DATA_WITH_STRUCT(RMSNormRegbaseReduceEmptyTilingData, tilingData, tiling);
        TPipe pipe;
        KernelRmsNormRegBaseReduceEmpty op(&pipe);
        op.Init(rstd, &tilingData);
        op.Process();
#endif
    } else if (TILING_KEY_IS(RMSNORM_REGBASE_SPLIT)) {
        GET_TILING_DATA_WITH_STRUCT(RMSNormTilingData, tiling_data_in, tiling);
        const RMSNormTilingData* __restrict tilingData = &tiling_data_in;
        KernelRmsNormRegBaseSplitD<DTYPE_X, DTYPE_GAMMA, false> op;
        op.Init(x, gamma, y, rstd, tilingData);
        op.Process();
    }
}

/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file instance_norm_apt.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "arch35/instance_norm_tiling_data.h"
#include "arch35/instance_norm_common.h"
#include "arch35/instance_norm_ar_full_reduce.h"
#include "arch35/instance_norm_ar_welford.h"
#include "arch35/instance_norm_ara_full_reduce.h"
#include "arch35/instance_norm_ara_welford.h"

using namespace AscendC;
using namespace InstanceNormOps;

#define TILINGKEY_AR_FULL_REDUCE 200000
#define TILINGKEY_AR_WELFORD 300000
#define TILINGKEY_ARA_FULL_REDUCE 400000
#define TILINGKEY_ARA_WELFORD 500000

extern "C" __global__ __aicore__ void instance_norm(
    GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mean_out, GM_ADDR variance_out, GM_ADDR workspace,
    GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(InstanceNormARFullReduceTilingData);
    if (TILING_KEY_IS(TILINGKEY_AR_FULL_REDUCE)) {
        GET_TILING_DATA_WITH_STRUCT(InstanceNormARFullReduceTilingData, tilingData, tiling);
        InstanceNormARFullReduce<DTYPE_X, DTYPE_BETA, DTYPE_MEAN> op(&tilingData);
        op.Init(x, gamma, beta, y, mean_out, variance_out);
        op.Process();
    } else if (TILING_KEY_IS(TILINGKEY_AR_WELFORD)) {
        GET_TILING_DATA_WITH_STRUCT(InstanceNormARWelfordTilingData, tilingData, tiling);
        InstanceNormARWelford<DTYPE_X, DTYPE_BETA, DTYPE_MEAN> op(&tilingData);
        op.Init(x, gamma, beta, y, mean_out, variance_out);
        op.Process();
    } else if (TILING_KEY_IS(TILINGKEY_ARA_FULL_REDUCE)) {
        GET_TILING_DATA_WITH_STRUCT(InstanceNormARAFullReduceTilingData, tiling_data_in, tiling);
        const InstanceNormARAFullReduceTilingData* __restrict tilingData = &tiling_data_in;
        InstanceNormARAFullReduce<DTYPE_X, DTYPE_BETA, DTYPE_MEAN> op(tilingData);
        op.Init(x, gamma, beta, y, mean_out, variance_out);
        op.Process();
    } else if (TILING_KEY_IS(TILINGKEY_ARA_WELFORD)) {
        GET_TILING_DATA_WITH_STRUCT(InstanceNormARAWelfordTilingData, tiling_data_in, tiling);
        const InstanceNormARAWelfordTilingData* __restrict tilingData = &tiling_data_in;
        InstanceNormARAWelford<DTYPE_X, DTYPE_BETA, DTYPE_MEAN> op(tilingData);
        op.Init(x, gamma, beta, y, mean_out, variance_out);
        op.Process();
    }
}

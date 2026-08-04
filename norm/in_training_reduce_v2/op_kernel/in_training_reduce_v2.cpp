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
 * \file in_training_reduce_v2.cpp
 * \brief in_training_reduce_v2 kernel entry (Ascend950 / arch35 only)
 */

#include "kernel_operator.h"
#include "arch35/in_training_reduce_v2_tiling_data.h"
#include "arch35/in_training_reduce_v2_common.h"
#include "arch35/in_training_reduce_v2_ar_full_reduce.h"

using namespace AscendC;
using namespace INTrainingReduceV2Ops;

#define TILINGKEY_AR_FULL_REDUCE 200000

extern "C" __global__ __aicore__ void in_training_reduce_v2(GM_ADDR x, GM_ADDR sum, GM_ADDR square_sum,
                                                            GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(INTrainingReduceV2ARFullReduceTilingData);
    if (TILING_KEY_IS(TILINGKEY_AR_FULL_REDUCE)) {
        GET_TILING_DATA_WITH_STRUCT(INTrainingReduceV2ARFullReduceTilingData, tilingData, tiling);
        INTrainingReduceV2ARFullReduce<DTYPE_X> op(&tilingData);
        op.Init(x, sum, square_sum);
        op.Process();
    }
}

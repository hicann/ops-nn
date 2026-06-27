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
 * \file threshold_v2_apt.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "threshold_dag.h"
#include "threshold_tiling_struct.h"
#include "atvoss/util/dfx.h"
#include "atvoss/elewise/elewise_sch.h"

using namespace AscendC;
using namespace ThresholdOp;

template <uint64_t schMode, uint8_t valueMode>
__global__ __aicore__ void threshold_v2(
    GM_ADDR x, GM_ADDR threshold, GM_ADDR value, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(EleBaseTilingDataV2);
    GET_TILING_DATA_WITH_STRUCT(EleBaseTilingDataV2, tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    TPipe pipe;
    if constexpr (std::is_same_v<DTYPE_X, bfloat16_t>) {
        if constexpr (valueMode == TPL_HAS_VALUE) {
            ElementwiseSch<schMode, typename ThresholdCastDag<DTYPE_X, float>::OpDag> sch(&tilingData, &pipe);
            sch.Init(x, threshold, value, y);
            sch.Process();
        } else {
            ElementwiseSch<schMode, typename ThresholdCastDagNoValue<DTYPE_X, float>::OpDag> sch(&tilingData, &pipe);
            sch.Init(x, threshold, y);
            sch.Process();
        }
    } else {
        if constexpr (valueMode == TPL_HAS_VALUE) {
            ElementwiseSch<schMode, typename ThresholdDag<DTYPE_X>::OpDag> sch(&tilingData, &pipe);
            sch.Init(x, threshold, value, y);
            sch.Process();
        } else {
            ElementwiseSch<schMode, typename ThresholdDagNoValue<DTYPE_X>::OpDag> sch(&tilingData, &pipe);
            sch.Init(x, threshold, y);
            sch.Process();
        }
    }
}
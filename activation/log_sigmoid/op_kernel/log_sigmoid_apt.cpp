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
 * \file log_sigmoid.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "atvoss/elewise/elewise_sch.h"
#include "atvoss/util/dfx.h"
#include "arch35/log_sigmoid_dag.h"
#include "arch35/log_sigmoid_struct.h"

using namespace AscendC;
using namespace LogSigmoidOp;

template <uint64_t schMode, uint64_t dType>
__global__ __aicore__ void log_sigmoid(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(EleBaseTilingDataV2);
    GET_TILING_DATA_WITH_STRUCT(EleBaseTilingDataV2, tilingData, tiling);
    TPipe pipe;
    if constexpr (dType == static_cast<uint64_t>(TPL_FP16)) {
        ElementwiseSch<schMode, LogSigmoidDag::LogSigmoidNeedCast<half>::OpDag> sch(&tilingData, &pipe);
        sch.Init(x, y);
        sch.Process();
    } else if constexpr (dType == static_cast<uint64_t>(TPL_BF16)) {
        ElementwiseSch<schMode, LogSigmoidDag::LogSigmoidNeedCast<bfloat16_t>::OpDag> sch(&tilingData, &pipe);
        sch.Init(x, y);
        sch.Process();
    } else if constexpr (dType == static_cast<uint64_t>(TPL_FP32)) {
        ElementwiseSch<schMode, LogSigmoidDag::LogSigmoidNoCast<float>::OpDag> sch(&tilingData, &pipe);
        sch.Init(x, y);
        sch.Process();
    }
}
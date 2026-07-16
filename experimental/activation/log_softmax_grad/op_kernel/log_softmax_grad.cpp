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
 * \file log_softmax_grad.cpp
 * \brief
 */

#include "reduce_mid.h"
#include "reduce_tail.h"
#include "no_need_reduce.h"

using namespace NsLogSoftmaxGrad;

template <uint32_t SCH_MOD, bool IS_SMALL, bool IS_CONTIGUOUS>
__global__ __aicore__ void log_softmax_grad(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(LogSoftmaxGradTilingData);
    GET_TILING_DATA_WITH_STRUCT(LogSoftmaxGradTilingData, tilingData, tiling);

    TPipe pipe;
    constexpr int BUF_NUM = IS_SMALL ? 1 : 2;
    if constexpr (SCH_MOD == REDUCE_MID) {
        ReduceMid<DTYPE_INPUT_DY, IS_SMALL, IS_CONTIGUOUS, BUF_NUM> op;
        op.Init(tilingData, &pipe);
        op.Process(x, y, z);

    } else if constexpr (SCH_MOD == REDUCE_TAIL) {
        ReduceTail<DTYPE_INPUT_DY, IS_SMALL, IS_CONTIGUOUS, BUF_NUM> op;
        op.Init(tilingData, &pipe);
        op.Process(x, y, z);
    } else if constexpr (SCH_MOD == NO_NEED_REDUCE) {
        NoNeedReduce<DTYPE_INPUT_DY> op;
        op.Init(tilingData, &pipe);
        op.Process(x, y, z);
    }
}

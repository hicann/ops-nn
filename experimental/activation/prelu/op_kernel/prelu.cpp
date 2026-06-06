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
 * \file prelu.cpp
 * \brief Prelu 算子 kernel 入口
 */

#include "prelu.h"

template <uint32_t schMode>
__global__ __aicore__ void prelu(GM_ADDR x, GM_ADDR weight, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(PreluTilingData);
    GET_TILING_DATA_WITH_STRUCT(PreluTilingData, tilingData, tiling);

    AscendC::TPipe pipe;
#ifdef DTYPE_X
    NsPrelu::Prelu<DTYPE_X> op;
#else
    using KernelDtype = float;
    NsPrelu::Prelu<KernelDtype> op;
#endif
    if constexpr (schMode == PRELU_TPL_CHANNEL_FULL_L_MODE) {
        op.InitChannel(x, weight, y, &tilingData, &pipe);
        op.ProcessChannelFullL();
    } else if constexpr (schMode == PRELU_TPL_CHANNEL_NC_RESIDENT_WEIGHT_MODE) {
        op.InitChannelNcResidentWeightReuse(x, weight, y, &tilingData, &pipe);
        op.ProcessChannelNcResidentWeightReuse();
    } else if constexpr (schMode == PRELU_TPL_CHANNEL_NC_WEIGHT_REUSE_MODE) {
        op.InitChannelNcWeightReuse(x, weight, y, &tilingData, &pipe);
        op.ProcessChannelNcWeightReuse();
    } else if constexpr (schMode == PRELU_TPL_CHANNEL_NC_SPLIT_C_WEIGHT_REUSE_MODE) {
        op.InitChannelNcSplitCWeightReuse(x, weight, y, &tilingData, &pipe);
        op.ProcessChannelNcSplitCWeightReuse();
    } else if constexpr (schMode == PRELU_TPL_CHANNEL_SPLIT_L_MODE) {
        op.InitChannel(x, weight, y, &tilingData, &pipe);
        op.ProcessChannelSplitL();
    } else if constexpr (schMode == PRELU_TPL_CHANNEL_SPLIT_L_PARALLEL_MODE) {
        op.InitChannelSplitLParallel(x, weight, y, &tilingData, &pipe);
        op.ProcessChannelSplitLParallel();
    } else {
        op.InitScalar(x, weight, y, &tilingData, &pipe);
        op.ProcessScalar();
    }
}

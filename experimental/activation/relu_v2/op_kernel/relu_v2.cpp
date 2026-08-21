/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <type_traits>

#include "relu_v2.h"

using namespace AscendC;

template <typename D_T_X>
__global__ __aicore__ void relu_v2(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ReluV2TilingData);
    GET_TILING_DATA_WITH_STRUCT(ReluV2TilingData, tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    TPipe pipe;

    if constexpr (std::is_same_v<D_T_X, float> || std::is_same_v<D_T_X, half> || std::is_same_v<D_T_X, int32_t>) {
        NsReluV2::KernelReluV2<D_T_X> op;
        op.Init(x, y, &tilingData, &pipe);
        op.Process();
    } else if constexpr (std::is_same_v<D_T_X, bfloat16_t> || std::is_same_v<D_T_X, int16_t>) {
        NsReluV2::KernelReluV2Upcast<D_T_X, float> op;
        op.Init(x, y, &tilingData, &pipe);
        op.Process();
    } else if constexpr (std::is_same_v<D_T_X, int8_t>) {
        NsReluV2::KernelReluV2Upcast<D_T_X, half> op;
        op.Init(x, y, &tilingData, &pipe);
        op.Process();
    } else {
        NsReluV2::KernelReluV2VectorInt64<D_T_X> op;
        op.Init(x, y, &tilingData, &pipe);
        op.Process();
    }
}

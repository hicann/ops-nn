/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#include "arch35/ascend_requant_kernel.h"
#include "arch35/ascend_requant_tiling_data.h"

using TilingData4 = AscendRequantTilingData<4>;
using TilingData8 = AscendRequantTilingData<8>;

template <int RANK, bool DO_RELU>
__global__ __aicore__ void ascend_requant(GM_ADDR x, GM_ADDR req_scale, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_NONE_TILING;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    GM_ADDR ins[2] = {x, req_scale};
    GM_ADDR outs[1] = {y};

    if constexpr (RANK == 4) {
        GET_TILING_DATA_WITH_STRUCT(TilingData4, td, tiling);
        AscendRequantKernel<4, DO_RELU> kernel;
        kernel.Init(ins, outs, &td);
        kernel.Process();
    } else {
        GET_TILING_DATA_WITH_STRUCT(TilingData8, td, tiling);
        AscendRequantKernel<8, DO_RELU> kernel;
        kernel.Init(ins, outs, &td);
        kernel.Process();
    }
}

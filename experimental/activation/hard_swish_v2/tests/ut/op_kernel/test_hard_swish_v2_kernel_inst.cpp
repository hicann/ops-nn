/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include "../../../op_kernel/hard_swish_v2.h"

template <typename T>
__aicore__ inline void RunHardSwish(GM_ADDR x, GM_ADDR out, GM_ADDR tiling)
{
    NsHardSwishV2::HardSwishV2<T> op;
    op.Init(x, out, reinterpret_cast<HardSwishV2TilingData*>(tiling));
    op.Process();
}

extern "C" __global__ __aicore__ void hard_swish_v2_fp32_ut(GM_ADDR x, GM_ADDR out, GM_ADDR tiling)
{
    RunHardSwish<float>(x, out, tiling);
}

extern "C" __global__ __aicore__ void hard_swish_v2_fp16_ut(GM_ADDR x, GM_ADDR out, GM_ADDR tiling)
{
    RunHardSwish<half>(x, out, tiling);
}

extern "C" __global__ __aicore__ void hard_swish_v2_bf16_ut(GM_ADDR x, GM_ADDR out, GM_ADDR tiling)
{
    RunHardSwish<bfloat16_t>(x, out, tiling);
}

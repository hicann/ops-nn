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
 * \file arg_max_grad.cpp
 * \brief
 */
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "arch35/arg_max_grad_nd.h"
#include "arch35/arg_max_grad_tiling_data.h"
#include "arch35/arg_max_grad_tiling_key.h"

using namespace AscendC;

// dtype 由 def 的 dtype profile 驱动框架注入 DTYPE_VAR 分别实例化(float/half/int32_t/int8_t),
// 不进 TilingKey; 唯一的模板维度是 inner 是否为 1(算法分支, 见 arch35/arg_max_grad_tiling_key.h)。
template <uint64_t innerIsOne>
__global__ __aicore__ void arg_max_grad(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR y, GM_ADDR workspace,
                                        GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ArgMaxGradArch35TilingData);
    GET_TILING_DATA_WITH_STRUCT(ArgMaxGradArch35TilingData, tilingData, tiling);

    TPipe pipe;
    ArgMaxGrad::ArgMaxGradND<DTYPE_VAR, innerIsOne == 1> op;
    op.Init(var, indices, updates, y, &tilingData, &pipe);
    op.Process();
}

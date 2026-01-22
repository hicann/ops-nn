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
 * \file ada_layer_norm_quant_apt.cpp
 * \brief
 */

#include "../ada_layer_norm/arch35/ada_layer_norm_impl.h"

using namespace AdaLayerNormNS;

extern "C" __global__ __aicore__ void ada_layer_norm_quant(
    GM_ADDR x, GM_ADDR scale, GM_ADDR shift, GM_ADDR weight, GM_ADDR bias, GM_ADDR smooth_scales, GM_ADDR out,
    GM_ADDR quant_scale, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA(tilingData, tiling);

    GM_ADDR usrWorkspace = GetUserWorkspace(workspace);
    if (usrWorkspace == nullptr) {
        return;
    }

#define INIT_AND_PROCESS                              \
    op.InitQuant(&gmAddr, usrWorkspace, &tilingData); \
    op.Process()

    GmAddr gmAddr = {x, scale, shift, weight, bias, smooth_scales, out, nullptr, nullptr, quant_scale};
    if (TILING_KEY_IS(11)) {
        AdaLayerNormFullLoad<DTYPE_X, DTYPE_X, DTYPE_OUT, QUANT_OP_CODE> op;
        INIT_AND_PROCESS;
    } else if (TILING_KEY_IS(21)) {
        AdaLayerNormWelford<DTYPE_X, DTYPE_X, DTYPE_OUT, QUANT_OP_CODE> op;
        INIT_AND_PROCESS;
    }
}

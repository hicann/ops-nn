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
 * \file inplace_add_layer_norm.cpp
 * \brief
 */

#include "../../add_layer_norm/arch35/add_layer_norm.h"

extern "C" __global__ __aicore__ void inplace_add_layer_norm(GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR beta,
                                                             GM_ADDR bias, GM_ADDR y, GM_ADDR mean, GM_ADDR rstd,
                                                             GM_ADDR x, GM_ADDR workspace, GM_ADDR tiling)
{
    AddLayerNormImpl<DTYPE_X1>(x1, x2, gamma, beta, bias, y, mean, rstd, x, workspace, tiling);
}

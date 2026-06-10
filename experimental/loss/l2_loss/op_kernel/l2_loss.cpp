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
 * \file l2_loss.cpp
 * \brief
*/
#include "l2_loss.h"

template <uint32_t schMode>
__global__ __aicore__ void l2_loss(GM_ADDR x, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(L2LossTilingData);
    GET_TILING_DATA_WITH_STRUCT(L2LossTilingData, tilingData, tiling);
    NsL2Loss::L2Loss<DTYPE_X, schMode> op;
    op.Init(x, z, workspace, &tilingData);
    op.Process();
}

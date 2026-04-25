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
 * \file masked_scatter.cpp
 * \brief
 */

#include "masked_scatter.h"
#include "masked_scatter_exceed.h"

template <uint32_t updatesMode>
__global__ __aicore__ void masked_scatter(
    GM_ADDR x, GM_ADDR mask, GM_ADDR updates, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(MaskedScatterV1TilingData);
    GET_TILING_DATA_WITH_STRUCT(MaskedScatterV1TilingData, tilingData, tiling);
    if constexpr (updatesMode == 0) {
        MaskedScatterNS::MaskedScatter op;
        op.Init(x, mask, updates, y, &tilingData);
        op.Process();
    }
    if constexpr (updatesMode == 1) {
        MaskedScatterNS::MaskedScatterExceed op;
        op.Init(x, mask, updates, y, &tilingData);
        op.Process();
    }
}

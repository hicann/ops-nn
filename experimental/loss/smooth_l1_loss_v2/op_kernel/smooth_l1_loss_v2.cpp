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
 * \file smooth_l1_loss_v2.cpp
 * \brief
 */
#include "smooth_l1_loss_v2.h"

enum class SmoothL1LossV2TilingKey : uint32_t
{
    TILING_KEY_EXAMPLE_FLOAT = 0,
    TILING_KEY_EXAMPLE_OTHER = 1,
};

template <uint32_t schMode>
__global__ __aicore__ void smooth_l1_loss_v2(
    GM_ADDR predict, GM_ADDR label, GM_ADDR loss, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SmoothL1LossV2TilingData);
    GET_TILING_DATA_WITH_STRUCT(SmoothL1LossV2TilingData, tilingData, tiling);
    MySmoothL1LossV2::KernelSmoothL1LossV2<DTYPE_PREDICT, DTYPE_LABEL, DTYPE_LOSS> op;
    op.Init(predict, label, loss, workspace, tilingData);
    op.Process();
}
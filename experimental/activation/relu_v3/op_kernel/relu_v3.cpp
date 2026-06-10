/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "relu_v3.h"

enum class ReluV3TilingKey : uint32_t
{
    TILING_KEY = 0
};

template<uint32_t schMode>
__global__ __aicore__ void relu_v3(GM_ADDR x, GM_ADDR y, GM_ADDR mask, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(ReluV3TilingData);
    GET_TILING_DATA(tiling_data, tiling);
    if constexpr (schMode == static_cast<uint32_t>(ReluV3TilingKey::TILING_KEY))
        relu_v3<DTYPE_X>(x, y, mask, tiling_data);
}

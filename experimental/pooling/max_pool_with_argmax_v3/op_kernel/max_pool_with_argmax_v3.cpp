/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * Our normal copyright notice. Below are our remarks.
 *
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

#include "max_pool_with_argmax_v3.h"

template <uint32_t schMode>
__global__ __aicore__ void max_pool_with_argmax_v3(
    GM_ADDR x, GM_ADDR y, GM_ADDR argmax, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(MaxPoolWithArgmaxV3TilingData);
    GET_TILING_DATA_WITH_STRUCT(MaxPoolWithArgmaxV3TilingData, tilingData, tiling);

    if constexpr (schMode == MAXPOOL_TPL_SCH_MODE_FP16) {
        NsMaxPoolWithArgmaxV3::MaxPoolWithArgmaxV3<half> op;
        op.Init(x, y, argmax, &tilingData);
        op.Process();
    }
    if constexpr (schMode == MAXPOOL_TPL_SCH_MODE_FP32) {
        NsMaxPoolWithArgmaxV3::MaxPoolWithArgmaxV3<float> op;
        op.Init(x, y, argmax, &tilingData);
        op.Process();
    }
    if constexpr (schMode == MAXPOOL_TPL_SCH_MODE_BF16) {
        NsMaxPoolWithArgmaxV3::MaxPoolWithArgmaxV3<bfloat16_t> op;
        op.Init(x, y, argmax, &tilingData);
        op.Process();
    }
}

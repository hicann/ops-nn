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
 * \file foreach_asin.cpp
 * \brief kernel entry for foreach_asin
 */

#include "foreach_asin_simt.h"

using namespace AscendC;

template <uint32_t dtype>
__global__ __aicore__ void foreach_asin(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ForeachAsinTilingData);
    GET_TILING_DATA_WITH_STRUCT(ForeachAsinTilingData, tilingData, tiling);

    if constexpr (dtype == FOREACH_ASIN_TPL_DTYPE_FP32) {
        NsForeachAsin::Process<float>(x, y, &tilingData);
    } else if constexpr (dtype == FOREACH_ASIN_TPL_DTYPE_FP16) {
        NsForeachAsin::Process<half>(x, y, &tilingData);
    } else if constexpr (dtype == FOREACH_ASIN_TPL_DTYPE_BF16) {
        NsForeachAsin::Process<bfloat16_t>(x, y, &tilingData);
    }
}

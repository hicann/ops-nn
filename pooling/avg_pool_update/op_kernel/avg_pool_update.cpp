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
 * \file avg_pool_update.cpp
 * \brief Kernel entry for avg_pool_update operator
 *
 * Single template parameter:
 *   schMode (uint32_t): scene mode
 *     0 = ELEMWISE (element-wise division, the only scene)
 *   DTYPE_X1 macro auto-instantiates for fp16/fp32 dtype combinations.
 */

#include "arch35/avg_pool_update_simt.h"

template <uint32_t schMode>
__global__ __aicore__ void avg_pool_update(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(AvgPoolUpdateTilingData);
    GET_TILING_DATA_WITH_STRUCT(AvgPoolUpdateTilingData, tilingData, tiling);

    if constexpr (schMode == static_cast<uint32_t>(AVG_POOL_UPDATE_SCH_MODE_ELEMWISE)) {
        NsAvgPoolUpdate::Process<DTYPE_X1>(x1, x2, y, workspace, tiling);
    }
}

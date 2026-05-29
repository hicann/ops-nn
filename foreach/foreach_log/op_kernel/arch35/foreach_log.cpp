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
 * \file foreach_log.cpp
 * \brief kernel entry for foreach_log (arch35 / Ascend950)
 */

#include "foreach_log_simt.h"

enum class ForeachLogTilingKey : uint32_t {
    TILING_KEY_FLOAT = 0,
    TILING_KEY_FLOAT16 = 1,
    TILING_KEY_BF16 = 2,
};

template <uint32_t schMode>
__global__ __aicore__ void foreach_log(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ForeachLogTilingData);
    GET_TILING_DATA_WITH_STRUCT(ForeachLogTilingData, tilingData, tiling);

    const __gm__ ForeachLogTilingData* tilingGm =
        reinterpret_cast<const __gm__ ForeachLogTilingData*>(tiling);

    if constexpr (schMode == static_cast<uint32_t>(ForeachLogTilingKey::TILING_KEY_FLOAT)) {
        NsForeachLog::Process<float>(x, y, tilingGm);
    } else if constexpr (schMode == static_cast<uint32_t>(ForeachLogTilingKey::TILING_KEY_FLOAT16)) {
        NsForeachLog::Process<half>(x, y, tilingGm);
    } else if constexpr (schMode == static_cast<uint32_t>(ForeachLogTilingKey::TILING_KEY_BF16)) {
        NsForeachLog::Process<bfloat16_t>(x, y, tilingGm);
    }
}

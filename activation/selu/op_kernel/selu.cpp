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
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/*!
 * \file selu.cpp
 * \brief Selu kernel entry (arch35, Ascend950)
 *
 * def 驱动 dtype 模式：dtype 由 def 文件 DataType 列表驱动，构建系统注入 DTYPE_X 编译宏；
 * 模板参数 schMode 仅为占位调度维度（Selu 无算法/调度分支）。
 * 实际数据类型分支在 NsSelu::Selu<DTYPE_X> 内由 if constexpr 编译期分发：
 *   - float:       direct fp32 computation
 *   - half:        cast to fp32 -> compute -> cast back to half
 *   - bfloat16_t:  cast to fp32 -> compute -> cast back
 *   - int32_t:     cast to fp32 -> compute -> cast back
 *   - int8_t:      int8 -> half -> compute -> ceil negative -> int8
 *
 * Kernel function signature: x, y, workspace, tiling
 */

#include "arch35/selu.h"

enum class SeluTilingKey : uint32_t {
    TILING_KEY_SELU = SELU_SCH_MODE_0,
};

template <uint32_t schMode>
__global__ __aicore__ void selu(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SeluTilingData);
    GET_TILING_DATA_WITH_STRUCT(SeluTilingData, tilingData, tiling);
    if constexpr (schMode == static_cast<uint32_t>(SeluTilingKey::TILING_KEY_SELU)) {
        NsSelu::Selu<DTYPE_X> op;
        op.Init(x, y, &tilingData);
        op.Process();
    }
}

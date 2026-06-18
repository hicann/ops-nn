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
 * \file fake_quant_with_min_max_vars_per_channel_arch35.cpp
 * \brief FakeQuantWithMinMaxVarsPerChannel kernel 入口（仅 arch35 / DAV_3510）
 *
 * 模板参数（与 fake_quant_with_min_max_vars_per_channel_tiling_key.h 对应）：
 *   - D_T_X      : 输入 x 数据类型占位（仅 FQ_TPL_DT_FP32）
 *   - MODE       : PER_MODE = 2 (Native PC) — 占位
 *   - HAS_ZP     : 0 — 占位
 *   - ROUND_MODE : 0 (floor + 0.5, TF 兼容) — 占位
 *
 * 算子签名（固定）：
 *   kernel(x, min, max, y, workspace, tiling)
 */

#include "arch35/fake_quant_with_min_max_vars_per_channel.h"
#include "arch35/fake_quant_with_min_max_vars_per_channel_tiling_key.h"

template <int D_T_X, int MODE, int HAS_ZP, int ROUND_MODE>
__global__ __aicore__ void fake_quant_with_min_max_vars_per_channel(
    GM_ADDR x, GM_ADDR minIn, GM_ADDR maxIn, GM_ADDR y,
    GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(FakeQuantWithMinMaxVarsPerChannelTilingData);
    GET_TILING_DATA_WITH_STRUCT(FakeQuantWithMinMaxVarsPerChannelTilingData, tilingData, tiling);

    if constexpr (D_T_X == FQ_TPL_DT_FP32 && MODE == 2 && HAS_ZP == 0 && ROUND_MODE == 0) {
        NsFakeQuantPerChannel::FakeQuantPerChannelNativePC<float> op;
        op.Init(x, minIn, maxIn, y, &tilingData);
        op.Process();
    }
}

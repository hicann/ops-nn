/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fake_quant_with_min_max_vars_per_channel_gradient_tiling.h
 * \brief
 */
#ifndef OPS_BUILD_IN_OP_TILING_RUNTIME_FAKE_QUANT_WITH_MIN_MAX_VARS_PER_CHANNEL_GRADIENT_H
#define OPS_BUILD_IN_OP_TILING_RUNTIME_FAKE_QUANT_WITH_MIN_MAX_VARS_PER_CHANNEL_GRADIENT_H
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include "util/math_util.h"
#include "platform/platform_infos_def.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(FakeQuantWithMinMaxVarsPerChannelGradientTilingData)
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, rowsPerCore);
TILING_DATA_FIELD_DEF(uint32_t, tailRows);
TILING_DATA_FIELD_DEF(uint32_t, totalRows);
TILING_DATA_FIELD_DEF(uint32_t, channelNum);
TILING_DATA_FIELD_DEF(uint32_t, tileLen);
TILING_DATA_FIELD_DEF(int32_t, quantMin);
TILING_DATA_FIELD_DEF(int32_t, quantMax);
TILING_DATA_FIELD_DEF(uint32_t, dTileLen);
TILING_DATA_FIELD_DEF(uint32_t, numDChunks);
TILING_DATA_FIELD_DEF(uint32_t, splitMode);  // 0 = split rows, 1 = split d-chunks
TILING_DATA_FIELD_DEF(uint32_t, chunksPerCore);
TILING_DATA_FIELD_DEF(uint32_t, tailChunks);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(FakeQuantWithMinMaxVarsPerChannelGradient,
                           FakeQuantWithMinMaxVarsPerChannelGradientTilingData)

struct FakeQuantWithMinMaxVarsPerChannelGradientCompileInfo {
    int32_t totalCoreNum = 0;
    uint64_t ubSizePlatForm = 0;
};
} // namespace optiling
#endif // OPS_BUILD_IN_OP_TILING_RUNTIME_FAKE_QUANT_WITH_MIN_MAX_VARS_PER_CHANNEL_GRADIENT_H

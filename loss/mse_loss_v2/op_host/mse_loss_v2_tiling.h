/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mse_loss_v2_tiling.h
 * \brief
 * 
 * 
 * 
 */

#ifndef MSE_LOSS_V2_TILING_DEF_H
#define MSE_LOSS_V2_TILING_DEF_H

#include <vector>
#include <iostream>
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MSELossV2TilingData)
TILING_DATA_FIELD_DEF(uint64_t, coreNum);
TILING_DATA_FIELD_DEF(uint64_t, tailElems);
TILING_DATA_FIELD_DEF(uint64_t, bufferNum);
TILING_DATA_FIELD_DEF(float, scale);
TILING_DATA_FIELD_DEF(uint64_t, epochs);
TILING_DATA_FIELD_DEF(uint64_t, epochsForLastCore);
TILING_DATA_FIELD_DEF(uint64_t, coreLength);
TILING_DATA_FIELD_DEF(uint64_t, tileLength);
TILING_DATA_FIELD_DEF(uint64_t, tailTileLength);
TILING_DATA_FIELD_DEF(uint64_t, tailTileLengthForLastCore);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MSELossV2, MSELossV2TilingData)
extern ge::graphStatus TilingParse4MSELossV2(gert::TilingParseContext* context);
extern ge::graphStatus Tiling4MSELossV2(gert::TilingContext* context);
} // namespace optiling

#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_MSE_LOSS_V2_H

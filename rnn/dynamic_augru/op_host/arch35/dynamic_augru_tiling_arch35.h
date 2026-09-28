/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_RNN_DYNAMIC_AUGRU_OP_HOST_ARCH35_DYNAMIC_AUGRU_TILING_ARCH35_H_
#define OPS_RNN_DYNAMIC_AUGRU_OP_HOST_ARCH35_DYNAMIC_AUGRU_TILING_ARCH35_H_

#include "../dynamic_augru_tiling.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(DynamicAUGRUTilingData)
TILING_DATA_FIELD_DEF(int64_t, timeSize);
TILING_DATA_FIELD_DEF(int64_t, batchSize);
TILING_DATA_FIELD_DEF(int64_t, inputSize);
TILING_DATA_FIELD_DEF(int64_t, hiddenSize);
TILING_DATA_FIELD_DEF(int64_t, tileHidden);
TILING_DATA_FIELD_DEF(uint32_t, hasBiasInput);
TILING_DATA_FIELD_DEF(uint32_t, hasBiasHidden);
TILING_DATA_FIELD_DEF(uint32_t, hasInitH);
TILING_DATA_FIELD_DEF(uint32_t, sequenceMode);
TILING_DATA_FIELD_DEF(uint32_t, gateOrder);
TILING_DATA_FIELD_DEF(uint32_t, stateType);
TILING_DATA_FIELD_DEF(uint32_t, usedAicCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, usedAivCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, blockSize);
TILING_DATA_FIELD_DEF(uint32_t, vectorLength);
TILING_DATA_FIELD_DEF(uint64_t, inputProjectionOffset);
TILING_DATA_FIELD_DEF(uint64_t, hiddenProjectionOffset);
TILING_DATA_FIELD_DEF(uint64_t, stateFp32Offset);
TILING_DATA_FIELD_DEF(uint64_t, weightHiddenFp32Offset);
TILING_DATA_FIELD_DEF(uint64_t, userWorkspaceSize);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, inputMMTiling);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, hiddenMMTiling);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(DynamicAUGRU, DynamicAUGRUTilingData);

ge::graphStatus Tiling4DynamicAUGRUArch35(gert::TilingContext* context, const DynamicAUGRUCompileInfo* compileInfo);
ge::graphStatus Tiling4DynamicAUGRU(gert::TilingContext* context);
ge::graphStatus TilingPrepare4DynamicAUGRU(gert::TilingParseContext* context);
} // namespace optiling
#endif // OPS_RNN_DYNAMIC_AUGRU_OP_HOST_ARCH35_DYNAMIC_AUGRU_TILING_ARCH35_H_

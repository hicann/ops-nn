/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __OP_HOST_GROUP_NORM_SILU_QUANT_TILING_H__
#define __OP_HOST_GROUP_NORM_SILU_QUANT_TILING_H__

#include <cstdint>
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "tiling_base/tiling_base.h"
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
#include "tiling_base/tiling_base.h"
#include "op_common/op_host/util/platform_util.h"
#include "tiling_base/tiling_templates_registry.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GroupNormSiluQuantTilingData)
TILING_DATA_FIELD_DEF(int64_t, numGroups);
TILING_DATA_FIELD_DEF(int64_t, hwNum);
TILING_DATA_FIELD_DEF(int64_t, elemNum);
TILING_DATA_FIELD_DEF(int64_t, shapeC);
TILING_DATA_FIELD_DEF(int64_t, shapeD);
TILING_DATA_FIELD_DEF(int64_t, shapeQuantScale);
TILING_DATA_FIELD_DEF(int64_t, realCoreNum);
TILING_DATA_FIELD_DEF(int64_t, numPerCore);
TILING_DATA_FIELD_DEF(int64_t, numLastCore);
TILING_DATA_FIELD_DEF(int64_t, processSize);
TILING_DATA_FIELD_DEF(int64_t, loopNum);
TILING_DATA_FIELD_DEF(int64_t, loopTail);
TILING_DATA_FIELD_DEF(int64_t, innerLoopNum);
TILING_DATA_FIELD_DEF(int64_t, innerLoopTail);
TILING_DATA_FIELD_DEF(int64_t, tilingKey);
TILING_DATA_FIELD_DEF(float, epsilon);
TILING_DATA_FIELD_DEF(int64_t, activateSilu);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GroupNormSiluQuant, GroupNormSiluQuantTilingData)

struct GroupNormSiluQuantCompileInfo {
    int32_t totalCoreNum = 0;
    uint64_t ubSizePlatForm = 0;
};

enum class GroupNormSiluQuantTilingKey : int64_t
{
    TILINGKEY_HIGH_PERF_B16 = 1031,         // high performance and dtype is b16
};
} // namespace optiling
#endif // __OP_HOST_GROUP_NORM_SILU_QUANT_TILING_H__

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
 * \file situ_glu_group_tiling_arch35.h
 * \brief SituGluGroup tiling data and entry declarations (arch35 / Ascend 950 only).
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_SITU_GLU_GROUP_TILING_ARCH35_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_SITU_GLU_GROUP_TILING_ARCH35_H_

#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(SituGluGroupTilingData)
TILING_DATA_FIELD_DEF(int64_t, bs);
TILING_DATA_FIELD_DEF(int64_t, d);
TILING_DATA_FIELD_DEF(int64_t, splitD);
TILING_DATA_FIELD_DEF(int64_t, rowFactor);
TILING_DATA_FIELD_DEF(int64_t, dLoop);
TILING_DATA_FIELD_DEF(int64_t, dFactor);
TILING_DATA_FIELD_DEF(int64_t, tailDFactor);
TILING_DATA_FIELD_DEF(float, beta);
TILING_DATA_FIELD_DEF(float, linearBeta);
TILING_DATA_FIELD_DEF(int64_t, activateLeft);
TILING_DATA_FIELD_DEF(int64_t, ubSize);
TILING_DATA_FIELD_DEF(int64_t, groupIndexLoop);
TILING_DATA_FIELD_DEF(int64_t, groupIndexFactor);
TILING_DATA_FIELD_DEF(int64_t, tailGroupIndexFactor);
// Presence flags decided by Tiling. The kernel must follow these instead of only testing the
// input pointers: if an optional descriptor exists but its shape is null, Tiling treats it as
// absent and excludes it from the UB budget, so the kernel must not enter that path either.
TILING_DATA_FIELD_DEF(int64_t, hasWeight);
TILING_DATA_FIELD_DEF(int64_t, hasGroupIndex);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SituGluGroup, SituGluGroupTilingData)

// Intentionally empty: platform parameters (coreNum/ubSize) are queried directly in Tiling
// instead of being passed through CompileInfo.
struct SituGluGroupCompileInfo {};

ge::graphStatus Tiling4SituGluGroupArch35(gert::TilingContext* context);
ge::graphStatus TilingPrepareForSituGluGroup(gert::TilingParseContext* context);
} // namespace optiling
#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_SITU_GLU_GROUP_TILING_ARCH35_H_

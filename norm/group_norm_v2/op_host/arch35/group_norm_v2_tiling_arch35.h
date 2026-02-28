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
 * \file group_norm_v2_tiling_arch35.h
 * \brief
 */

#ifndef GROUP_NORM_V2_TILING_ARCH35_H
#define GROUP_NORM_V2_TILING_ARCH35_H

#include <cstdint>
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "op_host/tiling_base.h"
#include "op_api/runtime2_util.h"

namespace optiling {
struct GroupNormV2CompileInfo {
  int32_t coreNum;
  uint64_t ubSize;
};

BEGIN_TILING_DATA_DEF(GroupNormV2TilingData)
TILING_DATA_FIELD_DEF(int64_t, numGroups);
TILING_DATA_FIELD_DEF(int64_t, hwNum);
TILING_DATA_FIELD_DEF(int64_t, elemNum);
TILING_DATA_FIELD_DEF(int64_t, shapeC);
TILING_DATA_FIELD_DEF(int64_t, shapeD);
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
TILING_DATA_FIELD_DEF(int64_t, parallelN);
TILING_DATA_FIELD_DEF(int64_t, ubSize);
TILING_DATA_FIELD_DEF(int64_t, dichotomyAddPower);
TILING_DATA_FIELD_DEF(int64_t, dichotomyAddK);
TILING_DATA_FIELD_DEF(int64_t, dichotomyAddLastNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GroupNormV2, GroupNormV2TilingData);

enum class GroupNormV2TilingKey : int64_t {
  TILINGKEY_WELFORD_PERF = 1100,  // 950 use welford for mean/rstd, R is partial Load
  TILINGKEY_WELFORD_PERF_MIX_TYPE = 1101,
  TILINGKEY_TWOPASS_PERF = 1110,  // 950 use twopass for mean/rstd, R is full Load
  TILINGKEY_TWOPASS_PERF_MIX_TYPE = 1111,
  TILINGKEY_WELFORD_GENERALIZED = 1120,  // 950 basic template and use welford for mean/rstd, R is partial Load
  TILINGKEY_WELFORD_GENERALIZED_MIX_TYPE = 1121,
  TILINGKEY_TWOPASS_GENERALIZED = 1130,  // 950 basic template and use twopass for mean/rstd, R is full Load
  TILINGKEY_TWOPASS_GENERALIZED_MIX_TYPE = 1131
};

ge::graphStatus SetTilingData(gert::TilingContext* context);
}  // namespace optiling
#endif
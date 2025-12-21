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
 * \file fused_linear_online_max_sum_tiling.h
 * \brief
 */
#ifndef __OP_HOST_FUSED_LINEAR_ONLINE_MAX_SUM_TILING_H__
#define __OP_HOST_FUSED_LINEAR_ONLINE_MAX_SUM_TILING_H__

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(FusedLinearOnlineMaxSumTilingData)
  TILING_DATA_FIELD_DEF(uint64_t, m);
  TILING_DATA_FIELD_DEF(uint64_t, k);
  TILING_DATA_FIELD_DEF(uint64_t, n);
  TILING_DATA_FIELD_DEF(uint64_t, bufSize);
  TILING_DATA_FIELD_DEF(uint64_t, cubeCoreNum);
  TILING_DATA_FIELD_DEF(uint64_t, vecCoreNum);
  TILING_DATA_FIELD_DEF(uint64_t, batchTaksPerVecCore);
  TILING_DATA_FIELD_DEF(uint64_t, batchTaksTailVecCore);
  TILING_DATA_FIELD_DEF(uint64_t, targetTasksPerLoop);
  TILING_DATA_FIELD_DEF(float, vocabStartIndex);
  TILING_DATA_FIELD_DEF(float, vocabEndIndex);
  TILING_DATA_FIELD_DEF(uint64_t, initWorkspaceLength);
  TILING_DATA_FIELD_DEF(uint64_t, cubeCoreNumAligned);
  TILING_DATA_FIELD_DEF(uint64_t, matmulInputEmptyFlag);
  TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mmTiling);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(FusedLinearOnlineMaxSum, FusedLinearOnlineMaxSumTilingData)

struct FusedLinearOnlineMaxSumCompileInfo {};
}  // namespace optiling

#endif // __OP_HOST_FUSED_LINEAR_ONLINE_MAX_SUM_TILING_H__

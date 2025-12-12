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
 * \file expand_into_jagged_permute_tiling.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_EXPAND_INTO_JAGGED_PERMUTE_TILING_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_EXPAND_INTO_JAGGED_PERMUTE_TILING_H

#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
using namespace ge;
namespace optiling {
  const static int64_t SIZE = 512;
  constexpr int64_t TILINGKEY = 0;
  
  struct ExpandIntoJaggedPermuteParam {
  int64_t permute = 0;
  int64_t inputoffset = 0;
  int64_t outputoffset = 0;
  int64_t outputSize = 0;
  int64_t dtypeSize = 0;
  int64_t coreNum = 0;
  int64_t oneTaskLen = 0;
  int64_t taskNum = 0;
  int64_t blockDim = 0;
  int64_t tilingKey = 0;
  int64_t oneTaskInputOffsetLen = 0;
  int64_t cacheLine = 0;
  int64_t maxCoreMemery = 0;
  int64_t numTail = 0;
  int64_t offsetLen = 0;
};


struct ExpandIntoJaggedPermuteCompileInfo {};

BEGIN_TILING_DATA_DEF(ExpandIntoJaggedPermuteTilingData)
  TILING_DATA_FIELD_DEF(int64_t, realCoreNum);
  TILING_DATA_FIELD_DEF(int64_t, frontCoreNum);
  TILING_DATA_FIELD_DEF(int64_t, blockFactor);  // 任务数
  TILING_DATA_FIELD_DEF(int64_t, oneTaskLen);
  TILING_DATA_FIELD_DEF(int64_t, tailCoreBlockFactor);
  TILING_DATA_FIELD_DEF(int64_t, lastTaskLen);
  TILING_DATA_FIELD_DEF(int64_t, oneTaskOffsetLen);
  TILING_DATA_FIELD_DEF(int64_t, inputLen);
  TILING_DATA_FIELD_DEF(int64_t, offsetLen);
  TILING_DATA_FIELD_DEF(int64_t, outputSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ExpandIntoJaggedPermute, ExpandIntoJaggedPermuteTilingData)

} //namespce optiling
#endif  

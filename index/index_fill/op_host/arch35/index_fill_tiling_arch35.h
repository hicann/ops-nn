/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file index_fill_tiling_arch35.h
 * \brief
 */
#ifndef INDEX_FILL_TILING_ARCH35_H
#define INDEX_FILL_TILING_ARCH35_H

#include "register/tilingdata_base.h"

namespace optiling {
struct TilingDataStructIndexFillArch35 {
  uint32_t coreNum = 0;
  uint64_t N = 0; // x在axis上的维度值
  uint64_t indicesNum = 0; // 索引tensor长度
  uint64_t indicesProcessMode = 0; // 索引处理模式
  uint64_t frontCoreNumTaskIndices = 0;
  uint64_t tailCoreNumTaskIndices = 0;
  uint64_t frontCoreDataTaskIndices = 0;
  uint64_t tailCoreDataTaskIndices = 0;
  uint64_t ubSize = 0;
  uint64_t P = 0;
  uint64_t Q = 0;

  uint32_t tilingKey = 0;
};

struct IndexFillCompileInfoArch35 {
  int32_t totalCoreNum = 0;
  uint64_t ubSizePlatForm = 0;
  uint32_t sysWorkspaceSize = 0;
};
}
#endif // INDEX_FILL_TILING_ARCH35_H

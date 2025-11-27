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
 * \file index_fill_tiling.h
 * \brief
 */
#ifndef INDEX_FILL_TILING_H
#define INDEX_FILL_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(IndexFillTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, coreNum);
  TILING_DATA_FIELD_DEF(uint64_t, N); // x在axis上的维度值
  TILING_DATA_FIELD_DEF(uint64_t, indicesNum); // 索引tensor长度
  TILING_DATA_FIELD_DEF(uint64_t, indicesProcessMode); // 索引处理模式
  // N < coreNum * 16，按indicesNum分核。N >= coreNum * 16，按N分核，都使用下面四个分核参数
  TILING_DATA_FIELD_DEF(uint64_t, frontCoreNumTaskIndices);
  TILING_DATA_FIELD_DEF(uint64_t, tailCoreNumTaskIndices);
  TILING_DATA_FIELD_DEF(uint64_t, frontCoreDataTaskIndices); 
  TILING_DATA_FIELD_DEF(uint64_t, tailCoreDataTaskIndices);
  TILING_DATA_FIELD_DEF(uint64_t, ubSize);
  // 非尾轴场景所用核
  TILING_DATA_FIELD_DEF(uint64_t, P); // x在axis轴前面的轴乘积
  TILING_DATA_FIELD_DEF(uint64_t, Q); // x在axis轴后面的轴乘积
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(IndexFill, IndexFillTilingData)

struct IndexFillCompileInfo {
  int32_t totalCoreNum = 0;
  uint64_t ubSizePlatForm = 0;
};
}
#endif
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
 * \file inplace_scatter_add_tiling.h
 * \brief
 */
#ifndef INPLACE_SCATTER_ADD_TILING_H
#define INPLACE_SCATTER_ADD_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(InplaceScatterAddTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, M);
  TILING_DATA_FIELD_DEF(uint32_t, N);
  TILING_DATA_FIELD_DEF(uint32_t, K);
  TILING_DATA_FIELD_DEF(uint32_t, NAligned);
  TILING_DATA_FIELD_DEF(uint32_t, frontCoreNum);
  TILING_DATA_FIELD_DEF(uint32_t, tailCoreNum);
  TILING_DATA_FIELD_DEF(uint32_t, frontCoreIndicesNum);
  TILING_DATA_FIELD_DEF(uint32_t, tailCoreIndicesNum);
  TILING_DATA_FIELD_DEF(uint32_t, ubSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(InplaceScatterAdd, InplaceScatterAddTilingData)

struct InplaceScatterAddCompileInfo {
  int32_t totalCoreNum = 0;
  uint64_t ubSizePlatForm = 0;
};
}
#endif
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
 * \file pp_matmul_tiling.h
 * \brief
 * 
 * 
 * 
 * 
 * 
 * 
 */
#ifndef __OP_HOST_PP_MATMUL_TILING_H__
#define __OP_HOST_PP_MATMUL_TILING_H__
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(PpMatmulTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, batch);
  TILING_DATA_FIELD_DEF(uint32_t, m);
  TILING_DATA_FIELD_DEF(uint32_t, k);
  TILING_DATA_FIELD_DEF(uint32_t, n);
  TILING_DATA_FIELD_DEF(uint32_t, m0);
  TILING_DATA_FIELD_DEF(uint32_t, k0);
  TILING_DATA_FIELD_DEF(uint32_t, n0);
  TILING_DATA_FIELD_DEF(uint32_t, mLoop);
  TILING_DATA_FIELD_DEF(uint32_t, kLoop);
  TILING_DATA_FIELD_DEF(uint32_t, nLoop);
  TILING_DATA_FIELD_DEF(uint32_t, coreLoop);
  TILING_DATA_FIELD_DEF(uint32_t, swizzlCount);
  TILING_DATA_FIELD_DEF(uint32_t, tilingKey);
  TILING_DATA_FIELD_DEF(uint32_t, blockDim);
  TILING_DATA_FIELD_DEF(uint32_t, swizzlDirect);
  TILING_DATA_FIELD_DEF(uint32_t, splitk);
  TILING_DATA_FIELD_DEF(uint32_t, enShuffleK);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(PpMatMul, PpMatmulTilingData)
}
#endif // __OP_HOST_BATCH_MATMUL_V3_TILING_H__
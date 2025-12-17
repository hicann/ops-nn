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
 * \file matmul_v3_tiling.h
 * \brief
 * ATTENTION: MAKE SURE 'BEGIN_TILING_DATA_DEF' STAY IN THE SAME LINE
 * 
 *
 *
 *
 *
 *
 */
#ifndef __OP_HOST_MATMUL_V3_TILING_H__
#define __OP_HOST_MATMUL_V3_TILING_H__
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(L2cacheUseInfo)
  TILING_DATA_FIELD_DEF(uint32_t, l2CacheFlag);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(L2cacheUseInfoOp, L2cacheUseInfo);

BEGIN_TILING_DATA_DEF(L2cacheTilePara)
  TILING_DATA_FIELD_DEF(uint32_t, mTileCntL2);
  TILING_DATA_FIELD_DEF(uint32_t, nTileCntL2);
  TILING_DATA_FIELD_DEF(uint32_t, mTileBlock);
  TILING_DATA_FIELD_DEF(uint32_t, nTileBlock);
  TILING_DATA_FIELD_DEF(uint32_t, calOrder);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(L2cacheTileParaOp, L2cacheTilePara)

BEGIN_TILING_DATA_DEF(MatMulRunInfo)
  TILING_DATA_FIELD_DEF(uint32_t, transA);
  TILING_DATA_FIELD_DEF(uint32_t, transB);
  TILING_DATA_FIELD_DEF(uint32_t, nd2nzA);
  TILING_DATA_FIELD_DEF(uint32_t, nd2nzB);
  TILING_DATA_FIELD_DEF(uint32_t, isNzA);
  TILING_DATA_FIELD_DEF(uint32_t, isNzB);
  TILING_DATA_FIELD_DEF(uint32_t, isHf32);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MatMulRunInfoOp, MatMulRunInfo)

BEGIN_TILING_DATA_DEF(MatmulTilingData)
  TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, matmulTiling);
  TILING_DATA_FIELD_DEF_STRUCT(L2cacheTilePara, tileL2cacheTiling);
  TILING_DATA_FIELD_DEF_STRUCT(MatMulRunInfo, matmulRunInfo);
  TILING_DATA_FIELD_DEF_STRUCT(L2cacheUseInfo, l2cacheUseInfo);
  TILING_DATA_FIELD_DEF(uint32_t, baseAN);
  TILING_DATA_FIELD_DEF(uint32_t, baseAD);
  TILING_DATA_FIELD_DEF(uint32_t, baseBN);
  TILING_DATA_FIELD_DEF(uint32_t, baseBD);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MatMulV3, MatmulTilingData)
REGISTER_TILING_DATA_CLASS(MatmulTilingDataOp, MatmulTilingData)
}
#endif // __OP_HOST_MATMUL_V3_TILING_H__
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
 * \file transpose_batch_mat_mul_tiling.h
 * \brief
 */
#ifndef __OP_HOST_TRANSPOSE_BATCH_MAT_MUL_TILING_H__
#define __OP_HOST_TRANSPOSE_BATCH_MAT_MUL_TILING_H__
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "matmul/mat_mul_v3/op_host/op_tiling/matmul_v3_tiling.h"
#include "matmul/batch_mat_mul_v3/op_host/op_tiling/batch_mat_mul_v3_tiling.h"
#include "pp_matmul_tiling.h"

namespace optiling {
REGISTER_TILING_DATA_CLASS(TransposeBatchMatMul_36, PpMatmulTilingData)
REGISTER_TILING_DATA_CLASS(TransposeBatchMatMul_292, PpMatmulTilingData)

BEGIN_TILING_DATA_DEF(TBMMTilingData)
  TILING_DATA_FIELD_DEF_STRUCT(MatmulTilingData, matmulTiling);
  TILING_DATA_FIELD_DEF_STRUCT(MultiBatchInfo, multiBatchInfo);
  TILING_DATA_FIELD_DEF(int32_t, batchSplitFactor);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(TransposeBatchMatMul, TBMMTilingData)
}
#endif // __OP_HOST_TRANSPOSE_BATCH_MAT_MUL_TILING_H__
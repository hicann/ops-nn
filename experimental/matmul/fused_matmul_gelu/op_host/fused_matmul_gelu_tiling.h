/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_HOST_FUSED_MATMUL_GELU_TILING_H_
#define OP_HOST_FUSED_MATMUL_GELU_TILING_H_

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(FusedMatmulGeluTilingData)
TILING_DATA_FIELD_DEF(uint64_t, m);
TILING_DATA_FIELD_DEF(uint64_t, k);
TILING_DATA_FIELD_DEF(uint64_t, n);
TILING_DATA_FIELD_DEF(uint64_t, totalElement);
TILING_DATA_FIELD_DEF(uint64_t, bufSize);
TILING_DATA_FIELD_DEF(uint64_t, cubeCoreNum);
TILING_DATA_FIELD_DEF(uint64_t, vecCoreNum);
TILING_DATA_FIELD_DEF(uint64_t, vecTasksPerCore);
TILING_DATA_FIELD_DEF(uint64_t, vecTasksTailCore);
TILING_DATA_FIELD_DEF(uint64_t, elemsPerVecLoop);
TILING_DATA_FIELD_DEF(uint64_t, hasBias);
TILING_DATA_FIELD_DEF(uint64_t, approximate);
TILING_DATA_FIELD_DEF(uint64_t, matmulWorkspaceSize);
TILING_DATA_FIELD_DEF(uint64_t, cubeCoreNumAligned);

TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mmTiling);

// Original CMCT bridge plan extension.
// Keep this suffix after mmTiling so the existing kernel-side tiling ABI
// remains compatible with the stable workspace path.
TILING_DATA_FIELD_DEF(uint32_t, fmgUsedCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, fmgML1);
TILING_DATA_FIELD_DEF(uint32_t, fmgNL1);
TILING_DATA_FIELD_DEF(uint32_t, fmgKL1);
TILING_DATA_FIELD_DEF(uint32_t, fmgBaseM);
TILING_DATA_FIELD_DEF(uint32_t, fmgBaseN);
TILING_DATA_FIELD_DEF(uint32_t, fmgBaseK);
TILING_DATA_FIELD_DEF(uint32_t, fmgMTileCnt);
TILING_DATA_FIELD_DEF(uint32_t, fmgNTileCnt);
TILING_DATA_FIELD_DEF(uint32_t, fmgUseWorkspace);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(FusedMatmulGelu, FusedMatmulGeluTilingData)

struct FusedMatmulGeluCompileInfo {};

} // namespace optiling

#endif // OP_HOST_FUSED_MATMUL_GELU_TILING_H_

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
 * \file flat_quant_tiling.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_FLAT_QUANT_TILING_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_FLAT_QUANT_TILING_H

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {

struct FlatQuantCompileInfo {
  int64_t coreNum;
};

BEGIN_TILING_DATA_DEF(FlatQuantTilingData)
TILING_DATA_FIELD_DEF(uint8_t, dataType);
TILING_DATA_FIELD_DEF(int64_t, K);
TILING_DATA_FIELD_DEF(int64_t, M);
TILING_DATA_FIELD_DEF(int64_t, N);
TILING_DATA_FIELD_DEF(float, clipRatio);

TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, matmulTilingR);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, matmulTilingL);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(FlatQuant, FlatQuantTilingData)
}  // namespace optiling

#endif  // OPS_BUILT_IN_OP_TILING_RUNTIME_FLAT_QUANT_TILING_H

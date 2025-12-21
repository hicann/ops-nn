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
 * \file gelu_mul_tiling_def.h
 * \brief gelu_mul_tiling_def
 */
 
#ifndef GELU_MUL_TILING_DEF_H
#define GELU_MUL_TILING_DEF_H

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
struct GeluMulCompileInfo {};

BEGIN_TILING_DATA_DEF(GeluMulTilingData)

TILING_DATA_FIELD_DEF(int32_t, lastDimSize);
TILING_DATA_FIELD_DEF(int32_t, batchSize);
TILING_DATA_FIELD_DEF(int32_t, approximateMode);
TILING_DATA_FIELD_DEF(int32_t, PPMaxCalNum);
TILING_DATA_FIELD_DEF(uint32_t, needCoreNum);

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GeluMul, GeluMulTilingData)
}  // namespace optiling

#endif
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file embedding_tiling_simd.h
 * \brief SIMD tiling data structure and registration for Embedding operator.
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_EMBEDDING_TILING_SIMD_H
#define AIR_CXX_RUNTIME_V2_OP_IMPL_EMBEDDING_TILING_SIMD_H
#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(EmbeddingTilingDataSimdTwoDim)
TILING_DATA_FIELD_DEF(int16_t, needCoreNum);
TILING_DATA_FIELD_DEF(int32_t, indiceFactor);
TILING_DATA_FIELD_DEF(int32_t, dtypeSize);
TILING_DATA_FIELD_DEF(int64_t, gatherDimSize);
TILING_DATA_FIELD_DEF(int64_t, gatherSize);
TILING_DATA_FIELD_DEF(int64_t, innerSize);
TILING_DATA_FIELD_DEF(int64_t, blockFactor);
TILING_DATA_FIELD_DEF(int64_t, tailBlockFactor);
TILING_DATA_FIELD_DEF(int64_t, maxElement);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Embedding_1000000299, EmbeddingTilingDataSimdTwoDim)
REGISTER_TILING_DATA_CLASS(Embedding_1000000300, EmbeddingTilingDataSimdTwoDim)

} // namespace optiling
#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_EMBEDDING_TILING_SIMD_H

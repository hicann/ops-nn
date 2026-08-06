/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file embedding_dense_grad_tiling.h
 * \brief
 */
#ifndef EMBEDDING_DENSE_GRAD_TILING_H
#define EMBEDDING_DENSE_GRAD_TILING_H
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(EmbeddingDenseGradTilingData)
TILING_DATA_FIELD_DEF(uint64_t, dimSize)
TILING_DATA_FIELD_DEF(int64_t, numWeights)
TILING_DATA_FIELD_DEF(int64_t, paddingIdx)
TILING_DATA_FIELD_DEF(int32_t, scaleGradByFreq)
TILING_DATA_FIELD_DEF(uint64_t, formerBatchSize)
TILING_DATA_FIELD_DEF(uint64_t, tailBatchSize)
TILING_DATA_FIELD_DEF(uint64_t, scaleFormerCoreNum)
TILING_DATA_FIELD_DEF(uint64_t, scaleFormerBatchSize)
TILING_DATA_FIELD_DEF(uint64_t, scaleTailBatchSize)
TILING_DATA_FIELD_DEF(uint64_t, formerCoreNum)
TILING_DATA_FIELD_DEF(int64_t, ubProcessNum)
TILING_DATA_FIELD_DEF(uint64_t, scaleUbProcessNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(EmbeddingDenseGrad, EmbeddingDenseGradTilingData)

struct EmbeddingDenseGradCompileInfo {
    uint64_t totalCoreNum = 0;
    uint64_t ubSizePlatForm = 0;
};
} // namespace optiling
#endif // EMBEDDING_DENSE_GRAD_TILING_H

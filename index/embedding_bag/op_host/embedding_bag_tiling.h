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
 * \file embedding_bag_tiling.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_V2_EMBEDDING_BAG_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_V2_EMBEDDING_BAG_H
#include "register/tilingdata_base.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(EmbeddingBagTilingData)
TILING_DATA_FIELD_DEF(int64_t, formerOffsetNum);
TILING_DATA_FIELD_DEF(int64_t, tailOffsetNum);
TILING_DATA_FIELD_DEF(int64_t, numEmbeddings);
TILING_DATA_FIELD_DEF(int64_t, computeRepTime);
TILING_DATA_FIELD_DEF(int64_t, indicesMaxMoveLength);
TILING_DATA_FIELD_DEF(int64_t, usedCoreNum);
TILING_DATA_FIELD_DEF(int64_t, paddingIdx);
TILING_DATA_FIELD_DEF(int64_t, numIndices);
TILING_DATA_FIELD_DEF(int64_t, mode);
TILING_DATA_FIELD_DEF(int64_t, tilingKey);
TILING_DATA_FIELD_DEF(int64_t, hasPerSampleWeights);
/* for regbase */
TILING_DATA_FIELD_DEF(int64_t, nBags);
TILING_DATA_FIELD_DEF(int64_t, embeddingDim);
TILING_DATA_FIELD_DEF(int64_t, indiceSize);
TILING_DATA_FIELD_DEF(int64_t, rowTileNum);
TILING_DATA_FIELD_DEF(int64_t, colTileNum);
TILING_DATA_FIELD_DEF(int64_t, rowNormalNum);
TILING_DATA_FIELD_DEF(int64_t, colNormalNum);
TILING_DATA_FIELD_DEF(int64_t, rowTailNum);
TILING_DATA_FIELD_DEF(int64_t, colTailNum);
TILING_DATA_FIELD_DEF(int64_t, indicesFactor);
TILING_DATA_FIELD_DEF(int64_t, offsetsFactor);
TILING_DATA_FIELD_DEF(int64_t, weightDimFactor);
TILING_DATA_FIELD_DEF(int64_t, weightRowFactor);
TILING_DATA_FIELD_DEF(int64_t, isNeedSampleWeight);
TILING_DATA_FIELD_DEF(int64_t, indicesNumel);
TILING_DATA_FIELD_DEF(int64_t, indicesLimit);
TILING_DATA_FIELD_DEF(int64_t, sampleWeightNum);

END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(EmbeddingBag, EmbeddingBagTilingData)

ge::graphStatus TilingEmbeddingBag(gert::TilingContext* context);
ge::graphStatus TilingPrepareForEmbeddingBag(gert::TilingParseContext* context);

struct EmbeddingBagCompileInfo {
    int32_t totalCoreNum = 0;
    int64_t sysWorkspaceSize = 0;
    int64_t ubSizePlatForm = 0;
    bool isRegBase = false;
};
} // namespace optiling
#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_V2_EMBEDDING_BAG_H
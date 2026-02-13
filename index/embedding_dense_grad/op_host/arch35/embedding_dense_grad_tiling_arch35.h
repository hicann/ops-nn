/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file embedding_dense_grad_tiling_arch35.h
 * \brief embedding_dense_grad_tiling_arch35
 */

#ifndef EMBEDDING_DENSE_GRAD_TILING_ARCH35_H
#define EMBEDDING_DENSE_GRAD_TILING_ARCH35_H

#include <cstdint>
#include "tiling/tiling_api.h"
#include "register/op_impl_registry.h"

namespace optiling {


BEGIN_TILING_DATA_DEF(EmbeddingDenseGradSimdTilingData)
TILING_DATA_FIELD_DEF(int64_t, numWeights);
TILING_DATA_FIELD_DEF(int64_t, paddingIdx);
TILING_DATA_FIELD_DEF(int64_t, scaleGradByFreq);
TILING_DATA_FIELD_DEF(uint64_t, embeddingDim);
TILING_DATA_FIELD_DEF(uint32_t, indicesFactor);
TILING_DATA_FIELD_DEF(uint32_t, gradFactor);
TILING_DATA_FIELD_DEF(uint64_t, loopPerCoreIndice);
TILING_DATA_FIELD_DEF(uint64_t, loopPerCoreGrad);
TILING_DATA_FIELD_DEF(uint64_t, loopPerCoreIndiceFreq);
TILING_DATA_FIELD_DEF(uint32_t, loopPerCoreGradFreq);
TILING_DATA_FIELD_DEF(uint64_t, embeddingDimPerCore);
TILING_DATA_FIELD_DEF(uint64_t, embeddingDimLastCore);
TILING_DATA_FIELD_DEF(uint32_t, gradFactorPerRow);
TILING_DATA_FIELD_DEF(uint32_t, gradFactorPerRowTail);
TILING_DATA_FIELD_DEF(uint32_t, indicesFactorTail);
TILING_DATA_FIELD_DEF(uint32_t, indicesFactorFreq);
TILING_DATA_FIELD_DEF(uint32_t, indicesFactorFreqTail);
TILING_DATA_FIELD_DEF(uint32_t, gradFactorPerRowFreq);
TILING_DATA_FIELD_DEF(uint32_t, gradFactorPerRowTailFreq);
TILING_DATA_FIELD_DEF(uint32_t, sortSharedBufSize);
TILING_DATA_FIELD_DEF(uint32_t, baseACast);
TILING_DATA_FIELD_DEF(uint32_t, baseWCast);
TILING_DATA_FIELD_DEF(uint64_t, cntACast);
TILING_DATA_FIELD_DEF(uint64_t, cntWCast);
TILING_DATA_FIELD_DEF(uint32_t, tailACast);
TILING_DATA_FIELD_DEF(uint32_t, tailWCast);
TILING_DATA_FIELD_DEF(uint32_t, processBlock);
TILING_DATA_FIELD_DEF(uint32_t, clearBlock);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(EmbeddingDenseGrad, EmbeddingDenseGradSimdTilingData)

ge::graphStatus Tiling4EmbeddingDenseGradSimd(gert::TilingContext *context, uint32_t maxCoreNum,
                                              uint32_t ubSizePlatform, uint32_t maxThreadNum);

struct EmbeddingDenseGradACTilingParam {
    uint32_t blockDim{0};
    uint32_t clearBlockDim{0};
    uint32_t maxCoreNum{0};
    uint32_t maxThreadNum{0};
    uint32_t ubSizePlatform{0};
    int64_t numWeights{0};
    int64_t paddingIdx{0};
    int64_t tilingKey{0};
    uint32_t minBaseADim{0};
    uint32_t baseADim{0};
    uint32_t baseSDim{0};
    uint32_t baseWDim{0};
    uint64_t loopPerCoreIndice{0};
    uint64_t loopPerCoreGrad{0};
    uint64_t numelIndices{0};
    uint64_t embeddingDim{0};
    uint64_t embeddingDimPerCore{0};
    uint64_t embeddingDimLastCore{0};
    uint32_t indicesFactor{0};
    uint32_t gradFactor{0};
    uint32_t loopPerCoreRowNumTail{0};
    uint32_t indicesFactorTail{0};
    uint32_t gradFactorPerRow{0};
    uint32_t gradFactorPerRowTail{0};
    // Freq
    uint64_t loopPerCoreIndiceFreq{0};
    uint64_t loopPerCoreGradFreq{0};
    uint32_t scaleGradByFreq{0};
    uint32_t loopPerCoreRowNumFreq{0};
    uint32_t indicesFactorFreq{0};
    uint32_t gradFactorFreq{0};
    uint32_t loopPerCoreRowNumFreqTail{0};
    uint32_t indicesFactorFreqTail{0};
    uint32_t gradFactorPerRowFreq{0};
    uint32_t gradFactorPerRowTailFreq{0};
    // Cast
    uint32_t baseACast{0};
    uint32_t baseWCast{0};
    uint64_t cntACast{0};
    uint64_t cntWCast{0};
    uint32_t tailACast{0};
    uint32_t tailWCast{0};
    // Other
    uint32_t gradDtypeSize{0};
    uint32_t indicesDtypeSize{0};
    uint32_t sortSharedBufSize{0};
    uint32_t blockSize {32};
    ge::DataType gradDataType {ge::DT_FLOAT};
    bool isFullLoad{false};
};

}  // namespace optiling
#endif  // EMBEDDING_DENSE_GRAD_TILING_ARCH35_H

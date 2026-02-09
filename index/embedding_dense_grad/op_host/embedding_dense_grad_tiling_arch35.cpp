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
 * \file embedding_dense_grad_tiling_arch35.cpp
 * \brief embedding_dense_grad_tiling_arch35
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>

#include "tiling/tiling_api.h"
#include "log/log.h"
#include "util/platform_util.h"
#include "util/math_util.h"
#include "embedding_dense_grad_tiling_arch35.h"

using namespace Ops::Base;

namespace optiling {

constexpr uint32_t ASCENDC_TOOLS_WORKSPACE = 16777216;
constexpr uint32_t SORT_SHARE_BUFFER_FACTOR_2 = 512;
constexpr uint32_t SORT_STAT_PADDING = 64;
constexpr uint32_t SORT_CONDITION = 4;
constexpr uint32_t COMPUTE_INDICE = 100;
constexpr uint32_t DOUBLE_COUNT = 2;
constexpr uint32_t TENSOR_NUM = 8;
constexpr uint32_t SCALEINDEX = 2;
constexpr uint32_t TILING_UB_CUT_A = 1;
constexpr uint32_t TILING_UB_CUT_S = 2;
constexpr uint32_t TILING_UB_CUT_W = 3;
constexpr uint32_t B32_BLOCK_NUM = 8;
constexpr uint32_t MIN_CACHE_LINE_SIZE = 128;
constexpr uint32_t PROCESS_GROUP = 5;
constexpr uint32_t INDICES_TILING_LIMIT = 1024;
constexpr uint32_t INDICES_FULL_LOAD_BASE_KEY = 100;
constexpr uint32_t INDICES_FULL_LOAD_SCALE_KEY = 10;
constexpr uint32_t MIN_CLEAR_BLOCK_SIZE = 1024;

const std::map<ge::DataType, uint32_t> indicesTypeMap = {
    {ge::DT_INT32, 400},
    {ge::DT_INT64, 800},
};

const std::map<ge::DataType, uint32_t> gradTypeMap = {
        {ge::DT_FLOAT, 4}, {ge::DT_FLOAT16, 2}, {ge::DT_BF16, 1}
    };

const std::map<ge::DataType, uint32_t> gradTypeByte = {
        {ge::DT_FLOAT, 4}, {ge::DT_FLOAT16, 2}, {ge::DT_BF16, 2}
    };

static void SetTilingData(EmbeddingDenseGradSimdTilingData& tiling, EmbeddingDenseGradACTilingParam& tilingParams)
{
    tiling.set_numWeights(tilingParams.numWeights);
    tiling.set_paddingIdx(tilingParams.paddingIdx);
    tiling.set_scaleGradByFreq(tilingParams.scaleGradByFreq);
    tiling.set_embeddingDim(tilingParams.embeddingDim);
    tiling.set_indicesFactor(tilingParams.indicesFactor);
    tiling.set_gradFactor(tilingParams.gradFactor);
    tiling.set_loopPerCoreIndice(tilingParams.loopPerCoreIndice);
    tiling.set_loopPerCoreGrad(tilingParams.loopPerCoreGrad);
    tiling.set_loopPerCoreIndiceFreq(tilingParams.loopPerCoreIndiceFreq);
    tiling.set_loopPerCoreGradFreq(tilingParams.loopPerCoreGradFreq);
    tiling.set_embeddingDimPerCore(tilingParams.embeddingDimPerCore);
    tiling.set_embeddingDimLastCore(tilingParams.embeddingDimLastCore);
    tiling.set_gradFactorPerRow(tilingParams.gradFactorPerRow);
    tiling.set_gradFactorPerRowTail(tilingParams.gradFactorPerRowTail);
    tiling.set_indicesFactorTail(tilingParams.indicesFactorTail);
    tiling.set_indicesFactorFreq(tilingParams.indicesFactorFreq);
    tiling.set_indicesFactorFreqTail(tilingParams.indicesFactorFreqTail);
    tiling.set_gradFactorPerRowFreq(tilingParams.gradFactorPerRowFreq);
    tiling.set_gradFactorPerRowTailFreq(tilingParams.gradFactorPerRowTailFreq);
    tiling.set_sortSharedBufSize(tilingParams.sortSharedBufSize);
    tiling.set_baseACast(tilingParams.baseACast);
    tiling.set_baseWCast(tilingParams.baseWCast);
    tiling.set_cntACast(tilingParams.cntACast);
    tiling.set_cntWCast(tilingParams.cntWCast);
    tiling.set_tailACast(tilingParams.tailACast);
    tiling.set_tailWCast(tilingParams.tailWCast);
    tiling.set_processBlock(tilingParams.blockDim);
    tiling.set_clearBlock(tilingParams.clearBlockDim);
}

static uint32_t GetMaxSortTmpBuf(const EmbeddingDenseGradACTilingParam &tilingParams, int64_t sortDim) {
    std::vector<int64_t> shapeVec = {sortDim};
    ge::Shape srcShape(shapeVec);
    AscendC::SortConfig config;
    config.type = AscendC::SortType::RADIX_SORT;
    config.isDescend = false;
    config.hasSrcIndex = false;
    config.hasDstIndex = true;
    uint32_t maxValue = 0;
    uint32_t minValue = 0;
    ge::DataType indicesType = tilingParams.indicesDtypeSize == sizeof(int64_t) ? ge::DT_INT64 : ge::DT_INT32;
    GetSortMaxMinTmpSize(srcShape, indicesType, ge::DT_UINT32, false, config, maxValue, minValue);
    return maxValue;
}

static uint64_t CalUBTotalSize(EmbeddingDenseGradACTilingParam &tilingParams,
                               uint64_t baseADim, uint64_t baseSDim)
{
    uint32_t gradBlockAlignNum = tilingParams.blockSize / tilingParams.gradDtypeSize;
    uint32_t resBlockAlignNum = tilingParams.blockSize / sizeof(float);
    uint32_t indicesBlockAlignNum = tilingParams.blockSize / tilingParams.indicesDtypeSize;
    uint64_t gradDtypeAlign = baseSDim * CeilDiv(baseADim, static_cast<uint64_t>(gradBlockAlignNum)) * gradBlockAlignNum;
    uint64_t resB32Align = baseSDim * CeilDiv(baseADim, static_cast<uint64_t>(resBlockAlignNum)) * resBlockAlignNum;
    uint64_t indicesDtypeAlign = CeilDiv(baseSDim, static_cast<uint64_t>(indicesBlockAlignNum)) * indicesBlockAlignNum;
    uint64_t indicesSortBufCnt = 3;
    uint64_t indicesOtherBufCnt = 2;
    uint64_t sortNeedTmpSize = GetMaxSortTmpBuf(tilingParams, static_cast<int64_t>(indicesDtypeAlign));
    uint64_t oneIndicesBufSize = indicesDtypeAlign * tilingParams.indicesDtypeSize;

    uint64_t totalSize = \
        1 * gradDtypeAlign * tilingParams.gradDtypeSize +    // gradBuf
        1 * resB32Align * sizeof(float) +    // resBuf
        (indicesSortBufCnt + indicesOtherBufCnt) * oneIndicesBufSize +     // some indicesBuf
        SORT_STAT_PADDING + SORT_STAT_PADDING +   // sort padding
        sortNeedTmpSize;       // sort shared buf size

    return totalSize;
}

/**
 * @brief Find best baseSize in range [baseXoStart, baseXoEnd], use dichotomy algorithm.
 */
static uint64_t CalBestBaseSize(EmbeddingDenseGradACTilingParam &tilingParams,
                                uint64_t baseXoStart, uint64_t baseXoEnd, const uint32_t ubCutAxis)
{
    uint64_t baseXoMid;
    uint64_t tmpTotalSize = 0;
    baseXoEnd = baseXoEnd + 1;
    while (baseXoEnd - baseXoStart > 1) {
        baseXoMid = (baseXoStart + baseXoEnd) / DOUBLE_COUNT;
        if (ubCutAxis == TILING_UB_CUT_A) {
            tmpTotalSize = CalUBTotalSize(tilingParams, baseXoMid, tilingParams.baseSDim);
        } else if (ubCutAxis == TILING_UB_CUT_S) {
            tmpTotalSize = CalUBTotalSize(tilingParams, tilingParams.baseADim, baseXoMid);
        }
        if (tmpTotalSize <= tilingParams.ubSizePlatform) {
            baseXoStart = baseXoMid;
        } else {
            baseXoEnd = baseXoMid;
        }
    }
    return baseXoStart;
}

static ge::graphStatus TilingUBAndBlockBase(EmbeddingDenseGradACTilingParam &tilingParams)
{
    OP_LOGD("EmbeddingDenseGrad", "Enter TilingUBAndBlockBase.");
    tilingParams.baseWDim = tilingParams.numWeights;

    // 0. No cut
    uint64_t noCutTotalSizeTmp = CalUBTotalSize(tilingParams,
        tilingParams.embeddingDimPerCore, tilingParams.numelIndices);
    if (noCutTotalSizeTmp < tilingParams.ubSizePlatform) {
        tilingParams.baseADim = tilingParams.embeddingDimPerCore;
        tilingParams.baseSDim = tilingParams.numelIndices;
        return ge::GRAPH_SUCCESS;
    }

    // 1. Cut S
    OP_LOGD("EmbeddingDenseGrad", "Begin cut S.");
    uint64_t cutATotalSizeTmp = CalUBTotalSize(tilingParams, tilingParams.embeddingDimPerCore, PROCESS_GROUP);
    if (cutATotalSizeTmp < tilingParams.ubSizePlatform) {
        tilingParams.baseADim = tilingParams.embeddingDimPerCore;
        tilingParams.baseSDim = CalBestBaseSize(tilingParams, 1, tilingParams.numelIndices, TILING_UB_CUT_S);
        return ge::GRAPH_SUCCESS;
    }

    // 2. Cut A&S baseA=128B
    OP_LOGD("EmbeddingDenseGrad", "Begin cut A&S, baseA=128B.");
    uint64_t cutASTotalSizeTmp1 = CalUBTotalSize(tilingParams, tilingParams.minBaseADim, 1);
    if (cutASTotalSizeTmp1 < tilingParams.ubSizePlatform) {
        tilingParams.baseADim = tilingParams.minBaseADim;
        tilingParams.baseSDim = CalBestBaseSize(tilingParams, 1, tilingParams.numelIndices, TILING_UB_CUT_S);
        return ge::GRAPH_SUCCESS;
    }

    // 3. Cut A&S
    OP_LOGD("EmbeddingDenseGrad", "Begin cut A&S, baseA=8.");
    uint64_t cutASTotalSizeTmp2 = CalUBTotalSize(tilingParams, B32_BLOCK_NUM, 1);
    if (cutASTotalSizeTmp2 < tilingParams.ubSizePlatform) {
        tilingParams.baseADim = B32_BLOCK_NUM;
        tilingParams.baseSDim = CalBestBaseSize(tilingParams, 1, tilingParams.numelIndices, TILING_UB_CUT_S);
        return ge::GRAPH_SUCCESS;
    }

    OP_LOGE("EmbeddingDenseGrad", "Cal tiling failed, can not find one way to cut UB.");
    return ge::GRAPH_FAILED;
}

static void TilingUBAndBlockOther(EmbeddingDenseGradACTilingParam &tilingParams)
{
    uint32_t gradBlockAlignNum = tilingParams.blockSize / tilingParams.gradDtypeSize;
    tilingParams.gradFactor = tilingParams.baseSDim *
                              CeilDiv(tilingParams.baseADim, gradBlockAlignNum) * gradBlockAlignNum;
    tilingParams.loopPerCoreGrad = CeilDiv(tilingParams.embeddingDimPerCore, static_cast<uint64_t>(tilingParams.baseADim));
    tilingParams.gradFactorPerRow = tilingParams.baseADim;
    tilingParams.gradFactorPerRowTail = \
        tilingParams.embeddingDimPerCore - (tilingParams.loopPerCoreGrad - 1) * tilingParams.baseADim;

    tilingParams.loopPerCoreIndice = CeilDiv(tilingParams.numelIndices, static_cast<uint64_t>(tilingParams.baseSDim));
    tilingParams.loopPerCoreRowNumTail = \
        tilingParams.numelIndices - (tilingParams.loopPerCoreIndice - 1) * tilingParams.baseSDim;

    tilingParams.indicesFactor = tilingParams.baseSDim;
    tilingParams.indicesFactorTail = tilingParams.loopPerCoreRowNumTail;

    uint32_t indicesBlockAlignNum = tilingParams.blockSize / tilingParams.indicesDtypeSize;
    uint32_t indicesDtypeAlign = \
        CeilDiv(tilingParams.baseSDim, indicesBlockAlignNum) * indicesBlockAlignNum;
    tilingParams.sortSharedBufSize = GetMaxSortTmpBuf(tilingParams, indicesDtypeAlign);

    OP_LOGI("EmbeddingDenseGrad", "tilingParams: "
        "ubSizePlatform: %u, baseSDim: %u, baseADim: %u, baseWDim: %u, "
        "blockDim: %u, embeddingDim: %lu, embeddingDimPerCore: %lu, embeddingDimLastCore: %lu, "
        "gradFactor: %u, loopPerCoreIndice: %lu, indicesFactor: %u, indicesFactorTail: %u, "
        "loopPerCoreGrad: %lu, gradFactorPerRow: %u, gradFactorPerRowTail: %u, "
        "loopPerCoreRowNumTail: %u, sortSharedBufSize: %u.",
        tilingParams.ubSizePlatform, tilingParams.baseSDim, tilingParams.baseADim, tilingParams.baseWDim,
        tilingParams.blockDim, tilingParams.embeddingDim, tilingParams.embeddingDimPerCore, tilingParams.embeddingDimLastCore,
        tilingParams.gradFactor, tilingParams.loopPerCoreIndice, tilingParams.indicesFactor, tilingParams.indicesFactorTail,
        tilingParams.loopPerCoreGrad, tilingParams.gradFactorPerRow, tilingParams.gradFactorPerRowTail,
        tilingParams.loopPerCoreRowNumTail, tilingParams.sortSharedBufSize);
}

static uint64_t CalUBTotalSizeFreq(const EmbeddingDenseGradACTilingParam &tilingParams,
                                   uint64_t baseADim, uint64_t baseWeights)
{
    uint32_t weightBlockAlignNum = tilingParams.blockSize / tilingParams.gradDtypeSize;
    uint32_t indicesBlockAlignNum = tilingParams.blockSize / tilingParams.indicesDtypeSize;
    uint64_t resDtypeAlign = baseWeights * CeilDiv(baseADim, static_cast<uint64_t>(weightBlockAlignNum)) * weightBlockAlignNum;
    uint64_t freqIndicesDtypeAlign = CeilDiv(baseWeights, static_cast<uint64_t>(indicesBlockAlignNum)) * indicesBlockAlignNum;

    uint64_t totalSize = DOUBLE_COUNT * resDtypeAlign * sizeof(float) +
                         1 * freqIndicesDtypeAlign * tilingParams.indicesDtypeSize;
    return totalSize;
}

/**
 * @brief Find best baseSize in range [baseXoStart, baseXoEnd], use dichotomy algorithm.
 */
static uint64_t CalBestBaseSizeFreq(EmbeddingDenseGradACTilingParam &tilingParams,
                                    uint64_t baseXoStart, uint64_t baseXoEnd, const uint32_t ubCutAxis)
{
    uint64_t baseXoMid;
    uint64_t tmpTotalSize = 0;
    baseXoEnd = baseXoEnd + 1;
    while (baseXoEnd - baseXoStart > 1) {
        baseXoMid = (baseXoStart + baseXoEnd) / DOUBLE_COUNT;
        if (ubCutAxis == TILING_UB_CUT_A) {
            tmpTotalSize = CalUBTotalSizeFreq(tilingParams, baseXoMid, tilingParams.baseWDim);
        } else if (ubCutAxis == TILING_UB_CUT_W) {
            tmpTotalSize = CalUBTotalSizeFreq(tilingParams, tilingParams.baseADim, baseXoMid);
        }
        if (tmpTotalSize <= tilingParams.ubSizePlatform) {
            baseXoStart = baseXoMid;
        } else {
            baseXoEnd = baseXoMid;
        }
    }
    return baseXoStart;
}

static ge::graphStatus TilingUB4Freq(EmbeddingDenseGradACTilingParam &tilingParams)
{
    // 0. No cut
    uint64_t noCutTotalSizeTmp = CalUBTotalSizeFreq(tilingParams,
        tilingParams.embeddingDimPerCore, tilingParams.numWeights);
    if (noCutTotalSizeTmp < tilingParams.ubSizePlatform) {
        tilingParams.baseADim = tilingParams.embeddingDimPerCore;
        tilingParams.baseWDim = tilingParams.numWeights;
        return ge::GRAPH_SUCCESS;
    }

    // 1. Cut W
    uint64_t cutWTotalSizeTmp = CalUBTotalSizeFreq(tilingParams,
        tilingParams.embeddingDimPerCore, 1);
    if (cutWTotalSizeTmp < tilingParams.ubSizePlatform) {
        tilingParams.baseADim = tilingParams.embeddingDimPerCore;
        tilingParams.baseWDim = CalBestBaseSizeFreq(tilingParams,
            1, tilingParams.numWeights, TILING_UB_CUT_W);
        return ge::GRAPH_SUCCESS;
    }

    // 2. Cut A&W
    uint64_t cutAWTotalSizeTmp = CalUBTotalSizeFreq(tilingParams, tilingParams.minBaseADim, 1);
    if (cutAWTotalSizeTmp < tilingParams.ubSizePlatform) {
        tilingParams.baseWDim = 1;
        tilingParams.baseADim = CalBestBaseSizeFreq(tilingParams, 
            tilingParams.minBaseADim, tilingParams.embeddingDimPerCore, TILING_UB_CUT_A);
        return ge::GRAPH_SUCCESS;
    }

    OP_LOGE("EmbeddingDenseGrad", "[Freq] Cal tiling failed, can not find one way to cut UB.");
    return ge::GRAPH_FAILED;
}

static void TilingUB4FreqOther(EmbeddingDenseGradACTilingParam &tilingParams)
{
    tilingParams.gradFactorPerRowFreq = tilingParams.baseADim;
    tilingParams.loopPerCoreGradFreq = CeilDiv(tilingParams.embeddingDimPerCore, static_cast<uint64_t>(tilingParams.baseADim));
    tilingParams.gradFactorPerRowTailFreq = \
        tilingParams.embeddingDimPerCore - (tilingParams.loopPerCoreGradFreq - 1) * tilingParams.baseADim;

    tilingParams.loopPerCoreRowNumFreq = tilingParams.baseWDim;
    tilingParams.loopPerCoreIndiceFreq = CeilDiv(tilingParams.numWeights, static_cast<int64_t>(tilingParams.baseWDim));
    tilingParams.loopPerCoreRowNumFreqTail = \
        tilingParams.numWeights - (tilingParams.loopPerCoreIndiceFreq - 1) * tilingParams.baseWDim;

    tilingParams.indicesFactorFreq = tilingParams.loopPerCoreRowNumFreq;
    tilingParams.indicesFactorFreqTail = tilingParams.loopPerCoreRowNumFreqTail;

    OP_LOGI("EmbeddingDenseGrad", "[Freq]tilingParams: "
        "gradFactorPerRowFreq: %u, loopPerCoreGradFreq: %lu, gradFactorPerRowTailFreq: %u, "
        "loopPerCoreRowNumFreq: %u, loopPerCoreIndiceFreq: %lu, loopPerCoreRowNumFreqTail: %u, "
        "indicesFactorFreq: %u, indicesFactorFreqTail: %u.",
        tilingParams.gradFactorPerRowFreq, tilingParams.loopPerCoreGradFreq, tilingParams.gradFactorPerRowTailFreq,
        tilingParams.loopPerCoreRowNumFreq, tilingParams.loopPerCoreIndiceFreq, tilingParams.loopPerCoreRowNumFreqTail,
        tilingParams.indicesFactorFreq, tilingParams.indicesFactorFreqTail);
}

static uint64_t CalUBTotalSizeCast(const EmbeddingDenseGradACTilingParam &tilingParams,
                                  uint64_t baseADim, uint64_t baseWeights)
{
    uint64_t dtypeAlignNum = tilingParams.blockSize / tilingParams.gradDtypeSize;
    uint64_t gradElements = baseWeights * CeilDiv(baseADim, dtypeAlignNum) * dtypeAlignNum;

    return gradElements * sizeof(float) + gradElements * tilingParams.gradDtypeSize;
}

/**
 * @brief Find best baseSize in range [baseXoStart, baseXoEnd], use dichotomy algorithm.
 */
static uint32_t CalBestBaseSizeCast(EmbeddingDenseGradACTilingParam &tilingParams,
    uint64_t baseXoStart, uint64_t baseXoEnd, const uint32_t ubCutAxis)
{
    uint64_t baseXoMid;
    uint64_t tmpTotalSize = 0;
    baseXoEnd = baseXoEnd + 1;
    while (baseXoEnd - baseXoStart > 1) {
        baseXoMid = (baseXoStart + baseXoEnd) / DOUBLE_COUNT;
        if (ubCutAxis == TILING_UB_CUT_A) {
            tmpTotalSize = CalUBTotalSizeCast(tilingParams, baseXoMid, tilingParams.baseWDim);
        } else if (ubCutAxis == TILING_UB_CUT_W) {
            tmpTotalSize = CalUBTotalSizeCast(tilingParams, tilingParams.baseADim, baseXoMid);
        }
        if (tmpTotalSize <= tilingParams.ubSizePlatform) {
            baseXoStart = baseXoMid;
        } else {
            baseXoEnd = baseXoMid;
        }
    }
    return baseXoStart;
}

static ge::graphStatus TilingUB4Cast(EmbeddingDenseGradACTilingParam &tilingParams)
{
    // 0. No cut
    uint64_t noCutTotalSizeTmp = CalUBTotalSizeCast(tilingParams,
    tilingParams.embeddingDimPerCore, tilingParams.numWeights);
    if (noCutTotalSizeTmp < tilingParams.ubSizePlatform) {
        tilingParams.baseADim = tilingParams.embeddingDimPerCore;
        tilingParams.baseWDim = tilingParams.numWeights;
        return ge::GRAPH_SUCCESS;
    }

    // 1. Cut W
    uint64_t cutWTotalSizeTmp = CalUBTotalSizeCast(tilingParams,
        tilingParams.embeddingDimPerCore, 1);
        if (cutWTotalSizeTmp < tilingParams.ubSizePlatform) {
            tilingParams.baseADim = tilingParams.embeddingDimPerCore;
            tilingParams.baseWDim = CalBestBaseSizeCast(tilingParams,
                1, tilingParams.numWeights, TILING_UB_CUT_W);
        return ge::GRAPH_SUCCESS;
    }

    // 2. Cut A&W
    uint64_t cutAWTotalSizeTmp = CalUBTotalSizeCast(tilingParams, tilingParams.minBaseADim, 1);
    if (cutAWTotalSizeTmp < tilingParams.ubSizePlatform) {
        tilingParams.baseWDim = 1;
        tilingParams.baseADim = CalBestBaseSizeCast(tilingParams, 
            tilingParams.minBaseADim, tilingParams.embeddingDimPerCore, TILING_UB_CUT_A);
        return ge::GRAPH_SUCCESS;
    }

    OP_LOGE("EmbeddingDenseGrad", "[Freq] Cal tiling failed, can not find one way to cut UB.");
    return ge::GRAPH_FAILED;
}

static void TilingUB4CastOther(EmbeddingDenseGradACTilingParam &tilingParams)
{
    tilingParams.baseACast = tilingParams.baseADim;
    tilingParams.cntACast = CeilDiv(tilingParams.embeddingDimPerCore, static_cast<uint64_t>(tilingParams.baseACast));
    tilingParams.tailACast = \
        tilingParams.embeddingDimPerCore - (tilingParams.cntACast - 1) * tilingParams.baseACast;

    tilingParams.baseWCast = tilingParams.baseWDim;
    tilingParams.cntWCast = CeilDiv(tilingParams.numWeights, static_cast<int64_t>(tilingParams.baseWCast));
    tilingParams.tailWCast = \
        tilingParams.numWeights - (tilingParams.cntWCast - 1) * tilingParams.baseWCast;

    OP_LOGI("EmbeddingDenseGrad", "[Cast]tilingParams: "
        "baseACast: %u, baseWCast: %u, "
        "cntACast: %lu, cntWCast: %lu, "
        "tailACast: %u, tailWCast: %u.",
        tilingParams.baseACast, tilingParams.baseWCast,
        tilingParams.cntACast, tilingParams.cntWCast,
        tilingParams.tailACast, tilingParams.tailWCast);
}

static ge::graphStatus CalTilingParamForFullLoadIndices(EmbeddingDenseGradACTilingParam &tilingParams)
{
    uint32_t gradBlockAlignNum = tilingParams.blockSize / tilingParams.gradDtypeSize;
    uint32_t indicesBlockAlignNum = tilingParams.blockSize / tilingParams.indicesDtypeSize;
    indicesBlockAlignNum = CeilAlign(tilingParams.baseSDim, indicesBlockAlignNum);
    uint64_t indicesBufferNumber = 5;
    uint64_t sortNeedTmpSize = GetMaxSortTmpBuf(tilingParams, static_cast<int64_t>(indicesBlockAlignNum));
    uint64_t oneIndicesBufSize = indicesBlockAlignNum * tilingParams.indicesDtypeSize;
    int32_t remainUbSize = tilingParams.ubSizePlatform - SORT_STAT_PADDING - SORT_STAT_PADDING - sortNeedTmpSize - indicesBufferNumber * oneIndicesBufSize;
    int32_t processGradNum = FloorDiv(remainUbSize, static_cast<int32_t>(tilingParams.gradDtypeSize * DOUBLE_COUNT));
    processGradNum = FloorDiv(processGradNum, static_cast<int32_t>(tilingParams.baseSDim));
    int32_t perUbADimNum = FloorAlign(processGradNum, static_cast<int32_t>(gradBlockAlignNum));
    if (perUbADimNum < static_cast<int32_t>(tilingParams.minBaseADim) &&
        static_cast<uint64_t>(perUbADimNum) != tilingParams.embeddingDim) {
        return ge::GRAPH_FAILED;
    }

    tilingParams.baseADim = perUbADimNum;
    tilingParams.blockDim = std::min(CeilAlign(tilingParams.embeddingDim, static_cast<uint64_t>(perUbADimNum)),
                                    static_cast<uint64_t>(tilingParams.blockDim));
    tilingParams.embeddingDimPerCore = CeilDiv(tilingParams.embeddingDim, static_cast<uint64_t>(tilingParams.blockDim));
    tilingParams.embeddingDimLastCore = \
        tilingParams.embeddingDim - (tilingParams.blockDim - 1) * tilingParams.embeddingDimPerCore;

    tilingParams.gradFactor = tilingParams.baseSDim * CeilAlign(tilingParams.baseADim, gradBlockAlignNum);
    tilingParams.loopPerCoreGrad = CeilDiv(tilingParams.embeddingDimPerCore, static_cast<uint64_t>(tilingParams.baseADim));
    tilingParams.gradFactorPerRow = tilingParams.baseADim;
    tilingParams.gradFactorPerRowTail = \
        tilingParams.embeddingDimPerCore - (tilingParams.loopPerCoreGrad - 1) * tilingParams.baseADim;

    tilingParams.loopPerCoreIndice = 1; // indices 全载
    tilingParams.loopPerCoreRowNumTail = 0;

    tilingParams.indicesFactor = tilingParams.baseSDim;
    tilingParams.indicesFactorTail = 0;
    tilingParams.sortSharedBufSize = sortNeedTmpSize;

    OP_LOGI("EmbeddingDenseGrad", "tilingParams: "
        "baseSDim: %u, baseADim: %u, baseWDim: %u, blockDim: %u, "
        "embeddingDim: %lu, embeddingPerCore: %lu, embeddingLastCore: %lu, "
        "gradFactor: %u, loopPerCoreIndice: %lu, indicesFactor: %u, indicesFactorTail: %u, "
        "sortSharedBufSize: %u, loopPerCoreGrad: %lu, gradFactorPerRow: %u, gradFactorPerRowTail: %u, "
        "loopPerCoreRowNumTail: %u.",
        tilingParams.baseSDim, tilingParams.baseADim, tilingParams.baseWDim, tilingParams.blockDim,
        tilingParams.embeddingDim, tilingParams.embeddingDimPerCore, tilingParams.embeddingDimLastCore,
        tilingParams.gradFactor, tilingParams.loopPerCoreIndice, tilingParams.indicesFactor, tilingParams.indicesFactorTail,
        tilingParams.sortSharedBufSize, tilingParams.loopPerCoreGrad, tilingParams.gradFactorPerRow, tilingParams.gradFactorPerRowTail,
        tilingParams.loopPerCoreRowNumTail);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus DoOpTiling(const gert::TilingContext *context, EmbeddingDenseGradACTilingParam &tilingParams)
{
    // indices ub内全载情况
    if (tilingParams.numelIndices <= INDICES_TILING_LIMIT) {
        tilingParams.baseSDim = tilingParams.numelIndices;
        if (CalTilingParamForFullLoadIndices(tilingParams) == ge::GRAPH_SUCCESS) {
            tilingParams.isFullLoad = true;
            tilingParams.tilingKey = INDICES_FULL_LOAD_BASE_KEY;
            tilingParams.tilingKey = tilingParams.tilingKey + gradTypeMap.find(tilingParams.gradDataType)->second;
            if (tilingParams.scaleGradByFreq != static_cast<uint32_t>(0)) {
                tilingParams.tilingKey += INDICES_FULL_LOAD_SCALE_KEY;
            }

            return ge::GRAPH_SUCCESS;
        }
    }

    ge::graphStatus status = TilingUBAndBlockBase(tilingParams);
    OP_CHECK_IF(ge::GRAPH_SUCCESS != status,
                    OP_LOGE(context->GetNodeName(), "TilingUBAndBlockBase failed."),
                    return ge::GRAPH_FAILED);
    TilingUBAndBlockOther(tilingParams);

    if (tilingParams.scaleGradByFreq != static_cast<uint32_t>(0)) {
        OP_LOGI("EmbeddingDenseGrad", "Attr scale_grad_by_freq is True.");
        status = TilingUB4Freq(tilingParams);
        OP_CHECK_IF(ge::GRAPH_SUCCESS != status,
                        OP_LOGE(context->GetNodeName(), "TilingUB4Freq failed."),
                        return ge::GRAPH_FAILED);
        TilingUB4FreqOther(tilingParams);
    } else if (tilingParams.gradDataType != ge::DT_FLOAT) {
        OP_LOGI("EmbeddingDenseGrad", "Dtype is not fp32 and attr scale_grad_by_freq is False.");
        status = TilingUB4Cast(tilingParams);
        OP_CHECK_IF(ge::GRAPH_SUCCESS != status,
                        OP_LOGE(context->GetNodeName(), "TilingUB4Cast failed."),
                        return ge::GRAPH_FAILED);
        TilingUB4CastOther(tilingParams);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Tiling4EmbeddingDenseGradSimd(gert::TilingContext *context, uint32_t maxCoreNum,
                                              uint32_t ubSizePlatform, uint32_t maxThreadNum)
{
    OP_LOGD(context->GetNodeName(), "Tiling4EmbeddingDenseGradSimd is begin");
    EmbeddingDenseGradSimdTilingData tiling;
    EmbeddingDenseGradACTilingParam tilingParams;

    tilingParams.ubSizePlatform = ubSizePlatform;
    tilingParams.maxCoreNum = maxCoreNum;
    tilingParams.maxThreadNum = maxThreadNum;
    tilingParams.blockSize = GetUbBlockSize(context);

    auto gradTensor = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradTensor);
    const gert::Shape gradShape = gradTensor->GetStorageShape();
    int64_t gradDims = gradShape.GetDimNum();
    OP_CHECK_IF(gradDims <= 0,
                    OP_LOGE(context->GetNodeName(),
                        "GradDim should bigger than 0."),
                    return ge::GRAPH_FAILED);

    auto indicesTensor = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, indicesTensor);
    const gert::Shape indicesShape = indicesTensor->GetStorageShape();
    int64_t indicesDims = indicesShape.GetDimNum();

    uint32_t numelIndices = indicesShape.GetShapeSize();
    tilingParams.numelIndices = numelIndices;

    uint64_t gradShapeMuls = 1;
    uint64_t indicesShapeMuls = 1;
    for (uint32_t i = 0; i < gradDims - 1; i++) {
        gradShapeMuls *= gradShape.GetDim(i);
    }
    for (uint32_t i = 0; i < indicesDims; i++) {
        indicesShapeMuls *= indicesShape.GetDim(i);
    }
    OP_CHECK_IF(gradShapeMuls != indicesShapeMuls,
                    OP_LOGE(context->GetNodeName(),
                        "The dim product of indices shape is not equal to first several dim of grad."),
                    return ge::GRAPH_FAILED);

    uint64_t embeddingDim = gradShape.GetDim(gradDims - 1);
    tilingParams.embeddingDim = embeddingDim;

    auto indicesTensorDesc = context->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, indicesTensorDesc);
    auto indicesDataType = indicesTensorDesc->GetDataType();
    OP_CHECK_IF(indicesTypeMap.count(indicesDataType) == 0,
                    OP_LOGE(context->GetNodeName(), "Check DataType failed"),
                    return ge::GRAPH_FAILED);
    tilingParams.tilingKey += indicesTypeMap.find(indicesDataType)->second;
    tilingParams.indicesDtypeSize = indicesTypeMap.find(indicesDataType)->second / COMPUTE_INDICE;

    auto gradTensorDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradTensorDesc);
    tilingParams.gradDataType = gradTensorDesc->GetDataType();
    OP_CHECK_IF(gradTypeMap.count(tilingParams.gradDataType) == 0,
                    OP_LOGE(context->GetNodeName(), "Check DataType failed"),
                    return ge::GRAPH_FAILED);
    tilingParams.tilingKey += gradTypeMap.find(tilingParams.gradDataType)->second;
    tilingParams.gradDtypeSize = gradTypeByte.find(tilingParams.gradDataType)->second;

    auto const attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const auto* numWeights = attrs->GetAttrPointer<int64_t>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, numWeights);
    tilingParams.numWeights = *numWeights;
    const auto* paddingIdx = attrs->GetAttrPointer<int64_t>(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, paddingIdx);
    tilingParams.paddingIdx = *paddingIdx;
    const bool* scaleGradByFreq = attrs->GetAttrPointer<bool>(SCALEINDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, scaleGradByFreq);
    tilingParams.scaleGradByFreq = *scaleGradByFreq ? 1 : 0;

    // 按照128B分核
    tilingParams.minBaseADim = MIN_CACHE_LINE_SIZE / tilingParams.gradDtypeSize;
    tilingParams.embeddingDimPerCore = CeilDiv(tilingParams.embeddingDim, static_cast<uint64_t>(tilingParams.maxCoreNum));
    tilingParams.embeddingDimPerCore = std::max(static_cast<uint64_t>(tilingParams.minBaseADim), tilingParams.embeddingDimPerCore);
    tilingParams.embeddingDimPerCore = std::min(tilingParams.embeddingDim, tilingParams.embeddingDimPerCore);
    tilingParams.blockDim = static_cast<uint32_t>(CeilDiv(tilingParams.embeddingDim, tilingParams.embeddingDimPerCore));
    tilingParams.embeddingDimLastCore = \
        tilingParams.embeddingDim - (tilingParams.blockDim - 1) * tilingParams.embeddingDimPerCore;

    uint64_t totalOutputNum = tilingParams.numWeights * tilingParams.embeddingDim;
    uint64_t clearBaseBlock = MIN_CLEAR_BLOCK_SIZE / tilingParams.gradDtypeSize;
    totalOutputNum = CeilDiv(totalOutputNum, clearBaseBlock);
    uint64_t clearBlockNum = CeilDiv(totalOutputNum, static_cast<uint64_t>(tilingParams.maxCoreNum));
    tilingParams.clearBlockDim = CeilDiv(totalOutputNum, clearBlockNum);

    if (DoOpTiling(context, tilingParams) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    uint32_t usedCoreNum = std::max(tilingParams.clearBlockDim, tilingParams.blockDim);
    context->SetBlockDim(usedCoreNum);
    context->SetLocalMemorySize(tilingParams.ubSizePlatform);
    context->SetTilingKey(tilingParams.tilingKey);
    context->SetScheduleMode(1);

    SetTilingData(tiling, tilingParams);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t usrSize = (tilingParams.isFullLoad == false) ? tilingParams.numWeights * tilingParams.indicesDtypeSize + \
                     tilingParams.numWeights * tilingParams.embeddingDim * sizeof(float) : 0;
    size_t *workspace = context->GetWorkspaceSizes(1);
    workspace[0] = usrSize + ASCENDC_TOOLS_WORKSPACE;
    OP_LOGD(context->GetNodeName(), "Tiling4EmbeddingDenseGradSimd is end");
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

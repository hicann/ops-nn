/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file elu_grad_v2_tiling.cpp
 */
#include <algorithm>
#include <cstdint>
 
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/elu_grad_v2_tiling_data.h"
#include "../op_kernel/elu_grad_v2_tiling_key.h"
 
namespace optiling {
namespace {
constexpr uint32_t FLOAT32_SIZE = 4U;
constexpr uint32_t FLOAT16_SIZE = 2U;
constexpr uint32_t MAX_VECTOR_CORE_NUM = 64U;
constexpr uint32_t BLOCK_SIZE = ELU_GRAD_V2_BLOCK_BYTES;
constexpr uint32_t PIPE_QUEUE_COUNT = 3U;
constexpr uint32_t CHUNK_MASK_BUFFER_NUM = 1U;
constexpr uint32_t TARGET_PIPE_TILE_NUM_FLOAT_EXP = 4U;
constexpr uint32_t TARGET_PIPE_TILE_NUM_FLOAT_RESULT = 3U;
constexpr uint32_t TARGET_PIPE_TILE_NUM_FLOAT16_EXP = 2U;
constexpr uint32_t TARGET_PIPE_TILE_NUM_FLOAT16_RESULT = 3U;
constexpr uint32_t TARGET_PIPE_TILE_NUM_BFLOAT16 = 2U;
constexpr uint32_t TARGET_PIPE_TILE_NUM_MIXED_FAST = 2U;
constexpr uint32_t TARGET_PIPE_TILE_NUM_FLOAT16_FAST = 2U;
constexpr uint32_t SAFE_FLOAT16_GENERIC_EXP_TILE_CAP = 20480U;
constexpr uint32_t MIN_PIPE_TILE_NUM = 2U;
constexpr uint32_t SMALL_FAST_PATH_LIMIT = ELU_GRAD_V2_CORE_CHUNK;
constexpr uint32_t MIXED_SINGLE_TILE_LIMIT = 2U * ELU_GRAD_V2_CORE_CHUNK;
constexpr uint32_t WORKSPACE_SLOTS = 1U;
constexpr float ATTR_COMPARE_EPS = 1e-3F;
 
struct EluGradV2CompileInfo {
};
 
uint32_t GetTypeSize(ge::DataType dt)
{
    return dt == ge::DT_FLOAT ? FLOAT32_SIZE : FLOAT16_SIZE;
}
 
uint32_t AlignDown(uint32_t value, uint32_t factor)
{
    if (factor == 0U) {
        return value;
    }
    return (value / factor) * factor;
}
 
uint32_t AlignUp(uint32_t value, uint32_t factor)
{
    if (factor == 0U) {
        return value;
    }
    return ((value + factor - 1U) / factor) * factor;
}
 
uint32_t CeilDiv(uint32_t value, uint32_t divisor)
{
    if (divisor == 0U) {
        return 0U;
    }
    return (value + divisor - 1U) / divisor;
}
 
uint32_t Max(uint32_t lhs, uint32_t rhs)
{
    return lhs > rhs ? lhs : rhs;
}
 
uint32_t Min(uint32_t lhs, uint32_t rhs)
{
    return lhs < rhs ? lhs : rhs;
}
 
uint32_t GetElemsPerBlock(uint32_t typeSize)
{
    if (typeSize == 0U) {
        return 0U;
    }
    return BLOCK_SIZE / typeSize;
}

bool IsNearlyEqual(float lhs, float rhs)
{
    float diff = lhs > rhs ? (lhs - rhs) : (rhs - lhs);
    return diff <= ATTR_COMPARE_EPS;
}

bool IsFloat16AlignedMixedSignFastPath(
    [[maybe_unused]] ge::DataType dt,
    [[maybe_unused]] uint32_t totalLength,
    [[maybe_unused]] float alpha,
    [[maybe_unused]] float scale,
    [[maybe_unused]] float inputScale,
    [[maybe_unused]] bool isResult)
{
    if (dt != ge::DT_FLOAT16) {
        return false;
    }
    if (isResult || !IsNearlyEqual(alpha, 1.0F) || !IsNearlyEqual(scale, 1.0F) || !IsNearlyEqual(inputScale, 1.0F)) {
        return false;
    }
    const uint32_t blockElems = GetElemsPerBlock(GetTypeSize(dt));
    if (blockElems == 0U) {
        return false;
    }
    const bool isAligned = (totalLength % blockElems) == 0U;
    return isAligned && totalLength >= (4U * ELU_GRAD_V2_CORE_CHUNK);
}

bool IsMixedAlignedFastPath([[maybe_unused]] ge::DataType dt, [[maybe_unused]] uint32_t totalLength)
{
    if (dt != ge::DT_BF16) {
        return false;
    }
    const uint32_t blockElems = GetElemsPerBlock(GetTypeSize(dt));
    if (blockElems == 0U) {
        return false;
    }
    const bool isAligned = (totalLength % blockElems) == 0U;
    return isAligned && totalLength >= (4U * ELU_GRAD_V2_CORE_CHUNK);
}

uint32_t GetChunkFloatBufferNum(ge::DataType dt, bool isResult, bool isMixedAlignedFast, bool isFloat16AlignedFast)
{
    if (isMixedAlignedFast || isFloat16AlignedFast) {
        return 2U;
    }
    if (dt == ge::DT_FLOAT) {
        return 1U;
    }
    if (dt == ge::DT_FLOAT16) {
        return 3U;
    }
    return 3U;
}
 
uint32_t SelectComputeChunk(ge::DataType dt, bool isResult)
{
    if (dt == ge::DT_FLOAT) {
        return isResult ? ELU_GRAD_V2_FLOAT_RESULT_COMPUTE_CHUNK : ELU_GRAD_V2_FLOAT_EXP_COMPUTE_CHUNK;
    }
    return isResult ? ELU_GRAD_V2_MIXED_RESULT_COMPUTE_CHUNK : ELU_GRAD_V2_MIXED_EXP_COMPUTE_CHUNK;
}
 
uint32_t GetTargetPipeTileNum(ge::DataType dt, bool isResult, bool isMixedAlignedFast, bool isFloat16AlignedFast)
{
    if (isFloat16AlignedFast) {
        return TARGET_PIPE_TILE_NUM_FLOAT16_FAST;
    }
    if (isMixedAlignedFast) {
        return TARGET_PIPE_TILE_NUM_MIXED_FAST;
    }
    if (dt == ge::DT_FLOAT) {
        return isResult ? TARGET_PIPE_TILE_NUM_FLOAT_RESULT : TARGET_PIPE_TILE_NUM_FLOAT_EXP;
    }
     if (dt == ge::DT_FLOAT16) {
         return isResult ? TARGET_PIPE_TILE_NUM_FLOAT16_RESULT : TARGET_PIPE_TILE_NUM_FLOAT16_EXP;
     }
     return TARGET_PIPE_TILE_NUM_BFLOAT16;
}
 
uint32_t GetMaxTileDataNum(
    uint64_t ubSize, uint32_t typeSize, uint32_t bufferNum, uint32_t chunkFloatBufferNum, uint32_t computeChunk)
{
    const uint32_t blockElems = GetElemsPerBlock(typeSize);
    if (blockElems == 0U || bufferNum == 0U) {
        return 0U;
    }
 
    const uint64_t chunkScratchBytes =
        static_cast<uint64_t>(chunkFloatBufferNum) * computeChunk * sizeof(float) +
        static_cast<uint64_t>(CHUNK_MASK_BUFFER_NUM) * computeChunk * sizeof(uint8_t) +
        ELU_GRAD_V2_RESERVED_UB_BYTES;
    const uint64_t queueBytesPerElem = static_cast<uint64_t>(bufferNum) * PIPE_QUEUE_COUNT * typeSize;
 
    if (ubSize <= chunkScratchBytes || queueBytesPerElem == 0U) {
        return blockElems;
    }
 
    uint64_t queueCapacity = (ubSize - chunkScratchBytes) / queueBytesPerElem;
    if (queueCapacity == 0U) {
        return blockElems;
    }
 
    return Max(AlignDown(static_cast<uint32_t>(queueCapacity), blockElems), blockElems);
}
 
 } // namespace
 
static ge::graphStatus TilingParseForEluGradV2([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}
 
static ge::graphStatus TilingFuncEluGradV2(gert::TilingContext* context)
{
    auto* platformInfo = context->GetPlatformInfo();
    auto* inputShape = context->GetInputShape(0);
    auto* inputDesc = context->GetInputDesc(0);
    auto* outputDesc = context->GetOutputDesc(0);
    auto* attrs = context->GetAttrs();

    if (platformInfo == nullptr) {
        OP_LOGE(context, "TilingFuncEluGradV2: platformInfo is nullptr.");
        return ge::GRAPH_FAILED;
    }

    if (inputShape == nullptr) {
        OP_LOGE(context, "TilingFuncEluGradV2: inputShape is nullptr.");
        return ge::GRAPH_FAILED;
    }    

    if (inputDesc == nullptr) {
        OP_LOGE(context, "TilingFuncEluGradV2: inputDesc is nullptr.");
        return ge::GRAPH_FAILED;
    }      

    if (outputDesc == nullptr) {
        OP_LOGE(context, "TilingFuncEluGradV2: outputDesc is nullptr.");
        return ge::GRAPH_FAILED;
    } 

    if (attrs == nullptr) {
        OP_LOGE(context, "TilingFuncEluGradV2: attrs is nullptr.");
        return ge::GRAPH_FAILED;
    } 
     
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint64_t ubSize = 0U;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
 
    uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();
    coreNum = std::min(coreNum, MAX_VECTOR_CORE_NUM);
    if (coreNum == 0U) {
        OP_LOGE(context, "TilingFuncEluGradV2: coreNum is 0.");
        return ge::GRAPH_FAILED;
    }
 
    const float alpha = static_cast<float>(*(attrs->GetFloat(0)));
    const float scale = static_cast<float>(*(attrs->GetFloat(1)));
    const float inputScale = static_cast<float>(*(attrs->GetFloat(2)));
    const bool isResult = static_cast<bool>(*(attrs->GetBool(3)));
    uint32_t totalLength = inputShape->GetStorageShape().GetShapeSize();
    const ge::DataType inputDataType = inputDesc->GetDataType();
    const ge::DataType outputDataType = outputDesc->GetDataType();
    const bool isMixedOutput = isResult && (inputDataType != outputDataType);
    uint32_t typeSize = GetTypeSize(inputDataType);
    if (isMixedOutput) {
        typeSize = Max(typeSize, GetTypeSize(outputDataType));
    }
    if (typeSize == 0U) {
        OP_LOGE(context, "TilingFuncEluGradV2: typeSize is 0, unsupported input dtype.");
        return ge::GRAPH_FAILED;
    }
    const bool isFloat16AlignedFast =
        IsFloat16AlignedMixedSignFastPath(inputDataType, totalLength, alpha, scale, inputScale, isResult);
    const bool isMixedAlignedFast = IsMixedAlignedFastPath(inputDataType, totalLength);
    uint32_t schMode = isResult ? ELU_GRAD_V2_TPL_SCH_MODE_SAFE_RESULT : ELU_GRAD_V2_TPL_SCH_MODE_SAFE_EXP;
    const uint32_t computeChunk = SelectComputeChunk(inputDataType, isResult);
 
    auto* tiling = context->GetTilingData<EluGradV2TilingData>();
    size_t* workspace = context->GetWorkspaceSizes(WORKSPACE_SLOTS);
    if (tiling == nullptr) {
        OP_LOGE(context, "TilingFuncEluGradV2: tiling is nullptr.");
        return ge::GRAPH_FAILED;
    }

    if (workspace == nullptr) {
        OP_LOGE(context, "TilingFuncEluGradV2: workspace is nullptr.");
        return ge::GRAPH_FAILED;
    }
     
    workspace[0] = 0U;
 
    if (totalLength == 0U) {
        tiling->totalLength = 0U;
        tiling->tileDataNum = computeChunk;
        tiling->coreNum = 1U;
        tiling->bigCoreNum = 0U;
        tiling->bigCoreDataNum = 0U;
        tiling->smallCoreDataNum = 0U;
        tiling->lastCoreDataNum = 0U;
        tiling->bufferOpen = 0U;
        tiling->computeChunk = computeChunk;
        tiling->alpha = alpha;
        tiling->scale = scale;
        tiling->inputScale = inputScale;
        tiling->isResult = static_cast<uint8_t>(isResult);
        context->SetTilingKey(GET_TPL_TILING_KEY(schMode));
        context->SetBlockDim(1U);
        return ge::GRAPH_SUCCESS;
    }
 
    const uint32_t blockElems = GetElemsPerBlock(typeSize);
    if (blockElems == 0U) {
        OP_LOGE(context, "TilingFuncEluGradV2: blockElems is 0.");
        return ge::GRAPH_FAILED;
    }
    const uint32_t coreDispatchChunk = ELU_GRAD_V2_CORE_CHUNK;
    uint32_t usedCoreNum = std::min(coreNum, CeilDiv(totalLength, coreDispatchChunk));
    uint64_t totalBytes = static_cast<uint64_t>(totalLength) * typeSize;
    uint32_t totalBlockNum = static_cast<uint32_t>((totalBytes + BLOCK_SIZE - 1U) / BLOCK_SIZE);
    usedCoreNum = Min(usedCoreNum, Max(totalBlockNum, 1U));
    usedCoreNum = Max(usedCoreNum, 1U);
    const uint32_t avgCoreData = CeilDiv(totalLength, usedCoreNum);
    const uint32_t chunkFloatBufferNum =
        GetChunkFloatBufferNum(inputDataType, isResult, isMixedAlignedFast, isFloat16AlignedFast);
     const uint32_t maxTileDataNumDouble =
        GetMaxTileDataNum(ubSize, typeSize, ELU_GRAD_V2_MAX_BUFFER_NUM, chunkFloatBufferNum, computeChunk);
     const uint32_t maxTileDataNumSingle =
        GetMaxTileDataNum(ubSize, typeSize, 1U, chunkFloatBufferNum, computeChunk);
 
    if (totalLength <= SMALL_FAST_PATH_LIMIT) {
        schMode = isResult ? ELU_GRAD_V2_TPL_SCH_MODE_SINGLE_TILE_RESULT
                            : ELU_GRAD_V2_TPL_SCH_MODE_SINGLE_TILE_EXP;
        uint32_t tileDataNum = Max(AlignUp(totalLength, blockElems), blockElems);
        tileDataNum = Min(tileDataNum, maxTileDataNumSingle);
 
        tiling->totalLength = totalLength;
        tiling->tileDataNum = tileDataNum;
        tiling->coreNum = 1U;
        tiling->bigCoreNum = 0U;
        tiling->bigCoreDataNum = 0U;
        tiling->smallCoreDataNum = 0U;
        tiling->lastCoreDataNum = totalLength;
        tiling->bufferOpen = 0U;
        tiling->computeChunk = computeChunk;
        tiling->alpha = alpha;
        tiling->scale = scale;
        tiling->inputScale = inputScale;
        tiling->isResult = static_cast<uint8_t>(isResult);
        context->SetTilingKey(GET_TPL_TILING_KEY(schMode));
        context->SetBlockDim(1U);
        return ge::GRAPH_SUCCESS;
    }
 
    const bool isMixedType = inputDataType != ge::DT_FLOAT;
    const uint32_t targetPipeTileNum =
        GetTargetPipeTileNum(inputDataType, isResult, isMixedAlignedFast, isFloat16AlignedFast);

    uint32_t bufferOpen = 0U;
    uint32_t targetTileNum = 1U;
    const bool isDedicatedFastPath = isMixedAlignedFast || isFloat16AlignedFast;
    const bool needsSingleChunkGenericTile = !isDedicatedFastPath;
    const bool preferSingleTile =
        (!isDedicatedFastPath && inputDesc->GetDataType() == ge::DT_BF16 && isResult &&
            avgCoreData <= maxTileDataNumSingle) ||
        (!isDedicatedFastPath && isMixedType && avgCoreData <= MIXED_SINGLE_TILE_LIMIT);
    if (!needsSingleChunkGenericTile && !preferSingleTile && avgCoreData >= targetPipeTileNum * ELU_GRAD_V2_CORE_CHUNK) {
        bufferOpen = 1U;
        targetTileNum = targetPipeTileNum;
    } else if (!needsSingleChunkGenericTile && !preferSingleTile && !isMixedType &&
                avgCoreData >= MIN_PIPE_TILE_NUM * ELU_GRAD_V2_CORE_CHUNK) {
        bufferOpen = 1U;
        targetTileNum = MIN_PIPE_TILE_NUM;
    }
 
    uint32_t tileDataNum = AlignUp(CeilDiv(avgCoreData, targetTileNum), blockElems);
    tileDataNum = Max(tileDataNum, blockElems);
    if (needsSingleChunkGenericTile) {
        const bool preferDoubleBuffer = (inputDataType != ge::DT_FLOAT) || !isResult;
        bufferOpen = (preferDoubleBuffer && maxTileDataNumDouble >= computeChunk) ? 1U : 0U;
        tileDataNum = (bufferOpen != 0U) ? Min(computeChunk, maxTileDataNumDouble)
                                         : Min(computeChunk, maxTileDataNumSingle);
    } else if (bufferOpen != 0U) {
        tileDataNum = Min(tileDataNum, maxTileDataNumDouble);
    } else {
        tileDataNum = Min(AlignUp(avgCoreData, blockElems), maxTileDataNumSingle);
    }
    tileDataNum = Max(tileDataNum, blockElems);
 
    const uint32_t smallCoreBlockNum = totalBlockNum / usedCoreNum;
    const uint32_t bigCoreNum = totalBlockNum % usedCoreNum;
    const uint32_t bigCoreBlockNum = smallCoreBlockNum + (bigCoreNum > 0U ? 1U : 0U);
    const uint32_t smallCoreDataNum = smallCoreBlockNum * blockElems;
    const uint32_t bigCoreDataNum = bigCoreBlockNum * blockElems;
 
    uint32_t lastCoreStart = 0U;
    if (usedCoreNum > 1U) {
        uint32_t lastCoreIdx = usedCoreNum - 1U;
        if (lastCoreIdx < bigCoreNum) {
            lastCoreStart = lastCoreIdx * bigCoreDataNum;
        } else {
            lastCoreStart = bigCoreNum * bigCoreDataNum + (lastCoreIdx - bigCoreNum) * smallCoreDataNum;
        }
    }
    uint32_t lastCoreDataNum = totalLength > lastCoreStart ? totalLength - lastCoreStart : 0U;
    if (usedCoreNum == 1U) {
        lastCoreDataNum = totalLength;
    }
 
    const uint32_t biggestCoreDataNum = bigCoreNum > 0U ? bigCoreDataNum : smallCoreDataNum;
    if (bufferOpen != 0U && CeilDiv(biggestCoreDataNum, tileDataNum) < MIN_PIPE_TILE_NUM) {
        bufferOpen = 0U;
        tileDataNum = Min(AlignUp(avgCoreData, blockElems), maxTileDataNumSingle);
        tileDataNum = Max(tileDataNum, blockElems);
    }

    const bool isSingleTileDirect =
        (bufferOpen == 0U) && (tileDataNum >= biggestCoreDataNum) && (avgCoreData <= ELU_GRAD_V2_CORE_CHUNK);
    if (isSingleTileDirect) {
        schMode = isResult ? ELU_GRAD_V2_TPL_SCH_MODE_SINGLE_TILE_RESULT
                           : ELU_GRAD_V2_TPL_SCH_MODE_SINGLE_TILE_EXP;
    } else if (isFloat16AlignedFast) {
        schMode = ELU_GRAD_V2_TPL_SCH_MODE_FLOAT16_EXP_FAST;
    } else if (isMixedAlignedFast) {
        schMode = isResult ? ELU_GRAD_V2_TPL_SCH_MODE_MIXED_RESULT_FAST : ELU_GRAD_V2_TPL_SCH_MODE_MIXED_EXP_FAST;
    }
 
    tiling->totalLength = totalLength;
    tiling->tileDataNum = tileDataNum;
    tiling->coreNum = usedCoreNum;
    tiling->bigCoreNum = bigCoreNum;
    tiling->bigCoreDataNum = bigCoreDataNum;
    tiling->smallCoreDataNum = smallCoreDataNum;
    tiling->lastCoreDataNum = lastCoreDataNum;
    tiling->bufferOpen = bufferOpen;
    tiling->computeChunk = computeChunk;
    tiling->alpha = alpha;
    tiling->scale = scale;
    tiling->inputScale = inputScale;
    tiling->isResult = static_cast<uint8_t>(isResult);
 
    context->SetTilingKey(GET_TPL_TILING_KEY(schMode));
    context->SetBlockDim(usedCoreNum);
    return ge::GRAPH_SUCCESS;
}
 
IMPL_OP_OPTILING(EluGradV2).Tiling(TilingFuncEluGradV2).TilingParse<EluGradV2CompileInfo>(TilingParseForEluGradV2);
} // namespace optiling
 

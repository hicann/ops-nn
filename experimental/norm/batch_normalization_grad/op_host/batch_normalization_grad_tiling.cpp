/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Pei Haobo<@xiaopei-1>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file batch_normalization_grad_tiling.cpp
 * \brief BatchNormalizationGrad算子的tiling(分块)策略实现
 *
 * 本文件提供tiling逻辑，将计算任务按channel维度划分到多个AI Core上并行执行，
 * 支持整空间模式和空间分块模式。整空间模式下最大化 UB 利用率；当 numSpatial 超
 * 过 UB 容量时自动降级为空间分块模式。
 */

#include "log/log.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "register/op_impl_registry.h"
#include "../op_kernel/batch_normalization_grad_tiling_data.h"
#include "../op_kernel/batch_normalization_grad_tiling_key.h"

namespace optiling {

const uint32_t BLOCK_SIZE = 32;
const uint32_t BUFFER_NUM = 2;

// ---- ReduceSum 临时缓冲区大小计算 ----
// sharedTmpBuffer 最小空间 = ceil(count/repeat) * block * sizeof(T)
// 其中 repeat = 256/sizeof(T), block = 32/sizeof(T)
static uint64_t CalcReduceUb(uint64_t elemCount, uint64_t elemBytes)
{
    const uint64_t elemPerRepeat = 256u / elemBytes;
    const uint64_t elemPerBlock  = 32u / elemBytes;
    const uint64_t firstRepeats  = (elemCount + elemPerRepeat - 1u) / elemPerRepeat;
    const uint64_t alignedRepeats = ((firstRepeats + elemPerBlock - 1u) / elemPerBlock) * elemPerBlock;
    return alignedRepeats * elemPerBlock * elemBytes;
}

// ---- 整空间维度模式 UB 需求 ----
// 当 numSpatial 较小时，所有 batch × spatial 可一次装入 UB
static uint64_t CalcWholeSpatialUb(uint64_t paddedSpatial, uint64_t batchesPerTile, uint32_t typeLength)
{
    uint64_t tileElems = paddedSpatial * batchesPerTile;
    const uint64_t tqueUb = 3ULL * BUFFER_NUM * tileElems * typeLength;       // 3 TQue × 2-depth
    const uint64_t tbufUb = 6ULL * tileElems * sizeof(float);                  // 6 TBuf，全 float
    const uint64_t reduceUb = CalcReduceUb(tileElems, sizeof(float));
    const uint64_t scalarUb = 8ULL * sizeof(float);                            // scalarBuf = 32B
    return tqueUb + tbufUb + reduceUb + scalarUb;
}

// ---- 空间分块模式单 tile UB 需求 ----
// 当 numSpatial 超过 UB 容量时，查询单个 spatial tile 所需 UB
static uint64_t CalcSpatialTileUb(uint64_t tileSpatial, uint32_t typeLength, uint64_t batchesPerTile)
{
    const uint64_t tqueUb = 3ULL * BUFFER_NUM * tileSpatial * typeLength;     // TQue 按 tile 大小分配
    const uint64_t tbufUb = 6ULL * tileSpatial * sizeof(float);                // TBuf 按 tile 大小分配
    const uint64_t reduceUb = CalcReduceUb(tileSpatial, sizeof(float));
    const uint64_t scalarUb = 8ULL * sizeof(float);                            // scalarBuf 固定 32B
    const uint64_t batchScalarUb = 2ULL * batchesPerTile * sizeof(float);      // batchGBuf + batchGXhatBuf
    const uint64_t safetyUb = 2048ULL;                                         // 安全余量
    return tqueUb + tbufUb + reduceUb + scalarUb + batchScalarUb + safetyUb;
}

// ---- 二分搜索最大空间 tile 大小 ----
static uint64_t SearchSpatialTileSize(uint64_t numSpatial, uint32_t typeLength,
                                      uint64_t batchesPerTile, uint64_t ubLength,
                                      uint64_t elementsPerBlock)
{
    const uint64_t alignElems = elementsPerBlock;
    uint64_t lo = alignElems;
    uint64_t hi = numSpatial;
    uint64_t best = 0;

    while (lo <= hi) {
        uint64_t mid = lo + (hi - lo) / 2u;
        mid = (mid / alignElems) * alignElems;
        if (mid == 0) mid = alignElems;
        if (mid > numSpatial) mid = (numSpatial / alignElems) * alignElems;
        if (mid == 0) break;

        uint64_t alignedMid = ((mid + alignElems - 1u) / alignElems) * alignElems;
        if (CalcSpatialTileUb(alignedMid, typeLength, batchesPerTile) <= ubLength) {
            best = mid;
            lo = mid + alignElems;
        } else {
            if (mid <= alignElems) break;
            hi = mid - alignElems;
        }
    }
    return best;
}

struct BatchNormalizationGradCompileInfo {
};

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);

    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus BatchNormalizationGradTilingFunc(gert::TilingContext* context)
{
    // 1. 获取平台运行时信息
    uint64_t ubSize;
    int64_t hardwareCoreNum;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, hardwareCoreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    // 2. 获取数据类型
    auto dt = context->GetInputDesc(0)->GetDataType();
    uint32_t typeLength = 0;
    ge::TypeUtils::GetDataTypeLength(dt, typeLength);
    uint32_t elementsPerBlock = BLOCK_SIZE / typeLength;

    // 3. 解析 shape [N, C, H, W, ...]
    auto inputShape = context->GetInputShape(0)->GetStorageShape();
    uint32_t numDims = inputShape.GetDimNum();
    if (numDims < 2) return ge::GRAPH_FAILED;

    uint32_t numBatches  = static_cast<uint32_t>(inputShape.GetDim(0));
    uint32_t numChannels = static_cast<uint32_t>(inputShape.GetDim(1));
    uint32_t numSpatial  = 1;
    for (uint32_t i = 2; i < numDims; ++i) {
        numSpatial *= static_cast<uint32_t>(inputShape.GetDim(i));
    }
    if (numBatches == 0 || numChannels == 0 || numSpatial == 0) return ge::GRAPH_FAILED;

    uint32_t m = numBatches * numSpatial;
    float mFloat = static_cast<float>(m);

    // 4. 多核 channel 分配
    uint32_t coreNum = 1;
    bool useMultiCore = false;
    if (numChannels > 1) {
        coreNum = std::min(static_cast<uint32_t>(hardwareCoreNum), numChannels);
        if (coreNum > 1) useMultiCore = true;
    }
    uint32_t channelsPerCore = numChannels / coreNum;
    uint32_t tailChannels     = numChannels % coreNum;
    uint32_t maxChannelsPerCore = channelsPerCore + (tailChannels > 0 ? 1 : 0);
    uint32_t alignedCPC = ((maxChannelsPerCore + 7) / 8) * 8;

    // 5. 跨 batch stride 分析
    uint32_t batchByteSpan = numSpatial * typeLength;
    uint32_t paddedBatchBytes = ((batchByteSpan + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
    uint32_t paddedSpatial = paddedBatchBytes / typeLength;
    bool batchStrideAligned = ((numChannels * numSpatial * typeLength) % BLOCK_SIZE) == 0;
    bool needPerBatchLoad = ((uint64_t)(numChannels - 1) * numSpatial * typeLength > 65535);

    // 6. 精确 UB 计算：多核 accumBuf
    uint64_t accumUb = useMultiCore ? (alignedCPC * 2 * sizeof(float)) : 0;

    // 7. 第一步：整空间模式可行性检查
    uint64_t hotTbufUb   = 6ULL * paddedSpatial * sizeof(float);
    uint64_t hotTqueUb   = 3ULL * BUFFER_NUM * paddedSpatial * typeLength;
    uint64_t fixedUb     = hotTbufUb + hotTqueUb + 8ULL * sizeof(float) + accumUb + 2048ULL;
    uint64_t remainForBatchUb = (ubSize > fixedUb) ? (ubSize - fixedUb) : 0;
    uint64_t perBatchUb  = hotTbufUb + hotTqueUb;
    uint64_t maxBatchesByUB = (perBatchUb > 0) ? (remainForBatchUb / perBatchUb + 1) : 1;
    if (maxBatchesByUB == 0) maxBatchesByUB = 1;

    uint32_t batchesPerTile;
    if (batchStrideAligned) {
        batchesPerTile = std::min(numBatches, static_cast<uint32_t>(maxBatchesByUB));
    } else {
        if (numSpatial <= 64) {
            batchesPerTile = std::min(numBatches, static_cast<uint32_t>(maxBatchesByUB));
        } else {
            batchesPerTile = 1;
        }
    }
    uint32_t batchGroups = (numBatches + batchesPerTile - 1) / batchesPerTile;
    uint32_t tailBatches = numBatches % batchesPerTile;
    if (tailBatches == 0) tailBatches = batchesPerTile;

    // 8. 第二步：空间分块判定
    uint32_t spatialSplitMode = 0;
    uint32_t spatialTileSize  = numSpatial;
    uint32_t spatialTileNum   = 1;
    uint32_t spatialTailSize  = numSpatial;

    uint32_t alignedSpatial = ((numSpatial + elementsPerBlock - 1) / elementsPerBlock) * elementsPerBlock;
    if (alignedSpatial == 0) alignedSpatial = elementsPerBlock;

    uint64_t wholeUb = CalcWholeSpatialUb(paddedSpatial, batchesPerTile, typeLength);
    if (accumUb > 0) wholeUb += accumUb + 1024ULL;

    if (wholeUb > ubSize) {
        spatialSplitMode = 1;
        batchesPerTile = 1;
        batchGroups = numBatches;
        tailBatches = 1;

        uint64_t maxTile = SearchSpatialTileSize(numSpatial, typeLength, batchesPerTile, ubSize, elementsPerBlock);
        if (maxTile == 0) {
            OP_LOGE(context, "no valid spatial tile for ubSize=%lu S=%u dtypeBytes=%u", ubSize, numSpatial, typeLength);
            return ge::GRAPH_FAILED;
        }
        spatialTileSize = static_cast<uint32_t>(maxTile);
        spatialTileSize = (spatialTileSize / elementsPerBlock) * elementsPerBlock;
        if (spatialTileSize == 0) spatialTileSize = elementsPerBlock;

        spatialTileNum  = numSpatial / spatialTileSize;
        spatialTailSize = numSpatial % spatialTileSize;
        if (spatialTailSize > 0) {
            spatialTileNum += 1;
        } else {
            spatialTailSize = spatialTileSize;
        }
    }

    // 9. 获取 epsilon
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    const float *epsilonPoint = attrs->GetAttrPointer<float>(0);
    const float epsilon = (epsilonPoint != nullptr) ? *epsilonPoint : 1e-5f;

    // 10. 填充 tiling 数据
    BatchNormalizationGradTilingData* tiling = context->GetTilingData<BatchNormalizationGradTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    OP_CHECK_IF(
        memset_s(tiling, sizeof(BatchNormalizationGradTilingData), 0, sizeof(BatchNormalizationGradTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"),
        return ge::GRAPH_FAILED);

    tiling->numChannels = numChannels;
    tiling->numBatches = numBatches;
    tiling->numSpatial = numSpatial;
    tiling->m = m;
    tiling->mFloat = mFloat;
    tiling->coreNum = coreNum;
    tiling->channelsPerCore = channelsPerCore;
    tiling->tailChannels = tailChannels;
    tiling->spatialSplitMode = spatialSplitMode;
    tiling->spatialTileSize = spatialTileSize;
    tiling->spatialTileNum = spatialTileNum;
    tiling->spatialTailSize = spatialTailSize;
    tiling->batchStrideAligned = batchStrideAligned ? 1 : 0;
    tiling->batchesPerTile = batchesPerTile;
    tiling->batchGroups = batchGroups;
    tiling->tailBatches = tailBatches;
    tiling->paddedSpatial = paddedSpatial;
    tiling->epsilon = epsilon;
    tiling->blockSize = BLOCK_SIZE;
    tiling->typeLength = typeLength;
    tiling->elementsPerBlock = elementsPerBlock;
    tiling->needPerBatchLoad = needPerBatchLoad ? 1 : 0;

    context->SetBlockDim(coreNum);

    // 11. 数据类型合法性校验，避免非法 dtype 触发后续除零等异常。
    if (dt != ge::DT_FLOAT && dt != ge::DT_FLOAT16 && dt != ge::DT_BF16) {
        OP_LOGE(context, "get dtype error");
        return ge::GRAPH_FAILED;
    }
    context->SetTilingKey(GET_TPL_TILING_KEY(BN_GRAD_TPL_SCH_MODE_0));

    // 12. workspace：多核模式下各核通道不重叠，无需 workspace
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForBatchNormalizationGrad([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(BatchNormalizationGrad).Tiling(BatchNormalizationGradTilingFunc).TilingParse<BatchNormalizationGradCompileInfo>(TilingParseForBatchNormalizationGrad);
} // namespace optiling

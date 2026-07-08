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
 * \file cross_entropy_grad_tiling.cpp
 * \brief CrossEntropyGrad算子的tiling(分块)策略实现
 *
 * 本文件提供tiling逻辑，将2D计算任务按行和列划分为较小的块在AI Core上并行执行。
 * 采用基于UB容量约束的二分搜索求解最优tile行数和列数。
 * 当rowLen过大导致UB无法容纳完整行时，自动启用列分割模式。
 */

#include "log/log.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "../op_kernel/cross_entropy_grad_tiling_data.h"
#include "../op_kernel/cross_entropy_grad_tiling_key.h"

namespace optiling {

constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t MAX_TILE_ROWS = 256;
constexpr uint32_t WS_SYS_SIZE = 0U;

struct TilingCache {
    uint32_t rowLen;
    uint32_t totalRows;
    uint32_t typeBytes;
    uint32_t targetBytes;
    uint32_t blockDim;
    uint32_t tilingKey;
    CrossEntropyGradTilingData data;
    bool valid;
};

static thread_local TilingCache g_tilingCache = {0, 0, 0, 0, 0, 0, {}, false};

static uint32_t CalcUbUsageFull(uint32_t trn, uint32_t effectiveColLen, uint32_t typeBytes, uint32_t targetBytes)
{
    uint32_t colBytesAlign32 = ((effectiveColLen * typeBytes + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
    uint32_t colAlign32 = colBytesAlign32 / typeBytes;

    uint32_t elementsPerBlock = BLOCK_SIZE / typeBytes;
    uint32_t elementsPerRepeat = 256 / typeBytes;
    uint32_t firstMaxRepeat = (colAlign32 + elementsPerRepeat - 1) / elementsPerRepeat;
    uint32_t iter1OutputCount = firstMaxRepeat * 2;
    uint32_t workLocalElements = ((iter1OutputCount + elementsPerBlock - 1) / elementsPerBlock) * elementsPerBlock;
    uint32_t workLocalSize = ((workLocalElements * typeBytes + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
    uint32_t fixed = workLocalSize * 2;

    uint32_t total = fixed;
    total += 2u * trn * colBytesAlign32;
    total += 2u * trn * colBytesAlign32;
    total += 2u * ((trn * targetBytes + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
    total += trn * colBytesAlign32;
    total += ((trn * typeBytes + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
    total += ((trn * typeBytes + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
    return total;
}

// bf16 UB计算：Queue用bf16(2B)，临时缓冲用float(4B)
static uint32_t CalcUbUsageBf16(uint32_t trn, uint32_t effectiveColLen)
{
    uint32_t colBytesAlignBf16 = ((effectiveColLen * 2 + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
    uint32_t colBytesAlignF32 = ((effectiveColLen * 4 + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
    uint32_t colAlign32F32 = colBytesAlignF32 / 4;

    uint32_t elementsPerBlock = BLOCK_SIZE / 4;
    uint32_t elementsPerRepeat = 256 / 4;
    uint32_t firstMaxRepeat = (colAlign32F32 + elementsPerRepeat - 1) / elementsPerRepeat;
    uint32_t iter1OutputCount = firstMaxRepeat * 2;
    uint32_t workLocalElements = ((iter1OutputCount + elementsPerBlock - 1) / elementsPerBlock) * elementsPerBlock;
    uint32_t workLocalSize = ((workLocalElements * 4 + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
    uint32_t fixed = workLocalSize * 2;

    uint32_t total = fixed;
    total += 2u * trn * colBytesAlignBf16;                                         // logits queue bf16 (双缓冲)
    total += 2u * trn * colBytesAlignBf16;                                         // grad queue bf16 (双缓冲)
    total += 2u * ((trn * 4 + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;         // target queue int32 (双缓冲)
    total += trn * colBytesAlignF32;                                               // logitsFloatBuf (单缓冲)
    total += trn * colBytesAlignF32;                                               // gradFloatBuf (单缓冲)
    total += ((trn * 4 + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;              // maxBuf float (单缓冲)
    total += ((trn * 4 + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;              // sumBuf float (单缓冲)
    return total;
}

struct CrossEntropyGradCompileInfo {
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

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CrossEntropyGradTilingFunc(gert::TilingContext* context)
{
    // 1. 获取平台运行时信息
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    // 2. 获取输入信息
    auto dt = context->GetInputDesc(0)->GetDataType();
    auto targetDt = context->GetInputDesc(1)->GetDataType();
    auto inputShape = context->GetInputShape(0)->GetStorageShape();

    uint32_t typeBytes = 0;
    ge::TypeUtils::GetDataTypeLength(dt, typeBytes);
    uint32_t targetBytes = 0;
    ge::TypeUtils::GetDataTypeLength(targetDt, targetBytes);

    uint32_t dimNum = inputShape.GetDimNum();
    uint32_t totalRows = 1;
    uint32_t rowLen = 1;

    if (dimNum >= 2) {
        for (uint32_t i = 0; i < dimNum - 1; i++) {
            totalRows *= inputShape.GetDim(i);
        }
        rowLen = inputShape.GetDim(dimNum - 1);
    } else {
        totalRows = 1;
        rowLen = inputShape.GetShapeSize();
    }

    // 3. 获取WorkspaceSize
    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    // 4. 尝试命中 tiling 缓存
    // 数据类型合法性校验，避免非法 dtype 触发后续除零等异常。
    if (dt != ge::DT_FLOAT && dt != ge::DT_FLOAT16 && dt != ge::DT_BF16) {
        OP_LOGE(context, "unsupported dtype combination");
        return ge::GRAPH_FAILED;
    }

    auto& cache = g_tilingCache;
    if (cache.valid &&
        cache.rowLen == rowLen &&
        cache.totalRows == totalRows &&
        cache.typeBytes == typeBytes &&
        cache.targetBytes == targetBytes) {
        context->SetTilingKey(cache.tilingKey);
        context->SetBlockDim(cache.blockDim);

        CrossEntropyGradTilingData* tiling = context->GetTilingData<CrossEntropyGradTilingData>();
        OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
        *tiling = cache.data;

        return ge::GRAPH_SUCCESS;
    }

    // 5. Step 1: 正常模式 — 全行宽二分搜索最大 tileRowNum
    bool isBf16 = (dt == ge::DT_BF16);
    uint32_t tileRowNum = 0;
    {
        uint32_t lo = 1, hi = totalRows;
        while (lo < hi) {
            uint32_t mid = lo + (hi - lo + 1) / 2;
            uint32_t ubUsed = isBf16 ? CalcUbUsageBf16(mid, rowLen) : CalcUbUsageFull(mid, rowLen, typeBytes, targetBytes);
            if (ubUsed <= ubSize) lo = mid;
            else hi = mid - 1;
        }
        tileRowNum = lo;
        uint32_t ubUsed = isBf16 ? CalcUbUsageBf16(tileRowNum, rowLen) : CalcUbUsageFull(tileRowNum, rowLen, typeBytes, targetBytes);
        if (ubUsed > ubSize) {
            tileRowNum = 0;
        }
    }

    // 6. Step 2: 列分割模式 — 当一行都放不下时
    uint32_t colSplitMode = 0;
    uint32_t tileCols = rowLen;
    uint32_t numColPasses = 1;

    if (tileRowNum == 0) {
        colSplitMode = 1;

        uint32_t clo = 1, chi = rowLen;
        while (clo < chi) {
            uint32_t cmid = clo + (chi - clo + 1) / 2;
            uint32_t ubUsed = isBf16 ? CalcUbUsageBf16(1, cmid) : CalcUbUsageFull(1, cmid, typeBytes, targetBytes);
            if (ubUsed <= ubSize) clo = cmid;
            else chi = cmid - 1;
        }
        tileCols = clo;

        uint32_t alignUnit = isBf16 ? (BLOCK_SIZE / 2) : (BLOCK_SIZE / typeBytes);
        tileCols = (tileCols / alignUnit) * alignUnit;
        if (tileCols == 0) tileCols = alignUnit;

        numColPasses = (rowLen + tileCols - 1) / tileCols;

        uint32_t lo = 1, hi = totalRows;
        while (lo < hi) {
            uint32_t mid = lo + (hi - lo + 1) / 2;
            uint32_t ubUsed = isBf16 ? CalcUbUsageBf16(mid, tileCols) : CalcUbUsageFull(mid, tileCols, typeBytes, targetBytes);
            if (ubUsed <= ubSize) lo = mid;
            else hi = mid - 1;
        }
        tileRowNum = lo;
    }

    tileRowNum = (tileRowNum > MAX_TILE_ROWS) ? MAX_TILE_ROWS : tileRowNum;

    // 7. 核间负载均衡
    uint32_t usedCoreNum = (coreNum < totalRows) ? coreNum : totalRows;
    usedCoreNum = (usedCoreNum >= 1) ? usedCoreNum : 1;

    uint32_t baseRowsPerCore = totalRows / usedCoreNum;
    uint32_t bigCoreNum = totalRows % usedCoreNum;

    uint32_t smallCoreRowNum = baseRowsPerCore;
    uint32_t bigCoreRowNum = baseRowsPerCore + 1;

    uint32_t smallCoreTileNum = 0;
    uint32_t smallTailRowNum = 0;
    if (smallCoreRowNum > 0) {
        smallCoreTileNum = smallCoreRowNum / tileRowNum;
        smallTailRowNum = smallCoreRowNum % tileRowNum;
        if (smallTailRowNum > 0) {
            smallCoreTileNum += 1;
        }
    }

    uint32_t bigCoreTileNum = 0;
    uint32_t bigTailRowNum = 0;
    if (bigCoreRowNum > 0) {
        bigCoreTileNum = bigCoreRowNum / tileRowNum;
        bigTailRowNum = bigCoreRowNum % tileRowNum;
        if (bigTailRowNum > 0) {
            bigCoreTileNum += 1;
        }
    }

    // 8. 设置tiling数据
    CrossEntropyGradTilingData* tiling = context->GetTilingData<CrossEntropyGradTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    tiling->rowLen = rowLen;
    tiling->totalRows = totalRows;
    tiling->smallCoreRowNum = smallCoreRowNum;
    tiling->bigCoreRowNum = bigCoreRowNum;
    tiling->bigCoreNum = bigCoreNum;
    tiling->tileRowNum = tileRowNum;
    tiling->smallCoreTileNum = smallCoreTileNum;
    tiling->bigCoreTileNum = bigCoreTileNum;
    tiling->smallTailRowNum = smallTailRowNum;
    tiling->bigTailRowNum = bigTailRowNum;
    tiling->colSplitMode = colSplitMode;
    tiling->tileCols = tileCols;
    tiling->numColPasses = numColPasses;
    tiling->dataTypeId = dt;
    tiling->typeBytes = typeBytes;

    context->SetBlockDim(usedCoreNum);
    context->SetTilingKey(GET_TPL_TILING_KEY(CROSS_ENTROPY_GRAD_TPL_SCH_MODE_0));

    // 写入缓存
    cache.rowLen = rowLen;
    cache.totalRows = totalRows;
    cache.typeBytes = typeBytes;
    cache.targetBytes = targetBytes;
    cache.blockDim = usedCoreNum;
    cache.tilingKey = GET_TPL_TILING_KEY(CROSS_ENTROPY_GRAD_TPL_SCH_MODE_0);
    cache.data = *tiling;
    cache.valid = true;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForCrossEntropyGrad([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(CrossEntropyGrad).Tiling(CrossEntropyGradTilingFunc).TilingParse<CrossEntropyGradCompileInfo>(TilingParseForCrossEntropyGrad);
} // namespace optiling

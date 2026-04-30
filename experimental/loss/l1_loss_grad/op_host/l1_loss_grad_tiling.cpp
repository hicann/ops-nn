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
 * \file l1_loss_grad_tiling.cpp
 * \brief
 */
#include "log/log.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "graph/utils/type_utils.h"
#include "op_common/op_host/util/platform_util.h"
#include <cstring>
#include "../op_kernel/l1_loss_grad_tiling_data.h"
#include "../op_kernel/l1_loss_grad_tiling_key.h"

namespace optiling {

#define UB_BLOCK_NUM_FLOAT 12U
#define UB_BLOCK_NUM_NARROW 16U

struct L1LossGradCompileInfo {
    int32_t totalCoreNum = 0;
    int64_t ubSize = 0;
    bool isRegbase = false;
};

static ge::graphStatus TilingParseForL1LossGrad([[maybe_unused]] gert::TilingParseContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CalculateCoreBlockNums(
    uint64_t inputLengthAlgin, uint32_t blockSize, int64_t coreNum, uint64_t tileBlockNum, uint64_t inputBytes,
    uint64_t tileDataNum, uint64_t& smallCoreDataNum, uint64_t& bigCoreDataNum, uint64_t& smallTailDataNum,
    uint64_t& bigTailDataNum, uint64_t& finalSmallTileNum, uint64_t& finalBigTileNum, uint64_t& tailBlockNum)
{
    // Declare safe fallback constants to satisfy static divide-by-zero checks.
    const uint32_t safeBlockSize = (blockSize == 0) ? 1U : blockSize;
    const uint64_t safeCoreNum = (coreNum <= 0) ? 1U : static_cast<uint64_t>(coreNum);
    const uint64_t safeTileBlockNum = (tileBlockNum == 0) ? 1U : tileBlockNum;
    const uint64_t safeInputBytes = (inputBytes == 0) ? 1U : inputBytes;

    uint64_t everyCoreInputBlockNum = inputLengthAlgin / safeBlockSize / safeCoreNum;
    tailBlockNum = (inputLengthAlgin / safeBlockSize) % safeCoreNum;

    smallCoreDataNum = everyCoreInputBlockNum * safeBlockSize / safeInputBytes;
    uint64_t smallTileNum = everyCoreInputBlockNum / safeTileBlockNum;
    finalSmallTileNum = (everyCoreInputBlockNum % safeTileBlockNum) == 0 ? smallTileNum : smallTileNum + 1;
    smallTailDataNum = smallCoreDataNum - (tileDataNum * smallTileNum);
    smallTailDataNum = smallTailDataNum == 0 ? tileDataNum : smallTailDataNum;

    everyCoreInputBlockNum += 1;
    bigCoreDataNum = everyCoreInputBlockNum * safeBlockSize / safeInputBytes;
    uint64_t bigTileNum = everyCoreInputBlockNum / safeTileBlockNum;
    finalBigTileNum = (everyCoreInputBlockNum % safeTileBlockNum) == 0 ? bigTileNum : bigTileNum + 1;
    bigTailDataNum = bigCoreDataNum - tileDataNum * bigTileNum;
    bigTailDataNum = bigTailDataNum == 0 ? tileDataNum : bigTailDataNum;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus L1LossGradTilingFunc(gert::TilingContext* context)
{
    constexpr uint64_t RESERVED_UB_SIZE = 64U * 1024U;
    constexpr uint64_t FAST_PATH_BYTES = 64U * 1024U;
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    auto inputShapePtr = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShapePtr);
    OP_CHECK_IF(
        context->GetPlatformInfo() == nullptr, OP_LOGE(context, "context->GetPlatformInfo() is nullptr"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        context->GetInputDesc(1) == nullptr, OP_LOGE(context, "context->GetInputDesc(1) is nullptr"),
        return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();

    uint64_t inputNum = inputShapePtr->GetStorageShape().GetShapeSize();
    OP_CHECK_IF(inputNum == 0, OP_LOGE(context, "inputNum is 0"), return ge::GRAPH_FAILED);

    auto inputDataType = context->GetInputDesc(1)->GetDataType();
    uint64_t inputBytes = 0;
    uint32_t currentUbBlockNum = UB_BLOCK_NUM_FLOAT;
    if (inputDataType == ge::DT_FLOAT) {
        inputBytes = sizeof(float);
        currentUbBlockNum = UB_BLOCK_NUM_FLOAT;
    } else if (inputDataType == ge::DT_FLOAT16) {
        inputBytes = sizeof(uint16_t);
        currentUbBlockNum = UB_BLOCK_NUM_NARROW;
    } else if (inputDataType == ge::DT_BF16) {
        inputBytes = sizeof(uint16_t);
        currentUbBlockNum = UB_BLOCK_NUM_NARROW;
    } else {
        OP_LOGE(context, "unsupported data type for L1LossGrad");
        return ge::GRAPH_FAILED;
    }

    uint64_t inputLength = inputNum * inputBytes;
    uint32_t blockSize = Ops::Base::GetUbBlockSize(context);
    OP_CHECK_IF(blockSize == 0, OP_LOGE(context, "blockSize is 0"), return ge::GRAPH_FAILED);
    const uint32_t safeBlockSize = (blockSize == 0) ? 1U : blockSize;
    uint64_t inputLengthAlgin = ((inputLength + safeBlockSize - 1) / safeBlockSize) * safeBlockSize;

    bool isMeanReduction = true;
    auto* attrs = context->GetAttrs();
    if (attrs != nullptr && attrs->GetAttrNum() > 0) {
        const char* reductionStr = attrs->GetAttrPointer<char>(0);
        if (reductionStr != nullptr &&
            (std::strcmp(reductionStr, "sum") == 0 || std::strcmp(reductionStr, "none") == 0)) {
            isMeanReduction = false;
        }
    }
    float scaleValue = isMeanReduction ? (1.0f / static_cast<float>(inputNum)) : 1.0f;

    int64_t aivNum = ascendcPlatform.GetCoreNumAiv();
    if (aivNum == 0) {
        aivNum = ascendcPlatform.GetCoreNum();
    }
    OP_CHECK_IF(aivNum <= 0, OP_LOGE(context, "coreNum <= 0"), return ge::GRAPH_FAILED);

    int64_t optimalCoreNum = static_cast<int64_t>((inputNum + 1023) / 1024);
    if (optimalCoreNum == 0) {
        optimalCoreNum = 1;
    }

    uint64_t totalBlocks = inputLengthAlgin / safeBlockSize;
    if (totalBlocks == 0) {
        totalBlocks = 1;
    }

    int64_t coreNum = (optimalCoreNum < aivNum) ? optimalCoreNum : aivNum;
    coreNum = (static_cast<uint64_t>(coreNum) < totalBlocks) ? coreNum : static_cast<int64_t>(totalBlocks);

    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(
        ubSize <= RESERVED_UB_SIZE,
        OP_LOGE(context, "ubSize is too small after reserving profile headroom, ubSize=%lu", ubSize),
        return ge::GRAPH_FAILED);
    ubSize -= RESERVED_UB_SIZE;

    uint64_t tileBlockNum = (ubSize / safeBlockSize) / currentUbBlockNum;
    OP_CHECK_IF(tileBlockNum == 0, OP_LOGE(context, "tileBlockNum is 0"), return ge::GRAPH_FAILED);

    uint64_t tileDataNum = (tileBlockNum * safeBlockSize) / inputBytes;
    OP_CHECK_IF(tileDataNum == 0, OP_LOGE(context, "tileDataNum is 0"), return ge::GRAPH_FAILED);

    uint64_t smallCoreDataNum = 0;
    uint64_t bigCoreDataNum = 0;
    uint64_t smallTailDataNum = 0;
    uint64_t bigTailDataNum = 0;
    uint64_t finalSmallTileNum = 0;
    uint64_t finalBigTileNum = 0;
    uint64_t tailBlockNum = 0;

    CalculateCoreBlockNums(
        inputLengthAlgin, blockSize, coreNum, tileBlockNum, inputBytes, tileDataNum, smallCoreDataNum, bigCoreDataNum,
        smallTailDataNum, bigTailDataNum, finalSmallTileNum, finalBigTileNum, tailBlockNum);

    L1LossGradTilingData* tilingData = context->GetTilingData<L1LossGradTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tilingData);
    *tilingData = {};
    tilingData->smallCoreDataNum = smallCoreDataNum;
    tilingData->bigCoreDataNum = bigCoreDataNum;
    tilingData->tileDataNum = tileDataNum;
    tilingData->smallTailDataNum = smallTailDataNum;
    tilingData->bigTailDataNum = bigTailDataNum;
    tilingData->finalSmallTileNum = finalSmallTileNum;
    tilingData->finalBigTileNum = finalBigTileNum;
    tilingData->tailBlockNum = tailBlockNum;
    tilingData->scaleValue = scaleValue;

    context->SetBlockDim(coreNum);
    uint64_t tilingKey = GET_TPL_TILING_KEY(0);
    context->SetTilingKey(tilingKey);

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    if (currentWorkspace != nullptr) {
        if (inputLengthAlgin <= FAST_PATH_BYTES) {
            currentWorkspace[0] = sysWorkspaceSize;
        } else {
            currentWorkspace[0] = 16 * 1024 * 1024 + sysWorkspaceSize;
        }
    }

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(L1LossGrad).Tiling(L1LossGradTilingFunc).TilingParse<L1LossGradCompileInfo>(TilingParseForL1LossGrad);

} // namespace optiling
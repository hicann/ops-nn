/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cctype>
#include <string>
#include "log/log.h"
#include "util/math_util.h"
#include "op_host/tiling_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "op_host/tiling_templates_registry.h"
#include "../op_kernel/smooth_l1_loss_grad_v2_tiling_data.h"
#include "../op_kernel/smooth_l1_loss_grad_v2_tiling_key.h"
#include "util/platform_util.h"

namespace optiling {
using namespace Ops::NN::OpTiling;

#define UB_DATA_NUM_FLOAT 10U
#define UB_DATA_NUM_OTHER 12U
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t WS_SYS_SIZE = 0;
struct SmoothL1LossGradV2CompileInfo {
};

static ge::graphStatus TilingParseForSmoothL1LossGradV2([[maybe_unused]] gert::TilingParseContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OP_LOGE(context, "platform info is nullptr"), return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    coreNum = ascendcPlatform.GetCoreNum();
    OP_CHECK_IF(coreNum <= 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ubSize <= 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    size_t usrSize = 0;
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OP_LOGE(context, "platform info is nullptr"), return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_IF(
        currentWorkspace == nullptr, OP_LOGE(context, "workspace array is nullptr"), return ge::GRAPH_FAILED); //
    currentWorkspace[0] = usrSize + sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

static std::string GetReductionAttr(gert::TilingContext* context)
{
    std::string reduction = "sum";
    if (context == nullptr)
        return reduction;
    auto attrs = context->GetAttrs();
    if (attrs != nullptr) {
        auto attr = attrs->GetStr(1); // sigma(0), reduction(1)
        if (attr != nullptr) {
            reduction = *attr;
        }
    }
    std::transform(reduction.begin(), reduction.end(), reduction.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return reduction;
}

static float GetSigmaAttr(gert::TilingContext* context)
{
    float sigma = 1.0f;
    if (context == nullptr)
        return sigma;
    auto attrs = context->GetAttrs();
    if (attrs != nullptr) {
        auto attr = attrs->GetFloat(0);
        if (attr != nullptr) {
            sigma = *attr;
        }
    }
    return sigma;
}

static int GetReductionType(const std::string& reduction)
{
    if (reduction == "none" || reduction == "n")
        return 0;
    if (reduction == "mean" || reduction == "m")
        return 1;
    if (reduction == "sum" || reduction == "s")
        return 2;
    return -1;
}

static ge::graphStatus GetShapeAttrsInfo(
    gert::TilingContext* context, uint32_t blockSize, uint64_t ubSize, uint64_t& inputNum, uint64_t& inputBytes,
    uint64_t& tileBlockNum, uint64_t& tileDataNum, uint64_t& inputLengthAlgin32)
{
    OP_CHECK_IF(blockSize == 0, OP_LOGE(context, "blockSize is 0"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    auto shapeInputPtr = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, shapeInputPtr);
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_IF(inputDesc == nullptr, OP_LOGE(context, "input desc is nullptr"), return ge::GRAPH_FAILED);

    inputNum = shapeInputPtr->GetStorageShape().GetShapeSize();
    uint32_t typeLength = 0;
    ge::TypeUtils::GetDataTypeLength(inputDesc->GetDataType(), typeLength);
    uint64_t inputLength = inputNum * typeLength;
    if (inputNum == 0) {
        OP_LOGE(context, "inputNum is 0");
        return ge::GRAPH_FAILED;
    }
    inputBytes = inputLength / inputNum;
    if (inputBytes == 0) {
        OP_LOGE(context, "inputBytes is 0");
        return ge::GRAPH_FAILED;
    }
    uint64_t ubDataNumber = (inputDesc->GetDataType() == ge::DT_FLOAT) ? UB_DATA_NUM_FLOAT : UB_DATA_NUM_OTHER;
    OP_CHECK_IF(ubDataNumber == 0, OP_LOGE(context, "ubDataNumber is 0"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(BUFFER_NUM == 0, OP_LOGE(context, "BUFFER_NUM is 0"), return ge::GRAPH_FAILED);
    if (blockSize == 0) {
        OP_LOGE(context, "blockSize is 0");
        return ge::GRAPH_FAILED;
    } else {
        tileBlockNum = (ubSize / blockSize) / ubDataNumber / BUFFER_NUM;
    }
    if (tileBlockNum == 0) {
        OP_LOGE(context, "tileBlockNum is 0");
        return ge::GRAPH_FAILED;
    }
    tileDataNum = (tileBlockNum * blockSize) / inputBytes;
    if (tileDataNum == 0) {
        OP_LOGE(context, "tileDataNum is 0");
        return ge::GRAPH_FAILED;
    }

    // 限制不超过基于 UB 容量的最大安全值
    uint64_t queueBytesPerElem = BUFFER_NUM * (inputBytes * 4);
    uint64_t calcBytesPerElem = 3 * sizeof(float) + sizeof(uint8_t);
    uint64_t totalBytesPerElem = queueBytesPerElem + calcBytesPerElem;
    uint64_t maxSafeElem = ubSize / totalBytesPerElem;
    if (maxSafeElem == 0)
        maxSafeElem = 1;
    if (tileDataNum > maxSafeElem) {
        tileDataNum = maxSafeElem;
    }
    if (blockSize == 0) {
        OP_LOGE(context, "blockSize is 0");
        return ge::GRAPH_FAILED;
    } else {
        tileBlockNum = (ubSize / blockSize) / ubDataNumber / BUFFER_NUM;
    }
    inputLengthAlgin32 = (((inputLength + blockSize - 1) / blockSize) * blockSize);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CalculateCoreBlockNums(
    gert::TilingContext* context, uint32_t blockSize, uint64_t inputLengthAlgin32, int64_t coreNum,
    uint64_t tileBlockNum, uint64_t inputBytes, uint64_t tileDataNum, uint64_t& smallCoreDataNum,
    uint64_t& bigCoreDataNum, uint64_t& smallTailDataNum, uint64_t& bigTailDataNum, uint64_t& finalSmallTileNum,
    uint64_t& finalBigTileNum, uint64_t& tailBlockNum)
{
    if (blockSize == 0) {
        OP_LOGE(context, "blockSize is 0");
        return ge::GRAPH_FAILED;
    }
    if (coreNum == 0) {
        OP_LOGE(context, "coreNum is 0");
        return ge::GRAPH_FAILED;
    }
    if (tileBlockNum == 0) {
        OP_LOGE(context, "tileBlockNum is 0 in CalculateCoreBlockNums");
        return ge::GRAPH_FAILED;
    }
    if (inputBytes == 0) {
        OP_LOGE(context, "inputBytes is 0 in CalculateCoreBlockNums");
        return ge::GRAPH_FAILED;
    }
    uint64_t everyCoreInputBlockNum = inputLengthAlgin32 / blockSize / coreNum;
    tailBlockNum = (inputLengthAlgin32 / blockSize) % coreNum;
    smallCoreDataNum = everyCoreInputBlockNum * blockSize / inputBytes;
    uint64_t smallTileNum = everyCoreInputBlockNum / tileBlockNum;
    finalSmallTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? smallTileNum : smallTileNum + 1;
    smallTailDataNum = smallCoreDataNum - (tileDataNum * smallTileNum);
    smallTailDataNum = smallTailDataNum == 0 ? tileDataNum : smallTailDataNum;

    everyCoreInputBlockNum += 1;
    bigCoreDataNum = everyCoreInputBlockNum * blockSize / inputBytes;
    uint64_t bigTileNum = everyCoreInputBlockNum / tileBlockNum;
    finalBigTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? bigTileNum : bigTileNum + 1;
    bigTailDataNum = bigCoreDataNum - tileDataNum * bigTileNum;
    bigTailDataNum = bigTailDataNum == 0 ? tileDataNum : bigTailDataNum;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SmoothL1LossGradV2TilingFunc(gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    uint32_t blockSize = Ops::Base::GetUbBlockSize(context);
    OP_CHECK_IF(blockSize == 0, OP_LOGE(context, "blockSize is 0"), return ge::GRAPH_FAILED);
    SmoothL1LossGradV2TilingData* tiling = context->GetTilingData<SmoothL1LossGradV2TilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(SmoothL1LossGradV2TilingData), 0, sizeof(SmoothL1LossGradV2TilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    uint64_t ubSize;
    int64_t coreNum;
    ge::graphStatus ret = GetPlatformInfo(context, ubSize, coreNum);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    uint64_t inputNum, inputBytes, tileBlockNum, tileDataNum, inputLengthAlgin32;
    ret = GetShapeAttrsInfo(
        context, blockSize, ubSize, inputNum, inputBytes, tileBlockNum, tileDataNum, inputLengthAlgin32);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetShapeAttrsInfo error"), return ge::GRAPH_FAILED);

    const std::string reductionStr = GetReductionAttr(context);
    int reductionType = GetReductionType(reductionStr);
    OP_CHECK_IF(
        reductionType < 0, OP_LOGE(context, "Invalid reduction: %s", reductionStr.c_str()), return ge::GRAPH_FAILED);
    float sigma = GetSigmaAttr(context);
    if (tileDataNum >= inputNum) {
        coreNum = 1;
    } else {
        if (blockSize == 0) {
            OP_LOGE(context, "blockSize is 0");
            return ge::GRAPH_FAILED;
        } else {
            coreNum = (static_cast<uint64_t>(coreNum) < inputLengthAlgin32 / blockSize) ?
                          coreNum :
                          inputLengthAlgin32 / blockSize;
        }
    }
    uint64_t smallCoreDataNum, bigCoreDataNum, smallTailDataNum, bigTailDataNum, finalSmallTileNum, finalBigTileNum,
        tailBlockNum;
    ret = CalculateCoreBlockNums(
        context, blockSize, inputLengthAlgin32, coreNum, tileBlockNum, inputBytes, tileDataNum, smallCoreDataNum,
        bigCoreDataNum, smallTailDataNum, bigTailDataNum, finalSmallTileNum, finalBigTileNum, tailBlockNum);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "CalculateCoreBlockNums error"), return ge::GRAPH_FAILED);

    auto doutShape = context->GetInputShape(2);
    OP_CHECK_IF(doutShape == nullptr, OP_LOGE(context, "dout shape is nullptr"), return ge::GRAPH_FAILED);
    uint64_t doutDataNum = doutShape->GetStorageShape().GetShapeSize();

    tiling->smallCoreDataNum = smallCoreDataNum;
    tiling->bigCoreDataNum = bigCoreDataNum;
    tiling->tileDataNum = tileDataNum;
    tiling->smallTailDataNum = smallTailDataNum;
    tiling->bigTailDataNum = bigTailDataNum;
    tiling->finalSmallTileNum = finalSmallTileNum;
    tiling->finalBigTileNum = finalBigTileNum;
    tiling->tailBlockNum = tailBlockNum;
    tiling->totalDataNum = inputNum;
    tiling->doutDataNum = doutDataNum;
    tiling->sigma = sigma;
    tiling->reduction = reductionType;

    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);
    uint64_t tilingKey = GET_TPL_TILING_KEY(0);
    context->SetTilingKey(tilingKey);
    context->SetBlockDim(coreNum);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SmoothL1LossGradV2)
    .Tiling(SmoothL1LossGradV2TilingFunc)
    .TilingParse<SmoothL1LossGradV2CompileInfo>(TilingParseForSmoothL1LossGradV2);

} // namespace optiling
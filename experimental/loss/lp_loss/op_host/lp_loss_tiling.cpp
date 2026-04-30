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
 * \file lp_loss_tiling.cpp
 * \brief
 */

#include "log/log.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "op_host/tiling_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "graph/utils/type_utils.h"
#include "register/op_impl_registry.h"
#include "op_host/tiling_templates_registry.h"
#include "../op_kernel/lp_loss_tiling_data.h"
#include "../op_kernel/lp_loss_tiling_key.h"
#include <securec.h>
#include <cstring>

namespace optiling {

using namespace Ops::NN::OpTiling;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t TILE_SPLIT_NUM = 4096;
constexpr uint32_t DATA_NUM_32B = 3;
constexpr uint32_t DATA_NUM_16B_NO_REDUCE = 7;
constexpr uint32_t DATA_NUM_16B_REDUCE = 8;
constexpr uint32_t DATA_NUM_8B = 12;
constexpr uint32_t SINGLE_BUFFER_NUM = 1;
constexpr uint32_t DOUBLE_BUFFER_NUM = 2;

struct LpLossCompileInfo {
};

struct LpLossTilingInfo {
    uint64_t ubSize = 0;
    uint64_t blockSize = 0;
    int64_t coreNum = 0;
    uint64_t inputNum = 0;
    uint64_t inputBytes = 0;
    uint64_t tileBlockNum = 0;
    uint64_t tileDataNum = 0;
    uint64_t inputLengthAlign32 = 0;
    uint32_t bufferNum = 0;
    uint32_t reductionMode = 0;
    uint64_t smallCoreDataNum = 0;
    uint64_t bigCoreDataNum = 0;
    uint64_t smallTailDataNum = 0;
    uint64_t bigTailDataNum = 0;
    uint64_t finalSmallTileNum = 0;
    uint64_t finalBigTileNum = 0;
    uint64_t tailBlockNum = 0;
};

static ge::graphStatus TilingParseForLpLoss([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetReductionMode(gert::TilingContext* context, uint32_t& reductionMode)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);

    reductionMode = 1U; // default mean
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const int64_t* p = attrs->GetInt(0);
    OP_CHECK_IF(p == nullptr, OP_LOGE(context, "p is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(*p != 1, OP_LOGE(context, "LpLoss only supports p == 1"), return ge::GRAPH_FAILED);

    const char* reductionStr = attrs->GetStr(1);
    if (reductionStr != nullptr) {
        if (strcmp(reductionStr, "none") == 0) {
            reductionMode = 0U;
        } else if (strcmp(reductionStr, "mean") == 0) {
            reductionMode = 1U;
        } else if (strcmp(reductionStr, "sum") == 0) {
            reductionMode = 2U;
        } else {
            OP_LOGE(context, "Unsupported reduction mode: %s", reductionStr);
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, LpLossTilingInfo& tilingInfo)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);

    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OP_LOGE(context, "platformInfo is nullptr"), return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, tilingInfo.ubSize);
    tilingInfo.coreNum = ascendcPlatform.GetCoreNumAiv();
    tilingInfo.blockSize = GetUbBlockSize(context);
    OP_CHECK_IF(tilingInfo.coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(tilingInfo.ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(tilingInfo.blockSize == 0, OP_LOGE(context, "blockSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(
    gert::TilingContext* context, int64_t coreNum, uint32_t reductionMode, uint64_t blockSize)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(coreNum <= 0, OP_LOGE(context, "coreNum is invalid"), return ge::GRAPH_FAILED);
    size_t usrSize = 0;
    if (reductionMode != 0U) {
        size_t syncAllSize = static_cast<size_t>(coreNum) * blockSize;
        size_t reducePartialSize = static_cast<size_t>(coreNum) * sizeof(float);
        usrSize = syncAllSize + reducePartialSize;
    }
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OP_LOGE(context, "platformInfo is nullptr"), return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_IF(currentWorkspace == nullptr, OP_LOGE(context, "currentWorkspace is nullptr"), return ge::GRAPH_FAILED);
    currentWorkspace[0] = usrSize + sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

static uint64_t GetUbDataNumber(ge::DataType dataType, uint32_t reductionMode)
{
    switch (dataType) {
        case ge::DT_FLOAT:
        case ge::DT_INT32:
            return DATA_NUM_32B;
        case ge::DT_BF16:
        case ge::DT_FLOAT16:
            return (reductionMode == 0U) ? DATA_NUM_16B_NO_REDUCE : DATA_NUM_16B_REDUCE;
        case ge::DT_BOOL:
        case ge::DT_INT8:
        case ge::DT_UINT8:
            return DATA_NUM_8B;
        default:
            return DATA_NUM_32B;
    }
}

static ge::graphStatus GetInputMeta(
    gert::TilingContext* context, uint64_t& inputNum, uint64_t& inputBytes, ge::DataType& dataType)
{
    auto shape = context->GetInputShape(0);
    OP_CHECK_IF(shape == nullptr, OP_LOGE(context, "input shape is nullptr"), return ge::GRAPH_FAILED);
    const gert::Shape& inputShape = EnsureNotScalar(shape->GetStorageShape());
    inputNum = inputShape.GetShapeSize();
    OP_CHECK_IF(inputNum == 0, OP_LOGE(context, "input num is 0"), return ge::GRAPH_FAILED);

    auto labelShape = context->GetInputShape(1);
    OP_CHECK_IF(labelShape == nullptr, OP_LOGE(context, "label shape is nullptr"), return ge::GRAPH_FAILED);
    const gert::Shape& inputLabelShape = EnsureNotScalar(labelShape->GetStorageShape());
    OP_CHECK_IF(
        inputShape != inputLabelShape, OP_LOGE(context, "predict and label shapes must match"),
        return ge::GRAPH_FAILED);

    auto desc = context->GetInputDesc(0);
    OP_CHECK_IF(desc == nullptr, OP_LOGE(context, "input desc is nullptr"), return ge::GRAPH_FAILED);
    auto labelDesc = context->GetInputDesc(1);
    OP_CHECK_IF(labelDesc == nullptr, OP_LOGE(context, "label desc is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        desc->GetDataType() != labelDesc->GetDataType(), OP_LOGE(context, "predict and label dtypes must match"),
        return ge::GRAPH_FAILED);

    dataType = desc->GetDataType();
    uint32_t typeLength = 0;
    ge::TypeUtils::GetDataTypeLength(dataType, typeLength);
    OP_CHECK_IF(typeLength == 0, OP_LOGE(context, "typeLength is 0"), return ge::GRAPH_FAILED);
    inputBytes = typeLength;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetShapeAttrsInfo(gert::TilingContext* context, LpLossTilingInfo& tilingInfo)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    ge::DataType dataType{};
    OP_CHECK_IF(
        GetInputMeta(context, tilingInfo.inputNum, tilingInfo.inputBytes, dataType) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetInputMeta failed"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        GetReductionMode(context, tilingInfo.reductionMode) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetReductionMode failed"), return ge::GRAPH_FAILED);

    tilingInfo.inputLengthAlign32 =
        (((tilingInfo.inputNum * tilingInfo.inputBytes + tilingInfo.blockSize - 1) / tilingInfo.blockSize) *
         tilingInfo.blockSize);
    uint64_t ubDataNumber = GetUbDataNumber(dataType, tilingInfo.reductionMode);
    OP_CHECK_IF(ubDataNumber == 0, OP_LOGE(context, "ubDataNumber is 0"), return ge::GRAPH_FAILED);
    const uint64_t checkedUbDataNumber = ubDataNumber;

    // UB is per-core. Compare single-buffer need against per-core UB size only.
    uint64_t singleBufferNeedSize = tilingInfo.inputLengthAlign32 * checkedUbDataNumber;
    tilingInfo.bufferNum = (singleBufferNeedSize <= tilingInfo.ubSize) ? SINGLE_BUFFER_NUM : DOUBLE_BUFFER_NUM;

    tilingInfo.tileBlockNum = (tilingInfo.ubSize / tilingInfo.bufferNum / tilingInfo.blockSize) / checkedUbDataNumber;
    OP_CHECK_IF(tilingInfo.tileBlockNum == 0, OP_LOGE(context, "tileBlockNum is 0"), return ge::GRAPH_FAILED);

    tilingInfo.tileDataNum = (tilingInfo.tileBlockNum * tilingInfo.blockSize) / tilingInfo.inputBytes;
    OP_CHECK_IF(tilingInfo.tileDataNum == 0, OP_LOGE(context, "tileDataNum is 0"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CalculateTileInfo(
    gert::TilingContext* context, const LpLossTilingInfo& tilingInfo, uint64_t coreInputBlockNum, uint64_t& coreDataNum,
    uint64_t& tailDataNum, uint64_t& finalTileNum)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);

    coreDataNum = coreInputBlockNum * tilingInfo.blockSize / tilingInfo.inputBytes;

    uint64_t tileNum = coreInputBlockNum / tilingInfo.tileBlockNum;
    finalTileNum = (coreInputBlockNum % tilingInfo.tileBlockNum) == 0 ? tileNum : tileNum + 1;
    tailDataNum = coreDataNum - tilingInfo.tileDataNum * tileNum;
    tailDataNum = tailDataNum == 0 ? tilingInfo.tileDataNum : tailDataNum;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CalculateCoreBlockNums(gert::TilingContext* context, LpLossTilingInfo& tilingInfo)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);

    const uint64_t checkedCoreNum = static_cast<uint64_t>(tilingInfo.coreNum);
    uint64_t everyCoreInputBlockNum = tilingInfo.inputLengthAlign32 / tilingInfo.blockSize / checkedCoreNum;
    tilingInfo.tailBlockNum = (tilingInfo.inputLengthAlign32 / tilingInfo.blockSize) % checkedCoreNum;

    ge::graphStatus ret = CalculateTileInfo(
        context, tilingInfo, everyCoreInputBlockNum, tilingInfo.smallCoreDataNum, tilingInfo.smallTailDataNum,
        tilingInfo.finalSmallTileNum);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "CalculateTileInfo for small core failed"), return ret);

    everyCoreInputBlockNum += 1;
    ret = CalculateTileInfo(
        context, tilingInfo, everyCoreInputBlockNum, tilingInfo.bigCoreDataNum, tilingInfo.bigTailDataNum,
        tilingInfo.finalBigTileNum);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "CalculateTileInfo for big core failed"), return ret);

    return ge::GRAPH_SUCCESS;
}

static void SetTilingData(LpLossTilingData* tiling, const LpLossTilingInfo& tilingInfo)
{
    tiling->totalNum = tilingInfo.inputNum;
    tiling->reduction = tilingInfo.reductionMode;
    tiling->coreNum = static_cast<uint64_t>(tilingInfo.coreNum);
    tiling->blockSize = tilingInfo.blockSize;
    tiling->smallCoreDataNum = tilingInfo.smallCoreDataNum;
    tiling->bigCoreDataNum = tilingInfo.bigCoreDataNum;
    tiling->tileDataNum = tilingInfo.tileDataNum;
    tiling->smallTailDataNum = tilingInfo.smallTailDataNum;
    tiling->bigTailDataNum = tilingInfo.bigTailDataNum;
    tiling->finalSmallTileNum = tilingInfo.finalSmallTileNum;
    tiling->finalBigTileNum = tilingInfo.finalBigTileNum;
    tiling->tailBlockNum = tilingInfo.tailBlockNum;
    tiling->bufferNum = static_cast<uint64_t>(tilingInfo.bufferNum);
    tiling->meanFactor = (tilingInfo.inputNum > 0U) ? (1.0f / static_cast<float>(tilingInfo.inputNum)) : 0.0f;
}

static ge::graphStatus LpLossTilingFunc(gert::TilingContext* context)
{
    auto* tiling = context->GetTilingData<LpLossTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(LpLossTilingData), 0, sizeof(LpLossTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    LpLossTilingInfo tilingInfo;

    ge::graphStatus ret = GetPlatformInfo(context, tilingInfo);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPlatformInfo error"), return ret);

    ret = GetShapeAttrsInfo(context, tilingInfo);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetShapeAttrsInfo error"), return ret);

    // 禁用全局原子加，统一走 workspace + core0 汇总
    context->SetNeedAtomic(false);

    uint64_t calcCoreNum = (tilingInfo.inputNum + tilingInfo.tileDataNum - 1) / tilingInfo.tileDataNum;
    tilingInfo.coreNum = (calcCoreNum < static_cast<uint64_t>(tilingInfo.coreNum)) ? static_cast<int64_t>(calcCoreNum) :
                                                                                     tilingInfo.coreNum;
    tilingInfo.coreNum = tilingInfo.coreNum <= 0 ? 1 : tilingInfo.coreNum;

    ret = CalculateCoreBlockNums(context, tilingInfo);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "CalculateCoreBlockNums error"), return ret);

    SetTilingData(tiling, tilingInfo);

    OP_CHECK_IF(
        GetWorkspaceSize(context, tilingInfo.coreNum, tilingInfo.reductionMode, tilingInfo.blockSize) !=
            ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"), return ge::GRAPH_FAILED);

    uint64_t tilingKey = (tilingInfo.bufferNum == DOUBLE_BUFFER_NUM) ? GET_TPL_TILING_KEY(LPLOSS_TPL_SCH_MODE_0) :
                                                                       GET_TPL_TILING_KEY(LPLOSS_TPL_SCH_MODE_1);
    context->SetTilingKey(tilingKey);
    context->SetBlockDim(tilingInfo.coreNum);

    return ge::GRAPH_SUCCESS;
}

// tiling注册入口
IMPL_OP_OPTILING(LpLoss).Tiling(LpLossTilingFunc).TilingParse<LpLossCompileInfo>(TilingParseForLpLoss);
} // namespace optiling

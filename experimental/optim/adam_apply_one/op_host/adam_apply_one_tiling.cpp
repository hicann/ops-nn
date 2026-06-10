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
 * \file adam_apply_one_tiling.cpp
 * \brief adam_apply_one_tiling source file
 */
#include "log/log.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "../op_kernel/adam_apply_one_tiling_key.h"
#include "../op_kernel/adam_apply_one_tiling_data.h"

namespace optiling {
const uint32_t BUFFER_NUM = 1;

const uint32_t WS_SYS_SIZE = 16U * 1024U * 1024U;
const int32_t DIMS_LIMIT = 4;

#define UBDataNumberFloatFp16 14
#define UBDataNumberBf16 21

struct AdamApplyOneCompileInfo {};

struct AdamApplyOneTilingCalcParams {
    uint32_t blockSize;
    uint64_t ubSize;
    int64_t coreNum;
    int64_t totalIdx;
    uint32_t typeLength;
};

struct AdamApplyOneTilingCalcResult {
    uint32_t finalCoreNum;
    uint32_t smallCoreDataNum;
    uint32_t bigCoreDataNum;
    uint32_t tileDataNum;
    uint32_t smallTailDataNum;
    uint32_t bigTailDataNum;
    uint32_t finalSmallTileNum;
    uint32_t finalBigTileNum;
    uint32_t tailBlockNum;
};

// 获取平台信息如ubSize, coreNum
static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    // 获取ubsize coreNum
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum <= 0, OP_LOGE(context, "coreNum is <= 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// check input shape and output shape equal or not
static ge::graphStatus checkShape(gert::TilingContext* context)
{
    // 1st shape
    auto first = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, first);
    auto firstShape = first->GetStorageShape();

    // check input shape
    for (int i = 1; i < 10; i++) {
        auto input = context->GetInputShape(i);
        OP_CHECK_NULL_WITH_CONTEXT(context, input);
        OP_CHECK_IF(
            input->GetStorageShape().GetDimNum() != firstShape.GetDimNum(),
            OP_LOGE(context, "AdamApplyOne: input shape should equal"), return ge::GRAPH_FAILED);
    }

    for (int i = 0; i < 3; i++) {
        auto output = context->GetOutputShape(i);
        OP_CHECK_NULL_WITH_CONTEXT(context, output);
        OP_CHECK_IF(
            output->GetStorageShape().GetDimNum() != firstShape.GetDimNum(),
            OP_LOGE(context, "AdamApplyOne: output shape should equal"), return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

// 获取属性，shape信息
static ge::graphStatus GetShapeAttrsInfo(gert::TilingContext* context, int64_t& totalIdx, ge::DataType& dataType)
{
    // 获取输入shape信息
    auto input0 = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, input0);
    // 如果输入shape 是标量 转换为{1}，否则保持原 shape 不变
    auto input0Shape = input0->GetStorageShape();

    // shape校验
    OP_CHECK_IF(
        checkShape(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "checkShape error"), return ge::GRAPH_FAILED);

    totalIdx = 1;
    for (uint32_t i = 0; i < input0Shape.GetDimNum(); i++) {
        totalIdx *= input0Shape.GetDim(i);
    }

    // dtype校验
    const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    dataType = inputDesc->GetDataType();
    if (supportedDtype.count(dataType) == 0) {
        OP_LOGE(context, "invalid dtype");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InitTilingData(gert::TilingContext* context, AdamApplyOneTilingData*& tiling)
{
    tiling = context->GetTilingData<AdamApplyOneTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(AdamApplyOneTilingData), 0, sizeof(AdamApplyOneTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus HandleEmptyInput(gert::TilingContext* context)
{
    AdamApplyOneTilingData* tiling = nullptr;
    OP_CHECK_IF(
        InitTilingData(context, tiling) != ge::GRAPH_SUCCESS, OP_LOGE(context, "init tiling data error"),
        return ge::GRAPH_FAILED);
    context->SetBlockDim(1);
    context->SetTilingKey(0);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetTypeLength(gert::TilingContext* context, uint32_t& typeLength)
{
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    ge::TypeUtils::GetDataTypeLength(inputDesc->GetDataType(), typeLength);
    OP_CHECK_IF(typeLength == 0, OP_LOGE(context, "typeLength is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetUbDataNumber(gert::TilingContext* context, ge::DataType dataType, uint32_t& ubDataNumber)
{
    switch (dataType) {
        case ge::DT_FLOAT:
        case ge::DT_FLOAT16:
            ubDataNumber = UBDataNumberFloatFp16;
            return ge::GRAPH_SUCCESS;
        case ge::DT_BF16:
            ubDataNumber = UBDataNumberBf16;
            return ge::GRAPH_SUCCESS;
        default:
            OP_LOGE(context, "invalid dtype");
            return ge::GRAPH_FAILED;
    }
}

static AdamApplyOneTilingCalcResult CalcTilingData(const AdamApplyOneTilingCalcParams& params, uint32_t ubDataNumber)
{
    AdamApplyOneTilingCalcResult result{};
    uint64_t inputBytes = static_cast<uint64_t>(params.typeLength);
    uint64_t inputLengthBytes = static_cast<uint64_t>(params.totalIdx) * inputBytes;
    uint64_t tmp = params.ubSize / params.blockSize;
    uint32_t tileBlockNum = 1U;
    if (tmp > 0) {
        if (ubDataNumber == 0) {
            ubDataNumber = 1;
        }
        uint64_t tb = tmp / ubDataNumber;
        tileBlockNum = (tb == 0) ? 1U : static_cast<uint32_t>(tb);
    }

    result.tileDataNum = static_cast<uint32_t>((static_cast<uint64_t>(tileBlockNum) * params.blockSize) / inputBytes);
    uint64_t blocksTotal = (inputLengthBytes + params.blockSize - 1ULL) / params.blockSize;
    uint64_t coreNum64 = static_cast<uint64_t>(params.coreNum);
    coreNum64 = (coreNum64 > blocksTotal) ? blocksTotal : coreNum64;
    coreNum64 = (coreNum64 == 0ULL) ? 1ULL : coreNum64;
    if (result.tileDataNum >= params.totalIdx) {
        coreNum64 = 1ULL;
    }

    result.finalCoreNum = static_cast<uint32_t>(coreNum64);
    uint64_t everyCoreInputBlockNum = blocksTotal / coreNum64;
    result.tailBlockNum = static_cast<uint32_t>(blocksTotal % coreNum64);
    result.smallCoreDataNum = static_cast<uint32_t>(everyCoreInputBlockNum * params.blockSize / inputBytes);

    uint32_t smallTileNum = static_cast<uint32_t>(everyCoreInputBlockNum / static_cast<uint64_t>(tileBlockNum));
    result.finalSmallTileNum = ((everyCoreInputBlockNum % tileBlockNum) == 0) ? smallTileNum : (smallTileNum + 1);
    int64_t smallTailDataNum = static_cast<int64_t>(result.smallCoreDataNum) -
                               static_cast<int64_t>(result.tileDataNum) * static_cast<int64_t>(smallTileNum);
    result.smallTailDataNum = (smallTailDataNum <= 0) ? result.tileDataNum : static_cast<uint32_t>(smallTailDataNum);

    uint64_t bigEveryCoreBlockNum = everyCoreInputBlockNum + 1ULL;
    result.bigCoreDataNum = static_cast<uint32_t>(bigEveryCoreBlockNum * params.blockSize / inputBytes);
    uint32_t bigTileNum = static_cast<uint32_t>(bigEveryCoreBlockNum / static_cast<uint64_t>(tileBlockNum));
    result.finalBigTileNum = ((bigEveryCoreBlockNum % tileBlockNum) == 0) ? bigTileNum : (bigTileNum + 1);
    int64_t bigTailDataNum = static_cast<int64_t>(result.bigCoreDataNum) -
                             static_cast<int64_t>(result.tileDataNum) * static_cast<int64_t>(bigTileNum);
    result.bigTailDataNum = (bigTailDataNum <= 0) ? result.tileDataNum : static_cast<uint32_t>(bigTailDataNum);
    return result;
}

static void SetTilingData(AdamApplyOneTilingData* tiling, const AdamApplyOneTilingCalcResult& result)
{
    tiling->smallCoreDataNum = static_cast<int64_t>(result.smallCoreDataNum);
    tiling->bigCoreDataNum = static_cast<int64_t>(result.bigCoreDataNum);
    tiling->tileDataNum = static_cast<int64_t>(result.tileDataNum);
    tiling->smallTailDataNum = static_cast<int64_t>(result.smallTailDataNum);
    tiling->bigTailDataNum = static_cast<int64_t>(result.bigTailDataNum);
    tiling->finalSmallTileNum = static_cast<int64_t>(result.finalSmallTileNum);
    tiling->finalBigTileNum = static_cast<int64_t>(result.finalBigTileNum);
    tiling->tailBlockNum = static_cast<int64_t>(result.tailBlockNum);
}

// tiling 分发入口
// 可直接替换你的AdamApplyOneTilingFunc 内部实现（保留函数签名）
static ge::graphStatus AdamApplyOneTilingFunc(gert::TilingContext* context)
{
    uint32_t blockSize = Ops::Base::GetUbBlockSize(context);
    OP_CHECK_IF(blockSize == 0, OP_LOGE(context, "BLOCK_SIZE error"), return ge::GRAPH_FAILED);
    OP_LOGI(context, "enter filing func\n");

    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    int64_t totalIdx = 0;
    uint32_t typeLength = 0;
    uint32_t ubDataNumber = 0;
    ge::DataType dataType;
    AdamApplyOneTilingData* tiling = nullptr;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        GetShapeAttrsInfo(context, totalIdx, dataType) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetShapeAttrsInfo error"), return ge::GRAPH_FAILED);
    if (totalIdx <= 0) {
        return HandleEmptyInput(context);
    }
    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        InitTilingData(context, tiling) != ge::GRAPH_SUCCESS, OP_LOGE(context, "set tiling data error"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        GetTypeLength(context, typeLength) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetTypeLength error"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        GetUbDataNumber(context, dataType, ubDataNumber) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetUbDataNumber error"), return ge::GRAPH_FAILED);

    AdamApplyOneTilingCalcParams params{blockSize, ubSize, coreNum, totalIdx, typeLength};
    AdamApplyOneTilingCalcResult result = CalcTilingData(params, ubDataNumber);
    SetTilingData(tiling, result);
    OP_LOGI(
        context,
        "smallCoreDataNum: %u, bigCoreDataNum: %u, tileDataNum: %u, smallTailDataNum: %u, bigTailDataNum: %u, "
        "finalSmallTileNum: %u, finalBigTileNum: %u, tailBlockNum: %u",
        result.smallCoreDataNum, result.bigCoreDataNum, result.tileDataNum, result.smallTailDataNum,
        result.bigTailDataNum, result.finalSmallTileNum, result.finalBigTileNum, result.tailBlockNum);
    context->SetBlockDim(result.finalCoreNum);
    context->SetTilingKey(GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_0));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForAdamApplyOne([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(AdamApplyOne)
    .Tiling(AdamApplyOneTilingFunc)
    .TilingParse<AdamApplyOneCompileInfo>(TilingParseForAdamApplyOne);
} // namespace optiling

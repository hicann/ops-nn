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
 * \file gelu_tiling.cpp
 * \brief
 */
#include "log/log.h"
#include "util/math_util.h"
#include "op_host/tiling_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "op_host/tiling_templates_registry.h"
#include "../op_kernel/gelu_tiling_data.h"
#include "../op_kernel/gelu_tiling_key.h"
#include "op_common/op_host/util/platform_util.h"

namespace optiling {

using namespace Ops::NN::OpTiling;

#define BLOCK_SIZE Ops::Base::GetUbBlockSize(context)
#define OPT_CORE_SIZE 1024U
#define BLOCK_ALIGN_NUM 16U
#define UB_DATA_NUM_FLOAT 12U // for DT_FLOAT
#define UB_DATA_NUM_OTHER 14U // for other data types
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t WS_SYS_SIZE = 0;
struct GeluCompileInfo {};

static ge::graphStatus TilingParseForGelu([[maybe_unused]] gert::TilingParseContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context->GetPlatformInfo() == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
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
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = usrSize + sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetShapeAttrsInfo(
    gert::TilingContext* context, uint64_t ubSize, uint64_t& inputNum, uint64_t& typeLength, uint64_t& tileBlockNum,
    uint64_t& tileDataNum, uint64_t& inputLengthAlign32)
{
    OP_CHECK_IF(
        context == nullptr || context->GetInputShape(0) == nullptr, OP_LOGE(context, "context is nullptr"),
        return ge::GRAPH_FAILED);
    inputNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    uint32_t dtypeLen = 0;
    ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), dtypeLen);
    typeLength = dtypeLen;
    uint64_t inputLength = inputNum * typeLength;
    if (inputNum == 0) {
        OP_LOGE(context, "inputNum is 0");
        return ge::GRAPH_FAILED;
    }
    if (typeLength == 0) {
        OP_LOGE(context, "typeLength is 0");
        return ge::GRAPH_FAILED;
    }
    uint64_t ubDataNumber =
        (context->GetInputDesc(0)->GetDataType() == ge::DT_FLOAT) ? UB_DATA_NUM_FLOAT : UB_DATA_NUM_OTHER;
    tileBlockNum = (ubSize / BLOCK_SIZE) / ubDataNumber;
    tileBlockNum = tileBlockNum <= BLOCK_ALIGN_NUM ? tileBlockNum : tileBlockNum / BLOCK_ALIGN_NUM * BLOCK_ALIGN_NUM;
    tileDataNum = (tileBlockNum * BLOCK_SIZE) / typeLength;
    inputLengthAlign32 = (((inputLength + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CalculateCoreBlockNums(
    gert::TilingContext* context, uint64_t inputLengthAlign32, int64_t coreNum, uint64_t tileBlockNum,
    uint64_t typeLength, uint64_t tileDataNum, uint64_t& smallCoreDataNum, uint64_t& bigCoreDataNum,
    uint64_t& smallTailDataNum, uint64_t& bigTailDataNum, uint64_t& finalSmallTileNum, uint64_t& finalBigTileNum,
    uint64_t& tailBlockNum)
{
    if (0 == BLOCK_SIZE || 0 == coreNum || 0 == tileBlockNum || 0 == typeLength) {
        OP_LOGE(context, "BLOCK_SIZE is 0 or coreNum is 0 or tileBlockNum is 0 or typeLength is 0");
        return ge::GRAPH_FAILED;
    }
    uint64_t everyCoreInputBlockNum = inputLengthAlign32 / BLOCK_SIZE / coreNum;
    tailBlockNum = (inputLengthAlign32 / BLOCK_SIZE) % coreNum;
    smallCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / typeLength;
    uint64_t smallTileNum = everyCoreInputBlockNum / tileBlockNum;
    finalSmallTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? smallTileNum : smallTileNum + 1;
    smallTailDataNum = smallCoreDataNum - (tileDataNum * smallTileNum);
    smallTailDataNum = smallTailDataNum == 0 ? tileDataNum : smallTailDataNum;

    everyCoreInputBlockNum += 1;
    bigCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / typeLength;
    uint64_t bigTileNum = everyCoreInputBlockNum / tileBlockNum;
    finalBigTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? bigTileNum : bigTileNum + 1;
    bigTailDataNum = bigCoreDataNum - tileDataNum * bigTileNum;
    bigTailDataNum = bigTailDataNum == 0 ? tileDataNum : bigTailDataNum;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GeluTilingFunc(gert::TilingContext* context)
{
    GeluTilingData* tiling = context->GetTilingData<GeluTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(GeluTilingData), 0, sizeof(GeluTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    uint64_t ubSize;
    int64_t coreNum;
    int64_t usedcoreNum;
    ge::graphStatus ret = GetPlatformInfo(context, ubSize, coreNum);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    uint64_t inputNum, typeLength, tileBlockNum, tileDataNum, inputLengthAlign32;
    ret = GetShapeAttrsInfo(context, ubSize, inputNum, typeLength, tileBlockNum, tileDataNum, inputLengthAlign32);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetShapeAttrsInfo error"), return ge::GRAPH_FAILED);

    if (OPT_CORE_SIZE >= inputNum) {
        usedcoreNum = 1;
    } else {
        usedcoreNum = inputNum / OPT_CORE_SIZE > coreNum ? coreNum : inputNum / OPT_CORE_SIZE;
        if (inputNum % OPT_CORE_SIZE > 0) {
            if (usedcoreNum < coreNum) {
                usedcoreNum += 1;
            }
        }
    }

    uint64_t smallCoreDataNum, bigCoreDataNum, smallTailDataNum, bigTailDataNum;
    uint64_t finalSmallTileNum, finalBigTileNum, tailBlockNum;
    ret = CalculateCoreBlockNums(
        context, inputLengthAlign32, usedcoreNum, tileBlockNum, typeLength, tileDataNum, smallCoreDataNum,
        bigCoreDataNum, smallTailDataNum, bigTailDataNum, finalSmallTileNum, finalBigTileNum, tailBlockNum);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "CalculateCoreBlockNums error"), return ge::GRAPH_FAILED);

    tiling->smallCoreDataNum = static_cast<uint64_t>(smallCoreDataNum);
    tiling->bigCoreDataNum = static_cast<uint64_t>(bigCoreDataNum);
    tiling->tileDataNum = static_cast<uint64_t>(tileDataNum);
    tiling->smallTailDataNum = static_cast<uint64_t>(smallTailDataNum);
    tiling->bigTailDataNum = static_cast<uint64_t>(bigTailDataNum);
    tiling->finalSmallTileNum = static_cast<uint64_t>(finalSmallTileNum);
    tiling->finalBigTileNum = static_cast<uint64_t>(finalBigTileNum);
    tiling->tailBlockNum = static_cast<uint64_t>(tailBlockNum);

    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    uint64_t tilingKey = 0;
    tilingKey = GET_TPL_TILING_KEY(0);
    context->SetTilingKey(tilingKey);
    context->SetBlockDim(usedcoreNum);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Gelu).Tiling(GeluTilingFunc).TilingParse<GeluCompileInfo>(TilingParseForGelu);

} // namespace optiling

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file relu_grad_tiling.cpp
 * \brief
 */

#include "log/log.h"
#include "util/math_util.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "../op_kernel/relu_grad_tiling_data.h"
#include "../op_kernel/relu_grad_tiling_key.h"

namespace optiling {

using namespace Ops::NN::OpTiling;

#define BLOCK_SIZE 256U
#define UB_NUM_FLOAT_FLOAT16 7U
#define UB_NUM_BF16_INT8_UINT8 10U
#define UB_NUM_INT32 6U
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t WS_SYS_SIZE = 0;

struct ReluGradCompileInfo {};

static ge::graphStatus TilingParseForReluGrad([[maybe_unused]] gert::TilingParseContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// 获取平台信息如ubSize, coreNum
static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    // 获取ubsize coreNum
    OP_CHECK_IF(context->GetPlatformInfo() == nullptr, OP_LOGE(context, "context->GetPlatformInfo() is nullptr"), return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    coreNum = ascendcPlatform.GetCoreNum();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    size_t usrSize = 0;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t* currentWorkspace = context->GetWorkspaceSizes(
        1); // 通过框架获取workspace的指针，GetWorkspaceSizes入参为所需workspace的块数。当前限制使用一块。
    currentWorkspace[0] = usrSize + sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetShapeAttrsInfo(
    gert::TilingContext* context, uint64_t ubSize, uint64_t& inputNum, uint64_t& inputBytes, uint64_t& tileBlockNum,
    uint64_t& tileDataNum, uint64_t& inputLengthAlgin)
{
    OP_CHECK_IF(
        context == nullptr || context->GetInputShape(0) == nullptr, OP_LOGE(context, "context is nullptr"),
        return ge::GRAPH_FAILED);
    inputNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    uint32_t typeLength = 0;
    ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), typeLength);
    uint64_t inputLength = inputNum * typeLength;
    if (inputNum == 0) {
        OP_LOGE(context, "inputNum is nullptr");
        return ge::GRAPH_FAILED;
    }
    inputBytes = inputLength / inputNum;
    if (inputBytes == 0) {
        OP_LOGE(context, "inputBytes is nullptr");
        return ge::GRAPH_FAILED;
    }
    uint64_t ubDataNumber = UB_NUM_BF16_INT8_UINT8;
    if (context->GetInputDesc(0)->GetDataType() == ge::DT_INT32) {
        ubDataNumber = UB_NUM_INT32;
    } else if (context->GetInputDesc(0)->GetDataType() == ge::DT_FLOAT || context->GetInputDesc(0)->GetDataType() == ge::DT_FLOAT16) {
        ubDataNumber = UB_NUM_FLOAT_FLOAT16;
    }
    if (BLOCK_SIZE == 0 || ubDataNumber == 0) {
        OP_LOGE(context, "BLOCK_SIZE or ubDataNumber is nullptr");
        return ge::GRAPH_FAILED;
    }
    tileBlockNum = (ubSize / BLOCK_SIZE) / ubDataNumber;
    tileDataNum = (tileBlockNum * BLOCK_SIZE) / inputBytes;
    inputLengthAlgin = (((inputLength + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CalculateCoreBlockNums(
    gert::TilingContext* context, uint64_t inputLengthAlgin, int64_t coreNum, uint64_t tileBlockNum, uint64_t inputBytes, uint64_t tileDataNum,
    uint64_t& smallCoreDataNum, uint64_t& bigCoreDataNum, uint64_t& smallTailDataNum, uint64_t& bigTailDataNum, uint64_t& finalSmallTileNum,
    uint64_t& finalBigTileNum, uint64_t& tailBlockNum)
{
    if (0 == BLOCK_SIZE || 0 == coreNum || 0 == tileBlockNum || 0 == inputBytes) {
        OP_LOGE(context, "BLOCK_SIZE or coreNum or tileBlockNum or inputBytes is nullptr");
        return ge::GRAPH_FAILED;
    }
    uint64_t everyCoreInputBlockNum = inputLengthAlgin / BLOCK_SIZE / coreNum;
    tailBlockNum = (inputLengthAlgin / BLOCK_SIZE) % coreNum;
    smallCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / inputBytes;
    uint64_t smallTileNum = everyCoreInputBlockNum / tileBlockNum;
    finalSmallTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? smallTileNum : smallTileNum + 1;
    smallTailDataNum = smallCoreDataNum - (tileDataNum * smallTileNum);
    smallTailDataNum = smallTailDataNum == 0 ? tileDataNum : smallTailDataNum;

    everyCoreInputBlockNum += 1;
    bigCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / inputBytes;
    uint64_t bigTileNum = everyCoreInputBlockNum / tileBlockNum;
    finalBigTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? bigTileNum : bigTileNum + 1;
    bigTailDataNum = bigCoreDataNum - tileDataNum * bigTileNum;
    bigTailDataNum = bigTailDataNum == 0 ? tileDataNum : bigTailDataNum;

    return ge::GRAPH_SUCCESS;
}

// tiling 分发入口
static ge::graphStatus ReluGradTilingFunc(gert::TilingContext* context)
{
    // 1、获取平台运行信息
    uint64_t ubSize;
    int64_t coreNum;
    ge::graphStatus ret = GetPlatformInfo(context, ubSize, coreNum);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);
    // 2、获取shape、属性信息
    uint64_t inputNum, inputBytes, tileBlockNum, tileDataNum, inputLengthAlgin;
    ret = GetShapeAttrsInfo(context, ubSize, inputNum, inputBytes, tileBlockNum, tileDataNum, inputLengthAlgin);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetShapeAttrsInfo error"), return ge::GRAPH_FAILED);
    // 3、获取WorkspaceSize信息
    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    // 4、设置tiling信息
    ReluGradTilingData* tiling = context->GetTilingData<ReluGradTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(ReluGradTilingData), 0, sizeof(ReluGradTilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    // 计算coreNum
    if (tileDataNum >= inputNum) {
        coreNum = 1;
    } else {
        coreNum = (static_cast<uint64_t>(coreNum) < inputLengthAlgin / BLOCK_SIZE) ? coreNum :
                                                                                       inputLengthAlgin / BLOCK_SIZE;
    }
    // 计算每个core处理的数据块数
    uint64_t smallCoreDataNum, bigCoreDataNum, smallTailDataNum, bigTailDataNum;
    uint64_t finalSmallTileNum, finalBigTileNum, tailBlockNum;
    ret = CalculateCoreBlockNums(
        context, inputLengthAlgin, coreNum, tileBlockNum, inputBytes, tileDataNum, smallCoreDataNum, bigCoreDataNum,
        smallTailDataNum, bigTailDataNum, finalSmallTileNum, finalBigTileNum, tailBlockNum);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "CalculateCoreBlockNums error"), return ge::GRAPH_FAILED);
    // 设置tiling数据
    tiling->smallCoreDataNum = static_cast<uint64_t>(smallCoreDataNum);
    tiling->bigCoreDataNum = static_cast<uint64_t>(bigCoreDataNum);
    tiling->tileDataNum = static_cast<uint64_t>(tileDataNum);
    tiling->smallTailDataNum = static_cast<uint64_t>(smallTailDataNum);
    tiling->bigTailDataNum = static_cast<uint64_t>(bigTailDataNum);
    tiling->finalSmallTileNum = static_cast<uint64_t>(finalSmallTileNum);
    tiling->finalBigTileNum = static_cast<uint64_t>(finalBigTileNum);
    tiling->tailBlockNum = static_cast<uint64_t>(tailBlockNum);

    context->SetBlockDim(coreNum);
    uint64_t tilingKey = 0;
    tilingKey = GET_TPL_TILING_KEY(0);
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

// tiling注册入口.
IMPL_OP_OPTILING(ReluGrad).Tiling(ReluGradTilingFunc).TilingParse<ReluGradCompileInfo>(TilingParseForReluGrad);
} // namespace optiling

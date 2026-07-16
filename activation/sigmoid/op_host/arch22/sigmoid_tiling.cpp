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
 * \file sigmoid_tiling.cpp
 * \brief
 */
#include "log/log.h"
#include "util/math_util.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "util/platform_util.h"
#include "../../op_kernel/sigmoid_tiling_data.h"
#include "../../op_kernel/sigmoid_tiling_key.h"

namespace optiling {

using namespace Ops::NN::OpTiling;

#define BLOCK_SIZE 32U
const uint32_t DATA_NUM_BIT32 = 3;
const uint32_t DATA_NUM_BIT16 = 4;
const uint32_t DATA_NUM_BIT8 = 4;
const uint32_t TILE_SPLIT_NUM = 1024;
const uint32_t SINGLE_BUFFER_NUM = 1;
const uint32_t DOUBLE_BUFFER_NUM = 2;
const uint32_t UB_RESERVED_BYTE = 1024;
struct SigmoidCompileInfo {};

static ge::graphStatus TilingParseForSigmoid([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    // 获取ubsize coreNum
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_IF(currentWorkspace == nullptr, OP_LOGE(context, "GetWorkspaceSizes failed"), return ge::GRAPH_FAILED);
    currentWorkspace[0] = sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetShapeAttrsInfo(gert::TilingContext* context, uint64_t ubSize, uint64_t coreNum,
                                         uint64_t& inputNum, uint64_t& inputBytes, uint64_t& tileBlockNum,
                                         uint64_t& tileDataNum, uint64_t& inputLengthAlgin32, uint32_t& bufferNum)
{
    uint32_t typeLength = 0;
    ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), typeLength);
    uint64_t inputLength = inputNum * typeLength;
    if (inputNum == 0) {
        return ge::GRAPH_FAILED;
    }
    inputBytes = typeLength;
    auto dataType = context->GetInputDesc(0)->GetDataType();

    uint64_t ubDataNumber = DATA_NUM_BIT32;

    switch (dataType) {
        case ge::DT_FLOAT:
            ubDataNumber = DATA_NUM_BIT32;
            break;
        case ge::DT_FLOAT16:
        case ge::DT_BF16:
        case ge::DT_INT16:
            ubDataNumber = DATA_NUM_BIT16;
            break;
        case ge::DT_INT8:
        case ge::DT_UINT8:
            ubDataNumber = DATA_NUM_BIT8;
            break;
        default:
            OP_LOGE(context, "unsupported data type for sigmoid tiling");
            return ge::GRAPH_FAILED;
    }

    inputLengthAlgin32 = (((inputLength + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE);
    uint64_t singleBufferNeedSize = inputLengthAlgin32 * ubDataNumber;

    if (singleBufferNeedSize <= coreNum * ubSize && inputNum < TILE_SPLIT_NUM * coreNum) {
        bufferNum = SINGLE_BUFFER_NUM;
    } else {
        bufferNum = DOUBLE_BUFFER_NUM;
    }

    if (ubDataNumber == 0 || bufferNum == 0) {
        return ge::GRAPH_FAILED;
    }

    tileBlockNum = (ubSize / bufferNum / BLOCK_SIZE) / ubDataNumber;
    if (inputBytes == 0) {
        return ge::GRAPH_FAILED;
    }
    tileDataNum = (tileBlockNum * BLOCK_SIZE) / inputBytes;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CalculateCoreBlockNums(uint64_t inputLengthAlgin32, int64_t coreNum, uint64_t tileBlockNum,
                                              uint64_t inputBytes, uint64_t tileDataNum, uint64_t& smallCoreDataNum,
                                              uint64_t& bigCoreDataNum, uint64_t& smallTailDataNum,
                                              uint64_t& bigTailDataNum, uint64_t& finalSmallTileNum,
                                              uint64_t& finalBigTileNum, uint64_t& tailBlockNum)
{
    if (0 == BLOCK_SIZE || 0 == coreNum || 0 == tileBlockNum || 0 == inputBytes) {
        return ge::GRAPH_FAILED;
    }

    uint64_t everyCoreInputBlockNum = inputLengthAlgin32 / BLOCK_SIZE / coreNum;
    tailBlockNum = (inputLengthAlgin32 / BLOCK_SIZE) % coreNum;

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

static ge::graphStatus SigmoidTilingFunc(gert::TilingContext* context)
{
    SigmoidTilingData* tiling = context->GetTilingData<SigmoidTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(SigmoidTilingData), 0, sizeof(SigmoidTilingData)) != EOK,
                OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    uint64_t ubSize;
    int64_t coreNum;
    ge::graphStatus ret = GetPlatformInfo(context, ubSize, coreNum);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPlatformInfo error"), return ret);

    uint64_t inputNum, inputBytes, tileBlockNum, tileDataNum, inputLengthAlgin32;
    uint32_t bufferNum;
    ubSize -= UB_RESERVED_BYTE;
    inputNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    uint64_t calcCoreNum = inputNum / TILE_SPLIT_NUM;
    if (inputNum % TILE_SPLIT_NUM)
        calcCoreNum = calcCoreNum + 1;
    coreNum = (calcCoreNum < static_cast<uint64_t>(coreNum)) ? calcCoreNum : coreNum;

    ret = GetShapeAttrsInfo(context, ubSize, coreNum, inputNum, inputBytes, tileBlockNum, tileDataNum,
                            inputLengthAlgin32, bufferNum);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetShapeAttrsInfo error"), return ret);

    uint64_t smallCoreDataNum, bigCoreDataNum, smallTailDataNum, bigTailDataNum;
    uint64_t finalSmallTileNum, finalBigTileNum, tailBlockNum;
    ret = CalculateCoreBlockNums(inputLengthAlgin32, coreNum, tileBlockNum, inputBytes, tileDataNum, smallCoreDataNum,
                                 bigCoreDataNum, smallTailDataNum, bigTailDataNum, finalSmallTileNum, finalBigTileNum,
                                 tailBlockNum);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "CalculateCoreBlockNums error"), return ret);

    tiling->smallCoreDataNum = smallCoreDataNum;
    tiling->bigCoreDataNum = bigCoreDataNum;
    tiling->tileDataNum = tileDataNum;
    tiling->smallTailDataNum = smallTailDataNum;
    tiling->bigTailDataNum = bigTailDataNum;
    tiling->finalSmallTileNum = finalSmallTileNum;
    tiling->finalBigTileNum = finalBigTileNum;
    tiling->tailBlockNum = tailBlockNum;

    OP_CHECK_IF(GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"),
                return ge::GRAPH_FAILED);
    uint64_t tilingKey = 0;
    if (bufferNum == DOUBLE_BUFFER_NUM) {
        tilingKey = GET_TPL_TILING_KEY(0);
    } else {
        tilingKey = GET_TPL_TILING_KEY(1);
    }
    context->SetTilingKey(tilingKey);
    context->SetBlockDim(coreNum);

    return ge::GRAPH_SUCCESS;
}

// tiling注册入口.
IMPL_OP_OPTILING(Sigmoid).Tiling(SigmoidTilingFunc).TilingParse<SigmoidCompileInfo>(TilingParseForSigmoid);
} // namespace optiling

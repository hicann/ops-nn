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
 * \file mse_loss_grad_tiling.cpp
 * \brief
 */
#include "log/log.h"
#include "util/math_util.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "mse_loss_grad/op_kernel/mse_loss_grad_tiling_data.h"
#include "mse_loss_grad/op_kernel/mse_loss_grad_tiling_key.h"

namespace optiling {

constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t UB_ALIGN = 32;
constexpr uint32_t REPEAT_ALIGN = 256;
constexpr uint32_t GM_ALIGN = 512;
constexpr uint32_t RESERVED_UB_SIZE = 0; // 有些api需要预留ub空间
constexpr uint32_t MAX_TILEDATA = 4096;

struct MseLossGradCompileInfo {};
// 获取平台信息如ubSize, coreNum
static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    // 获取ubsize coreNum
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

void CalculateMseLossGradTiling(MseLossGradTilingData* tiling, uint64_t inputNum, uint32_t typeLength,
                                int64_t realCoreNum, gert::TilingContext* context)
{
    if (tiling == nullptr) {
        OP_LOGE(context, "tiling is nullptr");
        return;
    }
    if (typeLength == 0) {
        OP_LOGE(context, "typeLength is 0");
        return;
    }

    uint64_t elemsPerGmBlock = (GM_ALIGN / typeLength);
    uint64_t inputLengthAlgin512 = (inputNum + elemsPerGmBlock - 1) / elemsPerGmBlock * elemsPerGmBlock;
    uint64_t tileDataNum = MAX_TILEDATA;
    int64_t needCoreNum = (inputLengthAlgin512 + tileDataNum * BUFFER_NUM - 1) / (tileDataNum * BUFFER_NUM);
    int64_t coreNum = ((realCoreNum) < needCoreNum) ? realCoreNum : needCoreNum;
    uint64_t needCoreDataNum = ((inputLengthAlgin512 + coreNum - 1) / coreNum);
    if ((coreNum < realCoreNum / 4) && (needCoreDataNum > (MAX_TILEDATA / 2))) {
        coreNum = coreNum * 2;
        needCoreDataNum = ((inputLengthAlgin512 + coreNum - 1) / coreNum);
    }
    needCoreDataNum = ((inputLengthAlgin512 + coreNum - 1) / coreNum);
    uint32_t bufferNum = 2;
    uint32_t usedDb = 1;
    if (needCoreDataNum < (MAX_TILEDATA / 4 * 3)) {
        bufferNum = 1;
        usedDb = 0;
    }
    uint64_t needTileDataNum = (needCoreDataNum + bufferNum - 1) / bufferNum;
    needTileDataNum = (needTileDataNum + elemsPerGmBlock - 1) / elemsPerGmBlock * elemsPerGmBlock;
    tileDataNum = (tileDataNum < needTileDataNum) ? tileDataNum : needTileDataNum;
    uint64_t everyCoreInputBlockNum = inputLengthAlgin512 / elemsPerGmBlock / coreNum;
    uint64_t tailBlockNum = (inputLengthAlgin512 / elemsPerGmBlock) % coreNum;
    uint64_t smallCoreDataNum = everyCoreInputBlockNum * elemsPerGmBlock;
    uint64_t finalSmallTileNum = (smallCoreDataNum + tileDataNum - 1) / tileDataNum;
    uint64_t smallTailDataNum = smallCoreDataNum - (finalSmallTileNum - 1) * tileDataNum;
    uint64_t bigCoreDataNum = smallCoreDataNum + elemsPerGmBlock;
    uint64_t finalBigTileNum = (bigCoreDataNum + tileDataNum - 1) / tileDataNum;
    uint64_t bigTailDataNum = bigCoreDataNum - (finalBigTileNum - 1) * tileDataNum;
    tiling->smallCoreDataNum = smallCoreDataNum;
    tiling->bigCoreDataNum = bigCoreDataNum;
    tiling->finalBigTileNum = finalBigTileNum;
    tiling->finalSmallTileNum = finalSmallTileNum;
    tiling->tileDataNum = tileDataNum;
    tiling->smallTailDataNum = smallTailDataNum;
    tiling->bigTailDataNum = bigTailDataNum;
    tiling->tailBlockNum = tailBlockNum;
    tiling->usedDb = usedDb;
    context->SetBlockDim(coreNum);
}

// tiling 分发入口
static ge::graphStatus MseLossGradTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize;
    int64_t realCoreNum;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, realCoreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);
    const char* reduction = context->GetAttrs()->GetStr(0);
    const char* model = "mean";
    float reduceElts = 1.0;
    float cof = 1.0;
    MseLossGradTilingData* tiling = context->GetTilingData<MseLossGradTilingData>();

    auto inputStorageShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputStorageShape);
    auto shape_reduction = inputStorageShape->GetStorageShape();

    int32_t dimNum = shape_reduction.GetDimNum();
    if (strcmp(reduction, model) == 0) {
        for (int32_t i = 0; i < dimNum; i++) {
            reduceElts = reduceElts * shape_reduction.GetDim(i);
        }
        cof = 1.0f / reduceElts * 2.0f;
        tiling->cof = cof;
    } else {
        tiling->cof = 2.0;
    }
    uint64_t inputNum;
    inputNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    uint32_t typeLength = 0;
    ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(1)->GetDataType(), typeLength);
    CalculateMseLossGradTiling(tiling, inputNum, typeLength, realCoreNum, context);
    uint64_t doutNum = context->GetInputShape(2)->GetStorageShape().GetShapeSize();
    if (doutNum == 1) {
        uint64_t tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_1);
        context->SetTilingKey(tilingKey);
    } else {
        uint64_t tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_0);
        context->SetTilingKey(tilingKey);
    }
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForMseLossGrad([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

// tiling注册入口.
IMPL_OP_OPTILING(MseLossGrad)
    .Tiling(MseLossGradTilingFunc)
    .TilingParse<MseLossGradCompileInfo>(TilingParseForMseLossGrad);
} // namespace optiling

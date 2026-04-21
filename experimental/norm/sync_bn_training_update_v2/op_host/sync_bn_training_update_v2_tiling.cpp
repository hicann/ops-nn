/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Shi Xiangyang <@shi-xiangyang225>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software: you can redistribute it and/or modify it.
 * Licensed under the CANN Open Software License Agreement Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * See the LICENSE file at the root of the repository for the full text of the License.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file sync_bn_training_update_v2_tiling.cpp
 * \brief
 */
#include "log/log.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/sync_bn_training_update_v2_tiling_data.h"
#include "../op_kernel/sync_bn_training_update_v2_tiling_key.h"

namespace optiling {

constexpr uint32_t BUFFER_NUM = 2;
constexpr uint64_t DATA_CACHE_CLEAN_NEED = 64;//64b
constexpr uint64_t UB_DATA_NUMBER_32 = 64;

struct SyncBnTrainingUpdateV2CompileInfo {};
struct SyncBnTrainingUpdateV2ShapeInfo {
    uint64_t inputNum{0};
    uint64_t inputBytes{0};
    uint64_t tileBlockNum{0};
    uint64_t tileDataNum{0};
    uint64_t inputLengthAlign32{0};
    uint64_t smallCoreDataNum{0};
    uint64_t bigCoreDataNum{0};
    uint64_t smallTailDataNum{0};
    uint64_t bigTailDataNum{0};
    uint64_t finalSmallTileNum{0};
    uint64_t finalBigTileNum{0};
    uint64_t tailBlockNum{0};
    uint64_t blockSize{0};

    uint64_t Ndim{0};
    uint64_t Cdim{0};
    uint64_t Hdim{0};
    uint64_t Wdim{0};
    float momentum{0.9f};
};

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    // 获取ubsize coreNum
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    coreNum = ascendcPlatform.GetCoreNum();
    OP_CHECK_IF(coreNum <= 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ubSize <= 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context, SyncBnTrainingUpdateV2ShapeInfo& info)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    size_t usrSize = info.Cdim * DATA_CACHE_CLEAN_NEED;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t* currentWorkspace = context->GetWorkspaceSizes(
        1); // 通过框架获取workspace的指针，GetWorkspaceSizes入参为所需workspace的块数。当前限制使用一块。
    currentWorkspace[0] = usrSize + sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetShapeAttrsInfo(gert::TilingContext* context, uint64_t ubSize, SyncBnTrainingUpdateV2ShapeInfo& info)
{
    OP_CHECK_IF(
        context == nullptr || context->GetInputShape(0) == nullptr, OP_LOGE(context, "context is nullptr"),
        return ge::GRAPH_FAILED);
    info.inputNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    uint32_t typeLength = 0;
    ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), typeLength);
    uint64_t inputLength = info.inputNum * typeLength;
    if (info.inputNum == 0) {
        return ge::GRAPH_FAILED;
    }
    info.inputBytes = inputLength / info.inputNum;
    auto dataType = context->GetInputDesc(0)->GetDataType();
    uint64_t ubDataNumber = UB_DATA_NUMBER_32;
    OP_CHECK_IF(
        dataType != ge::DT_FLOAT, OP_LOGE(context, "dataType is error"),
        return ge::GRAPH_FAILED);
    info.tileBlockNum = (ubSize / BUFFER_NUM / info.blockSize) / ubDataNumber;
    if (info.inputBytes == 0) {
        return ge::GRAPH_FAILED;
    }
    info.tileDataNum = (info.tileBlockNum * info.blockSize) / info.inputBytes;
    info.inputLengthAlign32 = (((inputLength + info.blockSize - 1) / info.blockSize) * info.blockSize);

    auto attrs = context->GetAttrs();
    info.momentum = 0.9f;  // 默认值
    if(attrs != nullptr) {
        if (attrs->GetFloat(0) != nullptr ) {
            info.momentum = *(attrs->GetFloat(0));
        }
    }
    info.Ndim = context->GetInputShape(0)->GetStorageShape().GetDim(0);
    info.Cdim = context->GetInputShape(0)->GetStorageShape().GetDim(1);
    info.Hdim = context->GetInputShape(0)->GetStorageShape().GetDim(2);
    info.Wdim = context->GetInputShape(0)->GetStorageShape().GetDim(3);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CalculateCoreBlockNums(int64_t coreNum, SyncBnTrainingUpdateV2ShapeInfo& info, gert::TilingContext* context)
{   
    OP_CHECK_IF(
        0 == coreNum || 0 == info.tileBlockNum || 0 == info.inputBytes, OP_LOGE(context, "input is null"),
        return ge::GRAPH_FAILED);
    uint64_t everyCoreInputBlockNum = info.inputLengthAlign32 / info.blockSize / coreNum;
    info.tailBlockNum = (info.inputLengthAlign32 / info.blockSize) % coreNum;
    info.smallCoreDataNum = everyCoreInputBlockNum * info.blockSize / info.inputBytes;
    uint64_t smallTileNum = everyCoreInputBlockNum / info.tileBlockNum;
    info.finalSmallTileNum = (everyCoreInputBlockNum % info.tileBlockNum) == 0 ? smallTileNum : smallTileNum + 1;
    info.smallTailDataNum = info.smallCoreDataNum - (info.tileDataNum * smallTileNum);
    info.smallTailDataNum = info.smallTailDataNum == 0 ? info.tileDataNum : info.smallTailDataNum;

    everyCoreInputBlockNum += 1;
    info.bigCoreDataNum = everyCoreInputBlockNum * info.blockSize / info.inputBytes;
    uint64_t bigTileNum = everyCoreInputBlockNum / info.tileBlockNum;
    info.finalBigTileNum = (everyCoreInputBlockNum % info.tileBlockNum) == 0 ? bigTileNum : bigTileNum + 1;
    info.bigTailDataNum = info.bigCoreDataNum - info.tileDataNum * bigTileNum;
    info.bigTailDataNum = info.bigTailDataNum == 0 ? info.tileDataNum : info.bigTailDataNum;

    return ge::GRAPH_SUCCESS;
}
// tiling 分发入口
static ge::graphStatus SyncBnTrainingUpdateV2TilingFunc(gert::TilingContext* context)
{
    // SyncBnTrainingUpdateV2TilingData tiling;
    SyncBnTrainingUpdateV2TilingData* tiling = context->GetTilingData<SyncBnTrainingUpdateV2TilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(SyncBnTrainingUpdateV2TilingData), 0, sizeof(SyncBnTrainingUpdateV2TilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);
    //获取平台运行信息
    uint64_t ubSize;
    int64_t coreNum;
    ge::graphStatus ret = GetPlatformInfo(context, ubSize, coreNum);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);
    //获取输入数据信息
    SyncBnTrainingUpdateV2ShapeInfo shapeInfo;
    shapeInfo.blockSize = Ops::Base::GetUbBlockSize(context);
    ret = GetShapeAttrsInfo(context, ubSize, shapeInfo);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetShapeAttrsInfo error"), return ge::GRAPH_FAILED);
    //计算coreNum
    if (shapeInfo.tileDataNum >= shapeInfo.inputNum) {
        coreNum = 1;
    }
    else {
        // There is at least 32B of data on each core, satisfying several settings for several cores. The maximum number of audits is the actual number of audits
        coreNum = (static_cast<uint64_t>(coreNum) < shapeInfo.inputLengthAlign32 / shapeInfo.blockSize) ? coreNum : shapeInfo.inputLengthAlign32 / shapeInfo.blockSize;
    }
    //计算每个core处理的数据块数
    ret = CalculateCoreBlockNums(coreNum, shapeInfo, context);
    OP_CHECK_IF(
        ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "CalculateCoreBlockNums is error"),
        return ret);
    //设置tiling数据
    tiling->smallCoreDataNum = static_cast<uint32_t>(shapeInfo.smallCoreDataNum);
    tiling->bigCoreDataNum = static_cast<uint32_t>(shapeInfo.bigCoreDataNum);
    tiling->tileDataNum = static_cast<uint32_t>(shapeInfo.tileDataNum);
    tiling->smallTailDataNum = static_cast<uint32_t>(shapeInfo.smallTailDataNum);
    tiling->bigTailDataNum = static_cast<uint32_t>(shapeInfo.bigTailDataNum);
    tiling->finalSmallTileNum = static_cast<uint32_t>(shapeInfo.finalSmallTileNum);
    tiling->finalBigTileNum = static_cast<uint32_t>(shapeInfo.finalBigTileNum);
    tiling->tailBlockNum = static_cast<uint32_t>(shapeInfo.tailBlockNum);

    tiling->momentum = static_cast<float>(shapeInfo.momentum);
    tiling->Ndim = static_cast<uint32_t>(shapeInfo.Ndim);
    tiling->Cdim = static_cast<uint32_t>(shapeInfo.Cdim);
    tiling->Hdim = static_cast<uint32_t>(shapeInfo.Hdim);
    tiling->Wdim = static_cast<uint32_t>(shapeInfo.Wdim);

    //计算workspace大小
    OP_CHECK_IF(GetWorkspaceSize(context, shapeInfo) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"), return ge::GRAPH_FAILED);
    context->SetBlockDim(coreNum);
    // 设置tilingKey.
    uint32_t tilingKey = 0;
    tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_0);
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForSyncBnTrainingUpdateV2([[maybe_unused]] gert::TilingParseContext* context)
{   
    return ge::GRAPH_SUCCESS;
}

// tiling注册入口.
IMPL_OP_OPTILING(SyncBnTrainingUpdateV2).Tiling(SyncBnTrainingUpdateV2TilingFunc).TilingParse<SyncBnTrainingUpdateV2CompileInfo>(TilingParseForSyncBnTrainingUpdateV2);
} // namespace optiling
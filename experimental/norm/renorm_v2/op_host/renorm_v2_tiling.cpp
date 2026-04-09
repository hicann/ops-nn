/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Cao Xiaojuan <@c15503545287>
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
 * \file renorm_v2_tiling.cpp
 * \brief
*/

#include "log/log.h"
#include "util/math_util.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "renorm_v2/op_kernel/renorm_v2_tiling_data.h"
#include "renorm_v2/op_kernel/renorm_v2_tiling_key.h"

namespace optiling {

constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t WS_SYS_SIZE = 16U * 1024U * 1024U; // 16MB
constexpr uint64_t UB_DATA_NUMBER_FLOAT = 10;
constexpr uint32_t DATA_CACHE_CLEAN_NEED = 64; // 64B
constexpr uint32_t SLOT_STRIDE = DATA_CACHE_CLEAN_NEED / sizeof(float);

struct RenormV2CompileInfo {};

struct RenormV2CompileInfoShapeInfo {
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
    
    float p{2.0f};
    int64_t axis{0};
    uint64_t rows{0};
    uint64_t cols{0};
    float max_norm{1.0f};      // 最大范数值
};

// 平台信息
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

// 工作空间
static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context, RenormV2CompileInfoShapeInfo& info)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    uint32_t workspaceCount = 0;
    if (info.axis == 0) {
        workspaceCount = info.rows;     // 按行规约
    } else if (info.axis == 1) {
        workspaceCount = info.cols;     // 按列规约
    }
    size_t usrSize = workspaceCount * SLOT_STRIDE * sizeof(float); // 用户部分
    size_t* currentWorkspace = context->GetWorkspaceSizes(1); // 通过框架获取workspace的指针，GetWorkspaceSizes入参为所需workspace的块数。当前限制使用一块。
    currentWorkspace[0] = usrSize + sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

// 形状属性
static ge::graphStatus GetShapeAttrsInfo(gert::TilingContext* context, uint64_t ubSize, RenormV2CompileInfoShapeInfo& info)
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
    uint64_t ubDataNumber = UB_DATA_NUMBER_FLOAT;
    
    switch (dataType) {
        case ge::DT_FLOAT16:
        case ge::DT_FLOAT:
            ubDataNumber = UB_DATA_NUMBER_FLOAT;
            break;
        default:
            return ge::GRAPH_FAILED;
    }
    
    info.tileBlockNum = (ubSize / BUFFER_NUM / BLOCK_SIZE) / ubDataNumber;
    if (info.inputBytes == 0) {
        return ge::GRAPH_FAILED;
    }
    
    info.tileDataNum = (info.tileBlockNum * BLOCK_SIZE) / info.inputBytes;
    info.inputLengthAlign32 = (((inputLength + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE);
    
    // 获取属性参数
    auto attrs = context->GetAttrs();
    if (attrs) {
        // 获取范数阶数p
        if (attrs->GetFloat(0)) {
            info.p = *(attrs->GetFloat(0));
        }
        // 获取axis参数
        if (attrs->GetInt(1)) {
            info.axis = *(attrs->GetInt(1));
        }
        // 获取max_norm参数
        if (attrs->GetFloat(2)) {
            info.max_norm = *(attrs->GetFloat(2));
        }
    }
    
    // 获取输入张量形状，假设是2D张量 [rows, cols]
    auto storageShape = context->GetInputShape(0)->GetStorageShape();
    if (storageShape.GetDimNum() != 2) {
        OP_LOGE(context, "RenormV2 only supports 2D tensor");
        return ge::GRAPH_FAILED;
    }
    
    info.rows = storageShape.GetDim(0);
    info.cols = storageShape.GetDim(1);
    
    return ge::GRAPH_SUCCESS;
}

// 分块信息
static ge::graphStatus CalculateCoreBlockNums(int64_t coreNum, RenormV2CompileInfoShapeInfo& info)
{
    if (0 == coreNum || 0 == info.tileBlockNum || 0 == info.inputBytes) {
        return ge::GRAPH_FAILED;
    }
    
    uint64_t everyCoreInputBlockNum = info.inputLengthAlign32 / BLOCK_SIZE / coreNum;
    info.tailBlockNum = (info.inputLengthAlign32 / BLOCK_SIZE) % coreNum;
    
    info.smallCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / info.inputBytes;
    uint64_t smallTileNum = everyCoreInputBlockNum / info.tileBlockNum;
    info.finalSmallTileNum = (everyCoreInputBlockNum % info.tileBlockNum) == 0 ? smallTileNum : smallTileNum + 1;
    info.smallTailDataNum = info.smallCoreDataNum - (info.tileDataNum * smallTileNum);
    info.smallTailDataNum = info.smallTailDataNum == 0 ? info.tileDataNum : info.smallTailDataNum;
    
    everyCoreInputBlockNum += 1;
    info.bigCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / info.inputBytes;
    uint64_t bigTileNum = everyCoreInputBlockNum / info.tileBlockNum;
    info.finalBigTileNum = (everyCoreInputBlockNum % info.tileBlockNum) == 0 ? bigTileNum : bigTileNum + 1;
    info.bigTailDataNum = info.bigCoreDataNum - info.tileDataNum * bigTileNum;
    info.bigTailDataNum = info.bigTailDataNum == 0 ? info.tileDataNum : info.bigTailDataNum;
    
    return ge::GRAPH_SUCCESS;
}

// tiling 分发入口
static ge::graphStatus RenormV2TilingFunc(gert::TilingContext* context)
{
    RenormV2TilingData* tiling = context->GetTilingData<RenormV2TilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    
    OP_CHECK_IF(
        memset_s(tiling, sizeof(RenormV2TilingData), 0, sizeof(RenormV2TilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);
    
    // 获取平台运行信息
    uint64_t ubSize;
    int64_t coreNum;
    ge::graphStatus ret = GetPlatformInfo(context, ubSize, coreNum);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);
    
    // 获取输入数据信息
    RenormV2CompileInfoShapeInfo shapeInfo;
    ret = GetShapeAttrsInfo(context, ubSize, shapeInfo);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetShapeAttrsInfo error"), return ge::GRAPH_FAILED);
    
    // 计算coreNum
    if (shapeInfo.tileDataNum >= shapeInfo.inputNum) {
        coreNum = 1;
    } else {
        // There is at least 32B of data on each core, satisfying several settings for several cores. 
        // The maximum number of audits is the actual number of audits
        coreNum = (static_cast<uint64_t>(coreNum) < shapeInfo.inputLengthAlign32 / BLOCK_SIZE) 
                  ? coreNum : shapeInfo.inputLengthAlign32 / BLOCK_SIZE;
    }
    
    // 计算每个core处理的数据块数
    ret = CalculateCoreBlockNums(coreNum, shapeInfo);
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(context, "CalculateCoreBlockNums error");
        return ret;
    }
    
    // 设置tiling数据
    tiling->smallCoreDataNum = static_cast<uint32_t>(shapeInfo.smallCoreDataNum);
    tiling->bigCoreDataNum = static_cast<uint32_t>(shapeInfo.bigCoreDataNum);
    tiling->tileDataNum = static_cast<uint32_t>(shapeInfo.tileDataNum);
    tiling->smallTailDataNum = static_cast<uint32_t>(shapeInfo.smallTailDataNum);
    tiling->bigTailDataNum = static_cast<uint32_t>(shapeInfo.bigTailDataNum);
    tiling->finalSmallTileNum = static_cast<uint32_t>(shapeInfo.finalSmallTileNum);
    tiling->finalBigTileNum = static_cast<uint32_t>(shapeInfo.finalBigTileNum);
    tiling->tailBlockNum = static_cast<uint32_t>(shapeInfo.tailBlockNum);
    tiling->p = static_cast<float>(shapeInfo.p);
    tiling->axis = static_cast<int32_t>(shapeInfo.axis);
    tiling->rows = static_cast<uint32_t>(shapeInfo.rows);
    tiling->cols = static_cast<uint32_t>(shapeInfo.cols);
    tiling->max_norm = static_cast<float>(shapeInfo.max_norm);
    
    // 计算workspace大小
    ret = GetWorkspaceSize(context, shapeInfo);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"), return ge::GRAPH_FAILED);
    
    context->SetBlockDim(coreNum);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForRenormV2([[maybe_unused]] gert::TilingParseContext* context)
{   
    return ge::GRAPH_SUCCESS;
}

// tiling注册入口
IMPL_OP_OPTILING(RenormV2).Tiling(RenormV2TilingFunc).TilingParse<RenormV2CompileInfo>(TilingParseForRenormV2);

} // namespace optiling
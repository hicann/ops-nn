/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Zhou Jianhua <@LePenseur>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file max_pooling_grad_tiling.cpp
 * \brief
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "graph/utils/type_utils.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/max_pooling_grad_tiling_data.h"

namespace optiling {

constexpr uint32_t BLOCK_SIZE = 256; // CompareScalar/Select API 要求的字节对齐
constexpr uint32_t BUFFER_NUM = 1;   // 高级向量 API 场景使用单缓冲
constexpr uint32_t UB_PART_NUM = 8;  // UB 内 buffer 等效数量

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    auto aivCoreNum = ascendcPlatform.GetCoreNumAiv();
    coreNum = (aivCoreNum > 0) ? aivCoreNum : ascendcPlatform.GetCoreNum();
    if (ubSize == 0 || coreNum <= 0) {
        OP_LOGE(context->GetNodeName(), "invalid platform info: ubSize or coreNum is zero");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus MaxPoolingGradTilingFunc(gert::TilingContext* context)
{
    // 1. 获取平台信息
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    if (GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "get platform info failed");
        return ge::GRAPH_FAILED;
    }

    // 2. 获取输入信息
    uint64_t inputNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
    uint32_t typeLength = 0;
    ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), typeLength);
    if (typeLength == 0) {
        OP_LOGE(context->GetNodeName(), "unsupported dtype: typeLength is zero");
        return ge::GRAPH_FAILED;
    }
    uint64_t inputLength = inputNum * typeLength; // 总数据字节数
    uint32_t inputBytes = typeLength;             // 单个元素字节数

    // 3. UB tile 计算: 每个 tile 可处理的 256B block 数
    //    tileBlockNum = (ubSize / BLOCK_SIZE / BUFFER_NUM) / ubDataNumber
    //    tileDataNum  = tileBlockNum * BLOCK_SIZE / sizeof(T)
    uint64_t tileBlockNum = (ubSize / BLOCK_SIZE / BUFFER_NUM) / UB_PART_NUM;
    uint64_t tileDataNum = (tileBlockNum * BLOCK_SIZE) / inputBytes;
    if (tileDataNum == 0) {
        tileDataNum = 1; // 保底
    }

    // 4. 多核数据分配: 将数据按 BLOCK_SIZE 对齐后均匀分配
    uint64_t inputLengthAlign = ((inputLength + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
    if (coreNum > static_cast<int64_t>(inputLengthAlign / BLOCK_SIZE)) {
        coreNum = static_cast<int64_t>(inputLengthAlign / BLOCK_SIZE);
    }
    if (coreNum < 1) {
        coreNum = 1;
    }

    uint64_t everyCoreInputBlockNum = inputLengthAlign / BLOCK_SIZE / coreNum;
    uint64_t tailBlockNum = (inputLengthAlign / BLOCK_SIZE) % coreNum;

    // small core 数据量
    uint64_t smallCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / inputBytes;
    uint64_t smallTileNum = everyCoreInputBlockNum / tileBlockNum;
    uint64_t smallCoreLoopNum = ((everyCoreInputBlockNum % tileBlockNum) == 0) ? smallTileNum : smallTileNum + 1;
    uint64_t smallCoreTailDataNum = smallCoreDataNum - tileDataNum * smallTileNum;
    if (smallCoreTailDataNum == 0) {
        smallCoreTailDataNum = tileDataNum;
    }

    // big core 数据量 (比 small core 多一个 block)
    everyCoreInputBlockNum += 1;
    uint64_t bigCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / inputBytes;
    uint64_t bigTileNum = everyCoreInputBlockNum / tileBlockNum;
    uint64_t bigCoreLoopNum = ((everyCoreInputBlockNum % tileBlockNum) == 0) ? bigTileNum : bigTileNum + 1;
    uint64_t bigCoreTailDataNum = bigCoreDataNum - tileDataNum * bigTileNum;
    if (bigCoreTailDataNum == 0) {
        bigCoreTailDataNum = tileDataNum;
    }

    // 5. 填充 TilingData
    MaxPoolingGradTilingData* tiling = context->GetTilingData<MaxPoolingGradTilingData>();
    if (tiling == nullptr) {
        OP_LOGE(context->GetNodeName(), "tiling data is null");
        return ge::GRAPH_FAILED;
    }

    tiling->smallCoreDataNum = smallCoreDataNum;
    tiling->bigCoreDataNum = bigCoreDataNum;
    tiling->ubPartDataNum = tileDataNum;
    tiling->smallCoreTailDataNum = smallCoreTailDataNum;
    tiling->bigCoreTailDataNum = bigCoreTailDataNum;
    tiling->smallCoreLoopNum = smallCoreLoopNum;
    tiling->bigCoreLoopNum = bigCoreLoopNum;
    tiling->tailBlockNum = tailBlockNum;

    // Compute the last small core's actual valid data count.
    // 256B alignment may inflate data counts; the last core must respect inputNum.
    uint64_t totalCoreData = bigCoreDataNum * tailBlockNum + smallCoreDataNum * (coreNum - tailBlockNum);
    if (totalCoreData > inputNum && (coreNum - tailBlockNum) > 0) {
        uint64_t lastCoreStart = bigCoreDataNum * tailBlockNum + smallCoreDataNum * (coreNum - tailBlockNum - 1);
        tiling->lastCoreValidDataNum = (lastCoreStart < inputNum) ? (inputNum - lastCoreStart) : 0;
    } else {
        tiling->lastCoreValidDataNum = 0; // no clamp needed
    }

    // 6. 设置 BlockDim
    context->SetBlockDim(coreNum);

    // 7. workspace: CompareScalar/Select 为高级向量 API，需要系统 workspace
    size_t systemWorkspaceSize = platform_ascendc::PlatformAscendC(context->GetPlatformInfo()).GetLibApiWorkSpaceSize();
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    if (currentWorkspace == nullptr) {
        OP_LOGE(context->GetNodeName(), "workspace is null");
        return ge::GRAPH_FAILED;
    }
    currentWorkspace[0] = systemWorkspaceSize;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForMaxPoolingGrad([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct MaxPoolingGradCompileInfo {};

IMPL_OP_OPTILING(MaxPoolingGrad)
    .Tiling(MaxPoolingGradTilingFunc)
    .TilingParse<MaxPoolingGradCompileInfo>(TilingParseForMaxPoolingGrad);

} // namespace optiling

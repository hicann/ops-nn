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
 * \file sync_batch_norm_backward_elemt_tiling.cpp
 * \brief
 */

#include "log/log.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "op_host/tiling_util.h"               //tiling_base<8.5.0>; op_host<9.0.0>
#include "op_host/tiling_templates_registry.h" //tiling_base<8.5.0>; op_host<9.0.0>
#include "tiling/platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "../op_kernel/sync_batch_norm_backward_elemt_tilingdata.h"
#include "../op_kernel/sync_batch_norm_backward_elemt_tiling_key.h"

namespace optiling {
constexpr uint32_t GRAD_OUTPUT_INDEX = 0;
constexpr uint32_t SAVE_INPUT_INDEX = 1;
constexpr uint32_t MEAN_INDEX = 2;
constexpr uint32_t INVSTD_INDEX = 3;
constexpr uint32_t WEIGHT_INDEX = 4;
constexpr uint32_t MEAN_DY_INDEX = 5;
constexpr uint32_t MEAN_DY_XMU_INDEX = 6;
constexpr uint32_t GRAD_INPUT_INDEX = 0;
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t UB_ALIGN = 32;
constexpr uint32_t REPEAT_ALIGN = 256;
constexpr uint32_t GM_ALIGN = 512;
constexpr uint32_t RESERVED_UB_SIZE = 0; // 有些api需要预留ub空间
constexpr uint32_t MAX_TILEDATA = 2048;  // 最大可以到4864

struct SyncBatchNormBackwardElemtCompileInfo {};

// 获取平台信息如ubSize, coreNum
static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    // 获取ubsize coreNum
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

static ge::graphStatus CalcAndSetTilingData(uint64_t inputNum, uint32_t typeLength, int64_t realCoreNum,
                                            SyncBatchNormBackwardElemtTilingData* tiling, gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);

    uint64_t elemsPerGmBlock = (GM_ALIGN / typeLength);
    uint64_t inputLengthAlgin512 = (inputNum + elemsPerGmBlock - 1) / elemsPerGmBlock * elemsPerGmBlock;

    uint64_t max_tileDataNum = 0;

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    ge::DataType t_Dtype = inputDesc->GetDataType();

    auto input1Desc = context->GetInputDesc(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, input1Desc);
    ge::DataType t1_Dtype = input1Desc->GetDataType();

    if (t_Dtype == ge::DT_FLOAT16 && t1_Dtype == ge::DT_FLOAT16) {
        // float16 + float16
        max_tileDataNum = 6144;
    } else if (t_Dtype == ge::DT_BF16 && t1_Dtype == ge::DT_BF16) {
        // bf16 + bf16
        max_tileDataNum = 3072;
    } else if (t_Dtype == ge::DT_FLOAT16 && t1_Dtype == ge::DT_FLOAT) {
        // float16 + float
        max_tileDataNum = 3072;
    } else {
        // float + float
        max_tileDataNum = 3072;
    }
    uint64_t tileDataNum = max_tileDataNum;

    int64_t needCoreNum = (inputLengthAlgin512 + tileDataNum * BUFFER_NUM - 1) / (tileDataNum * BUFFER_NUM);
    int64_t coreNum = ((realCoreNum) < needCoreNum) ? realCoreNum : needCoreNum;
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    uint64_t needCoreDataNum = ((inputLengthAlgin512 + coreNum - 1) / coreNum);
    // 如果所需要的aicore数量少于1/4的核心数，将核心数*2
    if ((coreNum < realCoreNum / 4) && (needCoreDataNum > max_tileDataNum / 4)) {
        coreNum = coreNum * 2;
        needCoreDataNum = ((inputLengthAlgin512 + coreNum - 1) / coreNum);
    }
    uint32_t bufferNum = BUFFER_NUM;
    uint64_t usedDb = 1;
    if (needCoreDataNum < (max_tileDataNum / 2)) {
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
    // 填充tiling数据结构
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
    return ge::GRAPH_SUCCESS;
}

// tiling 分发入口
static ge::graphStatus TilingForSyncBatchNormBackwardElemt(gert::TilingContext* context)
{
    // 1、获取平台运行信息
    uint64_t ubSize;
    int64_t realCoreNum;
    if (context == nullptr) {
        OP_LOGE("SyncBatchNormBackwardElemt",
                "[SyncBatchNormBackwardElemt] SyncBatchNormBackwardElemtTiling got context is nullptr.");
        return ge::GRAPH_FAILED;
    }
    ge::graphStatus ret = GetPlatformInfo(context, ubSize, realCoreNum);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    SyncBatchNormBackwardElemtTilingData* tiling = context->GetTilingData<SyncBatchNormBackwardElemtTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    auto inputXIdx0 = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputXIdx0);
    uint64_t inputNum;
    inputNum = inputXIdx0->GetStorageShape().GetShapeSize();

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    uint32_t typeLength = 0;
    ge::TypeUtils::GetDataTypeLength(inputDesc->GetDataType(), typeLength);

    auto xInput = context->GetInputShape(GRAD_OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xInput);
    auto meanInput = context->GetInputShape(MEAN_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, meanInput);
    auto investdInput = context->GetInputShape(INVSTD_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, investdInput);
    auto weightInput = context->GetInputShape(WEIGHT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightInput);
    auto meandyInput = context->GetInputShape(MEAN_DY_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, meandyInput);
    auto meandyxmuInput = context->GetInputShape(MEAN_DY_XMU_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, meandyxmuInput);

    const gert::Shape& xInputShape = xInput->GetStorageShape();
    const gert::Shape& meanInputShape = meanInput->GetStorageShape();
    const gert::Shape& investdInputShape = investdInput->GetStorageShape();
    const gert::Shape& weightInputShape = weightInput->GetStorageShape();
    const gert::Shape& meandyInputShape = meandyInput->GetStorageShape();
    const gert::Shape& meandyxmuInputShape = meandyxmuInput->GetStorageShape();

    ret = CalcAndSetTilingData(inputNum, typeLength, realCoreNum, tiling, context);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS,
                OP_LOGE("CalcAndSetTilingData", "CalcAndSetTilingData failed with ret=%d", static_cast<int>(ret)),
                return ge::GRAPH_FAILED);

    ret = GetWorkspaceSize(context);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize error"), return ge::GRAPH_FAILED);

    uint64_t tilingKey = 0;
    tilingKey = GET_TPL_TILING_KEY(ELEMENTWISE_TPL_SCH_MODE_0);
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepareForSyncBatchNormBackwardElemt([[maybe_unused]] gert::TilingParseContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SyncBatchNormBackwardElemt)
    .Tiling(TilingForSyncBatchNormBackwardElemt)
    .TilingParse<SyncBatchNormBackwardElemtCompileInfo>(TilingPrepareForSyncBatchNormBackwardElemt);
} // namespace optiling

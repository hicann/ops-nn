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
 * \file thnn_fused_lstm_cell_tiling.cpp
 * \brief
 */

#include "thnn_fused_lstm_cell_tiling.h"
#include "log/log.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {

struct ThnnFusedLstmCellTilingInfo {
    gert::Shape cxShape;
    uint64_t B;
    uint64_t H;
    uint64_t BH;
    uint64_t col;  // 任务块的列数
    uint64_t taskTotal;  // 总任务块数
    uint64_t taskSingle;  // 单个核心需要处理的任务块数
    uint64_t ubByteSize;
    size_t sysWorkspaceByteSize;
    uint32_t taskSize;  // 按照fp32算，单个任务块能容纳（处理）的元素数量
    uint32_t tailSize;  // 单个门行尾任务块的元素数量
    uint32_t aivNum;
} info;
ThnnFusedLstmCellTilingData tiling;
const uint32_t bufCnt = 17;
const uint32_t BUFFER_NUM = 2;
const size_t INDEX_2 = 2;
const uint32_t SIZE_UNIT_BYTE = 256;
const uint32_t FP32CNT_UNIT = SIZE_UNIT_BYTE / sizeof(float);


ge::graphStatus GetInfo(gert::TilingContext* context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, info.ubByteSize);
    info.aivNum = ascendcPlatform.GetCoreNumAiv();
    if (info.aivNum == 0U) {
        OP_LOGE(context->GetNodeName(), "Invalid number of aicore!");
        return ge::GRAPH_FAILED;
    }
    info.sysWorkspaceByteSize = static_cast<size_t>(ascendcPlatform.GetLibApiWorkSpaceSize());
    info.cxShape = context->GetInputShape(INDEX_2)->GetOriginShape();
    return ge::GRAPH_SUCCESS;
}

void FillTilingInfo()
{
    info.B = static_cast<uint64_t>(info.cxShape[0]);
    info.H = static_cast<uint64_t>(info.cxShape[1]);
    info.BH = info.B * info.H;
    info.taskSize = (static_cast<uint32_t>(info.ubByteSize) / BUFFER_NUM - 8 * 1024) / bufCnt / SIZE_UNIT_BYTE * FP32CNT_UNIT;
    info.col = (info.H + static_cast<uint64_t>(info.taskSize) - 1ULL) / static_cast<uint64_t>(info.taskSize);
    info.tailSize = static_cast<uint32_t>(info.H) % info.taskSize;
    info.tailSize = (info.tailSize != 0U) ? info.tailSize : info.taskSize;
    info.taskTotal = info.col * info.B;
    info.taskSingle = (info.taskTotal + static_cast<uint64_t>(info.aivNum) - 1ULL) / static_cast<uint64_t>(info.aivNum);
}

void PrintTilingData(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "Start PrintTilingData()");
    OP_LOGD(context->GetNodeName(), "B = %llu.", info.B);
    OP_LOGD(context->GetNodeName(), "H = %llu.", info.H);
    OP_LOGD(context->GetNodeName(), "BH = %llu.", info.BH);
    OP_LOGD(context->GetNodeName(), "col = %llu.", info.col);
    OP_LOGD(context->GetNodeName(), "taskTotal = %llu.", info.taskTotal);
    OP_LOGD(context->GetNodeName(), "taskSingle = %llu.", info.taskSingle);
    OP_LOGD(context->GetNodeName(), "taskSize = %u.", info.taskSize);
    OP_LOGD(context->GetNodeName(), "tailSize = %u.", info.tailSize);
    OP_LOGD(context->GetNodeName(), "aivNum = %u.", info.aivNum);
    OP_LOGD(context->GetNodeName(), "End PrintTilingData()");
}

void SetTilingData()
{
    tiling.set_B(info.B);
    tiling.set_H(info.H);
    tiling.set_BH(info.BH);
    tiling.set_col(info.col);
    tiling.set_taskTotal(info.taskTotal);
    tiling.set_taskSingle(info.taskSingle);
    tiling.set_taskSize(info.taskSize);
    tiling.set_tailSize(info.tailSize);
}

ge::graphStatus ThnnFusedLstmCellTilingFunc(gert::TilingContext* context)
{
    // 信息获取
    if (GetInfo(context) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // 参数计算
    FillTilingInfo();
    // 参数打印
    PrintTilingData(context);
    // 参数填充
    SetTilingData();
    // tiling设定
    context->SetBlockDim(info.aivNum);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    // 设置workspace大小。未使用用户workspace
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = info.sysWorkspaceByteSize;
    OP_LOGD(
        context->GetNodeName(),
        "currentWorkspaceSize = %lld.",
        currentWorkspace[0]
    );
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ThnnFusedLstmCellTilingPrepare([[maybe_unused]] gert::TilingParseContext *context)
{
    return ge::GRAPH_SUCCESS;
}

}  // namespace

IMPL_OP_OPTILING(ThnnFusedLstmCell)
    .Tiling(ThnnFusedLstmCellTilingFunc)
    .TilingParse<ThnnFusedLstmCellCompileInfo>(ThnnFusedLstmCellTilingPrepare);

}  // namespace optiling
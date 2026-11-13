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
#include "error_util.h"
#include <limits>
#include <string>
#include <vector>

namespace optiling {
namespace {

std::string Shape2String(const gert::Shape& shape)
{
    std::string s = "[";
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        if (i != 0) {
            s += ", ";
        }
        s += std::to_string(shape.GetDim(i));
    }
    s += "]";
    return s;
}

struct ThnnFusedLstmCellTilingInfo {
    gert::Shape cxShape;
    uint64_t B;
    uint64_t H;
    uint64_t BH;
    uint64_t col;        // 任务块的列数
    uint64_t taskTotal;  // 总任务块数
    uint64_t taskSingle; // 单个核心需要处理的任务块数
    uint64_t ubByteSize;
    size_t sysWorkspaceByteSize;
    uint32_t taskSize; // 按照fp32算，单个任务块能容纳（处理）的元素数量
    uint32_t tailSize; // 单个门行尾任务块的元素数量
    uint32_t aivNum;
};
const uint32_t bufCnt = 17;
const uint32_t BUFFER_NUM = 2;
const size_t INDEX_2 = 2;
const size_t CX_DIM_NUM = 2;
const size_t INPUT_GATES_INDEX = 0;
const size_t HIDDEN_GATES_INDEX = 1;
const size_t INPUT_BIAS_INDEX = 3;
const size_t HIDDEN_BIAS_INDEX = 4;
const size_t OUTPUT_HY_INDEX = 0;
const size_t OUTPUT_CY_INDEX = 1;
const size_t OUTPUT_STORAGE_INDEX = 2;
const int64_t GATES_NUM = 4;
const uint32_t SIZE_UNIT_BYTE = 256;
const uint32_t FP32CNT_UNIT = SIZE_UNIT_BYTE / sizeof(float);
const uint32_t UB_RSV_SIZE = 8 * 1024;

ge::graphStatus ValidateInputShape(const gert::TilingContext* context, size_t index, const std::string& paramName,
                                   const std::vector<int64_t>& expectedDims)
{
    auto input = context->GetInputShape(index);
    if (input == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), paramName.c_str());
        return ge::GRAPH_FAILED;
    }
    auto shape = input->GetStorageShape();
    if (shape.GetDimNum() != expectedDims.size()) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            context->GetNodeName(), paramName.c_str(), std::to_string(shape.GetDimNum()),
            "rank of " + paramName + " must be " + std::to_string(expectedDims.size()));
        return ge::GRAPH_FAILED;
    }
    for (size_t i = 0; i < expectedDims.size(); ++i) {
        if (shape.GetDim(i) != expectedDims[i]) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                context->GetNodeName(), paramName.c_str(), Shape2String(shape),
                "dimension " + std::to_string(i) + " of " + paramName + " must be " + std::to_string(expectedDims[i]));
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ValidateOutputShape(const gert::TilingContext* context, size_t index, const std::string& paramName,
                                    const std::vector<int64_t>& expectedDims)
{
    auto output = context->GetOutputShape(index);
    if (output == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), paramName.c_str());
        return ge::GRAPH_FAILED;
    }
    auto shape = output->GetStorageShape();
    if (shape.GetDimNum() != expectedDims.size()) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            context->GetNodeName(), paramName.c_str(), std::to_string(shape.GetDimNum()),
            "rank of " + paramName + " must be " + std::to_string(expectedDims.size()));
        return ge::GRAPH_FAILED;
    }
    for (size_t i = 0; i < expectedDims.size(); ++i) {
        if (shape.GetDim(i) != expectedDims[i]) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                context->GetNodeName(), paramName.c_str(), Shape2String(shape),
                "dimension " + std::to_string(i) + " of " + paramName + " must be " + std::to_string(expectedDims[i]));
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckParamsShape(const gert::TilingContext* context)
{
    // cx的rank已在GetInfo校验(恒为2);以cx的(B,H)为准校验跨输入/输出shape关系:
    // gates=(B,4H)、cx=(B,H)、bias=(4H,)、hy/cy=(B,H)、storage=(B,4H)
    // shape取storage口径: 算子注册面为ND格式(GE构图与aclnn路径的连续化转换均保证
    // tiling期storage shape与origin shape恒等),即kernel按B*H线性寻址的物理形状,
    // 与GetInfo/FillTilingInfo按origin shape计算的B/H数值恒同
    auto cxShapePtr = context->GetInputShape(INDEX_2);
    if (cxShapePtr == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "cx");
        return ge::GRAPH_FAILED;
    }
    auto cxShape = cxShapePtr->GetStorageShape();
    const int64_t batch = cxShape.GetDim(0);
    const int64_t hidden = cxShape.GetDim(1);
    if (batch < 0 || hidden < 0) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "cx", Shape2String(cxShape),
                                              "dimensions of cx must be non-negative");
        return ge::GRAPH_FAILED;
    }
    if (hidden > std::numeric_limits<int64_t>::max() / GATES_NUM) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "cx", Shape2String(cxShape),
                                              "dimension 1 of cx is too large");
        return ge::GRAPH_FAILED;
    }
    const std::vector<int64_t> gatesDims = {batch, GATES_NUM * hidden};
    const std::vector<int64_t> commonDims = {batch, hidden};
    const std::vector<int64_t> biasDims = {GATES_NUM * hidden};

    if (ValidateInputShape(context, INPUT_GATES_INDEX, "inputGates", gatesDims) != ge::GRAPH_SUCCESS ||
        ValidateInputShape(context, HIDDEN_GATES_INDEX, "hiddenGates", gatesDims) != ge::GRAPH_SUCCESS ||
        ValidateInputShape(context, INPUT_BIAS_INDEX, "inputBias", biasDims) != ge::GRAPH_SUCCESS ||
        ValidateInputShape(context, HIDDEN_BIAS_INDEX, "hiddenBias", biasDims) != ge::GRAPH_SUCCESS ||
        ValidateOutputShape(context, OUTPUT_HY_INDEX, "hy", commonDims) != ge::GRAPH_SUCCESS ||
        ValidateOutputShape(context, OUTPUT_CY_INDEX, "cy", commonDims) != ge::GRAPH_SUCCESS ||
        ValidateOutputShape(context, OUTPUT_STORAGE_INDEX, "storage", gatesDims) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetInfo(const gert::TilingContext* context, ThnnFusedLstmCellTilingInfo* info)
{
    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        OP_LOGE(context->GetNodeName(), "Failed to get platform info.");
        return ge::GRAPH_FAILED;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, info->ubByteSize);
    info->aivNum = ascendcPlatform.GetCoreNumAiv();
    if (info->aivNum == 0U) {
        OP_LOGE(context->GetNodeName(), "Invalid number of aicore!");
        return ge::GRAPH_FAILED;
    }
    info->sysWorkspaceByteSize = static_cast<size_t>(ascendcPlatform.GetLibApiWorkSpaceSize());
    auto cxShape = context->GetInputShape(INDEX_2);
    if (cxShape == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "cx");
        return ge::GRAPH_FAILED;
    }
    auto cxOriginShape = cxShape->GetOriginShape();
    if (cxOriginShape.GetDimNum() != CX_DIM_NUM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "cx",
                                                 std::to_string(cxOriginShape.GetDimNum()), "rank of cx must be 2");
        return ge::GRAPH_FAILED;
    }
    info->cxShape = cxOriginShape;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FillTilingInfo(const gert::TilingContext* context, ThnnFusedLstmCellTilingInfo* info)
{
    info->B = static_cast<uint64_t>(info->cxShape[0]);
    info->H = static_cast<uint64_t>(info->cxShape[1]);
    info->BH = info->B * info->H;
    if (info->ubByteSize <= static_cast<uint64_t>(BUFFER_NUM) * UB_RSV_SIZE) {
        OP_LOGE(context->GetNodeName(), "UB size %zu is too small for tiling.", static_cast<size_t>(info->ubByteSize));
        return ge::GRAPH_FAILED;
    }
    info->taskSize = (static_cast<uint32_t>(info->ubByteSize) / BUFFER_NUM - UB_RSV_SIZE) / bufCnt / SIZE_UNIT_BYTE *
                     FP32CNT_UNIT;
    if (info->taskSize == 0U) {
        OP_LOGE(context->GetNodeName(), "Invalid task size 0 from UB size %zu.", static_cast<size_t>(info->ubByteSize));
        return ge::GRAPH_FAILED;
    }
    info->col = (info->H + static_cast<uint64_t>(info->taskSize) - 1ULL) / static_cast<uint64_t>(info->taskSize);
    // H % taskSize 恒小于 taskSize(<=uint32上限),先在uint64域取模再收窄,避免H>=2^32时先截断致tailSize与col失配
    info->tailSize = static_cast<uint32_t>(info->H % static_cast<uint64_t>(info->taskSize));
    info->tailSize = (info->tailSize != 0U) ? info->tailSize : info->taskSize;
    info->taskTotal = info->col * info->B;
    info->taskSingle = (info->taskTotal + static_cast<uint64_t>(info->aivNum) - 1ULL) /
                       static_cast<uint64_t>(info->aivNum);
    return ge::GRAPH_SUCCESS;
}

void PrintTilingData(const gert::TilingContext* context, const ThnnFusedLstmCellTilingInfo& info)
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

void SetTilingData(const ThnnFusedLstmCellTilingInfo& info, ThnnFusedLstmCellTilingData* tiling)
{
    tiling->set_B(info.B);
    tiling->set_H(info.H);
    tiling->set_BH(info.BH);
    tiling->set_col(info.col);
    tiling->set_taskTotal(info.taskTotal);
    tiling->set_taskSingle(info.taskSingle);
    tiling->set_taskSize(info.taskSize);
    tiling->set_tailSize(info.tailSize);
}

ge::graphStatus ThnnFusedLstmCellTilingFunc(gert::TilingContext* context)
{
    ThnnFusedLstmCellTilingInfo info{};
    ThnnFusedLstmCellTilingData tiling{};

    // 信息获取
    if (GetInfo(context, &info) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // 跨输入/输出shape关系校验(GE路tiling为唯一host校验点)
    if (CheckParamsShape(context) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // 参数计算
    if (FillTilingInfo(context, &info) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // 参数打印
    PrintTilingData(context, info);
    // 参数填充
    SetTilingData(info, &tiling);
    // tiling设定
    context->SetBlockDim(info.aivNum);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    // 设置workspace大小。未使用用户workspace
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = info.sysWorkspaceByteSize;
    OP_LOGD(context->GetNodeName(), "currentWorkspaceSize = %lld.", currentWorkspace[0]);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ThnnFusedLstmCellTilingPrepare([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

} // namespace

IMPL_OP_OPTILING(ThnnFusedLstmCell)
    .Tiling(ThnnFusedLstmCellTilingFunc)
    .TilingParse<ThnnFusedLstmCellCompileInfo>(ThnnFusedLstmCellTilingPrepare);

} // namespace optiling

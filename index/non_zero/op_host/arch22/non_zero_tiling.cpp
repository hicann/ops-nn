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
 * \file non_zero_tiling.cpp
 * \brief
 */

#include <string>
#include <vector>

#include "log/log.h"
#include "platform/platform_info.h"
#include "tiling/tiling_api.h"
#include "register/op_impl_registry.h"
#include "non_zero_tiling.h"

namespace optiling {
constexpr size_t INPUT_X_IDX = 0;
constexpr size_t ATTR_TRANSPOSE_IDX = 0;
constexpr int64_t INT32_BYTE = 4;
constexpr int64_t WORKSPACE_ALIGN = 64;

namespace {
ge::graphStatus ReportTilingFailure(const char* opName, const std::string& detail)
{
    const std::vector<const char*> keys = {"opName", "reason"};
    const std::vector<const char*> values = {
        opName, "Operator tiling execution failed. Preserve the execution logs and contact operator support."};
    REPORT_PREDEFINED_ERR_MSG("EZ1008", keys, values);
    OP_LOGE_WITHOUT_REPORT(opName, "%s", detail.c_str());
    return ge::GRAPH_FAILED;
}
} // namespace

static int64_t AlignUp(int64_t value, int64_t align) { return ((value + align - 1) / align) * align; }

ge::graphStatus NonZeroAscendCTilingImpl::GetCompileInfo()
{
    auto compileInfo = context_->GetCompileInfo<NonZeroCompileInfo>();
    if (compileInfo == nullptr) {
        return ReportTilingFailure(context_->GetNodeName(), "compile info is null");
    }
    coreNum_ = compileInfo->coreNum;
    ubSize_ = compileInfo->ubSize;
    if (coreNum_ <= 0 || ubSize_ <= 0) {
        return ReportTilingFailure(context_->GetNodeName(), "invalid platform resources: AIV core count " +
                                                                std::to_string(coreNum_) + ", UB size " +
                                                                std::to_string(ubSize_));
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NonZeroAscendCTilingImpl::CalcShapeInfo()
{
    auto xStorage = context_->GetInputShape(INPUT_X_IDX);
    if (xStorage == nullptr) {
        return ReportTilingFailure(context_->GetNodeName(), "input x storage shape is null");
    }
    auto xShape = xStorage->GetStorageShape();
    rank_ = static_cast<int64_t>(xShape.GetDimNum());
    OP_CHECK_IF(rank_ <= 0 || rank_ > MAX_DIM_NUM,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "x", std::to_string(rank_).c_str(),
                                                         "rank must be in [1, 8]"),
                return ge::GRAPH_FAILED);

    numel_ = 1;
    for (int64_t i = 0; i < MAX_DIM_NUM; i++) {
        shape_[i] = (i < rank_) ? xShape.GetDim(i) : 1;
        if (i < rank_) {
            numel_ *= shape_[i];
        }
    }

    // 输入元素字节数：bool/int8/uint8=1、fp16=2、bf16 用 -2 编码、fp32/int32=4（kernel 按位模式判零）
    auto xDesc = context_->GetInputDesc(INPUT_X_IDX);
    if (xDesc == nullptr) {
        return ReportTilingFailure(context_->GetNodeName(), "input x descriptor is null");
    }
    ge::DataType xDtype = xDesc->GetDataType();
    switch (xDtype) {
        case ge::DT_BOOL:
        case ge::DT_INT8:
        case ge::DT_UINT8:
            xElemSize_ = 1;
            break;
        case ge::DT_FLOAT16:
            xElemSize_ = 2;
            break;
        case ge::DT_BF16:
            xElemSize_ = -2; // 符号位标记 bf16，kernel 侧取绝对值得到字节数并据此选择 Cast 目标类型
            break;
        default:
            xElemSize_ = 4;
            break;
    }

    // 输出元素字节数：int32=4、int64=8。GE 图模式（jit_compile）下 NonZero 内部输出为 int32 + 后置 Cast，
    // kernel 据此切换输出值与 outShape 刷新协议的写出位宽
    auto yDesc = context_->GetOutputDesc(0);
    if (yDesc == nullptr) {
        return ReportTilingFailure(context_->GetNodeName(), "output y descriptor is null");
    }
    ge::DataType yDtype = yDesc->GetDataType();
    OP_CHECK_IF(yDtype != ge::DT_INT32 && yDtype != ge::DT_INT64,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "y", Ops::Base::ToString(yDtype).c_str(),
                                                      "set the dtype attribute to int32 or int64"),
                return ge::GRAPH_FAILED);
    yElemSize_ = (yDtype == ge::DT_INT32) ? INT32_BYTE : 8;

    auto attrs = context_->GetAttrs();
    if (attrs == nullptr) {
        return ReportTilingFailure(context_->GetNodeName(), "runtime attributes are null");
    }
    const bool* transposePtr = attrs->GetAttrPointer<bool>(ATTR_TRANSPOSE_IDX);
    if (transposePtr == nullptr) {
        return ReportTilingFailure(context_->GetNodeName(), "transpose attribute is missing from the tiling context");
    }
    // transpose=1 → 输出 [dim, num]；transpose=0 → 输出 [num, dim]
    transpose_ = *transposePtr ? 1 : 0;

    // 切核：每核至少 8 个元素，前 realCoreNum-1 核均分 8 的倍数，尾核取余
    realCoreNum_ = std::min<int64_t>(std::max<int64_t>(coreNum_, 1), std::max<int64_t>(numel_ / 8, 1));
    realCoreNum_ = (realCoreNum_ < 1) ? 1 : realCoreNum_;
    if (realCoreNum_ > 1) {
        numPerCore8_ = (numel_ / (8 * realCoreNum_)) * 8;
        tailNum_ = numel_ - (realCoreNum_ - 1) * numPerCore8_;
    } else {
        numPerCore8_ = 0;
        tailNum_ = numel_;
    }

    return ge::GRAPH_SUCCESS;
}

size_t NonZeroAscendCTilingImpl::CalcWorkspaceSize()
{
    int64_t userWorkspaceSize = AlignUp(realCoreNum_ * 8 * INT32_BYTE, WORKSPACE_ALIGN);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    size_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    return sysWorkspaceSize + static_cast<size_t>(userWorkspaceSize);
}

ge::graphStatus NonZeroAscendCTilingImpl::Init()
{
    if (GetCompileInfo() != ge::GRAPH_SUCCESS) {
        OP_LOGE_WITHOUT_REPORT(context_->GetNodeName(), "NonZero arch22 tiling GetCompileInfo failed.");
        return ge::GRAPH_FAILED;
    }
    if (CalcShapeInfo() != ge::GRAPH_SUCCESS) {
        OP_LOGE_WITHOUT_REPORT(context_->GetNodeName(), "NonZero arch22 tiling CalcShapeInfo failed.");
        return ge::GRAPH_FAILED;
    }

    context_->GetWorkspaceSizes(1)[0] = CalcWorkspaceSize();
    context_->SetBlockDim(static_cast<int32_t>(realCoreNum_));
    context_->SetScheduleMode(1);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NonZeroAscendCTilingImpl::SetKernelTiling()
{
    tilingData_.set_numel(numel_);
    tilingData_.set_rank(rank_);
    tilingData_.set_realCoreNum(realCoreNum_);
    tilingData_.set_numPerCore8(numPerCore8_);
    tilingData_.set_tailNum(tailNum_);
    tilingData_.set_transpose(transpose_ | ((yElemSize_ == 4) ? 2 : 0));
    tilingData_.set_xElemSize(xElemSize_);
    tilingData_.set_shape(shape_);

    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());

    OP_LOGD(context_->GetNodeName(),
            "NonZero arch22 tiling: numel=%ld, rank=%ld, realCoreNum=%ld, numPerCore8=%ld, tailNum=%ld, "
            "transpose=%ld, xElemSize=%ld",
            numel_, rank_, realCoreNum_, numPerCore8_, tailNum_, transpose_, xElemSize_);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForNonZeroArch22(gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<NonZeroCompileInfo>();
    if (compileInfo == nullptr) {
        return ReportTilingFailure(context->GetNodeName(), "tiling parse output compile info is null");
    }
    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        return ReportTilingFailure(context->GetNodeName(), "tiling parse platform info is null");
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);

    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    if (compileInfo->coreNum <= 0) {
        return ReportTilingFailure(context->GetNodeName(),
                                   "invalid AIV core count " + std::to_string(compileInfo->coreNum));
    }

    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->ubSize = static_cast<int64_t>(ubSize);
    if (compileInfo->ubSize <= 0) {
        return ReportTilingFailure(context->GetNodeName(), "invalid UB size " + std::to_string(compileInfo->ubSize));
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingFuncForNonZeroArch22(gert::TilingContext* context)
{
    NonZeroAscendCTilingImpl tilingImpl(context);
    if (tilingImpl.Init() != ge::GRAPH_SUCCESS) {
        OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "NonZero arch22 tiling init failed.");
        return ge::GRAPH_FAILED;
    }
    return tilingImpl.SetKernelTiling();
}

IMPL_OP_OPTILING(NonZero)
    .Tiling(TilingFuncForNonZeroArch22)
    .TilingParse<NonZeroCompileInfo>(TilingParseForNonZeroArch22);
} // namespace optiling

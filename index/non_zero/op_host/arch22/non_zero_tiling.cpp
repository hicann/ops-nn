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

static int64_t AlignUp(int64_t value, int64_t align) { return ((value + align - 1) / align) * align; }

ge::graphStatus NonZeroAscendCTilingImpl::GetCompileInfo()
{
    auto compileInfo = context_->GetCompileInfo<NonZeroCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo);
    coreNum_ = compileInfo->coreNum;
    ubSize_ = compileInfo->ubSize;
    OP_CHECK_IF(
        coreNum_ <= 0 || ubSize_ <= 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "coreNum, ubSize",
                                              (std::to_string(coreNum_) + ", " + std::to_string(ubSize_)).c_str(),
                                              "value must be greater than 0"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NonZeroAscendCTilingImpl::CalcShapeInfo()
{
    auto xStorage = context_->GetInputShape(INPUT_X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xStorage);
    auto xShape = xStorage->GetStorageShape();
    rank_ = static_cast<int64_t>(xShape.GetDimNum());
    OP_CHECK_IF(rank_ <= 0 || rank_ > MAX_DIM_NUM,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "rank", std::to_string(rank_).c_str(),
                                                      "value must be in [1, 8]"),
                return ge::GRAPH_FAILED);

    numel_ = 1;
    for (int64_t i = 0; i < MAX_DIM_NUM; i++) {
        shape_[i] = (i < rank_) ? xShape.GetDim(i) : 1;
        if (i < rank_) {
            numel_ *= shape_[i];
        }
    }

    // 坐标反解用 stride：mulInDimList[d] = shape[d+1] * ... * shape[rank-1]
    mulInDimList_[rank_ - 1] = 1;
    for (int64_t d = rank_ - 2; d >= 0; d--) {
        mulInDimList_[d] = mulInDimList_[d + 1] * shape_[d + 1];
    }
    for (int64_t d = rank_; d < MAX_DIM_NUM; d++) {
        mulInDimList_[d] = 1;
    }

    // 输入元素字节数：bool/int8/uint8=1、fp16/bf16=2、fp32/int32=4（kernel 按位模式判零）
    auto xDesc = context_->GetInputDesc(INPUT_X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
    ge::DataType xDtype = xDesc->GetDataType();
    xIsBf16_ = (xDtype == ge::DT_BF16) ? 1 : 0;
    switch (xDtype) {
        case ge::DT_BOOL:
        case ge::DT_INT8:
        case ge::DT_UINT8:
            xElemSize_ = 1;
            break;
        case ge::DT_FLOAT16:
        case ge::DT_BF16:
            xElemSize_ = 2;
            break;
        default:
            xElemSize_ = 4;
            break;
    }

    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const bool* transposePtr = attrs->GetAttrPointer<bool>(ATTR_TRANSPOSE_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, transposePtr);
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
    // 尾核不 align8，保证 totalCapacity == numel
    tailCapacity_ = tailNum_;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NonZeroAscendCTilingImpl::CalcWorkspaceSize(size_t* workspaceSize)
{
    int64_t userWorkspaceSize = AlignUp(realCoreNum_ * 8 * INT32_BYTE, WORKSPACE_ALIGN);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    size_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    *workspaceSize = sysWorkspaceSize + static_cast<size_t>(userWorkspaceSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus NonZeroAscendCTilingImpl::Init()
{
    if (GetCompileInfo() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "NonZero arch22 tiling GetCompileInfo failed.");
        return ge::GRAPH_FAILED;
    }
    if (CalcShapeInfo() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "NonZero arch22 tiling CalcShapeInfo failed.");
        return ge::GRAPH_FAILED;
    }

    size_t workspaceSize = 0;
    if (CalcWorkspaceSize(&workspaceSize) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "NonZero arch22 tiling CalcWorkspaceSize failed.");
        return ge::GRAPH_FAILED;
    }
    context_->GetWorkspaceSizes(1)[0] = workspaceSize;
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
    tilingData_.set_tailCapacity(tailCapacity_);
    tilingData_.set_transpose(transpose_);
    tilingData_.set_xElemSize(xElemSize_);
    tilingData_.set_xIsBf16(xIsBf16_);
    tilingData_.set_shape(shape_);
    tilingData_.set_mulInDimList(mulInDimList_);

    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());

    OP_LOGD(context_->GetNodeName(),
            "NonZero arch22 tiling: numel=%ld, rank=%ld, realCoreNum=%ld, numPerCore8=%ld, tailNum=%ld, "
            "tailCapacity=%ld, transpose=%ld",
            numel_, rank_, realCoreNum_, numPerCore8_, tailNum_, tailCapacity_, transpose_);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForNonZeroArch22(gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<NonZeroCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);

    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((compileInfo->coreNum <= 0), OP_LOGE(context->GetNodeName(), "Failed to get core num."),
                return ge::GRAPH_FAILED);

    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->ubSize = static_cast<int64_t>(ubSize);
    OP_CHECK_IF((compileInfo->ubSize <= 0), OP_LOGE(context->GetNodeName(), "Failed to get ub size."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingFuncForNonZeroArch22(gert::TilingContext* context)
{
    NonZeroAscendCTilingImpl tilingImpl(context);
    if (tilingImpl.Init() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "NonZero arch22 tiling init failed.");
        return ge::GRAPH_FAILED;
    }
    return tilingImpl.SetKernelTiling();
}

IMPL_OP_OPTILING(NonZero)
    .Tiling(TilingFuncForNonZeroArch22)
    .TilingParse<NonZeroCompileInfo>(TilingParseForNonZeroArch22);
} // namespace optiling

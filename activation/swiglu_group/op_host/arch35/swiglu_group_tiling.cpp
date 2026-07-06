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
 * \file swiglu_group_tiling.cpp
 * \brief
 */

#include <algorithm>
#include <cmath>
#include <sstream>
#include "swiglu_group_tiling.h"

using namespace ge;
namespace optiling {
namespace {
constexpr uint64_t WORKSPACE_SIZE = 32;
int64_t CeilDiv(int64_t x, int64_t y)
{
    if (y != 0) {
        return (x + y - 1) / y;
    }
    return x;
}
int64_t DownAlign(int64_t x, int64_t y)
{
    if (y == 0) {
        return x;
    }
    return (x / y) * y;
}
int64_t RoundUp(int64_t x, int64_t y) { return CeilDiv(x, y) * y; }

constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t NUM_TWO = 2;
constexpr int64_t DOUBLE_BUFFER = 2;
constexpr int64_t B16_BYTES = 2;
constexpr int64_t B32_BYTES = 4;
constexpr int64_t B16_ALIGN_NUM = BLOCK_SIZE / B16_BYTES;
constexpr int64_t B32_ALIGN_NUM = BLOCK_SIZE / B32_BYTES;
// Ascend950 cacheline size is 512B. Split d-factor by one cacheline so each chunk is cacheline
// aligned; the element count per cacheline depends on the x dtype size (2 bytes for fp16/bf16,
// 4 bytes for float32).
constexpr int64_t ASCEND950_CACHE_LINE_BYTES = 512;
constexpr size_t ATTR_INDEX_CLAMP_LIMIT = 0;
constexpr size_t INPUT_INDEX_X = 0;
constexpr size_t INPUT_INDEX_WEIGHT = 1;
constexpr size_t INPUT_INDEX_GROUP_INDEX = 2;
constexpr size_t OUTPUT_INDEX_Y = 0;
constexpr size_t CACHE_LINE_SIZE = 128;
constexpr float DEFAULT_CLAMP_LIMIT = -1.0f;
constexpr int64_t SWIGLU_GROUP_TILING_KEY = 1000;

int64_t ShapeElementNum(const gert::Shape& shape)
{
    int64_t elementNum = 1;
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        elementNum *= shape.GetDim(i);
    }
    return elementNum;
}
} // namespace

ge::graphStatus SwigluGroupTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        auto compileInfoPtr = context_->GetCompileInfo<SwigluGroupCompileInfo>();
        OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context_->GetNodeName(), "compile info is null"),
                    return ge::GRAPH_FAILED);
        coreNum_ = compileInfoPtr->coreNum;
        ubSize_ = compileInfoPtr->ubSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        coreNum_ = ascendcPlatform.GetCoreNumAiv();
        uint64_t ubSizePlatForm;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        ubSize_ = ubSizePlatForm;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupTiling::GetClampLimitAttr(const gert::RuntimeAttrs* attrs)
{
    auto clampLimitAttr = attrs->GetAttrPointer<float>(ATTR_INDEX_CLAMP_LIMIT);
    if (clampLimitAttr != nullptr) {
        // DEFAULT_CLAMP_LIMIT means user did not pass clamp_limit.
        if (*clampLimitAttr != DEFAULT_CLAMP_LIMIT) {
            OP_CHECK_IF(!(*clampLimitAttr > 0.0f),
                        OP_LOGE(context_->GetNodeName(), "attr clamp_limit should be greater than 0.0, got %f.",
                                *clampLimitAttr),
                        return ge::GRAPH_FAILED);
            clampLimit_ = *clampLimitAttr;
            hasClampLimit_ = 1;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupTiling::GetAttr()
{
    auto* attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    if (GetClampLimitAttr(attrs) == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupTiling::CheckWeightInfo()
{
    auto weightDesc = context_->GetOptionalInputDesc(INPUT_INDEX_WEIGHT);
    if (weightDesc != nullptr) {
        auto weightDtype = weightDesc->GetDataType();
        OP_CHECK_IF((weightDtype != ge::DT_FLOAT),
                    OP_LOGE(context_->GetNodeName(), "input weight dtype should be FLOAT, got %d.",
                            static_cast<int>(weightDtype)),
                    return ge::GRAPH_FAILED);
        auto weightShape = context_->GetOptionalInputShape(INPUT_INDEX_WEIGHT);
        if (weightShape != nullptr) {
            auto weightStorageShape = weightShape->GetStorageShape();
            auto weightElementNum = ShapeElementNum(weightStorageShape);
            OP_CHECK_IF((weightElementNum != bs_),
                        OP_LOGE(context_->GetNodeName(),
                                "input weight element num should be equal to input x outer dim product, got %ld, "
                                "expected %ld.",
                                weightElementNum, bs_),
                        return ge::GRAPH_FAILED);
            hasWeight_ = true;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupTiling::CheckGroupIndexInfo()
{
    auto groupIndexDesc = context_->GetOptionalInputDesc(INPUT_INDEX_GROUP_INDEX);
    if (groupIndexDesc != nullptr) {
        auto groupIndexDtype = groupIndexDesc->GetDataType();
        OP_CHECK_IF((groupIndexDtype != ge::DT_INT64),
                    OP_LOGE(context_->GetNodeName(), "input group_index dtype should be INT64, got %d.",
                            static_cast<int>(groupIndexDtype)),
                    return ge::GRAPH_FAILED);
        auto groupIndexShape = context_->GetOptionalInputShape(INPUT_INDEX_GROUP_INDEX);
        if (groupIndexShape != nullptr) {
            auto groupIndexStorageShape = groupIndexShape->GetStorageShape();
            g_ = 1;
            for (size_t i = 0; i < groupIndexStorageShape.GetDimNum(); i++) {
                g_ = g_ * groupIndexStorageShape.GetDim(i);
            }
            // Empty tensor is not supported: a passed group_index must have a positive element count.
            OP_CHECK_IF((g_ <= 0),
                        OP_LOGE(context_->GetNodeName(),
                                "input group_index is empty tensor, which is not supported, got element num %ld.", g_),
                        return ge::GRAPH_FAILED);
            hasGroupIndex_ = true;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupTiling::CheckOutputInfo(ge::DataType xDtype)
{
    auto yDesc = context_->GetOutputDesc(OUTPUT_INDEX_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yDesc);
    auto yDtype = yDesc->GetDataType();
    OP_CHECK_IF(
        (yDtype != xDtype),
        OP_LOGE(context_->GetNodeName(), "output y dtype should be same as input x, got y dtype %d, x dtype %d.",
                static_cast<int>(yDtype), static_cast<int>(xDtype)),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupTiling::GetShapeAttrsInfoInner()
{
    // (b, s, hc_mix)
    auto shapeX = context_->GetInputShape(INPUT_INDEX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context_, shapeX);

    auto xStorageShape = shapeX->GetStorageShape();
    auto xDesc = context_->GetInputDesc(INPUT_INDEX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
    auto xDtype = xDesc->GetDataType();
    OP_CHECK_IF((xDtype != ge::DT_FLOAT16 && xDtype != ge::DT_BF16 && xDtype != ge::DT_FLOAT),
                OP_LOGE(context_->GetNodeName(), "input x dtype only support FLOAT16, BFLOAT16 or FLOAT, got %d.",
                        static_cast<int>(xDtype)),
                return ge::GRAPH_FAILED);
    xElemBytes_ = (xDtype == ge::DT_FLOAT) ? B32_BYTES : B16_BYTES;
    auto xDimNum = xStorageShape.GetDimNum();
    OP_CHECK_IF((xDimNum == 0), OP_LOGE(context_->GetNodeName(), "input x dim num should be greater than 0."),
                return ge::GRAPH_FAILED);
    bs_ = 1;
    for (size_t i = 0; i < xDimNum - 1; i++) {
        bs_ = bs_ * xStorageShape.GetDim(i);
    }
    // Empty tensor is not supported, so every dim must be positive. The last dim is checked below;
    // bs_ is the product of the remaining dims, which is positive only when none of them is 0.
    OP_CHECK_IF((bs_ <= 0),
                OP_LOGE(context_->GetNodeName(),
                        "input x is empty tensor, which is not supported, got outer dim product %ld.", bs_),
                return ge::GRAPH_FAILED);
    d_ = xStorageShape.GetDim(xDimNum - 1);
    OP_CHECK_IF((d_ <= 0 || d_ % NUM_TWO != 0),
                OP_LOGE(context_->GetNodeName(), "input x last dim should be positive and divisible by %ld, got %ld.",
                        NUM_TWO, d_),
                return ge::GRAPH_FAILED);

    if (CheckWeightInfo() == ge::GRAPH_FAILED || CheckGroupIndexInfo() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }

    // Get Attrs
    if (GetAttr() == ge::GRAPH_FAILED) {
        OP_LOGE(context_->GetNodeName(), "Get attr failed.");
        return ge::GRAPH_FAILED;
    }

    if (CheckOutputInfo(xDtype) == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }

    splitD_ = d_ / 2;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupTiling::CalcGroupIndexTiling()
{
    if (!hasGroupIndex_) {
        return ge::GRAPH_SUCCESS;
    }

    gFactor_ = g_;
    int64_t groupIndexSize = RoundUp(gFactor_, BLOCK_SIZE / sizeof(int64_t)) * DOUBLE_BUFFER * sizeof(int64_t);
    int64_t groupIndexSumSize = BLOCK_SIZE;
    if (groupIndexSize + groupIndexSumSize <= static_cast<int64_t>(ubSize_)) {
        gLoop_ = 1;
        tailGFactor_ = gFactor_;
        return ge::GRAPH_SUCCESS;
    }

    int64_t base = 2;
    int64_t maxBase = std::max(g_, base);
    while (base <= maxBase) {
        gFactor_ = CeilDiv(g_, base);
        groupIndexSize = RoundUp(gFactor_, BLOCK_SIZE / sizeof(int64_t)) * DOUBLE_BUFFER * sizeof(int64_t);
        if (groupIndexSize + groupIndexSumSize < static_cast<int64_t>(ubSize_)) {
            break;
        }
        base++;
    }
    gFactor_ = std::max(gFactor_, static_cast<int64_t>(1));
    if (gFactor_ > static_cast<int64_t>(CACHE_LINE_SIZE / sizeof(int64_t))) {
        gFactor_ = DownAlign(gFactor_, CACHE_LINE_SIZE / sizeof(int64_t));
        gFactor_ = std::max(gFactor_, static_cast<int64_t>(1));
    }
    gLoop_ = CeilDiv(g_, gFactor_);
    tailGFactor_ = g_ % gFactor_ == 0 ? gFactor_ : g_ % gFactor_;
    return ge::GRAPH_SUCCESS;
}

void SwigluGroupTiling::InitCoreTiling()
{
    rowOfFormerBlock_ = CeilDiv(bs_, static_cast<int64_t>(coreNum_));
    usedCoreNums_ = std::min(CeilDiv(bs_, rowOfFormerBlock_), static_cast<int64_t>(coreNum_));
    rowOfTailBlock_ = bs_ - (usedCoreNums_ - 1) * rowOfFormerBlock_;
}

void SwigluGroupTiling::SetFullDTiling()
{
    dLoop_ = 1;
    dFactor_ = splitD_;
    tailDFactor_ = dFactor_;
}

void SwigluGroupTiling::SetSplitDTiling()
{
    dLoop_ = CeilDiv(splitD_, dFactor_);
    tailDFactor_ = splitD_ % dFactor_ == 0 ? dFactor_ : splitD_ % dFactor_;
}

void SwigluGroupTiling::SetRowLoopTiling()
{
    rowLoopOfFormerBlock_ = CeilDiv(rowOfFormerBlock_, rowFactor_);
    rowLoopOfTailBlock_ = CeilDiv(rowOfTailBlock_, rowFactor_);
    tailRowFactorOfFormerBlock_ = rowOfFormerBlock_ % rowFactor_ == 0 ? rowFactor_ : rowOfFormerBlock_ % rowFactor_;
    tailRowFactorOfTailBlock_ = rowOfTailBlock_ % rowFactor_ == 0 ? rowFactor_ : rowOfTailBlock_ % rowFactor_;
}

int64_t SwigluGroupTiling::AddWeightSize(int64_t totalSize, int64_t rowFactor) const
{
    return hasWeight_ ? totalSize + RoundUp(rowFactor, B32_ALIGN_NUM) * B32_BYTES * DOUBLE_BUFFER : totalSize;
}

int64_t SwigluGroupTiling::CalcTotalSize(int64_t rowFactor, int64_t dFactor) const
{
    // x0, x1 and y share the x dtype (2 bytes for fp16/bf16, 4 bytes for float32).
    int64_t alignNum = BLOCK_SIZE / xElemBytes_;
    int64_t bufBytes = rowFactor * RoundUp(dFactor, alignNum) * xElemBytes_ * DOUBLE_BUFFER;
    int64_t totalSize = bufBytes + bufBytes + bufBytes; // x0 + x1 + y
    return AddWeightSize(totalSize, rowFactor);
}

void SwigluGroupTiling::CalcDAndRowFactorTiling(int64_t rowOnceLoop, int64_t dStep)
{
    rowFactor_ = rowOnceLoop;
    if (CalcTotalSize(rowOnceLoop, splitD_) <= static_cast<int64_t>(ubSize_)) {
        SetFullDTiling();
    } else {
        int64_t base = 1;
        int64_t maxBase = CeilDiv(splitD_, dStep);
        dFactor_ = dStep;
        while (base <= maxBase) {
            int64_t tryDFactor = base * dStep;
            if (tryDFactor > splitD_) {
                tryDFactor = splitD_;
            }
            if (CalcTotalSize(rowOnceLoop, tryDFactor) > static_cast<int64_t>(ubSize_)) {
                break;
            }
            dFactor_ = tryDFactor;
            if (tryDFactor == splitD_) {
                break;
            }
            base++;
        }
        SetSplitDTiling();
    }

    if (dFactor_ == splitD_) {
        int64_t lo = 1;
        int64_t hi = rowOfFormerBlock_;
        while (lo < hi) {
            int64_t mid = lo + (hi - lo + 1) / 2;
            if (CalcTotalSize(mid, dFactor_) <= static_cast<int64_t>(ubSize_)) {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        rowFactor_ = lo;
    }

    SetRowLoopTiling();
}

ge::graphStatus SwigluGroupTiling::CalcOpTiling()
{
    ge::graphStatus status = CalcGroupIndexTiling();
    if (status == ge::GRAPH_FAILED) {
        return status;
    }
    InitCoreTiling();
    int64_t rowOnceLoop = std::min(rowOfFormerBlock_, static_cast<int64_t>(1));
    int64_t dChunk = ASCEND950_CACHE_LINE_BYTES / xElemBytes_; // elements per cacheline
    CalcDAndRowFactorTiling(rowOnceLoop, dChunk);
    SetTilingData();
    return ge::GRAPH_SUCCESS;
}

void SwigluGroupTiling::SetTilingData()
{
    tilingData_.set_bs(bs_);
    tilingData_.set_d(d_);
    tilingData_.set_splitD(splitD_);
    tilingData_.set_rowOfFormerBlock(rowOfFormerBlock_);
    tilingData_.set_rowOfTailBlock(rowOfTailBlock_);
    tilingData_.set_rowLoopOfFormerBlock(rowLoopOfFormerBlock_);
    tilingData_.set_rowLoopOfTailBlock(rowLoopOfTailBlock_);
    tilingData_.set_rowFactor(rowFactor_);
    tilingData_.set_tailRowFactorOfFormerBlock(tailRowFactorOfFormerBlock_);
    tilingData_.set_tailRowFactorOfTailBlock(tailRowFactorOfTailBlock_);
    tilingData_.set_dLoop(dLoop_);
    tilingData_.set_dFactor(dFactor_);
    tilingData_.set_tailDFactor(tailDFactor_);
    tilingData_.set_clampLimit(clampLimit_);
    tilingData_.set_g(g_);
    tilingData_.set_ubSize(ubSize_);
    tilingData_.set_gLoop(gLoop_);
    tilingData_.set_gFactor(gFactor_);
    tilingData_.set_tailGFactor(tailGFactor_);
    tilingData_.set_coreNum(coreNum_);
    tilingData_.set_hasClampLimit(hasClampLimit_);
}

void SwigluGroupTiling::SetTilingKey()
{
    tilingKey_ = SWIGLU_GROUP_TILING_KEY;
    context_->SetTilingKey(tilingKey_);
}

ge::graphStatus SwigluGroupTiling::DoOpTiling()
{
    if (GetPlatformInfo() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }

    if (GetShapeAttrsInfoInner() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }

    if (CalcOpTiling() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }

    if (GetWorkspaceSize() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }

    if (PostTiling() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    SetTilingKey();

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupTiling::GetWorkspaceSize()
{
    workspaceSize_ = WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwigluGroupTiling::PostTiling()
{
    if (hasGroupIndex_) {
        context_->SetBlockDim(coreNum_);
    } else {
        context_->SetBlockDim(usedCoreNums_);
    }
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    workspaces[0] = workspaceSize_;
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForSwigluGroup(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForSwigluGroup(gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("SwigluGroup", "Tiling context is null"), return ge::GRAPH_FAILED);
    SwigluGroupTiling swigluGroupTiling(context);
    return swigluGroupTiling.DoOpTiling();
}

IMPL_OP_OPTILING(SwigluGroup)
    .Tiling(TilingForSwigluGroup)
    .TilingParse<SwigluGroupCompileInfo>(TilingPrepareForSwigluGroup);
} // namespace optiling

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
 * \file embedding_tiling_no_contiguous.cpp
 * \brief
 */
#include "embedding_no_contiguous_tiling.h"
#include "embedding_tiling_simt.h"

using Ops::NN::Optiling::TilingRegistry;
namespace optiling {
static constexpr int64_t IN_X_IDX = 0;
static constexpr int64_t IN_INDICES_IDX = 1;
static constexpr int64_t OUT_Y_IDX = 0;
static constexpr size_t TWO = 2;
static constexpr int64_t SIMT_CACHE_SIZE = static_cast<int64_t>(128 * 1024);
static constexpr uint64_t NO_CONTIGUOUS_BASE_KEY = 100;
static constexpr uint64_t NO_CONTIGUOUS_SIMD_BASE_KEY_INT32 = 150;
static constexpr uint64_t NO_CONTIGUOUS_SIMD_BASE_KEY_INT64 = 151;
static constexpr uint64_t DIGIT_TEN = 10;
static constexpr int64_t INT32_MAX_BOUND = 2147483647;
static constexpr size_t SUPPORT_DIM = 2;
static constexpr int64_t BUFFER_NUM = 2;
static constexpr int64_t INDICES_SIZE = 8192;

bool EmbeddingNoContiguousTiling::IsCapable()
{
    // 1、x and indices non-contiguous, y contiguous,
    // x and indices dimNum is 2
    if ((!IsContiguous(xShape_, xStride_) || !IsContiguous(indicesShape_, indicesStride_)) &&
        xShape_.GetDimNum() == SUPPORT_DIM && indicesShape_.GetDimNum() == SUPPORT_DIM) {
        return true;
    }
    return false;
}

ge::graphStatus EmbeddingNoContiguousTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        auto compileInfoPtr = reinterpret_cast<const EmbeddingCompileInfo*>(context_->GetCompileInfo());
        OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context_, "compile info is null"), return ge::GRAPH_FAILED);
        totalCoreNum_ = static_cast<int64_t>(compileInfoPtr->coreNum);
        ubSize_ = static_cast<int64_t>(compileInfoPtr->ubSize);
        OP_LOGD(opName_, "Get aivNum from compileInfo is: %ld", totalCoreNum_);
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        totalCoreNum_ = static_cast<int64_t>(ascendcPlatform.GetCoreNumAiv());
        uint64_t ubSizePlatForm;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        ubSize_ = static_cast<int64_t>(ubSizePlatForm);
        OP_LOGD(opName_, "Get aivNum from ascendcPlatform is: %ld", totalCoreNum_);
    }
    OP_CHECK_IF(
        (totalCoreNum_ <= 0 || ubSize_ <= 0),
        OP_LOGE(
            opName_,
            "coreNum and ubSize should not be smaller than 0, but got coreNum [%ld] and ubSize [%ld], please check.",
            totalCoreNum_, ubSize_),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

inline bool EmbeddingNoContiguousTiling::ParamTypeIsInvalid(ge::DataType& x)
{
    std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT,     ge::DT_FLOAT16,   ge::DT_BF16,  ge::DT_BOOL,
                                             ge::DT_INT8,      ge::DT_UINT8,     ge::DT_INT16, ge::DT_UINT16,
                                             ge::DT_INT32,     ge::DT_UINT32,    ge::DT_INT64, ge::DT_UINT64,
                                             ge::DT_COMPLEX64, ge::DT_COMPLEX32, ge::DT_DOUBLE};
    return supportedDtype.count(x) == 0;
}

ge::graphStatus EmbeddingNoContiguousTiling::GetContiguousTensorInfo(gert::Shape& shape, gert::Stride& stride,
                                                                     size_t idx, bool isOut)
{
    auto xStorageShape = isOut ? context_->GetOutputShape(idx) : context_->GetInputShape(idx);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xStorageShape);
    shape = xStorageShape->GetStorageShape();
    stride.SetDimNum(shape.GetDimNum());
    int32_t maxDim = static_cast<int32_t>(shape.GetDimNum()) - 1;
    int64_t xStride = 1;
    for (int32_t j = maxDim; j >= 0; --j) {
        stride.SetStride(j, xStride);
        xStride *= shape.GetDim(j);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus EmbeddingNoContiguousTiling::GetTensorInfo(gert::Shape& shape, gert::Stride& stride, size_t idx,
                                                           bool isOut)
{
    if (isOut) {
        GetContiguousTensorInfo(shape, stride, idx, isOut);
    } else {
        bool isView = context_->InputIsView(idx);
        if (isView) {
            auto stridePtr = context_->GetInputStride(idx);
            if (stridePtr == nullptr || stridePtr->GetDimNum() == 0) {
                GetContiguousTensorInfo(shape, stride, idx, isOut);
            } else {
                auto xStorageShape = context_->GetInputShape(idx);
                OP_CHECK_NULL_WITH_CONTEXT(context_, xStorageShape);
                shape = xStorageShape->GetShape();
                stride = *(stridePtr);
            }
        } else {
            GetContiguousTensorInfo(shape, stride, idx, isOut);
        }
    }
    std::string info = isOut ? "output" : "input";
    if (shape.GetDimNum() != stride.GetDimNum()) {
        std::string dimMsg = std::to_string(shape.GetDimNum()) + " and " + std::to_string(stride.GetDimNum());
        std::string reasonMsg = info + " and stride's shape dim should be equal";
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(context_->GetNodeName(), info + " and stride", dimMsg.c_str(),
                                                  reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

inline bool EmbeddingNoContiguousTiling::IsEnableInt64(gert::Shape shape, gert::Stride stride)
{
    int64_t maxSize = 0;
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        maxSize = std::max(maxSize, shape.GetDim(i) * stride[i]);
    }
    return maxSize > INT32_MAX_BOUND;
}

ge::graphStatus EmbeddingNoContiguousTiling::CheckInAndOutDtype()
{
    // check x & indices & y type
    auto xDesc = context_->GetInputDesc(IN_X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
    xDtype_ = xDesc->GetDataType();
    if (ParamTypeIsInvalid(xDtype_)) {
        OP_LOGE_FOR_INVALID_DTYPE(
            context_->GetNodeName(), "x", ge::TypeUtils::DataTypeToSerialString(xDtype_).c_str(),
            "float, float16, bfloat16, bool, int8, uint8, int16, uint16, int32, uint32, int64, uint64, complex64, "
            "complex32 and double");
        return ge::GRAPH_FAILED;
    }
    auto indexDesc = context_->GetInputDesc(IN_INDICES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, indexDesc);
    indicesDtype_ = indexDesc->GetDataType();
    if ((indicesDtype_ != ge::DataType::DT_INT32) && (indicesDtype_ != ge::DataType::DT_INT64)) {
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "indices",
                                  ge::TypeUtils::DataTypeToSerialString(indicesDtype_).c_str(), "int32 and int64");
        return ge::GRAPH_FAILED;
    }
    auto yDesc = context_->GetOutputDesc(OUT_Y_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yDesc);
    auto yDtype = yDesc->GetDataType();
    if (yDtype != xDtype_) {
        std::string dtypeMsg = Ops::Base::ToString(yDtype) + " and " + Ops::Base::ToString(xDtype_);
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "x and y", dtypeMsg.c_str(),
                                               "The input x and output y should have same dtype");
        return ge::GRAPH_FAILED;
    }
    xDtypeSize_ = ge::GetSizeByDataType(xDtype_);
    OP_CHECK_IF(xDtypeSize_ <= 0, OP_LOGE(opName_, "get xDtypeSize fail"), return ge::GRAPH_FAILED);
    indicesDtypeSize_ = ge::GetSizeByDataType(indicesDtype_);
    OP_CHECK_IF(indicesDtypeSize_ <= 0, OP_LOGE(opName_, "get indexDtypeSize fail"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus EmbeddingNoContiguousTiling::CheckOutShape()
{
    int64_t yDimSize = static_cast<int64_t>(yShape_.GetDimNum());
    int64_t indicesDimSize = static_cast<int64_t>(indicesShape_.GetDimNum());
    if (yDimSize != indicesDimSize + 1) {
        std::string dimExpect = std::to_string(indicesDimSize + 1);
        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "y", std::to_string(yDimSize).c_str(), dimExpect.c_str());
        return ge::GRAPH_FAILED;
    }
    gatherSize_ = static_cast<int64_t>(xShape_.GetDim(0));
    innerSize_ = static_cast<int64_t>(xShape_.GetDim(1));
    int64_t indicesDim0Size = static_cast<int64_t>(indicesShape_.GetDim(0));
    indicesDim1Size_ = static_cast<int64_t>(indicesShape_.GetDim(1));
    int64_t yDim0Size = static_cast<int64_t>(yShape_.GetDim(0));
    int64_t yDim1Size = static_cast<int64_t>(yShape_.GetDim(1));
    int64_t yDim2Size = static_cast<int64_t>(yShape_.GetDim(TWO));

    if (indicesDim0Size != yDim0Size) {
        std::string dimMsg = std::to_string(indicesDim0Size) + " and " + std::to_string(yDim0Size);
        std::string reasonMsg = "indices and y's shape dim0 should be equal";
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(context_->GetNodeName(), "indices and y", dimMsg.c_str(),
                                                  reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    if (indicesDim1Size_ != yDim1Size) {
        std::string dimMsg = std::to_string(indicesDim1Size_) + " and " + std::to_string(yDim1Size);
        std::string reasonMsg = "indices and y's shape dim1 should be equal";
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(context_->GetNodeName(), "indices and y", dimMsg.c_str(),
                                                  reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    if (innerSize_ != yDim2Size) {
        std::string dimMsg = std::to_string(yDim2Size) + " and " + std::to_string(innerSize_);
        std::string reasonMsg = "y's shape dim2 and x's shape dim1 should be equal";
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(context_->GetNodeName(), "y's shape dim2 and x's shape dim1",
                                                  dimMsg.c_str(), reasonMsg.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

inline bool EmbeddingNoContiguousTiling::IsContiguous(const gert::Shape& xShape, const gert::Stride& xStride)
{
    int64_t validStride = 1;
    for (int64_t i = static_cast<int64_t>(xShape.GetDimNum()) - 1; i >= 0; i--) {
        if (xShape[i] == 1) {
            continue;
        }
        if (validStride != xStride[i]) {
            return false;
        }
        validStride *= xShape[i];
    }
    return true;
}

ge::graphStatus EmbeddingNoContiguousTiling::GetShapeAttrsInfo()
{
    opName_ = context_->GetNodeName();
    ge::graphStatus ret = CheckInAndOutDtype();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    GetTensorInfo(xShape_, xStride_, IN_X_IDX, false);
    GetTensorInfo(indicesShape_, indicesStride_, IN_INDICES_IDX, false);
    GetTensorInfo(yShape_, yStride_, OUT_Y_IDX, true);

    ySize_ = yShape_.GetShapeSize();
    return ge::GRAPH_SUCCESS;
}

void EmbeddingNoContiguousTiling::CalcuCore()
{
    if (enableInt64_) {
        threadNum_ = threadNum_ / NUM_TWO;
    }
    while ((threadNum_ >= NUM_TWO * SMALL_CASE_THREAD_NUM) &&
           (Ops::Base::CeilDiv(ySize_, threadNum_) < (totalCoreNum_ / NUM_TWO))) {
        threadNum_ = threadNum_ / NUM_TWO;
    }
    perCoreElements_ = Ops::Base::CeilDiv(ySize_, totalCoreNum_);
    if (ySize_ < threadNum_) {
        usedCoreNum_ = 1;
        perCoreElements_ = ySize_;
        lastCoreElements_ = ySize_;
        return;
    }
    perCoreElements_ = (perCoreElements_ + threadNum_ - 1) / threadNum_ * threadNum_;
    usedCoreNum_ = Ops::Base::CeilDiv(ySize_, perCoreElements_);
    lastCoreElements_ = ySize_ - perCoreElements_ * (usedCoreNum_ - 1);
}

bool EmbeddingNoContiguousTiling::IsPcieThrough()
{
#if defined(METADEF_VERSION_NUM) && METADEF_VERSION_NUM >= 90200000
    return context_->GetPcieThroughFlag();
#else
    return false;
#endif
}

bool EmbeddingNoContiguousTiling::IsUseSimdForNoContiguous()
{
    if (IsPcieThrough()) {
        return true;
    }
    return false;
}

void EmbeddingNoContiguousTiling::SimdCalcuCore()
{
    int64_t aivNum = totalCoreNum_;
    int64_t gatherSize = static_cast<int64_t>(indicesShape_.GetShapeSize());
    if (gatherSize == 0) {
        usedCoreNum_ = 1;
        perCoreElements_ = 0;
        lastCoreElements_ = 0;
        return;
    }
    int64_t blockFactor = gatherSize / aivNum;
    int64_t tailBlockFactor = gatherSize - blockFactor * aivNum;
    usedCoreNum_ = blockFactor > 0 ? aivNum : tailBlockFactor;
    perCoreElements_ = blockFactor;
    lastCoreElements_ = tailBlockFactor;
}

void EmbeddingNoContiguousTiling::SetSimdTilingData()
{
    int64_t ubBlockSize = static_cast<int64_t>(Ops::Base::GetUbBlockSize(context_));
    int64_t ubAviable = (ubSize_ - INDICES_SIZE) / ubBlockSize * ubBlockSize / xDtypeSize_ / BUFFER_NUM;
    int32_t indiceFactor = INDICES_SIZE / indicesDtypeSize_;
    int64_t blockFactor = perCoreElements_;
    int64_t tailBlockFactor = lastCoreElements_;

    simdTilingData_.set_needCoreNum(static_cast<int16_t>(usedCoreNum_));
    simdTilingData_.set_indiceFactor(indiceFactor);
    simdTilingData_.set_dtypeSize(static_cast<int32_t>(xDtypeSize_));
    simdTilingData_.set_gatherDimSize(gatherSize_);
    simdTilingData_.set_gatherSize(static_cast<int64_t>(indicesShape_.GetShapeSize()));
    simdTilingData_.set_innerSize(innerSize_);
    simdTilingData_.set_indicesDim1Size(indicesDim1Size_);
    simdTilingData_.set_xDim0Stride(xDim0Stride_);
    simdTilingData_.set_xDim1Stride(xDim1Stride_);
    simdTilingData_.set_indicesDim0Stride(indicesDim0Stride_);
    simdTilingData_.set_indicesDim1Stride(indicesDim1Stride_);
    simdTilingData_.set_blockFactor(blockFactor);
    simdTilingData_.set_tailBlockFactor(tailBlockFactor);
    simdTilingData_.set_maxElement(ubAviable);
}

void EmbeddingNoContiguousTiling::ShowSimdTilingData()
{
    std::ostringstream info;
    info << "tilingKey: " << GetTilingKey();
    info << ", needCoreNum: " << simdTilingData_.get_needCoreNum();
    info << ", indiceFactor: " << simdTilingData_.get_indiceFactor();
    info << ", dtypeSize: " << simdTilingData_.get_dtypeSize();
    info << ", gatherDimSize: " << simdTilingData_.get_gatherDimSize();
    info << ", gatherSize: " << simdTilingData_.get_gatherSize();
    info << ", innerSize: " << simdTilingData_.get_innerSize();
    info << ", indicesDim1Size: " << simdTilingData_.get_indicesDim1Size();
    info << ", xDim0Stride: " << simdTilingData_.get_xDim0Stride();
    info << ", xDim1Stride: " << simdTilingData_.get_xDim1Stride();
    info << ", indicesDim0Stride: " << simdTilingData_.get_indicesDim0Stride();
    info << ", indicesDim1Stride: " << simdTilingData_.get_indicesDim1Stride();
    info << ", blockFactor: " << simdTilingData_.get_blockFactor();
    info << ", tailBlockFactor: " << simdTilingData_.get_tailBlockFactor();
    info << ", maxElement: " << simdTilingData_.get_maxElement();
    OP_LOGI(opName_, "%s", info.str().c_str());
}

void EmbeddingNoContiguousTiling::SetTilingData()
{
    m_tilingData_.set_threadNum(threadNum_);
    m_tilingData_.set_ySize(ySize_);
    m_tilingData_.set_gatherSize(gatherSize_);
    m_tilingData_.set_innerSize(innerSize_);
    m_tilingData_.set_indicesDim1Size(indicesDim1Size_);
    m_tilingData_.set_xDim0Stride(xDim0Stride_);
    m_tilingData_.set_xDim1Stride(xDim1Stride_);
    m_tilingData_.set_indicesDim0Stride(indicesDim0Stride_);
    m_tilingData_.set_indicesDim1Stride(indicesDim1Stride_);
    m_tilingData_.set_needCoreNum(usedCoreNum_);
    m_tilingData_.set_perCoreElements(perCoreElements_);
    m_tilingData_.set_lastCoreElements(lastCoreElements_);
}

void EmbeddingNoContiguousTiling::DumpTilingInfo()
{
    if (isSimd_) {
        ShowSimdTilingData();
        return;
    }
    std::ostringstream info;
    info << "tilingKey: " << GetTilingKey();
    info << ", threadNum: " << m_tilingData_.get_threadNum();
    info << ", ySize: " << m_tilingData_.get_ySize();
    info << ", gatherSize: " << m_tilingData_.get_gatherSize();
    info << ", innerSize: " << m_tilingData_.get_innerSize();
    info << ", indicesDim1Size: " << m_tilingData_.get_indicesDim1Size();
    info << ", xDim0Stride: " << m_tilingData_.get_xDim0Stride();
    info << ", xDim1Stride: " << m_tilingData_.get_xDim1Stride();
    info << ", indicesDim0Stride: " << m_tilingData_.get_indicesDim0Stride();
    info << ", indicesDim1Stride: " << m_tilingData_.get_indicesDim1Stride();
    info << ", needCoreNum: " << m_tilingData_.get_needCoreNum();
    info << ", perCoreElements: " << m_tilingData_.get_perCoreElements();
    info << ", lastCoreElements: " << m_tilingData_.get_lastCoreElements();
    OP_LOGI(opName_, "%s", info.str().c_str());
}

ge::graphStatus EmbeddingNoContiguousTiling::DoOpTiling()
{
    ge::graphStatus ret = CheckOutShape();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    xDim0Stride_ = static_cast<int64_t>(xStride_[0]);
    xDim1Stride_ = static_cast<int64_t>(xStride_[1]);
    indicesDim0Stride_ = static_cast<int64_t>(indicesStride_[0]);
    indicesDim1Stride_ = static_cast<int64_t>(indicesStride_[1]);

    // 非连续场景要用size*stride去判断
    bool xEnableInt64 = IsEnableInt64(xShape_, xStride_);
    bool indicesEnableInt64 = IsEnableInt64(indicesShape_, indicesStride_);
    bool yEnableInt64 = IsEnableInt64(yShape_, yStride_);
    enableInt64_ = (xEnableInt64 || indicesEnableInt64 || yEnableInt64) ? 1 : 0;

    isSimd_ = IsUseSimdForNoContiguous();
    if (isSimd_) {
        SimdCalcuCore();
        SetSimdTilingData();
    } else {
        CalcuCore();
        SetTilingData();
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus EmbeddingNoContiguousTiling::DoLibApiTiling() { return ge::GRAPH_SUCCESS; }

uint64_t EmbeddingNoContiguousTiling::GetTilingKey() const
{
    uint64_t key = 0;
    if (isSimd_) {
        key = (indicesDtype_ == ge::DT_INT64) ? NO_CONTIGUOUS_SIMD_BASE_KEY_INT64 : NO_CONTIGUOUS_SIMD_BASE_KEY_INT32;
    } else {
        key = NO_CONTIGUOUS_BASE_KEY;
        key += static_cast<uint64_t>(enableInt64_ * DIGIT_TEN + xDtypeSize_);
    }
    return key;
}

ge::graphStatus EmbeddingNoContiguousTiling::GetWorkspaceSize()
{
    auto workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = DEFAULT_WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus EmbeddingNoContiguousTiling::PostTiling()
{
    context_->SetBlockDim(usedCoreNum_);
    if (!isSimd_) {
        uint64_t localMemorySize = static_cast<uint64_t>(ubSize_) - static_cast<uint64_t>(SIMT_CACHE_SIZE);
        context_->SetLocalMemorySize(localMemorySize);
    }

    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData());
    if (isSimd_) {
        if (simdTilingData_.GetDataSize() > context_->GetRawTilingData()->GetCapacity()) {
            return ge::GRAPH_FAILED;
        }
        simdTilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(),
                                     context_->GetRawTilingData()->GetCapacity());
        context_->GetRawTilingData()->SetDataSize(simdTilingData_.GetDataSize());
    } else {
        if (m_tilingData_.GetDataSize() > context_->GetRawTilingData()->GetCapacity()) {
            return ge::GRAPH_FAILED;
        }
        m_tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(),
                                   context_->GetRawTilingData()->GetCapacity());
        context_->GetRawTilingData()->SetDataSize(m_tilingData_.GetDataSize());
    }
    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE("Embedding", EmbeddingNoContiguousTiling, 0);
} // namespace optiling

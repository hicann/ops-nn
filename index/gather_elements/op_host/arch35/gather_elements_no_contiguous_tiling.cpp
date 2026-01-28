/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file gather_elements_no_contiguous_tiling.cc
 * \brief
 */

#include "gather_elements_no_contiguous_tiling.h"
#include "platform/platform_info.h"
#include "gather_elements_tiling.h"
#include "util/math_util.h"

using namespace std;

namespace optiling {
static constexpr int64_t IN_X_IDX = 0;
static constexpr int64_t IN_INDEX_IDX = 1;
static constexpr int64_t OUT_Y_IDX = 0;
static constexpr int64_t ATTR_DIM_IDX = 0;
static constexpr size_t TWO = 2;
static constexpr size_t THREE = 3;
static constexpr int64_t SIMT_CACHE_LINE = 128;
static int64_t SIMT_CACHE_SIZE = static_cast<int64_t>(128 * 1024);
static constexpr uint64_t NO_CONTIGUOUS_BASE_KEY = 20000;
static constexpr uint64_t INDEX_TRANSPOSE_BASE_KEY = 1000;
static constexpr int64_t INT32_MAX_BOUND = 2147483647;
static constexpr size_t MAX_SUPPORT_DIM_NUM = 8;

bool GatherElementsNoContiguousTiling::IsCapable()
{
    if ((!IsContiguous(xShape_, xStride_) || !IsContiguous(indexShape_, indexStride_) ) 
        && indexShape_.GetDimNum() <= MAX_SUPPORT_DIM_NUM) {
        return true;
    }
    return false;
}

ge::graphStatus GatherElementsNoContiguousTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        auto compileInfoPtr = reinterpret_cast<const GatherElementsCompileInfo *>(context_->GetCompileInfo());
        OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context_, "compile info is null"),
            return ge::GRAPH_FAILED);
        coreNum_ = static_cast<int64_t>(compileInfoPtr->core_num);
        ubSize_ = static_cast<int64_t>(compileInfoPtr->ub_size);
        OP_LOGD(opName_, "Get aivNum form compileInfo is: %ld", coreNum_);
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        coreNum_ = static_cast<int64_t>(ascendcPlatform.GetCoreNumAiv());
        uint64_t ubSizePlatForm;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        ubSize_ = static_cast<int64_t>(ubSizePlatForm);
        OP_LOGD(opName_, "Get aivNum form ascendcPlatform is: %ld", coreNum_);
    }
    OP_CHECK_IF((coreNum_ <= 0 || ubSize_ <= 0),
        OP_LOGE(opName_,
        "coreNum and ubSize should not be samller than 0, but got coreNum [%ld] and ubSize [%ld], please check.",
        coreNum_, ubSize_), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

inline bool GatherElementsNoContiguousTiling::ParamTypeIsInvalid(ge::DataType &x)
{
    std::set<ge::DataType> supportedDtype = {
        ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16,  ge::DT_BOOL,
        ge::DT_INT8,  ge::DT_UINT8,   ge::DT_INT16, ge::DT_UINT16,
        ge::DT_INT32, ge::DT_UINT32,  ge::DT_INT64, ge::DT_UINT64 };
    return supportedDtype.count(x) == 0;
}

ge::graphStatus GatherElementsNoContiguousTiling::GetContiguousTensorInfo(gert::Shape &shape, gert::Stride &stride, size_t idx, bool isOut) 
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

ge::graphStatus GatherElementsNoContiguousTiling::GetTensorInfo(gert::Shape &shape, gert::Stride &stride, size_t idx, bool isOut) 
{
    if (isOut) {
        GetContiguousTensorInfo(shape, stride, idx , isOut);
    } else {
        bool isView = context_->InputIsView(idx);
        if (isView) {
            auto* inputStride = context_->GetInputStride(idx);
            if (inputStride == nullptr || inputStride->GetDimNum() == 0 ) {
                GetContiguousTensorInfo(shape, stride, idx , isOut);
            } else {
                stride = *inputStride;
                shape = context_->GetInputShape(idx)->GetShape();
            } 
        } else {
            GetContiguousTensorInfo(shape, stride, idx , isOut);
        }
    }
    std::string info = isOut ? "output" : "input";
    OP_CHECK_IF(shape.GetDimNum() != stride.GetDimNum(),
        OP_LOGE(opName_, "shape's dimNum [%lu] should be equal to strid's dimNum [%lu] for [%s] [%lu]", 
            shape.GetDimNum(), stride.GetDimNum(), info.c_str(), idx),
        return ge::GRAPH_FAILED); 
    return ge::GRAPH_SUCCESS;
}

inline bool GatherElementsNoContiguousTiling::IsEnableInt64(gert::Shape shape, gert::Stride stride) 
{
    int64_t maxSize = 0;
    for(size_t i = 0; i < shape.GetDimNum(); i++) {
        maxSize = std::max(maxSize, shape.GetDim(i) * stride[i]);
    }
    return maxSize > INT32_MAX_BOUND;
}

ge::graphStatus GatherElementsNoContiguousTiling::GetInAndOutInfo()
{
    // check x & index & y type
    auto xDesc = context_->GetRequiredInputDesc(IN_X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
    xDtype_ = xDesc->GetDataType();
    OP_CHECK_IF(ParamTypeIsInvalid(xDtype_), OP_LOGE(opName_,
        "x dtype should be float,float16,bfloat16,bool,int8,uint8,int16,uint16,int32,uint32,int64,uint64, but got [%s], please check.",
        Ops::Base::ToString(xDtype_).c_str()), return ge::GRAPH_FAILED);
    auto indexDesc = context_->GetRequiredInputDesc(IN_INDEX_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, indexDesc);
    indexDtype_ = indexDesc->GetDataType();
    OP_CHECK_IF((indexDtype_ != ge::DataType::DT_INT32) && (indexDtype_ != ge::DataType::DT_INT64),
        OP_LOGE(opName_, "index dtype should be int32,int64, please check."),
        return ge::GRAPH_FAILED);
    auto yDesc = context_->GetOutputDesc(OUT_Y_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yDesc);
    auto yDtype = yDesc->GetDataType();
    OP_CHECK_IF(yDtype != xDtype_, OP_LOGE(opName_,
        "The input x and output y should have same dtype, please check."),
        return ge::GRAPH_FAILED);
    GetTensorInfo(xShape_, xStride_, IN_X_IDX, false);
    GetTensorInfo(indexShape_, indexStride_, IN_INDEX_IDX, false);
    GetTensorInfo(yShape_, yStride_, OUT_Y_IDX, true);
    dimSize_ = static_cast<int64_t>(indexShape_.GetDimNum());

    OP_CHECK_IF(dimSize_ > MAX_DIM_LEN_EIGHT,
        OP_LOGE(opName_, "dimSize should not larger than 8, got %ld.", dimSize_),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(yShape_ != indexShape_,
        OP_LOGE(opName_, "The input index and output y should have same shape."),
        return ge::GRAPH_FAILED);

    // 非连续场景要用size*stride去判断
    bool xEnableInt64 = IsEnableInt64(xShape_, xStride_);
    bool indexEnableInt64 = IsEnableInt64(indexShape_, indexStride_);
    bool yEnableInt64 = IsEnableInt64(yShape_, yStride_);
    enableInt64_ = (xEnableInt64 || indexEnableInt64 || yEnableInt64) ? 1 : 0;
    ySize_ = indexShape_.GetShapeSize();
    xDtypeSize_ = ge::GetSizeByDataType(xDtype_);
    OP_CHECK_IF((xDtypeSize_ == -1),
        OP_LOGE(opName_, "get xDtypeSize fail"),
        return ge::GRAPH_FAILED);
        indexDtypeSize_ = ge::GetSizeByDataType(indexDtype_);
    OP_CHECK_IF((indexDtypeSize_ == -1),
        OP_LOGE(opName_, "get indexDtypeSize fail"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

inline bool GatherElementsNoContiguousTiling::IsContiguous(const gert::Shape &xShape, const gert::Stride &xStride)
{
    int64_t validStride = 1;
    for(int64_t i = static_cast<int64_t>(xShape.GetDimNum()) - 1; i >= 0; i--) {
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

inline bool GatherElementsNoContiguousTiling::CheckIsIndexTranspose()
{
    if (xShape_.GetDimNum() == THREE && IsContiguous(xShape_, xStride_) && IsContiguous(yShape_, yStride_) && axis_ == TWO
        && indexStride_[0] == 0 && indexShape_[1] * indexShape_[TWO] * indexDtypeSize_ > SIMT_CACHE_SIZE
        && yShape_[TWO] * xDtypeSize_ > SIMT_CACHE_LINE) {
        return true;
    }
    return false;
}

inline ge::graphStatus GatherElementsNoContiguousTiling::GetAttrInfo()
{
    auto const attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    auto *axis = attrs->GetAttrPointer<int32_t>(ATTR_DIM_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, axis);

    auto axisVal = static_cast<int64_t>(*axis);
    OP_CHECK_IF(axisVal < -dimSize_ || axisVal >= dimSize_,
        OP_LOGE(opName_, "axis value should between with [%ld, %ld], but got %ld.", -dimSize_,
        dimSize_ - 1, axisVal),
        return ge::GRAPH_FAILED);

    axis_ = axisVal < 0 ? axisVal + dimSize_ : axisVal;
    for (int64_t i = 0; i < dimSize_; i++) {
        if (i == axis_) {
            continue;
        }
        OP_CHECK_IF(xShape_.GetDim(i) < indexShape_.GetDim(i),
            OP_LOGE(opName_,
            "x should larger than or equal to index of each dim value, except axis."),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GatherElementsNoContiguousTiling::GetShapeAttrsInfo()
{
    opName_ = context_->GetNodeName();

    auto res = GetInAndOutInfo();
    if (res != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    auto resAttr = GetAttrInfo();
    isIndexTranspose_ = CheckIsIndexTranspose();
    if (resAttr != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return resAttr;
}

void GatherElementsNoContiguousTiling::CalcuCore()
{
    if (Ops::Base::CeilDiv(ySize_, threadNum_) < (coreNum_ / NUM_TWO)) {
        threadNum_ = SMALL_CASE_THREAD_NUM;
    }
    if (ySize_ < threadNum_) {
        usedCoreNum_ = 1;
        perCoreNum_ = ySize_;
        tailCoreNum_ = ySize_;
        return;
    }

    // 对齐到threadNum_的倍数
    perCoreNum_ = Ops::Base::CeilDiv(ySize_, coreNum_);
    usedCoreNum_ = Ops::Base::CeilDiv(ySize_, perCoreNum_);
    tailCoreNum_ = ySize_ - perCoreNum_ * (usedCoreNum_ - 1);
}

void GatherElementsNoContiguousTiling::SetTilingData()
{
    int64_t indexShape[ARRAY_LEN_EIGHT] = {0};
    int64_t xShape[ARRAY_LEN_EIGHT] = {0};
    int64_t xStride[ARRAY_LEN_EIGHT] = {0};
    int64_t indexStride[ARRAY_LEN_EIGHT] = {0};
    int64_t yStride[ARRAY_LEN_EIGHT] = {0};
    int64_t preDimNum = ARRAY_LEN_EIGHT - indexShape_.GetDimNum();
    for (int64_t i = 0; i < static_cast<int64_t>(indexShape_.GetDimNum()); i++) {
        indexShape[preDimNum + i] = indexShape_.GetDim(i);
        xShape[preDimNum + i] = xShape_.GetDim(i);
        xStride[preDimNum + i] = xStride_[i];
        indexStride[preDimNum + i] = indexStride_[i];
        yStride[preDimNum + i] = yStride_[i];
    }
    m_tilingData_.set_axis(axis_);
    m_tilingData_.set_usedCore(usedCoreNum_);
    m_tilingData_.set_perCoreNum(perCoreNum_);
    m_tilingData_.set_tailCoreNum(tailCoreNum_);
    m_tilingData_.set_indexShape(indexShape);
    m_tilingData_.set_xShape(xShape);
    m_tilingData_.set_xStride(xStride);
    m_tilingData_.set_indexStride(indexStride);
    m_tilingData_.set_yStride(yStride);
}

template <typename T>
static std::string ToString(const T* value, size_t size) {
  std::string r = "[";
  for (size_t i = 0; i < size; i++) {
    r = r + std::to_string(value[i]) + ", ";
  }
  r = r + "]";
  return r;
}

void GatherElementsNoContiguousTiling::PrintTilingData()
{
    OP_LOGI(opName_, "axis = %ld, usedCore = %ld, perCoreNum = %ld, tailCoreNum = %ld, \
xStride = %s, indexStride = %s, yStride = %s, indexShape = %s, xShape = %s",
        m_tilingData_.get_axis(), m_tilingData_.get_usedCore(),
        m_tilingData_.get_perCoreNum(), m_tilingData_.get_tailCoreNum(),
        ToString(m_tilingData_.get_xStride(), MAX_DIM_LEN_EIGHT).c_str(),
        ToString(m_tilingData_.get_indexStride(), MAX_DIM_LEN_EIGHT).c_str(),
        ToString(m_tilingData_.get_yStride(), MAX_DIM_LEN_EIGHT).c_str(),
        ToString(m_tilingData_.get_indexShape(), MAX_DIM_LEN_EIGHT).c_str(),
        ToString(m_tilingData_.get_xShape(), MAX_DIM_LEN_EIGHT).c_str());
}

bool GatherElementsNoContiguousTiling::CanMerge(int64_t* xShape, int64_t* xStride, int index) {
    return (xStride[index] == xShape[index + 1] * xStride[index + 1]);
}

void GatherElementsNoContiguousTiling::InitVector(vector<int64_t> &tempXShape, vector<int64_t> &tempXStride, vector<int64_t> &tempIndexShape, vector<int64_t> &tempIndexStride, vector<int64_t> &tempYStride)
{
    tempXShape.resize(xShape_.GetDimNum());
    tempXStride.resize(xStride_.GetDimNum());
    tempIndexShape.resize(indexShape_.GetDimNum());
    tempIndexStride.resize(indexStride_.GetDimNum());
    tempYStride.resize(yStride_.GetDimNum());
    for (int64_t i = 0; i < static_cast<int64_t>(xShape_.GetDimNum()); i++) {
        tempXShape[i] = xShape_.GetDim(i);
        tempXStride[i] = xStride_[i];
    }
    for (int64_t i = 0; i < static_cast<int64_t>(indexShape_.GetDimNum()); i++) {
        tempIndexShape[i] = indexShape_.GetDim(i);
        tempIndexStride[i] = indexStride_[i];
        tempYStride[i] = yStride_[i];
    }
}

void GatherElementsNoContiguousTiling::DoCoalesce(vector<int64_t> &tempXShape, vector<int64_t> &tempXStride, vector<int64_t> &tempIndexShape, vector<int64_t> &tempIndexStride, vector<int64_t> &tempYStride)
{
    bool changed = true;
    while (changed && tempXShape.size() > 1) {
        changed = false;
        for (int64_t i = static_cast<int64_t>(tempXShape.size()) - 2; i >= 0; i--) {
            const bool is_gather_dim = (i == axis_ ||i + 1 == axis_);
            if (is_gather_dim || tempXShape[i] != tempIndexShape[i] || tempXShape[i + 1] != tempIndexShape[i + 1]) {
                //无法合轴
                continue;
            }
            bool merge_x = CanMerge(tempXShape.data(), tempXStride.data(), i);
            bool merge_indices = CanMerge(tempIndexShape.data(), tempIndexStride.data(), i);
            bool merge_y = CanMerge(tempIndexShape.data(), tempYStride.data(), i);
            if (!merge_x || !merge_indices || !merge_y) {
                continue;
            }
            tempXShape[i] *= tempXShape[i+1];
            tempXStride[i] = tempXStride[i+1];
            tempIndexShape[i] *= tempIndexShape[i+1];
            tempIndexStride[i] = tempIndexStride[i+1];
            tempYStride[i] = tempYStride[i+1];
            tempXShape.erase(tempXShape.begin() + i + 1);
            tempXStride.erase(tempXStride.begin() + i + 1);
            tempIndexShape.erase(tempIndexShape.begin() + i + 1);
            tempIndexStride.erase(tempIndexStride.begin() + i + 1);
            tempYStride.erase(tempYStride.begin() + i + 1);
            if (i < axis_) {
                axis_--;
            }
            changed = true;
            break;
        }
    }
}

void GatherElementsNoContiguousTiling::UpdateResultFromVector(vector<int64_t> &tempXShape, vector<int64_t> &tempXStride, vector<int64_t> &tempIndexShape, vector<int64_t> &tempIndexStride, vector<int64_t> &tempYStride)
{
    xShape_.SetDimNum(tempXShape.size());
    xStride_.SetDimNum(tempXStride.size());
    indexShape_.SetDimNum(tempIndexShape.size());
    indexStride_.SetDimNum(tempIndexStride.size());
    yStride_.SetDimNum(tempYStride.size());
    for (size_t i = 0; i < tempXShape.size(); i++) {
        xShape_[i] = tempXShape[i];
        xStride_[i] = tempXStride[i];
    }
    for (size_t i = 0; i < tempIndexShape.size(); i++) {
        indexShape_[i] = tempIndexShape[i];
        indexStride_[i] = tempIndexStride[i];
        yStride_[i] = tempYStride[i];
    }
    dimSize_ = static_cast<int64_t>(indexShape_.GetDimNum());
}

void GatherElementsNoContiguousTiling::CoalesceGatherElements()
{
    vector<int64_t> tempXShape, tempXStride, tempIndexShape, tempIndexStride, tempYStride;
    InitVector(tempXShape, tempXStride, tempIndexShape, tempIndexStride, tempYStride);
    DoCoalesce(tempXShape, tempXStride, tempIndexShape, tempIndexStride, tempYStride);
    UpdateResultFromVector(tempXShape, tempXStride, tempIndexShape, tempIndexStride, tempYStride);
}

ge::graphStatus GatherElementsNoContiguousTiling::DoOpTiling()
{
    if (!isIndexTranspose_) {
        CoalesceGatherElements();
    }
    CalcuCore();
    SetTilingData();
    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GatherElementsNoContiguousTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t GatherElementsNoContiguousTiling::GetTilingKey() const
{
    uint64_t key = NO_CONTIGUOUS_BASE_KEY;
    key += static_cast<uint64_t>(dimSize_ * DIGIT_HUNDRED + enableInt64_ * DIGIT_TEN + axis_);
    if (isIndexTranspose_) {
        key += INDEX_TRANSPOSE_BASE_KEY;
    }
    return key;
}

ge::graphStatus GatherElementsNoContiguousTiling::GetWorkspaceSize()
{
    auto workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = DEFAULT_WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GatherElementsNoContiguousTiling::PostTiling()
{
    context_->SetBlockDim(usedCoreNum_);
    uint64_t localMemorySize = static_cast<uint64_t>(ubSize_) - static_cast<uint64_t>(SIMT_CACHE_SIZE);
    context_->SetLocalMemorySize(localMemorySize);

    if (m_tilingData_.GetDataSize() > context_->GetRawTilingData()->GetCapacity()) {
        return ge::GRAPH_FAILED;
    }
    m_tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(m_tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(GatherElements, GatherElementsNoContiguousTiling, 10);
} // namespace optiling
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
 * \file gather_elements_tiling_arch35.cpp
 * \brief
 */

#include "gather_elements_tiling_arch35.h"
#include "platform/platform_info.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "op_common/atvoss/reduce/reduce_tiling.h"

namespace optiling {
static constexpr int64_t IN_X_IDX = 0;
static constexpr int64_t IN_INDEX_IDX = 1;
static constexpr int64_t OUT_Y_IDX = 0;
static constexpr int64_t ATTR_DIM_IDX = 0;
static constexpr int64_t MIN_IDX_AFTER_AXIS = 32;
static constexpr int64_t MIN_RATIO = 32;
static constexpr int64_t TWO = 2;
static constexpr int64_t THREE = 3;
static constexpr int64_t INT8_BYTES = 1;
static constexpr int64_t INT16_BYTES = 2;
static constexpr int64_t INT64_BYTES = 8;
static constexpr int64_t UINT16_MAX_NUM = 65535;
static int64_t SIMT_CACHE_SIZE = static_cast<int64_t>(32 * 1024);
static constexpr int64_t DOUBLE = 2;

void GatherElementsSimtTiling::Reset()
{
    opName_ = nullptr;
}

bool GatherElementsSimtTiling::IsCapable()
{
    return true;
}

ge::graphStatus GatherElementsSimtTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        auto compileInfoPtr = static_cast<const GatherElementsCompileInfo *>(context_->GetCompileInfo());
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
    aicoreParams_.blockDim = coreNum_;
    aicoreParams_.ubSize = static_cast<int64_t>(ubSize_ - SIMT_CACHE_SIZE);
    OP_CHECK_IF((aicoreParams_.ubSize <= 0),
        OP_LOGE(opName_,
        "ubSize should be bigger than SIMT_CACHE_SIZE [%ld], but got ubSize [%ld], please check.",
        SIMT_CACHE_SIZE, ubSize_), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

inline bool GatherElementsSimtTiling::ParamTypeIsInvalid(ge::DataType &x)
{
    std::set<ge::DataType> supportedDtype = {
        ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16,  ge::DT_BOOL,
        ge::DT_INT8,  ge::DT_UINT8,   ge::DT_INT16, ge::DT_UINT16,
        ge::DT_INT32, ge::DT_UINT32,  ge::DT_INT64, ge::DT_UINT64 };
    return supportedDtype.count(x) == 0;
}

ge::graphStatus GatherElementsSimtTiling::GetInAndOutInfo()
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

    auto xStorageShape = context_->GetInputShape(IN_X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xStorageShape);
    xShape_ = xStorageShape->GetStorageShape();
    enableInt64_ = (xShape_.GetShapeSize() > INT32_MAX_BOUND) ? 1 : 0;

    auto indexStorageShape = context_->GetInputShape(IN_INDEX_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, indexStorageShape);
    indexShape_ = indexStorageShape->GetStorageShape();

    auto yTensor = context_->GetOutputShape(OUT_Y_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, yTensor);
    yShape_ = yTensor->GetStorageShape();

    dimSize_ = static_cast<int64_t>(indexShape_.GetDimNum());
    OP_CHECK_IF(dimSize_ > MAX_DIM_LEN_EIGHT,
        OP_LOGE(opName_, "dimSize should not larger than 8, got %ld.", dimSize_),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(yShape_ != indexShape_,
        OP_LOGE(opName_, "The input index and output y should have same shape."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus GatherElementsSimtTiling::GetAttrInfo()
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

ge::graphStatus GatherElementsSimtTiling::GetShapeAttrsInfo()
{
    opName_ = context_->GetNodeName();

    auto res = GetInAndOutInfo();
    if (res != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    auto resAttr = GetAttrInfo();
    if (resAttr != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return resAttr;
}

inline int GatherElementsSimtTiling::HandleCount(int &count, int i, int j)
{
    int val = 1;
    for (int k = i; k < j; k++) {
        val *= xShape_[k];
    }
    if (j <= axis_) {
        count += (j - i - 1);
    }
    return val;
} 

void GatherElementsSimtTiling::CalculateFullLoadCondition(int64_t xDtypeSize, int64_t indexDtypeSize) {
    xAfterAxis_ = xShape_.GetDim(axis_);
    idxAfterAxis_ = indexShape_.GetDim(axis_);
    for (int64_t i = axis_ + 1; i < dimSize_; i++) {
        xAfterAxis_ = xAfterAxis_ * xShape_.GetDim(i);
        idxAfterAxis_ = idxAfterAxis_ * indexShape_.GetDim(i);
    }
    OP_CHECK_IF((idxAfterAxis_ == static_cast<int64_t>(0)),
        OP_LOGE(opName_, "idxAfterAxis_ is 0"),
        return );
    int64_t xPerAxisEncludeDim1 = 1;
    int64_t idxPerAxisEncludeDim1 = 1;
    for (int64_t i = 1; i < axis_; i++) {
        xPerAxisEncludeDim1 = xPerAxisEncludeDim1 * xShape_.GetDim(i);
        idxPerAxisEncludeDim1 = idxPerAxisEncludeDim1 * indexShape_.GetDim(i);
    }
    uint32_t blockSize = Ops::Base::GetUbBlockSize(context_);
    uint64_t xPerUbSize = static_cast<uint64_t>(xAfterAxis_ * xDtypeSize);
    uint64_t yPerUbSize = static_cast<uint64_t>(idxAfterAxis_ * xDtypeSize);
    uint64_t idxPerUbSize = static_cast<uint64_t>(idxAfterAxis_ * indexDtypeSize);
    xLoadInUbNum_ = ubSize_ / ((Ops::Base::CeilAlign(xPerUbSize, static_cast<uint64_t>(blockSize)) +
                    Ops::Base::CeilAlign(yPerUbSize, static_cast<uint64_t>(blockSize)) + Ops::Base::CeilAlign(idxPerUbSize, static_cast<uint64_t>(blockSize))) * 
                    DOUBLE);
    xUbFactor_ = xAfterAxis_ * xLoadInUbNum_;
    indexLoadInUbNum_ = xLoadInUbNum_;
    if (xLoadInUbNum_ > 0 && xShape_.GetDim(axis_) <= UINT16_MAX_NUM && idxAfterAxis_ > MIN_IDX_AFTER_AXIS && dimSize_ - axis_ <= THREE &&
        xPerAxisEncludeDim1 / idxPerAxisEncludeDim1 <= TWO && xAfterAxis_ / idxAfterAxis_ < MIN_RATIO) {
        isFullLoad_ = 1;
    }
}

void GatherElementsSimtTiling::ComputeCoreNum()
{
    int64_t outPutPerAxis = 1;
    int64_t perCoreNum = 0;
    int64_t tailCoreNum = 0;
    for (int64_t i = 0; i < axis_; i++) {
        outPutPerAxis = outPutPerAxis * yShape_.GetDim(i);
    }

    if (coreNum_ >= outPutPerAxis) {
        usedCoreNum_ = outPutPerAxis;
        perCoreNum = 1;
        tailCoreNum = 1;
        xLoadInUbNum_ = 1;
        indexLoadInUbNum_ = 1;
    } else {
        perCoreNum = Ops::Base::CeilDiv(outPutPerAxis, coreNum_);
        usedCoreNum_ = Ops::Base::CeilDiv(outPutPerAxis, perCoreNum);
        tailCoreNum = outPutPerAxis - (usedCoreNum_ - 1) * perCoreNum;
        indexLoadInUbNum_ = indexLoadInUbNum_ > perCoreNum ? perCoreNum : indexLoadInUbNum_;
    }
    m_tilingData_.set_perCoreNum(perCoreNum);
    m_tilingData_.set_tailCoreNum(tailCoreNum);
}

static std::vector<int64_t> ToVector(const gert::Shape& shape)
{
    size_t shapeSize = shape.GetDimNum();
    std::vector<int64_t> shapeVec(shapeSize, 0);

    for (size_t i = 0; i < shapeSize; i++) {
        shapeVec[i] = shape.GetDim(i);
    }
    return shapeVec;
}

void GatherElementsSimtTiling::ComputeStride()
{
    xShape8d_.SetDimNum(MAX_DIM_LEN_EIGHT);
    indexShape8d_.SetDimNum(MAX_DIM_LEN_EIGHT);
    for (int64_t i = 0; i < MAX_DIM_LEN_EIGHT; i++) {
        if (i < MAX_DIM_LEN_EIGHT - static_cast<int64_t>(xShape_.GetDimNum())) {
            xShape8d_.SetDim(i, 0);
            indexShape8d_.SetDim(i, 0);
        } else {
            xShape8d_.SetDim(i, xShape_.GetDim(i - (MAX_DIM_LEN_EIGHT - xShape_.GetDimNum())));
            indexShape8d_.SetDim(i, indexShape_.GetDim(i - (MAX_DIM_LEN_EIGHT - indexShape_.GetDimNum())));
        }
    }

    xStride_.SetDimNum(MAX_DIM_LEN_EIGHT);
    indexStride_.SetDimNum(MAX_DIM_LEN_EIGHT);
    for (int64_t index = 0; index < MAX_DIM_LEN_EIGHT; index++) {
        int64_t xStride = 1;
        int64_t indexStride = 1;
        for (int64_t j = index + 1; j < MAX_DIM_LEN_EIGHT; ++j) {
            xStride *= xShape8d_[j];
            indexStride *= indexShape8d_[j];
        }
        if (index < (MAX_DIM_LEN_EIGHT - static_cast<int64_t>(xShape_.GetDimNum()))) {
            xStride_.SetDim(index, 1);
            indexStride_.SetDim(index, 1);
        } else {
            xStride_.SetDim(index, xStride);
            indexStride_.SetDim(index, indexStride);
        }
    }
    xShapeArr_ = ToVector(xShape8d_);
    indexShapeArr_ = ToVector(indexShape8d_);
    xStrideArr_ = ToVector(xStride_);
    indexStrideArr_ = ToVector(indexStride_);
    OP_LOGD(opName_, "after xShape convert to vector = %s", Ops::Base::VectorToString(xShapeArr_).c_str());
    OP_LOGD(opName_, "after indexShape convert to vector = %s", Ops::Base::VectorToString(indexShapeArr_).c_str());
    OP_LOGD(opName_, "after xStride convert to vector = %s", Ops::Base::VectorToString(xStrideArr_).c_str());
    OP_LOGD(opName_, "after indexStride convert to vector = %s", Ops::Base::VectorToString(indexStrideArr_).c_str());

    int64_t lengthIsEightArray[MAX_DIM_LEN_EIGHT] = {0, 0, 0, 0, 0, 0, 0, 0};
    std::copy(xStrideArr_.begin(), xStrideArr_.end(), lengthIsEightArray);
    m_tilingData_.set_xStrideArr(lengthIsEightArray);
    std::copy(indexStrideArr_.begin(), indexStrideArr_.end(), lengthIsEightArray);
    m_tilingData_.set_indexStrideArr(lengthIsEightArray);
}

void GatherElementsSimtTiling::SetTilingData()
{
    int64_t indexShape[TILING_ARRAY_LEN_EIGHT] = {0};
    int64_t xShape[TILING_ARRAY_LEN_EIGHT] = {0};
    for (int64_t i = 0; i < static_cast<int64_t>(indexShape_.GetDimNum()); i++) {
        indexShape[i] = indexShape_.GetDim(i);
        xShape[i] = xShape_.GetDim(i);
    }
    m_tilingData_.set_axis(axis_);
    m_tilingData_.set_usedCore(usedCoreNum_);
    m_tilingData_.set_indexShape(indexShape);
    m_tilingData_.set_xShape(xShape);
    m_tilingData_.set_xLoadInUbNum(xLoadInUbNum_);
    m_tilingData_.set_indexLoadInUbNum(indexLoadInUbNum_);
    m_tilingData_.set_xUbFactor(xUbFactor_);
    m_tilingData_.set_indexUbFactor(indexUbFactor_);
    m_tilingData_.set_xAfterAxis(xAfterAxis_);
    m_tilingData_.set_idxAfterAxis(idxAfterAxis_);
}

void GatherElementsSimtTiling::MergeAxis()
{
    int count = 0;
    int i = 0;
    while (i < dimSize_) {
        if (xShape_[i] != indexShape_[i] || i == axis_) {
            xShapeMerge_.AppendDim(xShape_[i]);
            indexShapeMerge_.AppendDim(indexShape_[i]);
        } else {
            int j = i;
            while (j < dimSize_ && xShape_[j] == indexShape_[j] && j != axis_) {
                j++;
            }
            if (j - i > 1) {
                int val = HandleCount(count, i, j);
                xShapeMerge_.AppendDim(val);
                indexShapeMerge_.AppendDim(val);
            } else {
                xShapeMerge_.AppendDim(xShape_[i]);
                indexShapeMerge_.AppendDim(indexShape_[i]);
            }
            i = j - 1;
        }
        i++;
    }

    axis_ -= static_cast<int64_t>(count);
    dimSize_ = static_cast<int64_t>(indexShapeMerge_.GetDimNum());
    OP_LOGD(opName_, "after xShapeMerge is = %s", Ops::Base::ToString(xShapeMerge_).c_str());
    OP_LOGD(opName_, "after indexShapeMerge is = %s", Ops::Base::ToString(indexShapeMerge_).c_str());

    m_tilingData_.set_axis(axis_);
}

void GatherElementsSimtTiling::ReductionDim()
{
    // compute size
    ySize_ = indexShapeMerge_.GetShapeSize();

    // conert to 8 dim
    xShape8d_.SetDimNum(MAX_DIM_LEN_EIGHT);
    indexShape8d_.SetDimNum(MAX_DIM_LEN_EIGHT);
    for (int i = 0; i < MAX_DIM_LEN_EIGHT; i++) {
        if (i < (MAX_DIM_LEN_EIGHT - dimSize_)) {
            xShape8d_.SetDim(i, 0);
            indexShape8d_.SetDim(i, 0);
        } else {
            xShape8d_.SetDim(i, xShapeMerge_[i + dimSize_ - MAX_DIM_LEN_EIGHT]);
            indexShape8d_.SetDim(i, indexShapeMerge_[i + dimSize_ - MAX_DIM_LEN_EIGHT]);
        }
    }

    xStride_.SetDimNum(MAX_DIM_LEN_EIGHT);
    indexStride_.SetDimNum(MAX_DIM_LEN_EIGHT);
    for (int index = 0; index < MAX_DIM_LEN_EIGHT; index++) {
        int64_t xStride = 1;
        int64_t indexStride = 1;
        for (int j = index + 1; j < MAX_DIM_LEN_EIGHT; ++j) {
            xStride *= xShape8d_[j];
            indexStride *= indexShape8d_[j];
        }
        if (index < (MAX_DIM_LEN_EIGHT - dimSize_)) {
            xStride_.SetDim(index, 1);
            indexStride_.SetDim(index, 1);
        } else {
            xStride_.SetDim(index, xStride);
            indexStride_.SetDim(index, indexStride);
        }
    }
    xShapeArr_ = ToVector(xShape8d_);
    xStrideArr_ = ToVector(xStride_);
    indexShapeArr_ = ToVector(indexShape8d_);
    indexStrideArr_ = ToVector(indexStride_);
    OP_LOGD(opName_, "after xShape convert to vector = %s", Ops::Base::VectorToString(xShapeArr_).c_str());
    OP_LOGD(opName_, "after xStride convert to vector = %s", Ops::Base::VectorToString(xStrideArr_).c_str());
    OP_LOGD(opName_, "after indexShape convert to vector = %s", Ops::Base::VectorToString(indexShapeArr_).c_str());
    OP_LOGD(opName_, "after indexStride convert to vector = %s", Ops::Base::VectorToString(indexStrideArr_).c_str());

    int64_t tempEightArray[MAX_DIM_LEN_EIGHT] = {0, 0, 0, 0, 0, 0, 0, 0};
    std::copy(xStrideArr_.begin(), xStrideArr_.end(), tempEightArray);
    m_tilingData_.set_xStrideArr(tempEightArray);
    std::copy(indexStrideArr_.begin(), indexStrideArr_.end(), tempEightArray);
    m_tilingData_.set_indexStrideArr(tempEightArray);
    
    int64_t lengthIsEightArray[MAX_DIM_LEN_EIGHT] = {0, 0, 0, 0, 0, 0, 0, 0};
    std::copy(indexShapeArr_.begin(), indexShapeArr_.end(), lengthIsEightArray);
    m_tilingData_.set_indexShape(lengthIsEightArray);
    std::copy(xShapeArr_.begin(), xShapeArr_.end(), lengthIsEightArray);
    m_tilingData_.set_xShape(lengthIsEightArray);
}

void GatherElementsSimtTiling::CalcuCore()
{
    if (Ops::Base::CeilDiv(ySize_, threadNum_) < (coreNum_ / NUM_TWO)) {
        threadNum_ = SMALL_CASE_THREAD_NUM;
    }
    if (ySize_ < threadNum_) {
        m_tilingData_.set_usedCore(1);
        m_tilingData_.set_perCoreNum(ySize_);
        m_tilingData_.set_tailCoreNum(ySize_);
        return;
    }

    // 对齐到threadNum_的倍数
    int64_t perCoreNum = Ops::Base::CeilDiv(ySize_, coreNum_);
    int64_t usedCore = Ops::Base::CeilDiv(ySize_, perCoreNum);
    int64_t tailCoreNum = ySize_ - perCoreNum * (usedCore - 1);
    m_tilingData_.set_usedCore(usedCore);
    m_tilingData_.set_perCoreNum(perCoreNum);
    m_tilingData_.set_tailCoreNum(tailCoreNum);
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

void GatherElementsSimtTiling::PrintTilingData()
{
    OP_LOGI(opName_, "axis = %ld, usedCore = %ld, perCoreNum = %ld, tailCoreNum = %ld, \
xStrideArr = %s, indexStrideArr = %s",
        m_tilingData_.get_axis(), m_tilingData_.get_usedCore(),
        m_tilingData_.get_perCoreNum(), m_tilingData_.get_tailCoreNum(),
        ToString(m_tilingData_.get_xStrideArr(), MAX_DIM_LEN_EIGHT).c_str(),
        ToString(m_tilingData_.get_indexStrideArr(), MAX_DIM_LEN_EIGHT).c_str());
}
void GatherElementsSimtTiling::GetMagicAndShift()
{
    constexpr uint32_t MAX_DIM_LEN = 8;
    constexpr int64_t QUICK_DIV_NUM_32 = 32;
    for (int64_t i = static_cast<int64_t>(MAX_DIM_LEN) - dimSize_; i < M_SHIFT_OFFSET; i++) {
        shift_[i] = std::ceil(std::log2(indexStrideArr_[i]));
        magic_[i] = 
            std::ceil(std::exp2(shift_[i] + QUICK_DIV_NUM_32) / indexStrideArr_[i]) - std::exp2(QUICK_DIV_NUM_32);
    }
    m_tilingData_.set_shift(shift_);
    m_tilingData_.set_magic(magic_);
    return ;
}

ge::graphStatus GatherElementsSimtTiling::DoOpTiling()
{
    int64_t xDtypeSize = ge::GetSizeByDataType(xDtype_);
    OP_CHECK_IF((xDtypeSize == -1),
        OP_LOGE(opName_, "get xDtypeSize fail"),
        return ge::GRAPH_FAILED);
    int64_t indexDtypeSize = ge::GetSizeByDataType(indexDtype_);
    OP_CHECK_IF((indexDtypeSize == -1),
        OP_LOGE(opName_, "get indexDtypeSize fail"),
        return ge::GRAPH_FAILED);
    CalculateFullLoadCondition(xDtypeSize, indexDtypeSize);
    if (isFullLoad_ > 0) {
        ComputeCoreNum();
        indexUbFactor_ = idxAfterAxis_ * indexLoadInUbNum_;
        ComputeStride();
        SetTilingData();
        return ge::GRAPH_SUCCESS;
    }
    MergeAxis();
    ReductionDim();
    CalcuCore();
    GetMagicAndShift();
    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GatherElementsSimtTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t GatherElementsSimtTiling::GetTilingKey() const
{
    if (isFullLoad_) {
        return static_cast<uint64_t>(DIGIT_THOUSAND + dimSize_ * DIGIT_HUNDRED + axis_ * DIGIT_TEN);
    }
    return static_cast<uint64_t>(dimSize_ * DIGIT_HUNDRED + enableInt64_ * DIGIT_TEN + axis_);
}

ge::graphStatus GatherElementsSimtTiling::GetWorkspaceSize()
{
    auto workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = DEFAULT_WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GatherElementsSimtTiling::PostTiling()
{
    if (isFullLoad_) {
        context_->SetBlockDim(usedCoreNum_);
    } else {
        context_->SetBlockDim(m_tilingData_.get_usedCore());
    }
    if (m_tilingData_.GetDataSize() > context_->GetRawTilingData()->GetCapacity()) {
        return ge::GRAPH_FAILED;
    }
    m_tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(m_tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(GatherElements, GatherElementsSimtTiling, 90);
} // namespace optiling
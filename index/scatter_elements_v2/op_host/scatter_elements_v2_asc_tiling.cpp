/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file scatter_elements_tiling.cc
 * \brief
 */

#include "scatter_elements_v2_asc_tiling.h"
#include <vector>
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "platform/platform_info.h"
#include "error_util.h"
#include "scatter_elements_v2_tiling.h"

using namespace AscendC;
namespace optiling
{
constexpr int64_t DATA_IDX = 0;
constexpr int64_t INDICES_IDX = 1;
constexpr int64_t UPDATES_IDX = 2;
constexpr int64_t ATTR_AXIS_IDX = 0;
constexpr int64_t ATTR_REDUCTION_IDX = 1;

constexpr uint64_t REDUCTION_NONE = 0;
constexpr uint64_t REDUCTION_ADD = 1;
constexpr uint64_t REDUCTION_MUL = 2;

constexpr int64_t DB_BUFFER = 1;
constexpr int64_t ACTIVE_NODES_NUM = 2;
constexpr int64_t GM_ALIGN = 512;
constexpr int64_t USE_UB_MAX_SIZE = 65536;  // 64K
constexpr int64_t MAX_THREAD_NUM = 512;
constexpr int64_t MAX_INT32_NUM = 2147483647;
constexpr int64_t DCACHE_SIZE = 131072;                // 128K
constexpr int64_t ASCENDC_TOOLS_WORKSPACE = 16777216;  // 16M
constexpr int64_t DETERM_DB_BUFFER = 2;
constexpr int64_t DOUBLE_COUNT = 2;
constexpr int64_t BASE_S_MAX = 256;
constexpr int64_t UB_MIN_FACTOR = 1024;
constexpr int64_t SIMT_UB_RES_SIZE = 640;
constexpr uint32_t MAX_SORT_SPACE = 10240;

static const std::set<ge::DataType> SCAT_ELE_NONE_DTYPE = {ge::DT_FLOAT,  ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT64,
                                                           ge::DT_INT32,  ge::DT_INT16,   ge::DT_INT8, ge::DT_UINT8,
                                                           ge::DT_DOUBLE, ge::DT_BOOL};
static const std::set<ge::DataType> SCAT_ELE_ADD_DTYPE = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16,
                                                          ge::DT_INT64, ge::DT_INT32,   ge::DT_INT16,
                                                          ge::DT_INT8,  ge::DT_UINT8,   ge::DT_BOOL};
static const std::set<ge::DataType> SCAT_ELE_ADD_DETERM_DTYPE = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};
static const std::set<ge::DataType> SCAT_ELE_MUL_DTYPE = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT64,
                                                          ge::DT_INT32, ge::DT_INT16,   ge::DT_INT8, ge::DT_UINT8};

static const std::map<std::string, uint64_t> SCAT_ELE_REDUCTION = {{"none", 0}, {"add", 1}, {"mul", 2}};

template <typename T>
static std::string ToString(const T* value, size_t size) {
  std::string r = "[";
  for (size_t i = 0; i < size; i++) {
    r = r + std::to_string(value[i]) + ", ";
  }
  r = r + "]";
  return r;
}

bool ScatterElementsV2AscTiling::IsCapable()
{
    return true;
}

ge::graphStatus ScatterElementsV2AscTiling::GetPlatformInfo()
{
    auto compileInfo = reinterpret_cast<const ScatterElementsV2CompileInfo*>(context_->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo);
    totalCoreNum_ = compileInfo->totalCoreNum;
    ubSize_ = compileInfo->ubSizePlatForm;
    OP_TILING_CHECK((ubSize_ <= DCACHE_SIZE),
                    VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "ub size less than Dcache Size"),
                    return ge::GRAPH_FAILED);
    ubSize_ = ubSize_ - DCACHE_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterElementsV2AscTiling::GetShapeAttrsInfo()
{
    auto const attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    auto axis = attrs->GetAttrPointer<int64_t>(ATTR_AXIS_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, axis);
    int64_t dim = static_cast<int64_t>(*axis);

    auto dataShapePtr = context_->GetInputShape(DATA_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dataShapePtr);
    auto dataShape = dataShapePtr->GetStorageShape();
    rank_ = static_cast<int16_t>(dataShape.GetDimNum());

    int16_t dimMax = std::max(-1 * rank_, rank_ - 1);
    int16_t dimMin = std::min(-1 * rank_, rank_ - 1);
    OP_TILING_CHECK(
        (dim > dimMax || dim < dimMin),
        VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
                                        "attr axis must be in range of [%d, %d], but got [%ld].", dimMin, dimMax, dim),
        return ge::GRAPH_FAILED);

    dim_ = dim < 0 ? static_cast<int16_t>(dim) + rank_ : static_cast<int16_t>(dim);

    const char* reduction = attrs->GetAttrPointer<char>(ATTR_REDUCTION_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, reduction);
    std::string reductionStr = reduction;
    auto it = SCAT_ELE_REDUCTION.find(reductionStr);
    bool reductionInValid = it == SCAT_ELE_REDUCTION.end();
    OP_TILING_CHECK(reductionInValid,
                    VECTOR_INNER_ERR_REPORT_TILIING(
                        context_->GetNodeName(), "attr reduction only support none, add, mul currently, please check."),
                    return ge::GRAPH_FAILED);
    reduction_ = it->second;

    OP_TILING_CHECK(CheckInputDtype() != ge::GRAPH_SUCCESS,
                    VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "input dtype check failed."),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckInputShape() != ge::GRAPH_SUCCESS,
                    VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "input shape check failed."),
                    return ge::GRAPH_FAILED);

    bool isDetermType = reduction_ == 0 ||
                        (reduction_ == 1 && SCAT_ELE_ADD_DETERM_DTYPE.find(dtype_) != SCAT_ELE_ADD_DETERM_DTYPE.end());
    if (context_->GetDeterministic() && isDetermType) {
        isDeterministic_ = 1;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterElementsV2AscTiling::CheckXDtype(const ge::DataType dtype)
{
    if (reduction_ == REDUCTION_NONE) {
        OP_TILING_CHECK(
            (SCAT_ELE_NONE_DTYPE.find(dtype) == SCAT_ELE_NONE_DTYPE.end()),
            VECTOR_INNER_ERR_REPORT_TILIING(
                context_->GetNodeName(),
                "for reduction = none, data dtype only support float32, float16, double, uint8, int8, int32, \
int16, bool, int64, bfloat16 currently, please check."),
            return ge::GRAPH_FAILED);
    } else if (reduction_ == REDUCTION_ADD) {
        OP_TILING_CHECK((SCAT_ELE_ADD_DTYPE.find(dtype) == SCAT_ELE_ADD_DTYPE.end()),
                        VECTOR_INNER_ERR_REPORT_TILIING(
                            context_->GetNodeName(),
                            "for reduction = add, data dtype only support float32, float16, uint8, int8, int32, \
int16, bool, int64, bfloat16 currently, please check."),
                        return ge::GRAPH_FAILED);
    } else if (reduction_ == REDUCTION_MUL) {
        OP_TILING_CHECK((SCAT_ELE_MUL_DTYPE.find(dtype) == SCAT_ELE_MUL_DTYPE.end()),
                        VECTOR_INNER_ERR_REPORT_TILIING(
                            context_->GetNodeName(),
                            "for reduction = mul, data dtype only support float32, float16, uint8, int8, int32, \
int16, int64, bfloat16 currently, please check."),
                        return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterElementsV2AscTiling::CheckInputDtype()
{
    auto dataPtr = context_->GetInputDesc(DATA_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dataPtr);
    dtype_ = dataPtr->GetDataType();
    ge::graphStatus ret = CheckXDtype(dtype_);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    typeSize_ = ge::GetSizeByDataType(dtype_);
    OP_TILING_CHECK(typeSize_ <= 0, VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "get dataType size fail."),
                    return ge::GRAPH_FAILED);

    auto indicesPtr = context_->GetInputDesc(INDICES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, indicesPtr);
    indicesDtype_ = indicesPtr->GetDataType();
    bool dtypeInValid = indicesDtype_ != ge::DT_INT32 && indicesDtype_ != ge::DT_INT64;
    OP_TILING_CHECK(dtypeInValid,
                    VECTOR_INNER_ERR_REPORT_TILIING(
                        context_->GetNodeName(), "indices dtype only support int32 and int64 currently, please check."),
                    return ge::GRAPH_FAILED);

    auto updatesPtr = context_->GetInputDesc(UPDATES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, updatesPtr);
    auto updatesDtype = updatesPtr->GetDataType();
    OP_TILING_CHECK((updatesDtype != dtype_),
                    VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
                                                    "expected updates dtype to be equal to data dtype, please check."),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

bool ScatterElementsV2AscTiling::CompareShape(const gert::Shape& shape1, const gert::Shape& shape2, int16_t dim)
{
    int16_t inputShapeSize = static_cast<int16_t>(shape1.GetDimNum());
    for (int16_t i = 0; i < inputShapeSize; ++i) {
        if (i != dim) {
            if (shape1.GetDim(i) > shape2.GetDim(i)) {
                return false;
            }
        }
    }
    return true;
}

void ScatterElementsV2AscTiling::ComputeShape(const gert::Shape& dataShape, const gert::Shape& indicesShape,
                                         const gert::Shape& updatesShape)
{
    int16_t inputShapeSize = static_cast<int16_t>(dataShape.GetDimNum());
    int16_t j = inputShapeSize - 1;
    for (; j >= 0; j--) {
        if (dataShape.GetDim(j) != indicesShape.GetDim(j) || dataShape.GetDim(j) != updatesShape.GetDim(j)) {
            break;
        }
    }

    int16_t axisSame = j + 1;
    int16_t combAxis = std::max(axisSame, static_cast<int16_t>(dim_ + 1));
    for (int16_t i = 0; i < combAxis; ++i) {
        dataCurSize_.push_back(static_cast<uint64_t>(dataShape.GetDim(i)));
        indicesCurSize_.push_back(static_cast<uint64_t>(indicesShape.GetDim(i)));
        updatesCurSize_.push_back(static_cast<uint64_t>(updatesShape.GetDim(i)));
    }

    if (combAxis <= rank_ - 1) {
        uint64_t lastAxis = 1;
        for (int16_t i = combAxis; i < rank_; ++i) {
            lastAxis *= static_cast<uint64_t>(dataShape.GetDim(i));
        }

        dataCurSize_.push_back(lastAxis);
        indicesCurSize_.push_back(lastAxis);
        updatesCurSize_.push_back(lastAxis);
        rank_ = combAxis + 1;
    }
}

uint64_t ScatterElementsV2AscTiling::GetStride(const std::vector<uint64_t>& shapeList, int16_t start)
{
    if (start < 0 || start >= rank_) {
        return 0;
    }
    uint64_t stride = 1;
    for (int16_t i = start; i < rank_; ++i) {
        stride *= shapeList[i];
    }
    return stride;
}

void ScatterElementsV2AscTiling::ComputeStride()
{
    for (int16_t i = 0; i < rank_ - 1; ++i) {
        dataStride_[i] = GetStride(dataCurSize_, i + 1);
        indicesStride_[i] = GetStride(indicesCurSize_, i + 1);
        updatesStride_[i] = GetStride(updatesCurSize_, i + 1);
    }
}

void ScatterElementsV2AscTiling::CombineIndicesAxis()
{
    for (int16_t i = 0; i < dim_; ++i) {
        preAxis_ *= indicesCurSize_[i];
    }

    midAxis_ = indicesCurSize_[dim_];

    for (int16_t j = dim_ + 1; j < rank_; ++j) {
        afterAxis_ *= indicesCurSize_[j];
    }
    return;
}

ge::graphStatus ScatterElementsV2AscTiling::CheckInputShape()
{
    auto dataShapePtr = context_->GetInputShape(DATA_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dataShapePtr);
    auto dataShape = dataShapePtr->GetStorageShape();
    dataAxis_ = dataShape.GetShapeSize();

    auto indicesShapePtr = context_->GetInputShape(INDICES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, indicesShapePtr);
    auto indicesShape = indicesShapePtr->GetStorageShape();
    int16_t indicesDimNum = static_cast<int16_t>(indicesShape.GetDimNum());
    allAxis_ = indicesShape.GetShapeSize();

    auto updatesShapePtr = context_->GetInputShape(UPDATES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, updatesShapePtr);
    auto updatesShape = updatesShapePtr->GetStorageShape();
    int16_t updatesDimNum = static_cast<int16_t>(updatesShape.GetDimNum());
    updatesAxis_ = updatesShape.GetShapeSize();

    OP_TILING_CHECK(
        (indicesDimNum != rank_),
        VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
                                        "indices must have the same number of dimensions as data, please check."),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        (indicesDimNum != updatesDimNum),
        VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(),
                                        "indices must have the same number of dimensions as updates, please check."),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(!CompareShape(indicesShape, updatesShape),
                    VECTOR_INNER_ERR_REPORT_TILIING(
                        context_->GetNodeName(),
                        "expected indices shape to be smaller size than data shape in each dimension, please check."),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        !CompareShape(indicesShape, dataShape, dim_),
        VECTOR_INNER_ERR_REPORT_TILIING(
            context_->GetNodeName(),
            "expected indices shape to be smaller size than updates shape apart from dimension %d, please check.",
            dim_),
        return ge::GRAPH_FAILED);

    ComputeShape(dataShape, indicesShape, updatesShape);
    ComputeStride();
    return ge::GRAPH_SUCCESS;
}

void ScatterElementsV2AscTiling::GetCastTypeSize()
{
    if (reduction_ == REDUCTION_ADD && (dtype_ == ge::DT_INT16 || dtype_ == ge::DT_INT8 || dtype_ == ge::DT_UINT8)) {
        castTypeSize_ = ge::GetSizeByDataType(ge::DT_INT32);
    } else if (reduction_ == REDUCTION_ADD && (dtype_ == ge::DT_BOOL)) {
        castTypeSize_ = ge::GetSizeByDataType(ge::DT_FLOAT16);
    }
}

uint32_t ScatterElementsV2AscTiling::GetMaxSortTmpBuf([[maybe_unused]] int64_t sortDim)
{
    return MAX_SORT_SPACE;
}

/**
 * @brief Find best baseSize in range [baseXoStart, baseXoEnd], use dichotomy algorithm.
 */
int64_t ScatterElementsV2AscTiling::CalBestBaseSize(int64_t baseXoStart, int64_t baseXoEnd)
{
    int64_t baseXoMid;
    int64_t tmpTotalSize = 0;
    baseXoEnd = baseXoEnd + 1;
    while (baseXoEnd - baseXoStart > 1) {
        baseXoMid = (baseXoStart + baseXoEnd) / DOUBLE_COUNT;
        int64_t sortDim = baseS_ * baseXoMid;
        int64_t sortNeedTmpSize = static_cast<int64_t>(GetMaxSortTmpBuf(sortDim));
        tmpTotalSize = sortDim * indicesTypeSize_ * DETERM_DB_BUFFER + ubBlockSize_ +  // indocesQue
                       sortDim * indicesTypeSize_ + ubBlockSize_ +                     // sortedkeyBuf
                       sortDim * sizeof(uint32_t) + ubBlockSize_ +                     // sortedIdxBuf
                       sortNeedTmpSize + ubBlockSize_;                                 // sort shared buf size
        if (tmpTotalSize <= ubSize_) {
            baseXoStart = baseXoMid;
        } else {
            baseXoEnd = baseXoMid;
        }
    }
    return baseXoStart;
}

ge::graphStatus ScatterElementsV2AscTiling::DoOpTiling()
{
    int64_t usedCoreNumAlignTotal = Ops::Base::CeilDiv(allAxis_, MAX_THREAD_NUM);
    usedCoreNumAlignTotal = std::min(usedCoreNumAlignTotal, totalCoreNum_);

    int64_t usedCoreNumAlignData = Ops::Base::CeilDiv(dataAxis_, static_cast<int64_t>(totalCoreNum_));
    usedCoreNumAlignData = std::max(usedCoreNumAlignData, UB_MIN_FACTOR);
    usedCoreNumAlignData = Ops::Base::CeilDiv(dataAxis_, usedCoreNumAlignData);

    usedCoreNum_ = usedCoreNumAlignTotal > usedCoreNumAlignData ? usedCoreNumAlignTotal : usedCoreNumAlignData;

    ubSize_ = std::min(ubSize_, USE_UB_MAX_SIZE);
    GetCastTypeSize();
    int64_t ubLength = 0;
    if (reduction_ == REDUCTION_ADD &&
        (dtype_ == ge::DT_INT16 || dtype_ == ge::DT_INT8 || dtype_ == ge::DT_UINT8 || dtype_ == ge::DT_BOOL)) {
        ubLength = ubSize_ / DB_BUFFER / ACTIVE_NODES_NUM / castTypeSize_;
    } else {
        ubLength = ubSize_ / DB_BUFFER / typeSize_;
    }
    int64_t oneBlockNum = Ops::Base::GetUbBlockSize(context_) / typeSize_;
    loopLength_ = Ops::Base::FloorAlign(ubLength, oneBlockNum);
    if (isDeterministic_) {
        CombineIndicesAxis();
        indicesTypeSize_ = ge::GetSizeByDataType(indicesDtype_);
        OP_TILING_CHECK(indicesTypeSize_ <= 0,
                        VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "get indicesType size fail."),
                        return ge::GRAPH_FAILED);
        ubBlockSize_ = Ops::Base::GetUbBlockSize(context_);
        baseS_ = std::min(midAxis_, static_cast<int64_t>(BASE_S_MAX / indicesTypeSize_));
        int64_t aSplitDim = afterAxis_;
        if (preAxis_ > afterAxis_) {
            aSplitDim = preAxis_;
        }
        indicesNormBlockData_ = Ops::Base::CeilDiv(aSplitDim, usedCoreNum_);
        int64_t tmpSize = baseS_;
        bool isPatternASA = afterAxis_ != 1 && preAxis_ != 1;
        if (isPatternASA) {
            if (preAxis_ > afterAxis_) {
                tmpSize *= afterAxis_;
            } else {
                tmpSize *= preAxis_;
            }
        }
        indicesNormBlockData_ =
            std::max(indicesNormBlockData_, static_cast<int64_t>(UB_MIN_FACTOR / indicesTypeSize_ / tmpSize));
        indicesUsedCoreNum_ = Ops::Base::CeilDiv(aSplitDim, indicesNormBlockData_);
        indicesTailBlockData_ = aSplitDim - (indicesUsedCoreNum_ - 1) * indicesNormBlockData_;
        int64_t aDim = indicesUsedCoreNum_ == 1 ? indicesTailBlockData_ : indicesNormBlockData_;
        if (isPatternASA) {
            if (preAxis_ > afterAxis_) {
                aDim *= afterAxis_;
            } else {
                aDim *= preAxis_;
            }
        }
        baseA_ = CalBestBaseSize(1, aDim);
        int64_t sortDim = baseS_ * baseA_;
        sortSharedBufSize_ = GetMaxSortTmpBuf(sortDim);
    }

    tilingData_.set_dim(dim_);
    tilingData_.set_rank(rank_);
    tilingData_.set_loopLength(loopLength_);
    tilingData_.set_allAxis(allAxis_);
    tilingData_.set_dataAxis(dataAxis_);
    tilingData_.set_updatesAxis(updatesAxis_);
    tilingData_.set_dataStride(dataStride_);
    tilingData_.set_indicesStride(indicesStride_);
    tilingData_.set_updatesStride(updatesStride_);
    tilingData_.set_preAxis(preAxis_);
    tilingData_.set_midAxis(midAxis_);
    tilingData_.set_afterAxis(afterAxis_);
    tilingData_.set_indicesNormBlockData(indicesNormBlockData_);
    tilingData_.set_indicesUsedCoreNum(indicesUsedCoreNum_);
    tilingData_.set_indicesTailBlockData(indicesTailBlockData_);
    tilingData_.set_baseS(baseS_);
    tilingData_.set_baseA(baseA_);
    tilingData_.set_isDeterministic(isDeterministic_);
    tilingData_.set_sortSharedBufSize(sortSharedBufSize_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterElementsV2AscTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t ScatterElementsV2AscTiling::GetTilingKey() const
{
    uint64_t tilingKey = 1000000;
    uint64_t factorStart = 100;
    uint64_t factor = 10;

    // 后两位 dtype
    if (reduction_ == REDUCTION_NONE) {
        tilingKey += typeSize_;
    } else {
        tilingKey += static_cast<uint64_t>(dtype_);
    }
    // 百位 reduction
    tilingKey += factorStart * reduction_;
    // 千位 indices dtype
    uint64_t thousandDigit = indicesDtype_ == ge::DT_INT32 ? 0 : 1;
    tilingKey += factorStart * factor * thousandDigit;
    uint64_t wanDigit = allAxis_ > MAX_INT32_NUM ? 1 : 0;
    tilingKey += factorStart * factor * factor * wanDigit;
    return tilingKey;
}

ge::graphStatus ScatterElementsV2AscTiling::GetWorkspaceSize()
{
    workspaceSize_ = ASCENDC_TOOLS_WORKSPACE;
    int64_t dataWsSize = 0;
    int64_t updatesWsSize = 0;
    if (castTypeSize_ != 0) {
        dataWsSize = Ops::Base::CeilAlign(dataAxis_ * castTypeSize_, GM_ALIGN);
        updatesWsSize = Ops::Base::CeilAlign(updatesAxis_ * castTypeSize_, GM_ALIGN);
        workspaceSize_ += dataWsSize + updatesWsSize;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterElementsV2AscTiling::PostTiling()
{
    auto workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = workspaceSize_;
    tilingKey_ = GetTilingKey();
    context_->SetTilingKey(tilingKey_);
    context_->SetBlockDim(usedCoreNum_);
    context_->SetScheduleMode(1);
    auto res = context_->SetLocalMemorySize(ubSize_ + SIMT_UB_RES_SIZE);
    OP_TILING_CHECK(
        (res != ge::GRAPH_SUCCESS),
        VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "SetLocalMemorySize ubSize = %ld failed.", ubSize_),
        return ge::GRAPH_FAILED);
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

void ScatterElementsV2AscTiling::DumpTilingInfo()
{
    std::ostringstream info;
    info << "usedCoreNum: " << usedCoreNum_;
    info << ", dim: " << tilingData_.get_dim();
    info << ", rank: " << tilingData_.get_rank();
    info << ", dataStride:" << ToString(tilingData_.get_dataStride(), TILING_ARRAY_LEN).c_str();
    info << ", indicesStride: " << ToString(tilingData_.get_indicesStride(), TILING_ARRAY_LEN).c_str();
    info << ", updatesStride: " << ToString(tilingData_.get_updatesStride(), TILING_ARRAY_LEN).c_str();
    info << ", loopLength: " << tilingData_.get_loopLength();
    info << ", allAxis: " << tilingData_.get_allAxis();
    info << ", dataAxis: " << tilingData_.get_dataAxis();
    info << ", updatesAxis: " << tilingData_.get_updatesAxis();
    info << ", preAxis: " << tilingData_.get_preAxis();
    info << ", midAxis: " << tilingData_.get_midAxis();
    info << ", afterAxis: " << tilingData_.get_afterAxis();
    info << ", indicesUsedCoreNum: " << tilingData_.get_indicesUsedCoreNum();
    info << ", indicesNormBlockData: " << tilingData_.get_indicesNormBlockData();
    info << ", indicesTailBlockData: " << tilingData_.get_indicesTailBlockData();
    info << ", baseS: " << tilingData_.get_baseS();
    info << ", baseA: " << tilingData_.get_baseA();
    info << ", sortSharedBufSize: " << tilingData_.get_sortSharedBufSize();
    info << ", isDeterministic: " << tilingData_.get_isDeterministic();
    info << ", tilingKey_: " << tilingKey_;
    OP_LOGI(context_->GetNodeName(), "%s", info.str().c_str());
}

REGISTER_TILING_TEMPLATE("ScatterElementsV2", ScatterElementsV2AscTiling, 0);
}  // namespace optiling

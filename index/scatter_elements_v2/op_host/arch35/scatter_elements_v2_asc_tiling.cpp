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
#include "scatter_elements_v2_base_tiling.h"

using namespace AscendC;
namespace optiling {
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
constexpr int64_t USE_UB_MAX_SIZE = 65536; // 64K
constexpr int64_t MAX_THREAD_NUM = 512;
constexpr int64_t MAX_INT32_NUM = 2147483647;
constexpr int64_t MAX_INT16_NUM = 32767;              // int16 key 上限(keySize 分档阈值)
constexpr int64_t DCACHE_SIZE = 131072;               // 128K
constexpr int64_t ASCENDC_TOOLS_WORKSPACE = 16777216; // 16M
constexpr int64_t DETERM_DB_BUFFER = 2;
constexpr int64_t DOUBLE_COUNT = 2;
constexpr int64_t BASE_S_MAX = 256;
constexpr int64_t UB_MIN_FACTOR = 1024;
constexpr int64_t SIMT_UB_RES_SIZE = 640;
constexpr uint32_t MAX_SORT_SPACE = 10240;
constexpr int64_t PHASE_THREAD_NUM = 1024;  // Phase1/Phase3 每核 stride-loop 粒度（按索引总数切核）
constexpr int64_t STATIC_UB_ESTIMATE = 512; // WithSorted SIMT strides/参数缓冲（SortLib tiling 预算扣除）
// 排序模板准入门槛
constexpr int64_t SORT_ADMIT_MID_OUTER_RATIO = 50; // 索引轴主导门槛：midAxis/RATIO 不小于 outerAxisNum 才准入
constexpr int64_t SORT_ADMIT_HALF_CORE_RATIO = 2; // float 半核判定：A 轴分核数*RATIO 需小于总核数
// tilingKey 前缀（与 apt.cpp 的 TILING_KEY_IS 宏对表）
constexpr uint64_t SCAC_ELE_DETERM_KEY_BASE = 1000000; // 确定性模板前缀基值（1xxxxxx）
constexpr uint64_t SCAC_ELE_SORT_KEY_PREFIX = 1000000; // 排序模板前缀提升步长（1xxxxxx → 2xxxxxx）

static const std::set<ge::DataType> SCAT_ELE_NONE_DTYPE = {ge::DT_FLOAT,  ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT64,
                                                           ge::DT_INT32,  ge::DT_INT16,   ge::DT_INT8, ge::DT_UINT8,
                                                           ge::DT_DOUBLE, ge::DT_BOOL};
static const std::set<ge::DataType> SCAT_ELE_ADD_DTYPE = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16,
                                                          ge::DT_INT64, ge::DT_INT32,   ge::DT_INT16,
                                                          ge::DT_INT8,  ge::DT_UINT8,   ge::DT_BOOL};
static const std::set<ge::DataType> SCAT_ELE_ADD_DETERM_DTYPE = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};
// 排序模板（WithSorted / tilingKey 前缀 2xxxxxx）add 的 dtype 白名单：在 FP16/FP32/BF16 基础上
// 放宽到 int8/int16/int32
static const std::set<ge::DataType> SCAT_ELE_SORT_DETERM_DTYPE = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16,
                                                                  ge::DT_INT8,  ge::DT_INT16,   ge::DT_INT32};
static const std::set<ge::DataType> SCAT_ELE_MUL_DTYPE = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT64,
                                                          ge::DT_INT32, ge::DT_INT16,   ge::DT_INT8, ge::DT_UINT8};

static const std::map<std::string, uint64_t> SCAT_ELE_REDUCTION = {{"none", 0}, {"add", 1}, {"mul", 2}};

template <typename T>
static std::string ToString(const T* value, size_t size)
{
    std::string r = "[";
    for (size_t i = 0; i < size; i++) {
        r = r + std::to_string(value[i]) + ", ";
    }
    r = r + "]";
    return r;
}

// WithSorted workspace 分段 128B 对齐
static int64_t WithSortedAlignUp128(int64_t value)
{
    constexpr int64_t align = 128;
    return (value + align - 1) / align * align;
}

bool ScatterElementsV2AscTiling::IsCapable() { return true; }

ge::graphStatus ScatterElementsV2AscTiling::GetPlatformInfo()
{
    auto compileInfo = reinterpret_cast<const ScatterElementsV2CompileInfoArch35*>(context_->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo);
    totalCoreNum_ = compileInfo->totalCoreNum;
    ubSize_ = compileInfo->ubSizePlatForm;
    if (ubSize_ <= DCACHE_SIZE) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            context_->GetNodeName(), "ubSize_, DCACHE_SIZE",
            (std::to_string(static_cast<int32_t>(ubSize_)) + ", " + std::to_string(static_cast<int32_t>(DCACHE_SIZE)))
                .c_str(),
            "ubSize must be less than Dcache Size");
        return ge::GRAPH_FAILED;
    }
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
    if (dim > dimMax || dim < dimMin) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "axis", std::to_string(dim).c_str(),
                                              "axis must be in range[-rank, rank-1]");
        return ge::GRAPH_FAILED;
    }

    dim_ = dim < 0 ? static_cast<int16_t>(dim) + rank_ : static_cast<int16_t>(dim);

    const char* reduction = attrs->GetAttrPointer<char>(ATTR_REDUCTION_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, reduction);
    std::string reductionStr = reduction;
    auto it = SCAT_ELE_REDUCTION.find(reductionStr);
    bool reductionInValid = it == SCAT_ELE_REDUCTION.end();
    if (reductionInValid) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "reduction", reductionStr.c_str(),
                                              "reduction must be in [none, add, mul]");
        return ge::GRAPH_FAILED;
    }
    reduction_ = it->second;

    if (CheckInputDtype() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (CheckInputShape() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    bool isDetermDtype = (reduction_ == REDUCTION_NONE) ||
                         (reduction_ == REDUCTION_ADD &&
                          SCAT_ELE_ADD_DETERM_DTYPE.find(dtype_) != SCAT_ELE_ADD_DETERM_DTYPE.end());
    if (context_->GetDeterministic() && isDetermDtype) {
        isDeterministic_ = 1;
    }

    // 排序模板 dtype 准入（仅决定 isSortDeterministic_，即是否具备进入排序模板的 dtype 资格）：
    //   add —— FP16/FP32/BF16 + int8/int16/int32（SCAT_ELE_SORT_DETERM_DTYPE）；
    //   none —— 仅 int 类型。
    bool isSortDetermDtype = (reduction_ == REDUCTION_ADD &&
                              SCAT_ELE_SORT_DETERM_DTYPE.find(dtype_) != SCAT_ELE_SORT_DETERM_DTYPE.end()) ||
                             (reduction_ == REDUCTION_NONE &&
                              (dtype_ == ge::DT_INT8 || dtype_ == ge::DT_INT16 || dtype_ == ge::DT_INT32 ||
                               dtype_ == ge::DT_UINT8 || dtype_ == ge::DT_INT64));
    if (context_->GetDeterministic() && isSortDetermDtype) {
        isSortDeterministic_ = 1;
    }

    // === 排序模板预计算（linear_index/srcPos 统一按 max(data, updates) 元素数定 key 宽）===
    indicesTotalNum_ = allAxis_;
    int64_t maxElem = dataAxis_ > updatesAxis_ ? dataAxis_ : updatesAxis_;
    keySize_ = (maxElem <= MAX_INT16_NUM) ? 2 : (maxElem <= MAX_INT32_NUM) ? 4 : 8;
    keyDtype_ = (keySize_ == 2) ? ge::DT_INT16 : (keySize_ == 4) ? ge::DT_INT32 : ge::DT_INT64;
    countMode_ = SortLib::IsInt32Safe(indicesTotalNum_) ? 0 : 1;
    // perm（排序索引）位宽跟随计数模式：countMode_=0 即元素数 <= 2^30，int32 可安全表示 [0, N)；
    // 否则用 int64 perm，避免索引截断。
    permSize_ = (countMode_ == 0) ? 4 : 8;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterElementsV2AscTiling::CheckXDtype(const ge::DataType dtype)
{
    if (reduction_ == REDUCTION_NONE) {
        if (SCAT_ELE_NONE_DTYPE.find(dtype) == SCAT_ELE_NONE_DTYPE.end()) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                context_->GetNodeName(), "data", std::to_string(static_cast<int32_t>(dtype)).c_str(),
                "When reduction=none, dtype must be in [DT_FLOAT, DT_FLOAT16, DT_BF16, DT_INT64, DT_INT32, DT_INT16, "
                "DT_INT8, DT_UINT8, DT_DOUBLE, DT_BOOL]");
            return ge::GRAPH_FAILED;
        }
    } else if (reduction_ == REDUCTION_ADD) {
        if (SCAT_ELE_ADD_DTYPE.find(dtype) == SCAT_ELE_ADD_DTYPE.end()) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "data",
                                                  std::to_string(static_cast<int32_t>(dtype)).c_str(),
                                                  "When reduction=add, dtype must be in [DT_FLOAT, DT_FLOAT16, "
                                                  "DT_BF16, DT_INT64, DT_INT32, DT_INT16, DT_INT8, DT_UINT8, DT_BOOL]");
            return ge::GRAPH_FAILED;
        }
    } else if (reduction_ == REDUCTION_MUL) {
        if (SCAT_ELE_MUL_DTYPE.find(dtype) == SCAT_ELE_MUL_DTYPE.end()) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "data",
                                                  std::to_string(static_cast<int32_t>(dtype)).c_str(),
                                                  "When reduction=mul, dtype must be in [DT_FLOAT, DT_FLOAT16, "
                                                  "DT_BF16, DT_INT64, DT_INT32, DT_INT16, DT_INT8, DT_UINT8]");
            return ge::GRAPH_FAILED;
        }
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
    if (typeSize_ <= 0) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "data",
                                              std::to_string(static_cast<int32_t>(dtype_)).c_str(),
                                              "dtype size is invalid");
        return ge::GRAPH_FAILED;
    }

    auto indicesPtr = context_->GetInputDesc(INDICES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, indicesPtr);
    indicesDtype_ = indicesPtr->GetDataType();
    bool dtypeInValid = indicesDtype_ != ge::DT_INT32 && indicesDtype_ != ge::DT_INT64;
    if (dtypeInValid) {
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "indices",
                                  std::to_string(static_cast<int32_t>(indicesDtype_)).c_str(),
                                  "indices dType must be in [DT_INT32, DT_INT64]");
        return ge::GRAPH_FAILED;
    }

    auto updatesPtr = context_->GetInputDesc(UPDATES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, updatesPtr);
    auto updatesDtype = updatesPtr->GetDataType();
    if (updatesDtype != dtype_) {
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            context_->GetNodeName(), "data, updates",
            (std::to_string(static_cast<int32_t>(dtype_)) + ", " + std::to_string(static_cast<int32_t>(updatesDtype)))
                .c_str(),
            "the dtype of data and updates must be same");
        return ge::GRAPH_FAILED;
    }
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
    const char* opName_ = "ScatterElementsV2";
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

    if (indicesDimNum != rank_) {
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            opName_, "indices, data", (std::to_string(indicesDimNum) + ", " + std::to_string(rank_)).c_str(),
            "the dimNum of indices and data must be same");
        return ge::GRAPH_FAILED;
    }

    if (indicesDimNum != updatesDimNum) {
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            opName_, "indices, updates", (std::to_string(indicesDimNum) + ", " + std::to_string(updatesDimNum)).c_str(),
            "the dimNum of indices and updates must be same");
        return ge::GRAPH_FAILED;
    }

    if (!CompareShape(indicesShape, updatesShape)) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(opName_, "indices, data", "indices_shape, data_shape",
                                               "each indices shape dim must be less than data shape");
        return ge::GRAPH_FAILED;
    }

    if (!CompareShape(indicesShape, dataShape, dim_)) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            opName_, "indices, data", "indices_shape, data_shape",
            "the shape of indices must be less than or equal to the shape of data in each dimension except dim");
        return ge::GRAPH_FAILED;
    }

    ComputeShape(dataShape, indicesShape, updatesShape);
    ComputeStride();

    // shapeMode：逐维比较原始 storage shape，任一维 data != indices → SUBSET(1)
    shapeMode_ = 0;
    for (int16_t i = 0; i < rank_; ++i) {
        if (dataShape.GetDim(i) != indicesShape.GetDim(i)) {
            shapeMode_ = 1;
            break;
        }
    }
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

uint32_t ScatterElementsV2AscTiling::GetMaxSortTmpBuf(int64_t sortDim)
{
    std::vector<int64_t> shapeVec = {sortDim};
    ge::Shape srcShape(shapeVec);
    AscendC::SortConfig config;
    config.type = AscendC::SortType::RADIX_SORT;
    config.isDescend = false;
    config.hasSrcIndex = false;
    config.hasDstIndex = true;
    uint32_t maxValue = 0;
    uint32_t minValue = 0;
    GetSortMaxMinTmpSize(srcShape, indicesDtype_, ge::DT_UINT32, false, config, maxValue, minValue);
    return maxValue;
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
        tmpTotalSize = sortDim * indicesTypeSize_ * DETERM_DB_BUFFER + ubBlockSize_ + // indocesQue
                       sortDim * indicesTypeSize_ + ubBlockSize_ +                    // sortedkeyBuf
                       sortDim * sizeof(uint32_t) + ubBlockSize_ +                    // sortedIdxBuf
                       sortNeedTmpSize + ubBlockSize_;                                // sort shared buf size
        if (tmpTotalSize <= ubSize_) {
            baseXoStart = baseXoMid;
        } else {
            baseXoEnd = baseXoMid;
        }
    }
    return baseXoStart;
}

// 排序模板准入 —— 整型分支。
// int8/uint8：放宽至多维（rank_ <= 8），int16/int32/int64：仍仅一维（rank_ == 1）准入。
bool ScatterElementsV2AscTiling::IsSortAdmittedInt() const
{
    if (dtype_ == ge::DT_INT8 || dtype_ == ge::DT_UINT8) {
        return rank_ <= 8;
    }
    return rank_ == 1;
}

// 排序模板准入 —— 浮点分支（fp32/fp16/bf16）。
// 只有当原确定性模板明显吃不满核（A 轴分核数 < 总核数一半）时，排序模板才接管。
bool ScatterElementsV2AscTiling::IsSortAdmittedFloat(int64_t aAxisCoreNum) const
{
    return aAxisCoreNum * SORT_ADMIT_HALF_CORE_RATIO < totalCoreNum_; // 乘比例比较避免整除截断
}

bool ScatterElementsV2AscTiling::IsSortTemplateAdmitted(int64_t aAxisCoreNum) const
{
    // 排序模板 dtype 白名单：add 走 SCAT_ELE_SORT_DETERM_DTYPE（FP + int8/16/32）；none 仅 int 类型。
    bool sortDtypeOk = SCAT_ELE_SORT_DETERM_DTYPE.find(dtype_) != SCAT_ELE_SORT_DETERM_DTYPE.end() ||
                       (reduction_ == REDUCTION_NONE &&
                        (dtype_ == ge::DT_INT8 || dtype_ == ge::DT_INT16 || dtype_ == ge::DT_INT32 ||
                         dtype_ == ge::DT_UINT8 || dtype_ == ge::DT_INT64));
    if (!sortDtypeOk) {
        return false;
    }
    int64_t outerAxisNum = preAxis_ * afterAxis_;
    if (midAxis_ / SORT_ADMIT_MID_OUTER_RATIO < outerAxisNum) {
        return false;
    }
    bool isIntDtype = dtype_ == ge::DT_INT8 || dtype_ == ge::DT_UINT8 || dtype_ == ge::DT_INT16 ||
                      dtype_ == ge::DT_INT32 || dtype_ == ge::DT_INT64;
    bool isFloatDtype = dtype_ == ge::DT_FLOAT || dtype_ == ge::DT_FLOAT16 || dtype_ == ge::DT_BF16;
    if (isIntDtype) {
        if (!IsSortAdmittedInt()) {
            return false;
        }
    } else if (isFloatDtype) {
        if (!IsSortAdmittedFloat(aAxisCoreNum)) {
            return false;
        }
    } else {
        return false;
    }
    // index-count 切核收益判定：按索引总数切核的核数须多于原确定性模板 A 轴切核的核数
    int64_t normBlockData = std::max(Ops::Base::CeilDiv(indicesTotalNum_, totalCoreNum_),
                                     static_cast<int64_t>(UB_MIN_FACTOR));
    int64_t idxNumCoreNum = Ops::Base::CeilDiv(indicesTotalNum_, normBlockData);
    return idxNumCoreNum > aAxisCoreNum;
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
    // 公共基础：原确定性模板与排序模板都要 pre/mid/afterAxis 与 indices 基础字段。
    if (isDeterministic_ || isSortDeterministic_) {
        CombineIndicesAxis();
        indicesTypeSize_ = ge::GetSizeByDataType(indicesDtype_);
        if (indicesTypeSize_ <= 0) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "indices",
                                                  std::to_string(static_cast<int32_t>(indicesDtype_)).c_str(),
                                                  "the size of indices dtype is invalid");
            return ge::GRAPH_FAILED;
        }
    }

    // === 原确定性模板：A 轴切核 ===
    if (isDeterministic_ || isSortDeterministic_) {
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
        indicesNormBlockData_ = std::max(indicesNormBlockData_,
                                         static_cast<int64_t>(UB_MIN_FACTOR / indicesTypeSize_ / tmpSize));
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

    // === 排序模板：独立、优先准入
    if (isSortDeterministic_ && IsSortTemplateAdmitted(indicesUsedCoreNum_)) {
        isSortDeterm_ = true;
        // SortLib 单核/多核自动切换：总元素数小且能塞下则 isSingleCore，否则多核 radix。
        sortR_ = SortLib::SortTilingCompute(
            indicesTotalNum_, totalCoreNum_, static_cast<uint64_t>(ubSize_ - STATIC_UB_ESTIMATE),
            static_cast<uint32_t>(keySize_), static_cast<uint32_t>(permSize_), countMode_ == 0, keyDtype_);
        if (sortR_.errCode != SortLib::SORT_TILING_OK) {
            // 库报错（UB 不足最小内核）时回落原确定性模板（1xxxxxx 前缀），
            // 避免无效核数/workspace 泄漏。
            isSortDeterm_ = false;
            sortR_ = SortLib::SortTilingResult{};
        }
    }
    if (isSortDeterm_) {
        multiSortWsBytes_ = sortR_.workspaceBytes;
        sortUsedCoreNum_ = (sortR_.coreNumNeed > 0) ? static_cast<int64_t>(sortR_.coreNumNeed) : 1;

        // workspace 布局（各段 128B 对齐，与 kernel Init 严格对应）
        int64_t n = indicesTotalNum_;
        wsLinearIdxOff_ = WithSortedAlignUp128(multiSortWsBytes_);
        wsSortedOff_ = WithSortedAlignUp128(wsLinearIdxOff_ + n * keySize_);
        wsPermOff_ = WithSortedAlignUp128(wsSortedOff_ + n * keySize_);
        wsUserSize_ = wsPermOff_ + n * permSize_; // linearIdx + sorted + perm 基准
        if (shapeMode_ == 1) {
            wsSrcPosOff_ = WithSortedAlignUp128(wsUserSize_);
            wsUserSize_ = wsSrcPosOff_ + n * keySize_;
        } else {
            wsSrcPosOff_ = 0;
        }
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

ge::graphStatus ScatterElementsV2AscTiling::DoLibApiTiling() { return ge::GRAPH_SUCCESS; }

uint64_t ScatterElementsV2AscTiling::GetTilingKey() const
{
    uint64_t tilingKey = SCAC_ELE_DETERM_KEY_BASE;
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
    uint64_t wanDigit = 0;
    if (allAxis_ > MAX_INT32_NUM || dataAxis_ > MAX_INT32_NUM || updatesAxis_ > MAX_INT32_NUM) {
        wanDigit = 1;
    }

    tilingKey += factorStart * factor * factor * wanDigit;
    // 排序模板前缀提升：1xxxxxx → 2xxxxxx，触发 KernelScatterElementsWithSorted 模板路径。
    if (isSortDeterm_) {
        tilingKey += SCAC_ELE_SORT_KEY_PREFIX;
    }
    return tilingKey;
}

ge::graphStatus ScatterElementsV2AscTiling::GetWorkspaceSize()
{
    workspaceSize_ = ASCENDC_TOOLS_WORKSPACE;
    if (isSortDeterm_) {
        workspaceSize_ += wsUserSize_; // 排序模板 userWs（SortLib ws + linearIdx/sorted/perm[+srcPos]）
    } else if (castTypeSize_ != 0) {
        int64_t dataWsSize = Ops::Base::CeilAlign(dataAxis_ * castTypeSize_, GM_ALIGN);
        int64_t updatesWsSize = Ops::Base::CeilAlign(updatesAxis_ * castTypeSize_, GM_ALIGN);
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
    if (isSortDeterm_) {
        // 各阶段核数：Phase1/Phase3 按索引总数（每核至少 PHASE_THREAD_NUM），Sort 用 sortR.coreNumNeed
        int64_t phaseCores = Ops::Base::CeilDiv(indicesTotalNum_, PHASE_THREAD_NUM);
        phaseCores = std::min(phaseCores, totalCoreNum_);
        if (phaseCores < 1) {
            phaseCores = 1;
        }
        int64_t blockDim = std::max(phaseCores, sortUsedCoreNum_);
        if (blockDim < 1) {
            blockDim = 1;
        }
        context_->SetBlockDim(blockDim);
    } else {
        context_->SetBlockDim(usedCoreNum_);
    }
    context_->SetScheduleMode(1);
    auto res = context_->SetLocalMemorySize(ubSize_ + SIMT_UB_RES_SIZE);
    if (res != ge::GRAPH_SUCCESS) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "ubSize_", std::to_string(ubSize_).c_str(),
                                              "SetLocalMemorySize failed");
    }
    // 仅排序模板启用时写入：sortTiling 只被 _WS 前缀内核读取，非排序案例保持默认全 0，
    // 避免无条件写入无效数据（该嵌套 struct 经 sortTiling{nullptr} 默认初始化，无脏值）。
    if (isSortDeterm_) {
        tilingData_.sortTiling.set_indicesTotalNum(indicesTotalNum_);
        tilingData_.sortTiling.set_keySize(keySize_);
        tilingData_.sortTiling.set_permSize(permSize_);
        tilingData_.sortTiling.set_countMode(countMode_);
        tilingData_.sortTiling.set_shapeMode(shapeMode_);
        tilingData_.sortTiling.set_dimNormalized(dim_);
        tilingData_.sortTiling.set_sortUsedCoreNum(static_cast<uint32_t>(sortUsedCoreNum_));
        tilingData_.sortTiling.set_numTileData(sortR_.numTileData);
        tilingData_.sortTiling.set_tileCount(sortR_.tileCount);
        tilingData_.sortTiling.set_activeCores(sortR_.activeCores);
        tilingData_.sortTiling.set_tmpUbSize(sortR_.tmpUbSize);
        tilingData_.sortTiling.set_isSingleCore(sortR_.isSingleCore);
        tilingData_.sortTiling.set_wsLinearIdxOff(wsLinearIdxOff_);
        tilingData_.sortTiling.set_wsSortedOff(wsSortedOff_);
        tilingData_.sortTiling.set_wsPermOff(wsPermOff_);
        tilingData_.sortTiling.set_wsSrcPosOff(wsSrcPosOff_);
    }

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
    info << ", isSortDeterministic: " << isSortDeterministic_;
    info << ", isSortDeterm: " << isSortDeterm_;
    info << ", sortUsedCoreNum: " << sortUsedCoreNum_;
    info << ", shapeMode: " << shapeMode_;
    info << ", keySize: " << keySize_;
    info << ", permSize: " << permSize_;
    info << ", indicesTotalNum: " << indicesTotalNum_;
    info << ", wsUserSize: " << wsUserSize_;
    info << ", tilingKey_: " << tilingKey_;
    OP_LOGI(context_->GetNodeName(), "%s", info.str().c_str());
}

REGISTER_TILING_TEMPLATE("ScatterElementsV2", ScatterElementsV2AscTiling, 0);
} // namespace optiling

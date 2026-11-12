/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file scatter_add_with_sorted_tiling_base.cpp
 * \brief scatter_add_with_sorted_tiling_base
 */

#include "scatter_add_with_sorted_tiling_base.h"
#include "log/log.h"
#include <string>

using namespace AscendC;
using namespace ge;

namespace optiling {

constexpr int64_t VAR_IDX = 0;
constexpr int64_t SORTED_INDEX_IDX = 2;
constexpr int64_t UPDATES_IDX = 1;
constexpr int64_t POS_IDX = 3;

static constexpr int64_t BASE_BLOCK_ALIGN = 512;
static constexpr int64_t SINGLE_CORE_THRESHOLD = 4 * 1024;
static constexpr int64_t BLOCK_TILING_THRES = 512;
static constexpr int64_t INNER_ADD_NUM = 128;

static const std::set<ge::DataType> INDICES_DTYPE_SET = {ge::DT_INT32, ge::DT_INT64};
static const std::set<ge::DataType> VAR_DTYPE_SET = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_BF16};

static std::string ToString(std::set<ge::DataType> supportDtypes)
{
    std::stringstream ss;
    for (const auto& element : supportDtypes) {
        ss << Ops::Base::ToString(element) << " ";
    }

    return ss.str();
}

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

ge::graphStatus ScatterAddWithSortedBaseTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_IF(
        platformInfo == nullptr,
        OP_LOGE_FOR_INVALID_CONFIG_WITH_REASON(opName, "tiling", "platform_info", "null", "fail to get platform info"),
        return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    // 960系列NPU架构号为9201/9202，兼容看护环境的旧op_common无开放常量，直接比较架构号数值
    static constexpr uint64_t ASCEND960_NPU_ARCH = 9201;
    static constexpr uint64_t ASCEND960PR_NPU_ARCH = 9202;
    isAscend96_ = static_cast<uint64_t>(ascendcPlatform.GetCurNpuArch()) == ASCEND960_NPU_ARCH ||
                  static_cast<uint64_t>(ascendcPlatform.GetCurNpuArch()) == ASCEND960PR_NPU_ARCH;
    auto aivNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((aivNum <= 0), OP_LOGE_FOR_INVALID_VALUE(opName, "aivNum", std::to_string(aivNum).c_str(), "> 0"),
                return ge::GRAPH_FAILED);
    totalCoreNum_ = aivNum;
    uint64_t ubSizePlatForm = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    ubSize_ = ubSizePlatForm;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterAddWithSortedBaseTiling::GetShapeAttrsInfo()
{
    auto var = context_->GetInputShape(VAR_IDX);
    OP_CHECK_IF(var == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName, "var"), return ge::GRAPH_FAILED);
    auto varShape = var->GetStorageShape();
    varSize_ = varShape.GetShapeSize();
    varShape_[0] = varShape.GetDim(0);
    varShape_[1] = (varShape_[0] != 0) ? varSize_ / varShape_[0] : 0;
    auto indices = context_->GetInputShape(SORTED_INDEX_IDX);
    OP_CHECK_IF(indices == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName, "sorted_index"), return ge::GRAPH_FAILED);
    auto indiceShape = indices->GetStorageShape();
    indicesNum_ = indiceShape.GetShapeSize();

    auto updates = context_->GetInputShape(UPDATES_IDX);
    OP_CHECK_IF(updates == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName, "value"), return ge::GRAPH_FAILED);
    auto updateShape = updates->GetStorageShape();
    updatesSize_ = updateShape.GetShapeSize();
    uint64_t updatesDims = updateShape.GetDimNum();
    if (updatesDims == 0 || (updatesDims == 1 && updatesSize_ == 1)) {
        isUpdateScalar_ = 1;
    } else {
        const auto ret = CheckUpdatesShape(varShape, indiceShape, updateShape);
        if (ret != ge::GRAPH_SUCCESS) {
            OP_LOGE_WITHOUT_REPORT(opName, "CheckUpdatesShape failed, rc=%u; propagate the parameter error.", ret);
            return ret;
        }
    }

    auto pos = context_->GetInputShape(POS_IDX);
    if (pos != nullptr) {
        auto posShape = pos->GetStorageShape();
        OP_CHECK_IF(static_cast<uint64_t>(posShape.GetShapeSize()) != indicesNum_,
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        opName, "pos, sorted_index",
                        (std::to_string(posShape.GetShapeSize()) + ", " + std::to_string(indicesNum_)).c_str(),
                        "pos and sorted_index must have the same element count"),
                    return ge::GRAPH_FAILED);
        hasPos_ = true;
    }

    const auto ret = CheckInputDtype();
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE_WITHOUT_REPORT(opName, "CheckInputDtype failed, rc=%u; propagate the parameter error.", ret);
        return ret;
    }

    bool isFloat = (varDtype_ == ge::DT_FLOAT || varDtype_ == ge::DT_FLOAT16 || varDtype_ == ge::DT_BF16);
    if (context_->GetDeterministic() && !isUpdateScalar_ && isFloat) {
        isDeterminTemplate_ = 1;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterAddWithSortedBaseTiling::CheckInputDtype()
{
    auto indicesPtr = context_->GetInputDesc(SORTED_INDEX_IDX);
    OP_CHECK_IF(indicesPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName, "sorted_index"), return ge::GRAPH_FAILED);
    indicesDtype_ = indicesPtr->GetDataType();
    OP_CHECK_IF(
        (INDICES_DTYPE_SET.find(indicesDtype_) == INDICES_DTYPE_SET.end()),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName, "sorted_index", Ops::Base::ToString(indicesDtype_).c_str(),
                                              ("sorted_index dtype must be in " + ToString(INDICES_DTYPE_SET)).c_str()),
        return ge::GRAPH_FAILED);
    indicesDtypeSize_ = ge::GetSizeByDataType(indicesDtype_);
    OP_CHECK_IF(
        indicesDtypeSize_ <= 0,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName, "sorted_index", Ops::Base::ToString(indicesDtype_).c_str(),
                                              "failed to get dtype size, dtype may be unsupported"),
        return ge::GRAPH_FAILED);

    if (hasPos_) {
        auto posPtr = context_->GetInputDesc(POS_IDX);
        OP_CHECK_IF(posPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName, "pos"), return ge::GRAPH_FAILED);
        posDtype_ = posPtr->GetDataType();
        OP_CHECK_IF(
            (INDICES_DTYPE_SET.find(posDtype_) == INDICES_DTYPE_SET.end()),
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName, "pos", Ops::Base::ToString(posDtype_).c_str(),
                                                  ("pos dtype must be in " + ToString(INDICES_DTYPE_SET)).c_str()),
            return ge::GRAPH_FAILED);
        posDtypeSize_ = ge::GetSizeByDataType(posDtype_);
        OP_CHECK_IF(posDtypeSize_ <= 0,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName, "pos", Ops::Base::ToString(posDtype_).c_str(),
                                                          "failed to get dtype size, dtype may be unsupported"),
                    return ge::GRAPH_FAILED);
    }

    auto dataPtr = context_->GetInputDesc(VAR_IDX);
    OP_CHECK_IF(dataPtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName, "var"), return ge::GRAPH_FAILED);
    varDtype_ = dataPtr->GetDataType();
    OP_CHECK_IF((VAR_DTYPE_SET.find(varDtype_) == VAR_DTYPE_SET.end()),
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName, "var", Ops::Base::ToString(varDtype_).c_str(),
                                                      ("var dtype must be in " + ToString(VAR_DTYPE_SET)).c_str()),
                return ge::GRAPH_FAILED);
    varTypeSize_ = ge::GetSizeByDataType(varDtype_);
    OP_CHECK_IF(varTypeSize_ <= 0,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName, "var", Ops::Base::ToString(varDtype_).c_str(),
                                                      "failed to get dtype size, dtype may be unsupported"),
                return ge::GRAPH_FAILED);
    auto updatePtr = context_->GetInputDesc(UPDATES_IDX);
    OP_CHECK_IF(updatePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName, "value"), return ge::GRAPH_FAILED);
    auto updatesType = updatePtr->GetDataType();
    OP_CHECK_IF((VAR_DTYPE_SET.find(updatesType) == VAR_DTYPE_SET.end()),
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName, "value", Ops::Base::ToString(updatesType).c_str(),
                                                      ("value dtype must be in " + ToString(VAR_DTYPE_SET)).c_str()),
                return ge::GRAPH_FAILED);
    updatesDtypeSize_ = ge::GetSizeByDataType(updatesType);
    OP_CHECK_IF(
        (updatesType != varDtype_),
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            opName, "var, value", (Ops::Base::ToString(varDtype_) + ", " + Ops::Base::ToString(updatesType)).c_str(),
            "var and value must have the same dtype"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterAddWithSortedBaseTiling::CheckUpdatesShape(const gert::Shape& varShape,
                                                                  const gert::Shape& indicesShape,
                                                                  const gert::Shape& updatesShape)
{
    uint64_t varDimNum = static_cast<uint64_t>(varShape.GetDimNum());
    uint64_t indicesDimNum = static_cast<uint64_t>(indicesShape.GetDimNum());
    uint64_t updatesDimNum = static_cast<uint64_t>(updatesShape.GetDimNum());
    OP_CHECK_IF(
        (updatesDimNum != indicesDimNum + varDimNum - 1),
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            opName, "value, sorted_index, var",
            (std::to_string(updatesDimNum) + ", " + std::to_string(indicesDimNum) + ", " + std::to_string(varDimNum))
                .c_str(),
            "updatesDimNum must equal indicesDimNum + varDimNum - 1"),
        return ge::GRAPH_FAILED);
    for (uint64_t i = 0; i < indicesDimNum; i++) {
        OP_CHECK_IF(
            (static_cast<uint32_t>(updatesShape.GetDim(i)) != static_cast<uint32_t>(indicesShape.GetDim(i))),
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                opName, "value, sorted_index",
                (std::to_string(updatesShape.GetDim(i)) + ", " + std::to_string(indicesShape.GetDim(i))).c_str(),
                "first indicesDimNum dimensions of updates must equal indices dimensions"),
            return ge::GRAPH_FAILED);
    }

    for (uint64_t i = 1; i < varDimNum; i++) {
        OP_CHECK_IF(
            (static_cast<uint32_t>(updatesShape.GetDim(i + indicesDimNum - 1)) !=
             static_cast<uint32_t>(varShape.GetDim(i))),
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                opName, "value, var",
                (std::to_string(updatesShape.GetDim(i + indicesDimNum - 1)) + ", " + std::to_string(varShape.GetDim(i)))
                    .c_str(),
                "remaining dimensions of updates must equal var dimensions (excluding first dim)"),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

bool ScatterAddWithSortedBaseTiling::IsCapable() { return true; }

ge::graphStatus ScatterAddWithSortedBaseTiling::DoOpTiling() { return ge::GRAPH_SUCCESS; }

ge::graphStatus ScatterAddWithSortedBaseTiling::DoLibApiTiling() { return ge::GRAPH_SUCCESS; }

uint64_t ScatterAddWithSortedBaseTiling::GetTilingKey() const { return 0; }

ge::graphStatus ScatterAddWithSortedBaseTiling::GetWorkspaceSize() { return ge::GRAPH_SUCCESS; }

ge::graphStatus ScatterAddWithSortedBaseTiling::PostTiling() { return ge::GRAPH_SUCCESS; }

} // namespace optiling

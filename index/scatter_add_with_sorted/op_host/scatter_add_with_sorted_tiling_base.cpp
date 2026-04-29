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
        ss << element << " ";
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
    OP_CHECK_IF(platformInfo == nullptr, OP_LOGE(opName, "fail to get platform info"), return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    auto aivNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((aivNum <= 0), OP_LOGE(opName, "fail to get coreNum."), return ge::GRAPH_FAILED);
    totalCoreNum_ = aivNum;
    uint64_t ubSizePlatForm = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    ubSize_ = ubSizePlatForm;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterAddWithSortedBaseTiling::GetShapeAttrsInfo()
{
    auto var = context_->GetInputShape(VAR_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, var);
    auto varShape = var->GetStorageShape();
    varSize_ = varShape.GetShapeSize();
    varShape_[0] = varShape.GetDim(0);
    varShape_[1] = (varShape_[0] != 0) ? varSize_ / varShape_[0] : 0;
    auto indices = context_->GetInputShape(SORTED_INDEX_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, indices);
    auto indiceShape = indices->GetStorageShape();
    indicesNum_ = indiceShape.GetShapeSize();

    auto updates = context_->GetInputShape(UPDATES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, updates);
    auto updateShape = updates->GetStorageShape();
    updatesSize_ = updateShape.GetShapeSize();
    uint64_t updatesDims = updateShape.GetDimNum();

    if (updatesDims == 0 || (updatesDims == 1 && updatesSize_ == 1)) {
        isUpdateScalar_ = 1;
    } else {
        OP_CHECK_IF(
            CheckUpdatesShape(varShape, indiceShape, updateShape) != ge::GRAPH_SUCCESS,
            OP_LOGE(opName, "update shape check failed."), return ge::GRAPH_FAILED);
    }

    auto pos = context_->GetInputShape(POS_IDX);
    if (pos != nullptr) {
        auto posShape = pos->GetStorageShape();
        OP_CHECK_IF(
            static_cast<uint64_t>(posShape.GetShapeSize()) != indicesNum_,
            OP_LOGE(opName, "pos shape must be equal to indices shape."), return ge::GRAPH_FAILED);
        hasPos_ = true;
    }

    OP_CHECK_IF(
        CheckInputDtype() != ge::GRAPH_SUCCESS, OP_LOGE(opName, "input dtype check failed."), return ge::GRAPH_FAILED);

    bool isFloat = (varDtype_ == ge::DT_FLOAT || varDtype_ == ge::DT_FLOAT16 || varDtype_ == ge::DT_BF16);
    if (context_->GetDeterministic() && !isUpdateScalar_ && isFloat) {
        isDeterminTemplate_ = 1;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterAddWithSortedBaseTiling::CheckInputDtype()
{
    auto indicesPtr = context_->GetInputDesc(SORTED_INDEX_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, indicesPtr);
    indicesDtype_ = indicesPtr->GetDataType();
    OP_CHECK_IF(
        (INDICES_DTYPE_SET.find(indicesDtype_) == INDICES_DTYPE_SET.end()),
        OP_LOGE(
            opName, "indices data dtype only support %s, currently, please check.",
            ToString(INDICES_DTYPE_SET).c_str()),
        return ge::GRAPH_FAILED);
    indicesDtypeSize_ = ge::GetSizeByDataType(indicesDtype_);
    OP_CHECK_IF(indicesDtypeSize_ <= 0, OP_LOGE(opName, "get indicesDtype size fail."), return ge::GRAPH_FAILED);

    if (hasPos_) {
        auto posPtr = context_->GetInputDesc(POS_IDX);
        OP_CHECK_NULL_WITH_CONTEXT(context_, posPtr);
        posDtype_ = posPtr->GetDataType();
        OP_CHECK_IF(
            (INDICES_DTYPE_SET.find(posDtype_) == INDICES_DTYPE_SET.end()),
            OP_LOGE(
                opName, "pos data dtype only support %s, currently, please check.",
                ToString(INDICES_DTYPE_SET).c_str()),
            return ge::GRAPH_FAILED);
        posDtypeSize_ = ge::GetSizeByDataType(posDtype_);
        OP_CHECK_IF(posDtypeSize_ <= 0, OP_LOGE(opName, "get posDtype size fail."), return ge::GRAPH_FAILED);
    }

    auto dataPtr = context_->GetInputDesc(VAR_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dataPtr);
    varDtype_ = dataPtr->GetDataType();
    OP_CHECK_IF(
        (VAR_DTYPE_SET.find(varDtype_) == VAR_DTYPE_SET.end()),
        OP_LOGE(opName, "var data dtype only support %s, please check.", ToString(VAR_DTYPE_SET).c_str()),
        return ge::GRAPH_FAILED);
    varTypeSize_ = ge::GetSizeByDataType(varDtype_);
    OP_CHECK_IF(varTypeSize_ <= 0, OP_LOGE(opName, "get dataType size fail."), return ge::GRAPH_FAILED);
    auto updatePtr = context_->GetInputDesc(UPDATES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, updatePtr);
    auto updatesType = updatePtr->GetDataType();
    OP_CHECK_IF(
        (VAR_DTYPE_SET.find(updatesType) == VAR_DTYPE_SET.end()),
        OP_LOGE(opName, "updates data dtype only support %s currently, please check.", ToString(VAR_DTYPE_SET).c_str()),
        return ge::GRAPH_FAILED);
    updatesDtypeSize_ = ge::GetSizeByDataType(updatesType);
    OP_CHECK_IF(
        (updatesType != varDtype_), OP_LOGE(opName, "expected updates dtype to be equal to var dtype, please check."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterAddWithSortedBaseTiling::CheckUpdatesShape(
    const gert::Shape& varShape, const gert::Shape& indicesShape, const gert::Shape& updatesShape)
{
    uint64_t varDimNum = static_cast<uint64_t>(varShape.GetDimNum());
    uint64_t indicesDimNum = static_cast<uint64_t>(indicesShape.GetDimNum());
    uint64_t updatesDimNum = static_cast<uint64_t>(updatesShape.GetDimNum());
    OP_CHECK_IF(
        (updatesDimNum != indicesDimNum + varDimNum - 1),
        OP_LOGE(opName, "updatesDimNum must have the same number of indicesDimNum add varDimNum - 1, please check."),
        return ge::GRAPH_FAILED);
    for (uint64_t i = 0; i < indicesDimNum; i++) {
        OP_CHECK_IF(
            (static_cast<uint32_t>(updatesShape.GetDim(i)) != static_cast<uint32_t>(indicesShape.GetDim(i))),
            OP_LOGE(
                opName,
                "updatesShape should be equal to the shape of 'indices' concats the shape of 'var' except for the "
                "first dimension."),
            return ge::GRAPH_FAILED);
    }

    for (uint64_t i = 1; i < varDimNum; i++) {
        OP_CHECK_IF(
            (static_cast<uint32_t>(updatesShape.GetDim(i + indicesDimNum - 1)) !=
             static_cast<uint32_t>(varShape.GetDim(i))),
            OP_LOGE(
                opName,
                "updatesShape should be equal to the shape of 'indices' concats the shape of 'var' except for the "
                "first dimension."),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

bool ScatterAddWithSortedBaseTiling::IsCapable()
{
    return true;
}

ge::graphStatus ScatterAddWithSortedBaseTiling::DoOpTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterAddWithSortedBaseTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t ScatterAddWithSortedBaseTiling::GetTilingKey() const
{
    return 0;
}

ge::graphStatus ScatterAddWithSortedBaseTiling::GetWorkspaceSize()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterAddWithSortedBaseTiling::PostTiling()
{
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
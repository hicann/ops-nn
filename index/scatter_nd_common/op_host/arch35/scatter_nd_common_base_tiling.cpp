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
 * \file scatter_nd_common_base_tiling.cpp
 * \brief scatter_nd_common_base_tiling
 */

#include "scatter_nd_common_base_tiling.h"

using namespace AscendC;
using namespace ge;

namespace optiling {
constexpr uint64_t ASCENDC_WORKSPACE = static_cast<uint64_t>(16) * 1024 * 1024;
static constexpr uint64_t CASTMODE1 = 1;   // int32 Cast int16
static constexpr uint64_t CASTMODE2 = 2;   // int64 Cast int32
static constexpr uint64_t CASTMODE3 = 3;   // int64 Cast int16
static constexpr uint64_t CASTMODE4 = 4;   // int32 Cast uint8
static constexpr uint64_t CASTMODE5 = 5;   // int64 Cast uint8

static constexpr uint16_t RANK_MIN_VALUE = 1;
static constexpr uint16_t RANK_MAX_VALUE = 7;


ge::graphStatus ScatterNdCommonBaseTiling::GetCastType()
{
    indiceCastDtype_ = indiceDtype_;

    if (indiceDtype_ == ge::DT_INT32) {
        if (varInAxis_ < UINT8_MAX) {
            indiceCastMode_ = CASTMODE4;          // int32 Cast uint8
            indiceCastDtype_ = ge::DT_UINT8;
        } else if (varInAxis_ < INT16_MAX) {
            indiceCastMode_ = CASTMODE1;          // int32 Cast int16
            indiceCastDtype_ = ge::DT_INT16;
        }
    } else {
        if (varInAxis_ < UINT8_MAX) {
            indiceCastMode_ = CASTMODE5;          // int64 Cast uint8
            indiceCastDtype_ = ge::DT_UINT8;
        } else if (varInAxis_ < INT16_MAX) {
            indiceCastMode_ = CASTMODE3;          // int64 Cast int16
            indiceCastDtype_ = ge::DT_INT16;
        } else if (varInAxis_ < INT32_MAX) {
            indiceCastMode_ = CASTMODE2;          // int64 Cast int32
            indiceCastDtype_ = ge::DT_INT32;
        }
    }

    if (indiceCastMode_ != 0) {
        indiceCastDtypeSize_ = ge::GetSizeByDataType(indiceCastDtype_);
    }
    return ge::GRAPH_SUCCESS;
}

void ScatterNdCommonBaseTiling::SetStride()
{
    auto outPutShape = context_->GetOutputShape(OUTPUT_IDX_SHAPE)->GetStorageShape();
    strideList_[rankSize_ - ONE] = static_cast<uint64_t>(1);
    for (int16_t dim = static_cast<int16_t>(rankSize_ - TWO); dim >= 0; --dim) {
        strideList_[dim] = strideList_[dim + 1] * outPutShape.GetDim(dim + 1);
    }
}


ge::graphStatus ScatterNdCommonBaseTiling::GetPlatformInfo()
{
    auto compileInfo = context_->GetCompileInfo<ScatterNdCommonCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, compileInfo);
    totalCoreNum_ = compileInfo->core_num;
    ubSize_ = compileInfo->ub_size;

    OP_CHECK_IF(totalCoreNum_ <= 0, OP_LOGE(context_, "GetPlatformInfo get corenum <= 0"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ubSize_ <= 0, OP_LOGE(context_, "GetPlatformInfo get ub size <= 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}


uint32_t ScatterNdCommonBaseTiling::GetSortTmpSize(ge::DataType dataType, uint32_t lastAxisNum, bool isDescend)
{
    std::vector<int64_t> shapeVec = {lastAxisNum};
    ge::Shape srcShape(shapeVec);
    AscendC::SortConfig config;
    config.type = AscendC::SortType::RADIX_SORT;
    config.isDescend = isDescend;
    config.hasSrcIndex = false;
    config.hasDstIndex = true;
    uint32_t maxValue = 0;
    uint32_t minValue = 0;
    AscendC::GetSortMaxMinTmpSize(srcShape, dataType, ge::DT_UINT32, false, config, maxValue, minValue);
    OP_LOGI("RadixSortTilingForAscendC", "Need tmp buffer %u byte for ac sort api", maxValue);
    return maxValue;
}


ge::graphStatus ScatterNdCommonBaseTiling::GetShapeAttrsInfo()
{
    auto opName = context_->GetNodeName();
    OP_LOGD(opName, "GetShapeAttrsInfo begin.");

    auto var = context_->GetInputTensor(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, var);
    auto varShapeSize = var->GetShapeSize();
    OP_CHECK_IF((varShapeSize <= 0),
            OP_LOGE(opName, "var shape size is invalid(%ld)", varShapeSize),
            return ge::GRAPH_FAILED);
    auto varDesc = context_->GetInputDesc(INPUT_IDX_UPDATES);
    OP_CHECK_NULL_WITH_CONTEXT(context_, varDesc);
    auto varDtype = varDesc->GetDataType();
    varTypeSize_ = ge::GetSizeByDataType(varDtype);
    OP_CHECK_IF(
        varTypeSize_ <= 0,
        OP_LOGE(context_, "varTypeSize must be greater than 0, varTypeSize: %ld", varTypeSize_),
        return ge::GRAPH_FAILED);

    auto indices = context_->GetInputTensor(INPUT_IDX_INDICES);
    OP_CHECK_NULL_WITH_CONTEXT(context_, indices);
    indiceShapeSize_ = indices->GetShapeSize();
    OP_CHECK_IF((indiceShapeSize_ < 0),
            OP_LOGE(opName,
            "update shape size is invalid(%ld)", indiceShapeSize_), return ge::GRAPH_FAILED);
    auto indicesDesc = context_->GetInputDesc(INPUT_IDX_INDICES);
    OP_CHECK_NULL_WITH_CONTEXT(context_, indicesDesc);
    indiceDtype_ = indicesDesc->GetDataType();
    indicesTypeSize_ = ge::GetSizeByDataType(indiceDtype_);
    
    auto indiceShape = indices->GetStorageShape();
    auto indiceDims = indiceShape.GetDimNum();
    rankSize_ = indiceShape.GetDim(indiceDims - 1);
    OP_CHECK_IF(
        (RANK_MIN_VALUE > static_cast<uint16_t>(rankSize_) || static_cast<uint16_t>(rankSize_) > RANK_MAX_VALUE),
        OP_LOGE(opName,
        "rankSize_ %u out of range[1, 7], please check.", rankSize_),
        return ge::GRAPH_FAILED);
    
    auto updates = context_->GetInputTensor(INPUT_IDX_UPDATES);
    OP_CHECK_NULL_WITH_CONTEXT(context_, updates);
    updateShapeSize_ = updates->GetShapeSize();
    OP_CHECK_IF((updateShapeSize_ < 0),
                    OP_LOGE(opName,
                    "update shape size is invalid(%ld)", updateShapeSize_), return ge::GRAPH_FAILED);

    auto updateDesc = context_->GetInputDesc(INPUT_IDX_UPDATES);
    OP_CHECK_NULL_WITH_CONTEXT(context_, updateDesc);
    updateDtype_ = updateDesc->GetDataType();
    OP_CHECK_IF(
            (updateDtype_ != varDtype),
            OP_LOGE(opName, "updates [%s] and var [%s] must have the same dtype.",
                                          Ops::Base::ToString(updateDtype_).c_str(), Ops::Base::ToString(varDtype).c_str()),
          return ge::GRAPH_FAILED);

    auto outputShape = context_->GetOutputShape(OUTPUT_IDX_SHAPE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputShape);
    auto shapeValue = outputShape->GetStorageShape();
    uint64_t shapeRank = shapeValue.GetDimNum();
    OP_CHECK_IF((shapeRank < rankSize_),
            OP_LOGE(opName,
            "shapeRank %lu less than rank %u, please check.", shapeRank, rankSize_), 
            return ge::GRAPH_FAILED);

    for (uint64_t idx = 0; idx < shapeRank; idx++) {
      outPutShape_[idx] = shapeValue.GetDim(idx);
      outputShapeSize_ *= outPutShape_[idx];
    }

    if (indiceShapeSize_ == 0 || updateShapeSize_ == 0) {
        return ge::GRAPH_SUCCESS;
    }
    // indicesAxis_ equal updatesInAxis
    indicesAxis_ = indiceShapeSize_ / rankSize_; // g
    afterAxis_ = updateShapeSize_ / indicesAxis_; // n
    varInAxis_ = varShapeSize / afterAxis_; // m
    if (varInAxis_ < INT32_MAX) { // rank维索引合一可能超过int32最大值
        outOfSetTypeSize_ = indicesTypeSize_;
        outOfSetDtype_ = indiceDtype_;
    } else {
        outOfSetTypeSize_ = sizeof(int64_t);
        outOfSetDtype_ = ge::DataType::DT_INT64;
    }

    return ge::GRAPH_SUCCESS;
}



bool ScatterNdCommonBaseTiling::IsCapable()
{
    return true;
}

ge::graphStatus ScatterNdCommonBaseTiling::DoOpTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterNdCommonBaseTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t ScatterNdCommonBaseTiling::GetTilingKey() const
{
    return 0;
}

ge::graphStatus ScatterNdCommonBaseTiling::GetWorkspaceSize()
{
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = ASCENDC_WORKSPACE;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ScatterNdCommonBaseTiling::PostTiling()
{
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

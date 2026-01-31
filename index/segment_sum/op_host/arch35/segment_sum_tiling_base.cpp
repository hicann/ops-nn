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
 * \file segment_sum_tiling_base.cpp
 * \brief segment_sum_tiling_base
 */

#include "segment_sum_tiling_base.h"

using namespace AscendC;
using namespace ge;

namespace optiling {
static constexpr uint32_t INPUT_DATA_INDEX = 0;
static constexpr uint32_t INPUT_SEGMENT_IDS_INDEX = 1;
static constexpr uint32_t OUTPUT_Y_INDEX = 0;

static const std::map<ge::DataType, uint32_t> dataTypeMap = {{ge::DT_FLOAT, 0}, {ge::DT_FLOAT16, 1}, {ge::DT_BF16, 2},
                                                             {ge::DT_INT32, 3}, {ge::DT_INT64, 4},   {ge::DT_UINT32, 5},
                                                             {ge::DT_UINT64, 6}};

static const std::map<ge::DataType, uint32_t> indexTypeMap = {
    {ge::DT_INT32, 100},
    {ge::DT_INT64, 200},
};

ge::graphStatus SegmentSumBaseTiling::GetPlatformInfo()
{
    auto compileInfo =
        reinterpret_cast<const SegmentSumCompileInfo*>(context_->GetCompileInfo());
    OPS_CHECK_NULL_WITH_CONTEXT(context_, compileInfo);
    totalCoreNum_ = compileInfo->core_num;
    ubSize_ = compileInfo->ub_size;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SegmentSumBaseTiling::CheckInputDtype()
{
    auto dataDtypePtr = context_->GetInputDesc(INPUT_DATA_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, dataDtypePtr);
    auto segmentIdsDtypePtr = context_->GetInputDesc(INPUT_SEGMENT_IDS_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, segmentIdsDtypePtr);

    dataType_ = dataDtypePtr->GetDataType();
    idType_ = segmentIdsDtypePtr->GetDataType();
    auto dataTypeIter = dataTypeMap.find(dataType_);
    auto indexTypeIter = indexTypeMap.find(idType_);
    OP_CHECK_IF(
        dataTypeIter == dataTypeMap.end(),
        OP_LOGE(context_->GetNodeName(), "data dtype does not support !"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        indexTypeIter == indexTypeMap.end(),
        OP_LOGE(context_->GetNodeName(), "segment_ids dtype does not support !"),
        return ge::GRAPH_FAILED);

    valueTypeBytes_ = ge::GetSizeByDataType(dataType_);
    idTypeBytes_ = ge::GetSizeByDataType(idType_);
    
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SegmentSumBaseTiling::GetShapeAttrsInfo()
{
    OP_CHECK_IF(
        CheckInputDtype() != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "input dtype check failed."), 
        return ge::GRAPH_FAILED);

    auto dataShapePtr = context_->GetInputShape(INPUT_DATA_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, dataShapePtr);
    auto dataShape = dataShapePtr->GetStorageShape();

    auto segmentIdsShapePtr = context_->GetInputShape(INPUT_SEGMENT_IDS_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, segmentIdsShapePtr);
    auto segmentIdsShape = segmentIdsShapePtr->GetStorageShape();

    for (size_t i = 0; i < dataShape.GetDimNum(); ++i) {
        OP_CHECK_IF(
            dataShape.GetDim(i) < 0,
            OP_LOGE(
                context_->GetNodeName(), "Dimension must be >= 0, but got %ld", dataShape.GetDim(i)),
            return ge::GRAPH_FAILED);
        
        innerDim_ *= (i != 0) ? dataShape.GetDim(i) : 1;
    }
    outerDim_ = dataShape.GetDim(0);
    
    OP_CHECK_IF(
        outerDim_ != segmentIdsShape.GetDim(0),
        OP_LOGE(
            context_->GetNodeName(), "the dimension 0 of data shape should be same with segment_ids."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        segmentIdsShape.GetDimNum() != 1,
        OP_LOGE(context_->GetNodeName(), "segment_ids dimension should be 1 but got %zu", segmentIdsShape.GetDimNum()),
        return ge::GRAPH_FAILED);

    dataShapeSize_ = dataShape.GetShapeSize();
    auto segmentIdsShapeSize = segmentIdsShape.GetShapeSize();
    if (dataShapeSize_ == 0 && segmentIdsShapeSize == 0) {
        OP_LOGI(context_->GetNodeName(), "input and segment_ids are empty, return an empty tensor.");
        return ge::GRAPH_SUCCESS;
    }

    ubBlockSize_ = Ops::Base::GetUbBlockSize(context_);

    auto outputY = context_->GetOutputShape(OUTPUT_Y_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, outputY);
    auto yShape = Ops::Base::EnsureNotScalar(outputY->GetStorageShape());
    segmentNum_ = yShape.GetDim(0);
    
    return ge::GRAPH_SUCCESS;
}

bool SegmentSumBaseTiling::IsCapable()
{
    return true;
}

ge::graphStatus SegmentSumBaseTiling::DoOpTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SegmentSumBaseTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t SegmentSumBaseTiling::GetTilingKey() const
{
    return 0;
}

ge::graphStatus SegmentSumBaseTiling::GetWorkspaceSize()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SegmentSumBaseTiling::PostTiling()
{
    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling
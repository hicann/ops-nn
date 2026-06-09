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
 * \file sorted_sparse_segment_mean_grad_tiling_base.cpp
 * \brief
 */
#include "sorted_sparse_segment_mean_grad_tiling_base.h"
#include "op_common/atvoss/broadcast/broadcast_tiling.h"

using namespace AscendC;
using namespace ge;

namespace optiling {
static constexpr int64_t DTYPE_BYTES_4 = 4;
static constexpr int64_t INPUT_X = 0;
static constexpr int64_t INPUT_INDICES = 1;
static constexpr int64_t INPUT_INDICES_LOCATION = 2;
static constexpr int64_t INPUT_SEGMENT_IDS = 3;
static constexpr int64_t OUTPUT_DIM0 = 4;
static constexpr int64_t OUTPUT_Y = 0;

static constexpr int64_t FLOAT16_SIZE = 2;
static constexpr int64_t FLOAT32_SIZE = 4;


void SortedSparseSegmentMeanGradBaseTiling::PrintHardwareData() const
{
    OP_LOGD("SortedSparseSegmentMeanGradBaseTiling", "[SortedSparseSegmentMeanGrad] PrintHardwareData start running");

    std::ostringstream info;
    info << "hardwareData.ubSize: " << hardwareData.ubSize << std::endl;
    info << "hardwareData.coreNum: " << hardwareData.coreNum << std::endl;

    OP_LOGI("SortedSparseSegmentMeanGrad", "%s", info.str().c_str());
}

ge::graphStatus SortedSparseSegmentMeanGradBaseTiling::GetPlatformInfo()
{
    auto platformPtr = context_->GetPlatformInfo();
    if (platformPtr == nullptr) {
        auto compileInfoPtr = reinterpret_cast<const SortedSparseSegmentMeanGradCompileInfo*>(context_->GetCompileInfo());
        OP_TILING_CHECK(
            compileInfoPtr == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "compile info is null"),
            return ge::GRAPH_FAILED);
        hardwareData.coreNum = compileInfoPtr->coreNum;
        hardwareData.ubSize = compileInfoPtr->ubSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformPtr);
        hardwareData.coreNum = ascendcPlatform.GetCoreNumAiv();

        uint64_t ubSizePlatform;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatform);
        hardwareData.ubSize = static_cast<int64_t>(ubSizePlatform);
    }

    if (hardwareData.coreNum == 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            context_->GetNodeName(), "coreNum",  "0", "platform coreNum must be greater than 0");
        return ge::GRAPH_FAILED;
    }
    
    PrintHardwareData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SortedSparseSegmentMeanGradBaseTiling::GetXInfoAndCheck()
{
    auto inputX = context_->GetInputShape(INPUT_X);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, inputX);
    xShape_ = Ops::Base::EnsureNotScalar(inputX->GetStorageShape());

    if (xShape_.GetDimNum() < 1) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            context_->GetNodeName(), "x", std::to_string(xShape_.GetDimNum()).c_str(),
            "dimNum of x must be greater than or equal to 1");
        return ge::GRAPH_FAILED;
    }

    if (xShape_.GetDim(0) <= 0) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            context_->GetNodeName(), "x", std::to_string(xShape_.GetDim(0)).c_str(), "dim0 of x must be greater than 0");
        return ge::GRAPH_FAILED;
    }

    if (xShape_.GetShapeSize() <= 0) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
            context_->GetNodeName(), "x", std::to_string(xShape_.GetShapeSize()).c_str(), "shapeSize of x must be greater than 0");
        return ge::GRAPH_FAILED;
    }
    inputData.segmentNum = xShape_.GetDim(0);
    inputData.innerSize = 1;
    for (size_t i = 1; i < xShape_.GetDimNum(); i++) {
        inputData.innerSize *= xShape_.GetDim(i);
    }

    auto inputDesc = context_->GetInputDesc(INPUT_X);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, inputDesc);
    inputData.inputDtype = inputDesc->GetDataType();
    if (inputData.inputDtype != ge::DataType::DT_FLOAT16 && inputData.inputDtype != ge::DataType::DT_FLOAT &&
        inputData.inputDtype != ge::DataType::DT_BF16) {
        OP_LOGE_FOR_INVALID_DTYPE(
            context_->GetNodeName(), "x", std::to_string(static_cast<int32_t>(inputData.inputDtype)).c_str(),
            "[DT_FLOAT16, DT_FLOAT, DT_BF16]");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SortedSparseSegmentMeanGradBaseTiling::GetYInfoAndCheck()
{
    auto outputDesc = context_->GetOutputDesc(0);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, outputDesc);
    if (outputDesc->GetDataType() != inputData.inputDtype) {
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            context_->GetNodeName(), "x, y",
            (std::to_string(static_cast<int32_t>(inputData.inputDtype)) + ", " +
             std::to_string(static_cast<int32_t>(outputDesc->GetDataType())))
                .c_str(),
            "dtype of x must be same as dtype of y");
        return ge::GRAPH_FAILED;
    }

    auto outputY = context_->GetOutputShape(OUTPUT_Y);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, outputY);
    auto yShape = Ops::Base::EnsureNotScalar(outputY->GetStorageShape());
    auto yShapeDimNum = yShape.GetDimNum();
    if (xShape_.GetDimNum() != yShapeDimNum) {
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            context_->GetNodeName(), "x, y", (std::to_string(xShape_.GetDimNum()) + ", " + std::to_string(yShapeDimNum)).c_str(),
            "dimNum of x must equal dimNum of y");
        return ge::GRAPH_FAILED;
    }

    for (size_t i = 1; i < yShapeDimNum; i++) {
        if (xShape_.GetDim(i) != yShape.GetDim(i)) {
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                context_->GetNodeName(), "x, y", (std::to_string(xShape_.GetDim(i)) + ", " + std::to_string(yShape.GetDim(i))).c_str(),
                "shape of x must match the shape of output y starting from the first dim");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SortedSparseSegmentMeanGradBaseTiling::GetIndicesInfoAndCheck()
{
    // inputIndices信息获取
    auto inputIndices = context_->GetInputShape(INPUT_INDICES);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, inputIndices);
    indicesShape_ = Ops::Base::EnsureNotScalar(inputIndices->GetStorageShape());

    if (indicesShape_.GetDimNum() != 1) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "indices", std::to_string(indicesShape_.GetDimNum()).c_str(), "1");
        return ge::GRAPH_FAILED;
    }

    if (indicesShape_.GetShapeSize() <= 0) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
            context_->GetNodeName(), "indices", std::to_string(indicesShape_.GetShapeSize()).c_str(),
            "shapeSize of indices must be greater than 0");
        return ge::GRAPH_FAILED;
    }
    inputData.outterSize = indicesShape_.GetShapeSize();

    auto inputIndicesDesc = context_->GetInputDesc(INPUT_INDICES);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, inputIndicesDesc);
    inputData.indicesDtype = inputIndicesDesc->GetDataType();
    if (inputData.indicesDtype != ge::DataType::DT_INT32 && inputData.indicesDtype != ge::DataType::DT_INT64) {
        OP_LOGE_FOR_INVALID_DTYPE(
            context_->GetNodeName(), "indices", std::to_string(static_cast<int32_t>(inputData.indicesDtype)).c_str(),
            "[DT_INT32, DT_INT64]");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SortedSparseSegmentMeanGradBaseTiling::GetSegmentIdsInfoAndCheck()
{
    // inputSegmentIds信息获取
    auto inputSegmentIds = context_->GetInputShape(INPUT_SEGMENT_IDS);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, inputSegmentIds);
    auto segmentIdsShape = Ops::Base::EnsureNotScalar(inputSegmentIds->GetStorageShape());
    if (indicesShape_.GetShapeSize() != segmentIdsShape.GetShapeSize()) {
        OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
            context_->GetNodeName(), "indices, segment_ids",
            (std::to_string(indicesShape_.GetShapeSize()) + ", " + std::to_string(segmentIdsShape.GetShapeSize()))
                .c_str(),
            "shapeSize of indices must equal shapeSize of segment_ids");
        return ge::GRAPH_FAILED;
    }
    if (segmentIdsShape.GetDimNum() != 1) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(context_->GetNodeName(), "segment_ids", std::to_string(segmentIdsShape.GetDimNum()).c_str(), "1");
        return ge::GRAPH_FAILED;
    }

    auto inputSegmentIdsDesc = context_->GetInputDesc(INPUT_SEGMENT_IDS);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, inputSegmentIdsDesc);
    inputData.segmentIdsDtype = inputSegmentIdsDesc->GetDataType();
    if (inputData.segmentIdsDtype != ge::DataType::DT_INT32 && inputData.segmentIdsDtype != ge::DataType::DT_INT64) {
        OP_LOGE_FOR_INVALID_DTYPE(
            context_->GetNodeName(), "segment_ids", std::to_string(static_cast<int32_t>(inputData.segmentIdsDtype)).c_str(),
            "[DT_INT32, DT_INT64]");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SortedSparseSegmentMeanGradBaseTiling::GetOutputDim0InfoAndCheck()
{
    // outputDim0信息获取
    auto outputDim0 = context_->GetInputShape(OUTPUT_DIM0);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, outputDim0);
    auto outputDim0Shape = Ops::Base::EnsureNotScalar(outputDim0->GetStorageShape());
    if (outputDim0Shape.GetShapeSize() != 1 || outputDim0Shape.GetDimNum() != 1) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            context_->GetNodeName(), "outputDim0", std::to_string(outputDim0Shape.GetDimNum()).c_str(),
            "dimNum and shapeSize of outputDim0 must both be 1");
        return ge::GRAPH_FAILED;
    }
    auto outputDim0Desc = context_->GetInputDesc(OUTPUT_DIM0);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, outputDim0Desc);
    if (outputDim0Desc->GetDataType() != ge::DataType::DT_INT32) {
        OP_LOGE_FOR_INVALID_DTYPE(
            context_->GetNodeName(), "outputDim0", std::to_string(static_cast<int32_t>(outputDim0Desc->GetDataType())).c_str(),
            "DT_INT32");
        return ge::GRAPH_FAILED;
    }
    const gert::Tensor* outputDim0Tensor = context_->GetInputTensor(OUTPUT_DIM0);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, outputDim0Tensor);
    inputData.outputDim0 = static_cast<int32_t>(outputDim0Tensor->GetData<int32_t>()[0]);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SortedSparseSegmentMeanGradBaseTiling::GetIndicesLocationInfoAndCheck()
{
    // inputIndicesLocation信息获取
    auto inputIndicesLocation = context_->GetInputShape(INPUT_INDICES_LOCATION);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, inputIndicesLocation);
    auto indicesLocationShape = Ops::Base::EnsureNotScalar(inputIndicesLocation->GetStorageShape());
    if (indicesShape_.GetShapeSize() != indicesLocationShape.GetShapeSize()) {
        OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
            context_->GetNodeName(), "indices, indicesLocation",
            (std::to_string(indicesShape_.GetShapeSize()) + ", " + std::to_string(indicesLocationShape.GetShapeSize()))
                .c_str(),
            "shapeSize of indices must equal shapeSize of indicesLocation");
        return ge::GRAPH_FAILED;
    }
    if (indicesLocationShape.GetDimNum() != 1) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(
            context_->GetNodeName(), "indicesLocation", std::to_string(indicesLocationShape.GetDimNum()).c_str(), "1");
        return ge::GRAPH_FAILED;
    }

    auto inputIndicesLocationDesc = context_->GetInputDesc(INPUT_INDICES_LOCATION);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, inputIndicesLocationDesc);
    inputData.indicesLocationDtype = inputIndicesLocationDesc->GetDataType();
    if (inputData.indicesLocationDtype != ge::DataType::DT_INT32 &&
        inputData.indicesLocationDtype != ge::DataType::DT_INT64) {
        OP_LOGE_FOR_INVALID_DTYPE(
            context_->GetNodeName(), "indicesLocation", std::to_string(static_cast<int32_t>(inputData.indicesLocationDtype)).c_str(),
            "[DT_INT32, DT_INT64]");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SortedSparseSegmentMeanGradBaseTiling::GetShapeAttrsInfo()
{
    OP_LOGD("SortedSparseSegmentMeanGradBaseTiling", "[SortedSparseSegmentMeanGrad] enter GetShapeAttrsInfo");

    if (GetXInfoAndCheck() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (GetYInfoAndCheck() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (GetIndicesInfoAndCheck() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (GetSegmentIdsInfoAndCheck() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (GetOutputDim0InfoAndCheck() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (GetIndicesLocationInfoAndCheck() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    PrintInputData();
    return ge::GRAPH_SUCCESS;
}

void SortedSparseSegmentMeanGradBaseTiling::PrintInputData() const
{
    OP_LOGD("SortedSparseSegmentMeanGradBaseTiling", "[SortedSparseSegmentMeanGrad] PrintInputData start running");

    std::ostringstream info;
    info << "inputData.innerSize: " << inputData.innerSize << std::endl;
    info << "inputData.segmentNum: " << inputData.segmentNum << std::endl;
    info << "inputData.outterSize: " << inputData.outterSize << std::endl;
    info << "inputData.inputDtype: " << inputData.inputDtype << std::endl;
    info << "inputData.indicesDtype: " << inputData.indicesDtype << std::endl;
    info << "inputData.segmentIdsDtype: " << inputData.segmentIdsDtype << std::endl;
    info << "inputData.indicesLocationDtype: " << inputData.indicesLocationDtype << std::endl;
    info << "inputData.outputDim0: " << inputData.outputDim0 << std::endl;

    OP_LOGI("SortedSparseSegmentMeanGrad", "%s", info.str().c_str());
}

bool SortedSparseSegmentMeanGradBaseTiling::IsCapable()
{
    return true;
}

ge::graphStatus SortedSparseSegmentMeanGradBaseTiling::DoOpTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SortedSparseSegmentMeanGradBaseTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t SortedSparseSegmentMeanGradBaseTiling::GetTilingKey() const
{
    return 0;
}

ge::graphStatus SortedSparseSegmentMeanGradBaseTiling::GetWorkspaceSize()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SortedSparseSegmentMeanGradBaseTiling::PostTiling()
{
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
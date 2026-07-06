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
 * \file sparse_segment_mean_tiling_base.cpp
 * \brief
 */

#include "sparse_segment_mean_tiling_base.h"
#include "op_common/atvoss/broadcast/broadcast_tiling.h"

using namespace AscendC;
using namespace ge;

namespace optiling {
static constexpr int64_t DTYPE_BYTES_4 = 4;
static constexpr int64_t INPUT_X = 0;
static constexpr int64_t OUTPUT_Y = 0;
static constexpr int64_t INPUT_INDICES = 1;
static constexpr int64_t INPUT_SEGMENT_IDS = 2;

static constexpr int64_t FLOAT16_SIZE = 2;
static constexpr int64_t FLOAT32_SIZE = 4;
static constexpr int64_t INT32_SIZE = 4;
static constexpr int64_t INT64_SIZE = 8;

void SparseSegmentMeanBaseTiling::PrintHardwareData() const
{
    OP_LOGD("SparseSegmentMeanBaseTiling", "[SparseSegmentMean] PrintHardwareData start running");

    std::ostringstream info;
    info << "hardwareData.ubSize: " << hardwareData.ubSize << std::endl;
    info << "hardwareData.coreNum: " << hardwareData.coreNum << std::endl;

    OP_LOGI("SparseSegmentMean", "%s", info.str().c_str());
}

ge::graphStatus SparseSegmentMeanBaseTiling::GetPlatformInfo()
{
    auto platformPtr = context_->GetPlatformInfo();
    if (platformPtr == nullptr) {
        auto compileInfoPtr = reinterpret_cast<const SparseSegmentMeanCompileInfo*>(context_->GetCompileInfo());
        OP_TILING_CHECK(compileInfoPtr == nullptr, CUBE_INNER_ERR_REPORT(context_, "compile info is null"),
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

    OP_TILING_CHECK(hardwareData.coreNum == 0, CUBE_INNER_ERR_REPORT(context_, "coreNum is 0"),
                    return ge::GRAPH_FAILED);

    PrintHardwareData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseSegmentMeanBaseTiling::GetXAndYInfoAndCheck()
{
    auto inputX = context_->GetInputShape(INPUT_X);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputX);
    auto xShape = Ops::Base::EnsureNotScalar(inputX->GetStorageShape());

    auto xShapeDimNum = xShape.GetDimNum();
    OP_TILING_CHECK(
        xShapeDimNum < 1,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "x", std::to_string(xShapeDimNum).c_str(),
                                                 "The shape of x must be greater than or equal to 1."),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        xShape.GetDim(0) <= 0,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "x", std::to_string(xShape.GetDim(0)).c_str(),
                                                 "The 0 axis of x must be greater than 0."),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(xShape.GetShapeSize() <= 0,
                    OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(context_->GetNodeName(), "x",
                                                              std::to_string(xShape.GetShapeSize()).c_str(),
                                                              "The shape size of x must be greater than 0."),
                    return ge::GRAPH_FAILED);
    inputData.gatherSize = xShape.GetDim(0);
    inputData.innerSize = 1;
    for (size_t i = 1; i < xShapeDimNum; i++) {
        inputData.innerSize *= xShape.GetDim(i);
    }

    auto inputDesc = context_->GetInputDesc(INPUT_X);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputDesc);
    inputData.inputDtype = inputDesc->GetDataType();
    if (inputData.inputDtype != ge::DataType::DT_FLOAT16 && inputData.inputDtype != ge::DataType::DT_FLOAT &&
        inputData.inputDtype != ge::DataType::DT_BF16) {
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "x", std::to_string(inputData.inputDtype).c_str(),
                                  "float16, float32 and bfloat16");
        return ge::GRAPH_FAILED;
    }

    inputData.inputBytes = inputData.inputDtype == ge::DataType::DT_FLOAT ? FLOAT32_SIZE : FLOAT16_SIZE;
    auto outputDesc = context_->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputDesc);
    if (outputDesc->GetDataType() != inputData.inputDtype) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "y",
                                              std::to_string(outputDesc->GetDataType()).c_str(),
                                              "The dtype of y must be the same as dtype of x.");
        return ge::GRAPH_FAILED;
    }

    auto outputY = context_->GetOutputShape(OUTPUT_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputY);
    auto yShape = Ops::Base::EnsureNotScalar(outputY->GetStorageShape());
    inputData.segmentNum = yShape.GetDim(0);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseSegmentMeanBaseTiling::GetIndicesAndSegmentIdsInfoAndCheck()
{
    auto inputIndices = context_->GetInputShape(INPUT_INDICES);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputIndices);
    auto indicesShape = Ops::Base::EnsureNotScalar(inputIndices->GetStorageShape());

    OP_TILING_CHECK(indicesShape.GetDimNum() != 1,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "indices",
                                                             std::to_string(indicesShape.GetDimNum()).c_str(),
                                                             "The shape dim of indices must be 1."),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(indicesShape.GetShapeSize() <= 0,
                    OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(context_->GetNodeName(), "indices",
                                                              std::to_string(indicesShape.GetShapeSize()).c_str(),
                                                              "The shape size of indices must be greater than 0."),
                    return ge::GRAPH_FAILED);
    inputData.outterSize = indicesShape.GetShapeSize();

    auto inputIndicesDesc = context_->GetInputDesc(INPUT_INDICES);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputIndicesDesc);
    inputData.indicesDtype = inputIndicesDesc->GetDataType();
    inputData.indicesBytes = inputData.indicesDtype == ge::DataType::DT_INT32 ? INT32_SIZE : INT64_SIZE;
    if (inputData.indicesDtype != ge::DataType::DT_INT32 && inputData.indicesDtype != ge::DataType::DT_INT64) {
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "indices", std::to_string(inputData.indicesDtype).c_str(),
                                  "int32, int64");
        return ge::GRAPH_FAILED;
    }

    auto inputSegmentIds = context_->GetInputShape(INPUT_SEGMENT_IDS);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputSegmentIds);
    auto segmentIdsShape = Ops::Base::EnsureNotScalar(inputSegmentIds->GetStorageShape());
    OP_TILING_CHECK(indicesShape.GetShapeSize() != segmentIdsShape.GetShapeSize(),
                    OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                        context_->GetNodeName(), "indices", std::to_string(indicesShape.GetShapeSize()).c_str(),
                        "The shape size of indices must be equal to the shape size of segment_ids."),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(segmentIdsShape.GetDimNum() != 1,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "segment_ids",
                                                             std::to_string(segmentIdsShape.GetDimNum()).c_str(),
                                                             "The shape dim of segment_ids must be 1."),
                    return ge::GRAPH_FAILED);

    auto inputSegmentIdsDesc = context_->GetInputDesc(INPUT_SEGMENT_IDS);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputSegmentIdsDesc);
    inputData.segmentIdsDtype = inputSegmentIdsDesc->GetDataType();
    inputData.segmentIdsBytes = inputData.segmentIdsDtype == ge::DataType::DT_INT32 ? INT32_SIZE : INT64_SIZE;
    if (inputData.segmentIdsDtype != ge::DataType::DT_INT32 && inputData.segmentIdsDtype != ge::DataType::DT_INT64) {
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "segment_ids",
                                  std::to_string(inputData.segmentIdsDtype).c_str(), "int32, int64");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseSegmentMeanBaseTiling::GetShapeAttrsInfo()
{
    OP_LOGD("SparseSegmentMeanBaseTiling", "[SparseSegmentMean] enter GetShapeAttrsInfo");

    OP_TILING_CHECK(GetXAndYInfoAndCheck() != ge::GRAPH_SUCCESS,
                    OP_LOGD(context_->GetNodeName(), "input x and output y check failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(GetIndicesAndSegmentIdsInfoAndCheck() != ge::GRAPH_SUCCESS,
                    OP_LOGD(context_->GetNodeName(), "input indices and segment_ids check failed."),
                    return ge::GRAPH_FAILED);

    PrintInputData();
    return ge::GRAPH_SUCCESS;
}

void SparseSegmentMeanBaseTiling::PrintInputData() const
{
    OP_LOGD("SparseSegmentMeanBaseTiling", "[SparseSegmentMean] enter PrintInputData");

    std::ostringstream info;
    info << "inputData.innerSize: " << inputData.innerSize << std::endl;
    info << "inputData.gatherSize: " << inputData.gatherSize << std::endl;
    info << "inputData.segmentNum: " << inputData.segmentNum << std::endl;
    info << "inputData.outterSize: " << inputData.outterSize << std::endl;
    info << "inputData.inputDtype: " << inputData.inputDtype << std::endl;
    info << "inputData.indicesDtype: " << inputData.indicesDtype << std::endl;
    info << "inputData.segmentIdsDtype: " << inputData.segmentIdsDtype << std::endl;
    info << "inputData.inputBytes: " << inputData.inputBytes << std::endl;
    info << "inputData.indicesBytes: " << inputData.indicesBytes << std::endl;
    info << "inputData.segmentIdsBytes: " << inputData.segmentIdsBytes << std::endl;

    OP_LOGI("SparseSegmentMean", "%s", info.str().c_str());
}

bool SparseSegmentMeanBaseTiling::IsCapable() { return true; }

ge::graphStatus SparseSegmentMeanBaseTiling::DoOpTiling() { return ge::GRAPH_SUCCESS; }

ge::graphStatus SparseSegmentMeanBaseTiling::DoLibApiTiling() { return ge::GRAPH_SUCCESS; }

uint64_t SparseSegmentMeanBaseTiling::GetTilingKey() const { return 0; }

ge::graphStatus SparseSegmentMeanBaseTiling::GetWorkspaceSize() { return ge::GRAPH_SUCCESS; }

ge::graphStatus SparseSegmentMeanBaseTiling::PostTiling() { return ge::GRAPH_SUCCESS; }
} // namespace optiling
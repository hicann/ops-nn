/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file in_training_reduce_v2_tiling_base_arch35.cpp
 * \brief
 */
#include <vector>
#include <algorithm>
#include <limits>
#include "in_training_reduce_v2_tiling.h"

using namespace ge;
using namespace Ops::Base;

namespace {
constexpr int64_t NCHW_DIM_NUM = 4;
constexpr int64_t NCDHW_DIM_NUM = 5;
constexpr int64_t ND_MIN_DIM_NUM = 2;
constexpr int64_t ND_MAX_DIM_NUM = 8;

constexpr int64_t DIM_0 = 0;
constexpr int64_t DIM_1 = 1;
constexpr int64_t DIM_2 = 2;

const std::vector<ge::DataType> DTYPE_LIST = {ge::DataType::DT_FLOAT16, ge::DataType::DT_FLOAT};

bool CheckedMulNonNegative(int64_t lhs, int64_t rhs, int64_t& result)
{
    if (lhs < 0 || rhs < 0 || (lhs != 0 && rhs > std::numeric_limits<int64_t>::max() / lhs)) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

bool IsChannelLast(ge::Format format) { return format == FORMAT_NHWC || format == FORMAT_NDHWC; }
} // namespace

namespace optiling {
const char* INTrainingReduceV2RegbaseTilingBase::GetNodeName() const
{
    const char* nodeName = context_->GetNodeName();
    return nodeName == nullptr ? "nil" : nodeName;
}

ge::graphStatus INTrainingReduceV2RegbaseTilingBase::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    aicoreParams_.blockDim = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    aicoreParams_.ubSize = ubSize;
    ubBlockSize = Ops::Base::GetUbBlockSize(context_);
    vectorLength = Ops::Base::GetVRegSize(context_);
    vlfp32 = vectorLength / sizeof(float);
    OP_CHECK_IF(
        vlfp32 <= 0 || ubBlockSize <= 0 || aicoreParams_.blockDim == 0 || aicoreParams_.ubSize == 0,
        OP_LOGE(GetNodeName(), "invalid platform info: vectorLength=%ld, ubBlockSize=%ld, blockDim=%lu, ubSize=%lu",
                vectorLength, ubBlockSize, aicoreParams_.blockDim, aicoreParams_.ubSize),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus INTrainingReduceV2RegbaseTilingBase::GetShapeAttrsInfo()
{
    if (context_ == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto xShape = context_->GetInputShape(INPUT_X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShape);
    xStorageShape = xShape->GetStorageShape();
    if (CheckShapeAllNotNegative(xStorageShape) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    auto xDesc = context_->GetInputDesc(INPUT_X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xDesc);
    dataType = xDesc->GetDataType();
    format = xDesc->GetFormat().GetStorageFormat();

    if (ParseShapeByFormat() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (CheckDtypeValid() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return CheckOutputShapes();
}

// N 位于首维；NHWC/NDHWC 的 C 位于末维，其余既有格式的 C 位于 dim1。
ge::graphStatus INTrainingReduceV2RegbaseTilingBase::ParseAndCheckNC()
{
    a1 = xStorageShape.GetDim(DIM_0);
    a0 = xStorageShape.GetDim(IsChannelLast(format) ? xStorageShape.GetDimNum() - 1 : DIM_1);
    OP_CHECK_IF(
        a1 <= 0 || a0 <= 0,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(GetNodeName(), "x", ToString(xStorageShape).c_str(),
                                              "The N-dimension and C-dimension of input x must be positive numbers"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus INTrainingReduceV2RegbaseTilingBase::ParseShapeByFormat()
{
    int64_t xDimNum = xStorageShape.GetDimNum();
    if (format == FORMAT_NCHW || format == FORMAT_NHWC) {
        OP_CHECK_IF(xDimNum != NCHW_DIM_NUM,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        GetNodeName(), "x", std::to_string(xDimNum).c_str(),
                        "The shape dim of input x must be 4 when the format of x is NCHW or NHWC"),
                    return ge::GRAPH_FAILED);
        if (ParseAndCheckNC() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    } else if (format == FORMAT_NCDHW || format == FORMAT_NDHWC) {
        OP_CHECK_IF(xDimNum != NCDHW_DIM_NUM,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        GetNodeName(), "x", std::to_string(xDimNum).c_str(),
                        "The shape dim of input x must be 5 when the format of x is NCDHW or NDHWC"),
                    return ge::GRAPH_FAILED);
        if (ParseAndCheckNC() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    } else if (format == FORMAT_ND) {
        OP_CHECK_IF(xDimNum < ND_MIN_DIM_NUM || xDimNum > ND_MAX_DIM_NUM,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        GetNodeName(), "x", std::to_string(xDimNum).c_str(),
                        "The shape dim of input x must be in the range of [2, 8] when the format of x is ND"),
                    return ge::GRAPH_FAILED);
        if (ParseAndCheckNC() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    } else {
        OP_LOGE_FOR_INVALID_FORMAT(GetNodeName(), "x", ToString(format).c_str(), "NCHW, NHWC, NCDHW, NDHWC or ND");
        return ge::GRAPH_FAILED;
    }

    r = 1;
    const int64_t spatialBegin = IsChannelLast(format) ? DIM_1 : DIM_2;
    const int64_t spatialEnd = IsChannelLast(format) ? xDimNum - 1 : xDimNum;
    for (int64_t i = spatialBegin; i < spatialEnd; ++i) {
        int64_t nextR = 0;
        OP_CHECK_IF(
            !CheckedMulNonNegative(r, xStorageShape.GetDim(i), nextR),
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(GetNodeName(), "x", ToString(xStorageShape).c_str(),
                                                  "The product of spatial dimensions must not exceed int64 range"),
            return ge::GRAPH_FAILED);
        r = nextR;
    }
    int64_t totalRows = 0;
    int64_t totalElements = 0;
    OP_CHECK_IF(!CheckedMulNonNegative(a1, a0, totalRows),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(GetNodeName(), "x", ToString(xStorageShape).c_str(),
                                                      "N*C must not exceed int64 range"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!CheckedMulNonNegative(totalRows, r, totalElements),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(GetNodeName(), "x", ToString(xStorageShape).c_str(),
                                                      "The input element count must not exceed int64 range"),
                return ge::GRAPH_FAILED);
    totalRows_ = static_cast<uint64_t>(totalRows);
    totalElements_ = static_cast<uint64_t>(totalElements);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus INTrainingReduceV2RegbaseTilingBase::CheckDtypeValid()
{
    // 校验 x 数据类型 ∈ {fp16, fp32}
    OP_CHECK_IF(std::find(DTYPE_LIST.begin(), DTYPE_LIST.end(), dataType) == DTYPE_LIST.end(),
                OP_LOGE_FOR_INVALID_DTYPE(GetNodeName(), "x", ToString(dataType).c_str(), "FLOAT or FLOAT16"),
                return ge::GRAPH_FAILED);

    // 校验输出 sum / square_sum 数据类型恒 fp32
    auto sumDesc = context_->GetOutputDesc(OUTPUT_SUM_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, sumDesc);
    OP_CHECK_IF(sumDesc->GetDataType() != ge::DT_FLOAT,
                OP_LOGE_FOR_INVALID_DTYPE(GetNodeName(), "sum", ToString(sumDesc->GetDataType()).c_str(), "FLOAT"),
                return ge::GRAPH_FAILED);

    auto squareSumDesc = context_->GetOutputDesc(OUTPUT_SQUARE_SUM_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, squareSumDesc);
    OP_CHECK_IF(
        squareSumDesc->GetDataType() != ge::DT_FLOAT,
        OP_LOGE_FOR_INVALID_DTYPE(GetNodeName(), "square_sum", ToString(squareSumDesc->GetDataType()).c_str(), "FLOAT"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus INTrainingReduceV2RegbaseTilingBase::CheckOutputShapes()
{
    const size_t rank = xStorageShape.GetDimNum();
    const size_t channelAxis = IsChannelLast(format) ? rank - 1 : DIM_1;
    for (uint32_t index : {OUTPUT_SUM_INDEX, OUTPUT_SQUARE_SUM_INDEX}) {
        const char* outputName = index == OUTPUT_SUM_INDEX ? "sum" : "square_sum";
        auto outputShape = context_->GetOutputShape(index);
        OP_CHECK_NULL_WITH_CONTEXT(context_, outputShape);
        const auto& shape = outputShape->GetStorageShape();
        OP_CHECK_IF(shape.GetDimNum() != rank,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(GetNodeName(), outputName, ToString(shape).c_str(),
                                                          "Output rank must match the input rank"),
                    return ge::GRAPH_FAILED);
        for (size_t axis = 0; axis < rank; ++axis) {
            const int64_t expected = (axis == DIM_0 || axis == channelAxis) ? xStorageShape.GetDim(axis) : 1;
            OP_CHECK_IF(
                shape.GetDim(axis) != expected,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(GetNodeName(), outputName, ToString(shape).c_str(),
                                                      "Output must retain N/C and set each spatial dimension to 1"),
                return ge::GRAPH_FAILED);
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus INTrainingReduceV2RegbaseTilingBase::CheckShapeAllNotNegative(const gert::Shape& shape)
{
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        OP_CHECK_IF(shape.GetDim(i) < 0,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(GetNodeName(), "x", ToString(shape).c_str(),
                                                          "Input x has negative axes"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus INTrainingReduceV2RegbaseTilingBase::DoLibApiTiling() { return ge::GRAPH_SUCCESS; }

ge::graphStatus INTrainingReduceV2RegbaseTilingBase::GetWorkspaceSize()
{
    // 计算 workspace 大小
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    workspaceSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

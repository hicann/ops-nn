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
constexpr int64_t DIM_3 = 3;
constexpr int64_t DIM_4 = 4;

const std::vector<ge::DataType> DTYPE_LIST = {ge::DataType::DT_FLOAT16, ge::DataType::DT_FLOAT};
} // namespace

namespace optiling {
ge::graphStatus INTrainingReduceV2RegbaseTilingBase::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    auto compileInfoPtr = reinterpret_cast<const INTrainingReduceV2CompileInfo*>(context_->GetCompileInfo());
    OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context_->GetNodeName(), "compile info is null"),
                return ge::GRAPH_FAILED);
    vlfp32 = compileInfoPtr->vectorLength / sizeof(float);
    ubBlockSize = compileInfoPtr->ubBlockSize;
    vectorLength = compileInfoPtr->vectorLength;

    if (platformInfo != nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        aicoreParams_.blockDim = ascendcPlatform.GetCoreNumAiv();
        uint64_t ubSizePlatForm;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
        aicoreParams_.ubSize = ubSizePlatForm;
    } else {
        aicoreParams_.blockDim = compileInfoPtr->coreNum;
        aicoreParams_.ubSize = compileInfoPtr->ubSize;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus INTrainingReduceV2RegbaseTilingBase::GetShapeAttrsInfo()
{
    if (context_ == nullptr) {
        OP_LOGE("INTrainingReduceV2", "TilingContext is nullptr.");
        return ge::GRAPH_FAILED;
    }

    auto xShape = context_->GetInputShape(INPUT_X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShape);
    xStorageShape = xShape->GetStorageShape();
    if (CheckShapeAllNotNegative(xStorageShape) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "Not supported shape info.");
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
        OP_LOGE(context_->GetNodeName(), "Not supported datatype info.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// AR 布局 [a1=N, a0=C, r=空间轴]：N / C 的取值与正数校验三种 format 完全一致，抽出复用。
ge::graphStatus INTrainingReduceV2RegbaseTilingBase::ParseAndCheckNC()
{
    a1 = xStorageShape.GetDim(DIM_0);
    a0 = xStorageShape.GetDim(DIM_1);
    OP_CHECK_IF(
        a1 <= 0 || a0 <= 0,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "x", ToString(xStorageShape).c_str(),
                                              "The N-dimension and C-dimension of input x must be positive numbers"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus INTrainingReduceV2RegbaseTilingBase::ParseShapeByFormat()
{
    int64_t xDimNum = xStorageShape.GetDimNum();
    if (format == FORMAT_NCHW) {
        OP_CHECK_IF(
            xDimNum != NCHW_DIM_NUM,
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->GetNodeName(), "x", std::to_string(xDimNum).c_str(),
                                                     "The shape dim of input x must be 4 when the format of x is NCHW"),
            return ge::GRAPH_FAILED);
        if (ParseAndCheckNC() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        r = xStorageShape.GetDim(DIM_2) * xStorageShape.GetDim(DIM_3);
    } else if (format == FORMAT_NCDHW) {
        OP_CHECK_IF(xDimNum != NCDHW_DIM_NUM,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        context_->GetNodeName(), "x", std::to_string(xDimNum).c_str(),
                        "The shape dim of input x must be 5 when the format of x is NCDHW"),
                    return ge::GRAPH_FAILED);
        if (ParseAndCheckNC() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        r = xStorageShape.GetDim(DIM_2) * xStorageShape.GetDim(DIM_3) * xStorageShape.GetDim(DIM_4);
    } else if (format == FORMAT_ND) {
        OP_CHECK_IF(xDimNum < ND_MIN_DIM_NUM || xDimNum > ND_MAX_DIM_NUM,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        context_->GetNodeName(), "x", std::to_string(xDimNum).c_str(),
                        "The shape dim of input x must be in the range of [2, 8] when the format of x is ND"),
                    return ge::GRAPH_FAILED);
        if (ParseAndCheckNC() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        r = xStorageShape.GetShapeSize() / a1 / a0;
    } else {
        OP_LOGE_FOR_INVALID_FORMAT(context_->GetNodeName(), "x", ToString(format).c_str(), "NCHW, NCDHW or ND");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus INTrainingReduceV2RegbaseTilingBase::CheckDtypeValid()
{
    // 校验 x 数据类型 ∈ {fp16, fp32}
    OP_CHECK_IF(std::find(DTYPE_LIST.begin(), DTYPE_LIST.end(), dataType) == DTYPE_LIST.end(),
                OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "x", ToString(dataType).c_str(), "FLOAT or FLOAT16"),
                return ge::GRAPH_FAILED);

    // 校验输出 sum / square_sum 数据类型恒 fp32
    auto sumDesc = context_->GetOutputDesc(OUTPUT_SUM_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, sumDesc);
    OP_CHECK_IF(
        sumDesc->GetDataType() != ge::DT_FLOAT,
        OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "sum", ToString(sumDesc->GetDataType()).c_str(), "FLOAT"),
        return ge::GRAPH_FAILED);

    auto squareSumDesc = context_->GetOutputDesc(OUTPUT_SQUARE_SUM_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, squareSumDesc);
    OP_CHECK_IF(squareSumDesc->GetDataType() != ge::DT_FLOAT,
                OP_LOGE_FOR_INVALID_DTYPE(context_->GetNodeName(), "square_sum",
                                          ToString(squareSumDesc->GetDataType()).c_str(), "FLOAT"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus INTrainingReduceV2RegbaseTilingBase::CheckShapeAllNotNegative(gert::Shape& shape)
{
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        OP_CHECK_IF(shape.GetDim(i) < 0,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "x", ToString(shape).c_str(),
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

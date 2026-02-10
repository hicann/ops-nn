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
 * \file max_pool_grad_with_argmax_v3_tiling_base.cpp
 * \brief
 */

#include <cstdint>
#include "tiling_base/tiling_templates_registry.h"
#include "log/log.h"
#include "error_util.h"
#include "platform/platform_info.h"
#include "../../../max_pool_with_argmax_v3/op_host/arch35/max_pool_with_argmax_v3_tiling.h"
#include "max_pool_grad_with_argmax_v3_tiling_base.h"

using namespace AscendC;
using namespace ge;

namespace optiling {

static constexpr int64_t DIMS_FOUR = 4;
static constexpr int64_t ATTR_INDEX_KSIZE = 0;
static constexpr int64_t ATTR_INDEX_STRIDES = 1;
static constexpr int64_t ATTR_INDEX_PADS = 2;
static constexpr int64_t ATTR_INDEX_DTYPE = 3;
static constexpr int64_t ATTR_INDEX_DILATION = 4;
static constexpr int64_t ATTR_INDEX_CEIL_MODE = 5;
static constexpr int64_t ATTR_INDEX_FORMAT = 6;
static constexpr int64_t DIM_ZERO = 0;
static constexpr int64_t DIM_ONE = 1;
static constexpr int64_t DIM_TWO = 2;
static constexpr int64_t DIM_THREE = 3;
static constexpr int64_t DTYPE_INT32 = 3;
static constexpr int64_t DTYPE_INT64 = 9;
static constexpr int64_t INPUT_X = 0;
static constexpr int64_t INPUT_GRAD = 1;
static constexpr int64_t INPUT_ARGMAX = 2;

static inline int64_t DivRtn(int64_t x, int64_t y)
{
    if (y == 0) {
        return 0;
    }
    int64_t q = x / y;
    int64_t r = x % y;
    if ((r != 0) && ((r < 0) != (y < 0))) {
        --q;
    }
    return q;
}

static bool CheckGradShape(const MaxPoolGradWithArgmaxInputInfoCommon& inputData)
{
    int64_t tmpH = inputData.hX + 2 * inputData.hPad - inputData.hDilation * (inputData.hKernel - 1) - 1;
    if (inputData.ceilMode) {
        tmpH += (inputData.hStride - 1);
    }
    int64_t tmpHGrad = DivRtn(tmpH, inputData.hStride) + 1;

    int64_t tmpW = inputData.wX + 2 * inputData.wPad - inputData.wDilation * (inputData.wKernel - 1) - 1;
    if (inputData.ceilMode) {
        tmpW += (inputData.wStride - 1);
    }
    int64_t tmpWGrad = DivRtn(tmpW, inputData.wStride) + 1;

    if (inputData.ceilMode) {
        if ((tmpHGrad - 1) * inputData.hStride >= inputData.hX + inputData.hPad) {
            tmpHGrad = tmpHGrad - 1;
        }
        if ((tmpWGrad - 1) * inputData.wStride >= inputData.wX + inputData.wPad) {
            tmpWGrad = tmpWGrad - 1;
        }
    }

    if (tmpHGrad != inputData.hGrad || tmpWGrad != inputData.wGrad || inputData.nX != inputData.nGrad ||
        inputData.cX != inputData.cGrad) {
        OP_LOGE(
            "MaxPoolGradWithArgmaxV3", "grad shape expected nchw [%ld, %ld,%ld, %ld], but got [%ld, %ld,%ld, %ld]",
            inputData.nX, inputData.cX, tmpHGrad, tmpWGrad, inputData.nGrad, inputData.cGrad, inputData.hGrad,
            inputData.wGrad);
        return false;
    }
    return true;
}

static inline bool IsGreaterThanInt32Max(const MaxPoolGradWithArgmaxInputInfoCommon& inputData)
{
    if (inputData.indexDtype == ge::DataType::DT_INT32) {
        return false;
    }

    int64_t planeSize = inputData.hX * inputData.wX;
    return planeSize > static_cast<int64_t>(INT32_MAX);
}

ge::graphStatus MaxPoolGradWithArgmaxV3BaseTiling::GetShapeAttrsInfo()
{
    auto inputX = context_->GetInputShape(INPUT_X);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputX);
    auto xShape = Ops::Base::EnsureNotScalar(inputX->GetStorageShape());

    OP_TILING_CHECK(
        xShape.GetDimNum() != DIMS_FOUR,
        VECTOR_INNER_ERR_REPORT_TILIING(
            context_->GetNodeName(), "MaxPoolGradWithArgmaxV3: input shape dim = %zu, should be equal 4",
            xShape.GetDimNum()),
        return ge::GRAPH_FAILED);

    OP_TILING_CHECK(
        xShape.GetShapeSize() <= 0,
        VECTOR_INNER_ERR_REPORT_TILIING(
            context_->GetNodeName(), "MaxPoolGradWithArgmaxV3: input shape size %ld less than zero failed",
            xShape.GetShapeSize()),
        return ge::GRAPH_FAILED);

    auto inputDesc = context_->GetInputDesc(INPUT_X);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputDesc);
    inputData.inputDtype = inputDesc->GetDataType();
    if (inputData.inputDtype != ge::DataType::DT_BF16 && inputData.inputDtype != ge::DataType::DT_FLOAT16 &&
        inputData.inputDtype != ge::DataType::DT_FLOAT) {
        VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "MaxPoolGradWithArgmaxV3: invalid dtype");
        return ge::GRAPH_FAILED;
    }

    auto inputGrad = context_->GetInputShape(INPUT_GRAD);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputGrad);
    auto gradShape = Ops::Base::EnsureNotScalar(inputGrad->GetStorageShape());

    OP_TILING_CHECK(
        gradShape.GetShapeSize() <= 0,
        VECTOR_INNER_ERR_REPORT_TILIING(
            context_->GetNodeName(), "MaxPoolGradWithArgmaxV3: grad shape size %ld less than zero failed",
            gradShape.GetShapeSize()),
        return ge::GRAPH_FAILED);
    inputData.gradShapeSize = gradShape.GetShapeSize();
    auto inputArgmax = context_->GetInputShape(INPUT_ARGMAX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputArgmax);
    auto argmaxShape = Ops::Base::EnsureNotScalar(inputArgmax->GetStorageShape());
    OP_TILING_CHECK(
        argmaxShape.GetShapeSize() <= 0,
        VECTOR_INNER_ERR_REPORT_TILIING(
            context_->GetNodeName(), "MaxPoolGradWithArgmaxV3: argmax shape size %ld less than zero failed",
            argmaxShape.GetShapeSize()),
        return ge::GRAPH_FAILED);

    auto inputArgmaxDesc = context_->GetInputDesc(INPUT_ARGMAX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputArgmaxDesc);
    auto argmaxDtype = inputArgmaxDesc->GetDataType();
    if (argmaxDtype != ge::DataType::DT_INT32 && argmaxDtype != ge::DataType::DT_INT64) {
        VECTOR_INNER_ERR_REPORT_TILIING(
            context_->GetNodeName(), "MaxPoolGradWithArgmaxV3: argmax dtype only support int32, int64, but got [%s].",
            Ops::Base::ToString(argmaxDtype).c_str());
        return ge::GRAPH_FAILED;
    }

    OP_TILING_CHECK(
        gradShape != argmaxShape,
        VECTOR_INNER_ERR_REPORT_TILIING(
            context_->GetNodeName(), "MaxPoolGradWithArgmaxV3: argmax shape is not same as grad shape"),
        return ge::GRAPH_FAILED);

    auto outY = context_->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outY);
    auto yShape = Ops::Base::EnsureNotScalar(outY->GetStorageShape());
    OP_TILING_CHECK(
        yShape != xShape,
        VECTOR_INNER_ERR_REPORT_TILIING(
            context_->GetNodeName(), "MaxPoolGradWithArgmaxV3: output shape is not same as input shape"),
        return ge::GRAPH_FAILED);

    auto runtimeAttrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, runtimeAttrs);

    const char* inputFormatPtr = runtimeAttrs->GetAttrPointer<char>(ATTR_INDEX_FORMAT);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputFormatPtr);
    if (strncmp(inputFormatPtr, "NCHW", sizeof("NCHW") / sizeof(char)) == 0) {
        inputData.inputFormat = ge::Format::FORMAT_NCHW;
        inputData.nX = xShape.GetDim(DIM_ZERO);
        inputData.cX = xShape.GetDim(DIM_ONE);
        inputData.hX = xShape.GetDim(DIM_TWO);
        inputData.wX = xShape.GetDim(DIM_THREE);
        inputData.nGrad = gradShape.GetDim(DIM_ZERO);
        inputData.cGrad = gradShape.GetDim(DIM_ONE);
        inputData.hGrad = gradShape.GetDim(DIM_TWO);
        inputData.wGrad = gradShape.GetDim(DIM_THREE);
    } else if (strncmp(inputFormatPtr, "NHWC", sizeof("NHWC") / sizeof(char)) == 0) {
        inputData.inputFormat = ge::Format::FORMAT_NHWC;
        inputData.nX = xShape.GetDim(DIM_ZERO);
        inputData.cX = xShape.GetDim(DIM_THREE);
        inputData.hX = xShape.GetDim(DIM_ONE);
        inputData.wX = xShape.GetDim(DIM_TWO);
        inputData.nGrad = gradShape.GetDim(DIM_ZERO);
        inputData.cGrad = gradShape.GetDim(DIM_THREE);
        inputData.hGrad = gradShape.GetDim(DIM_ONE);
        inputData.wGrad = gradShape.GetDim(DIM_TWO);
    } else {
        VECTOR_INNER_ERR_REPORT_TILIING(
            context_->GetNodeName(), "MaxPoolGradWithArgmaxV3: input format [%s] is invalid", inputFormatPtr);
        return ge::GRAPH_FAILED;
    }

    const gert::TypedContinuousVector<int64_t>* kernelSizePtr = runtimeAttrs->GetListInt(ATTR_INDEX_KSIZE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, kernelSizePtr);
    inputData.hKernel = *(kernelSizePtr->GetData());
    inputData.wKernel = *(kernelSizePtr->GetData() + 1);
    OP_TILING_CHECK(
        inputData.hKernel <= 0 || inputData.wKernel <= 0,
        VECTOR_INNER_ERR_REPORT_TILIING(
            context_->GetNodeName(), "MaxPoolGradWithArgmaxV3: kernel shape [%ld, %ld] is invalid", inputData.hKernel,
            inputData.wKernel),
        return ge::GRAPH_FAILED);

    const gert::TypedContinuousVector<int64_t>* stridePtr = runtimeAttrs->GetListInt(ATTR_INDEX_STRIDES);
    OP_CHECK_NULL_WITH_CONTEXT(context_, stridePtr);
    inputData.hStride = *(stridePtr->GetData());
    inputData.wStride = *(stridePtr->GetData() + 1);
    OP_TILING_CHECK(
        inputData.hStride <= 0 || inputData.wStride <= 0,
        VECTOR_INNER_ERR_REPORT_TILIING(
            context_->GetNodeName(), "MaxPoolGradWithArgmaxV3: stride shape [%ld, %ld] is invalid", inputData.hStride,
            inputData.wStride),
        return ge::GRAPH_FAILED);

    const gert::TypedContinuousVector<int64_t>* paddingPtr = runtimeAttrs->GetListInt(ATTR_INDEX_PADS);
    OP_CHECK_NULL_WITH_CONTEXT(context_, paddingPtr);
    inputData.hPad = *(paddingPtr->GetData());
    inputData.wPad = *(paddingPtr->GetData() + 1);
    OP_TILING_CHECK(
        inputData.hPad > inputData.hKernel / 2 || inputData.wPad > inputData.wKernel / 2,
        VECTOR_INNER_ERR_REPORT_TILIING(
            context_->GetNodeName(), "MaxPoolGradWithArgmaxV3: pad shape [%ld, %ld] is invalid", inputData.hPad,
            inputData.wPad),
        return ge::GRAPH_FAILED);

    const gert::TypedContinuousVector<int64_t>* dilationPtr = runtimeAttrs->GetListInt(ATTR_INDEX_DILATION);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dilationPtr);

    inputData.hDilation = *(dilationPtr->GetData());
    inputData.wDilation = *(dilationPtr->GetData() + 1);
    OP_TILING_CHECK(
        inputData.hDilation <= 0 || inputData.wDilation <= 0,
        VECTOR_INNER_ERR_REPORT_TILIING(
            context_->GetNodeName(), "MaxPoolGradWithArgmaxV3: dilation shape [%ld, %ld] is invalid",
            inputData.hDilation, inputData.wDilation),
        return ge::GRAPH_FAILED);

    const bool* ceilModePtr = runtimeAttrs->GetAttrPointer<bool>(ATTR_INDEX_CEIL_MODE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, ceilModePtr);
    inputData.ceilMode = *ceilModePtr;

    OP_TILING_CHECK(
        !CheckGradShape(inputData),
        VECTOR_INNER_ERR_REPORT_TILIING(context_->GetNodeName(), "MaxPoolGradWithArgmaxV3: grad shape is invalid"),
        return ge::GRAPH_FAILED);

    const int* indexDtypePtr = runtimeAttrs->GetAttrPointer<int>(ATTR_INDEX_DTYPE);
    OP_CHECK_NULL_WITH_CONTEXT(context_, indexDtypePtr);
    switch (*indexDtypePtr) {
        case DTYPE_INT32:
            inputData.indexDtype = ge::DataType::DT_INT32;
            break;
        case DTYPE_INT64:
            inputData.indexDtype = ge::DataType::DT_INT64;
            break;
        default:
            inputData.indexDtype = ge::DataType::DT_INT32;
            break;
    }

    if (IsGreaterThanInt32Max(inputData)) {
        inputData.isInt32Meet = 0;
    } else {
        inputData.isInt32Meet = 1;
    }

    PrintInputData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPoolGradWithArgmaxV3BaseTiling::PostTiling()
{
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
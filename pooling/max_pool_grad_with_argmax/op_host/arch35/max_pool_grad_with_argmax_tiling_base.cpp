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
 * \file max_pool_grad_with_argmax_tiling_base.cpp
 * \brief
 */

#include "max_pool_grad_with_argmax_tiling.h"
#include "graph/utils/type_utils.h"
#include "exe_graph/runtime/infer_shape_context.h"

#include "platform/platform_info.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include "op_common/op_host/util/platform_util.h"

using namespace AscendC;
using namespace ge;

namespace optiling {
static constexpr int64_t DIMS_FOUR = 4;
static constexpr int64_t ATTR_INDEX_KSIZE = 0;
static constexpr int64_t ATTR_INDEX_STRIDES = 1;
static constexpr int64_t ATTR_INDEX_PADDING = 2;
static constexpr int64_t ATTR_INDEX_INCLUDE_BATCH_IN_INDEX = 3;
static constexpr int64_t ATTR_INDEX_FORMAT = 4;
static constexpr int64_t DIM_ZERO = 0;
static constexpr int64_t DIM_ONE = 1;
static constexpr int64_t DIM_TWO = 2;
static constexpr int64_t DIM_THREE = 3;
static constexpr int64_t DTYPE_INT32 = 3;
static constexpr int64_t DTYPE_INT64 = 9;
static constexpr int64_t INPUT_X = 0;
static constexpr int64_t INPUT_GRAD = 1;
static constexpr int64_t INPUT_ARGMAX = 2;

ge::graphStatus MaxPoolGradWithArgmaxBaseTiling::GetShapeAttrsInfo()
{
    const char* opName_ = "MaxPoolGradWithArgmax";
    OP_LOGD(opName_, "MaxPoolGradWithArgmaxBaseTiling::GetShapeAttrsInfo()");
    auto inputX = context_->GetInputShape(INPUT_X);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, inputX);
    auto xShape = Ops::Base::EnsureNotScalar(inputX->GetStorageShape());
    if (xShape.GetDimNum() != DIMS_FOUR) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(opName_, "x", std::to_string(xShape.GetDimNum()).c_str(), "4");
        return ge::GRAPH_FAILED;
    }

    if (xShape.GetShapeSize() <= 0) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(opName_, "x", std::to_string(xShape.GetShapeSize()).c_str(),
                                                  "input shape size must be greater than 0");
        return ge::GRAPH_FAILED;
    }

    auto inputDesc = context_->GetInputDesc(INPUT_X);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, inputDesc);
    inputData.inputDtype = inputDesc->GetDataType();
    if (inputData.inputDtype != ge::DataType::DT_BF16 && inputData.inputDtype != ge::DataType::DT_FLOAT16 &&
        inputData.inputDtype != ge::DataType::DT_FLOAT) {
        OP_LOGE_FOR_INVALID_DTYPE(opName_, "x", std::to_string(static_cast<int32_t>(inputData.inputDtype)),
                                  "[DT_FLOAT16, DT_FLOAT, DT_BF16]");
        return ge::GRAPH_FAILED;
    }

    auto inputGrad = context_->GetInputShape(INPUT_GRAD);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, inputGrad);
    auto gradShape = Ops::Base::EnsureNotScalar(inputGrad->GetStorageShape());
    if (gradShape.GetShapeSize() <= 0) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(opName_, "grad", std::to_string(gradShape.GetShapeSize()).c_str(),
                                                  "grad shape size must be greater than 0");
        return ge::GRAPH_FAILED;
    }

    auto inputArgmax = context_->GetInputShape(INPUT_ARGMAX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, inputArgmax);
    auto argmaxShape = Ops::Base::EnsureNotScalar(inputArgmax->GetStorageShape());
    if (argmaxShape.GetShapeSize() <= 0) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(opName_, "argmax", std::to_string(argmaxShape.GetShapeSize()).c_str(),
                                                  "argmax shape size must be greater than 0");
        return ge::GRAPH_FAILED;
    }

    auto inputArgmaxDesc = context_->GetInputDesc(INPUT_ARGMAX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputArgmaxDesc);
    auto argmaxDtype = inputArgmaxDesc->GetDataType();
    if (argmaxDtype != ge::DataType::DT_INT32 && argmaxDtype != ge::DataType::DT_INT64) {
        OP_LOGE_FOR_INVALID_DTYPE(opName_, "argmax", std::to_string(static_cast<int32_t>(argmaxDtype)),
                                  "[DT_INT32, DT_INT64]");
        return ge::GRAPH_FAILED;
    }
    inputData.indexDtype = argmaxDtype;

    if (gradShape != argmaxShape) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(opName_, "argmax, grad", "argmax_shape, grad_shape",
                                               "argmax shape must equal grad shape");
        return ge::GRAPH_FAILED;
    }

    auto outY = context_->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outY);
    auto yShape = Ops::Base::EnsureNotScalar(outY->GetStorageShape());
    if (yShape != xShape) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(opName_, "y, x", "y_shape, x_shape",
                                               "output shape must equal input shape");
        return ge::GRAPH_FAILED;
    }

    auto runtimeAttrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, runtimeAttrs);

    const char* inputFormatPtr = runtimeAttrs->GetAttrPointer<char>(ATTR_INDEX_FORMAT);
    if (inputFormatPtr && strncmp(inputFormatPtr, "NCHW", sizeof("NCHW") / sizeof(char)) == 0) {
        inputData.inputFormat = ge::Format::FORMAT_NCHW;
        inputData.nX = xShape.GetDim(DIM_ZERO);
        inputData.cX = xShape.GetDim(DIM_ONE);
        inputData.hX = xShape.GetDim(DIM_TWO);
        inputData.wX = xShape.GetDim(DIM_THREE);
        inputData.nGrad = gradShape.GetDim(DIM_ZERO);
        inputData.cGrad = gradShape.GetDim(DIM_ONE);
        inputData.hGrad = gradShape.GetDim(DIM_TWO);
        inputData.wGrad = gradShape.GetDim(DIM_THREE);
    } else if (!inputFormatPtr || strncmp(inputFormatPtr, "NHWC", sizeof("NHWC") / sizeof(char)) == 0) {
        inputData.inputFormat = ge::Format::FORMAT_NHWC;
        inputData.nX = xShape.GetDim(DIM_ZERO);
        inputData.hX = xShape.GetDim(DIM_ONE);
        inputData.wX = xShape.GetDim(DIM_TWO);
        inputData.cX = xShape.GetDim(DIM_THREE);
        inputData.nGrad = gradShape.GetDim(DIM_ZERO);
        inputData.hGrad = gradShape.GetDim(DIM_ONE);
        inputData.wGrad = gradShape.GetDim(DIM_TWO);
        inputData.cGrad = gradShape.GetDim(DIM_THREE);
    } else {
        OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(opName_, "x", inputFormatPtr, "dataFormat must be NCHW or NHWC");
        return ge::GRAPH_FAILED;
    }
    if (inputData.inputFormat == ge::Format::FORMAT_NHWC) {
        const gert::TypedContinuousVector<int64_t>* kernelSizePtr = runtimeAttrs->GetListInt(ATTR_INDEX_KSIZE);
        OPS_CHECK_NULL_WITH_CONTEXT(context_, kernelSizePtr);
        inputData.hKernel = kernelSizePtr->GetData()[DIM_ONE];
        inputData.wKernel = kernelSizePtr->GetData()[DIM_TWO];
        if (kernelSizePtr->GetData()[DIM_ZERO] != 1 || kernelSizePtr->GetData()[DIM_THREE] != 1 ||
            inputData.hKernel <= 0 || inputData.wKernel <= 0) {
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                opName_, "ksize[NHWC]",
                (std::string("[") + std::to_string(kernelSizePtr->GetData()[DIM_ZERO]) + "," +
                 std::to_string(kernelSizePtr->GetData()[DIM_ONE]) + "," +
                 std::to_string(kernelSizePtr->GetData()[DIM_TWO]) + "," +
                 std::to_string(kernelSizePtr->GetData()[DIM_THREE]) + "]")
                    .c_str(),
                "NHWC ksize[0] and ksize[3] must be 1, ksize[H] and ksize[W] must be > 0");
            return ge::GRAPH_FAILED;
        }

        const gert::TypedContinuousVector<int64_t>* stridePtr = runtimeAttrs->GetListInt(ATTR_INDEX_STRIDES);
        OPS_CHECK_NULL_WITH_CONTEXT(context_, stridePtr);
        inputData.hStride = stridePtr->GetData()[DIM_ONE];
        inputData.wStride = stridePtr->GetData()[DIM_TWO];
        if (stridePtr->GetData()[DIM_ZERO] != 1 || stridePtr->GetData()[DIM_THREE] != 1 || inputData.hStride <= 0 ||
            inputData.wStride <= 0) {
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                opName_, "strides[NHWC]",
                (std::string("[") + std::to_string(stridePtr->GetData()[DIM_ZERO]) + "," +
                 std::to_string(stridePtr->GetData()[DIM_ONE]) + "," + std::to_string(stridePtr->GetData()[DIM_TWO]) +
                 "," + std::to_string(stridePtr->GetData()[DIM_THREE]) + "]")
                    .c_str(),
                "NHWC strides[0] and strides[3] must be 1, strides[H] and strides[W] must be > 0");
            return ge::GRAPH_FAILED;
        }
    } else if (inputData.inputFormat == ge::Format::FORMAT_NCHW) {
        const gert::TypedContinuousVector<int64_t>* kernelSizePtr = runtimeAttrs->GetListInt(ATTR_INDEX_KSIZE);
        OPS_CHECK_NULL_WITH_CONTEXT(context_, kernelSizePtr);
        inputData.hKernel = kernelSizePtr->GetData()[DIM_TWO];
        inputData.wKernel = kernelSizePtr->GetData()[DIM_THREE];
        if (kernelSizePtr->GetData()[DIM_ZERO] != 1 || kernelSizePtr->GetData()[DIM_ONE] != 1 ||
            inputData.hKernel <= 0 || inputData.wKernel <= 0) {
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                opName_, "ksize[NCHW]",
                (std::string("[") + std::to_string(kernelSizePtr->GetData()[DIM_ZERO]) + "," +
                 std::to_string(kernelSizePtr->GetData()[DIM_ONE]) + "," +
                 std::to_string(kernelSizePtr->GetData()[DIM_TWO]) + "," +
                 std::to_string(kernelSizePtr->GetData()[DIM_THREE]) + "]")
                    .c_str(),
                "NCHW ksize[0] and ksize[1] must be 1, ksize[H] and ksize[W] must be > 0");
            return ge::GRAPH_FAILED;
        }
        const gert::TypedContinuousVector<int64_t>* stridePtr = runtimeAttrs->GetListInt(ATTR_INDEX_STRIDES);
        OPS_CHECK_NULL_WITH_CONTEXT(context_, stridePtr);
        inputData.hStride = stridePtr->GetData()[DIM_TWO];
        inputData.wStride = stridePtr->GetData()[DIM_THREE];
        if (stridePtr->GetData()[DIM_ZERO] != 1 || stridePtr->GetData()[DIM_ONE] != 1 || inputData.hStride <= 0 ||
            inputData.wStride <= 0) {
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                opName_, "strides[NCHW]",
                (std::string("[") + std::to_string(stridePtr->GetData()[DIM_ZERO]) + "," +
                 std::to_string(stridePtr->GetData()[DIM_ONE]) + "," + std::to_string(stridePtr->GetData()[DIM_TWO]) +
                 "," + std::to_string(stridePtr->GetData()[DIM_THREE]) + "]")
                    .c_str(),
                "NCHW strides[0] and strides[1] must be 1, strides[H] and strides[W] must be > 0");
            return ge::GRAPH_FAILED;
        }
    } else {
        OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(opName_, "x", "unknown", "dataFormat must be NCHW or NHWC");
        return ge::GRAPH_FAILED;
    }

    const char* padMode = runtimeAttrs->GetAttrPointer<char>(ATTR_INDEX_PADDING);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, padMode);
    std::string padModeStr = padMode;
    if (IsInvalidPaddingMode(padModeStr)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "padding", padModeStr.c_str(), "padding must be VALID or SAME");
        return ge::GRAPH_FAILED;
    }
    if (padModeStr == "VALID") {
        inputData.hPad = inputData.wPad = 0;
    } else if (padModeStr == "SAME") {
        int64_t hPadNeed = std::max(int64_t{0},
                                    (inputData.hGrad - 1) * inputData.hStride + inputData.hKernel - inputData.hX);
        inputData.hPad = hPadNeed / DIGIT_TWO;

        int64_t wPadNeed = std::max(int64_t{0},
                                    (inputData.wGrad - 1) * inputData.wStride + inputData.wKernel - inputData.wX);
        inputData.wPad = wPadNeed / DIGIT_TWO;
    }

    if (inputData.hPad > inputData.hKernel / DIGIT_TWO || inputData.wPad > inputData.wKernel / DIGIT_TWO) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            opName_, "hPad, wPad",
            (std::string("hPad=") + std::to_string(inputData.hPad) + ", wPad=" + std::to_string(inputData.wPad))
                .c_str(),
            "hPad must <= hKernel/2, wPad must <= wKernel/2");
        return ge::GRAPH_FAILED;
    }

    if (!CheckGradShape(inputData, padModeStr)) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(opName_, "grad", "grad_shape",
                                               "grad shape is invalid for the given padMode and kernel/stride config");
        return ge::GRAPH_FAILED;
    }

    if (IsGreaterThanInt32MaxNHWC(inputData)) {
        inputData.isInt32Meet = 0;
    } else {
        inputData.isInt32Meet = 1;
    }

    PrintInputData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPoolGradWithArgmaxBaseTiling::PostTiling()
{
    OP_LOGD("MaxPoolGradWithArgmax", "MaxPoolGradWithArgmaxBaseTiling::PostTiling");
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
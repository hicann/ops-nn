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
 * \file max_pool_v3_tiling_base.cpp
 * \brief
 */

#include "exe_graph/runtime/shape.h"
#include "op_host/tiling_util.h"
#include "max_pool_v3_tiling.h"

using namespace ge;

namespace optiling {
const int32_t INPUT_IDX_X = 0;
const int32_t NCHW_DIMS = 4;
const int32_t KERNEL_POS = 0;
const int32_t STRIDE_POS = 1;
const int32_t PADDING_MODE_POS = 2;
const int32_t PADDING_POS = 3;
const int32_t FORMAT_POS = 4;
const int32_t GLOBAL_POOLING_POS = 5;
const int32_t CEIL_POS = 6;
const uint32_t WS_SYS_SIZE = 16U * 1024U * 1024U;
static const int32_t MP_MAX_2D_DIM_ZERO = 0;
static const int32_t MP_MAX_2D_DIM_ONE = 1;
static const int32_t MP_MAX_2D_DIM_TWO = 2;
static const int32_t MP_MAX_2D_DIM_THREE = 3;
static const int32_t DIGIT_TWO = 2;

// Integer division rounding to -Infinity
static inline int64_t DivRtn(const int64_t x, const int64_t y)
{
    if (y == 0) {
        return 0;
    }
    int64_t q = x / y;
    int64_t r = x % y;
    if ((r != 0) && ((r < 0) != (y < 0))) {
        --q;
    };
    return q;
}

ge::graphStatus MaxPoolV3BaseTiling::GetPlatformInfo()
{
    auto platformPtr = context_->GetPlatformInfo();
    if (platformPtr == nullptr) {
        auto compileInfoPtr = reinterpret_cast<const MaxPoolV3CompileInfo*>(context_->GetCompileInfo());
        if (compileInfoPtr == nullptr) {
            OP_LOGE(context_, "compile info is null");
            return ge::GRAPH_FAILED;
        }
        coreNum = compileInfoPtr->coreNum;
        ubSize = compileInfoPtr->ubSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformPtr);
        coreNum = ascendcPlatform.GetCoreNum();

        uint64_t ubSizePlatform;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatform);
        ubSize = static_cast<uint64_t>(ubSizePlatform);
    }
    if (coreNum == 0) {
        OP_LOGE(context_, "coreNum is 0");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

bool MaxPoolV3BaseTiling::IsInvalidType(const DataType dataType)
{
    const std::set<ge::DataType> supportedDtype = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16,  ge::DT_INT64,
                                                   ge::DT_INT32, ge::DT_UINT32,  ge::DT_INT16, ge::DT_UINT16,
                                                   ge::DT_INT8,  ge::DT_UINT8};
    bool dtypeInValid = (supportedDtype.count(dataType) == 0);
    return dtypeInValid;
}

bool MaxPoolV3BaseTiling::IsInvalidPaddingMode(std::string padMode)
{
    const std::set<std::string> supportedPadModelIST = {"CALCULATED", "SAME", "VALID"};
    bool padModeInValid = (supportedPadModelIST.count(padMode) == 0);
    return padModeInValid;
}

ge::graphStatus MaxPoolV3BaseTiling::CheckShape(
    gert::Shape& inputShape, gert::Shape& outputShape, const ge::Format& inputFormat)
{
    const char* opName_ = "MaxPoolV3";
    if (inputShape.GetDimNum() != NCHW_DIMS) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(opName_, "x", std::to_string(inputShape.GetDimNum()).c_str(), "4");
        return ge::GRAPH_FAILED;
    }
    if (outputShape.GetDimNum() != NCHW_DIMS) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(opName_, "y", std::to_string(outputShape.GetDimNum()).c_str(), "4");
        return ge::GRAPH_FAILED;
    }
    if (inputShape.GetShapeSize() == 0 && outputShape.GetShapeSize() == 0) {
        return ge::GRAPH_SUCCESS;
    }

    if (inputShape.GetShapeSize() <= 0) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(opName_, "x",
            std::to_string(inputShape.GetShapeSize()).c_str(), "shape size must be greater than 0");
        return ge::GRAPH_FAILED;
    }

    if (outputShape.GetShapeSize() <= 0) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(opName_, "y",
            std::to_string(outputShape.GetShapeSize()).c_str(), "shape size must be greater than 0");
        return ge::GRAPH_FAILED;
    }

    int32_t nDim = MP_MAX_2D_DIM_ZERO;
    int32_t cDim = MP_MAX_2D_DIM_ONE;
    if (inputFormat == ge::Format::FORMAT_NHWC) {
        nDim = MP_MAX_2D_DIM_ZERO;
        cDim = MP_MAX_2D_DIM_THREE;
    }
    if (inputShape.GetDim(nDim) != outputShape.GetDim(nDim)) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(opName_, "x",
            std::to_string(inputShape.GetDim(nDim)).c_str(),
            ("dim-n must be equal between x and y, expected " + std::to_string(outputShape.GetDim(nDim))).c_str());
        return ge::GRAPH_FAILED;
    }
    if (inputShape.GetDim(cDim) != outputShape.GetDim(cDim)) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(opName_, "x",
            std::to_string(inputShape.GetDim(cDim)).c_str(),
            ("dim-c must be equal between x and y, expected " + std::to_string(outputShape.GetDim(cDim))).c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPoolV3BaseTiling::GetShapeAndDtype()
{
    const char* opName_ = "MaxPoolV3";
    auto inputX = context_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputX);
    auto inputShape = Ops::NN::OpTiling::EnsureNotScalar(inputX->GetStorageShape());
    auto outX = context_->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outX);
    auto outShape = Ops::NN::OpTiling::EnsureNotScalar(outX->GetStorageShape());
    auto inputDesc = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputDesc);
    dtype = inputDesc->GetDataType();
    if (IsInvalidType(dtype)) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(opName_, "x",
            std::to_string(static_cast<int32_t>(dtype)).c_str(),
            "dtype not in supported list [0(DT_FLOAT), 1(DT_FLOAT16), 27(DT_BF16), 9(DT_INT64), 3(DT_INT32), 19(DT_UINT32), 5(DT_INT16), 18(DT_UINT16), 2(DT_INT8), 17(DT_UINT8)]");
        return ge::GRAPH_FAILED;
    }
    dtypeSize = ge::GetSizeByDataType(dtype);
    if (dtypeSize <= 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "x dtype size",
            std::to_string(dtypeSize).c_str(), "dtypeSize must be greater than 0");
        return ge::GRAPH_FAILED;
    }

    auto runtimeAttrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, runtimeAttrs);

    if (CheckShape(inputShape, outShape, inputData.inputFormat) != ge::GRAPH_SUCCESS) {
        OP_LOGE(opName_, "MaxPoolV3: check shape failed");
        return ge::GRAPH_FAILED;
    }
    nDim_ = MP_MAX_2D_DIM_ZERO;
    cDim_ = MP_MAX_2D_DIM_ONE;
    hDim_ = MP_MAX_2D_DIM_TWO;
    wDim_ = MP_MAX_2D_DIM_THREE;
    if (inputData.inputFormat == ge::Format::FORMAT_NCHW) {
        inputData.batches = inputShape.GetDim(MP_MAX_2D_DIM_ZERO) * inputShape.GetDim(MP_MAX_2D_DIM_ONE);
        inputData.channels = 1;
    } else if (inputData.inputFormat == ge::Format::FORMAT_NHWC) {
        nDim_ = MP_MAX_2D_DIM_ZERO;
        hDim_ = MP_MAX_2D_DIM_ONE;
        wDim_ = MP_MAX_2D_DIM_TWO;
        cDim_ = MP_MAX_2D_DIM_THREE;
        inputData.batches = inputShape.GetDim(MP_MAX_2D_DIM_ZERO);
        inputData.channels = inputShape.GetDim(MP_MAX_2D_DIM_THREE);
    } else {
        OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(opName_, "x",
            std::to_string(static_cast<int32_t>(inputData.inputFormat)).c_str(),
            "only support NCHW and NHWC");
        return ge::GRAPH_FAILED;
    }
    inputData.inputShape = {inputShape.GetDim(hDim_), inputShape.GetDim(wDim_)};
    inputData.outShape = {outShape.GetDim(hDim_), outShape.GetDim(wDim_)};
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPoolV3BaseTiling::GetKernelInfo()
{
    const char* opName_ = (std::string(context_->GetNodeType()) == "MaxPoolV2") ? "MaxPoolV2" : "MaxPoolV3";
    if (std::string(context_->GetNodeType()) == "MaxPoolV2") {
        auto kernelSize = context_->GetInputTensor(INPUT_KSIZE_IDX);
        OP_CHECK_NULL_WITH_CONTEXT(context_, kernelSize);
        if (kernelSize->GetShapeSize() != NCHW_DIMS) {
            OP_LOGE_FOR_INVALID_LISTSIZE(opName_, "Length of kernel_size",
                std::to_string(kernelSize->GetShapeSize()).c_str(), "4");
            return ge::GRAPH_FAILED;
        }
        int32_t hKernelSize = kernelSize->GetData<int32_t>()[hDim_];
        int32_t wKernelSize = kernelSize->GetData<int32_t>()[wDim_];
        inputData.kernelSize = {hKernelSize, wKernelSize};
        if (hKernelSize <= 0 || wKernelSize <= 0) {
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(opName_, "ksize[H], ksize[W]",
                (std::to_string(hKernelSize) + ", " + std::to_string(wKernelSize)).c_str(),
                "ksize of H and W dimensions must be greater than 0");
            return ge::GRAPH_FAILED;
        }
        auto stride = context_->GetInputTensor(INPUT_STRIDES_IDX);
        OP_CHECK_NULL_WITH_CONTEXT(context_, stride);
        if (stride->GetShapeSize() != NCHW_DIMS) {
            OP_LOGE_FOR_INVALID_LISTSIZE(opName_, "Length of strides",
                std::to_string(stride->GetShapeSize()).c_str(), "4");
            return ge::GRAPH_FAILED;
        }

        int32_t hStride = stride->GetData<int32_t>()[hDim_];
        int32_t wStride = stride->GetData<int32_t>()[wDim_];

        inputData.stride = {hStride, wStride};
        if (hStride <= 0 || wStride <= 0) {
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(opName_, "strides[H], strides[W]",
                (std::to_string(hStride) + ", " + std::to_string(wStride)).c_str(),
                "strides of H and W dimensions must be greater than 0");
            return ge::GRAPH_FAILED;
        }
    } else {
        auto runtimeAttrs = context_->GetAttrs();
        OP_CHECK_NULL_WITH_CONTEXT(context_, runtimeAttrs);
        auto kernelSize = runtimeAttrs->GetListInt(KERNEL_POS);
        OP_CHECK_NULL_WITH_CONTEXT(context_, kernelSize);
        if (kernelSize->GetSize() != NCHW_DIMS) {
            OP_LOGE_FOR_INVALID_LISTSIZE(opName_, "Length of kernel_size",
                std::to_string(kernelSize->GetSize()).c_str(), "4");
            return ge::GRAPH_FAILED;
        }

        int64_t hKernelSize = kernelSize->GetData()[hDim_];
        int64_t wKernelSize = kernelSize->GetData()[wDim_];
        inputData.kernelSize = {hKernelSize, wKernelSize};
        if (hKernelSize <= 0 || wKernelSize <= 0) {
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(opName_, "ksize[H], ksize[W]",
                (std::to_string(hKernelSize) + ", " + std::to_string(wKernelSize)).c_str(),
                "ksize of H and W dimensions must be greater than 0");
            return ge::GRAPH_FAILED;
        }
        auto stride = runtimeAttrs->GetListInt(STRIDE_POS);
        OP_CHECK_NULL_WITH_CONTEXT(context_, stride);
        if (stride->GetSize() != NCHW_DIMS) {
            OP_LOGE_FOR_INVALID_LISTSIZE(opName_, "Length of strides",
                std::to_string(stride->GetSize()).c_str(), "4");
            return ge::GRAPH_FAILED;
        }

        int64_t hStride = stride->GetData()[hDim_];
        int64_t wStride = stride->GetData()[wDim_];

        inputData.stride = {hStride, wStride};
        if (hStride <= 0 || wStride <= 0) {
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(opName_, "strides[H], strides[W]",
                (std::to_string(hStride) + ", " + std::to_string(wStride)).c_str(),
                "strides of H and W dimensions must be greater than 0");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPoolV3BaseTiling::GetPadInfo()
{
    const char* opName_ = "MaxPoolV3";
    auto runtimeAttrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, runtimeAttrs);

    if (padModeStr_ == "CALCULATED") {
        auto padding = runtimeAttrs->GetListInt(PADDING_POS);
        OP_CHECK_NULL_WITH_CONTEXT(context_, padding);
        if (padding->GetSize() != NCHW_DIMS) {
            OP_LOGE_FOR_INVALID_LISTSIZE(opName_, "Length of padding",
                std::to_string(padding->GetSize()).c_str(), "4");
            return ge::GRAPH_FAILED;
        }
        int64_t topPad = padding->GetData()[TOP_PAD_INDEX];
        int64_t bottomPad = padding->GetData()[BOTTOM_PAD_INDEX];
        int64_t leftPad = padding->GetData()[LEFT_PAD_INDEX];
        int64_t rightPad = padding->GetData()[RIGHT_PAD_INDEX];

        inputData.pad = {topPad, bottomPad, leftPad, rightPad};
        if (topPad >= inputData.kernelSize[H_DIM] || topPad < 0 || bottomPad >= inputData.kernelSize[H_DIM] || bottomPad < 0 ||
            leftPad >= inputData.kernelSize[W_DIM] || leftPad < 0 || rightPad >= inputData.kernelSize[W_DIM] || rightPad < 0) {
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(opName_, "top_pad, bottom_pad, left_pad, right_pad",
                (std::to_string(topPad) + ", " + std::to_string(bottomPad) + ", " +
                 std::to_string(leftPad) + ", " + std::to_string(rightPad)).c_str(),
                ("padding must be >= 0 and < corresponding kernel size, kernel H=" +
                 std::to_string(inputData.kernelSize[H_DIM]) + ", W=" +
                 std::to_string(inputData.kernelSize[W_DIM])).c_str());
            return ge::GRAPH_FAILED;
        }
    } else if (padModeStr_ == "VALID") {
        inputData.pad = {0, 0, 0, 0};
    } else if (padModeStr_ == "SAME") {
        int64_t hPadNeed = std::max(
            int64_t{0},
            (inputData.outShape[0] - 1) * inputData.stride[0] + inputData.kernelSize[0] - inputData.inputShape[0]);
        int64_t topPad = hPadNeed / DIGIT_TWO;
        int64_t bottomPad = hPadNeed - topPad;
        int64_t wPadNeed = std::max(
            int64_t{0},
            (inputData.outShape[1] - 1) * inputData.stride[1] + inputData.kernelSize[1] - inputData.inputShape[1]);
        int64_t leftPad = wPadNeed / DIGIT_TWO;
        int64_t rightPad = wPadNeed - leftPad;
        inputData.pad = {topPad, bottomPad, leftPad, rightPad};
    } else {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "padding_mode",
            padModeStr_.c_str(), "only support CALCULATED, SAME, VALID");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPoolV3BaseTiling::GetAttrsInfo()
{
    const char* opName_ = (std::string(context_->GetNodeType()) == "MaxPoolV2") ? "MaxPoolV2" : "MaxPoolV3";
    auto runtimeAttrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, runtimeAttrs);
    const char* padMode = nullptr;
    if (std::string(context_->GetNodeType()) == "MaxPoolV2") {
        padMode = runtimeAttrs->GetAttrPointer<char>(0);
    } else {
        padMode = runtimeAttrs->GetAttrPointer<char>(PADDING_MODE_POS);
    }
    if (padMode != nullptr) {
        padModeStr_ = padMode;
    } else {
        padModeStr_ = "CALCULATED";
    }
    if (IsInvalidPaddingMode(padModeStr_)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "padding_mode",
            padModeStr_.c_str(), "only support CALCULATED, SAME, VALID");
        return ge::GRAPH_FAILED;
    }
    inputData.ceilMode = false;
    inputData.globalPooling = false;
    if (std::string(context_->GetNodeType()) == "MaxPoolV3") {
        const bool* ceilModePtr = runtimeAttrs->GetAttrPointer<bool>(CEIL_POS);
        if (ceilModePtr != nullptr) {
            inputData.ceilMode = *ceilModePtr;
        }
        const bool* globalPoolingPtr = runtimeAttrs->GetAttrPointer<bool>(GLOBAL_POOLING_POS);
        if (globalPoolingPtr != nullptr) {
            inputData.globalPooling = *globalPoolingPtr;
        }
    }
    std::string inputFormatStr("NCHW");
    const char* inputFormat = nullptr;
    if (std::string(context_->GetNodeType()) == "MaxPoolV2") {
        inputFormat = runtimeAttrs->GetAttrPointer<char>(1);
    } else {
        inputFormat = runtimeAttrs->GetAttrPointer<char>(FORMAT_POS);
    }
    if (inputFormat != nullptr) {
        inputFormatStr = inputFormat;
    }
    if (inputFormatStr == "NCHW") {
        inputData.inputFormat = ge::Format::FORMAT_NCHW;
    } else if (inputFormatStr == "NHWC") {
        inputData.inputFormat = ge::Format::FORMAT_NHWC;
    } else {
        OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(opName_, "x",
            inputFormatStr.c_str(), "only support NCHW and NHWC");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPoolV3BaseTiling::CheckOutPutShapeForValid()
{
    const char* opName_ = "MaxPoolV3";
    int64_t expectedH = (inputData.inputShape[0] - inputData.kernelSize[0] + inputData.stride[0]) / inputData.stride[0];
    int64_t expectedW = (inputData.inputShape[1] - inputData.kernelSize[1] + inputData.stride[1]) / inputData.stride[1];
    if (inputData.outShape[0] != expectedH || inputData.outShape[1] != expectedW) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName_, "y",
            (std::to_string(inputData.outShape[0]) + ", " + std::to_string(inputData.outShape[1])).c_str(),
            ("when padmode is VALID, output shape must be (" + std::to_string(expectedH) + ", " + std::to_string(expectedW) + ")").c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPoolV3BaseTiling::CheckOutPutShapeForSame()
{
    return ge::GRAPH_SUCCESS;
}

int64_t MaxPoolV3BaseTiling::InferCalculateOutShape(
    const int64_t ksize, const int64_t padL, const int64_t padR, const int64_t stride, const bool ceilMode,
    int64_t dimSize)
{
    int64_t tmpTotalInput = dimSize + padL + padR - ksize;
    if (ceilMode) {
        tmpTotalInput += stride - 1;
    }
    int64_t outputSize = DivRtn(tmpTotalInput, stride) + 1;

    if (ceilMode) {
        if ((outputSize - 1) * stride >= dimSize + padL) {
            --outputSize;
        }
    }
    return outputSize;
}

ge::graphStatus MaxPoolV3BaseTiling::CheckOutPutShapeForCalculated()
{
    const char* opName_ = "MaxPoolV3";
    int64_t expectedH = InferCalculateOutShape(
        inputData.kernelSize[0], inputData.pad[TOP_PAD_INDEX], inputData.pad[BOTTOM_PAD_INDEX], inputData.stride[0],
        inputData.ceilMode, inputData.inputShape[H_DIM]);
    int64_t expectedW = InferCalculateOutShape(
        inputData.kernelSize[1], inputData.pad[LEFT_PAD_INDEX], inputData.pad[RIGHT_PAD_INDEX], inputData.stride[1],
        inputData.ceilMode, inputData.inputShape[W_DIM]);
    if (inputData.outShape[0] > expectedH || inputData.outShape[1] > expectedW) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName_, "y",
            (std::to_string(inputData.outShape[0]) + ", " + std::to_string(inputData.outShape[1])).c_str(),
            ("when padmode is CALCULATED, output shape must not exceed (" + std::to_string(expectedH) + ", " + std::to_string(expectedW) + ")").c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPoolV3BaseTiling::CheckOutPutShape()
{
    if (padModeStr_ == "CALCULATED") {
        return CheckOutPutShapeForCalculated();
    } else if (padModeStr_ == "VALID") {
        return CheckOutPutShapeForValid();
    } else if (padModeStr_ == "SAME") {
        return CheckOutPutShapeForSame();
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPoolV3BaseTiling::GetShapeAttrsInfo()
{
    const char* opName_ = "MaxPoolV3";
    if (GetAttrsInfo() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_, "GetAttrsInfo fail.");
        return ge::GRAPH_FAILED;
    }
    if (GetShapeAndDtype() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_, "GetShapeAndDtype fail.");
        return ge::GRAPH_FAILED;
    }

    if (inputData.globalPooling) {
        if ((inputData.inputShape[0] != 0 && inputData.outShape[0] != 1) ||
            (inputData.inputShape[1] != 0 && inputData.outShape[1] != 1)) {
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(opName_, "y[h-dim], y[w-dim]",
                (std::to_string(inputData.outShape[0]) + ", " + std::to_string(inputData.outShape[1])).c_str(),
                "when global_pooling is true, output shape in h-dim and w-dim must be 1 if the corresponding input shape is not zero");
            return ge::GRAPH_FAILED;
        }
        inputData.pad = {0, 0, 0, 0};
        inputData.stride = inputData.inputShape;
        inputData.kernelSize = inputData.inputShape;
        return ge::GRAPH_SUCCESS;
    }
    if (GetKernelInfo() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_, "GetKernelInfo fail.");
        return ge::GRAPH_FAILED;
    }
    if (GetPadInfo() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_, "GetPadInfo fail.");
        return ge::GRAPH_FAILED;
    }
    if (CheckOutPutShape() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_, "CheckOutPutShape fail.");
        return ge::GRAPH_FAILED;
    }
    RefineShape();
    return ge::GRAPH_SUCCESS;
}

void MaxPoolV3BaseTiling::RefineShape()
{
    if (inputData.outShape[H_DIM] == 1) {
        if (inputData.kernelSize[H_DIM] >= inputData.inputShape[H_DIM] + inputData.pad[TOP_PAD_INDEX]) {
            inputData.kernelSize[H_DIM] = inputData.inputShape[H_DIM];
            inputData.pad[TOP_PAD_INDEX] = 0;
            inputData.pad[BOTTOM_PAD_INDEX] = 0;
        } else {
            inputData.kernelSize[H_DIM] = inputData.kernelSize[H_DIM] - inputData.pad[TOP_PAD_INDEX];
            inputData.pad[TOP_PAD_INDEX] = 0;
            inputData.pad[BOTTOM_PAD_INDEX] = 0;
        }
        inputData.stride[H_DIM] = inputData.kernelSize[H_DIM];
    }

    if (inputData.outShape[W_DIM] == 1) {
        if (inputData.kernelSize[W_DIM] >= inputData.inputShape[W_DIM] + inputData.pad[LEFT_PAD_INDEX]) {
            inputData.kernelSize[W_DIM] = inputData.inputShape[W_DIM];
            inputData.pad[LEFT_PAD_INDEX] = 0;
            inputData.pad[RIGHT_PAD_INDEX] = 0;
        } else {
            inputData.kernelSize[W_DIM] = inputData.kernelSize[W_DIM] - inputData.pad[LEFT_PAD_INDEX];
            inputData.pad[LEFT_PAD_INDEX] = 0;
            inputData.pad[RIGHT_PAD_INDEX] = 0;
        }
        inputData.stride[W_DIM] = inputData.kernelSize[W_DIM];
    }
}

bool MaxPoolV3BaseTiling::IsCapable()
{
    return true;
}

ge::graphStatus MaxPoolV3BaseTiling::DoOpTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPoolV3BaseTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t MaxPoolV3BaseTiling::GetTilingKey() const
{
    return 0;
}

ge::graphStatus MaxPoolV3BaseTiling::GetWorkspaceSize()
{
    uint32_t sysWorkspace = WS_SYS_SIZE;
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = sysWorkspace;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPoolV3BaseTiling::PostTiling()
{
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
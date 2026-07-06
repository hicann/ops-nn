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
 * \file max_pool_grad_tiling_base.cpp
 * \brief
 */

#include "error_util.h"
#include "max_pool_grad_tiling.h"
#include "op_host/tiling_util.h"

namespace optiling {

constexpr size_t CHW_DIM_NUM = 3U;
constexpr size_t C_DIM_OFFSET = 3;
constexpr size_t H_DIM_OFFSET = 2;
constexpr size_t W_DIM_OFFSET = 1;
constexpr size_t H_ATTR_INDEX_NHWC = 1;
constexpr size_t W_ATTR_INDEX_NHWC = 2;
constexpr size_t H_ATTR_INDEX_NCHW = 2;
constexpr size_t W_ATTR_INDEX_NCHW = 3;

constexpr uint32_t X1_INDEX = 0;
constexpr uint32_t X2_INDEX = 1;
constexpr uint32_t GRAD_INDEX = 2;
constexpr size_t KSIZE_ATTR_INDEX = 0U;
constexpr size_t STRIDES_ATTR_INDEX = 1U;
constexpr size_t PADDING_ATTR_INDEX = 2U;
constexpr size_t DATA_FORMAT_ATTR_INDEX = 3U;

constexpr uint32_t Y_INDEX = 0;
constexpr uint64_t NUM_TWO = 2;
constexpr size_t HW_DIM_NUM = 3;
constexpr uint32_t MAX_BLOCK_COUNT = 4095;

constexpr size_t NC_DIM_NUM = 2;
constexpr size_t NCHW_DIM_NUM = 4;
constexpr size_t PADS_ATTR_INDEX = 3U;
constexpr size_t CEIL_MODE_ATTR_INDEX = 5U;

static std::string GetDataFormat(gert::TilingContext* context)
{
    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return "";
    }
    const char* data_format = attrs->GetAttrPointer<char>(DATA_FORMAT_ATTR_INDEX);
    if (data_format == nullptr) {
        return "";
    }
    return std::string(data_format);
}

static bool CheckX2GradShapeEqual(gert::TilingContext* context, const gert::StorageShape* x2Shape,
                                  const gert::StorageShape* gradShape, size_t dimNum)
{
    for (size_t i = 0; i < dimNum; i++) {
        uint64_t x2DimValue = x2Shape->GetStorageShape().GetDim(i);
        uint64_t gradDimValue = gradShape->GetStorageShape().GetDim(i);
        OP_TILING_CHECK(x2DimValue != gradDimValue,
                        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                            context->GetNodeName(), "x2, grad",
                            (std::to_string(x2DimValue) + ", " + std::to_string(gradDimValue)).c_str(),
                            ("the dim values of x2 and grad must be equal, but x2[" + std::to_string(i) +
                             "]=" + std::to_string(x2DimValue) + ", grad[" + std::to_string(i) +
                             "]=" + std::to_string(gradDimValue))
                                .c_str()),
                        return false);
    }
    return true;
}

static bool CheckNCDimEqual(gert::TilingContext* context, const gert::StorageShape* x1Shape,
                            const gert::StorageShape* x2Shape, const std::string& format, size_t dimNum)
{
    uint32_t cPosIdx = (format == "NHWC") ? dimNum - 1 : dimNum - 4;
    uint64_t xNDim = (dimNum == CHW_DIM_NUM) ? 1 : x1Shape->GetStorageShape().GetDim(0);
    uint64_t gradNDim = (dimNum == CHW_DIM_NUM) ? 1 : x2Shape->GetStorageShape().GetDim(0);
    uint64_t xCDim = x1Shape->GetStorageShape().GetDim(cPosIdx);
    uint64_t gradCDim = x2Shape->GetStorageShape().GetDim(cPosIdx);
    OP_TILING_CHECK(
        (xNDim != gradNDim) || (xCDim != gradCDim),
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "grad, x1",
                                               ("N=" + std::to_string(gradNDim) + "," + std::to_string(gradCDim) +
                                                " and N=" + std::to_string(xNDim) + "," + std::to_string(xCDim))
                                                   .c_str(),
                                               "the N,C dims of grad and x1 must be equal"),
        return false);
    return true;
}

bool MaxPoolGradTilingBase::CheckInputShape()
{
    const gert::StorageShape* x1Shape = context_->GetInputShape(X1_INDEX);
    const gert::StorageShape* x2Shape = context_->GetInputShape(X2_INDEX);
    const gert::StorageShape* gradShape = context_->GetInputShape(GRAD_INDEX);
    size_t X1DimNum = Ops::Base::EnsureNotScalar(x1Shape->GetStorageShape()).GetDimNum();
    size_t X2DimNum = Ops::Base::EnsureNotScalar(x2Shape->GetStorageShape()).GetDimNum();
    size_t gradDimNum = Ops::Base::EnsureNotScalar(gradShape->GetStorageShape()).GetDimNum();

    std::string data_formatStr = GetDataFormat(context_);
    OP_CHECK_IF(
        !(data_formatStr == "NCHW" || data_formatStr == "NHWC"),
        OP_LOGE_FOR_INVALID_FORMAT(context_->GetNodeName(), "data_format", data_formatStr.c_str(), "NHWC or NCHW"),
        return false);

    OP_CHECK_IF(
        ((X1DimNum != NCHW_DIM_NUM) || (X2DimNum != NCHW_DIM_NUM) || (gradDimNum != NCHW_DIM_NUM)),
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            context_->GetNodeName(), "input",
            (std::to_string(X1DimNum) + ", " + std::to_string(X2DimNum) + ", " + std::to_string(gradDimNum)).c_str(),
            ("all inputs must have " + std::to_string(NCHW_DIM_NUM) +
             " dims, but got xdim=" + std::to_string(X1DimNum) + ", x2dim=" + std::to_string(X2DimNum) +
             ", gradDim=" + std::to_string(gradDimNum))
                .c_str()),
        return false);

    for (uint32_t i = 0; i < X1DimNum; i++) {
        uint64_t xDimVal = x1Shape->GetStorageShape().GetDim(i);
        OP_CHECK_IF(xDimVal == 0,
                    OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                        context_->GetNodeName(), "x", std::to_string(xDimVal).c_str(),
                        ("the dim" + std::to_string(i) + " of x should not be 0").c_str()),
                    return false);
    }

    if (!CheckX2GradShapeEqual(context_, x2Shape, gradShape, X1DimNum)) {
        return false;
    }
    return CheckNCDimEqual(context_, x1Shape, x2Shape, data_formatStr, X1DimNum);
}

ge::graphStatus MaxPoolGradTilingBase::CheckInputDtype()
{
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(X1_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(X2_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(GRAD_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOutputDesc(Y_INDEX));
    auto x1DataType = context_->GetInputDesc(X1_INDEX)->GetDataType();
    auto x2DataType = context_->GetInputDesc(X2_INDEX)->GetDataType();
    auto gradDataType = context_->GetInputDesc(GRAD_INDEX)->GetDataType();

    OP_CHECK_IF(!(x1DataType == x2DataType && x2DataType == gradDataType),
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context_->GetNodeName(), "x1, x2, grads",
                                                       (ge::TypeUtils::DataTypeToSerialString(x1DataType) + ", " +
                                                        ge::TypeUtils::DataTypeToSerialString(x2DataType) + ", " +
                                                        ge::TypeUtils::DataTypeToSerialString(gradDataType))
                                                           .c_str(),
                                                       "the dtypes of x1, x2, grads must be the same"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((x1DataType != ge::DT_FLOAT) && (x1DataType != ge::DT_FLOAT16) && (x1DataType != ge::DT_BF16),
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->GetNodeName(), "x1",
                                                      ge::TypeUtils::DataTypeToSerialString(x1DataType).c_str(),
                                                      "the dtype of x1 must be fp32, fp16, or bf16"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckKSizeStridesForNhwc(gert::TilingContext* context, const int64_t* kSizeVector,
                                                const int64_t* stridesVector)
{
    OP_TILING_CHECK(
        kSizeVector[0] != 1 || kSizeVector[C_DIM_OFFSET] != 1,
        OP_LOGE_FOR_INVALID_VALUE(
            context->GetNodeName(), ("ksize[0], ksize[" + std::to_string(C_DIM_OFFSET) + "]").c_str(),
            (std::to_string(kSizeVector[0]) + ", " + std::to_string(kSizeVector[C_DIM_OFFSET])).c_str(), "1, 1"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        stridesVector[0] != 1 || stridesVector[C_DIM_OFFSET] != 1,
        OP_LOGE_FOR_INVALID_VALUE(
            context->GetNodeName(), ("strides[0], strides[" + std::to_string(C_DIM_OFFSET) + "]").c_str(),
            (std::to_string(stridesVector[0]) + ", " + std::to_string(stridesVector[C_DIM_OFFSET])).c_str(), "1, 1"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckKSizeStridesForNchw(gert::TilingContext* context, const int64_t* kSizeVector,
                                                const int64_t* stridesVector)
{
    OP_TILING_CHECK(kSizeVector[0] != 1 || kSizeVector[1] != 1,
                    OP_LOGE_FOR_INVALID_VALUE(
                        context->GetNodeName(), "ksize[0], ksize[1]",
                        (std::to_string(kSizeVector[0]) + ", " + std::to_string(kSizeVector[1])).c_str(), "1, 1"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(stridesVector[0] != 1 || stridesVector[1] != 1,
                    OP_LOGE_FOR_INVALID_VALUE(
                        context->GetNodeName(), "strides[0], strides[1]",
                        (std::to_string(stridesVector[0]) + ", " + std::to_string(stridesVector[1])).c_str(), "1, 1"),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckKSizeStridesPositive(gert::TilingContext* context, int32_t kSizeDimNum,
                                                 int32_t stridesDimNum, const int64_t* kSizeVector,
                                                 const int64_t* stridesVector)
{
    for (int32_t i = 0; i < kSizeDimNum; i++) {
        OP_TILING_CHECK(kSizeVector[i] <= 0,
                        OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), ("ksize[" + std::to_string(i) + "]").c_str(),
                                                  std::to_string(kSizeVector[i]).c_str(), "positive integer"),
                        return ge::GRAPH_FAILED);
    }
    for (int32_t i = 0; i < stridesDimNum; i++) {
        OP_TILING_CHECK(
            stridesVector[i] <= 0,
            OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), ("strides[" + std::to_string(i) + "]").c_str(),
                                      std::to_string(stridesVector[i]).c_str(), "positive integer"),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPoolGradTilingBase::CheckAttrVal()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    auto kSizeVector = attrs->GetListInt(KSIZE_ATTR_INDEX)->GetData();
    auto stridesVector = attrs->GetListInt(STRIDES_ATTR_INDEX)->GetData();
    int32_t kSizeDimNum = attrs->GetListInt(KSIZE_ATTR_INDEX)->GetSize();
    int32_t stridesDimNum = attrs->GetListInt(STRIDES_ATTR_INDEX)->GetSize();

    std::string data_formatStr = GetDataFormat(context_);
    ge::graphStatus ret = (data_formatStr == "NHWC") ? CheckKSizeStridesForNhwc(context_, kSizeVector, stridesVector) :
                                                       CheckKSizeStridesForNchw(context_, kSizeVector, stridesVector);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    return CheckKSizeStridesPositive(context_, kSizeDimNum, stridesDimNum, kSizeVector, stridesVector);
}

ge::graphStatus MaxPoolGradTilingBase::CheckAttrShape()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    int32_t kSizeDimNum = attrs->GetListInt(KSIZE_ATTR_INDEX)->GetSize();
    int32_t stridesDimNum = attrs->GetListInt(STRIDES_ATTR_INDEX)->GetSize();
    const char* padMode = attrs->GetAttrPointer<char>(PADDING_ATTR_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, padMode);
    std::string padModeStr = padMode;
    OP_TILING_CHECK(IsInvalidPaddingModeWithCalculated(padModeStr),
                    OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "padding", padModeStr.c_str(), "VALID or SAME"),
                    return ge::GRAPH_FAILED);

    // Check attr dim num
    OP_CHECK_IF(
        (kSizeDimNum != NCHW_DIM_NUM),
        OP_LOGE_FOR_INVALID_LISTSIZE(context_->GetNodeName(), "ksize", std::to_string(kSizeDimNum).c_str(), "4"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (stridesDimNum != NCHW_DIM_NUM),
        OP_LOGE_FOR_INVALID_LISTSIZE(context_->GetNodeName(), "strides", std::to_string(stridesDimNum).c_str(), "4"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPoolGradTilingBase::SetInputParams()
{
    const gert::Shape xShape = context_->GetInputShape(X1_INDEX)->GetStorageShape();
    const gert::Shape gradShape = context_->GetInputShape(GRAD_INDEX)->GetStorageShape();
    size_t xDimNum = xShape.GetDimNum();
    auto attrs = context_->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const char* data_format = attrs->GetAttrPointer<char>(DATA_FORMAT_ATTR_INDEX);
    std::string data_formatStr = data_format;

    uint32_t cPosIdx = xDimNum - C_DIM_OFFSET;
    uint32_t hPosIdx = xDimNum - H_DIM_OFFSET;
    uint32_t wPosIdx = xDimNum - W_DIM_OFFSET;

    inputData.inputFormat = ge::Format::FORMAT_NCHW;

    if (data_formatStr == "NHWC") {
        inputData.inputFormat = ge::Format::FORMAT_NHWC;
        hPosIdx = hPosIdx - 1;
        wPosIdx = wPosIdx - 1;
        cPosIdx = xDimNum - 1;
    }

    inputData.nX = (xDimNum == CHW_DIM_NUM) ? 1 : xShape.GetDim(0);
    inputData.cX = xShape.GetDim(cPosIdx);
    inputData.hX = xShape.GetDim(hPosIdx);
    inputData.wX = xShape.GetDim(wPosIdx);
    inputData.nGrad = (xDimNum == CHW_DIM_NUM) ? 1 : gradShape.GetDim(0);
    inputData.cGrad = gradShape.GetDim(cPosIdx);
    inputData.hGrad = gradShape.GetDim(hPosIdx);
    inputData.wGrad = gradShape.GetDim(wPosIdx);
    inputData.gradShapeSize = gradShape.GetShapeSize();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPoolGradTilingBase::SetAttrParams()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    int32_t kSizeDimNum = attrs->GetListInt(KSIZE_ATTR_INDEX)->GetSize();
    int32_t stridesDimNum = attrs->GetListInt(STRIDES_ATTR_INDEX)->GetSize();
    auto kSizeVector = attrs->GetListInt(KSIZE_ATTR_INDEX)->GetData();
    auto stridesVector = attrs->GetListInt(STRIDES_ATTR_INDEX)->GetData();

    std::string data_formatStr = GetDataFormat(context_);

    if (data_formatStr == "NHWC") {
        inputData.hKernel = kSizeVector[H_ATTR_INDEX_NHWC];
        inputData.wKernel = (kSizeDimNum == 1) ? inputData.hKernel : kSizeVector[W_ATTR_INDEX_NHWC];
    } else {
        inputData.hKernel = kSizeVector[H_ATTR_INDEX_NCHW];
        inputData.wKernel = (kSizeDimNum == 1) ? inputData.hKernel : kSizeVector[W_ATTR_INDEX_NCHW];
    }

    if (stridesDimNum == 0) {
        inputData.hStride = inputData.hKernel;
        inputData.wStride = inputData.wKernel;
    } else {
        if (data_formatStr == "NHWC") {
            inputData.hStride = stridesVector[H_ATTR_INDEX_NHWC];
            inputData.wStride = (stridesDimNum == 1) ? inputData.hStride : stridesVector[W_ATTR_INDEX_NHWC];
        } else {
            inputData.hStride = stridesVector[H_ATTR_INDEX_NCHW];
            inputData.wStride = (stridesDimNum == 1) ? inputData.hStride : stridesVector[W_ATTR_INDEX_NCHW];
        }
    }

    const char* padMode = attrs->GetAttrPointer<char>(PADDING_ATTR_INDEX);
    std::string padModeStr = padMode;
    if (padModeStr == "VALID") {
        inputData.hPad = 0;
        inputData.wPad = 0;
    } else if (padModeStr == "SAME") {
        int64_t hPadNeed = std::max(int64_t{0},
                                    (inputData.hGrad - 1) * inputData.hStride + inputData.hKernel - inputData.hX);
        inputData.hPad = hPadNeed / DIGIT_TWO;
        int64_t wPadNeed = std::max(int64_t{0},
                                    (inputData.wGrad - 1) * inputData.wStride + inputData.wKernel - inputData.wX);
        inputData.wPad = wPadNeed / DIGIT_TWO;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPoolGradTilingBase::CheckInputValid()
{
    const uint64_t kh = inputData.hKernel;
    const uint64_t kw = inputData.wKernel;
    const uint64_t pHTop = inputData.hPad;
    const uint64_t pWTop = inputData.wPad;
    const uint64_t dilationH = inputData.hDilation;
    const uint64_t dilationW = inputData.wDilation;

    OP_CHECK_IF((pHTop >= kh) || (pWTop >= kw),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context_->GetNodeName(), "padding",
                    ("hPad=" + std::to_string(pHTop) + ", wPad=" + std::to_string(pWTop)).c_str(),
                    ("padSize must be smaller than kernelSize (hKernel=" + std::to_string(kh) +
                     ", wKernel=" + std::to_string(kw) + ")")
                        .c_str()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((pHTop >= (kh - 1) * dilationH + 1) || (pWTop >= (kw - 1) * dilationW + 1),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context_->GetNodeName(), "padding",
                    ("hPad=" + std::to_string(pHTop) + ", wPad=" + std::to_string(pWTop)).c_str(),
                    ("padSize must be smaller than (kernelSize - 1) * dilation + 1 "
                     "(hDilation=" +
                     std::to_string(dilationH) + ", wDilation=" + std::to_string(dilationW) + ")")
                        .c_str()),
                return ge::GRAPH_FAILED);
    // Check outerDim invaild
    auto attrs = context_->GetAttrs();
    const char* padMode = attrs->GetAttrPointer<char>(PADDING_ATTR_INDEX);
    std::string padModeStr = padMode;

    OP_TILING_CHECK(
        !CheckGradShape(inputData, padModeStr),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            context_->GetNodeName(), "grad",
            ("hGrad=" + std::to_string(inputData.hGrad) + ", wGrad=" + std::to_string(inputData.wGrad)).c_str(),
            "grad shape is invalid for the given pooling parameters"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

void MaxPoolGradTilingBase::SetOtherInputParams()
{
    inputData.inputDtype = context_->GetInputDesc(X1_INDEX)->GetDataType();
    inputData.isInt32Meet = IsGreaterThanInt32MaxNCHW(inputData) ? 1 : 0;
}

ge::graphStatus MaxPoolGradTilingBase::GetShapeAttrsInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    (void)ascendcPlatform.GetSocVersion();
    if (!Ops::NN::OpTiling::IsRegbaseSocVersion(context_)) {
        // Skip the current template
        return ge::GRAPH_PARAM_INVALID;
    }

    OP_LOGD(context_->GetNodeName(), "Enter MaxPoolGradTilingBase GetShapeAttrsInfo.");
    OP_CHECK_IF(ge::GRAPH_SUCCESS != CheckInputDtype(), OP_LOGE(context_->GetNodeName(), "The input dtype is invalid."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!CheckInputShape(), OP_LOGE(context_->GetNodeName(), "The input relationship is invalid."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ge::GRAPH_SUCCESS != CheckAttrShape(), OP_LOGE(context_->GetNodeName(), "The attr shape is invalid."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ge::GRAPH_SUCCESS != CheckAttrVal(), OP_LOGE(context_->GetNodeName(), "The attr value is invalid."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ge::GRAPH_SUCCESS != SetInputParams(), OP_LOGE(context_->GetNodeName(), "Set input shape failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ge::GRAPH_SUCCESS != SetAttrParams(), OP_LOGE(context_->GetNodeName(), "Set attr shape failed."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ge::GRAPH_SUCCESS != CheckInputValid(), OP_LOGE(context_->GetNodeName(), "The input shape is invalid."),
                return ge::GRAPH_FAILED);
    SetOtherInputParams();
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

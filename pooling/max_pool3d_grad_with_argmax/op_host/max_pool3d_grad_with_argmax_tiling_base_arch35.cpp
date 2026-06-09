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
 * \file max_pool3d_grad_with_argmax_tiling_base_arch35.cpp
 * \brief
 */

#include "error_util.h"
#include "op_host/tiling_util.h"
#include "max_pool3d_grad_with_argmax_tiling_arch35.h"

namespace optiling {

constexpr size_t CDHW_DIM_NUM = 4U;
constexpr size_t DATA_FORMAT_ATTR_INDEX = 5U;
constexpr size_t C_DIM_OFFSET = 4; // pos = dim - offset
constexpr size_t D_DIM_OFFSET = 3;
constexpr size_t H_DIM_OFFSET = 2;
constexpr size_t W_DIM_OFFSET = 1;
constexpr size_t D_ATTR_INDEX = 0;
constexpr size_t H_ATTR_INDEX = 1;
constexpr size_t W_ATTR_INDEX = 2;
constexpr int64_t WS_SYS_SIZE = 16 * 1024 * 1024;
static const gert::Shape g_vec_1_shape = {1};

static const gert::Shape& EnsureNotScalar(const gert::Shape& inShape)
{
    if (inShape.IsScalar()) {
        return g_vec_1_shape;
    }
    return inShape;
}

static inline bool IsGreaterThanInt32Max(const MaxPool3DGradWithArgmaxInputInfo& inputData)
{
    if (inputData.indexDtype == ge::DataType::DT_INT32) {
        return false;
    }

    int64_t cubeSize = inputData.dX * inputData.hX * inputData.wX;
    return cubeSize > static_cast<int64_t>(INT32_MAX);
}

bool MaxPool3DGradWithArgmaxTilingBaseV35::CheckInputShape()
{
    const char* opName_ = "MaxPool3DGradWithArgmax";
    const gert::StorageShape* xShape = context_->GetInputShape(X_INDEX);
    const gert::StorageShape* gradShape = context_->GetInputShape(GRAD_INDEX);
    const gert::StorageShape* argmaxShape = context_->GetInputShape(ARGMAX_INDEX);
    size_t xDimNum = EnsureNotScalar(xShape->GetStorageShape()).GetDimNum();
    size_t gradDimNum = EnsureNotScalar(gradShape->GetStorageShape()).GetDimNum();
    size_t argmaxDimNum = EnsureNotScalar(argmaxShape->GetStorageShape()).GetDimNum();
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const char* data_format = attrs->GetAttrPointer<char>(DATA_FORMAT_ATTR_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, data_format);
    std::string data_formatStr = data_format;

    if (!(data_formatStr == "NCDHW" || data_formatStr == "NDHWC")) {
        OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(opName_, "data_format", data_format, "expect [NDHWC] or [NCDHW].");
        return false;
    }

    if (((xDimNum != NCDHW_DIM_NUM) || (gradDimNum != NCDHW_DIM_NUM) || (argmaxDimNum != NCDHW_DIM_NUM)) &&
        ((xDimNum != CDHW_DIM_NUM) || (gradDimNum != CDHW_DIM_NUM) || (argmaxDimNum != CDHW_DIM_NUM))) {
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            opName_, "x, grad, argmax",
            (std::to_string(xDimNum) + ", " + std::to_string(gradDimNum) + ", " + std::to_string(argmaxDimNum)).c_str(),
            "Input dim num should equal 5 or 4");
        return false;
    }
    for (uint32_t i = 0; i < xDimNum; i++) {
        if (xShape->GetStorageShape().GetDim(i) == 0) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                opName_, "x", std::to_string(xShape->GetStorageShape().GetDim(i)).c_str(),
                "Input x shape can not be 0.");
            return false;
        }
    }

    // gradShape&argmaxShape's shape should be equal
    for (size_t i = 0; i < xDimNum; i++) {
        uint64_t gradDimValue = gradShape->GetStorageShape().GetDim(i);
        uint64_t argmaxDimValue = argmaxShape->GetStorageShape().GetDim(i);
        if (gradDimValue != argmaxDimValue) {
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                opName_, "grad, argmax", (std::to_string(gradDimValue) + ", " + std::to_string(argmaxDimValue)).c_str(),
                (std::string("Input dim check invalid, grad[") + std::to_string(i) + "] != argmax[" +
                 std::to_string(i) + "]")
                    .c_str());
            return false;
        }
    }

    // Input NCDim should be equal
    uint32_t cPosIdx = (data_formatStr == "NDHWC") ? xDimNum - 1 : xDimNum - 4;
    uint64_t xNDim = (xDimNum == CDHW_DIM_NUM) ? 1 : xShape->GetStorageShape().GetDim(0);
    uint64_t gradNDim = (gradDimNum == CDHW_DIM_NUM) ? 1 : gradShape->GetStorageShape().GetDim(0);
    uint64_t xCDim = xShape->GetStorageShape().GetDim(cPosIdx);
    uint64_t gradCDim = gradShape->GetStorageShape().GetDim(cPosIdx);
    if ((xNDim != gradNDim) || (xCDim != gradCDim)) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            opName_, "x, grad",
            (std::to_string(xNDim) + ", " + std::to_string(xCDim) + " vs " + std::to_string(gradNDim) + ", " +
             std::to_string(gradCDim))
                .c_str(),
            "Input N,C dim check invalid, not equal.");
        return false;
    }

    return true;
}

ge::graphStatus MaxPool3DGradWithArgmaxTilingBaseV35::CheckInputDtype()
{
    const char* opName_ = "MaxPool3DGradWithArgmax";
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(X_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(GRAD_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(ARGMAX_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOutputDesc(Y_INDEX));
    auto xDataType = context_->GetInputDesc(X_INDEX)->GetDataType();
    auto gradDataType = context_->GetInputDesc(GRAD_INDEX)->GetDataType();
    auto argmaxDataType = context_->GetInputDesc(ARGMAX_INDEX)->GetDataType();
    auto yOutDataType = context_->GetOutputDesc(Y_INDEX)->GetDataType();

    if (xDataType != gradDataType) {
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            opName_, "x, grad",
            (std::to_string(static_cast<int32_t>(xDataType)) + ", " +
             std::to_string(static_cast<int32_t>(gradDataType)))
                .c_str(),
            "x data type not equal grad data type.");
        return ge::GRAPH_FAILED;
    }
    if (xDataType != yOutDataType) {
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            opName_, "x, y",
            (std::to_string(static_cast<int32_t>(xDataType)) + ", " +
             std::to_string(static_cast<int32_t>(yOutDataType)))
                .c_str(),
            "x data type not equal y data type.");
        return ge::GRAPH_FAILED;
    }
    if ((xDataType != ge::DT_FLOAT) && (xDataType != ge::DT_FLOAT16) && (xDataType != ge::DT_BF16)) {
        std::string xDataTypeStr = std::to_string(static_cast<int32_t>(xDataType));
        OP_LOGE_FOR_INVALID_DTYPE(opName_, "x", xDataTypeStr.c_str(), "[0(DT_FLOAT), 1(DT_FLOAT16), 27(DT_BF16)]");
        return ge::GRAPH_FAILED;
    }
    if ((argmaxDataType != ge::DT_INT32) && (argmaxDataType != ge::DT_INT64)) {
        std::string argmaxDataTypeStr = std::to_string(static_cast<int32_t>(argmaxDataType));
        OP_LOGE_FOR_INVALID_DTYPE(opName_, "argmax", argmaxDataTypeStr.c_str(), "[3(DT_INT32), 9(DT_INT64)]");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPool3DGradWithArgmaxTilingBaseV35::CheckAttrShape()
{
    const char* opName_ = "MaxPool3DGradWithArgmax";
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    int32_t kSizeDimNum = attrs->GetListInt(KSIZE_ATTR_INDEX)->GetSize();
    int32_t stridesDimNum = attrs->GetListInt(STRIDES_ATTR_INDEX)->GetSize();
    int32_t padsDimNum = attrs->GetListInt(PADS_ATTR_INDEX)->GetSize();
    int32_t dilationsDimNum = attrs->GetListInt(DILATION_ATTR_INDEX)->GetSize();

    if ((kSizeDimNum != DHW_DIM_NUM) && (kSizeDimNum != 1)) {
        OP_LOGE_FOR_INVALID_LISTSIZE(opName_, "Length of kSize", std::to_string(kSizeDimNum).c_str(), "3 or 1");
        return ge::GRAPH_FAILED;
    }
    if ((stridesDimNum != DHW_DIM_NUM) && (stridesDimNum != 1) && (stridesDimNum != 0)) {
        OP_LOGE_FOR_INVALID_LISTSIZE(opName_, "Length of strides", std::to_string(stridesDimNum).c_str(), "3, 1, or 0");
        return ge::GRAPH_FAILED;
    }
    if ((padsDimNum != DHW_DIM_NUM) && (padsDimNum != 1)) {
        OP_LOGE_FOR_INVALID_LISTSIZE(opName_, "Length of pads", std::to_string(padsDimNum).c_str(), "3 or 1");
        return ge::GRAPH_FAILED;
    }
    if ((dilationsDimNum != DHW_DIM_NUM) && (dilationsDimNum != 1)) {
        OP_LOGE_FOR_INVALID_LISTSIZE(opName_, "Length of dilations", std::to_string(dilationsDimNum).c_str(), "3 or 1");
        return ge::GRAPH_FAILED;
    }

    auto kSizeVector = attrs->GetListInt(KSIZE_ATTR_INDEX)->GetData();
    auto stridesVector = attrs->GetListInt(STRIDES_ATTR_INDEX)->GetData();
    auto padsVector = attrs->GetListInt(PADS_ATTR_INDEX)->GetData();
    auto dilationsVector = attrs->GetListInt(DILATION_ATTR_INDEX)->GetData();
    for (uint32_t i = 0; i < static_cast<uint32_t>(kSizeDimNum); i++) {
        if (kSizeVector[i] <= 0) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                opName_, (std::string("kSize[") + std::to_string(i) + "]").c_str(),
                std::to_string(kSizeVector[i]).c_str(), "kSize value should bigger than 0");
            return ge::GRAPH_FAILED;
        }
    }
    for (uint32_t i = 0; i < static_cast<uint32_t>(stridesDimNum); i++) {
        if (stridesVector[i] <= 0) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                opName_, (std::string("strides[") + std::to_string(i) + "]").c_str(),
                std::to_string(stridesVector[i]).c_str(), "strides value should bigger than 0");
            return ge::GRAPH_FAILED;
        }
    }
    for (uint32_t i = 0; i < static_cast<uint32_t>(padsDimNum); i++) {
        if (padsVector[i] < 0) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                opName_, (std::string("pads[") + std::to_string(i) + "]").c_str(),
                std::to_string(padsVector[i]).c_str(), "pads value should bigger or equal 0");
            return ge::GRAPH_FAILED;
        }
    }
    for (uint32_t i = 0; i < static_cast<uint32_t>(dilationsDimNum); i++) {
        if (dilationsVector[i] <= 0) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                opName_, (std::string("dilations[") + std::to_string(i) + "]").c_str(),
                std::to_string(dilationsVector[i]).c_str(), "dilations value should bigger than 0");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPool3DGradWithArgmaxTilingBaseV35::SetInputParams()
{
    const gert::Shape xShape = context_->GetInputShape(X_INDEX)->GetStorageShape();
    const gert::Shape gradShape = context_->GetInputShape(GRAD_INDEX)->GetStorageShape();
    size_t xDimNum = xShape.GetDimNum();
    auto attrs = context_->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const char* data_format = attrs->GetAttrPointer<char>(DATA_FORMAT_ATTR_INDEX);
    std::string data_formatStr = data_format;

    uint32_t cPosIdx = xDimNum - C_DIM_OFFSET;
    uint32_t dPosIdx = xDimNum - D_DIM_OFFSET;
    uint32_t hPosIdx = xDimNum - H_DIM_OFFSET;
    uint32_t wPosIdx = xDimNum - W_DIM_OFFSET;

    inputData.inputFormat = ge::Format::FORMAT_NCDHW;

    if (data_formatStr == "NDHWC") {
        inputData.inputFormat = ge::Format::FORMAT_NDHWC;
        dPosIdx = dPosIdx - 1;
        hPosIdx = hPosIdx - 1;
        wPosIdx = wPosIdx - 1;
        cPosIdx = xDimNum - 1;
    }

    inputData.nX = (xDimNum == CDHW_DIM_NUM) ? 1 : xShape.GetDim(0);
    inputData.cX = xShape.GetDim(cPosIdx);
    inputData.dX = xShape.GetDim(dPosIdx);
    inputData.hX = xShape.GetDim(hPosIdx);
    inputData.wX = xShape.GetDim(wPosIdx);
    inputData.nGrad = (xDimNum == CDHW_DIM_NUM) ? 1 : gradShape.GetDim(0);
    inputData.cGrad = gradShape.GetDim(cPosIdx);
    inputData.dGrad = gradShape.GetDim(dPosIdx);
    inputData.hGrad = gradShape.GetDim(hPosIdx);
    inputData.wGrad = gradShape.GetDim(wPosIdx);
    inputData.gradShapeSize = gradShape.GetShapeSize();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPool3DGradWithArgmaxTilingBaseV35::SetAttrParams()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    int32_t kSizeDimNum = attrs->GetListInt(KSIZE_ATTR_INDEX)->GetSize();
    int32_t stridesDimNum = attrs->GetListInt(STRIDES_ATTR_INDEX)->GetSize();
    int32_t padsDimNum = attrs->GetListInt(PADS_ATTR_INDEX)->GetSize();
    int32_t dilationsDimNum = attrs->GetListInt(DILATION_ATTR_INDEX)->GetSize();
    auto kSizeVector = attrs->GetListInt(KSIZE_ATTR_INDEX)->GetData();
    auto stridesVector = attrs->GetListInt(STRIDES_ATTR_INDEX)->GetData();
    auto padsVector = attrs->GetListInt(PADS_ATTR_INDEX)->GetData();
    auto dilationsVector = attrs->GetListInt(DILATION_ATTR_INDEX)->GetData();
    bool ceilMode = *attrs->GetBool(CEIL_MODE_ATTR_INDEX);
    inputData.ceilMode = ceilMode;
    inputData.dKernel = kSizeVector[D_ATTR_INDEX];
    inputData.hKernel = (kSizeDimNum == 1) ? inputData.dKernel : kSizeVector[H_ATTR_INDEX];
    inputData.wKernel = (kSizeDimNum == 1) ? inputData.dKernel : kSizeVector[W_ATTR_INDEX];
    if (stridesDimNum == 0) {
        inputData.dStride = inputData.dKernel;
        inputData.hStride = inputData.hKernel;
        inputData.wStride = inputData.wKernel;
    } else {
        inputData.dStride = stridesVector[D_ATTR_INDEX];
        inputData.hStride = (stridesDimNum == 1) ? inputData.dStride : stridesVector[H_ATTR_INDEX];
        inputData.wStride = (stridesDimNum == 1) ? inputData.dStride : stridesVector[W_ATTR_INDEX];
    }
    inputData.dPad = padsVector[D_ATTR_INDEX];
    inputData.hPad = (padsDimNum == 1) ? inputData.dPad : padsVector[H_ATTR_INDEX];
    inputData.wPad = (padsDimNum == 1) ? inputData.dPad : padsVector[W_ATTR_INDEX];
    inputData.dDilation = dilationsVector[D_ATTR_INDEX];
    inputData.hDilation = (dilationsDimNum == 1) ? inputData.dDilation : dilationsVector[H_ATTR_INDEX];
    inputData.wDilation = (dilationsDimNum == 1) ? inputData.dDilation : dilationsVector[W_ATTR_INDEX];
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPool3DGradWithArgmaxTilingBaseV35::CheckInputValid()
{
    const char* opName_ = "MaxPool3DGradWithArgmax";
    const uint64_t kd = inputData.dKernel;
    const uint64_t kh = inputData.hKernel;
    const uint64_t kw = inputData.wKernel;
    const uint64_t sd = inputData.dStride;
    const uint64_t sh = inputData.hStride;
    const uint64_t sw = inputData.wStride;
    const uint64_t pDTop = inputData.dPad;
    const uint64_t pHTop = inputData.hPad;
    const uint64_t pWTop = inputData.wPad;
    const uint64_t dilationD = inputData.dDilation;
    const uint64_t dilationH = inputData.hDilation;
    const uint64_t dilationW = inputData.wDilation;

    if ((pDTop > (kd / 2)) || (pHTop > (kh / 2)) || (pWTop > (kw / 2))) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            opName_, "pD, pH, pW",
            (std::to_string(pDTop) + ", " + std::to_string(pHTop) + ", " + std::to_string(pWTop)).c_str(),
            "padSize should smaller than kernelSize div 2");
        return ge::GRAPH_FAILED;
    }
    if ((pDTop > ((kd - 1) * dilationD + 1) / 2) || (pHTop > ((kh - 1) * dilationH + 1) / 2) ||
        (pWTop > ((kw - 1) * dilationW + 1) / 2)) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            opName_, "pD, pH, pW",
            (std::to_string(pDTop) + ", " + std::to_string(pHTop) + ", " + std::to_string(pWTop)).c_str(),
            "padSize should smaller than ((kernelSize - 1) * dilation + 1) / 2.");
        return ge::GRAPH_FAILED;
    }
    // Check outerDim invaild
    int64_t doExpected, hoExpected, woExpected;
    if (inputData.ceilMode) {
        doExpected = Ops::Base::CeilDiv((inputData.dX + NUM_TWO * pDTop + sd - dilationD * (kd - 1) - 1), sd);
        hoExpected = Ops::Base::CeilDiv((inputData.hX + NUM_TWO * pHTop + sh - dilationH * (kh - 1) - 1), sh);
        woExpected = Ops::Base::CeilDiv((inputData.wX + NUM_TWO * pWTop + sw - dilationW * (kw - 1) - 1), sw);
    } else {
        doExpected = (inputData.dX + NUM_TWO * pDTop + sd - dilationD * (kd - 1) - 1) / sd;
        hoExpected = (inputData.hX + NUM_TWO * pHTop + sh - dilationH * (kh - 1) - 1) / sh;
        woExpected = (inputData.wX + NUM_TWO * pWTop + sw - dilationW * (kw - 1) - 1) / sw;
    }
    doExpected = ((doExpected - 1) * sd >= inputData.dX + pDTop) ? doExpected - 1 : doExpected;
    hoExpected = ((hoExpected - 1) * sh >= inputData.hX + pHTop) ? hoExpected - 1 : hoExpected;
    woExpected = ((woExpected - 1) * sw >= inputData.wX + pWTop) ? woExpected - 1 : woExpected;
    if ((doExpected <= 0) || (doExpected != inputData.dGrad) || (hoExpected <= 0) || (hoExpected != inputData.hGrad) ||
        (woExpected <= 0) || (woExpected != inputData.wGrad)) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            opName_, "do, ho, wo",
            (std::to_string(doExpected) + ", " + std::to_string(hoExpected) + ", " + std::to_string(woExpected))
                .c_str(),
            "OuterDim size invalid");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

void MaxPool3DGradWithArgmaxTilingBaseV35::SetOtherInputParams()
{
    inputData.inputDtype = context_->GetInputDesc(X_INDEX)->GetDataType();
    inputData.indexDtype = context_->GetInputDesc(ARGMAX_INDEX)->GetDataType();
    inputData.isInt32Meet = IsGreaterThanInt32Max(inputData) ? 0 : 1;
}

ge::graphStatus MaxPool3DGradWithArgmaxTilingBaseV35::GetShapeAttrsInfo()
{
    if (!Ops::NN::OpTiling::IsRegbaseSocVersion(context_)) {
        // Skip the current template
        return ge::GRAPH_PARAM_INVALID;
    }

    OP_LOGD(context_->GetNodeName(), "Enter MaxPool3DGradWithArgmaxTilingBaseV35 GetShapeAttrsInfo.");
    const char* opName_ = "MaxPool3DGradWithArgmax";
    if (ge::GRAPH_SUCCESS != CheckInputDtype()) {
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            opName_, "x, grad, argmax", "invalid_dtypes", "The input dtype is invalid.");
        return ge::GRAPH_FAILED;
    }
    if (!CheckInputShape()) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            opName_, "x, grad, argmax", "invalid_shapes", "The input relationship is invalid.");
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != CheckAttrShape()) {
        OP_LOGE_FOR_INVALID_LISTSIZE(opName_, "Length of attr", "invalid_size", "3, 1, or 0");
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != SetInputParams()) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName_, "input", "invalid_shape", "Set input shape failed.");
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != SetAttrParams()) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "attr", "invalid_value", "Set attr shape failed.");
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != CheckInputValid()) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(opName_, "d, h, w", "invalid_values", "The input shape is invalid.");
        return ge::GRAPH_FAILED;
    }
    SetOtherInputParams();
    return ge::GRAPH_SUCCESS;
}

bool MaxPool3DGradWithArgmaxTilingBaseV35::IsCapable() { return false; }

ge::graphStatus MaxPool3DGradWithArgmaxTilingBaseV35::DoOpTiling() { return ge::GRAPH_SUCCESS; }

ge::graphStatus MaxPool3DGradWithArgmaxTilingBaseV35::DoLibApiTiling() { return ge::GRAPH_SUCCESS; }

ge::graphStatus MaxPool3DGradWithArgmaxTilingBaseV35::GetPlatformInfo()
{
    const char* opName_ = "MaxPool3DGradWithArgmax";
    auto platformPtr = context_->GetPlatformInfo();
    if (platformPtr == nullptr) {
        auto compileInfoPtr = static_cast<const Tiling4MaxPool3DGradWithArgmaxCompileInfo*>(context_->GetCompileInfo());
        if (compileInfoPtr == nullptr) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "compileInfo", "null", "compile info is null");
            return ge::GRAPH_FAILED;
        }
        coreNum_ = compileInfoPtr->totalCoreNum;
        ubSize_ = compileInfoPtr->maxUbSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformPtr);
        coreNum_ = ascendcPlatform.GetCoreNumAiv();

        uint64_t ubSizePlatform;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatform);
        ubSize_ = static_cast<int64_t>(ubSizePlatform);
    }

    if (coreNum_ == 0) {
        OP_LOGE_FOR_INVALID_CONFIG_WITH_REASON(opName_, "0", "coreNum", "MaxPool3DGradWithArgmax", "coreNum is 0");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPool3DGradWithArgmaxTilingBaseV35::GetWorkspaceSize()
{
    auto sys_workspace = WS_SYS_SIZE;
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = sys_workspace;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPool3DGradWithArgmaxTilingBaseV35::PostTiling() { return ge::GRAPH_SUCCESS; }

uint64_t MaxPool3DGradWithArgmaxTilingBaseV35::GetTilingKey() const { return 0; }
} // namespace optiling

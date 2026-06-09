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

#include "log/log.h"
#include "error_util.h"
#include "max_pool3d_grad_tiling.h"

#include "platform/platform_info.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include "op_common/op_host/util/platform_util.h"
#include "op_host/tiling_util.h"

namespace optiling {

constexpr size_t CDHW_DIM_NUM = 4U;
constexpr size_t C_DIM_OFFSET = 4; // pos = dim - offset
constexpr size_t D_DIM_OFFSET = 3;
constexpr size_t H_DIM_OFFSET = 2;
constexpr size_t W_DIM_OFFSET = 1;
constexpr size_t D_ATTR_INDEX = 2;
constexpr size_t H_ATTR_INDEX = 3;
constexpr size_t W_ATTR_INDEX = 4;

constexpr uint32_t ORIG_X_INDEX = 0;
constexpr uint32_t ORIG_Y_INDEX = 1;
constexpr uint32_t GRADS_INDEX = 2;
constexpr size_t KSIZE_ATTR_INDEX = 0U;
constexpr size_t STRIDES_ATTR_INDEX = 1U;
constexpr size_t PADDING_ATTR_INDEX = 2U;
constexpr size_t PADS_ATTR_INDEX = 3U;
constexpr size_t DATA_FORMAT_ATTR_INDEX = 4U;

constexpr size_t PADS_SIZE = 6;
constexpr uint32_t Y_INDEX = 0;
constexpr uint64_t NUM_TWO = 2;
constexpr size_t DHW_DIM_NUM = 3;
constexpr uint32_t MAX_BLOCK_COUNT = 4095;

// 参数常量
constexpr size_t NC_DIM_NUM = 2;
constexpr size_t NCDHW_DIM_NUM = 5;

static inline bool IsGreaterThanInt32Max(const Pool3DGradNCDHWInputInfo& inputData)
{
    int64_t cubeSize = inputData.dX * inputData.hX * inputData.wX;
    return cubeSize > static_cast<int64_t>(INT32_MAX);
}

bool MaxPool3DGradTilingBase::CheckInputShape()
{
    const char* opName_ = "MaxPool3DGrad";
    const gert::StorageShape* origXShape = context_->GetInputShape(ORIG_X_INDEX);
    const gert::StorageShape* origYShape = context_->GetInputShape(ORIG_Y_INDEX);
    const gert::StorageShape* gradsShape = context_->GetInputShape(GRADS_INDEX);
    size_t origXDimNum = Ops::Base::EnsureNotScalar(origXShape->GetStorageShape()).GetDimNum();
    size_t origYDimNum = Ops::Base::EnsureNotScalar(origYShape->GetStorageShape()).GetDimNum();
    size_t gradsDimNum = Ops::Base::EnsureNotScalar(gradsShape->GetStorageShape()).GetDimNum();
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    const char* data_format = attrs->GetAttrPointer<char>(DATA_FORMAT_ATTR_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, data_format);
    std::string data_formatStr = data_format;

    if (!(data_formatStr == "NCDHW" || data_formatStr == "NDHWC")) {
        OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(
            opName_, "Attribute data_format", data_format, "expect [NDHWC] or [NCDHW]");
        return false;
    }

    if ((origXDimNum != NCDHW_DIM_NUM) || (origYDimNum != NCDHW_DIM_NUM) || (gradsDimNum != NCDHW_DIM_NUM)) {
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            opName_, "orig_x, orig_y, grads",
            (std::to_string(origXDimNum) + ", " + std::to_string(origYDimNum) + ", " + std::to_string(gradsDimNum))
                .c_str(),
            "Input dim num should equal 5");
        return false;
    }
    for (uint32_t i = 0; i < origXDimNum; i++) {
        if (origXShape->GetStorageShape().GetDim(i) == 0) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                opName_, "orig_x", std::to_string(origXShape->GetStorageShape().GetDim(i)).c_str(),
                "Input x shape dimension can not be 0");
            return false;
        }
    }

    for (size_t i = 0; i < origXDimNum; i++) {
        uint64_t origYDimValue = origYShape->GetStorageShape().GetDim(i);
        uint64_t gradsDimValue = gradsShape->GetStorageShape().GetDim(i);
        if (origYDimValue != gradsDimValue) {
            OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                opName_, "orig_y, grads",
                (std::to_string(origYDimValue) + ", " + std::to_string(gradsDimValue)).c_str(),
                "orig_y and grads dimensions should be equal");
            return false;
        }
    }

    uint32_t cPosIdx = (data_formatStr == "NDHWC") ? origXDimNum - 1 : origXDimNum - 4;
    uint64_t xNDim = (origXDimNum == CDHW_DIM_NUM) ? 1 : origXShape->GetStorageShape().GetDim(0);
    uint64_t gradNDim = (origYDimNum == CDHW_DIM_NUM) ? 1 : origYShape->GetStorageShape().GetDim(0);
    uint64_t xCDim = origXShape->GetStorageShape().GetDim(cPosIdx);
    uint64_t gradCDim = origYShape->GetStorageShape().GetDim(cPosIdx);
    if ((xNDim != gradNDim) || (xCDim != gradCDim)) {
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            opName_, "orig_x, grads",
            (std::to_string(xNDim) + "," + std::to_string(xCDim) + " vs " + std::to_string(gradNDim) + "," +
             std::to_string(gradCDim))
                .c_str(),
            "N and C dimensions of orig_x and grads must be equal");
        return false;
    }

    return true;
}

ge::graphStatus MaxPool3DGradTilingBase::CheckInputDtype()
{
    const char* opName_ = "MaxPool3DGrad";
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(ORIG_X_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(ORIG_Y_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(GRADS_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOutputDesc(Y_INDEX));
    auto xDataType = context_->GetInputDesc(ORIG_X_INDEX)->GetDataType();
    auto origYDataType = context_->GetInputDesc(ORIG_Y_INDEX)->GetDataType();
    auto gradsDataType = context_->GetInputDesc(GRADS_INDEX)->GetDataType();

    if (!(xDataType == origYDataType && origYDataType == gradsDataType)) {
        std::string xDtypeStr = std::to_string(static_cast<int32_t>(xDataType));
        std::string origYDtypeStr = std::to_string(static_cast<int32_t>(origYDataType));
        std::string gradsDtypeStr = std::to_string(static_cast<int32_t>(gradsDataType));
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            opName_, "orig_x, orig_y, grads", (xDtypeStr + ", " + origYDtypeStr + ", " + gradsDtypeStr).c_str(),
            "orig_x, orig_y, grads data type must be same");
        return ge::GRAPH_FAILED;
    }
    if ((xDataType != ge::DT_FLOAT) && (xDataType != ge::DT_FLOAT16) && (xDataType != ge::DT_BF16)) {
        std::string xDtypeStr = std::to_string(static_cast<int32_t>(xDataType));
        OP_LOGE_FOR_INVALID_DTYPE(opName_, "orig_x", xDtypeStr.c_str(), "[DT_FLOAT, DT_FLOAT16, DT_BF16]");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPool3DGradTilingBase::CheckAttrVal()
{
    const char* opName_ = "MaxPool3DGrad";
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    int32_t kSizeDimNum = attrs->GetListInt(KSIZE_ATTR_INDEX)->GetSize();
    int32_t stridesDimNum = attrs->GetListInt(STRIDES_ATTR_INDEX)->GetSize();
    int32_t padsDimNum = attrs->GetListInt(PADS_ATTR_INDEX)->GetSize();
    auto kSizeVector = attrs->GetListInt(KSIZE_ATTR_INDEX)->GetData();
    auto stridesVector = attrs->GetListInt(STRIDES_ATTR_INDEX)->GetData();

    const char* data_format = attrs->GetAttrPointer<char>(DATA_FORMAT_ATTR_INDEX);
    std::string data_formatStr = data_format;

    if (data_formatStr == "NDHWC") {
        if (kSizeVector[0] != 1 || kSizeVector[C_DIM_OFFSET] != 1) {
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                opName_, "kSize[0], kSize[4]",
                (std::to_string(kSizeVector[0]) + ", " + std::to_string(kSizeVector[C_DIM_OFFSET])).c_str(),
                "kSize[0] and kSize[4] should equal 1 in NDHWC format");
            return ge::GRAPH_FAILED;
        }
        if (stridesVector[0] != 1 || stridesVector[C_DIM_OFFSET] != 1) {
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                opName_, "strides[0], strides[4]",
                (std::to_string(stridesVector[0]) + ", " + std::to_string(stridesVector[C_DIM_OFFSET])).c_str(),
                "strides[0] and strides[4] should equal 1 in NDHWC format");
            return ge::GRAPH_FAILED;
        }
    } else {
        if (kSizeVector[0] != 1 || kSizeVector[1] != 1) {
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                opName_, "kSize[0], kSize[1]",
                (std::to_string(kSizeVector[0]) + ", " + std::to_string(kSizeVector[1])).c_str(),
                "kSize[0] and kSize[1] should equal 1 in NCDHW format");
            return ge::GRAPH_FAILED;
        }
        if (stridesVector[0] != 1 || stridesVector[1] != 1) {
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                opName_, "strides[0], strides[1]",
                (std::to_string(stridesVector[0]) + ", " + std::to_string(stridesVector[1])).c_str(),
                "strides[0] and strides[1] should equal 1 in NCDHW format");
            return ge::GRAPH_FAILED;
        }
    }
    for (uint32_t i = 0; i < static_cast<uint32_t>(kSizeDimNum); i++) {
        if (kSizeVector[i] <= 0) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                opName_, "kSize", std::to_string(kSizeVector[i]).c_str(), "kSize value should be bigger than 0");
            return ge::GRAPH_FAILED;
        }
    }
    for (uint32_t i = 0; i < static_cast<uint32_t>(stridesDimNum); i++) {
        if (stridesVector[i] <= 0) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                opName_, "strides", std::to_string(stridesVector[i]).c_str(), "strides value should be bigger than 0");
            return ge::GRAPH_FAILED;
        }
    }
    auto padsVector = attrs->GetListInt(PADS_ATTR_INDEX)->GetData();
    for (uint32_t i = 0; i < static_cast<uint32_t>(padsDimNum); i++) {
        if (padsVector[i] < 0) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                opName_, "pads", std::to_string(padsVector[i]).c_str(), "pads value should be bigger or equal 0");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPool3DGradTilingBase::CheckAttrShape()
{
    const char* opName_ = "MaxPool3DGrad";
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    int32_t kSizeDimNum = attrs->GetListInt(KSIZE_ATTR_INDEX)->GetSize();
    int32_t stridesDimNum = attrs->GetListInt(STRIDES_ATTR_INDEX)->GetSize();
    int32_t padsDimNum = attrs->GetListInt(PADS_ATTR_INDEX)->GetSize();

    const char* padMode = attrs->GetAttrPointer<char>(PADDING_ATTR_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, padMode);
    std::string padModeStr = padMode;
    if (IsInvalidPaddingModeWithCalculated(padModeStr)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            opName_, "padMode", padMode, "only support VALID, SAME or CALCULATED padding mode");
        return ge::GRAPH_FAILED;
    }

    if (kSizeDimNum != NCDHW_DIM_NUM) {
        OP_LOGE_FOR_INVALID_LISTSIZE(opName_, "Length of kSize", std::to_string(kSizeDimNum).c_str(), "5");
        return ge::GRAPH_FAILED;
    }
    if (stridesDimNum != NCDHW_DIM_NUM) {
        OP_LOGE_FOR_INVALID_LISTSIZE(opName_, "Length of strides", std::to_string(stridesDimNum).c_str(), "5");
        return ge::GRAPH_FAILED;
    }
    if (padsDimNum != PADS_SIZE) {
        OP_LOGE_FOR_INVALID_LISTSIZE(opName_, "Length of pads", std::to_string(padsDimNum).c_str(), "6");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPool3DGradTilingBase::SetInputParams()
{
    const gert::Shape xShape = context_->GetInputShape(ORIG_X_INDEX)->GetStorageShape();
    const gert::Shape gradShape = context_->GetInputShape(GRADS_INDEX)->GetStorageShape();
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

ge::graphStatus MaxPool3DGradTilingBase::SetAttrParams()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    int32_t kSizeDimNum = attrs->GetListInt(KSIZE_ATTR_INDEX)->GetSize();
    int32_t stridesDimNum = attrs->GetListInt(STRIDES_ATTR_INDEX)->GetSize();
    auto kSizeVector = attrs->GetListInt(KSIZE_ATTR_INDEX)->GetData();
    auto stridesVector = attrs->GetListInt(STRIDES_ATTR_INDEX)->GetData();
    size_t dAttrIdx = D_ATTR_INDEX;
    size_t hAttrIdx = H_ATTR_INDEX;
    size_t wAttrIdx = W_ATTR_INDEX;
    if (inputData.inputFormat == ge::Format::FORMAT_NDHWC) {
        dAttrIdx = D_ATTR_INDEX - 1;
        hAttrIdx = H_ATTR_INDEX - 1;
        wAttrIdx = W_ATTR_INDEX - 1;
    }
    inputData.dKernel = kSizeVector[dAttrIdx];
    inputData.hKernel = (kSizeDimNum == 1) ? inputData.dKernel : kSizeVector[hAttrIdx];
    inputData.wKernel = (kSizeDimNum == 1) ? inputData.dKernel : kSizeVector[wAttrIdx];
    if (stridesDimNum == 0) {
        inputData.dStride = inputData.dKernel;
        inputData.hStride = inputData.hKernel;
        inputData.wStride = inputData.wKernel;
    } else {
        inputData.dStride = stridesVector[dAttrIdx];
        inputData.hStride = (stridesDimNum == 1) ? inputData.dStride : stridesVector[hAttrIdx];
        inputData.wStride = (stridesDimNum == 1) ? inputData.dStride : stridesVector[wAttrIdx];
    }

    const char* padMode = attrs->GetAttrPointer<char>(PADDING_ATTR_INDEX);
    std::string padModeStr = padMode;
    if (padModeStr == "VALID") {
        inputData.dPad = 0;
        inputData.hPad = 0;
        inputData.wPad = 0;
    } else if (padModeStr == "SAME") {
        int64_t dPadNeed =
            std::max(int64_t{0}, (inputData.dGrad - 1) * inputData.dStride + inputData.dKernel - inputData.dX);
        inputData.dPad = dPadNeed / DIGIT_TWO;
        int64_t hPadNeed =
            std::max(int64_t{0}, (inputData.hGrad - 1) * inputData.hStride + inputData.hKernel - inputData.hX);
        inputData.hPad = hPadNeed / DIGIT_TWO;
        int64_t wPadNeed =
            std::max(int64_t{0}, (inputData.wGrad - 1) * inputData.wStride + inputData.wKernel - inputData.wX);
        inputData.wPad = wPadNeed / DIGIT_TWO;
    } else if (padModeStr == "CALCULATED") {
        auto padsVector = attrs->GetListInt(PADS_ATTR_INDEX)->GetData();
        inputData.dPad = padsVector[0];
        inputData.hPad = padsVector[2];
        inputData.wPad = padsVector[4];
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPool3DGradTilingBase::CheckInputValid()
{
    const char* opName_ = "MaxPool3DGrad";
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
            opName_, "dPad, hPad, wPad",
            (std::to_string(pDTop) + ", " + std::to_string(pHTop) + ", " + std::to_string(pWTop)).c_str(),
            "padSize should be smaller than kernelSize / 2");
        return ge::GRAPH_FAILED;
    }
    if ((pDTop > ((kd - 1) * dilationD + 1) / 2) || (pHTop > ((kh - 1) * dilationH + 1) / 2) ||
        (pWTop > ((kw - 1) * dilationW + 1) / 2)) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            opName_, "dPad, hPad, wPad",
            (std::to_string(pDTop) + ", " + std::to_string(pHTop) + ", " + std::to_string(pWTop)).c_str(),
            "padSize should be smaller than ((kernelSize - 1) * dilation + 1) / 2");
        return ge::GRAPH_FAILED;
    }

    auto attrs = context_->GetAttrs();
    const char* padMode = attrs->GetAttrPointer<char>(PADDING_ATTR_INDEX);
    std::string padModeStr = padMode;
    int64_t doExpected = 0;
    int64_t hoExpected = 0;
    int64_t woExpected = 0;
    if (padModeStr == "VALID") {
        doExpected = (inputData.dX - inputData.dKernel + inputData.dStride) / inputData.dStride;
        hoExpected = (inputData.hX - inputData.hKernel + inputData.hStride) / inputData.hStride;
        woExpected = (inputData.wX - inputData.wKernel + inputData.wStride) / inputData.wStride;
    } else if (padModeStr == "SAME") {
        doExpected = (inputData.dX + inputData.dStride - 1) / inputData.dStride;
        hoExpected = (inputData.hX + inputData.hStride - 1) / inputData.hStride;
        woExpected = (inputData.wX + inputData.wStride - 1) / inputData.wStride;
    } else if (padModeStr == "CALCULATED") {
        doExpected = (inputData.dX + inputData.dPad * 2 - inputData.dDilation * (inputData.dKernel - 1) - 1) /
                         inputData.dStride +
                     1;
        hoExpected = (inputData.hX + inputData.hPad * 2 - inputData.hDilation * (inputData.hKernel - 1) - 1) /
                         inputData.hStride +
                     1;
        woExpected = (inputData.wX + inputData.wPad * 2 - inputData.wDilation * (inputData.wKernel - 1) - 1) /
                         inputData.wStride +
                     1;
    }

    if ((doExpected <= 0) || (doExpected != inputData.dGrad) || (hoExpected <= 0) || (hoExpected != inputData.hGrad) ||
        (woExpected <= 0) || (woExpected != inputData.wGrad)) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            opName_, "do, ho, wo",
            (std::to_string(doExpected) + ", " + std::to_string(hoExpected) + ", " + std::to_string(woExpected))
                .c_str(),
            "OuterDim size must be positive and match grads dimensions");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

void MaxPool3DGradTilingBase::SetOtherInputParams()
{
    inputData.inputDtype = context_->GetInputDesc(ORIG_X_INDEX)->GetDataType();
    inputData.isInt32Meet = IsGreaterThanInt32Max(inputData) ? 1 : 0;
}

ge::graphStatus MaxPool3DGradTilingBase::GetShapeAttrsInfo()
{
    const char* opName_ = "MaxPool3DGrad";
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    auto socVersion = ascendcPlatform.GetSocVersion();
    if (!Ops::NN::OpTiling::IsRegbaseSocVersion(context_)) {
        return ge::GRAPH_PARAM_INVALID;
    }

    OP_LOGD(context_->GetNodeName(), "Enter MaxPool3DGradTilingBase GetShapeAttrsInfo.");
    if (ge::GRAPH_SUCCESS != CheckInputDtype()) {
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            opName_, "orig_x, orig_y, grads", "invalid_dtypes", "The input dtype is invalid");
        return ge::GRAPH_FAILED;
    }
    if (!CheckInputShape()) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            opName_, "orig_x, orig_y, grads", "invalid_shapes", "The input relationship is invalid");
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != CheckAttrShape()) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "attr", "invalid_value", "The attr shape is invalid");
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != CheckAttrVal()) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "attr", "invalid_value", "The attr value is invalid");
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != SetInputParams()) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName_, "input", "invalid_shape", "Set input shape failed");
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != SetAttrParams()) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "attr", "invalid_value", "Set attr params failed");
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != CheckInputValid()) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(opName_, "input", "invalid_shapes", "The input shape is invalid");
        return ge::GRAPH_FAILED;
    }
    SetOtherInputParams();
    return ge::GRAPH_SUCCESS;
}

bool MaxPool3DGradTilingBase::IsCapable() { return false; }

ge::graphStatus MaxPool3DGradTilingBase::DoOpTiling() { return ge::GRAPH_SUCCESS; }

ge::graphStatus MaxPool3DGradTilingBase::DoLibApiTiling() { return ge::GRAPH_SUCCESS; }

ge::graphStatus MaxPool3DGradTilingBase::GetPlatformInfo()
{
    auto platformPtr = context_->GetPlatformInfo();
    if (platformPtr == nullptr) {
        auto compileInfoPtr = static_cast<const Tiling4Pool3DGradCompileInfo*>(context_->GetCompileInfo());
        OP_CHECK_IF(
            compileInfoPtr == nullptr, OP_LOGE(context_->GetNodeName(), "compile info is null"),
            return ge::GRAPH_FAILED);
        coreNum_ = compileInfoPtr->totalCoreNum;
        ubSize_ = compileInfoPtr->maxUbSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformPtr);
        coreNum_ = ascendcPlatform.GetCoreNumAiv();

        uint64_t ubSizePlatform;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatform);
        ubSize_ = static_cast<int64_t>(ubSizePlatform);
    }

    OP_CHECK_IF(coreNum_ == 0, OP_LOGE(context_->GetNodeName(), "coreNum is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPool3DGradTilingBase::GetWorkspaceSize()
{
    auto sys_workspace = WS_SYS_SIZE;
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = sys_workspace;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPool3DGradTilingBase::PostTiling() { return ge::GRAPH_SUCCESS; }

uint64_t MaxPool3DGradTilingBase::GetTilingKey() const { return 0; }
} // namespace optiling

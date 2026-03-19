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
#include "max_pool3d_grad_tiling.h"

#include "platform/platform_info.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include "op_common/op_host/util/platform_util.h"

namespace optiling {

constexpr size_t CDHW_DIM_NUM = 4U;
constexpr size_t C_DIM_OFFSET = 4;  // pos = dim - offset
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

    // data_format should be NCDHW
    OP_CHECK_IF(!(data_formatStr == "NCDHW" || data_formatStr == "NDHWC"),
                OP_LOGE(context_->GetNodeName(), "ATTR data_format is %s ,expect [NDHWC] or [NCDHW].", data_format),
                return false);

    // origXDimNum should be 5
    OP_CHECK_IF(((origXDimNum != NCDHW_DIM_NUM) || (origYDimNum != NCDHW_DIM_NUM) || (gradsDimNum != NCDHW_DIM_NUM)),
                OP_LOGE(context_->GetNodeName(),
                        "Input dim num should equal = %lu or %lu, actual is xDim: %lu, gradDim: %lu, argmaxDim: %lu.",
                        NCDHW_DIM_NUM, CDHW_DIM_NUM, origXDimNum, origYDimNum, gradsDimNum),
                return false);
    for (uint32_t i = 0; i < origXDimNum; i++) {
        OP_CHECK_IF(origXShape->GetStorageShape().GetDim(i) == 0,
                    OP_LOGE(context_->GetNodeName(), "Input x shape can not be 0."), return false);
    }

    // origYShape&gradsShape's shape should be equal
    for (size_t i = 0; i < origXDimNum; i++) {
        uint64_t origYDimValue = origYShape->GetStorageShape().GetDim(i);
        uint64_t gradsDimValue = gradsShape->GetStorageShape().GetDim(i);
        OP_CHECK_IF(origYDimValue != gradsDimValue,
                    OP_LOGE(context_->GetNodeName(),
                            "Input dim check invalid, orig_y[%lu] is %lu, grads[%lu] is %lu, not equal.", i,
                            origYDimValue, i, gradsDimValue),
                    return false);
    }

    // Input NCDim should be equal
    uint32_t cPosIdx = (data_formatStr == "NDHWC") ? origXDimNum - 1 : origXDimNum - 4;
    uint64_t xNDim = (origXDimNum == CDHW_DIM_NUM) ? 1 : origXShape->GetStorageShape().GetDim(0);
    uint64_t gradNDim = (origYDimNum == CDHW_DIM_NUM) ? 1 : origYShape->GetStorageShape().GetDim(0);
    uint64_t xCDim = origXShape->GetStorageShape().GetDim(cPosIdx);
    uint64_t gradCDim = origYShape->GetStorageShape().GetDim(cPosIdx);
    OP_CHECK_IF((xNDim != gradNDim) || (xCDim != gradCDim),
                OP_LOGE(context_->GetNodeName(), "Input N,C dim check invalid, grad(%lu,%lu), x(%lu,%lu), not equal.",
                        gradNDim, gradCDim, xNDim, xCDim),
                return false);

    return true;
}

ge::graphStatus MaxPool3DGradTilingBase::CheckInputDtype()
{
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(ORIG_X_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(ORIG_Y_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(GRADS_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOutputDesc(Y_INDEX));
    auto xDataType = context_->GetInputDesc(ORIG_X_INDEX)->GetDataType();
    auto origYDataType = context_->GetInputDesc(ORIG_Y_INDEX)->GetDataType();
    auto gradsDataType = context_->GetInputDesc(GRADS_INDEX)->GetDataType();

    OP_CHECK_IF(!(xDataType == origYDataType && origYDataType == gradsDataType),
                OP_LOGE(context_->GetNodeName(), "Data type invalid, orig_x, orig_y, grads data type not same."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((xDataType != ge::DT_FLOAT) && (xDataType != ge::DT_FLOAT16) && (xDataType != ge::DT_BF16),
                OP_LOGE(context_->GetNodeName(), "Data type invalid, x data type not fp32/fp16/bf16."),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPool3DGradTilingBase::CheckAttrVal()
{
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
        OP_CHECK_IF((kSizeVector[0] != 1 || kSizeVector[C_DIM_OFFSET] != 1),
                OP_LOGE(context_->GetNodeName(), "Attr value invalid, kSize[%u]-kSize[%u] is %ld-%ld, should equle to 1.",
                        0, C_DIM_OFFSET, kSizeVector[0], kSizeVector[C_DIM_OFFSET]),
                return ge::GRAPH_FAILED);
        OP_CHECK_IF((stridesVector[0] != 1 || stridesVector[C_DIM_OFFSET] != 1),
                OP_LOGE(context_->GetNodeName(), "Attr value invalid, strides[%u]-strides[%u] is %ld-%ld, should equle to 1.",
                        0, C_DIM_OFFSET, stridesVector[0], stridesVector[C_DIM_OFFSET]),
                return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF((kSizeVector[0] != 1 || kSizeVector[1] != 1),
            OP_LOGE(context_->GetNodeName(), "Attr value invalid, kSize[%u]-kSize[%u] is %ld-%ld, should equle to 1.",
                    0, 1, kSizeVector[0], kSizeVector[1]),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF((stridesVector[0] != 1 || stridesVector[1] != 1),
            OP_LOGE(context_->GetNodeName(), "Attr value invalid, strides[%u]-strides[%u] is %ld-%ld, should equle to 1.",
                    0, 1, stridesVector[0], stridesVector[1]),
            return ge::GRAPH_FAILED);
    }
    for (uint32_t i = 0; i < static_cast<uint32_t>(kSizeDimNum); i++) {
        OP_CHECK_IF((kSizeVector[i] <= 0),
                    OP_LOGE(context_->GetNodeName(), "Attr value invalid, kSize[%u] is %ld, should bigger than 0.",
                            i, kSizeVector[i]),
                    return ge::GRAPH_FAILED);
    }
    for (uint32_t i = 0; i < static_cast<uint32_t>(stridesDimNum); i++) {
        OP_CHECK_IF((stridesVector[i] <= 0),
                    OP_LOGE(context_->GetNodeName(), "Attr value invalid, strides[%u] is %ld, should bigger than 0.",
                            i, stridesVector[i]),
                    return ge::GRAPH_FAILED);
    }
    auto padsVector = attrs->GetListInt(PADS_ATTR_INDEX)->GetData();
    for (uint32_t i = 0; i < static_cast<uint32_t>(padsDimNum); i++) {
        OP_CHECK_IF((padsVector[i] < 0),
                    OP_LOGE(context_->GetNodeName(), "Attr value invalid, pads[%u] is %ld, should bigger or equal 0.",
                            i, padsVector[i]),
                    return ge::GRAPH_FAILED);
    }
}

ge::graphStatus MaxPool3DGradTilingBase::CheckAttrShape()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    int32_t kSizeDimNum = attrs->GetListInt(KSIZE_ATTR_INDEX)->GetSize();
    int32_t stridesDimNum = attrs->GetListInt(STRIDES_ATTR_INDEX)->GetSize();
    int32_t padsDimNum = attrs->GetListInt(PADS_ATTR_INDEX)->GetSize();

    const char* padMode = attrs->GetAttrPointer<char>(PADDING_ATTR_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, padMode);
        std::string padModeStr = padMode;
        OP_TILING_CHECK(
            IsInvalidPaddingModeWithCalculated(padModeStr),
            VECTOR_INNER_ERR_REPORT_TILIING(context_, "MaxPool3DGradTilingBase: not support padmode %s", padModeStr.c_str()),
            return ge::GRAPH_FAILED);

    // Check attr dim num
    OP_CHECK_IF((kSizeDimNum != NCDHW_DIM_NUM),
                OP_LOGE(context_->GetNodeName(), "Attr kSize dim num invalid, dim num should equal 5."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((stridesDimNum != NCDHW_DIM_NUM),
                OP_LOGE(context_->GetNodeName(), "Attr strides dim num invalid, dim num should equal 5."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(padsDimNum != PADS_SIZE,
                OP_LOGE(context_->GetNodeName(), "Attr pads dim num invalid, dim num should equal 6."),
                return ge::GRAPH_FAILED);

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

    const char* padMode = attrs->GetAttrPointer<char>(PADDING_ATTR_INDEX);
    std::string padModeStr = padMode;
    if (padModeStr == "VALID") {
        inputData.dPad = 0;
        inputData.hPad = 0;
        inputData.wPad = 0;
    } else if (padModeStr == "SAME") {
        int64_t dPadNeed = std::max(int64_t{0}, (inputData.dGrad - 1) * inputData.dStride +
                                                inputData.dKernel - inputData.dX);
        inputData.dPad = dPadNeed / DIGIT_TWO;
        int64_t hPadNeed = std::max(int64_t{0}, (inputData.hGrad - 1) * inputData.hStride +
                                                inputData.hKernel - inputData.hX);
        inputData.hPad = hPadNeed / DIGIT_TWO;
        int64_t wPadNeed = std::max(int64_t{0}, (inputData.wGrad - 1) * inputData.wStride +
                                                inputData.wKernel - inputData.wX);
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

    // check 1
    OP_CHECK_IF((pDTop > (kd / 2)) || (pHTop > (kh / 2)) || (pWTop > (kw / 2)),
                OP_LOGE(context_->GetNodeName(), "Attr size invalid, padSize should smaller than kernelSize div 2"),
                return ge::GRAPH_FAILED);
    // check 2
    OP_CHECK_IF((pDTop > ((kd - 1) * dilationD + 1) / 2) || (pHTop > ((kh - 1) * dilationH + 1) / 2) ||
                    (pWTop > ((kw - 1) * dilationW + 1) / 2),
                OP_LOGE(context_->GetNodeName(),
                        "Attr size invalid, padSize should smaller than ((kernelSize - 1) * dilation + 1) / 2."),
                return ge::GRAPH_FAILED);
    // check 3
    // Check outerDim invalid
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
      doExpected = (inputData.dX + inputData.dStride -1) / inputData.dStride;
      hoExpected = (inputData.hX + inputData.hStride -1) / inputData.hStride;
      woExpected = (inputData.wX + inputData.wStride -1) / inputData.wStride;
    }

    OP_CHECK_IF(
        (doExpected <= 0) || (doExpected != inputData.dGrad) || (hoExpected <= 0) || (hoExpected != inputData.hGrad) ||
            (woExpected <= 0) || (woExpected != inputData.wGrad),
        OP_LOGE(context_->GetNodeName(), "OuterDim size invalid, doExpected: %ld, hoExpected: %ld, woExpected: %ld.",
                doExpected, hoExpected, woExpected),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

void MaxPool3DGradTilingBase::SetOtherInputParams()
{
    inputData.inputDtype = context_->GetInputDesc(ORIG_X_INDEX)->GetDataType();
    inputData.isInt32Meet = IsGreaterThanInt32Max(inputData) ? 1 : 0;
}

ge::graphStatus MaxPool3DGradTilingBase::GetShapeAttrsInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    auto socVersion = ascendcPlatform.GetSocVersion();
    if (socVersion != platform_ascendc::SocVersion::ASCEND950) {
        // Skip the current template
        return ge::GRAPH_PARAM_INVALID;
    }

    OP_LOGD(context_->GetNodeName(), "Enter MaxPool3DGradTilingBase GetShapeAttrsInfo.");
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

bool MaxPool3DGradTilingBase::IsCapable()
{
    return false;
}

ge::graphStatus MaxPool3DGradTilingBase::DoOpTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPool3DGradTilingBase::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MaxPool3DGradTilingBase::GetPlatformInfo()
{
    auto platformPtr = context_->GetPlatformInfo();
    if (platformPtr == nullptr) {
        auto compileInfoPtr =
            static_cast<const Tiling4Pool3DGradCompileInfo*>(context_->GetCompileInfo());
        OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context_->GetNodeName(), "compile info is null"),
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

ge::graphStatus MaxPool3DGradTilingBase::PostTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t MaxPool3DGradTilingBase::GetTilingKey() const
{
    return 0;
}
}  // namespace optiling

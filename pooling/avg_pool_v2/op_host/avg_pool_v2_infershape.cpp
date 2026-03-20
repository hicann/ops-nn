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
 * \file avg_pool_v2_infershape.cpp
 * \brief
 */
#include "pooling/avg_pool/op_host/avg_pool_infershape_common.h"

template <class T>
bool CheckPoolingPadsPositive(const std::string& opName,
                              const std::string& paddingModeStr,
                              T& inputs)
{
    int64_t pad_needed_h = 0;
    int64_t pad_needed_w = 0;
    if (paddingModeStr == "SAME") {
        pad_needed_h = (inputs.outH - 1) * inputs.strideH + inputs.windowH - inputs.inH;
        pad_needed_w = (inputs.outW - 1) * inputs.strideW + inputs.windowW - inputs.inW;
        if (pad_needed_h < 0  or pad_needed_w < 0) {
            std::string err_msg = OtherErrMsg("pad_needed should be positive");
            OP_LOGE(opName.c_str(), "%s", err_msg.c_str());
            return false;
        }
    }
    return true;
}
namespace ops {
// proto input
const size_t X_IDX_AVGPOOLV2 = 0;
// proto output
const size_t OUT_IDX_AVGPOOLV2 = 0;
// proto attributes
const size_t KSIZE_IDX_AVGPOOLV2 = 0;
const size_t STRIDES_IDX_AVGPOOLV2 = 1;
const size_t PADDING_MODE_IDX_AVGPOOLV2 = 2;
const size_t PADS_IDX_AVGPOOLV2 = 3;
const size_t DATA_FORMAT_IDX_AVGPOOLV2 = 4;
const size_t GLOBAL_POOLING_IDX_AVGPOOLV2 = 5;
const size_t CEIL_MODE_IDX_AVGPOOLV2 = 6;
const size_t EXCLUSIVE_IDX_AVGPOOLV2 = 7;
// support information
const size_t DIM_SIZE4 = 4;
const size_t PAD_SIZE_LIMIT = 4;
// pad index
const size_t TOP_IDX_PAD = 0;
const size_t BOTTOM_IDX_PAD = 1;
const size_t LEFT_IDX_PAD = 2;
const size_t RIGHT_IDX_PAD = 3;

struct AvgPoolV2Inputs {
    // fmap index 4D
    int64_t xnPosition = 0;
    int64_t xcPosition = 0;
    int64_t xhPosition = 0;
    int64_t xwPosition = 0;
    // shape of fmap 4D
    int64_t inN = -1;
    int64_t inC = -1;
    int64_t inH = -1;
    int64_t inW = -1;
    // shape of output 4D
    int64_t outN = 0;
    int64_t outC = 0;
    int64_t outH = 0;
    int64_t outW = 0;
    // shape of stride
    int32_t strideH = 0;
    int32_t strideW = 0;
    // shape of window
    int32_t windowH = 0;
    int32_t windowW = 0;
    // pad size
    int64_t padtop = 0;
    int64_t padbottom = 0;
    int64_t padleft = 0;
    int64_t padright = 0;
};

static graphStatus GetAvgPoolV2XShape(const gert::CompileTimeTensorDesc* tensorDescIn,
                                      const std::string& opName,
                                      const InferShapeContext* context,
                                      const char* dataFormatArray,
                                      AvgPoolV2Inputs& inputs)
{
    // Get fmap Dim
    const Format xdataFormat = tensorDescIn->GetOriginFormat();
    std::string dataFormat = format2str.at(xdataFormat);
    if ((dataFormat != "NCHW" && dataFormat != "NHWC") and (dataFormat != dataFormatArray)) {
        std::string errMsg = OtherErrMsg("attr data format is wrong.");
        OP_LOGE(opName.c_str(), "%s", errMsg.c_str());
        return GRAPH_FAILED;
    }

    if (dataFormat.length() != DIM_SIZE4) {
        std::string errMsg =
            GetAttrValueErrMsg("Input format dim", std::to_string(dataFormat.length()), ConcatString(DIM_SIZE4));
        OP_LOGE(opName.c_str(), "%s", errMsg.c_str());
        return GRAPH_FAILED;
    }

    bool get_dim_in_format = GetDimInFormat(opName, dataFormat, "N", inputs.xnPosition) &&
                             GetDimInFormat(opName, dataFormat, "C", inputs.xcPosition) &&
                             GetDimInFormat(opName, dataFormat, "H", inputs.xhPosition) &&
                             GetDimInFormat(opName, dataFormat, "W", inputs.xwPosition);
    if (!get_dim_in_format) {
        return GRAPH_FAILED;
    }

    const gert::Shape* shapeIn = context->GetInputShape(X_IDX_AVGPOOLV2);
    OP_LOGE_IF(shapeIn == nullptr, GRAPH_FAILED, opName, "fmap is null.");
    OP_LOGE_IF(shapeIn->GetDimNum() != SUPPORTED_DIM_NUM, GRAPH_FAILED, opName,
        "Not support input xShape dimnum %lu.", shapeIn->GetDimNum());

    // Set x shape into structure
    inputs.inN = shapeIn->GetDim(inputs.xnPosition);
    inputs.inC = shapeIn->GetDim(inputs.xcPosition);
    inputs.inH = shapeIn->GetDim(inputs.xhPosition);
    inputs.inW = shapeIn->GetDim(inputs.xwPosition);

    OP_LOGE_IF(inputs.inC < 1, GRAPH_FAILED, opName,
        "x channel should be greater than or equal to 1. actual is: %ld", inputs.inC);

    return GRAPH_SUCCESS;
}

static graphStatus SetAvgPoolV2Output(InferShapeContext* context, AvgPoolV2Inputs& inputs, const std::string& opName)
{
    const gert::CompileTimeTensorDesc* tensordescOutput = context->GetOutputDesc(OUT_IDX_AVGPOOLV2);
    OP_LOGE_IF(tensordescOutput == nullptr, GRAPH_FAILED, opName, "Get output failed.");
    auto formatOut = tensordescOutput->GetOriginFormat();
    auto shapeOut = context->GetOutputShape(OUT_IDX_AVGPOOLV2);
    OP_LOGE_IF(shapeOut == nullptr, GRAPH_FAILED, opName, "Get shapeOut failed.");
    shapeOut->SetDimNum(SUPPORTED_DIM_NUM);
    // NC1HWC0(NCHW/NHWC)
    if (formatOut == Format::FORMAT_NCHW || formatOut == Format::FORMAT_NHWC) {
        shapeOut->SetDim(inputs.xnPosition, inputs.outN);
        shapeOut->SetDim(inputs.xcPosition, inputs.outC);
        shapeOut->SetDim(inputs.xhPosition, inputs.outH);
        shapeOut->SetDim(inputs.xwPosition, inputs.outW);
    } else {
        OP_LOGE(opName.c_str(), "%s", "output y format is not correct! format should be NCHW or NHWC.");
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

static graphStatus InferShapeForAvgPoolV2(InferShapeContext* context)
{
    const std::string opName = context->GetNodeName();
    OP_LOGD(opName, "Enter avgpoolv2 shape infer.");

    AvgPoolV2Inputs inputs;

    // Define runtime attrs of avgpoolv2
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OP_LOGE_IF(attrs == nullptr, GRAPH_FAILED, opName, "Get attrs failed.");

    // Get input desc
    const gert::CompileTimeTensorDesc* tensorDescIn = context->GetInputDesc(X_IDX_AVGPOOLV2);
    OP_LOGE_IF(tensorDescIn == nullptr, GRAPH_FAILED, opName, "Get fmap failed.");

    // Get output desc
    const gert::CompileTimeTensorDesc* tensorDescOutput = context->GetOutputDesc(OUT_IDX_AVGPOOLV2);
    OP_LOGE_IF(tensorDescOutput == nullptr, GRAPH_FAILED, opName, "Get output failed.");

    // Get input ksize
    const gert::ContinuousVector* ksizePtr = attrs->GetAttrPointer<gert::ContinuousVector>(KSIZE_IDX_AVGPOOLV2);
    OP_LOGE_IF(ksizePtr == nullptr, GRAPH_FAILED, opName, "Get ksize failed.");
    const int64_t* ksizeArray = static_cast<const int64_t*>(ksizePtr->GetData());
    OP_LOGE_IF(ksizeArray == nullptr, GRAPH_FAILED, opName, "ksize is null");

    if (ksizePtr->GetSize() != DIM_SIZE4) {
        std::string errMsg =
            OtherErrMsg(GetAttrValueErrMsg("ksizePtr", std::to_string(ksizePtr->GetSize()), ConcatString(DIM_SIZE4)));
        OP_LOGE(opName.c_str(), "%s", errMsg.c_str());
        return GRAPH_FAILED;
    }

    // Get input strides
    const gert::ContinuousVector* stridesPtr = attrs->GetAttrPointer<gert::ContinuousVector>(STRIDES_IDX_AVGPOOLV2);
    OP_LOGE_IF(stridesPtr == nullptr, GRAPH_FAILED, opName, "Get strides failed.");
    const int64_t* stridesArray = static_cast<const int64_t*>(stridesPtr->GetData());
    OP_LOGE_IF(stridesArray == nullptr, GRAPH_FAILED, opName, "Stride is null.");

    if (stridesPtr->GetSize() != DIM_SIZE4) {
        std::string errMsg =
            GetAttrValueErrMsg("stridesPtr", std::to_string(stridesPtr->GetSize()), ConcatString(DIM_SIZE4));
        OP_LOGE(opName.c_str(), "%s", errMsg.c_str());
        return GRAPH_FAILED;
    }

    // Get input data_format
    const gert::ContinuousVector* dataFormatPtr =
        attrs->GetAttrPointer<gert::ContinuousVector>(DATA_FORMAT_IDX_AVGPOOLV2);
    OP_LOGE_IF(dataFormatPtr == nullptr, GRAPH_FAILED, opName, "Get data format failed.");
    const char* dataFormatArray = static_cast<const char*>(dataFormatPtr->GetData());
    OP_LOGE_IF(dataFormatArray == nullptr, GRAPH_FAILED, opName, "data format is null.");

    // Get input padding mode
    const char* paddingModePtr = attrs->GetAttrPointer<char>(PADDING_MODE_IDX_AVGPOOLV2);
    std::string paddingModeStr = paddingModePtr == nullptr ? "NULL" : paddingModePtr;

    // Get input pads
    const gert::ContinuousVector* padsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(PADS_IDX_AVGPOOLV2);
    OP_LOGE_IF(padsPtr == nullptr, GRAPH_FAILED, opName, "Get pads failed.");
    const int64_t* padsArray = static_cast<const int64_t*>(padsPtr->GetData());
    OP_LOGE_IF(padsArray == nullptr, GRAPH_FAILED, opName, "pads is null.");

    auto pSize = padsPtr->GetSize();
    OP_LOGE_IF(pSize != PAD_SIZE_LIMIT, GRAPH_FAILED, context->GetNodeName(),
        "pads list should be 4D, actual is: %ld.", pSize);

    // Set pads into structure-[AvgPoolV2Inputs]
    inputs.padtop = padsArray[TOP_IDX_PAD];
    inputs.padbottom = padsArray[BOTTOM_IDX_PAD];
    inputs.padleft = padsArray[LEFT_IDX_PAD];
    inputs.padright = padsArray[RIGHT_IDX_PAD];

    // Get input global padding
    const bool* globalPoolingPtr = attrs->GetAttrPointer<bool>(GLOBAL_POOLING_IDX_AVGPOOLV2);
    OP_LOGE_IF(globalPoolingPtr == nullptr, GRAPH_FAILED, opName, "Get global pooling failed.");

    // Get input ceil_mode
    const bool* ceilModeFlag = attrs->GetAttrPointer<bool>(CEIL_MODE_IDX_AVGPOOLV2);
    OP_LOGE_IF(ceilModeFlag == nullptr, GRAPH_FAILED, opName, "Get ceil mode failed.");

    // Get input shape of avgpoolv2
    if (GetAvgPoolV2XShape(tensorDescIn, opName, context, dataFormatArray, inputs) != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }

    if (*globalPoolingPtr) {
        inputs.windowH = static_cast<int32_t>(inputs.inH);
        inputs.windowW = static_cast<int32_t>(inputs.inW);
        inputs.strideH = static_cast<int32_t>(inputs.inH);
        inputs.strideW = static_cast<int32_t>(inputs.inW);
        inputs.padtop = 0;
        inputs.padbottom = 0;
        inputs.padleft = 0;
        inputs.padright = 0;
    } else {
        inputs.windowH = ksizeArray[inputs.xhPosition];
        inputs.windowW = ksizeArray[inputs.xwPosition];
        inputs.strideH = stridesArray[inputs.xhPosition];
        inputs.strideW = stridesArray[inputs.xwPosition];
        OP_CHECK_IF(inputs.windowH <= 0 || inputs.windowW <= 0,
            OP_LOGE(opName.c_str(), "%s", "ksize should be more than 0."), return GRAPH_FAILED);
        OP_CHECK_IF(inputs.strideH <= 0 || inputs.strideW <= 0,
            OP_LOGE(opName.c_str(), "%s", "stride should be more than 0."), return GRAPH_FAILED);
    }
    inputs.outN = inputs.inN;
    inputs.outC = inputs.inC;

    if (paddingModeStr == "SAME") {
        OP_CHECK_IF(inputs.strideH == 0 || inputs.strideW == 0,
            OP_LOGE(opName.c_str(), "%s", "stride should not be 0."), return GRAPH_FAILED);
        inputs.outH = (inputs.inH + inputs.strideH - 1) / inputs.strideH;
        inputs.outW = (inputs.inW + inputs.strideW - 1) / inputs.strideW;
    } else if (paddingModeStr == "VALID") {
        OP_CHECK_IF(inputs.strideH == 0 || inputs.strideW == 0,
            OP_LOGE(opName.c_str(), "%s", "stride should not be 0."), return GRAPH_FAILED);
        if (*ceilModeFlag) {
            inputs.outH = (inputs.inH - inputs.windowH + inputs.padtop + inputs.padbottom + inputs.strideH - 1)
                / inputs.strideH + 1;
            inputs.outW = (inputs.inW - inputs.windowW + inputs.padleft + inputs.padright + inputs.strideW - 1)
                / inputs.strideW + 1;
        } else {
            inputs.outH = (inputs.inH - inputs.windowH + inputs.strideH) / inputs.strideH;
            inputs.outW = (inputs.inW - inputs.windowW + inputs.strideW) / inputs.strideW;
        }
    } else if (paddingModeStr == "CALCULATED"){
        OP_CHECK_IF(inputs.strideH == 0 || inputs.strideW == 0,
            OP_LOGE(opName.c_str(), "%s", "stride should not be 0."), return GRAPH_FAILED);
        if (*ceilModeFlag) {
            inputs.outH = (inputs.inH - inputs.windowH + inputs.padtop + inputs.padbottom + inputs.strideH - 1)
                / inputs.strideH + 1;
            inputs.outW = (inputs.inW - inputs.windowW + inputs.padleft + inputs.padright + inputs.strideW - 1)
                / inputs.strideW + 1;
        } else {
            inputs.outH = (inputs.inH - inputs.windowH + inputs.padtop + inputs.padbottom) / inputs.strideH + 1;
            inputs.outW = (inputs.inW - inputs.windowW + inputs.padleft + inputs.padright) / inputs.strideW + 1;
        }
    } else {
        return GRAPH_FAILED;
    }

    // TBE not support postive actual_pads now
    bool positive = CheckPoolingPadsPositive(opName, paddingModeStr, inputs);
    OP_CHECK_IF(!positive, OP_LOGE(opName.c_str(), "%s", "check pooing pads positive failed."),
        return GRAPH_FAILED);
    // Set output shape
    if (SetAvgPoolV2Output(context, inputs, opName) != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }
    OP_LOGD(opName, "leave AvgPoolV2 shape infer!");
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(AvgPoolV2).InferShape(InferShapeForAvgPoolV2);
}
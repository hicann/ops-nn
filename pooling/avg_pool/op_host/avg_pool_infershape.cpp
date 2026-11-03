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
 * \file avg_pool_infershape.cpp
 * \brief
 */
#include "log/log.h"
#include "avg_pool_infershape_common.h"
#include "util/shape_util.h"
std::string GetInputFormatNotSupportErrMsg(const std::string& param_name, const std::string& expected_format_list,
                                           const std::string& data_format)
{
    std::string msg = ConcatString("[", param_name, "], has wrong format [", data_format, "], it should be in ",
                                   expected_format_list);
    return msg;
}

namespace ops {
// proto input
const size_t X_IDX_AVGPOOL = 0;
// proto output
const size_t OUT_IDX_AVGPOOL = 0;
// proto attributes
const size_t KSIZE_IDX_AVGPOOL = 0;
const size_t STRIDES_IDX_AVGPOOL = 1;
const size_t PADDING_IDX_AVGPOOL = 2;
const size_t DATA_FORMAT_IDX_AVGPOOL = 3;
// support information
const size_t DIM_SIZE4 = 4;
const size_t SUPPORTED_DIM_NUM = 4;
constexpr int64_t UNKNOWN_DIM_VALUE = -1;

struct AvgPoolInputs {
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
};

static graphStatus GetAvgPoolXShape(const gert::CompileTimeTensorDesc* tensorDescIn, const std::string& opName,
                                    const InferShapeContext* context, const char* dataFormatArray,
                                    AvgPoolInputs& inputs)
{
    // Get fmap Dim
    // 图推理阶段算子输入 desc 的 origin format 可能是泛化占位 ND，不在 format2str 映射表中，
    // 若直接 format2str.at() 会因键缺失抛出 std::out_of_range。数据格式以属性 data_format 为准，
    // 故对映射表做存在性保护，origin format 不在表中时回退到属性传入的格式。
    const Format xdataFormat = tensorDescIn->GetOriginFormat();
    auto formatIter = format2str.find(xdataFormat);
    std::string dataFormat;
    if (formatIter != format2str.end()) {
        dataFormat = formatIter->second;
    } else if (dataFormatArray != nullptr) {
        dataFormat = dataFormatArray;
    }
    if (dataFormat != "NCHW" && dataFormat != "NHWC") {
        OP_LOGE_FOR_INVALID_FORMAT(opName, "data_format", dataFormat.c_str(), "NCHW or NHWC");
        return GRAPH_FAILED;
    }

    bool get_dim_in_format = GetDimInFormat(opName, dataFormat, "N", inputs.xnPosition) &&
                             GetDimInFormat(opName, dataFormat, "C", inputs.xcPosition) &&
                             GetDimInFormat(opName, dataFormat, "H", inputs.xhPosition) &&
                             GetDimInFormat(opName, dataFormat, "W", inputs.xwPosition);
    if (!get_dim_in_format) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape* shapeIn = context->GetInputShape(X_IDX_AVGPOOL);
    OP_LOGE_IF(shapeIn == nullptr, ge::GRAPH_FAILED, opName, "fmap is null.");
    if (shapeIn->GetDimNum() != SUPPORTED_DIM_NUM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(opName, "x", std::to_string(shapeIn->GetDimNum()).c_str(),
                                                 "shape dim must be 4");
        return GRAPH_FAILED;
    }

    // Set x shape into structure
    inputs.inN = shapeIn->GetDim(inputs.xnPosition);
    inputs.inC = shapeIn->GetDim(inputs.xcPosition);
    inputs.inH = shapeIn->GetDim(inputs.xhPosition);
    inputs.inW = shapeIn->GetDim(inputs.xwPosition);

    OP_LOGE_IF(inputs.inC != UNKNOWN_DIM_VALUE && inputs.inC < 1, ge::GRAPH_FAILED, opName,
               "x channel should be greater than or equal to 1. actual is: %ld", inputs.inC);

    return ge::GRAPH_SUCCESS;
}

static graphStatus GetAvgPoolWindowAndStride(const int64_t* ksizeArray, const int64_t* stridesArray,
                                             AvgPoolInputs& inputs)
{
    inputs.windowH = ksizeArray[inputs.xhPosition];
    inputs.windowW = ksizeArray[inputs.xwPosition];
    inputs.strideH = stridesArray[inputs.xhPosition];
    inputs.strideW = stridesArray[inputs.xwPosition];

    return GRAPH_SUCCESS;
}

static graphStatus SetAvgPoolOutput(InferShapeContext* context, AvgPoolInputs& inputs, const std::string& opName)
{
    const gert::CompileTimeTensorDesc* tensordescOutput = context->GetOutputDesc(OUT_IDX_AVGPOOL);
    OP_LOGE_IF(tensordescOutput == nullptr, GRAPH_FAILED, opName, "Get output failed.");
    auto formatOut = tensordescOutput->GetOriginFormat();
    auto shapeOut = context->GetOutputShape(OUT_IDX_AVGPOOL);
    OP_LOGE_IF(shapeOut == nullptr, GRAPH_FAILED, opName, "Get output shape failed.");

    shapeOut->SetDimNum(SUPPORTED_DIM_NUM);
    // NC1HWC0(NCHW/NHWC)
    if (formatOut == Format::FORMAT_NCHW || formatOut == Format::FORMAT_NHWC) {
        shapeOut->SetDim(inputs.xnPosition, inputs.outN);
        shapeOut->SetDim(inputs.xcPosition, inputs.outC);
        shapeOut->SetDim(inputs.xhPosition, inputs.outH);
        shapeOut->SetDim(inputs.xwPosition, inputs.outW);
    } else {
        OP_LOGE_FOR_INVALID_FORMAT(opName, "y", std::to_string(static_cast<int32_t>(formatOut)).c_str(),
                                   "NCHW or NHWC");
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

static graphStatus InferShapeForAvgPool(InferShapeContext* context)
{
    OP_LOGI(context->GetNodeName(), "***runtime2.0 inferShape***");
    const std::string opName = context->GetNodeName();
    OP_LOGD(opName, "Enter avgpool shape infer.");

    AvgPoolInputs inputs;

    // Define runtime attrs of AvgPool
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OP_LOGE_IF(attrs == nullptr, ge::GRAPH_FAILED, opName, "Get attrs failed.");

    // Get input desc
    const gert::CompileTimeTensorDesc* tensorDescIn = context->GetInputDesc(X_IDX_AVGPOOL);
    OP_LOGE_IF(tensorDescIn == nullptr, GRAPH_FAILED, opName, "Get fmap failed.");

    // Get output desc
    const gert::CompileTimeTensorDesc* tensorDescOutput = context->GetOutputDesc(OUT_IDX_AVGPOOL);
    OP_LOGE_IF(tensorDescOutput == nullptr, GRAPH_FAILED, opName, "Get output failed.");

    // Get input ksize
    const gert::ContinuousVector* ksizePtr = attrs->GetAttrPointer<gert::ContinuousVector>(KSIZE_IDX_AVGPOOL);
    OP_LOGE_IF(ksizePtr == nullptr, GRAPH_FAILED, opName, "Get ksize failed.");
    const int64_t* ksizeArray = reinterpret_cast<const int64_t*>(ksizePtr->GetData());
    OP_LOGE_IF(ksizeArray == nullptr, GRAPH_FAILED, opName, "ksize is null");

    if (ksizePtr->GetSize() != DIM_SIZE4) {
        OP_LOGE_FOR_INVALID_LISTSIZE(opName, "ksize", std::to_string(ksizePtr->GetSize()).c_str(), "4");
        return GRAPH_FAILED;
    }

    // Get input stride
    const gert::ContinuousVector* stridesPtr = attrs->GetAttrPointer<gert::ContinuousVector>(STRIDES_IDX_AVGPOOL);
    OP_LOGE_IF(stridesPtr == nullptr, GRAPH_FAILED, opName, "Get strides failed.");
    const int64_t* stridesArray = reinterpret_cast<const int64_t*>(stridesPtr->GetData());
    OP_LOGE_IF(stridesArray == nullptr, GRAPH_FAILED, opName, "Stride is null.");

    if (stridesPtr->GetSize() != DIM_SIZE4) {
        OP_LOGE_FOR_INVALID_LISTSIZE(opName, "strides", std::to_string(stridesPtr->GetSize()).c_str(), "4");
        return GRAPH_FAILED;
    }

    // Get input padding mode
    const char* paddingPtr = attrs->GetAttrPointer<char>(PADDING_IDX_AVGPOOL);
    OP_LOGE_IF(paddingPtr == nullptr, GRAPH_FAILED, opName, "Get pads failed.");
    std::string padStr = paddingPtr == nullptr ? "NULL" : paddingPtr;
    if (padStr != "SAME" && padStr != "VALID") {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "padding_mode", padStr.c_str(),
                                              "padding_mode must be SAME or VALID");
        return GRAPH_FAILED;
    }

    // Get input data_format
    const char* dataFormatArray = attrs->GetAttrPointer<char>(DATA_FORMAT_IDX_AVGPOOL);

    // Get input shape
    const gert::Shape* xShape = context->GetInputShape(X_IDX_AVGPOOL);
    OP_LOGE_IF(xShape == nullptr, GRAPH_FAILED, opName, "fmap is null.");
    if (Ops::Base::IsUnknownRank(*xShape)) {
        OP_LOGD(opName, "x shape is unknown rank, set output shape to unknown rank.");
        gert::Shape* outputShape = context->GetOutputShape(OUT_IDX_AVGPOOL);
        OP_LOGE_IF(outputShape == nullptr, GRAPH_FAILED, opName, "Get output shape failed.");
        Ops::Base::SetUnknownRank(*outputShape);
        return GRAPH_SUCCESS;
    }
    if (GetAvgPoolXShape(tensorDescIn, opName, context, dataFormatArray, inputs) != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }

    if (GetAvgPoolWindowAndStride(ksizeArray, stridesArray, inputs) != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }

    if (inputs.strideH <= 0 || inputs.strideW <= 0) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            opName, "strideH, strideW",
            (std::to_string(inputs.strideH) + ", " + std::to_string(inputs.strideW)).c_str(),
            "stride values must be > 0");
        return GRAPH_FAILED;
    }

    if (inputs.windowH <= 0 || inputs.windowW <= 0) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            opName, "windowH, windowW",
            (std::to_string(inputs.windowH) + ", " + std::to_string(inputs.windowW)).c_str(),
            "ksize values must be > 0");
        return GRAPH_FAILED;
    }

    inputs.outN = inputs.inN;
    inputs.outC = inputs.inC;
    if (inputs.inH == UNKNOWN_DIM_VALUE || inputs.inW == UNKNOWN_DIM_VALUE) {
        OP_LOGD(opName, "x H/W dim is unknown, set output H/W dim to unknown.");
        inputs.outH = UNKNOWN_DIM_VALUE;
        inputs.outW = UNKNOWN_DIM_VALUE;
    } else if (padStr == "SAME") {
        inputs.outH = (inputs.inH + inputs.strideH - 1) / inputs.strideH;
        inputs.outW = (inputs.inW + inputs.strideW - 1) / inputs.strideW;
    } else {
        inputs.outH = (inputs.inH - inputs.windowH + inputs.strideH) / inputs.strideH;
        inputs.outW = (inputs.inW - inputs.windowW + inputs.strideW) / inputs.strideW;
    }

    // Set output shape
    if (SetAvgPoolOutput(context, inputs, opName) != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }
    OP_LOGD(opName, "leave AvgPool shape infer!");
    return GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(AvgPool).InferShape(InferShapeForAvgPool);
} // namespace ops

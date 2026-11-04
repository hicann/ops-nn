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
#include "log/log.h"
#include "pooling/avg_pool/op_host/avg_pool_infershape_common.h"
#include "util/shape_util.h"

template <class T>
bool CheckPoolingPadsPositive(const std::string& opName, const std::string& paddingModeStr, T& inputs)
{
    int64_t pad_needed_h = 0;
    int64_t pad_needed_w = 0;
    if (paddingModeStr == "SAME") {
        pad_needed_h = (inputs.outH - 1) * inputs.strideH + inputs.windowH - inputs.inH;
        pad_needed_w = (inputs.outW - 1) * inputs.strideW + inputs.windowW - inputs.inW;
        if (pad_needed_h < 0 or pad_needed_w < 0) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                opName, "pad_needed", (std::to_string(pad_needed_h) + ", " + std::to_string(pad_needed_w)).c_str(),
                "pad_needed must be >= 0");
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
constexpr int64_t UNKNOWN_DIM_VALUE = -1;
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

static graphStatus GetAvgPoolV2XShape(const gert::CompileTimeTensorDesc* tensorDescIn, const std::string& opName,
                                      const InferShapeContext* context, const char* dataFormatArray,
                                      AvgPoolV2Inputs& inputs)
{
    // Get fmap Dim
    // 图推理阶段算子输入 desc 的 origin format 可能是泛化占位 ND，不在 format2str 映射表中，
    // 若直接 format2str.at() 会因键缺失抛出 std::out_of_range。数据格式以属性 data_format 为准，
    // 故对映射表做存在性保护，origin format 不在表中时回退到属性传入的格式。
    const Format xdataFormat = tensorDescIn->GetOriginFormat();
    auto formatIter = format2str.find(xdataFormat);
    std::string dataFormat = (formatIter != format2str.end()) ? formatIter->second : std::string(dataFormatArray);
    if ((dataFormat != "NCHW" && dataFormat != "NHWC") and (dataFormat != dataFormatArray)) {
        OP_LOGE_FOR_INVALID_FORMAT(opName, "data_format", dataFormat.c_str(), "NCHW or NHWC");
        return GRAPH_FAILED;
    }

    if (dataFormat.length() != DIM_SIZE4) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "data_format_dim", std::to_string(dataFormat.length()).c_str(),
                                              "data_format length must be 4");
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

    OP_LOGE_IF(inputs.inC != UNKNOWN_DIM_VALUE && inputs.inC < 1, GRAPH_FAILED, opName,
               "x channel should be greater than or equal to 1. actual is: %ld", inputs.inC);

    return GRAPH_SUCCESS;
}

static graphStatus SetAvgPoolV2Output(InferShapeContext* context, AvgPoolV2Inputs& inputs, const std::string& opName)
{
    auto shapeOut = context->GetOutputShape(OUT_IDX_AVGPOOLV2);
    OP_LOGE_IF(shapeOut == nullptr, GRAPH_FAILED, opName, "Get shapeOut failed.");

    // 输出 shape 按 data_format 决定的 N/C/H/W 位置设置，不依赖图推理阶段可能为 ND 占位的输出 desc format
    shapeOut->SetDimNum(SUPPORTED_DIM_NUM);
    shapeOut->SetDim(inputs.xnPosition, inputs.outN);
    shapeOut->SetDim(inputs.xcPosition, inputs.outC);
    shapeOut->SetDim(inputs.xhPosition, inputs.outH);
    shapeOut->SetDim(inputs.xwPosition, inputs.outW);
    return GRAPH_SUCCESS;
}

static graphStatus InferShapeForAvgPoolV2(InferShapeContext* context)
{
    OP_LOGI(context->GetNodeName(), "***runtime2.0 inferShape***");
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
        OP_LOGE_FOR_INVALID_LISTSIZE(opName, "ksize", std::to_string(ksizePtr->GetSize()).c_str(), "4");
        return GRAPH_FAILED;
    }

    // Get input strides
    const gert::ContinuousVector* stridesPtr = attrs->GetAttrPointer<gert::ContinuousVector>(STRIDES_IDX_AVGPOOLV2);
    OP_LOGE_IF(stridesPtr == nullptr, GRAPH_FAILED, opName, "Get strides failed.");
    const int64_t* stridesArray = static_cast<const int64_t*>(stridesPtr->GetData());
    OP_LOGE_IF(stridesArray == nullptr, GRAPH_FAILED, opName, "Stride is null.");

    if (stridesPtr->GetSize() != DIM_SIZE4) {
        OP_LOGE_FOR_INVALID_LISTSIZE(opName, "strides", std::to_string(stridesPtr->GetSize()).c_str(), "4");
        return GRAPH_FAILED;
    }

    // Get input data_format
    const char* dataFormatArray = attrs->GetAttrPointer<char>(DATA_FORMAT_IDX_AVGPOOLV2);
    OP_LOGE_IF(dataFormatArray == nullptr, GRAPH_FAILED, opName, "Get data format failed.");

    // Get input padding mode
    const char* paddingModePtr = attrs->GetAttrPointer<char>(PADDING_MODE_IDX_AVGPOOLV2);
    std::string paddingModeStr = paddingModePtr == nullptr ? "NULL" : paddingModePtr;

    // Get input pads
    const gert::ContinuousVector* padsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(PADS_IDX_AVGPOOLV2);
    OP_LOGE_IF(padsPtr == nullptr, GRAPH_FAILED, opName, "Get pads failed.");
    const int64_t* padsArray = static_cast<const int64_t*>(padsPtr->GetData());
    OP_LOGE_IF(padsArray == nullptr, GRAPH_FAILED, opName, "pads is null.");

    auto pSize = padsPtr->GetSize();
    OP_LOGE_IF(pSize != PAD_SIZE_LIMIT, GRAPH_FAILED, context->GetNodeName(), "pads list should be 4D, actual is: %ld.",
               pSize);

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
    const gert::Shape* xShape = context->GetInputShape(X_IDX_AVGPOOLV2);
    OP_LOGE_IF(xShape == nullptr, GRAPH_FAILED, opName, "fmap is null.");
    if (Ops::Base::IsUnknownRank(*xShape)) {
        OP_LOGD(opName, "x shape is unknown rank, set output shape to unknown rank.");
        gert::Shape* outputShape = context->GetOutputShape(OUT_IDX_AVGPOOLV2);
        OP_LOGE_IF(outputShape == nullptr, GRAPH_FAILED, opName, "Get output shape failed.");
        Ops::Base::SetUnknownRank(*outputShape);
        return ge::GRAPH_SUCCESS;
    }
    if (GetAvgPoolV2XShape(tensorDescIn, opName, context, dataFormatArray, inputs) != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }

    inputs.outN = inputs.inN;
    inputs.outC = inputs.inC;
    if (inputs.inH == UNKNOWN_DIM_VALUE || inputs.inW == UNKNOWN_DIM_VALUE) {
        OP_LOGD(opName, "x H/W dim is unknown, set output H/W dim to unknown.");
        inputs.outH = UNKNOWN_DIM_VALUE;
        inputs.outW = UNKNOWN_DIM_VALUE;
        if (SetAvgPoolV2Output(context, inputs, opName) != GRAPH_SUCCESS) {
            return GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
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
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                        opName, "windowH, windowW",
                        (std::to_string(inputs.windowH) + ", " + std::to_string(inputs.windowW)).c_str(),
                        "ksize values must be > 0"),
                    return GRAPH_FAILED);
        OP_CHECK_IF(inputs.strideH <= 0 || inputs.strideW <= 0,
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                        opName, "strideH, strideW",
                        (std::to_string(inputs.strideH) + ", " + std::to_string(inputs.strideW)).c_str(),
                        "stride values must be > 0"),
                    return GRAPH_FAILED);
    }

    if (paddingModeStr == "SAME") {
        OP_CHECK_IF(inputs.strideH == 0 || inputs.strideW == 0,
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                        opName, "strideH, strideW",
                        (std::to_string(inputs.strideH) + ", " + std::to_string(inputs.strideW)).c_str(),
                        "stride values must not be 0"),
                    return GRAPH_FAILED);
        inputs.outH = (inputs.inH + inputs.strideH - 1) / inputs.strideH;
        inputs.outW = (inputs.inW + inputs.strideW - 1) / inputs.strideW;
    } else if (paddingModeStr == "VALID") {
        OP_CHECK_IF(inputs.strideH == 0 || inputs.strideW == 0,
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                        opName, "strideH, strideW",
                        (std::to_string(inputs.strideH) + ", " + std::to_string(inputs.strideW)).c_str(),
                        "stride values must not be 0"),
                    return GRAPH_FAILED);
        if (*ceilModeFlag) {
            inputs.outH = (inputs.inH - inputs.windowH + inputs.padtop + inputs.padbottom + inputs.strideH - 1) /
                              inputs.strideH +
                          1;
            inputs.outW = (inputs.inW - inputs.windowW + inputs.padleft + inputs.padright + inputs.strideW - 1) /
                              inputs.strideW +
                          1;
        } else {
            inputs.outH = (inputs.inH - inputs.windowH + inputs.strideH) / inputs.strideH;
            inputs.outW = (inputs.inW - inputs.windowW + inputs.strideW) / inputs.strideW;
        }
    } else if (paddingModeStr == "CALCULATED") {
        OP_CHECK_IF(inputs.strideH == 0 || inputs.strideW == 0,
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                        opName, "strideH, strideW",
                        (std::to_string(inputs.strideH) + ", " + std::to_string(inputs.strideW)).c_str(),
                        "stride values must not be 0"),
                    return GRAPH_FAILED);
        if (*ceilModeFlag) {
            inputs.outH = (inputs.inH - inputs.windowH + inputs.padtop + inputs.padbottom + inputs.strideH - 1) /
                              inputs.strideH +
                          1;
            inputs.outW = (inputs.inW - inputs.windowW + inputs.padleft + inputs.padright + inputs.strideW - 1) /
                              inputs.strideW +
                          1;
            if ((inputs.outH - 1) * inputs.strideH >= inputs.inH + inputs.padtop) {
                --inputs.outH;
            }
            if ((inputs.outW - 1) * inputs.strideW >= inputs.inW + inputs.padleft) {
                --inputs.outW;
            }
        } else {
            inputs.outH = (inputs.inH - inputs.windowH + inputs.padtop + inputs.padbottom) / inputs.strideH + 1;
            inputs.outW = (inputs.inW - inputs.windowW + inputs.padleft + inputs.padright) / inputs.strideW + 1;
        }
    } else {
        return GRAPH_FAILED;
    }

    // TBE not support postive actual_pads now
    bool positive = CheckPoolingPadsPositive(opName, paddingModeStr, inputs);
    OP_CHECK_IF(
        !positive,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "pads", "check_failed", "computed pad values must be >= 0"),
        return GRAPH_FAILED);
    // Set output shape
    if (SetAvgPoolV2Output(context, inputs, opName) != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }
    OP_LOGD(opName, "leave AvgPoolV2 shape infer!");
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_INFERSHAPE(AvgPoolV2).InferShape(InferShapeForAvgPoolV2);
} // namespace ops

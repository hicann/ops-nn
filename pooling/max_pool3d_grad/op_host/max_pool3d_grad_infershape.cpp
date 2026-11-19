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
 * \file max_pool3d_grad_infershape_arch35.cpp
 * \brief
 */
#include <algorithm>
#include <string>
#include "error_util.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "graph/utils/type_utils.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "util/shape_util.h"
#include "platform/platform_infos_def.h"
#include "platform/platform_info.h"

using namespace ge;
namespace ops {
static constexpr size_t KSIZE_ATTR_INDEX = 0U;
static constexpr size_t STRIDES_ATTR_INDEX = 1U;
static constexpr size_t PADDING_ATTR_INDEX = 2U;
static constexpr size_t PADS_ATTR_INDEX = 3U;
static constexpr size_t DATA_FORMAT_ATTR_INDEX = 4U;
static constexpr size_t INDEX_OUT_MAX = 0;
static constexpr size_t ATTR_LIST_SHAPE_SIZE = 5;
static constexpr size_t INDEX_ZERO = 0;
static constexpr size_t INDEX_ONE = 1;
static constexpr size_t INDEX_TWO = 2;
static constexpr size_t CDHW_DIM_NUM = 4;
static constexpr size_t NUMBER_TWO = 2;
static constexpr size_t PARAM_NUM = 4;
static constexpr size_t SHAPE_D_DIM = 2;
static constexpr size_t SHAPE_H_DIM = 3;
static constexpr size_t SHAPE_W_DIM = 4;
static constexpr size_t PADS_SIZE = 6;
static constexpr int64_t UNKNOWN_DIM_VALUE_ = -1LL;
static constexpr int64_t PAD_HALF_DIVISOR = 2;
static constexpr size_t PADS_D_BEGIN_INDEX = 0;
static constexpr size_t PADS_D_END_INDEX = 1;
static constexpr size_t PADS_H_BEGIN_INDEX = 2;
static constexpr size_t PADS_H_END_INDEX = 3;
static constexpr size_t PADS_W_BEGIN_INDEX = 4;
static constexpr size_t PADS_W_END_INDEX = 5;

inline ge::graphStatus SetAllUnknownDim(const int64_t rank, gert::Shape* output_shape)
{
    OP_CHECK_IF(output_shape == nullptr,
                REPORT_PREDEFINED_ERR_MSG(
                    "EZ1014", (std::vector<const char*>{"opName", "reason"}),
                    (std::vector<const char*>{"MaxPool3DGrad",
                                              "SetAllUnknownDim: output_shape is null; InferShape cannot continue"})),
                return ge::GRAPH_FAILED);
    output_shape->SetDimNum(rank);
    for (int64_t i = 0; i < rank; ++i) {
        output_shape->SetDim(i, UNKNOWN_DIM_VALUE_);
    }
    OP_LOGD("SetAllUnknownDim", "set all dim = -1, output = %s", Ops::Base::ToString(*output_shape).c_str());
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus CheckPadInfoForMaxPool3DGrad(const gert::InferShapeContext* context)
{
    const char* opName_ = "MaxPool3DGrad";
    auto attrs = context->GetAttrs();
    OP_CHECK_IF(
        attrs == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context).c_str(), "attributes",
                                                 "required input, descriptor, attribute or attribute data is missing"),
        return ge::GRAPH_FAILED);
    auto padsPtr = attrs->GetAttrPointer<char>(PADDING_ATTR_INDEX);
    OP_CHECK_IF(
        padsPtr == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context).c_str(), "padding",
                                                 "required input, descriptor, attribute or attribute data is missing"),
        return ge::GRAPH_FAILED);
    std::string padding(padsPtr);
    fe::PlatformInfo platform_info;
    fe::OptionalInfo optional_info;
    OP_CHECK_IF((fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_info, optional_info) !=
                 ge::GRAPH_SUCCESS),
                REPORT_PREDEFINED_ERR_MSG(
                    "EZ1014", (std::vector<const char*>{"opName", "reason"}),
                    (std::vector<const char*>{Ops::Base::GetOpInfo(context).c_str(), "Cannot get platform info!"})),
                return ge::GRAPH_FAILED);
    OP_LOGD(context, "soc version is %s", platform_info.str_info.short_soc_version.c_str());
    if (padding != "SAME" && padding != "VALID" && padding != "CALCULATED") {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "padMode", padding.c_str(),
                                              "only support SAME, VALID or CALCULATED padding mode");
        return ge::GRAPH_FAILED;
    }
    if (padding == "CALCULATED") {
        auto pads = attrs->GetAttrPointer<gert::ContinuousVector>(PADS_ATTR_INDEX);
        OP_CHECK_IF(pads == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                        Ops::Base::GetOpInfo(context).c_str(), "pads",
                        "required input, descriptor, attribute or attribute data is missing"),
                    return ge::GRAPH_FAILED);
        if (pads->GetSize() != PADS_SIZE) {
            OP_LOGE_FOR_INVALID_LISTSIZE(opName_, "Length of pads", std::to_string(pads->GetSize()).c_str(), "6");
            return ge::GRAPH_FAILED;
        }

        auto pads_data = static_cast<const int64_t*>(pads->GetData());
        for (uint32_t i = 0; i < static_cast<uint32_t>(pads->GetSize()); i++) {
            if (pads_data[i] < 0) {
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, (std::string("pads[") + std::to_string(i) + "]").c_str(),
                                                      std::to_string(pads_data[i]).c_str(),
                                                      "pads value should be >= 0");
                return ge::GRAPH_FAILED;
            }
        }
    }
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus CheckAttrInfo(const gert::InferShapeContext* context)
{
    const char* opName_ = "MaxPool3DGrad";
    auto attrs = context->GetAttrs();
    OP_CHECK_IF(
        attrs == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context).c_str(), "attributes",
                                                 "required input, descriptor, attribute or attribute data is missing"),
        return ge::GRAPH_FAILED);

    auto ksize = attrs->GetAttrPointer<gert::ContinuousVector>(KSIZE_ATTR_INDEX);
    OP_CHECK_IF(
        ksize == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context).c_str(), "ksize",
                                                 "required input, descriptor, attribute or attribute data is missing"),
        return ge::GRAPH_FAILED);
    if (ksize->GetSize() != ATTR_LIST_SHAPE_SIZE) {
        OP_LOGE_FOR_INVALID_LISTSIZE(opName_, "Length of kSize", std::to_string(ksize->GetSize()).c_str(), "5");
        return GRAPH_FAILED;
    }

    auto ksize_data = static_cast<const int64_t*>(ksize->GetData());
    for (uint32_t i = 0; i < static_cast<uint32_t>(ksize->GetSize()); i++) {
        if (ksize_data[i] <= 0) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, (std::string("kSize[") + std::to_string(i) + "]").c_str(),
                                                  std::to_string(ksize_data[i]).c_str(),
                                                  "kSize value should be bigger than 0");
            return ge::GRAPH_FAILED;
        }
    }

    auto strides = attrs->GetAttrPointer<gert::ContinuousVector>(STRIDES_ATTR_INDEX);
    OP_CHECK_IF(
        strides == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context).c_str(), "strides",
                                                 "required input, descriptor, attribute or attribute data is missing"),
        return ge::GRAPH_FAILED);
    if (strides->GetSize() != ATTR_LIST_SHAPE_SIZE) {
        OP_LOGE_FOR_INVALID_LISTSIZE(opName_, "Length of strides", std::to_string(strides->GetSize()).c_str(), "5");
        return GRAPH_FAILED;
    }

    auto strides_data = static_cast<const int64_t*>(strides->GetData());
    for (uint32_t i = 0; i < static_cast<uint32_t>(strides->GetSize()); i++) {
        if (strides_data[i] <= 0) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, (std::string("strides[") + std::to_string(i) + "]").c_str(),
                                                  std::to_string(strides_data[i]).c_str(),
                                                  "strides value should be bigger than 0");
            return ge::GRAPH_FAILED;
        }
    }
    auto ret = CheckPadInfoForMaxPool3DGrad(context);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT("InferShapeForMaxPool3DGrad", "CheckAttrInfo return failure"),
                return ge::GRAPH_FAILED);

    const char* data_format = attrs->GetAttrPointer<char>(DATA_FORMAT_ATTR_INDEX);
    OP_CHECK_IF(
        data_format == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context).c_str(), "data_format",
                                                 "required input, descriptor, attribute or attribute data is missing"),
        return ge::GRAPH_FAILED);
    std::string dataFormatStr(data_format);
    OP_CHECK_IF(
        dataFormatStr != "NDHWC" && dataFormatStr != "NCDHW",
        OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(opName_, "data_format", data_format, "expect [NDHWC] or [NCDHW]"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CalculatePadsForMaxPool3DGrad(const gert::InferShapeContext* context, const gert::Shape* xShape)
{
    const char* opName_ = "MaxPool3DGrad";
    auto attrs = context->GetAttrs();
    OP_CHECK_IF(
        attrs == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context).c_str(), "attributes",
                                                 "required input, descriptor, attribute or attribute data is missing"),
        return ge::GRAPH_FAILED);

    const int64_t xDimNum = static_cast<int64_t>(xShape->GetDimNum());
    OP_CHECK_IF(
        xDimNum < static_cast<int64_t>(CDHW_DIM_NUM),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "length of orig_x shape", std::to_string(xDimNum).c_str(),
                                              "rank of orig_x should be 4(CDHW or DHWC) or 5(NCDHW or NDHWC)"),
        return ge::GRAPH_FAILED);

    const char* padding = attrs->GetAttrPointer<char>(PADDING_ATTR_INDEX);
    OP_CHECK_IF(
        padding == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context).c_str(), "padding",
                                                 "required input, descriptor, attribute or attribute data is missing"),
        return ge::GRAPH_FAILED);
    std::string paddingStr(padding);
    if (paddingStr == "CALCULATED") {
        return ge::GRAPH_SUCCESS;
    }

    int64_t pads[6] = {0, 0, 0, 0, 0, 0};
    if (paddingStr == "SAME") {
        const char* dataFormat = attrs->GetAttrPointer<char>(DATA_FORMAT_ATTR_INDEX);
        OP_CHECK_IF(dataFormat == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                        Ops::Base::GetOpInfo(context).c_str(), "data_format",
                        "required input, descriptor, attribute or attribute data is missing"),
                    return ge::GRAPH_FAILED);
        std::string dataFormatStr(dataFormat);

        size_t dDimIdx = static_cast<size_t>(xDimNum) - (CDHW_DIM_NUM - 1);
        size_t hDimIdx = static_cast<size_t>(xDimNum) - (CDHW_DIM_NUM - 2);
        size_t wDimIdx = static_cast<size_t>(xDimNum) - 1;
        size_t dAttrIdx = SHAPE_D_DIM;
        size_t hAttrIdx = SHAPE_H_DIM;
        size_t wAttrIdx = SHAPE_W_DIM;
        if (dataFormatStr == "NDHWC") {
            dDimIdx -= 1;
            hDimIdx -= 1;
            wDimIdx -= 1;
            dAttrIdx -= 1;
            hAttrIdx -= 1;
            wAttrIdx -= 1;
        }

        auto ksize = attrs->GetAttrPointer<gert::ContinuousVector>(KSIZE_ATTR_INDEX);
        OP_CHECK_IF(ksize == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                        Ops::Base::GetOpInfo(context).c_str(), "ksize",
                        "required input, descriptor, attribute or attribute data is missing"),
                    return ge::GRAPH_FAILED);
        auto strides = attrs->GetAttrPointer<gert::ContinuousVector>(STRIDES_ATTR_INDEX);
        OP_CHECK_IF(strides == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                        Ops::Base::GetOpInfo(context).c_str(), "strides",
                        "required input, descriptor, attribute or attribute data is missing"),
                    return ge::GRAPH_FAILED);
        const auto ksizeData = static_cast<const int64_t*>(ksize->GetData());
        const auto stridesData = static_cast<const int64_t*>(strides->GetData());
        const int64_t kernelD = ksizeData[dAttrIdx];
        const int64_t kernelH = ksizeData[hAttrIdx];
        const int64_t kernelW = ksizeData[wAttrIdx];
        const int64_t strideD = stridesData[dAttrIdx];
        const int64_t strideH = stridesData[hAttrIdx];
        const int64_t strideW = stridesData[wAttrIdx];
        const int64_t inputD = xShape->GetDim(dDimIdx);
        const int64_t inputH = xShape->GetDim(hDimIdx);
        const int64_t inputW = xShape->GetDim(wDimIdx);

        const int64_t tailsD = inputD % strideD;
        const int64_t tailsH = inputH % strideH;
        const int64_t tailsW = inputW % strideW;
        const int64_t padD = std::max(int64_t{0}, (tailsD > 0 ? kernelD - tailsD : kernelD - strideD));
        const int64_t padH = std::max(int64_t{0}, (tailsH > 0 ? kernelH - tailsH : kernelH - strideH));
        const int64_t padW = std::max(int64_t{0}, (tailsW > 0 ? kernelW - tailsW : kernelW - strideW));
        pads[PADS_D_BEGIN_INDEX] = padD / PAD_HALF_DIVISOR;
        pads[PADS_D_END_INDEX] = padD / PAD_HALF_DIVISOR + padD % PAD_HALF_DIVISOR;
        pads[PADS_H_BEGIN_INDEX] = padH / PAD_HALF_DIVISOR;
        pads[PADS_H_END_INDEX] = padH / PAD_HALF_DIVISOR + padH % PAD_HALF_DIVISOR;
        pads[PADS_W_BEGIN_INDEX] = padW / PAD_HALF_DIVISOR;
        pads[PADS_W_END_INDEX] = padW / PAD_HALF_DIVISOR + padW % PAD_HALF_DIVISOR;
    }

    OP_LOGD(context->GetNodeName(), "MaxPool3DGrad padding is %s, calculated pads is [%s]", paddingStr.c_str(),
            (std::to_string(pads[0]) + ", " + std::to_string(pads[1]) + ", " + std::to_string(pads[2]) + ", " +
             std::to_string(pads[3]) + ", " + std::to_string(pads[4]) + ", " + std::to_string(pads[5]))
                .c_str());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferShapeForMaxPool3DGrad(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    OP_LOGD(context->GetNodeName(), "runtime2.0 MaxPool3DGrad infershape running");
    auto inputXDesc = context->GetInputDesc(0);
    OP_CHECK_IF(
        inputXDesc == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context).c_str(), "orig_x",
                                                 "required input, descriptor, attribute or attribute data is missing"),
        return ge::GRAPH_FAILED);
    auto inputXFormat = inputXDesc->GetOriginFormat();
    OP_CHECK_IF(inputXFormat != FORMAT_ND && inputXFormat != FORMAT_NCDHW && inputXFormat != FORMAT_NDHWC,
                OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(context->GetNodeName(), "orig_x",
                                                        std::to_string(static_cast<int32_t>(inputXFormat)).c_str(),
                                                        "format only supports ND, NCDHW, NDHWC"),
                return ge::GRAPH_FAILED);

    auto inputYDesc = context->GetInputDesc(1);
    OP_CHECK_IF(
        inputYDesc == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context).c_str(), "orig_y",
                                                 "required input, descriptor, attribute or attribute data is missing"),
        return ge::GRAPH_FAILED);
    auto gradsDesc = context->GetInputDesc(2);
    OP_CHECK_IF(
        gradsDesc == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context).c_str(), "grads",
                                                 "required input, descriptor, attribute or attribute data is missing"),
        return ge::GRAPH_FAILED);
    const ge::DataType xDtype = inputXDesc->GetDataType();
    const ge::DataType yDtype = inputYDesc->GetDataType();
    const ge::DataType gradsDtype = gradsDesc->GetDataType();
    OP_CHECK_IF(xDtype != yDtype || xDtype != gradsDtype,
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context->GetNodeName(), "orig_x, orig_y, grads",
                                                       (ge::TypeUtils::DataTypeToSerialString(xDtype) + ", " +
                                                        ge::TypeUtils::DataTypeToSerialString(yDtype) + ", " +
                                                        ge::TypeUtils::DataTypeToSerialString(gradsDtype))
                                                           .c_str(),
                                                       "the dtypes of orig_x, orig_y and grads must be the same"),
                return ge::GRAPH_FAILED);

    auto ret = CheckAttrInfo(context);
    OP_CHECK_IF(ret != GRAPH_SUCCESS,
                OP_LOGE_WITHOUT_REPORT("InferShapeForMaxPool3DGrad", "CheckAttrInfo return unsuccessful"),
                return ge::GRAPH_FAILED);

    const gert::Shape* xShape = context->GetInputShape(0);
    OP_CHECK_IF(
        xShape == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context).c_str(), "orig_x",
                                                 "required input, descriptor, attribute or attribute data is missing"),
        return ge::GRAPH_FAILED);
    const gert::Shape* origYShape = context->GetInputShape(1);
    OP_CHECK_IF(
        origYShape == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context).c_str(), "orig_y",
                                                 "required input, descriptor, attribute or attribute data is missing"),
        return ge::GRAPH_FAILED);
    const gert::Shape* gradsShape = context->GetInputShape(2);
    OP_CHECK_IF(
        gradsShape == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context).c_str(), "grads",
                                                 "required input, descriptor, attribute or attribute data is missing"),
        return ge::GRAPH_FAILED);
    gert::Shape* yShape = context->GetOutputShape(0);
    OP_CHECK_IF(yShape == nullptr,
                REPORT_PREDEFINED_ERR_MSG("EZ1014", (std::vector<const char*>{"opName", "reason"}),
                                          (std::vector<const char*>{Ops::Base::GetOpInfo(context).c_str(),
                                                                    "yShape is null; InferShape cannot continue"})),
                return ge::GRAPH_FAILED);
    size_t xDimNum = xShape->GetDimNum();

    if (Ops::Base::IsUnknownShape(*xShape) || Ops::Base::IsUnknownShape(*origYShape) ||
        Ops::Base::IsUnknownShape(*gradsShape)) {
        SetAllUnknownDim(xDimNum, yShape);
        OP_LOGD(context->GetNodeName(), "runtime2.0 MaxPool3DGrad infershape handle unknown shape.");
        return ge::GRAPH_SUCCESS;
    }

    if (Ops::Base::IsUnknownRank(*xShape)) {
        Ops::Base::SetUnknownRank(*yShape);
        OP_LOGD(context->GetNodeName(), "runtime2.0 MaxPool3DGrad infershape handle unknown rank.");
        return ge::GRAPH_SUCCESS;
    }

    auto padsRet = CalculatePadsForMaxPool3DGrad(context, xShape);
    OP_CHECK_IF(
        padsRet != GRAPH_SUCCESS,
        OP_LOGE_WITHOUT_REPORT("InferShapeForMaxPool3DGrad", "CalculatePadsForMaxPool3DGrad return unsuccessful"),
        return ge::GRAPH_FAILED);

    yShape->SetDimNum(xDimNum);
    *yShape = *xShape;

    OP_LOGD(context->GetNodeName(), "runtime2.0 MaxPool3DGrad infershape run success.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MaxPool3DGrad).InferShape(InferShapeForMaxPool3DGrad);
} // namespace ops

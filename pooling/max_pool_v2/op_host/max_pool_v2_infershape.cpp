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
 * \file max_pool_v2_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/shape_util.h"
#include <algorithm>

using namespace ge;
namespace ops {
constexpr int64_t UNKNOWN_DIM_VALUE_ = -1LL;
constexpr int64_t UNKNOWN_RANK_DIM_VALUE_ = -2LL;
constexpr size_t SHAPE_NHWC_SIZE = 4;
constexpr size_t INSHAPE_DIM_NUM = 4;
constexpr size_t INPUT_KSIZE_IDX = 1;
constexpr size_t INPUT_STRIDES_IDX = 2;
constexpr size_t INDEX_PADDING = 0;
constexpr size_t INDEX_DATA_FORMAT = 1;
constexpr size_t INDEX_ZERO = 0;
constexpr size_t INDEX_ONE = 1;
constexpr size_t INDEX_TWO = 2;
constexpr size_t INDEX_THREE = 3;
constexpr int32_t KSIZE_STRIDES_FIXED_DIM_VALUE = 1;
constexpr int64_t MIN_STRIDES_HW_VALUE = 1LL;
constexpr int64_t MAX_STRIDES_HW_VALUE = 63LL;
constexpr int64_t MAX_KSIZE_HW_PRODUCT = 255LL;

using InferShapePaddingFunc = ge::graphStatus (*)(gert::InferShapeContext*, size_t, size_t);

static int64_t SameUpdateDim(const int64_t ksize, const int64_t strides, int64_t dimSize)
{
    if (dimSize == UNKNOWN_DIM_VALUE_) {
        return UNKNOWN_DIM_VALUE_;
    }
    return (strides == 0) ? (dimSize - ksize + 1) : ((dimSize - ksize + strides) / strides);
}

static ge::graphStatus InferShapePaddingValid(gert::InferShapeContext* context, size_t hDim, size_t wDim)
{
    const char* opName_ = "MaxPoolV2";
    auto inShape = context->GetInputShape(0);
    OP_CHECK_IF(
        inShape == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "x", "required input x shape is missing"),
        return ge::GRAPH_FAILED);
    auto outShape = context->GetOutputShape(0);
    OP_CHECK_IF(
        outShape == nullptr,
        REPORT_PREDEFINED_ERR_MSG("EZ1014", (std::vector<const char*>{"opName", "reason"}),
                                  (std::vector<const char*>{context->GetNodeName(),
                                                            "outShape is null; shape/type inference cannot continue"})),
        return ge::GRAPH_FAILED);
    *outShape = *inShape;
    auto ksize = context->GetInputTensor(INPUT_KSIZE_IDX);
    OP_CHECK_IF(ksize == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "ksize",
                                                         "required parameter ksize is missing"),
                return ge::GRAPH_FAILED);
    if (ksize->GetShapeSize() != SHAPE_NHWC_SIZE) {
        OP_LOGE_FOR_INVALID_LISTSIZE(opName_, "Length of ksize", std::to_string(ksize->GetShapeSize()).c_str(), "4");
        return GRAPH_FAILED;
    }
    int64_t hksizeData = static_cast<int64_t>(ksize->GetData<int32_t>()[hDim]);
    int64_t wksizeData = static_cast<int64_t>(ksize->GetData<int32_t>()[wDim]);
    auto strides = context->GetInputTensor(INPUT_STRIDES_IDX);
    OP_CHECK_IF(strides == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "strides",
                                                         "required parameter strides is missing"),
                return ge::GRAPH_FAILED);
    if (strides->GetShapeSize() != SHAPE_NHWC_SIZE) {
        OP_LOGE_FOR_INVALID_LISTSIZE(opName_, "Length of strides", std::to_string(strides->GetShapeSize()).c_str(),
                                     "4");
        return GRAPH_FAILED;
    }
    int64_t hstridesData = static_cast<int64_t>(strides->GetData<int32_t>()[hDim]);
    int64_t wstridesData = static_cast<int64_t>(strides->GetData<int32_t>()[wDim]);
    if (hstridesData <= 0 || wstridesData <= 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            opName_, "strides[h,w]", (std::to_string(hstridesData) + ", " + std::to_string(wstridesData)).c_str(),
            "strides must be greater than 0");
        return GRAPH_FAILED;
    }
    int64_t dimSize = inShape->GetDim(hDim);
    outShape->SetDim(hDim, SameUpdateDim(hksizeData, hstridesData, dimSize));
    dimSize = inShape->GetDim(wDim);
    outShape->SetDim(wDim, SameUpdateDim(wksizeData, wstridesData, dimSize));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapePaddingSame(gert::InferShapeContext* context, size_t hDim, size_t wDim)
{
    const char* opName_ = "MaxPoolV2";
    auto inShape = context->GetInputShape(0);
    OP_CHECK_IF(
        inShape == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "x", "required input x shape is missing"),
        return ge::GRAPH_FAILED);
    auto outShape = context->GetOutputShape(0);
    OP_CHECK_IF(
        outShape == nullptr,
        REPORT_PREDEFINED_ERR_MSG("EZ1014", (std::vector<const char*>{"opName", "reason"}),
                                  (std::vector<const char*>{context->GetNodeName(),
                                                            "outShape is null; shape/type inference cannot continue"})),
        return ge::GRAPH_FAILED);
    *outShape = *inShape;
    auto strides = context->GetInputTensor(INPUT_STRIDES_IDX);
    OP_CHECK_IF(strides == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "strides",
                                                         "required parameter strides is missing"),
                return ge::GRAPH_FAILED);
    if (static_cast<uint64_t>(strides->GetShapeSize()) != SHAPE_NHWC_SIZE) {
        OP_LOGE_FOR_INVALID_LISTSIZE(opName_, "Length of strides",
                                     std::to_string(static_cast<uint64_t>(strides->GetShapeSize())).c_str(), "4");
        return GRAPH_FAILED;
    }
    int64_t hstridesData = static_cast<int64_t>(strides->GetData<int32_t>()[hDim]);
    int64_t wstridesData = static_cast<int64_t>(strides->GetData<int32_t>()[wDim]);
    if (hstridesData <= 0 || wstridesData <= 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            opName_, "strides[h,w]", (std::to_string(hstridesData) + ", " + std::to_string(wstridesData)).c_str(),
            "strides must be greater than 0");
        return GRAPH_FAILED;
    }
    int64_t dimSize = inShape->GetDim(hDim);
    outShape->SetDim(hDim, SameUpdateDim(1, hstridesData, dimSize));
    dimSize = inShape->GetDim(wDim);
    outShape->SetDim(wDim, SameUpdateDim(1, wstridesData, dimSize));
    return ge::GRAPH_SUCCESS;
}

static const std::vector<std::pair<std::string, InferShapePaddingFunc>> kFuncMap = {
    {"SAME", InferShapePaddingSame},
    {"VALID", InferShapePaddingValid},
};

static ge::graphStatus InferShape4MaxPoolV2(gert::InferShapeContext* context)
{
    const char* opName_ = "MaxPoolV2";
    auto inShape = context->GetInputShape(0);
    OP_CHECK_IF(
        inShape == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "x", "required input x shape is missing"),
        return ge::GRAPH_FAILED);
    auto outShape = context->GetOutputShape(0);
    OP_CHECK_IF(
        outShape == nullptr,
        REPORT_PREDEFINED_ERR_MSG("EZ1014", (std::vector<const char*>{"opName", "reason"}),
                                  (std::vector<const char*>{context->GetNodeName(),
                                                            "outShape is null; shape/type inference cannot continue"})),
        return ge::GRAPH_FAILED);
    if (Ops::Base::IsUnknownRank(*inShape)) {
        Ops::Base::SetUnknownShape(INSHAPE_DIM_NUM, *outShape);
        return ge::GRAPH_SUCCESS;
    }

    auto attrs = context->GetAttrs();
    OP_CHECK_IF(
        attrs == nullptr,
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1014", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{context->GetNodeName(), "attrs is null; shape/type inference cannot continue"})),
        return ge::GRAPH_FAILED);

    const char* dataFormatPtr = attrs->GetAttrPointer<char>(INDEX_DATA_FORMAT);
    std::string dataFormatStr = dataFormatPtr == nullptr ? "NHWC" : dataFormatPtr;
    if (dataFormatStr != "NHWC" && dataFormatStr != "NCHW") {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "data_format", dataFormatStr.c_str(), "must be in (NHWC, NCHW)");
        return GRAPH_FAILED;
    }
    size_t hDim = dataFormatStr == "NHWC" ? INDEX_ONE : INDEX_TWO;
    size_t wDim = dataFormatStr == "NHWC" ? INDEX_TWO : INDEX_THREE;

    auto paddingMode = attrs->GetAttrPointer<char>(INDEX_PADDING);
    OP_CHECK_IF(paddingMode == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "padding",
                                                         "required parameter padding is missing"),
                return ge::GRAPH_FAILED);

    auto ksize = context->GetInputTensor(INPUT_KSIZE_IDX);
    OP_CHECK_IF(ksize == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "ksize",
                                                         "required parameter ksize is missing"),
                return ge::GRAPH_FAILED);
    if (ksize->GetShapeSize() != SHAPE_NHWC_SIZE) {
        OP_LOGE_FOR_INVALID_LISTSIZE(opName_, "Length of ksize", std::to_string(ksize->GetShapeSize()).c_str(), "4");
        return GRAPH_FAILED;
    }
    auto ksizeData = ksize->GetData<int32_t>();
    OP_CHECK_IF(ksizeData == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "ksize",
                                                         "required parameter ksize data is missing"),
                return ge::GRAPH_FAILED);

    auto strides = context->GetInputTensor(INPUT_STRIDES_IDX);
    OP_CHECK_IF(strides == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "strides",
                                                         "required parameter strides is missing"),
                return ge::GRAPH_FAILED);
    if (strides->GetShapeSize() != SHAPE_NHWC_SIZE) {
        OP_LOGE_FOR_INVALID_LISTSIZE(opName_, "Length of strides", std::to_string(strides->GetShapeSize()).c_str(),
                                     "4");
        return GRAPH_FAILED;
    }
    auto stridesData = strides->GetData<int32_t>();
    OP_CHECK_IF(stridesData == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "strides",
                                                         "required parameter strides data is missing"),
                return ge::GRAPH_FAILED);

    if (dataFormatStr == "NHWC") {
        if (ksizeData[INDEX_ZERO] != KSIZE_STRIDES_FIXED_DIM_VALUE ||
            ksizeData[INDEX_THREE] != KSIZE_STRIDES_FIXED_DIM_VALUE ||
            stridesData[INDEX_ZERO] != KSIZE_STRIDES_FIXED_DIM_VALUE ||
            stridesData[INDEX_THREE] != KSIZE_STRIDES_FIXED_DIM_VALUE) {
            std::string attrMsg = "[" + std::to_string(ksizeData[INDEX_ZERO]) + ", " +
                                  std::to_string(ksizeData[INDEX_THREE]) + "], [" +
                                  std::to_string(stridesData[INDEX_ZERO]) + ", " +
                                  std::to_string(stridesData[INDEX_THREE]) + "]";
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                opName_, "ksize[0,3], strides[0,3]", attrMsg.c_str(),
                "MaxPoolV2 only supports pooling across height/width and other ksize/strides dimension should be one");
            return GRAPH_FAILED;
        }
    } else {
        if (ksizeData[INDEX_ZERO] != KSIZE_STRIDES_FIXED_DIM_VALUE ||
            ksizeData[INDEX_ONE] != KSIZE_STRIDES_FIXED_DIM_VALUE ||
            stridesData[INDEX_ZERO] != KSIZE_STRIDES_FIXED_DIM_VALUE ||
            stridesData[INDEX_ONE] != KSIZE_STRIDES_FIXED_DIM_VALUE) {
            std::string attrMsg = "[" + std::to_string(ksizeData[INDEX_ZERO]) + ", " +
                                  std::to_string(ksizeData[INDEX_ONE]) + "], [" +
                                  std::to_string(stridesData[INDEX_ZERO]) + ", " +
                                  std::to_string(stridesData[INDEX_ONE]) + "]";
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                opName_, "ksize[0,1], strides[0,1]", attrMsg.c_str(),
                "MaxPoolV2 only supports pooling across height/width and other ksize/strides dimension should be one");
            return GRAPH_FAILED;
        }
    }

    int64_t ksizeHwProduct = static_cast<int64_t>(ksizeData[hDim]) * static_cast<int64_t>(ksizeData[wDim]);
    if (ksizeHwProduct > MAX_KSIZE_HW_PRODUCT) {
        std::string attrMsg = "[" + std::to_string(ksizeData[hDim]) + ", " + std::to_string(ksizeData[wDim]) + "]";
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            opName_, "ksize[h,w]", attrMsg.c_str(),
            "the product of ksize in H and W dimensions should be less than or equal to 255");
        return GRAPH_FAILED;
    }
    if (stridesData[hDim] < MIN_STRIDES_HW_VALUE || stridesData[hDim] > MAX_STRIDES_HW_VALUE ||
        stridesData[wDim] < MIN_STRIDES_HW_VALUE || stridesData[wDim] > MAX_STRIDES_HW_VALUE) {
        std::string attrMsg = "[" + std::to_string(stridesData[hDim]) + ", " + std::to_string(stridesData[wDim]) + "]";
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(opName_, "strides[h,w]", attrMsg.c_str(),
                                               "strides of H and W dimensions should be in range [1, 63]");
        return GRAPH_FAILED;
    }

    auto it = std::find_if(kFuncMap.begin(), kFuncMap.end(),
                           [&paddingMode](const std::pair<std::string, InferShapePaddingFunc>& item) -> bool {
                               return item.first == paddingMode;
                           });
    if (it == kFuncMap.end()) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName_, "paddingMode", paddingMode, "must be in (VALID, SAME)");
        return GRAPH_FAILED;
    }

    // when paddingMode in (VALID, SAME)
    return it->second(context, hDim, wDim);
}

IMPL_OP_INFERSHAPE(MaxPoolV2)
    .InputsDataDependency({INPUT_KSIZE_IDX, INPUT_STRIDES_IDX})
    .InferShape(InferShape4MaxPoolV2);
} // namespace ops

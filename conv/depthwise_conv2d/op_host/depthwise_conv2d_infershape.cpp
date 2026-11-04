/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file depthwise_conv2d_infershape.cpp
 * \brief
 */

#include <algorithm>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "error_util.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/infer_shape_range_context.h"
#include "graph/utils/type_utils.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/shape_util.h"

namespace DepthwiseConv2DInfer {
using gert::InferShapeContext;

// Cache full input shapes seen in the first infer call; fuzzy-build generalizes storage shape to -1
// in later calls, the cached real shape keeps validation identical to RT1.0 OpDesc-shape behavior.
static std::mutex gDwShapeCacheMutex;
static std::unordered_map<std::string, gert::Shape> gDwShapeCache;
constexpr size_t gDwShapeCacheLimit = 1024;

static bool HasUnknownDim(const gert::Shape& shape)
{
    if (shape.GetDimNum() == 0) {
        return true;
    }
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        if (shape.GetDim(i) < 0) {
            return true;
        }
    }
    return false;
}

static gert::Shape GetCachedShape(const gert::Shape& shape, const char* nodeName, size_t idx)
{
    std::string key = std::string(nodeName == nullptr ? "" : nodeName) + ":" + std::to_string(idx);
    if (shape.GetDimNum() != 4) {
        return shape; // unknown rank or abnormal dimNum: no cache
    }
    if (!HasUnknownDim(shape)) {
        std::lock_guard<std::mutex> lock(gDwShapeCacheMutex);
        if (gDwShapeCache.size() >= gDwShapeCacheLimit) {
            gDwShapeCache.clear();
        }
        gDwShapeCache[key] = shape;
        return shape;
    }
    {
        std::lock_guard<std::mutex> lock(gDwShapeCacheMutex);
        auto it = gDwShapeCache.find(key);
        if (it != gDwShapeCache.end()) {
            gert::Shape cached = it->second;
            gDwShapeCache.erase(it);
            return cached;
        }
    }
    return shape;
}

using gert::InferShapeContext;
// proto input
const size_t X_IDX_DEPTHWISE = 0;
const size_t W_IDX_DEPTHWISE = 1;
// proto output
const size_t Y_IDX_DEPTHWISE = 0;
// proto attribute
const size_t STRIDES_IDX_DEPTHWISE = 0;
const size_t DILATIONS_IDX_DEPTHWISE = 1;
const size_t PADS_IDX_DEPTHWISE = 2;
const size_t DATA_FORMAT_IDX_DEPTHWISE = 3;
const size_t OFFSET_X_IDX_DEPTHWISE = 4;
// customized attribute, registered by IMPL_OP_INFERSHAPE PrivateAttr in proto order
const size_t PADDING_IDX_DEPTHWISE = 5;
const size_t AUTO_PAD_IDX_DEPTHWISE = 6;
const char* const PADDING = "_padding";
const char* const AUTO_PAD = "padding";
const int64_t PADDING_UNSET_SENTINEL = -1;
const std::vector<int64_t> DEFAULT_ATTR_VAL_PADDING = {-1, -1, -1, -1};
const char* const DEFAULT_ATTR_VAL_AUTOPAD = "";

// dim index
const int32_t N_DIM_IDX_NCHW = 0;
const int32_t C_DIM_IDX_NCHW = 1;
const int32_t H_DIM_IDX_NCHW = 2;
const int32_t W_DIM_IDX_NCHW = 3;
const int32_t N_DIM_IDX_NHWC = 0;
const int32_t C_DIM_IDX_NHWC = 3;
const int32_t H_DIM_IDX_NHWC = 1;
const int32_t W_DIM_IDX_NHWC = 2;
const int32_t N_DIM_IDX_HWCN = 3;
const int32_t C_DIM_IDX_HWCN = 2;
const int32_t H_DIM_IDX_HWCN = 0;
const int32_t W_DIM_IDX_HWCN = 1;
// pad index
const int32_t TOP_IDX_PAD = 0;
const int32_t BOTTOM_IDX_PAD = 1;
const int32_t LEFT_IDX_PAD = 2;
const int32_t RIGHT_IDX_PAD = 3;

// support information
const size_t SUPPORTED_DIM_NUM = 4;
const size_t PAD_SIZE_LIMIT = 4;
const size_t STRIDE_SIZE_LIMIT = 4;
const size_t DILATION_SIZE_LIMIT = 4;

const int64_t ZERO_CHANNEL = 0;
const int64_t UNKNOWN_DIM = -1;
const int64_t ZERO_TENSOR_DYN_RANGE_LOWER_BOUND = 0;
const int64_t DYN_RANGE_LOWER_BOUND = 1;
const int64_t DYN_RANGE_UPPER_BOUND = 4096;

struct DepthwiseInputShapes {
    int64_t in = 0;
    int64_t ic = 0;
    int64_t ih = 0;
    int64_t iw = 0;
    int64_t kn = 0;
    int64_t kc = 0;
    int64_t kh = 0;
    int64_t kw = 0;
    bool unknownRankX = false;
    bool unknownShapeX = false;
};

struct DepthwiseOutputShape {
    int64_t on = 0;
    int64_t oc = 0;
    int64_t oh = 0;
    int64_t ow = 0;
    bool isZeroTensor = false;
};

struct DepthwiseAttrs {
    int64_t strh = 0;
    int64_t strw = 0;
    int64_t dilh = 0;
    int64_t dilw = 0;
    int64_t padt = 0;
    int64_t padb = 0;
    int64_t padl = 0;
    int64_t padr = 0;
};

struct DimIdx {
    int32_t batch = N_DIM_IDX_NCHW;
    int32_t channel = C_DIM_IDX_NCHW;
    int32_t height = H_DIM_IDX_NCHW;
    int32_t width = W_DIM_IDX_NCHW;
};

struct TensorFmtShape {
    ge::Format format = ge::Format::FORMAT_RESERVED;
    gert::Shape cachedShape;
    const gert::Shape* shape = nullptr;
};

struct FillHwAttrParam {
    DimIdx xIdx;
    size_t attrIdx;
    size_t sizeLimit;
    const char* attrName;
};

struct HwDimRangeIn {
    bool useSame = false;
    int64_t stride = 0;
    int64_t dilation = 0;
    int64_t pad = 0;
    int64_t kernelLow = 0;
    int64_t kernelHigh = 0;
    int64_t inLow = 0;
    int64_t inHigh = 0;
};

struct ShapeRangeBundle {
    const gert::Range<gert::Shape>* xShapeRange = nullptr;
    const gert::Range<gert::Shape>* filterShapeRange = nullptr;
    gert::Range<gert::Shape>* yShapeRange = nullptr;
};
} // namespace DepthwiseConv2DInfer

namespace Ops {
namespace NN {
namespace Conv {
using namespace DepthwiseConv2DInfer;

static const char* NodeNameOrNil(const char* name) { return (name == nullptr) ? "nil" : name; }

static DimIdx GetDimIdx(const ge::Format format)
{
    if (format == ge::Format::FORMAT_NHWC) {
        return {N_DIM_IDX_NHWC, C_DIM_IDX_NHWC, H_DIM_IDX_NHWC, W_DIM_IDX_NHWC};
    }
    if (format == ge::Format::FORMAT_HWCN) {
        return {N_DIM_IDX_HWCN, C_DIM_IDX_HWCN, H_DIM_IDX_HWCN, W_DIM_IDX_HWCN};
    }
    return {N_DIM_IDX_NCHW, C_DIM_IDX_NCHW, H_DIM_IDX_NCHW, W_DIM_IDX_NCHW};
}

static ge::graphStatus ReportInvalidFormat(const char* nodeName, const char* param, const char* expected,
                                           ge::Format format)
{
    const ge::AscendString formatStr = ge::TypeUtils::FormatToAscendString(format);
    const char* actual = (formatStr.GetString() == nullptr) ? "NONE" : formatStr.GetString();
    OP_LOGE_FOR_INVALID_FORMAT(nodeName, param, actual, expected);
    return ge::GRAPH_FAILED;
}

static ge::graphStatus CheckFilterFormat(const char* nodeName, ge::Format filterFormat)
{
    if (filterFormat == ge::Format::FORMAT_NCHW || filterFormat == ge::Format::FORMAT_NHWC ||
        filterFormat == ge::Format::FORMAT_HWCN) {
        return ge::GRAPH_SUCCESS;
    }
    return ReportInvalidFormat(nodeName, "filter", "NCHW, NHWC or HWCN", filterFormat);
}

static ge::graphStatus GetInputFmtShape(InferShapeContext* context, size_t idx, TensorFmtShape& tensor)
{
    const gert::CompileTimeTensorDesc* desc = context->GetInputDesc(idx);
    OPS_CHECK_NULL_WITH_CONTEXT(context, desc);
    tensor.format = desc->GetOriginFormat();
    OP_LOGE_IF(tensor.format == ge::Format::FORMAT_RESERVED, ge::GRAPH_FAILED, context->GetNodeName(),
               "Get format failed: %d.", tensor.format);
    const gert::Shape* shape = context->GetInputShape(idx);
    OPS_CHECK_NULL_WITH_CONTEXT(context, shape);
    tensor.cachedShape = GetCachedShape(*shape, context->GetNodeName(), idx);
    tensor.shape = &tensor.cachedShape;
    return ge::GRAPH_SUCCESS;
}

static void FillXDims(const gert::Shape& shape, const DimIdx& idx, DepthwiseInputShapes& shapes)
{
    shapes.in = shape.GetDim(idx.batch);
    shapes.ic = shape.GetDim(idx.channel);
    shapes.ih = shape.GetDim(idx.height);
    shapes.iw = shape.GetDim(idx.width);
}

static void FillFilterDims(const gert::Shape& shape, const DimIdx& idx, DepthwiseInputShapes& shapes)
{
    shapes.kn = shape.GetDim(idx.batch);
    shapes.kc = shape.GetDim(idx.channel);
    shapes.kh = shape.GetDim(idx.height);
    shapes.kw = shape.GetDim(idx.width);
}

static const char* GetAttrCStr(const gert::RuntimeAttrs* attrs, size_t idx)
{
    if (attrs == nullptr || attrs->GetAttrNum() <= idx) {
        return nullptr;
    }
    return attrs->GetAttrPointer<char>(idx);
}

// x dims are mapped by the data_format attr.
static DimIdx GetDimIdxByFormatStr(const char* dataFormat)
{
    const std::string fmtStr = (dataFormat == nullptr) ? "" : dataFormat;
    return fmtStr == "NHWC" ? DimIdx{N_DIM_IDX_NHWC, C_DIM_IDX_NHWC, H_DIM_IDX_NHWC, W_DIM_IDX_NHWC} :
                              DimIdx{N_DIM_IDX_NCHW, C_DIM_IDX_NCHW, H_DIM_IDX_NCHW, W_DIM_IDX_NCHW};
}

static DimIdx GetXDimIdxByFormatAttr(InferShapeContext* context)
{
    return GetDimIdxByFormatStr(GetAttrCStr(context->GetAttrs(), DATA_FORMAT_IDX_DEPTHWISE));
}

// unknown rank x keeps all x dims unknown, output is [-1, outC, -1, -1].
static ge::graphStatus GetDepthwiseXShapeDim(InferShapeContext* context, DepthwiseInputShapes& shapes)
{
    TensorFmtShape xTensor;
    if (GetInputFmtShape(context, X_IDX_DEPTHWISE, xTensor) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    shapes.unknownRankX = xTensor.shape != nullptr && Ops::Base::IsUnknownRank(*xTensor.shape);
    shapes.unknownShapeX = xTensor.shape != nullptr && Ops::Base::IsUnknownShape(*xTensor.shape);
    if (shapes.unknownRankX) {
        shapes.in = UNKNOWN_DIM;
        shapes.ic = UNKNOWN_DIM;
        shapes.ih = UNKNOWN_DIM;
        shapes.iw = UNKNOWN_DIM;
        return ge::GRAPH_SUCCESS;
    }
    if (xTensor.shape->GetDimNum() != SUPPORTED_DIM_NUM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "x", std::to_string(xTensor.shape->GetDimNum()).c_str(),
                                     std::to_string(SUPPORTED_DIM_NUM).c_str());
        return ge::GRAPH_FAILED;
    }
    FillXDims(*xTensor.shape, GetXDimIdxByFormatAttr(context), shapes);
    return ge::GRAPH_SUCCESS;
}

// weight must be static 4D.
static ge::graphStatus GetDepthwiseWShapeDim(InferShapeContext* context, DepthwiseInputShapes& shapes)
{
    TensorFmtShape filterTensor;
    if (GetInputFmtShape(context, W_IDX_DEPTHWISE, filterTensor) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckFilterFormat(context->GetNodeName(), filterTensor.format) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (filterTensor.shape->GetDimNum() != SUPPORTED_DIM_NUM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "filter",
                                     std::to_string(filterTensor.shape->GetDimNum()).c_str(),
                                     std::to_string(SUPPORTED_DIM_NUM).c_str());
        return ge::GRAPH_FAILED;
    }
    FillFilterDims(*filterTensor.shape, GetDimIdx(filterTensor.format), shapes);
    if (shapes.kh <= 0 || shapes.kw <= 0) {
        const std::string khw = "[" + std::to_string(shapes.kh) + "," + std::to_string(shapes.kw) + "]";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "filter", khw.c_str(), "kh and kw can't be 0");
        return ge::GRAPH_FAILED;
    }
    if (shapes.kh < 0 || shapes.kw < 0 || shapes.kn < 0 || shapes.kc < 0) {
        const std::string kwShape = "[" + std::to_string(shapes.kn) + "," + std::to_string(shapes.kc) + "," +
                                    std::to_string(shapes.kh) + "," + std::to_string(shapes.kw) + "]";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "filter", kwShape.c_str(),
                                              "only static weight is supported when x is dynamic");
        return ge::GRAPH_FAILED;
    }
    if (shapes.kh < UNKNOWN_DIM || shapes.kw < UNKNOWN_DIM || shapes.kn < UNKNOWN_DIM || shapes.kc < UNKNOWN_DIM) {
        const std::string kwShape = "[" + std::to_string(shapes.kn) + "," + std::to_string(shapes.kc) + "," +
                                    std::to_string(shapes.kh) + "," + std::to_string(shapes.kw) + "]";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "filter", kwShape.c_str(),
                                              "filter dim is invalid");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// x dtype must equal filter dtype except INT16 fmap + INT8 weight.
static ge::graphStatus CheckDepthwiseDtype(InferShapeContext* context)
{
    const gert::CompileTimeTensorDesc* xDesc = context->GetInputDesc(X_IDX_DEPTHWISE);
    OPS_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    const gert::CompileTimeTensorDesc* wDesc = context->GetInputDesc(W_IDX_DEPTHWISE);
    OPS_CHECK_NULL_WITH_CONTEXT(context, wDesc);
    const ge::DataType xDtype = xDesc->GetDataType();
    const ge::DataType wDtype = wDesc->GetDataType();
    const bool mixedPrecisionFlag = (xDtype == ge::DT_INT16) && (wDtype == ge::DT_INT8);
    if (!mixedPrecisionFlag && xDtype != wDtype) {
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context->GetNodeName(), "x, filter", "same dtype",
                                               "input x dtype must be the same as filter dtype, "
                                               "except for the INT16 fmap + INT8 weight mixed scene");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// data_format attr only supports NCHW and NHWC.
static ge::graphStatus CheckDataFormatAttr(InferShapeContext* context)
{
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const char* dataFormatPtr = GetAttrCStr(attrs, DATA_FORMAT_IDX_DEPTHWISE);
    if (dataFormatPtr == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    const std::string dataFormat(dataFormatPtr);
    if (dataFormat.compare("NCHW") != 0 && dataFormat.compare("NHWC") != 0) {
        OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "data_format", dataFormat.c_str(), "NCHW or NHWC");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus FillPositiveHwFromAttr(const gert::ExtendedKernelContext* context, const FillHwAttrParam& hwAttr,
                                              int64_t& height, int64_t& width)
{
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const gert::ContinuousVector* vec = attrs->GetAttrPointer<gert::ContinuousVector>(hwAttr.attrIdx);
    OPS_CHECK_NULL_WITH_CONTEXT(context, vec);
    const char* nodeName = NodeNameOrNil(context->GetNodeName());
    if (vec->GetSize() != hwAttr.sizeLimit) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, hwAttr.attrName, std::to_string(vec->GetSize()).c_str(),
                                     std::to_string(hwAttr.sizeLimit).c_str());
        return ge::GRAPH_FAILED;
    }
    const int64_t* arr = static_cast<const int64_t*>(vec->GetData());
    OPS_CHECK_NULL_WITH_CONTEXT(context, arr);
    const DimIdx idx = hwAttr.xIdx;
    height = arr[idx.height];
    width = arr[idx.width];
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetDepthwiseStrideAndDilation(InferShapeContext* context, DepthwiseAttrs& dwAttrs)
{
    const DimIdx attrIdx = GetXDimIdxByFormatAttr(context);
    const FillHwAttrParam strideAttr{attrIdx, STRIDES_IDX_DEPTHWISE, STRIDE_SIZE_LIMIT, "strides"};
    if (FillPositiveHwFromAttr(context, strideAttr, dwAttrs.strh, dwAttrs.strw) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const FillHwAttrParam dilationAttr{attrIdx, DILATIONS_IDX_DEPTHWISE, DILATION_SIZE_LIMIT, "dilations"};
    if (FillPositiveHwFromAttr(context, dilationAttr, dwAttrs.dilh, dwAttrs.dilw) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (dwAttrs.strh == 0 || dwAttrs.strw == 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "strides", "0", "stride is 0.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static void InferSamePadAxis(int64_t inSize, int64_t kSize, int64_t stride, int64_t dilation, int64_t& padFirst,
                             int64_t& padSecond)
{
    const int64_t dk = dilation * (kSize - 1) + 1;
    const int64_t tails = inSize % stride;
    const int64_t pad = std::max((tails > 0 ? dk - tails : dk - stride), static_cast<int64_t>(0));
    padFirst = pad >> 1;
    padSecond = (pad >> 1) + (pad & 1);
}

static ge::graphStatus LoadPadsArray(InferShapeContext* context, const gert::ContinuousVector* padsPtr,
                                     DepthwiseAttrs& attrs)
{
    if (padsPtr->GetSize() != PAD_SIZE_LIMIT) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(NodeNameOrNil(context->GetNodeName()), "pads",
                                     std::to_string(padsPtr->GetSize()).c_str(),
                                     std::to_string(PAD_SIZE_LIMIT).c_str());
        return ge::GRAPH_FAILED;
    }
    const int64_t* padsArray = static_cast<const int64_t*>(padsPtr->GetData());
    OPS_CHECK_NULL_WITH_CONTEXT(context, padsArray);
    attrs.padt = padsArray[TOP_IDX_PAD];
    attrs.padb = padsArray[BOTTOM_IDX_PAD];
    attrs.padl = padsArray[LEFT_IDX_PAD];
    attrs.padr = padsArray[RIGHT_IDX_PAD];
    return ge::GRAPH_SUCCESS;
}

// _padding overrides pads when set; sentinel means unset.
static bool ApplyPrivatePaddingAttrs(const gert::RuntimeAttrs* attrs, int64_t& padt, int64_t& padb, int64_t& padl,
                                     int64_t& padr)
{
    if (attrs == nullptr || attrs->GetAttrNum() <= PADDING_IDX_DEPTHWISE) {
        return false;
    }
    const gert::ContinuousVector* paddingPtr = attrs->GetAttrPointer<gert::ContinuousVector>(PADDING_IDX_DEPTHWISE);
    if (paddingPtr == nullptr || paddingPtr->GetSize() != PAD_SIZE_LIMIT) {
        return false;
    }
    const int64_t* padsArray = static_cast<const int64_t*>(paddingPtr->GetData());
    if (padsArray == nullptr) {
        return false;
    }
    if (padsArray[TOP_IDX_PAD] == PADDING_UNSET_SENTINEL && padsArray[BOTTOM_IDX_PAD] == PADDING_UNSET_SENTINEL &&
        padsArray[LEFT_IDX_PAD] == PADDING_UNSET_SENTINEL && padsArray[RIGHT_IDX_PAD] == PADDING_UNSET_SENTINEL) {
        return false;
    }
    padt = padsArray[TOP_IDX_PAD];
    padb = padsArray[BOTTOM_IDX_PAD];
    padl = padsArray[LEFT_IDX_PAD];
    padr = padsArray[RIGHT_IDX_PAD];
    return true;
}

// private attr "_padding" overwrites pads when it is set.
static ge::graphStatus ApplyPaddingListAttr(InferShapeContext* context, DepthwiseAttrs& attrs, bool& applied)
{
    const gert::RuntimeAttrs* attrArgs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrArgs);
    if (attrArgs->GetAttrNum() > PADDING_IDX_DEPTHWISE) {
        const gert::ContinuousVector* paddingPtr = attrArgs->GetAttrPointer<gert::ContinuousVector>(
            PADDING_IDX_DEPTHWISE);
        if (paddingPtr != nullptr && paddingPtr->GetSize() != PAD_SIZE_LIMIT) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(NodeNameOrNil(context->GetNodeName()), "pads",
                                         std::to_string(paddingPtr->GetSize()).c_str(),
                                         std::to_string(PAD_SIZE_LIMIT).c_str());
            return ge::GRAPH_FAILED;
        }
    }
    applied = ApplyPrivatePaddingAttrs(attrArgs, attrs.padt, attrs.padb, attrs.padl, attrs.padr);
    return ge::GRAPH_SUCCESS;
}

// private attr "padding" (SAME/VALID) overwrites pads.
static ge::graphStatus ApplyPaddingStrAttr(InferShapeContext* context, const DepthwiseInputShapes& shapes,
                                           DepthwiseAttrs& attrs)
{
    const char* nodeName = NodeNameOrNil(context->GetNodeName());
    const gert::RuntimeAttrs* attrArgs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrArgs);
    const char* paddingPtr = GetAttrCStr(attrArgs, AUTO_PAD_IDX_DEPTHWISE);
    const std::string padStr = (paddingPtr == nullptr) ? "" : paddingPtr;
    if (padStr.size() == 0) {
        return ge::GRAPH_SUCCESS;
    }
    if (padStr.compare("SAME") == 0) {
        InferSamePadAxis(shapes.ih, shapes.kh, attrs.strh, attrs.dilh, attrs.padt, attrs.padb);
        InferSamePadAxis(shapes.iw, shapes.kw, attrs.strw, attrs.dilw, attrs.padl, attrs.padr);
    } else if (padStr.compare("VALID") == 0) {
        attrs.padt = 0;
        attrs.padb = 0;
        attrs.padl = 0;
        attrs.padr = 0;
    } else {
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "padding", padStr.c_str(), "SAME or VALID");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetDepthwisePads(InferShapeContext* context, const DepthwiseInputShapes& shapes,
                                        DepthwiseAttrs& attrs)
{
    const char* nodeName = NodeNameOrNil(context->GetNodeName());
    const gert::RuntimeAttrs* attrArgs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrArgs);
    const gert::ContinuousVector* padsPtr = attrArgs->GetAttrPointer<gert::ContinuousVector>(PADS_IDX_DEPTHWISE);
    OPS_CHECK_NULL_WITH_CONTEXT(context, padsPtr);
    if (LoadPadsArray(context, padsPtr, attrs) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    bool paddingListApplied = false;
    if (ApplyPaddingListAttr(context, attrs, paddingListApplied) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (!paddingListApplied && ApplyPaddingStrAttr(context, shapes, attrs) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (attrs.padt < 0 || attrs.padb < 0 || attrs.padl < 0 || attrs.padr < 0) {
        const std::string pads = "[" + std::to_string(attrs.padt) + "," + std::to_string(attrs.padb) + "," +
                                 std::to_string(attrs.padl) + "," + std::to_string(attrs.padr) + "]";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeName, "pads", pads.c_str(),
                                              "pads must be greater than or equal to 0");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// mark zero tensor output when any input dim is 0.
static void InferStaticOutputIsZeroTensor(const DepthwiseInputShapes& shapes, DepthwiseOutputShape& yShape)
{
    if (shapes.in == 0 || shapes.ic == 0 || shapes.ih == 0 || shapes.iw == 0 || shapes.kn == 0 || shapes.kc == 0) {
        yShape.isZeroTensor = true;
    }
}

// reject non-zero input with negative output.
static ge::graphStatus CheckOutputZeroTensor(InferShapeContext* context, const DepthwiseInputShapes& shapes,
                                             const DepthwiseOutputShape& yShape)
{
    const char* nodeName = NodeNameOrNil(context->GetNodeName());
    const std::string shapesStr = "[" + std::to_string(shapes.in) + "," + std::to_string(shapes.ic) + "," +
                                  std::to_string(shapes.ih) + "," + std::to_string(shapes.iw) + "]";
    const std::string yStr = "[" + std::to_string(yShape.on) + "," + std::to_string(yShape.oc) + "," +
                             std::to_string(yShape.oh) + "," + std::to_string(yShape.ow) + "]";
    if (yShape.isZeroTensor) {
        if (yShape.on < 0 || yShape.oc < 0 || yShape.oh < 0 || yShape.ow < 0) {
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(nodeName, "x, y", (shapesStr + ", " + yStr).c_str(),
                                                   "zero tensor input infers a negative output value");
            return ge::GRAPH_FAILED;
        }
        if (yShape.on != 0 && yShape.oc != 0 && yShape.oh != 0 && yShape.ow != 0) {
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(nodeName, "x, y", (shapesStr + ", " + yStr).c_str(),
                                                   "zero tensor input and non-zero tensor output are not supported");
            return ge::GRAPH_FAILED;
        }
        OP_LOGD(nodeName, "Output is zero tensor.");
        return ge::GRAPH_SUCCESS;
    }
    return ge::GRAPH_SUCCESS;
}

static void InferDepthwiseOutputHw(const DepthwiseInputShapes& shapes, const DepthwiseAttrs& dwAttrs,
                                   DepthwiseOutputShape& yShape)
{
    yShape.oh = UNKNOWN_DIM;
    yShape.ow = UNKNOWN_DIM;
    if (shapes.ih > UNKNOWN_DIM && shapes.kh > 0) {
        yShape.oh = (shapes.ih + dwAttrs.padt + dwAttrs.padb - dwAttrs.dilh * (shapes.kh - 1) - 1) / dwAttrs.strh + 1;
    }
    if (shapes.iw > UNKNOWN_DIM && shapes.kw > 0) {
        yShape.ow = (shapes.iw + dwAttrs.padl + dwAttrs.padr - dwAttrs.dilw * (shapes.kw - 1) - 1) / dwAttrs.strw + 1;
    }
    yShape.on = shapes.in;
    yShape.oc = (shapes.kn == UNKNOWN_DIM || shapes.kc == UNKNOWN_DIM) ? UNKNOWN_DIM : shapes.kn * shapes.kc;
}

static ge::graphStatus SetDepthwiseYShape(InferShapeContext* context, const DepthwiseOutputShape& yShape)
{
    const char* nodeName = NodeNameOrNil(context->GetNodeName());
    gert::Shape* yShapePtr = context->GetOutputShape(Y_IDX_DEPTHWISE);
    OPS_CHECK_NULL_WITH_CONTEXT(context, yShapePtr);
    const gert::CompileTimeTensorDesc* yDesc = context->GetOutputDesc(Y_IDX_DEPTHWISE);
    OPS_CHECK_NULL_WITH_CONTEXT(context, yDesc);
    const ge::Format yFormat = yDesc->GetOriginFormat();
    OP_LOGE_IF(ge::Format::FORMAT_RESERVED == yFormat, ge::GRAPH_FAILED, nodeName, "get format failed: %d", yFormat);
    yShapePtr->SetDimNum(SUPPORTED_DIM_NUM);
    if (yFormat == ge::Format::FORMAT_NCHW || yFormat == ge::Format::FORMAT_NHWC) {
        const DimIdx idx = GetDimIdx(yFormat);
        yShapePtr->SetDim(idx.batch, yShape.on);
        yShapePtr->SetDim(idx.channel, yShape.oc);
        yShapePtr->SetDim(idx.height, yShape.oh);
        yShapePtr->SetDim(idx.width, yShape.ow);
        return ge::GRAPH_SUCCESS;
    }
    return ReportInvalidFormat(nodeName, "y", "NCHW or NHWC", yFormat);
}

// unknown rank x outputs [-1, outC, -1, -1], outC is from the static weight.
static ge::graphStatus SetUnknownRankYShape(InferShapeContext* context, int64_t outC)
{
    DepthwiseOutputShape yShape;
    yShape.on = UNKNOWN_DIM;
    yShape.oc = outC;
    yShape.oh = UNKNOWN_DIM;
    yShape.ow = UNKNOWN_DIM;
    if (SetDepthwiseYShape(context, yShape) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "Unknown rank, set y shape with unknown dims. Leave shape infer.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeForDepthwiseConv2D(InferShapeContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("DepthwiseConv2D", "%s is nullptr!", "context"), return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "Enter depthwise shape infer.");
    DepthwiseInputShapes shapes;
    if (GetDepthwiseXShapeDim(context, shapes) != ge::GRAPH_SUCCESS ||
        GetDepthwiseWShapeDim(context, shapes) != ge::GRAPH_SUCCESS ||
        CheckDepthwiseDtype(context) != ge::GRAPH_SUCCESS || CheckDataFormatAttr(context) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    DepthwiseAttrs dwAttrs;
    if (GetDepthwiseStrideAndDilation(context, dwAttrs) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (GetDepthwisePads(context, shapes, dwAttrs) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (shapes.unknownRankX) {
        const int64_t outC = (shapes.kn == UNKNOWN_DIM || shapes.kc == UNKNOWN_DIM) ? UNKNOWN_DIM :
                                                                                      shapes.kn * shapes.kc;
        return SetUnknownRankYShape(context, outC);
    }
    // dynamic fmap channel is completed by the static filter channel.
    const bool isDynamic = shapes.unknownShapeX && !shapes.unknownRankX;
    if (isDynamic && shapes.ic == UNKNOWN_DIM) {
        OP_LOGD(context->GetNodeName(), "input x channel is unknown, set in_channels[%ld] to filter channel[%ld]",
                shapes.ic, shapes.kc);
        shapes.ic = shapes.kc;
    }
    if (shapes.ic < 1) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "x", std::to_string(shapes.ic).c_str(),
                                              "x channel should be greater than or equal to 1");
        return ge::GRAPH_FAILED;
    }
    // dynamic weight zero tensor is illegal.
    if (isDynamic && shapes.kc == ZERO_CHANNEL) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "filter", std::to_string(shapes.kc).c_str(),
                                              "weight is zero tensor, its cin should be greater than 0");
        return ge::GRAPH_FAILED;
    }
    DepthwiseOutputShape yShape;
    InferDepthwiseOutputHw(shapes, dwAttrs, yShape);
    if (!isDynamic) {
        InferStaticOutputIsZeroTensor(shapes, yShape);
        if (CheckOutputZeroTensor(context, shapes, yShape) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    if (SetDepthwiseYShape(context, yShape) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(),
            "x_shape:(N:%ld, C:%ld, H:%ld, W:%ld); w_shape:(N:%ld, C:%ld, H:%ld, W:%ld); "
            "y_shape(N:%ld, C:%ld, H:%ld, W:%ld); strides(H:%ld, W:%ld); dilations(H:%ld, W:%ld); "
            "pads(T:%ld, B:%ld, L:%ld, R:%ld). Leave shape infer. ",
            shapes.in, shapes.ic, shapes.ih, shapes.iw, shapes.kn, shapes.kc, shapes.kh, shapes.kw, yShape.on,
            yShape.oc, yShape.oh, yShape.ow, dwAttrs.strh, dwAttrs.strw, dwAttrs.dilh, dwAttrs.dilw, dwAttrs.padt,
            dwAttrs.padb, dwAttrs.padl, dwAttrs.padr);
    return ge::GRAPH_SUCCESS;
}

static bool DepthwiseUseSameRangeFormula(const gert::RuntimeAttrs* attrs)
{
    bool useSame = false;
    const char* paddingPtr = GetAttrCStr(attrs, AUTO_PAD_IDX_DEPTHWISE);
    if (paddingPtr != nullptr) {
        const std::string paddingStr(paddingPtr);
        if (paddingStr.compare("SAME") == 0) {
            useSame = true;
        } else if (paddingStr.compare("VALID") == 0) {
            useSame = false;
        }
    }
    return useSame;
}

static ge::graphStatus GetDepthwisePadsForRange(gert::InferShapeRangeContext* context, int64_t& padH, int64_t& padW,
                                                bool& privatePadsApplied)
{
    padH = 0;
    padW = 0;
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const gert::ContinuousVector* padsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(PADS_IDX_DEPTHWISE);
    if (padsPtr != nullptr && padsPtr->GetSize() == PAD_SIZE_LIMIT) {
        const int64_t* padsArray = static_cast<const int64_t*>(padsPtr->GetData());
        OPS_CHECK_NULL_WITH_CONTEXT(context, padsArray);
        padH = padsArray[TOP_IDX_PAD] + padsArray[BOTTOM_IDX_PAD];
        padW = padsArray[LEFT_IDX_PAD] + padsArray[RIGHT_IDX_PAD];
    }
    int64_t padt = 0;
    int64_t padb = 0;
    int64_t padl = 0;
    int64_t padr = 0;
    privatePadsApplied = ApplyPrivatePaddingAttrs(attrs, padt, padb, padl, padr);
    if (privatePadsApplied) {
        padH = padt + padb;
        padW = padl + padr;
        return ge::GRAPH_SUCCESS;
    }
    const char* paddingPtr = GetAttrCStr(attrs, AUTO_PAD_IDX_DEPTHWISE);
    if (paddingPtr != nullptr && std::string(paddingPtr).compare("VALID") == 0) {
        padH = 0;
        padW = 0;
    }
    return ge::GRAPH_SUCCESS;
}

// SAME uses ceil div, otherwise the explicit formula.
static void InferDepthwiseHwDimRange(const HwDimRangeIn& in, int64_t& outLow, int64_t& outHigh)
{
    if (in.stride == 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("DepthwiseConv2D", "strides", std::to_string(in.stride).c_str(),
                                              "stride must be greater than 0");
        return;
    }
    if (in.useSame) {
        outLow = (in.inLow + in.stride - 1) / in.stride;
        outHigh = (in.inHigh + in.stride - 1) / in.stride;
    } else {
        outLow = (in.kernelLow != 0) ? (in.inLow + in.pad - in.dilation * (in.kernelLow - 1) - 1) / in.stride + 1 : 0;
        outHigh = (in.kernelHigh != 0) ? (in.inHigh + in.pad - in.dilation * (in.kernelHigh - 1) - 1) / in.stride + 1 :
                                         0;
    }
    const int64_t lowerBound = (in.inLow == ZERO_TENSOR_DYN_RANGE_LOWER_BOUND) ? ZERO_TENSOR_DYN_RANGE_LOWER_BOUND :
                                                                                 DYN_RANGE_LOWER_BOUND;
    outLow = std::max(outLow, lowerBound);
    outHigh = std::min(outHigh, DYN_RANGE_UPPER_BOUND);
    if (in.inHigh == UNKNOWN_DIM) {
        outHigh = in.inHigh;
    }
}

static ge::graphStatus GetRangeFormats(gert::InferShapeRangeContext* context, ge::Format& filterFormat)
{
    const char* nodeName = NodeNameOrNil(context->GetNodeName());
    const gert::CompileTimeTensorDesc* filterDesc = context->GetInputDesc(W_IDX_DEPTHWISE);
    OPS_CHECK_NULL_WITH_CONTEXT(context, filterDesc);
    filterFormat = filterDesc->GetOriginFormat();
    if (filterFormat != ge::Format::FORMAT_NCHW && filterFormat != ge::Format::FORMAT_NHWC &&
        filterFormat != ge::Format::FORMAT_HWCN) {
        return ReportInvalidFormat(nodeName, "filter", "NCHW, NHWC or HWCN", filterFormat);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetAndCheckShapeRanges(gert::InferShapeRangeContext* context, ShapeRangeBundle& ranges)
{
    const char* nodeName = NodeNameOrNil(context->GetNodeName());
    ranges.xShapeRange = context->GetInputShapeRange(X_IDX_DEPTHWISE);
    ranges.filterShapeRange = context->GetInputShapeRange(W_IDX_DEPTHWISE);
    ranges.yShapeRange = context->GetOutputShapeRange(Y_IDX_DEPTHWISE);
    OPS_CHECK_NULL_WITH_CONTEXT(context, ranges.xShapeRange);
    OPS_CHECK_NULL_WITH_CONTEXT(context, ranges.filterShapeRange);
    OPS_CHECK_NULL_WITH_CONTEXT(context, ranges.yShapeRange);
    OPS_CHECK_NULL_WITH_CONTEXT(context, ranges.xShapeRange->GetMax());
    OPS_CHECK_NULL_WITH_CONTEXT(context, ranges.xShapeRange->GetMin());
    OPS_CHECK_NULL_WITH_CONTEXT(context, ranges.filterShapeRange->GetMax());
    OPS_CHECK_NULL_WITH_CONTEXT(context, ranges.filterShapeRange->GetMin());
    OPS_CHECK_NULL_WITH_CONTEXT(context, ranges.yShapeRange->GetMax());
    OPS_CHECK_NULL_WITH_CONTEXT(context, ranges.yShapeRange->GetMin());
    if (ranges.xShapeRange->GetMax()->GetDimNum() != SUPPORTED_DIM_NUM ||
        ranges.xShapeRange->GetMin()->GetDimNum() != SUPPORTED_DIM_NUM) {
        const std::string dims = std::to_string(ranges.xShapeRange->GetMin()->GetDimNum()) + ", " +
                                 std::to_string(ranges.xShapeRange->GetMax()->GetDimNum());
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(nodeName, "x_range_min, x_range_max", dims.c_str(),
                                                  "x shape range dimension number must be 4");
        return ge::GRAPH_FAILED;
    }
    if (ranges.filterShapeRange->GetMax()->GetDimNum() != SUPPORTED_DIM_NUM ||
        ranges.filterShapeRange->GetMin()->GetDimNum() != SUPPORTED_DIM_NUM) {
        const std::string dims = std::to_string(ranges.filterShapeRange->GetMin()->GetDimNum()) + ", " +
                                 std::to_string(ranges.filterShapeRange->GetMax()->GetDimNum());
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(nodeName, "filter_range_min, filter_range_max", dims.c_str(),
                                                  "filter shape range dimension number must be 4");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// output C range is static from the weight,
// H/W range is deduced from the fmap range with the conv formula.
static void FillDepthwiseOutRange(gert::InferShapeRangeContext* context, const ShapeRangeBundle& ranges,
                                  const DimIdx& xIdx, const DimIdx& filterIdx, const DepthwiseAttrs& dwAttrs,
                                  int64_t padH, int64_t padW, bool useSame,
                                  std::vector<std::pair<int64_t, int64_t>>& yRange)
{
    const char* nodeName = NodeNameOrNil(context->GetNodeName());
    yRange[static_cast<size_t>(xIdx.batch)] = std::make_pair(ranges.xShapeRange->GetMin()->GetDim(xIdx.batch),
                                                             ranges.xShapeRange->GetMax()->GetDim(xIdx.batch));
    const int64_t knMin = ranges.filterShapeRange->GetMin()->GetDim(filterIdx.batch);
    const int64_t kcMin = ranges.filterShapeRange->GetMin()->GetDim(filterIdx.channel);
    const int64_t outC = (knMin == UNKNOWN_DIM || kcMin == UNKNOWN_DIM) ? UNKNOWN_DIM : knMin * kcMin;
    yRange[static_cast<size_t>(xIdx.channel)] = std::make_pair(outC, outC);
    const HwDimRangeIn heightRange{useSame,
                                   dwAttrs.strh,
                                   dwAttrs.dilh,
                                   padH,
                                   ranges.filterShapeRange->GetMin()->GetDim(filterIdx.height),
                                   ranges.filterShapeRange->GetMax()->GetDim(filterIdx.height),
                                   ranges.xShapeRange->GetMin()->GetDim(xIdx.height),
                                   ranges.xShapeRange->GetMax()->GetDim(xIdx.height)};
    InferDepthwiseHwDimRange(heightRange, yRange[static_cast<size_t>(xIdx.height)].first,
                             yRange[static_cast<size_t>(xIdx.height)].second);
    const HwDimRangeIn widthRange{useSame,
                                  dwAttrs.strw,
                                  dwAttrs.dilw,
                                  padW,
                                  ranges.filterShapeRange->GetMin()->GetDim(filterIdx.width),
                                  ranges.filterShapeRange->GetMax()->GetDim(filterIdx.width),
                                  ranges.xShapeRange->GetMin()->GetDim(xIdx.width),
                                  ranges.xShapeRange->GetMax()->GetDim(xIdx.width)};
    InferDepthwiseHwDimRange(widthRange, yRange[static_cast<size_t>(xIdx.width)].first,
                             yRange[static_cast<size_t>(xIdx.width)].second);
    OP_LOGD(nodeName, "output range is deduced from the fmap range and the static weight.");
}

static ge::graphStatus InferShapeRangeForDepthwiseConv2D(gert::InferShapeRangeContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("DepthwiseConv2D", "%s is nullptr!", "context"), return ge::GRAPH_FAILED);
    const char* nodeName = NodeNameOrNil(context->GetNodeName());
    OP_LOGD(nodeName, "Begin dynamic shape set range.");
    ge::Format filterFormat = ge::Format::FORMAT_RESERVED;
    if (GetRangeFormats(context, filterFormat) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    ShapeRangeBundle ranges;
    if (GetAndCheckShapeRanges(context, ranges) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    DepthwiseAttrs dwAttrs;
    const DimIdx attrIdx = GetDimIdxByFormatStr(GetAttrCStr(context->GetAttrs(), DATA_FORMAT_IDX_DEPTHWISE));
    const FillHwAttrParam strideAttr{attrIdx, STRIDES_IDX_DEPTHWISE, STRIDE_SIZE_LIMIT, "strides"};
    const FillHwAttrParam dilationAttr{attrIdx, DILATIONS_IDX_DEPTHWISE, DILATION_SIZE_LIMIT, "dilations"};
    if (FillPositiveHwFromAttr(context, strideAttr, dwAttrs.strh, dwAttrs.strw) != ge::GRAPH_SUCCESS ||
        FillPositiveHwFromAttr(context, dilationAttr, dwAttrs.dilh, dwAttrs.dilw) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (dwAttrs.strh == 0 || dwAttrs.strw == 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "strides", "0", "stride is 0.");
        return ge::GRAPH_FAILED;
    }
    int64_t padH = 0;
    int64_t padW = 0;
    bool privatePadsApplied = false;
    if (GetDepthwisePadsForRange(context, padH, padW, privatePadsApplied) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    std::vector<std::pair<int64_t, int64_t>> yRange(SUPPORTED_DIM_NUM);
    FillDepthwiseOutRange(context, ranges,
                          GetDimIdxByFormatStr(GetAttrCStr(context->GetAttrs(), DATA_FORMAT_IDX_DEPTHWISE)),
                          GetDimIdx(filterFormat), dwAttrs, padH, padW,
                          DepthwiseUseSameRangeFormula(context->GetAttrs()) && !privatePadsApplied, yRange);
    ranges.yShapeRange->GetMin()->SetDimNum(SUPPORTED_DIM_NUM);
    ranges.yShapeRange->GetMax()->SetDimNum(SUPPORTED_DIM_NUM);
    for (size_t dim = 0; dim < SUPPORTED_DIM_NUM; ++dim) {
        ranges.yShapeRange->GetMin()->SetDim(static_cast<int32_t>(dim), yRange[dim].first);
        ranges.yShapeRange->GetMax()->SetDim(static_cast<int32_t>(dim), yRange[dim].second);
    }
    OP_LOGD(nodeName, "output Range Min %s Max %s.", Shape2String(*ranges.yShapeRange->GetMin()).c_str(),
            Shape2String(*ranges.yShapeRange->GetMax()).c_str());
    return ge::GRAPH_SUCCESS;
}
} // namespace Conv

IMPL_OP_INFERSHAPE(DepthwiseConv2D)
    .InferShape(Ops::NN::Conv::InferShapeForDepthwiseConv2D)
    .InferShapeRange(Ops::NN::Conv::InferShapeRangeForDepthwiseConv2D)
    .PrivateAttr(DepthwiseConv2DInfer::PADDING, DepthwiseConv2DInfer::DEFAULT_ATTR_VAL_PADDING)
    .PrivateAttr(DepthwiseConv2DInfer::AUTO_PAD, DepthwiseConv2DInfer::DEFAULT_ATTR_VAL_AUTOPAD);
} // namespace NN
} // namespace Ops

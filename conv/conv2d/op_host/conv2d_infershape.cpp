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
 * \file conv2d_infershape.cpp
 * \brief
 */
#include <algorithm>
#include <string>
#include <utility>
#include <vector>
#include "error_util.h"
#include "graph/utils/type_utils.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/infer_shape_range_context.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/shape_util.h"

namespace Conv2DInfer {
using gert::InferShapeContext;
const size_t X_IDX_CONV2D = 0;
const size_t W_IDX_CONV2D = 1;
const size_t BIAS_IDX_CONV2D = 2;
const size_t OFFSET_W_IDX_CONV2D = 3;
const size_t Y_IDX_CONV2D = 0;
const size_t STRIDES_IDX_CONV2D = 0;
const size_t PADS_IDX_CONV2D = 1;
const size_t DILATIONS_IDX_CONV2D = 2;
const size_t GROUPS_IDX_CONV2D = 3;
const size_t PADDING_IDX_CONV2D = 6;
const size_t AUTO_PAD_IDX_CONV2D = 7;
const char* const PADDING = "padding";
const char* const AUTO_PAD = "auto_pad";
const char* const OP_IMPL_MODE = "_op_impl_mode_enum";
const char* const FIXED_SHIFT_VALUE = "fixed_shift_value";
const char* const DEFAULT_ATTR_VAL = "";

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
const int32_t TOP_IDX_PAD = 0;
const int32_t BOTTOM_IDX_PAD = 1;
const int32_t LEFT_IDX_PAD = 2;
const int32_t RIGHT_IDX_PAD = 3;

const size_t SUPPORTED_DIM_NUM = 4;
const size_t PAD_SIZE_LIMIT = 4;
const size_t STRIDE_SIZE_LIMIT = 4;
const size_t DILATION_SIZE_LIMIT = 4;

const int64_t ZERO_CHANNEL = 0;
const int64_t UNKNOWN_DIM = -1;
const int64_t ZERO_TENSOR_DYN_RANGE_LOWER_BOUND = 0;
const int64_t DYN_RANGE_LOWER_BOUND = 1;

struct Conv2DInputShapes {
    int64_t in = 0;
    int64_t ic = 0;
    int64_t ih = 0;
    int64_t iw = 0;
    int64_t kn = 0;
    int64_t kc = 0;
    int64_t kh = 0;
    int64_t kw = 0;
    bool unknownRankX = false;
    bool unknownRankW = false;
    bool unknownShapeX = false;
    bool unknownShapeW = false;
};

struct Conv2DOutputShape {
    int64_t on = 0;
    int64_t oc = 0;
    int64_t oh = 0;
    int64_t ow = 0;
    bool isZeroTensor = false;
};

struct Conv2DAttrs {
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
    const gert::Shape* shape = nullptr;
};

struct FillHwAttrParam {
    ge::Format xFormat;
    size_t attrIdx;
    size_t sizeLimit;
    const char* attrName;
};

struct PadsInferCtx {
    InferShapeContext* context;
    const gert::RuntimeAttrs* attrs;
    size_t padsSize;
    const Conv2DInputShapes& shapes;
    Conv2DAttrs& conv2DAttrs;
};

struct PadAfterConv {
    int64_t ihPad = UNKNOWN_DIM;
    int64_t iwPad = UNKNOWN_DIM;
};

struct CutPadAxisParam {
    int64_t inSize;
    int64_t kSize;
    int64_t padFirst;
    int64_t& padSecond;
    int64_t dilation;
    int64_t stride;
    int64_t preCalc = 0;
};

struct SamePadAxis {
    int64_t inSize;
    int64_t kSize;
    int64_t stride;
    int64_t dilation;
    int64_t& padFirst;
    int64_t& padSecond;
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

struct RangeCalcParam {
    ShapeRangeBundle ranges;
    DimIdx xIdx;
    DimIdx filterIdx;
    Conv2DAttrs attrs;
    int64_t padH = 0;
    int64_t padW = 0;
    bool useSame = false;
};

enum class SamePadMode : int32_t { SAME = 0, SAME_UPPER = 1, SAME_LOWER = 2 };
} // namespace Conv2DInfer

namespace Ops {
namespace NN {
namespace Conv {
using namespace Conv2DInfer;

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

static ge::graphStatus CheckXFormat(const char* nodeName, ge::Format xFormat)
{
    if (xFormat == ge::Format::FORMAT_NCHW || xFormat == ge::Format::FORMAT_NHWC) {
        return ge::GRAPH_SUCCESS;
    }
    return ReportInvalidFormat(nodeName, "x", "NCHW or NHWC", xFormat);
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
    tensor.shape = shape;
    return ge::GRAPH_SUCCESS;
}

static void FillXDims(const gert::Shape& shape, const DimIdx& idx, Conv2DInputShapes& shapes)
{
    shapes.in = shape.GetDim(idx.batch);
    shapes.ic = shape.GetDim(idx.channel);
    shapes.ih = shape.GetDim(idx.height);
    shapes.iw = shape.GetDim(idx.width);
}

static void FillFilterDims(const gert::Shape& shape, const DimIdx& idx, Conv2DInputShapes& shapes)
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

static void SetZeroPads(Conv2DAttrs& attrs)
{
    attrs.padt = 0;
    attrs.padb = 0;
    attrs.padl = 0;
    attrs.padr = 0;
}

static ge::graphStatus CheckConv2DOffsetW(InferShapeContext* context)
{
    const gert::Shape* offsetWShape = context->GetOptionalInputShape(OFFSET_W_IDX_CONV2D);
    if (offsetWShape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (offsetWShape->GetDimNum() != 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "offset_w", Shape2String(*offsetWShape).c_str(),
                                              "offset_w is not supported");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetConv2DXShapeDim(InferShapeContext* context, Conv2DInputShapes& shapes)
{
    TensorFmtShape xTensor;
    if (GetInputFmtShape(context, X_IDX_CONV2D, xTensor) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckXFormat(context->GetNodeName(), xTensor.format) != ge::GRAPH_SUCCESS) {
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
    FillXDims(*xTensor.shape, GetDimIdx(xTensor.format), shapes);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetConv2DWShapeDim(InferShapeContext* context, Conv2DInputShapes& shapes)
{
    TensorFmtShape filterTensor;
    if (GetInputFmtShape(context, W_IDX_CONV2D, filterTensor) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckFilterFormat(context->GetNodeName(), filterTensor.format) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    shapes.unknownRankW = filterTensor.shape != nullptr && Ops::Base::IsUnknownRank(*filterTensor.shape);
    shapes.unknownShapeW = filterTensor.shape != nullptr && Ops::Base::IsUnknownShape(*filterTensor.shape);
    if (shapes.unknownRankW) {
        shapes.kn = UNKNOWN_DIM;
        shapes.kc = UNKNOWN_DIM;
        shapes.kh = UNKNOWN_DIM;
        shapes.kw = UNKNOWN_DIM;
        return ge::GRAPH_SUCCESS;
    }
    if (filterTensor.shape->GetDimNum() != SUPPORTED_DIM_NUM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "filter",
                                     std::to_string(filterTensor.shape->GetDimNum()).c_str(),
                                     std::to_string(SUPPORTED_DIM_NUM).c_str());
        return ge::GRAPH_FAILED;
    }
    FillFilterDims(*filterTensor.shape, GetDimIdx(filterTensor.format), shapes);
    if (shapes.kh == 0 || shapes.kw == 0) {
        const std::string khw = "[" + std::to_string(shapes.kh) + "," + std::to_string(shapes.kw) + "]";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "filter", khw.c_str(),
                                              "kh and kw must be greater than 0");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetBiasChannelIdx(InferShapeContext* context, size_t biasDimNum, size_t& channelIdx)
{
    if (biasDimNum == SUPPORTED_DIM_NUM) {
        const gert::CompileTimeTensorDesc* biasDesc = context->GetInputDesc(BIAS_IDX_CONV2D);
        OPS_CHECK_NULL_WITH_CONTEXT(context, biasDesc);
        ge::Format biasFormat = biasDesc->GetOriginFormat();
        if (biasFormat == ge::Format::FORMAT_NCHW) {
            channelIdx = C_DIM_IDX_NCHW;
        } else if (biasFormat == ge::Format::FORMAT_NHWC) {
            channelIdx = C_DIM_IDX_NHWC;
        } else {
            const ge::AscendString biasFormatStr = ge::TypeUtils::FormatToAscendString(biasFormat);
            const char* actualFormat = (biasFormatStr.GetString() == nullptr) ? "NONE" : biasFormatStr.GetString();
            OP_LOGE_FOR_INVALID_FORMAT(context->GetNodeName(), "bias", actualFormat, "NCHW or NHWC");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckBiasDims(InferShapeContext* context, const gert::Shape* biasShape, size_t channelIdx,
                                     int64_t outChannel)
{
    const int64_t biasC = biasShape->GetDim(channelIdx);
    if (biasC != UNKNOWN_DIM && biasC != outChannel) {
        const std::string biasShapes = "[" + std::to_string(biasC) + "], [" + std::to_string(outChannel) + "]";
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "bias, filter", biasShapes.c_str(),
                                               "bias channel must be equal to out_channels");
        return ge::GRAPH_FAILED;
    }
    for (size_t dim = 0; dim < biasShape->GetDimNum(); dim++) {
        if (dim == channelIdx) {
            continue;
        }
        if (biasShape->GetDim(dim) != 1) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "bias", Shape2String(*biasShape).c_str(),
                                                  "bias dimensions other than channel must be 1");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckConv2DBias(InferShapeContext* context, int64_t outChannel)
{
    const gert::Shape* biasShape = context->GetOptionalInputShape(BIAS_IDX_CONV2D);
    if (biasShape == nullptr) {
        OP_LOGD(context->GetNodeName(), "No bias.");
        return ge::GRAPH_SUCCESS;
    }
    if (outChannel == UNKNOWN_DIM) {
        OP_LOGD(context->GetNodeName(), "Input bias's shape is dynamic.");
        return ge::GRAPH_SUCCESS;
    }
    if (Ops::Base::IsUnknownRank(*biasShape)) {
        OP_LOGD(context->GetNodeName(), "input bias's shape is [-2].");
        return ge::GRAPH_SUCCESS;
    }
    size_t biasDimNum = biasShape->GetDimNum();
    if (biasDimNum != 1 && biasDimNum != SUPPORTED_DIM_NUM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "bias", std::to_string(biasDimNum).c_str(),
                                                 "bias shape must be 1D or 4D");
        return ge::GRAPH_FAILED;
    }
    size_t channelIdx = 0;
    if (GetBiasChannelIdx(context, biasDimNum, channelIdx) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return CheckBiasDims(context, biasShape, channelIdx, outChannel);
}

static ge::graphStatus CheckGroupsAndInputChannel(InferShapeContext* context, const int64_t ic, const int64_t kc,
                                                  const int64_t groups)
{
    if (ic != kc * groups) {
        const gert::Shape* xShape = context->GetInputShape(X_IDX_CONV2D);
        OPS_CHECK_NULL_WITH_CONTEXT(context, xShape);
        const gert::Shape* filterShape = context->GetInputShape(W_IDX_CONV2D);
        OPS_CHECK_NULL_WITH_CONTEXT(context, filterShape);
        const std::string channelShapes = Shape2String(*xShape) + ", " + Shape2String(*filterShape);
        const std::string reason = "x channel must equal filter channel * groups, groups is " + std::to_string(groups);
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "x, filter", channelShapes.c_str(),
                                               reason.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckGroupsAndOutChannel(InferShapeContext* context, const int64_t outChannel,
                                                const int64_t groups)
{
    if (outChannel > 0 && groups > 0 && outChannel % groups != 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "groups", std::to_string(groups).c_str(),
                                              "out_channels must be divisible by groups");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckGroupsValue(InferShapeContext* context, const Conv2DInputShapes& shapes, int64_t groups)
{
    const bool unknownShape = shapes.unknownShapeX || shapes.unknownShapeW;
    if (groups == 0) {
        OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "groups", "0", "a positive integer");
        return ge::GRAPH_FAILED;
    }
    if (!unknownShape && groups < 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "groups", std::to_string(groups).c_str(),
                                              "groups must be greater than 0");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckGroupsWithChannels(InferShapeContext* context, const Conv2DInputShapes& shapes,
                                               int64_t groups)
{
    int64_t ic = shapes.ic;
    const int64_t kc = shapes.kc;
    if (ic < 0 && kc > 0 && groups > 0) {
        OP_LOGD(context->GetNodeName(), "input x channel is unknown, set in_channels[%ld] to kc * groups[%ld * %ld]",
                ic, kc, groups);
        ic = kc * groups;
    }
    if (ic < 0 || kc < 0 || groups < 0) {
        OP_LOGD(context->GetNodeName(), "Exist unknown shape, skip check: in_channels[%ld] = kc * groups[%ld * %ld]",
                ic, kc, groups);
        return ge::GRAPH_SUCCESS;
    }
    if (groups == 1 && kc != 0) {
        if (ic % kc == 0) {
            groups = ic / kc;
            OP_LOGD(context->GetNodeName(), "Attr groups is implicitly changed.");
        } else {
            const std::string channelShapes = "[" + std::to_string(ic) + "], [" + std::to_string(kc) + "]";
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "x, filter", channelShapes.c_str(),
                                                   "in_channels must be divisible by kernel_channels when groups is 1");
            return ge::GRAPH_FAILED;
        }
    }
    if (CheckGroupsAndInputChannel(context, ic, kc, groups) != ge::GRAPH_SUCCESS ||
        CheckGroupsAndOutChannel(context, shapes.kn, groups) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckConv2DGroups(InferShapeContext* context, const Conv2DInputShapes& shapes)
{
    if (shapes.unknownRankX || shapes.unknownRankW) {
        return ge::GRAPH_SUCCESS;
    }
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* groupsPtr = attrs->GetAttrPointer<int64_t>(GROUPS_IDX_CONV2D);
    OPS_CHECK_NULL_WITH_CONTEXT(context, groupsPtr);
    if (CheckGroupsValue(context, shapes, *groupsPtr) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return CheckGroupsWithChannels(context, shapes, *groupsPtr);
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
    if (hwAttr.xFormat == ge::Format::FORMAT_NCHW || hwAttr.xFormat == ge::Format::FORMAT_NHWC) {
        const DimIdx idx = GetDimIdx(hwAttr.xFormat);
        height = arr[idx.height];
        width = arr[idx.width];
    }
    if (height <= 0 || width <= 0) {
        const std::string hw = "[" + std::to_string(height) + "," + std::to_string(width) + "]";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeName, hwAttr.attrName, hw.c_str(),
                                              "height and width must be positive");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetConv2DStrideAndDilation(InferShapeContext* context, Conv2DAttrs& conv2DAttrs)
{
    const gert::CompileTimeTensorDesc* xDesc = context->GetInputDesc(X_IDX_CONV2D);
    OPS_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    const ge::Format xFormat = xDesc->GetOriginFormat();
    const FillHwAttrParam strideAttr{xFormat, STRIDES_IDX_CONV2D, STRIDE_SIZE_LIMIT, "strides"};
    if (FillPositiveHwFromAttr(context, strideAttr, conv2DAttrs.strh, conv2DAttrs.strw) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const FillHwAttrParam dilationAttr{xFormat, DILATIONS_IDX_CONV2D, DILATION_SIZE_LIMIT, "dilations"};
    return FillPositiveHwFromAttr(context, dilationAttr, conv2DAttrs.dilh, conv2DAttrs.dilw);
}

static void InferSamePadAxis(const SamePadAxis& axis, SamePadMode mode)
{
    if (axis.inSize < 0 || axis.kSize < 0) {
        axis.padFirst = UNKNOWN_DIM;
        axis.padSecond = UNKNOWN_DIM;
        return;
    }
    const int64_t tails = axis.inSize % axis.stride;
    const int64_t dk = axis.dilation * (axis.kSize - 1) + 1;
    int64_t pad = (tails > 0 ? dk - tails : dk - axis.stride);
    if (mode == SamePadMode::SAME) {
        pad = std::max(pad, static_cast<int64_t>(0));
    }
    const int64_t half = pad >> 1;
    const int64_t extra = pad & 1;
    if (mode == SamePadMode::SAME_LOWER) {
        axis.padFirst = half + extra;
        axis.padSecond = half;
    } else {
        axis.padFirst = half;
        axis.padSecond = half + extra;
    }
}

static void InferSamePads(const Conv2DInputShapes& shapes, Conv2DAttrs& attrs, SamePadMode mode)
{
    InferSamePadAxis({shapes.ih, shapes.kh, attrs.strh, attrs.dilh, attrs.padt, attrs.padb}, mode);
    InferSamePadAxis({shapes.iw, shapes.kw, attrs.strw, attrs.dilw, attrs.padl, attrs.padr}, mode);
}

static ge::graphStatus InferConv2DPadsWithPadding(const PadsInferCtx& ctx, const std::string& paddingStr)
{
    const char* nodeName = NodeNameOrNil(ctx.context->GetNodeName());
    OP_LOGD(nodeName, "padding str is %s.", paddingStr.c_str());
    if (paddingStr.size() == 0) {
        return ge::GRAPH_SUCCESS;
    } else if (paddingStr.compare("EXPLICIT") == 0) {
        if (ctx.padsSize != PAD_SIZE_LIMIT) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, "pads", std::to_string(ctx.padsSize).c_str(),
                                         std::to_string(PAD_SIZE_LIMIT).c_str());
            return ge::GRAPH_FAILED;
        }
    } else if (paddingStr.compare("SAME") == 0) {
        InferSamePads(ctx.shapes, ctx.conv2DAttrs, SamePadMode::SAME);
    } else if (paddingStr.compare("VALID") == 0) {
        SetZeroPads(ctx.conv2DAttrs);
    } else {
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "padding", paddingStr.c_str(), "EXPLICIT, SAME or VALID");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferConv2DPadsWithAutoPad(const PadsInferCtx& ctx, const std::string& autoPadStr)
{
    const char* nodeName = NodeNameOrNil(ctx.context->GetNodeName());
    OP_LOGD(nodeName, "auto_pad str is %s.", autoPadStr.c_str());
    if (autoPadStr.size() == 0) {
        return ge::GRAPH_SUCCESS;
    } else if (autoPadStr.compare("SAME_UPPER") == 0) {
        InferSamePads(ctx.shapes, ctx.conv2DAttrs, SamePadMode::SAME_UPPER);
    } else if (autoPadStr.compare("SAME_LOWER") == 0) {
        InferSamePads(ctx.shapes, ctx.conv2DAttrs, SamePadMode::SAME_LOWER);
    } else if (autoPadStr.compare("NOTSET") == 0) {
        if (ctx.padsSize != PAD_SIZE_LIMIT) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, "pads", std::to_string(ctx.padsSize).c_str(),
                                         std::to_string(PAD_SIZE_LIMIT).c_str());
            return ge::GRAPH_FAILED;
        }
    } else if (autoPadStr.compare("VALID") == 0) {
        SetZeroPads(ctx.conv2DAttrs);
    } else {
        OP_LOGE_FOR_INVALID_VALUE(nodeName, "auto_pad", autoPadStr.c_str(), "NOTSET, SAME_UPPER, SAME_LOWER or VALID");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CutPadAxis(CutPadAxisParam& axis)
{
    if (axis.inSize < 0 || axis.kSize <= 0) {
        return ge::GRAPH_SUCCESS;
    }
    axis.preCalc = axis.inSize + axis.padFirst + axis.padSecond - (axis.kSize - 1) * axis.dilation - 1;
    if (axis.preCalc < 0) {
        return ge::GRAPH_FAILED;
    }
    int64_t outSize = axis.preCalc / axis.stride + 1;
    int64_t remainder = axis.preCalc % axis.stride;
    if ((outSize == 1) && (remainder <= axis.padSecond)) {
        axis.padSecond -= remainder;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ReportCutPadFailed(const char* nodeName, const CutPadAxisParam& axis, bool isHeight)
{
    const char* paramName = isHeight ? "input_h, pads, kernel_h, dilations" : "input_w, pads, kernel_w, dilations";
    const char* reason = isHeight ?
                             "input_h + pad_t + pad_b - (kernel_h - 1) * dil_h - 1 must be greater than or equal to 0" :
                             "input_w + pad_l + pad_r - (kernel_w - 1) * dil_w - 1 must be greater than or equal to 0";
    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, paramName, std::to_string(axis.preCalc).c_str(), reason);
    return ge::GRAPH_FAILED;
}

static ge::graphStatus CutPads(const char* nodeName, const Conv2DInputShapes& shapes, Conv2DAttrs& attrs)
{
    CutPadAxisParam heightAxis{shapes.ih, shapes.kh, attrs.padt, attrs.padb, attrs.dilh, attrs.strh};
    if (CutPadAxis(heightAxis) != ge::GRAPH_SUCCESS) {
        return ReportCutPadFailed(nodeName, heightAxis, true);
    }
    CutPadAxisParam widthAxis{shapes.iw, shapes.kw, attrs.padl, attrs.padr, attrs.dilw, attrs.strw};
    if (CutPadAxis(widthAxis) != ge::GRAPH_SUCCESS) {
        return ReportCutPadFailed(nodeName, widthAxis, false);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckPositivePads(const char* nodeName, const Conv2DInputShapes& shapes,
                                         const Conv2DAttrs& conv2DAttrs)
{
    if (shapes.unknownRankX || shapes.unknownRankW || shapes.unknownShapeX || shapes.unknownShapeW) {
        return ge::GRAPH_SUCCESS;
    }
    if (conv2DAttrs.padt < 0 || conv2DAttrs.padb < 0 || conv2DAttrs.padl < 0 || conv2DAttrs.padr < 0) {
        const std::string pads = "[" + std::to_string(conv2DAttrs.padt) + "," + std::to_string(conv2DAttrs.padb) + "," +
                                 std::to_string(conv2DAttrs.padl) + "," + std::to_string(conv2DAttrs.padr) + "]";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeName, "pads", pads.c_str(),
                                              "pads must be greater than or equal to 0");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus LoadPadsArray(InferShapeContext* context, const gert::ContinuousVector* padsPtr,
                                     Conv2DAttrs& attrs)
{
    if (padsPtr->GetSize() == PAD_SIZE_LIMIT) {
        const int64_t* padsArray = static_cast<const int64_t*>(padsPtr->GetData());
        OPS_CHECK_NULL_WITH_CONTEXT(context, padsArray);
        attrs.padt = static_cast<int64_t>(padsArray[TOP_IDX_PAD]);
        attrs.padb = static_cast<int64_t>(padsArray[BOTTOM_IDX_PAD]);
        attrs.padl = static_cast<int64_t>(padsArray[LEFT_IDX_PAD]);
        attrs.padr = static_cast<int64_t>(padsArray[RIGHT_IDX_PAD]);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ApplyPaddingAttr(const PadsInferCtx& ctx, bool& padsOverwritten)
{
    if (ctx.attrs->GetAttrNum() <= PADDING_IDX_CONV2D) {
        return ge::GRAPH_SUCCESS;
    }
    const char* paddingPtr = ctx.attrs->GetAttrPointer<char>(PADDING_IDX_CONV2D);
    OPS_CHECK_NULL_WITH_CONTEXT(ctx.context, paddingPtr);
    const std::string paddingStr(paddingPtr);
    if (paddingStr.compare("SAME") == 0 || paddingStr.compare("VALID") == 0) {
        padsOverwritten = true;
    }
    return InferConv2DPadsWithPadding(ctx, paddingStr);
}

static ge::graphStatus ApplyAutoPadAttr(const PadsInferCtx& ctx, bool& padsOverwritten)
{
    if (ctx.attrs->GetAttrNum() <= AUTO_PAD_IDX_CONV2D) {
        return ge::GRAPH_SUCCESS;
    }
    const char* autoPadPtr = ctx.attrs->GetAttrPointer<char>(AUTO_PAD_IDX_CONV2D);
    OPS_CHECK_NULL_WITH_CONTEXT(ctx.context, autoPadPtr);
    const std::string autoPadStr(autoPadPtr);
    if (autoPadStr.compare("SAME_UPPER") == 0 || autoPadStr.compare("SAME_LOWER") == 0 ||
        autoPadStr.compare("VALID") == 0) {
        padsOverwritten = true;
    }
    return InferConv2DPadsWithAutoPad(ctx, autoPadStr);
}

static ge::graphStatus GetConv2DPads(InferShapeContext* context, const Conv2DInputShapes& shapes,
                                     Conv2DAttrs& conv2DAttrs)
{
    const char* nodeName = NodeNameOrNil(context->GetNodeName());
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const gert::ContinuousVector* padsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(PADS_IDX_CONV2D);
    OPS_CHECK_NULL_WITH_CONTEXT(context, padsPtr);
    if (LoadPadsArray(context, padsPtr, conv2DAttrs) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    bool padsOverwritten = false;
    const PadsInferCtx ctx{context, attrs, padsPtr->GetSize(), shapes, conv2DAttrs};
    if (ApplyPaddingAttr(ctx, padsOverwritten) != ge::GRAPH_SUCCESS ||
        ApplyAutoPadAttr(ctx, padsOverwritten) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (!padsOverwritten && padsPtr->GetSize() != PAD_SIZE_LIMIT) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, "pads", std::to_string(padsPtr->GetSize()).c_str(),
                                     std::to_string(PAD_SIZE_LIMIT).c_str());
        return ge::GRAPH_FAILED;
    }
    if (CutPads(nodeName, shapes, conv2DAttrs) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return CheckPositivePads(nodeName, shapes, conv2DAttrs);
}

static ge::graphStatus CheckConv2DInputWithPad(const char* nodeName, const Conv2DInputShapes& shapes, int64_t ihPad,
                                               int64_t iwPad)
{
    if ((shapes.ih > 0) && (shapes.kh > 0) && (shapes.iw > 0) && (shapes.kw > 0)) {
        if ((ihPad < 0) || (iwPad < 0)) {
            const char* nodeNameSafe = NodeNameOrNil(nodeName);
            const std::string padded = "[" + std::to_string(ihPad) + "," + std::to_string(iwPad) + "]";
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                nodeNameSafe, "x, filter", padded.c_str(),
                "image size after padding must be greater than or equal to filter size");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetConv2DYShape(InferShapeContext* context, const Conv2DOutputShape& outputShape)
{
    const char* nodeName = NodeNameOrNil(context->GetNodeName());
    gert::Shape* yShape = context->GetOutputShape(Y_IDX_CONV2D);
    OPS_CHECK_NULL_WITH_CONTEXT(context, yShape);
    const gert::CompileTimeTensorDesc* yDesc = context->GetOutputDesc(Y_IDX_CONV2D);
    OPS_CHECK_NULL_WITH_CONTEXT(context, yDesc);
    ge::Format yFormat = yDesc->GetOriginFormat();
    OP_LOGE_IF(ge::Format::FORMAT_RESERVED == yFormat, ge::GRAPH_FAILED, nodeName, "get format failed: %d", yFormat);
    const gert::CompileTimeTensorDesc* xDesc = context->GetInputDesc(X_IDX_CONV2D);
    OPS_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    const ge::Format xFormat = xDesc->GetOriginFormat();
    if (xFormat != yFormat) {
        const ge::AscendString xFormatStr = ge::TypeUtils::FormatToAscendString(xFormat);
        const ge::AscendString yFormatStr = ge::TypeUtils::FormatToAscendString(yFormat);
        const char* xActual = (xFormatStr.GetString() == nullptr) ? "NONE" : xFormatStr.GetString();
        const char* yActual = (yFormatStr.GetString() == nullptr) ? "NONE" : yFormatStr.GetString();
        const std::string formats = std::string(xActual) + ", " + yActual;
        OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(nodeName, "x, y", formats.c_str(),
                                                "x format must be the same as y format");
        return ge::GRAPH_FAILED;
    }
    yShape->SetDimNum(SUPPORTED_DIM_NUM);
    if (yFormat == ge::Format::FORMAT_NCHW || yFormat == ge::Format::FORMAT_NHWC) {
        const DimIdx idx = GetDimIdx(yFormat);
        yShape->SetDim(idx.batch, outputShape.on);
        yShape->SetDim(idx.channel, outputShape.oc);
        yShape->SetDim(idx.height, outputShape.oh);
        yShape->SetDim(idx.width, outputShape.ow);
        return ge::GRAPH_SUCCESS;
    }
    return ReportInvalidFormat(nodeName, "y", "NCHW or NHWC", yFormat);
}

static ge::graphStatus CheckOutputZeroTensor(InferShapeContext* context, const Conv2DInputShapes& shapes,
                                             Conv2DOutputShape& yShape)
{
    if (!yShape.isZeroTensor) {
        return ge::GRAPH_SUCCESS;
    }
    if (yShape.on < 0 || yShape.oc < 0 || yShape.oh < 0 || yShape.ow < 0) {
        const std::string incorrectShapes = "[" + std::to_string(shapes.in) + "," + std::to_string(shapes.ic) + "," +
                                            std::to_string(shapes.ih) + "," + std::to_string(shapes.iw) + "], [" +
                                            std::to_string(yShape.on) + "," + std::to_string(yShape.oc) + "," +
                                            std::to_string(yShape.oh) + "," + std::to_string(yShape.ow) + "]";
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "x, y", incorrectShapes.c_str(),
                                               "output infers a negative value");
        return ge::GRAPH_FAILED;
    }
    if (yShape.on != 0 && yShape.oc != 0 && yShape.oh != 0 && yShape.ow != 0) {
        const std::string incorrectShapes = "[" + std::to_string(shapes.in) + "," + std::to_string(shapes.ic) + "," +
                                            std::to_string(shapes.ih) + "," + std::to_string(shapes.iw) + "], [" +
                                            std::to_string(yShape.on) + "," + std::to_string(yShape.oc) + "," +
                                            std::to_string(yShape.oh) + "," + std::to_string(yShape.ow) + "]";
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "x, y", incorrectShapes.c_str(),
                                               "zero tensor input and non-zero tensor output are not supported");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckFcKcEqualInZeroTensor(InferShapeContext* context, const Conv2DInputShapes& shapes)
{
    if (shapes.ic != 0 && shapes.kc != 0) {
        return ge::GRAPH_SUCCESS;
    }
    if (shapes.ic != shapes.kc) {
        const std::string channels = "[" + std::to_string(shapes.ic) + "], [" + std::to_string(shapes.kc) + "]";
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "x, filter", channels.c_str(),
                                               "in zero tensor, input channel must be the same as filter channel");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetUnknownRankYShape(InferShapeContext* context)
{
    Conv2DOutputShape yShape;
    yShape.on = UNKNOWN_DIM;
    yShape.oc = UNKNOWN_DIM;
    yShape.oh = UNKNOWN_DIM;
    yShape.ow = UNKNOWN_DIM;
    if (SetConv2DYShape(context, yShape) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "Unknown rank, set y shape to [-1, -1, -1, -1]. Leave shape infer.");
    return ge::GRAPH_SUCCESS;
}

static void InferConv2DOutputHw(const Conv2DInputShapes& shapes, const Conv2DAttrs& conv2DAttrs, PadAfterConv& padAfter,
                                Conv2DOutputShape& yShape)
{
    padAfter.ihPad = UNKNOWN_DIM;
    padAfter.iwPad = UNKNOWN_DIM;
    yShape.oh = UNKNOWN_DIM;
    yShape.ow = UNKNOWN_DIM;
    if (shapes.ih > UNKNOWN_DIM && shapes.kh > 0) {
        padAfter.ihPad = shapes.ih + conv2DAttrs.padt + conv2DAttrs.padb - conv2DAttrs.dilh * (shapes.kh - 1) - 1;
        yShape.oh = padAfter.ihPad / conv2DAttrs.strh + 1;
    }
    if (shapes.iw > UNKNOWN_DIM && shapes.kw > 0) {
        padAfter.iwPad = shapes.iw + conv2DAttrs.padl + conv2DAttrs.padr - conv2DAttrs.dilw * (shapes.kw - 1) - 1;
        yShape.ow = padAfter.iwPad / conv2DAttrs.strw + 1;
    }
    yShape.on = shapes.in;
    yShape.oc = (shapes.ic == ZERO_CHANNEL) ? ZERO_CHANNEL : shapes.kn;
}

static ge::graphStatus CheckZeroTensorAndPad(InferShapeContext* context, const Conv2DInputShapes& shapes,
                                             Conv2DOutputShape& yShape, const PadAfterConv& padAfter, bool isDynamic)
{
    if (!isDynamic) {
        if (shapes.in == 0 || shapes.ic == 0 || shapes.ih == 0 || shapes.iw == 0 || shapes.kn == 0 || shapes.kc == 0) {
            OP_LOGD(context->GetNodeName(), "Input is zero Tensor!");
            yShape.isZeroTensor = true;
        }
        if (CheckFcKcEqualInZeroTensor(context, shapes) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    if (!yShape.isZeroTensor &&
        CheckConv2DInputWithPad(context->GetNodeName(), shapes, padAfter.ihPad, padAfter.iwPad) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (!isDynamic && CheckOutputZeroTensor(context, shapes, yShape) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeForConv2D(InferShapeContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("Conv2D", "%s is nullptr!", "context"), return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "Enter shape infer. ");
    Conv2DInputShapes shapes;
    if (CheckConv2DOffsetW(context) != ge::GRAPH_SUCCESS || GetConv2DXShapeDim(context, shapes) != ge::GRAPH_SUCCESS ||
        GetConv2DWShapeDim(context, shapes) != ge::GRAPH_SUCCESS ||
        CheckConv2DBias(context, shapes.kn) != ge::GRAPH_SUCCESS ||
        CheckConv2DGroups(context, shapes) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    Conv2DAttrs conv2DAttrs;
    if (GetConv2DStrideAndDilation(context, conv2DAttrs) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (shapes.unknownRankX || shapes.unknownRankW) {
        return SetUnknownRankYShape(context);
    }
    if (GetConv2DPads(context, shapes, conv2DAttrs) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    PadAfterConv padAfter;
    Conv2DOutputShape yShape;
    InferConv2DOutputHw(shapes, conv2DAttrs, padAfter, yShape);
    const bool isDynamic = (shapes.unknownShapeX && !shapes.unknownRankX) ||
                           (shapes.unknownShapeW && !shapes.unknownRankW);
    if (CheckZeroTensorAndPad(context, shapes, yShape, padAfter, isDynamic) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (SetConv2DYShape(context, yShape) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(),
            "x_shape:(N:%ld, C:%ld, H:%ld, W:%ld); w_shape:(N:%ld, C:%ld, H:%ld, W:%ld); "
            "y_shape(N:%ld, C:%ld, H:%ld, W:%ld); strides(H:%ld, W:%ld); dilations(H:%ld, W:%ld); "
            "pads(T:%ld, B:%ld, L:%ld, R:%ld). Leave shape infer. ",
            shapes.in, shapes.ic, shapes.ih, shapes.iw, shapes.kn, shapes.kc, shapes.kh, shapes.kw, yShape.on,
            yShape.oc, yShape.oh, yShape.ow, conv2DAttrs.strh, conv2DAttrs.strw, conv2DAttrs.dilh, conv2DAttrs.dilw,
            conv2DAttrs.padt, conv2DAttrs.padb, conv2DAttrs.padl, conv2DAttrs.padr);
    return ge::GRAPH_SUCCESS;
}

static bool Conv2DUseSameRangeFormula(const gert::RuntimeAttrs* attrs)
{
    bool useSame = false;
    const char* paddingPtr = GetAttrCStr(attrs, PADDING_IDX_CONV2D);
    if (paddingPtr != nullptr) {
        const std::string paddingStr(paddingPtr);
        if (paddingStr.compare("SAME") == 0) {
            useSame = true;
        } else if (paddingStr.compare("VALID") == 0) {
            useSame = false;
        }
    }
    const char* autoPadPtr = GetAttrCStr(attrs, AUTO_PAD_IDX_CONV2D);
    if (autoPadPtr != nullptr) {
        const std::string autoPadStr(autoPadPtr);
        if (autoPadStr.compare("SAME_UPPER") == 0 || autoPadStr.compare("SAME_LOWER") == 0) {
            useSame = true;
        } else if (autoPadStr.compare("VALID") == 0) {
            useSame = false;
        }
    }
    return useSame;
}

static ge::graphStatus GetConv2DPadsForRange(gert::InferShapeRangeContext* context, int64_t& padH, int64_t& padW)
{
    padH = 0;
    padW = 0;
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const gert::ContinuousVector* padsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(PADS_IDX_CONV2D);
    if (padsPtr != nullptr && padsPtr->GetSize() == PAD_SIZE_LIMIT) {
        const int64_t* padsArray = static_cast<const int64_t*>(padsPtr->GetData());
        OPS_CHECK_NULL_WITH_CONTEXT(context, padsArray);
        padH = padsArray[TOP_IDX_PAD] + padsArray[BOTTOM_IDX_PAD];
        padW = padsArray[LEFT_IDX_PAD] + padsArray[RIGHT_IDX_PAD];
    }
    const char* paddingPtr = GetAttrCStr(attrs, PADDING_IDX_CONV2D);
    if (paddingPtr != nullptr && std::string(paddingPtr).compare("VALID") == 0) {
        padH = 0;
        padW = 0;
    }
    const char* autoPadPtr = GetAttrCStr(attrs, AUTO_PAD_IDX_CONV2D);
    if (autoPadPtr != nullptr && std::string(autoPadPtr).compare("VALID") == 0) {
        padH = 0;
        padW = 0;
    }
    return ge::GRAPH_SUCCESS;
}

static void InferConv2DHwDimRange(const HwDimRangeIn& in, int64_t& outLow, int64_t& outHigh)
{
    if (in.stride <= 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("Conv2D", "strides", std::to_string(in.stride).c_str(),
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
    if (in.inHigh == UNKNOWN_DIM) {
        outHigh = in.inHigh;
    }
}

static ge::graphStatus GetRangeFormats(gert::InferShapeRangeContext* context, ge::Format& xFormat,
                                       ge::Format& filterFormat)
{
    const char* nodeName = NodeNameOrNil(context->GetNodeName());
    const gert::CompileTimeTensorDesc* xDesc = context->GetInputDesc(X_IDX_CONV2D);
    OPS_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    xFormat = xDesc->GetOriginFormat();
    if (xFormat != ge::Format::FORMAT_NCHW && xFormat != ge::Format::FORMAT_NHWC) {
        return ReportInvalidFormat(nodeName, "x", "NCHW or NHWC", xFormat);
    }
    const gert::CompileTimeTensorDesc* filterDesc = context->GetInputDesc(W_IDX_CONV2D);
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
    ranges.xShapeRange = context->GetInputShapeRange(X_IDX_CONV2D);
    ranges.filterShapeRange = context->GetInputShapeRange(W_IDX_CONV2D);
    ranges.yShapeRange = context->GetOutputShapeRange(Y_IDX_CONV2D);
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

static void FillConv2DOutRange(const RangeCalcParam& param, std::vector<std::pair<int64_t, int64_t>>& yRange)
{
    yRange[static_cast<size_t>(param.xIdx.batch)] = std::make_pair(
        param.ranges.xShapeRange->GetMin()->GetDim(param.xIdx.batch),
        param.ranges.xShapeRange->GetMax()->GetDim(param.xIdx.batch));
    const int64_t knMin = param.ranges.filterShapeRange->GetMin()->GetDim(param.filterIdx.batch);
    const int64_t knMax = param.ranges.filterShapeRange->GetMax()->GetDim(param.filterIdx.batch);
    yRange[static_cast<size_t>(param.xIdx.channel)] = std::make_pair(knMin, knMax);
    if (knMin == UNKNOWN_DIM && knMax == UNKNOWN_DIM) {
        yRange[static_cast<size_t>(param.xIdx.channel)] = std::make_pair(DYN_RANGE_LOWER_BOUND, UNKNOWN_DIM);
    }
    const HwDimRangeIn heightRange{param.useSame,
                                   param.attrs.strh,
                                   param.attrs.dilh,
                                   param.padH,
                                   param.ranges.filterShapeRange->GetMin()->GetDim(param.filterIdx.height),
                                   param.ranges.filterShapeRange->GetMax()->GetDim(param.filterIdx.height),
                                   param.ranges.xShapeRange->GetMin()->GetDim(param.xIdx.height),
                                   param.ranges.xShapeRange->GetMax()->GetDim(param.xIdx.height)};
    InferConv2DHwDimRange(heightRange, yRange[static_cast<size_t>(param.xIdx.height)].first,
                          yRange[static_cast<size_t>(param.xIdx.height)].second);
    const HwDimRangeIn widthRange{param.useSame,
                                  param.attrs.strw,
                                  param.attrs.dilw,
                                  param.padW,
                                  param.ranges.filterShapeRange->GetMin()->GetDim(param.filterIdx.width),
                                  param.ranges.filterShapeRange->GetMax()->GetDim(param.filterIdx.width),
                                  param.ranges.xShapeRange->GetMin()->GetDim(param.xIdx.width),
                                  param.ranges.xShapeRange->GetMax()->GetDim(param.xIdx.width)};
    InferConv2DHwDimRange(widthRange, yRange[static_cast<size_t>(param.xIdx.width)].first,
                          yRange[static_cast<size_t>(param.xIdx.width)].second);
}

static ge::graphStatus FillAndWriteRange(gert::InferShapeRangeContext* context, const ShapeRangeBundle& ranges,
                                         ge::Format xFormat, ge::Format filterFormat, const Conv2DAttrs& conv2DAttrs)
{
    const char* nodeName = NodeNameOrNil(context->GetNodeName());
    int64_t padH = 0;
    int64_t padW = 0;
    if (GetConv2DPadsForRange(context, padH, padW) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const RangeCalcParam calc{ranges,
                              GetDimIdx(xFormat),
                              GetDimIdx(filterFormat),
                              conv2DAttrs,
                              padH,
                              padW,
                              Conv2DUseSameRangeFormula(context->GetAttrs())};
    std::vector<std::pair<int64_t, int64_t>> yRange(SUPPORTED_DIM_NUM);
    FillConv2DOutRange(calc, yRange);
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

static ge::graphStatus InferShapeRangeForConv2D(gert::InferShapeRangeContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("Conv2D", "%s is nullptr!", "context"), return ge::GRAPH_FAILED);
    const char* nodeName = NodeNameOrNil(context->GetNodeName());
    OP_LOGD(nodeName, "Begin dynamic shape set range. ");
    ge::Format xFormat = ge::Format::FORMAT_RESERVED;
    ge::Format filterFormat = ge::Format::FORMAT_RESERVED;
    if (GetRangeFormats(context, xFormat, filterFormat) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    ShapeRangeBundle ranges;
    if (GetAndCheckShapeRanges(context, ranges) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    Conv2DAttrs conv2DAttrs;
    const FillHwAttrParam strideAttr{xFormat, STRIDES_IDX_CONV2D, STRIDE_SIZE_LIMIT, "strides"};
    const FillHwAttrParam dilationAttr{xFormat, DILATIONS_IDX_CONV2D, DILATION_SIZE_LIMIT, "dilations"};
    if (FillPositiveHwFromAttr(context, strideAttr, conv2DAttrs.strh, conv2DAttrs.strw) != ge::GRAPH_SUCCESS ||
        FillPositiveHwFromAttr(context, dilationAttr, conv2DAttrs.dilh, conv2DAttrs.dilw) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return FillAndWriteRange(context, ranges, xFormat, filterFormat, conv2DAttrs);
}
} // namespace Conv

IMPL_OP_INFERSHAPE(Conv2D)
    .InferShape(Ops::NN::Conv::InferShapeForConv2D)
    .InferShapeRange(Ops::NN::Conv::InferShapeRangeForConv2D)
    .PrivateAttr(Conv2DInfer::PADDING, Conv2DInfer::DEFAULT_ATTR_VAL)
    .PrivateAttr(Conv2DInfer::AUTO_PAD, Conv2DInfer::DEFAULT_ATTR_VAL)
    .PrivateAttr(Conv2DInfer::OP_IMPL_MODE, static_cast<int64_t>(0))
    .PrivateAttr(Conv2DInfer::FIXED_SHIFT_VALUE, static_cast<int64_t>(0));
} // namespace NN
} // namespace Ops

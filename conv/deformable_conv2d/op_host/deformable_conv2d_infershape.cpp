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
 * \file deformable_conv2d_infershape.cpp
 * \brief DeformableConv2D InferShape / InferShapeRange.
 */

#include <algorithm>
#include <string>
#include <utility>
#include <vector>
#include "error_util.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/infer_shape_range_context.h"
#include "graph/utils/type_utils.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/shape_util.h"

namespace DeformableConv2DInfer {
using gert::InferShapeContext;

// proto input
const size_t X_IDX_DFM = 0;
const size_t FILTER_IDX_DFM = 1;
const size_t OFFSETS_IDX_DFM = 2;
const size_t BIAS_IDX_DFM = 3;
// proto output
const size_t Y_IDX_DFM = 0;
// proto attribute order: strides, pads, dilations, groups, data_format, deformable_groups, modulated
const size_t STRIDES_IDX_DFM = 0;
const size_t PADS_IDX_DFM = 1;
const size_t DILATIONS_IDX_DFM = 2;
const size_t GROUPS_IDX_DFM = 3;
const size_t DEFORMABLE_GROUPS_IDX_DFM = 5;
const size_t MODULATED_IDX_DFM = 6;

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
// modulated v2 packs the mask into the offsets channel: 2 offsets + 1 mask per sampling point.
const int64_t OFFSET_PLUS_MASK = 3;

// support information
const size_t SUPPORTED_DIM_NUM = 4;
const size_t PAD_SIZE_LIMIT = 4;
const size_t STRIDE_SIZE_LIMIT = 4;
const size_t DILATION_SIZE_LIMIT = 4;

const int64_t UNKNOWN_DIM = -1;
const int64_t ZERO_TENSOR_DYN_RANGE_LOWER_BOUND = 0;
const int64_t DYN_RANGE_LOWER_BOUND = 1;

struct DfmInputShapes {
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

struct DfmOutputShape {
    int64_t on = 0;
    int64_t oc = 0;
    int64_t oh = 0;
    int64_t ow = 0;
    bool isZeroTensor = false;
};

struct DfmAttrs {
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

struct ShapeRangeBundle {
    const gert::Range<gert::Shape>* xShapeRange = nullptr;
    const gert::Range<gert::Shape>* filterShapeRange = nullptr;
    gert::Range<gert::Shape>* yShapeRange = nullptr;
};
} // namespace DeformableConv2DInfer

namespace Ops {
namespace NN {
namespace Conv {
using namespace DeformableConv2DInfer;

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

static ge::graphStatus GetInputFmtShape(InferShapeContext* context, size_t idx, const char* param, const char* expected,
                                        TensorFmtShape& tensor)
{
    const gert::CompileTimeTensorDesc* desc = context->GetInputDesc(idx);
    OPS_CHECK_NULL_WITH_CONTEXT(context, desc);
    tensor.format = desc->GetOriginFormat();
    if (tensor.format == ge::Format::FORMAT_RESERVED) {
        OP_LOGE_FOR_INVALID_FORMAT(NodeNameOrNil(context->GetNodeName()), param, "NONE", expected);
        return ge::GRAPH_FAILED;
    }
    const gert::Shape* shape = context->GetInputShape(idx);
    OPS_CHECK_NULL_WITH_CONTEXT(context, shape);
    tensor.shape = shape;
    return ge::GRAPH_SUCCESS;
}

static void FillXDims(const gert::Shape& shape, const DimIdx& idx, DfmInputShapes& shapes)
{
    shapes.in = shape.GetDim(idx.batch);
    shapes.ic = shape.GetDim(idx.channel);
    shapes.ih = shape.GetDim(idx.height);
    shapes.iw = shape.GetDim(idx.width);
}

static void FillFilterDims(const gert::Shape& shape, const DimIdx& idx, DfmInputShapes& shapes)
{
    shapes.kn = shape.GetDim(idx.batch);
    shapes.kc = shape.GetDim(idx.channel);
    shapes.kh = shape.GetDim(idx.height);
    shapes.kw = shape.GetDim(idx.width);
}

// x dims are mapped by the x origin format.
static ge::graphStatus GetDfmXShapeDim(InferShapeContext* context, TensorFmtShape& xTensor, DfmInputShapes& shapes)
{
    if (GetInputFmtShape(context, X_IDX_DFM, "x", "NCHW or NHWC", xTensor) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (xTensor.format != ge::Format::FORMAT_NCHW && xTensor.format != ge::Format::FORMAT_NHWC) {
        return ReportInvalidFormat(context->GetNodeName(), "x", "NCHW or NHWC", xTensor.format);
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

// filter only supports NCHW and HWCN; only kh/kw==0 is rejected.
static ge::graphStatus GetDfmFilterShapeDim(InferShapeContext* context, DfmInputShapes& shapes)
{
    TensorFmtShape filterTensor;
    if (GetInputFmtShape(context, FILTER_IDX_DFM, "filter", "NCHW or HWCN", filterTensor) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (filterTensor.format != ge::Format::FORMAT_NCHW && filterTensor.format != ge::Format::FORMAT_HWCN) {
        return ReportInvalidFormat(context->GetNodeName(), "filter", "NCHW or HWCN", filterTensor.format);
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
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "filter", khw.c_str(), "kh and kw can't be 0");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// x dtype must equal filter dtype; offsets/bias dtypes are not checked.
static ge::graphStatus CheckDfmDtype(InferShapeContext* context)
{
    const gert::CompileTimeTensorDesc* xDesc = context->GetInputDesc(X_IDX_DFM);
    OPS_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    const gert::CompileTimeTensorDesc* wDesc = context->GetInputDesc(FILTER_IDX_DFM);
    OPS_CHECK_NULL_WITH_CONTEXT(context, wDesc);
    const ge::DataType xDtype = xDesc->GetDataType();
    const ge::DataType wDtype = wDesc->GetDataType();
    if (xDtype != wDtype) {
        const ge::AscendString xStr = ge::TypeUtils::DataTypeToAscendString(xDtype);
        const ge::AscendString wStr = ge::TypeUtils::DataTypeToAscendString(wDtype);
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context->GetNodeName(), "x, filter", "same dtype",
                                               "input x dtype must be the same as filter dtype");
        OP_LOGD(context->GetNodeName(), "x dtype is: %s, filter dtype is: %s.",
                (xStr.GetString() == nullptr) ? "UNKNOWN" : xStr.GetString(),
                (wStr.GetString() == nullptr) ? "UNKNOWN" : wStr.GetString());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// optional bias must be 1D and its size must be equal to out_channels; a 0-dim bias is accepted.
static ge::graphStatus CheckDfmBias(InferShapeContext* context, const DfmInputShapes& shapes)
{
    const gert::Shape* biasShape = context->GetInputShape(BIAS_IDX_DFM);
    if (biasShape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (biasShape->GetDimNum() > 1) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "bias", std::to_string(biasShape->GetDimNum()).c_str(),
                                     "1");
        return ge::GRAPH_FAILED;
    }
    if (biasShape->GetDimNum() == 1 && biasShape->GetDim(0) != shapes.kn) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "bias",
                                              std::to_string(biasShape->GetDim(0)).c_str(),
                                              "input bias size should be equal to out_channels");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetDfmHwAttr(const gert::ExtendedKernelContext* context, size_t attrIdx, size_t sizeLimit,
                                    const char* attrName, const DimIdx& xIdx, int64_t& height, int64_t& width,
                                    int64_t& batch, int64_t& channel)
{
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const gert::ContinuousVector* vec = attrs->GetAttrPointer<gert::ContinuousVector>(attrIdx);
    OPS_CHECK_NULL_WITH_CONTEXT(context, vec);
    if (vec->GetSize() != sizeLimit) {
        const std::string reason = std::string(attrName) + " length must be " + std::to_string(sizeLimit);
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(NodeNameOrNil(context->GetNodeName()), attrName,
                                              std::to_string(vec->GetSize()).c_str(), reason.c_str());
        return ge::GRAPH_FAILED;
    }
    const int64_t* arr = static_cast<const int64_t*>(vec->GetData());
    OPS_CHECK_NULL_WITH_CONTEXT(context, arr);
    batch = arr[xIdx.batch];
    channel = arr[xIdx.channel];
    height = arr[xIdx.height];
    width = arr[xIdx.width];
    return ge::GRAPH_SUCCESS;
}

// Shared attr validation for both InferShape and InferShapeRange entries.
static ge::graphStatus GetDfmAttrs(const gert::ExtendedKernelContext* context, const DimIdx& xIdx, DfmAttrs& dfmAttrs)
{
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const char* nodeName = NodeNameOrNil(context->GetNodeName());
    int64_t strn = 0;
    int64_t strc = 0;
    int64_t diln = 0;
    int64_t dilc = 0;
    if (GetDfmHwAttr(context, STRIDES_IDX_DFM, STRIDE_SIZE_LIMIT, "strides", xIdx, dfmAttrs.strh, dfmAttrs.strw, strn,
                     strc) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (dfmAttrs.strh <= 0 || dfmAttrs.strw <= 0) {
        const std::string strHw = "[" + std::to_string(dfmAttrs.strh) + ", " + std::to_string(dfmAttrs.strw) + "]";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeName, "strides", strHw.c_str(),
                                              "strides H/W dimensions must be positive");
        return ge::GRAPH_FAILED;
    }
    if (strn != 1 || strc != 1) {
        const std::string strNc = "[" + std::to_string(strn) + ", " + std::to_string(strc) + "]";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "strides", strNc.c_str(),
                                              "strides N/C dimensions must be set to 1");
        return ge::GRAPH_FAILED;
    }
    if (GetDfmHwAttr(context, DILATIONS_IDX_DFM, DILATION_SIZE_LIMIT, "dilations", xIdx, dfmAttrs.dilh, dfmAttrs.dilw,
                     diln, dilc) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (dfmAttrs.dilh <= 0 || dfmAttrs.dilw <= 0) {
        const std::string dilHw = "[" + std::to_string(dfmAttrs.dilh) + ", " + std::to_string(dfmAttrs.dilw) + "]";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeName, "dilations", dilHw.c_str(),
                                              "dilations H/W dimensions must be positive");
        return ge::GRAPH_FAILED;
    }
    if (diln != 1 || dilc != 1) {
        const std::string dilNc = "[" + std::to_string(diln) + ", " + std::to_string(dilc) + "]";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "dilations", dilNc.c_str(),
                                              "dilations N/C dimensions must be set to 1");
        return ge::GRAPH_FAILED;
    }
    const gert::ContinuousVector* padsVec = attrs->GetAttrPointer<gert::ContinuousVector>(PADS_IDX_DFM);
    OPS_CHECK_NULL_WITH_CONTEXT(context, padsVec);
    if (padsVec->GetSize() != PAD_SIZE_LIMIT) {
        const std::string reason = "pads length must be " + std::to_string(PAD_SIZE_LIMIT);
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "pads", std::to_string(padsVec->GetSize()).c_str(),
                                              reason.c_str());
        return ge::GRAPH_FAILED;
    }
    const int64_t* padsArr = static_cast<const int64_t*>(padsVec->GetData());
    OPS_CHECK_NULL_WITH_CONTEXT(context, padsArr);
    dfmAttrs.padt = padsArr[TOP_IDX_PAD];
    dfmAttrs.padb = padsArr[BOTTOM_IDX_PAD];
    dfmAttrs.padl = padsArr[LEFT_IDX_PAD];
    dfmAttrs.padr = padsArr[RIGHT_IDX_PAD];
    if (dfmAttrs.padt < 0 || dfmAttrs.padb < 0 || dfmAttrs.padl < 0 || dfmAttrs.padr < 0) {
        const std::string pads = "[" + std::to_string(dfmAttrs.padt) + "," + std::to_string(dfmAttrs.padb) + "," +
                                 std::to_string(dfmAttrs.padl) + "," + std::to_string(dfmAttrs.padr) + "]";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeName, "pads", pads.c_str(),
                                              "pads must be greater than or equal to 0");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// mark zero tensor output when any input dim is 0.
static void InferStaticOutputIsZeroTensor(const DfmInputShapes& shapes, DfmOutputShape& yShape)
{
    if (shapes.in == 0 || shapes.ic == 0 || shapes.ih == 0 || shapes.iw == 0 || shapes.kn == 0 || shapes.kc == 0) {
        yShape.isZeroTensor = true;
    }
}

// reject zero tensor input with negative or all-non-zero output.
static ge::graphStatus CheckOutputZeroTensor(InferShapeContext* context, const DfmInputShapes& shapes,
                                             const DfmOutputShape& yShape)
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
    }
    return ge::GRAPH_SUCCESS;
}

// non-zero input after padding must still cover the filter.
static ge::graphStatus CheckDfmInputWithPad(InferShapeContext* context, const DfmInputShapes& shapes, int64_t ihPad,
                                            int64_t iwPad)
{
    if (shapes.ih > 0 && shapes.kh > 0 && shapes.iw > 0 && shapes.kw > 0 && (ihPad < 0 || iwPad < 0)) {
        const std::string pads = "ihPad:" + std::to_string(ihPad) + ", iwPad:" + std::to_string(iwPad);
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "x, filter", pads.c_str(),
                                               "image size after padding should be greater than or equal to "
                                               "filter size");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckDfmGroups(InferShapeContext* context, const DfmInputShapes& shapes)
{
    if (shapes.unknownRankX) {
        return ge::GRAPH_SUCCESS;
    }
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* groupsPtr = attrs->GetAttrPointer<int64_t>(GROUPS_IDX_DFM);
    OPS_CHECK_NULL_WITH_CONTEXT(context, groupsPtr);
    const int64_t groups = *groupsPtr;
    if (groups < 1) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "groups", std::to_string(groups).c_str(),
                                              "groups should be greater than or equal to 1");
        return ge::GRAPH_FAILED;
    }
    if (shapes.ic > 0 && shapes.ic != shapes.kc * groups) {
        const std::string xShape = "[" + std::to_string(shapes.in) + "," + std::to_string(shapes.ic) + "," +
                                   std::to_string(shapes.ih) + "," + std::to_string(shapes.iw) + "]";
        const std::string wShape = "[" + std::to_string(shapes.kn) + "," + std::to_string(shapes.kc) + "," +
                                   std::to_string(shapes.kh) + "," + std::to_string(shapes.kw) + "]";
        const std::string reason = "x channel should be equal to filter channel * groups(" + std::to_string(groups) +
                                   ")";
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "x, filter", (xShape + ", " + wShape).c_str(),
                                               reason.c_str());
        return ge::GRAPH_FAILED;
    }
    if (shapes.kn % groups != 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "filter", std::to_string(shapes.kn).c_str(),
                                              "out_channels should be divisible by groups");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckDfmOffsets(InferShapeContext* context, const DfmInputShapes& shapes,
                                       const DfmOutputShape& yShape)
{
    const char* nodeName = NodeNameOrNil(context->GetNodeName());
    TensorFmtShape offsetsTensor;
    if (GetInputFmtShape(context, OFFSETS_IDX_DFM, "offsets", "NCHW or NHWC", offsetsTensor) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (offsetsTensor.shape->GetDimNum() != SUPPORTED_DIM_NUM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, "offsets", std::to_string(offsetsTensor.shape->GetDimNum()).c_str(),
                                     std::to_string(SUPPORTED_DIM_NUM).c_str());
        return ge::GRAPH_FAILED;
    }
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* dfmGroupsPtr = attrs->GetAttrPointer<int64_t>(DEFORMABLE_GROUPS_IDX_DFM);
    OPS_CHECK_NULL_WITH_CONTEXT(context, dfmGroupsPtr);
    const int64_t dfmGroups = *dfmGroupsPtr;
    if (!shapes.unknownRankX && shapes.ic > 0 && (dfmGroups < 1 || shapes.ic % dfmGroups != 0)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "deformable_groups", std::to_string(dfmGroups).c_str(),
                                              "deformable_groups should be positive and can divide in_channels");
        return ge::GRAPH_FAILED;
    }
    // the restriction comes from DeformableOffsets.
    const bool* modulatedPtr = attrs->GetAttrPointer<bool>(MODULATED_IDX_DFM);
    OPS_CHECK_NULL_WITH_CONTEXT(context, modulatedPtr);
    if (!(*modulatedPtr)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "modulated", "false", "Currently modulated must be true.");
        return ge::GRAPH_FAILED;
    }
    if (offsetsTensor.format != ge::Format::FORMAT_NCHW && offsetsTensor.format != ge::Format::FORMAT_NHWC) {
        return ReportInvalidFormat(nodeName, "offsets", "NCHW or NHWC", offsetsTensor.format);
    }
    if (shapes.unknownRankX) {
        return ge::GRAPH_SUCCESS;
    }
    const int64_t expChannel = dfmGroups * shapes.kh * shapes.kw * OFFSET_PLUS_MASK;
    const DimIdx offsetsIdx = GetDimIdx(offsetsTensor.format);
    const int64_t offsetsChannel = offsetsTensor.shape->GetDim(offsetsIdx.channel);
    const bool isDynamic = shapes.unknownShapeX;
    if (isDynamic) {
        if (offsetsChannel != UNKNOWN_DIM && offsetsChannel != expChannel) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeName, "offsets", std::to_string(offsetsChannel).c_str(),
                                                  "input offsets channel shape should match deformable_groups * "
                                                  "kernel_h * kernel_w * 3");
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    }
    const bool dimMatched = offsetsTensor.shape->GetDim(offsetsIdx.batch) == shapes.in &&
                            offsetsTensor.shape->GetDim(offsetsIdx.channel) == expChannel &&
                            offsetsTensor.shape->GetDim(offsetsIdx.height) == yShape.oh &&
                            offsetsTensor.shape->GetDim(offsetsIdx.width) == yShape.ow;
    if (!dimMatched) {
        const std::string offsetsShape = "[" + std::to_string(offsetsTensor.shape->GetDim(offsetsIdx.batch)) + "," +
                                         std::to_string(offsetsTensor.shape->GetDim(offsetsIdx.channel)) + "," +
                                         std::to_string(offsetsTensor.shape->GetDim(offsetsIdx.height)) + "," +
                                         std::to_string(offsetsTensor.shape->GetDim(offsetsIdx.width)) + "]";
        const std::string expShape = "[" + std::to_string(shapes.in) + "," + std::to_string(expChannel) + "," +
                                     std::to_string(yShape.oh) + "," + std::to_string(yShape.ow) + "]";
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(nodeName, "offsets", (offsetsShape + ", expected " + expShape).c_str(),
                                               "input offsets shape should match the inferred shape");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static void InferDfmOutputHw(const DfmInputShapes& shapes, const DfmAttrs& dfmAttrs, DfmOutputShape& yShape)
{
    yShape.oh = UNKNOWN_DIM;
    yShape.ow = UNKNOWN_DIM;
    if (shapes.ih > UNKNOWN_DIM && shapes.kh > 0) {
        yShape.oh = (shapes.ih + dfmAttrs.padt + dfmAttrs.padb - dfmAttrs.dilh * (shapes.kh - 1) - 1) / dfmAttrs.strh +
                    1;
    }
    if (shapes.iw > UNKNOWN_DIM && shapes.kw > 0) {
        yShape.ow = (shapes.iw + dfmAttrs.padl + dfmAttrs.padr - dfmAttrs.dilw * (shapes.kw - 1) - 1) / dfmAttrs.strw +
                    1;
    }
    yShape.on = shapes.in;
    yShape.oc = shapes.kn;
}

static ge::graphStatus SetDfmYShape(InferShapeContext* context, const DfmOutputShape& yShape)
{
    const char* nodeName = NodeNameOrNil(context->GetNodeName());
    gert::Shape* yShapePtr = context->GetOutputShape(Y_IDX_DFM);
    OPS_CHECK_NULL_WITH_CONTEXT(context, yShapePtr);
    const gert::CompileTimeTensorDesc* yDesc = context->GetOutputDesc(Y_IDX_DFM);
    OPS_CHECK_NULL_WITH_CONTEXT(context, yDesc);
    const ge::Format yFormat = yDesc->GetOriginFormat();
    if (yFormat == ge::Format::FORMAT_RESERVED) {
        OP_LOGE_FOR_INVALID_FORMAT(nodeName, "y", "NONE", "NCHW or NHWC");
        return ge::GRAPH_FAILED;
    }
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

static ge::graphStatus InferShapeForDeformableConv2D(InferShapeContext* context)
{
    OPS_CHECK_NULL_WITH_CONTEXT(context, context);
    OP_LOGD(context->GetNodeName(), "Enter deformable conv2d shape infer.");
    DfmInputShapes shapes;
    TensorFmtShape xTensor;
    if (GetDfmXShapeDim(context, xTensor, shapes) != ge::GRAPH_SUCCESS ||
        GetDfmFilterShapeDim(context, shapes) != ge::GRAPH_SUCCESS || CheckDfmDtype(context) != ge::GRAPH_SUCCESS ||
        CheckDfmBias(context, shapes) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    DfmAttrs dfmAttrs;
    if (GetDfmAttrs(context, GetDimIdx(xTensor.format), dfmAttrs) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    DfmOutputShape yShape;
    InferDfmOutputHw(shapes, dfmAttrs, yShape);
    const bool isDynamic = shapes.unknownShapeX && !shapes.unknownRankX;
    if (shapes.unknownRankX) {
        yShape.on = UNKNOWN_DIM;
        yShape.oh = UNKNOWN_DIM;
        yShape.ow = UNKNOWN_DIM;
    }
    if (!isDynamic && !shapes.unknownRankX) {
        InferStaticOutputIsZeroTensor(shapes, yShape);
        if (CheckOutputZeroTensor(context, shapes, yShape) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    if (!yShape.isZeroTensor) {
        const int64_t ihPad = (shapes.ih > UNKNOWN_DIM && shapes.kh > 0) ?
                                  shapes.ih + dfmAttrs.padt + dfmAttrs.padb - dfmAttrs.dilh * (shapes.kh - 1) - 1 :
                                  0;
        const int64_t iwPad = (shapes.iw > UNKNOWN_DIM && shapes.kw > 0) ?
                                  shapes.iw + dfmAttrs.padl + dfmAttrs.padr - dfmAttrs.dilw * (shapes.kw - 1) - 1 :
                                  0;
        if (CheckDfmInputWithPad(context, shapes, ihPad, iwPad) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    if (CheckDfmGroups(context, shapes) != ge::GRAPH_SUCCESS ||
        CheckDfmOffsets(context, shapes, yShape) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (SetDfmYShape(context, yShape) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(),
            "x_shape:(N:%ld, C:%ld, H:%ld, W:%ld); filter_shape:(N:%ld, C:%ld, H:%ld, W:%ld); "
            "y_shape(N:%ld, C:%ld, H:%ld, W:%ld); strides(H:%ld, W:%ld); dilations(H:%ld, W:%ld); "
            "pads(T:%ld, B:%ld, L:%ld, R:%ld). Leave shape infer. ",
            shapes.in, shapes.ic, shapes.ih, shapes.iw, shapes.kn, shapes.kc, shapes.kh, shapes.kw, yShape.on,
            yShape.oc, yShape.oh, yShape.ow, dfmAttrs.strh, dfmAttrs.strw, dfmAttrs.dilh, dfmAttrs.dilw, dfmAttrs.padt,
            dfmAttrs.padb, dfmAttrs.padl, dfmAttrs.padr);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetDfmRangeFormats(gert::InferShapeRangeContext* context, ge::Format& filterFormat)
{
    const char* nodeName = NodeNameOrNil(context->GetNodeName());
    const gert::CompileTimeTensorDesc* filterDesc = context->GetInputDesc(FILTER_IDX_DFM);
    OPS_CHECK_NULL_WITH_CONTEXT(context, filterDesc);
    filterFormat = filterDesc->GetOriginFormat();
    if (filterFormat != ge::Format::FORMAT_NCHW && filterFormat != ge::Format::FORMAT_HWCN) {
        return ReportInvalidFormat(nodeName, "filter", "NCHW or HWCN", filterFormat);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetDfmShapeRanges(gert::InferShapeRangeContext* context, ShapeRangeBundle& ranges)
{
    ranges.xShapeRange = context->GetInputShapeRange(X_IDX_DFM);
    ranges.filterShapeRange = context->GetInputShapeRange(FILTER_IDX_DFM);
    ranges.yShapeRange = context->GetOutputShapeRange(Y_IDX_DFM);
    if (ranges.xShapeRange == nullptr) {
        // no x range data: the caller skips range setting
        return ge::GRAPH_SUCCESS;
    }
    if (ranges.filterShapeRange == nullptr || ranges.yShapeRange == nullptr) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// non-positive stride: log and return; the caller validates strides before this helper.
static void InferDfmHwDimRange(const char* nodeName, int64_t stride, int64_t dilation, int64_t pad, int64_t kernelLow,
                               int64_t kernelHigh, int64_t inLow, int64_t inHigh, int64_t& outLow, int64_t& outHigh)
{
    if (stride <= 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "strides", std::to_string(stride).c_str(),
                                              "stride must be greater than 0");
        return;
    }
    outLow = (inLow + pad - dilation * (kernelLow - 1) - 1) / stride + 1;
    outHigh = (inHigh + pad - dilation * (kernelHigh - 1) - 1) / stride + 1;
    const int64_t lowerBound = (inLow == ZERO_TENSOR_DYN_RANGE_LOWER_BOUND) ? ZERO_TENSOR_DYN_RANGE_LOWER_BOUND :
                                                                              DYN_RANGE_LOWER_BOUND;
    outLow = std::max(outLow, lowerBound);
    if (inHigh == UNKNOWN_DIM) {
        outHigh = inHigh;
    }
}

static ge::graphStatus InferShapeRangeForDeformableConv2D(gert::InferShapeRangeContext* context)
{
    OPS_CHECK_NULL_WITH_CONTEXT(context, context);
    const char* nodeName = NodeNameOrNil(context->GetNodeName());
    OP_LOGD(nodeName, "Begin deformable conv2d dynamic shape set range.");
    ge::Format filterFormat = ge::Format::FORMAT_RESERVED;
    if (GetDfmRangeFormats(context, filterFormat) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    ShapeRangeBundle ranges;
    if (GetDfmShapeRanges(context, ranges) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // no x range data: skip range setting and succeed
    if (ranges.xShapeRange == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    const gert::Shape* xMinShape = ranges.xShapeRange->GetMin();
    const gert::Shape* xMaxShape = ranges.xShapeRange->GetMax();
    // an x without range data skips range setting; a half-set range is rejected.
    if (xMinShape == nullptr && xMaxShape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (xMinShape == nullptr || xMaxShape == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "x", "null", "the x shape range must be set completely");
        return ge::GRAPH_FAILED;
    }
    // the filter must be 4D; an unknown-rank x ([-2]) skips the derivation; a known-rank x must be 4D.
    const gert::Shape* wMinShape = ranges.filterShapeRange->GetMin();
    const gert::Shape* wMaxShape = ranges.filterShapeRange->GetMax();
    if (wMinShape == nullptr || wMaxShape == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "filter", "null",
                                              "the filter shape range must be set when the x range is set");
        return ge::GRAPH_FAILED;
    }
    if (wMinShape->GetDimNum() != SUPPORTED_DIM_NUM || wMaxShape->GetDimNum() != SUPPORTED_DIM_NUM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, "filter", std::to_string(wMinShape->GetDimNum()).c_str(),
                                     std::to_string(SUPPORTED_DIM_NUM).c_str());
        return ge::GRAPH_FAILED;
    }
    if (Ops::Base::IsUnknownRank(*xMinShape) && Ops::Base::IsUnknownRank(*xMaxShape)) {
        return ge::GRAPH_SUCCESS;
    }
    if (xMinShape->GetDimNum() != SUPPORTED_DIM_NUM || xMaxShape->GetDimNum() != SUPPORTED_DIM_NUM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(nodeName, "x", std::to_string(xMinShape->GetDimNum()).c_str(),
                                     std::to_string(SUPPORTED_DIM_NUM).c_str());
        return ge::GRAPH_FAILED;
    }
    const gert::CompileTimeTensorDesc* xDesc = context->GetInputDesc(X_IDX_DFM);
    OPS_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    const DimIdx xIdx = GetDimIdx(xDesc->GetOriginFormat());
    const DimIdx filterIdx = GetDimIdx(filterFormat);
    DfmAttrs dfmAttrs;
    if (GetDfmAttrs(context, xIdx, dfmAttrs) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const int64_t padH = dfmAttrs.padt + dfmAttrs.padb;
    const int64_t padW = dfmAttrs.padl + dfmAttrs.padr;

    int64_t kernelH = -1;
    int64_t kernelW = -1;
    int64_t knRange = -1;
    const gert::TensorRange* filterTensorRange = context->GetInputTensorRange(FILTER_IDX_DFM);
    bool filterFromTensorRange = false;
    if (filterTensorRange != nullptr && filterTensorRange->GetMin() != nullptr) {
        const gert::Shape& filterShapeStatic = filterTensorRange->GetMin()->GetOriginShape();
        if (filterShapeStatic.GetDimNum() == SUPPORTED_DIM_NUM) {
            kernelH = filterShapeStatic.GetDim(filterIdx.height);
            kernelW = filterShapeStatic.GetDim(filterIdx.width);
            knRange = filterShapeStatic.GetDim(filterIdx.batch);
            filterFromTensorRange = true;
        }
    }
    if (!filterFromTensorRange) {
        // UT faker path: no TensorRange, read the kernel from the filter shape range min
        kernelH = ranges.filterShapeRange->GetMin()->GetDim(filterIdx.height);
        kernelW = ranges.filterShapeRange->GetMin()->GetDim(filterIdx.width);
        knRange = ranges.filterShapeRange->GetMin()->GetDim(filterIdx.batch);
    }
    // dynamic when max == -1 or min != max; static only when min == max != -1.
    auto isDynamicDim = [](const gert::Range<gert::Shape>* range, size_t dimIdx) -> bool {
        const int64_t lo = range->GetMin()->GetDim(dimIdx);
        const int64_t hi = range->GetMax()->GetDim(dimIdx);
        return (hi == UNKNOWN_DIM) || (lo != hi);
    };
    const bool hDynamic = isDynamicDim(ranges.xShapeRange, static_cast<size_t>(xIdx.height));
    const bool wDynamic = isDynamicDim(ranges.xShapeRange, static_cast<size_t>(xIdx.width));
    const int64_t xHStatic = ranges.xShapeRange->GetMin()->GetDim(xIdx.height);
    const int64_t xWStatic = ranges.xShapeRange->GetMin()->GetDim(xIdx.width);
    const int64_t ohStatic = (xHStatic > UNKNOWN_DIM && kernelH > 0) ?
                                 (xHStatic + padH - dfmAttrs.dilh * (kernelH - 1) - 1) / dfmAttrs.strh + 1 :
                                 UNKNOWN_DIM;
    const int64_t owStatic = (xWStatic > UNKNOWN_DIM && kernelW > 0) ?
                                 (xWStatic + padW - dfmAttrs.dilw * (kernelW - 1) - 1) / dfmAttrs.strw + 1 :
                                 UNKNOWN_DIM;

    std::vector<std::pair<int64_t, int64_t>> yRange(SUPPORTED_DIM_NUM);
    yRange[static_cast<size_t>(xIdx.batch)] = std::make_pair(ranges.xShapeRange->GetMin()->GetDim(xIdx.batch),
                                                             ranges.xShapeRange->GetMax()->GetDim(xIdx.batch));
    // output C range is the static kn from the weight
    yRange[static_cast<size_t>(xIdx.channel)] = std::make_pair(knRange, knRange);
    // derive the H/W range only when the x dim is dynamic
    if (hDynamic) {
        yRange[static_cast<size_t>(xIdx.height)].first = -1;
        yRange[static_cast<size_t>(xIdx.height)].second = -1;
        InferDfmHwDimRange(
            nodeName, dfmAttrs.strh, dfmAttrs.dilh, padH, kernelH, kernelH,
            ranges.xShapeRange->GetMin()->GetDim(xIdx.height), ranges.xShapeRange->GetMax()->GetDim(xIdx.height),
            yRange[static_cast<size_t>(xIdx.height)].first, yRange[static_cast<size_t>(xIdx.height)].second);
    } else {
        // the H range uses the inferred static output oh, not the input H
        yRange[static_cast<size_t>(xIdx.height)] = std::make_pair(ohStatic, ohStatic);
    }
    if (wDynamic) {
        yRange[static_cast<size_t>(xIdx.width)].first = -1;
        yRange[static_cast<size_t>(xIdx.width)].second = -1;
        InferDfmHwDimRange(
            nodeName, dfmAttrs.strw, dfmAttrs.dilw, padW, kernelW, kernelW,
            ranges.xShapeRange->GetMin()->GetDim(xIdx.width), ranges.xShapeRange->GetMax()->GetDim(xIdx.width),
            yRange[static_cast<size_t>(xIdx.width)].first, yRange[static_cast<size_t>(xIdx.width)].second);
    } else {
        // the W range uses the inferred static output ow, not the input W
        yRange[static_cast<size_t>(xIdx.width)] = std::make_pair(owStatic, owStatic);
    }
    if (ranges.yShapeRange->GetMin() == nullptr || ranges.yShapeRange->GetMax() == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "y", "null", "the output shape range must be set");
        return ge::GRAPH_FAILED;
    }
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

IMPL_OP_INFERSHAPE(DeformableConv2D)
    .InferShape(Ops::NN::Conv::InferShapeForDeformableConv2D)
    .InferShapeRange(Ops::NN::Conv::InferShapeRangeForDeformableConv2D);
} // namespace NN
} // namespace Ops

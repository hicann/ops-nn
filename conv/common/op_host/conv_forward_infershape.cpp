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
 * \file conv_common_infer.cpp
 * \brief
 */
#include "cube_util.h"
#include "util/shape_util.h"
#include "error_util.h"
#include "graph/utils/type_utils.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "register/op_impl_registry.h"
#include "log/log.h"

#define OPS_CHECK_NULL_WITH_CONTEXT_BOOL(context, ptr)                                               \
    if ((ptr) == nullptr) {                                                                          \
        const char* name = ((context)->GetNodeName() == nullptr) ? "nil" : (context)->GetNodeName(); \
        OP_LOGE(name, "%s is nullptr!", #ptr);                                                       \
        REPORT_INNER_ERR_MSG("EZ9999", "op[%s], %s is nullptr!", name, #ptr);                           \
        return false;                                                                                \
    }

namespace ConvForward {
// Conv common proto index
constexpr size_t X_IDX_CONV = 0;
constexpr size_t W_IDX_CONV = 1;
constexpr size_t BIAS_IDX_CONV = 2;
constexpr size_t BIAS_IDX_QUANT_CONV = 3;
constexpr size_t Y_IDX_CONV = 0;

constexpr size_t STRIDES_IDX_CONV = 0;
constexpr size_t PADS_IDX_CONV = 1;
constexpr size_t DILATIONS_IDX_CONV = 2;
constexpr size_t GROUPS_IDX_CONV = 3;
constexpr size_t PAD_MODE_IDX_CONV = 6;

// QuantConv attr idx define
constexpr size_t DTYPE_IDX_QUANT_CONV = 0;
constexpr size_t STRIDES_IDX_QUANT_CONV = 1;
constexpr size_t PADS_IDX_QUANT_CONV = 2;
constexpr size_t DILATIONS_IDX_QUANT_CONV = 3;
constexpr size_t GROUPS_IDX_QUANT_CONV = 4;
constexpr size_t ROUNDMODE_IDX_QUANT_CONV = 7;
constexpr size_t OFFSET_IDX_QUANT_CONV = 4;

// ExtendConv2D tensor and attr idx define
constexpr size_t BIAS_IDX_EXTEND_CONV = 2;
constexpr size_t SCALE0_IDX_EXTEND_CONV = 4;
constexpr size_t SCALE1_IDX_EXTEND_CONV = 7;
constexpr size_t RELUWEIGHT0_IDX_EXTEND_CONV = 5;
constexpr size_t RELUWEIGHT1_IDX_EXTEND_CONV = 8;
constexpr size_t CLIPVALUE0_IDX_EXTEND_CONV = 6;
constexpr size_t CLIPVALUE1_IDX_EXTEND_CONV = 9;
constexpr size_t Y0_IDX_EXTEND_CONV = 0;
constexpr size_t Y1_IDX_EXTEND_CONV = 1;

constexpr size_t STRIDES_IDX_EXTEND_CONV = 0;
constexpr size_t PADS_IDX_EXTEND_CONV = 1;
constexpr size_t DILATIONS_IDX_EXTEND_CONV = 2;
constexpr size_t GROUPS_IDX_EXTEND_CONV = 3;
constexpr size_t ROUND_MODE_IDX_EXTEND_CONV = 6;
constexpr size_t PAD_MODE_IDX_EXTEND_CONV = 7;
constexpr size_t ENABLE_RELU0_IDX_EXTEND_CONV = 9;
constexpr size_t ENABLE_RELU1_IDX_EXTEND_CONV = 10;
constexpr size_t DUAL_OUT_IDX_EXTEND_CONV = 11;
constexpr size_t DTYPE0_IDX_EXTEND_CONV = 12;
constexpr size_t DTYPE1_IDX_EXTEND_CONV = 13;

// Conv2D
constexpr size_t CONV2DV2_SUPPORTED_DIM_NUM = 4;
constexpr size_t CONV2DV2_PAD_SIZE_LIMIT = 4;

// Conv3D
constexpr size_t CONV3DV2_SUPPORTED_DIM_NUM = 5;
constexpr size_t CONV3DV2_PAD_SIZE_LIMIT = 6;
constexpr int32_t C_DIM_IDX_NCDHW = 1;

// Quant Conv3D proto attribute index
constexpr size_t PAD_MODE_IDX_QUANTCONV3D = 8;

constexpr int64_t PAD_HALF_DIV = 2;

// Conv common count params
constexpr size_t COUNT_PARAMS_WITHOUT_BIAS = 3;
constexpr size_t COUNT_PARAMS_WITH_BIAS = 4;

constexpr int64_t ZERO_TENSOR_DYN_RANGE_LOWER_BOUND = 0;
constexpr int64_t DYN_RANGE_LOWER_BOUND = 1;

constexpr size_t IDX_LIST_N_IDX = 0;
constexpr size_t IDX_LIST_C_IDX = 1;
constexpr size_t IDX_LIST_D_IDX = 2;
constexpr size_t IDX_LIST_H_IDX = 3;
constexpr size_t IDX_LIST_W_IDX = 4;

using gert::InferShapeContext;

struct ConvInputShapes {
    int64_t in = 0;
    int64_t ic = 0;
    int64_t id = 0; // conv2d is default value, skip check
    int64_t ih = 0;
    int64_t iw = 0;

    int64_t kn = 0;
    int64_t kc = 0;
    int64_t kd = 0; // conv2d is default value, skip check
    int64_t kh = 0;
    int64_t kw = 0;
    bool isInputZeroTensor = false;
    bool unknownShapeX = false;
    bool unknownShapeW = false;
    bool unknownRankX = false;
    bool unknownRankW = false;
    bool unknownRankBias = false;
};

struct ConvOutputShape {
    int64_t on = 0;
    int64_t oc = 0;
    int64_t od = 0; // conv2d is default value, skip check
    int64_t oh = 0;
    int64_t ow = 0;
    bool isOutputZeroTensor = false;
};

struct ConvAttrs {
    int64_t strd = 1; // conv2d is default value, skip check
    int64_t strh = 1;
    int64_t strw = 1;

    int64_t dild = 1; // conv2d is default value, skip check
    int64_t dilh = 1;
    int64_t dilw = 1;

    int64_t padh = -1; // conv2d is default value, skip check
    int64_t padt = -1; // conv2d is default value, skip check
    int64_t padu = -1;
    int64_t padd = -1;
    int64_t padl = -1;
    int64_t padr = -1;

    int64_t groups = 1;
    const char* padMode = "SPECIFIC";
};

enum class ConvOptype : int {
    CONV2DV2 = 0,
    CONV3DV2,
    QUANT_CONV2D,
    QUANT_CONV3D,
    EXTEND_CONV2D,
    INVALID
};

struct OpParamIdx {
    int32_t fMapIdx = 0;
    int32_t weightIdx = 0;
    int32_t biasIdx = 0;
    int32_t outputIdx = 0;
    int32_t strideIdx = 0;
    int32_t padIdx = 0;
    int32_t dilationIdx = 0;
    int32_t groupsIdx = 0;
    int32_t padModeIdx = 0;
    int32_t roundModeIdx = 0;
    // extend conv2d idx
    int32_t scale0Idx = 0;
    int32_t scale1Idx = 0;
    int32_t reluWeight0Idx = 0;
    int32_t reluWeight1Idx = 0;
    int32_t clipValue0Idx = 0;
    int32_t clipValue1Idx = 0;
    int32_t output1Idx = 0;
    int32_t enableRelu0Idx = 0;
    int32_t enableRelu1Idx = 0;
    int32_t dualOutIdx = 0;
};

struct ConvOpInfo : public ConvInputShapes, public ConvOutputShape, public ConvAttrs {
    ConvOptype opType = ConvOptype::INVALID;
    OpParamIdx paramIdx;
    bool isConv2DLike = false; // isConv2DLike includes Conv2DV2 & QuantConv2D
    bool isConv3DLike = false; // isConv3DLike includes Conv3DV2 & QuantConv3D
    std::vector<size_t> xFormatIdx = {0, 0, 0, 0, 0}; // N C D H W
    std::vector<size_t> wFormatIdx = {0, 0, 0, 0, 0}; // N C D H W
    std::vector<size_t> biasFormatIdx = {0, 0, 0, 0, 0}; // N C D H W
};
} // namespace

namespace Ops {
namespace NN {
namespace Conv {
using namespace ConvForward;

const std::unordered_map<ge::Format, std::vector<size_t>> FORMAT_TO_IDX_MAP = {
    {ge::Format::FORMAT_NHWC, {kNDimNHWCIdx, kCDimNHWCIdx, kDDimUnsupportIdx, kHDimNHWCIdx, kWDimNHWCIdx}},
    {ge::Format::FORMAT_NCHW, {kNDimNCHWIdx, kCDimNCHWIdx, kDDimUnsupportIdx, kHDimNCHWIdx, kWDimNCHWIdx}},
    {ge::Format::FORMAT_NCDHW, {kNDimNCDHWIdx, kCDimNCDHWIdx, kDDimNCDHWIdx, kHDimNCDHWIdx, kWDimNCDHWIdx}},
    {ge::Format::FORMAT_NDHWC, {kNDimNDHWCIdx, kCDimNDHWCIdx, kDDimNDHWCIdx, kHDimNDHWCIdx, kWDimNDHWCIdx}},
    {ge::Format::FORMAT_DHWCN, {kNDimDHWCNIdx, kCDimDHWCNIdx, kDDimDHWCNIdx, kHDimDHWCNIdx, kWDimDHWCNIdx}},
    {ge::Format::FORMAT_HWCN, {kNDimHWCNIdx, kCDimHWCNIdx, kDDimUnsupportIdx, kHDimHWCNIdx, kWDimHWCNIdx}},
    {ge::Format::FORMAT_ND, {kDDimUnsupportIdx, kDimNDIdx, kDDimUnsupportIdx, kDDimUnsupportIdx, kDDimUnsupportIdx}}
};

std::string DTypeToStr(const ge::DataType dtype) {
    return ge::TypeUtils::DataTypeToSerialString(dtype);
}

static bool GetConvUnknownRankXShape(const InferShapeContext* context, ConvOpInfo& opInfo)
{
    OP_LOGD(context->GetNodeName(), "Get Conv Unknown Rank X Shape." );
    // when unknownRankX is true, set -1 to x shape
    opInfo.in = -1;
    opInfo.ic = -1;
    opInfo.ih = -1;
    opInfo.iw = -1;
    if (opInfo.isConv3DLike) {
        opInfo.id = -1;
    }
    return true;
}

static bool GetConvUnknownRankWShape(const InferShapeContext* context, ConvOpInfo& opInfo)
{
    OP_LOGD(context->GetNodeName(), "Get Conv Unknown Rank W Shape.");
    // when unknownRankW is true, set -1 to w shape
    opInfo.kn = -1;
    opInfo.kc = -1;
    opInfo.kh = -1;
    opInfo.kw = -1;
    if (opInfo.isConv3DLike) {
        opInfo.kd = -1;
    }
    return true;
}

static void GetConvShapeIdx(const ge::Format format, std::vector<size_t>& idx)
{
    auto it = FORMAT_TO_IDX_MAP.find(format);
    if (it != FORMAT_TO_IDX_MAP.end()) {
        const std::vector<size_t>& indices = it->second;
        // all support format check in pre process
        if (idx.size() != indices.size()) {
            return;
        }
        for (size_t i = 0; i < indices.size(); ++i) {
            idx[i] = indices[i];
        }
    }
}

static bool GetConvXShape(const InferShapeContext* context, ConvOpInfo& opInfo) {
    const gert::Shape* xShape = context->GetInputShape(opInfo.paramIdx.fMapIdx);
    OPS_CHECK_NULL_WITH_CONTEXT_BOOL(context, xShape);

    opInfo.unknownShapeX = Ops::Base::IsUnknownShape(*xShape);
    opInfo.unknownRankX = Ops::Base::IsUnknownRank(*xShape);

    OP_LOGD(context->GetNodeName(), "Conv UnknownShape X: %d, UnknownRank X: %d.",
        opInfo.unknownShapeX, opInfo.unknownRankX);
    if (opInfo.unknownRankX) { // unknownRankX scene quit
        return GetConvUnknownRankXShape(context, opInfo);
    }

    if (opInfo.isConv3DLike) {
        OP_LOGE_IF(xShape->GetDimNum() != CONV3DV2_SUPPORTED_DIM_NUM, false, context->GetNodeName(),
            "The shape of xTensor should be %zu, actual dim num: %zu",
            CONV3DV2_SUPPORTED_DIM_NUM, xShape->GetDimNum());
    } else {
        OP_LOGE_IF(xShape->GetDimNum() != CONV2DV2_SUPPORTED_DIM_NUM, false, context->GetNodeName(),
            "The shape of xTensor should be %zu, actual dim num: %zu",
            CONV2DV2_SUPPORTED_DIM_NUM, xShape->GetDimNum());
    }

    opInfo.in = xShape->GetDim(opInfo.xFormatIdx[IDX_LIST_N_IDX]);
    opInfo.ic = xShape->GetDim(opInfo.xFormatIdx[IDX_LIST_C_IDX]);
    opInfo.ih = xShape->GetDim(opInfo.xFormatIdx[IDX_LIST_H_IDX]);
    opInfo.iw = xShape->GetDim(opInfo.xFormatIdx[IDX_LIST_W_IDX]);
    if (opInfo.isConv3DLike) {
        opInfo.id = xShape->GetDim(opInfo.xFormatIdx[IDX_LIST_D_IDX]);
    }
    return true;
}

static bool GetConvFilterShape(const InferShapeContext* context, ConvOpInfo& opInfo)
{
    const auto filterShape = context->GetInputShape(opInfo.paramIdx.weightIdx);
    OPS_CHECK_NULL_WITH_CONTEXT_BOOL(context, filterShape);

    opInfo.unknownShapeW = Ops::Base::IsUnknownShape(*filterShape);
    opInfo.unknownRankW = Ops::Base::IsUnknownRank(*filterShape);
    OP_LOGD(context->GetNodeName(), "Conv UnknownShape W: %d, UnknownRank W: %d.",
        opInfo.unknownShapeW, opInfo.unknownRankW);
    if (opInfo.unknownRankW) { // unknownRankW scene quit
        return GetConvUnknownRankWShape(context, opInfo);
    }

    // already checked in shape range infer logic, no need to use error manager here
    if (opInfo.isConv3DLike) {
        OP_LOGE_IF(filterShape->GetDimNum() != CONV3DV2_SUPPORTED_DIM_NUM, false, context->GetNodeName(),
            "The shape of the filter should be %zu, actual dim num: %zu",
            CONV3DV2_SUPPORTED_DIM_NUM, filterShape->GetDimNum());
    } else {
        OP_LOGE_IF(filterShape->GetDimNum() != CONV2DV2_SUPPORTED_DIM_NUM, false, context->GetNodeName(),
            "The shape of the filter should be %zu, actual dim num: %zu",
            CONV2DV2_SUPPORTED_DIM_NUM, filterShape->GetDimNum());
    }

    opInfo.kn = filterShape->GetDim(opInfo.wFormatIdx[IDX_LIST_N_IDX]);
    opInfo.kc = filterShape->GetDim(opInfo.wFormatIdx[IDX_LIST_C_IDX]);
    opInfo.kh = filterShape->GetDim(opInfo.wFormatIdx[IDX_LIST_H_IDX]);
    opInfo.kw = filterShape->GetDim(opInfo.wFormatIdx[IDX_LIST_W_IDX]);
    if (opInfo.isConv3DLike) {
        opInfo.kd = filterShape->GetDim(opInfo.wFormatIdx[IDX_LIST_D_IDX]);
    }

    if (opInfo.kh == 0 || opInfo.kw == 0) {
        OP_LOGE(context->GetNodeName(),
                "Input kh [%ld] or kw [%ld] can't be equal to 0.", opInfo.kh, opInfo.kw);
        return false;
    }

    return true;
}

static bool CheckConvBiasUnknownRank(const InferShapeContext* context, const gert::Shape* biasShape, ConvOpInfo &opInfo)
{
    opInfo.unknownRankBias = IsUnknownRank(biasShape);
    OP_LOGD(context->GetNodeName(), "Conv UnknownRank Bias: %d. ", opInfo.unknownRankBias);
    if (opInfo.unknownRankBias) { // unknownRankBias scene quit
        OP_LOGD(context->GetNodeName(), "Input bias's shape is [-2].");
        return true;
    }
    return false;
}

static bool CheckConvBiasDimLegal(const InferShapeContext* context, const gert::Shape* biasShape, ConvOpInfo &opInfo)
{
    // Check bias dim num
    size_t biasDimNum = biasShape->GetDimNum();
    size_t biasDimNumExpect = opInfo.isConv3DLike ? CONV3DV2_SUPPORTED_DIM_NUM : CONV2DV2_SUPPORTED_DIM_NUM;
    if (biasDimNum != static_cast<size_t>(1) && biasDimNum != biasDimNumExpect) {
        OP_LOGE(context->GetNodeName(), "Input bias shape should be 1 D or %zu D.", biasDimNumExpect);
        return false;
    }
    if (opInfo.kn == UNKNOWN_DIM_VALUE_) { // cout dynamic scene quit after dim check
        OP_LOGD(context->GetNodeName(), "Input bias's shape is dynamic.");
        return true;
    }
    // Get bias channel index
    ge::Format biasFormat = context->GetOptionalInputDesc(opInfo.paramIdx.biasIdx)->GetOriginFormat();
    GetConvShapeIdx(biasFormat, opInfo.biasFormatIdx);
    // Check bias other dim
    if (biasDimNum != biasDimNumExpect) {
        if (biasShape->GetDim(0) != opInfo.kn) {
            OP_LOGE(context->GetNodeName(), "Input bias size of dim_c: %ld should be equal to out_channels: %ld.",
                    biasShape->GetDim(opInfo.biasFormatIdx[IDX_LIST_C_IDX]), opInfo.kn);
            return false;
        }
        return true;
    }
    // Check bias channel
    if (biasShape->GetDim(opInfo.biasFormatIdx[IDX_LIST_C_IDX]) != opInfo.kn) {
        OP_LOGE(context->GetNodeName(), "Input bias size of dim_c: %ld should be equal to out_channels: %ld.",
                biasShape->GetDim(opInfo.biasFormatIdx[IDX_LIST_C_IDX]), opInfo.kn);
        return false;
    }
    for (size_t i = 0; i < biasDimNum; i++) {
        if (i == opInfo.biasFormatIdx[IDX_LIST_C_IDX]) {
            // NOTE: rt1.0 set kn to bias channel-dim when bias channel-dim is -1, rt2.0 no set interface, skip
            continue;
        }
        if (biasShape->GetDim(i) != 1) {
            OP_LOGE(context->GetNodeName(), "Input bias size of other dim except dim_c should be equal to 1.");
            return false;
        }
    }
    return true;
}

static bool CheckConvBias(InferShapeContext* context, ConvOpInfo &opInfo)
{
    const gert::Shape* biasShape = context->GetOptionalInputShape(opInfo.paramIdx.biasIdx);
    if (biasShape == nullptr) {
        OP_LOGD(context->GetNodeName(), "No bias.");
        return true;
    }
    if (CheckConvBiasUnknownRank(context, biasShape, opInfo)) {
        return true;
    }
    if (!CheckConvBiasDimLegal(context, biasShape, opInfo)) {
        return false;
    }

    return true;
}

template <typename T>
static bool GetConvStrides(const T* context, ConvOpInfo& opInfo)
{
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT_BOOL(context, attrs);
    const gert::ContinuousVector* stridesList = attrs->GetAttrPointer<gert::ContinuousVector>(opInfo.paramIdx.strideIdx);
    OPS_CHECK_NULL_WITH_CONTEXT_BOOL(context, stridesList);
    // already checked in first infer shape logic, double check just for security
    if (opInfo.isConv3DLike) {
        OP_LOGE_IF(stridesList->GetSize() != CONV3DV2_SUPPORTED_DIM_NUM, false, context->GetNodeName(),
            "The shape of the strides should be %zu, actual dim num: %zu",
            CONV3DV2_SUPPORTED_DIM_NUM, stridesList->GetSize());
    } else {
        OP_LOGE_IF(stridesList->GetSize() != CONV2DV2_SUPPORTED_DIM_NUM, false, context->GetNodeName(),
            "The shape of the strides should be %zu, actual dim num: %zu",
            CONV2DV2_SUPPORTED_DIM_NUM, stridesList->GetSize());
    }

    const int64_t* strides = static_cast<const int64_t *>(stridesList->GetData());
    opInfo.strh = strides[opInfo.xFormatIdx[IDX_LIST_H_IDX]];
    opInfo.strw = strides[opInfo.xFormatIdx[IDX_LIST_W_IDX]];
    if (opInfo.isConv3DLike) {
        opInfo.strd = strides[opInfo.xFormatIdx[IDX_LIST_D_IDX]];
    }

    if (opInfo.isConv3DLike && (opInfo.strd <= 0 || opInfo.strh <= 0 || opInfo.strw <= 0)) {
        OP_LOGE(context->GetNodeName(),
            "strides should be positive, actual is [%ld,%ld,%ld].", opInfo.strd, opInfo.strh, opInfo.strw);
        return false;
    } else if (opInfo.isConv2DLike && (opInfo.strh <= 0 || opInfo.strw <= 0)) {
        OP_LOGE(context->GetNodeName(),
            "strides should be positive, actual is [%ld,%ld].", opInfo.strh, opInfo.strw);
        return false;
    }

    return true;
}

template <typename T>
static bool GetConvDilations(const T* context, ConvOpInfo& opInfo)
{
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT_BOOL(context, attrs);
    const gert::ContinuousVector* dilationsList =
        attrs->GetAttrPointer<gert::ContinuousVector>(opInfo.paramIdx.dilationIdx);
    if (opInfo.isConv3DLike) {
        OP_LOGE_IF(dilationsList->GetSize() != CONV3DV2_SUPPORTED_DIM_NUM, false, context->GetNodeName(),
            "The shape of the dilations should be %zu, actual dim num: %zu",
            CONV3DV2_SUPPORTED_DIM_NUM, dilationsList->GetSize());
    } else {
        OP_LOGE_IF(dilationsList->GetSize() != CONV2DV2_SUPPORTED_DIM_NUM, false, context->GetNodeName(),
            "The shape of the dilations should be %zu, actual dim num: %zu",
            CONV2DV2_SUPPORTED_DIM_NUM, dilationsList->GetSize());
    }

    const int64_t* dilations = static_cast<const int64_t *>(dilationsList->GetData());
    opInfo.dilh = dilations[opInfo.xFormatIdx[IDX_LIST_H_IDX]];
    opInfo.dilw = dilations[opInfo.xFormatIdx[IDX_LIST_W_IDX]];
    if (opInfo.isConv3DLike) {
        opInfo.dild = dilations[opInfo.xFormatIdx[IDX_LIST_D_IDX]];
    }

    if (opInfo.isConv3DLike && (opInfo.dild <= 0 || opInfo.dilh <= 0 || opInfo.dilw <= 0)) {
        OP_LOGE(context->GetNodeName(),
            "dilations should be positive, actual is [%ld,%ld,%ld].", opInfo.dild, opInfo.dilh, opInfo.dilw);
        return false;
    } else if (opInfo.isConv2DLike && (opInfo.dilh <= 0 || opInfo.dilw <= 0)) {
        OP_LOGE(context->GetNodeName(),
            "dilations should be positive, actual is [%ld,%ld].", opInfo.dilh, opInfo.dilw);
        return false;
    }

    return true;
}

static bool CheckPositivePads(const char* nodeName, const ConvOpInfo& opInfo)
{
    bool negConv3DPadFlag = opInfo.isConv3DLike &&
        (opInfo.padh < 0 || opInfo.padt < 0 || opInfo.padu < 0 ||opInfo.padd < 0 ||
        opInfo.padl < 0 || opInfo.padr < 0);
    bool negConv2DPadFlag = opInfo.isConv2DLike &&
        (opInfo.padu < 0 || opInfo.padd < 0 || opInfo.padl < 0 || opInfo.padr < 0);
    bool staticRankAndShape = !opInfo.unknownShapeX && !opInfo.unknownRankX &&
                              !opInfo.unknownShapeW && !opInfo.unknownRankW;
    if (staticRankAndShape && negConv3DPadFlag) {
        OP_LOGE(nodeName, "pads should be positive, but the actual is [%ld,%ld,%ld,%ld,%ld,%ld].",
            opInfo.padh, opInfo.padt, opInfo.padu, opInfo.padd, opInfo.padl, opInfo.padr);
        return false;
    } else if (staticRankAndShape && negConv2DPadFlag) {
        OP_LOGE(nodeName, "pads should be positive, but the actual is [%ld,%ld,%ld,%ld].",
            opInfo.padu, opInfo.padd, opInfo.padl, opInfo.padr);
        return false;
    }
    return true;
}

static inline int64_t CalcPadNeededConv(const int64_t input, const int64_t kernel, const int64_t stride,
    const int64_t dilation, bool isDynamic)
{
    int64_t padNeeded = 0;
    if (isDynamic) {
        padNeeded = UNKNOWN_DIM_VALUE_ * PAD_HALF_DIV; // to make sure both sides of pads are -1
        return padNeeded;
    }
    int64_t tails = input % stride; // stride 0 check before
    int64_t dilateKernel = dilation * (kernel - 1) + 1;
    padNeeded = std::max((tails > 0 ? dilateKernel - tails : dilateKernel - stride), 0L);

    return padNeeded;
}

static void GetSamePadMode(ConvOpInfo& opInfo) {
    bool isDDynamic = (opInfo.id < 0 || opInfo.kd < 0);
    bool isHDynamic = (opInfo.ih < 0 || opInfo.kh < 0);
    bool isWDynamic = (opInfo.iw < 0 || opInfo.kw < 0);
    int64_t padDepth = CalcPadNeededConv(opInfo.id, opInfo.kd, opInfo.strd, opInfo.dild, isDDynamic);
    int64_t padHeight = CalcPadNeededConv(opInfo.ih, opInfo.kh, opInfo.strh, opInfo.dilh, isHDynamic);
    int64_t padWidth = CalcPadNeededConv(opInfo.iw, opInfo.kw, opInfo.strw, opInfo.dilw, isWDynamic);

    if (strcmp(opInfo.padMode, "SAME") == 0 || strcmp(opInfo.padMode, "SAME_UPPER") == 0) {
        opInfo.padh = opInfo.isConv2DLike ? 0 : padDepth / PAD_HALF_DIV;
        opInfo.padt = opInfo.isConv2DLike ? 0 : padDepth - opInfo.padh;
        opInfo.padu = padHeight / PAD_HALF_DIV;
        opInfo.padd = padHeight - opInfo.padu;
        opInfo.padl = padWidth / PAD_HALF_DIV;
        opInfo.padr = padWidth - opInfo.padl;
    } else {
        // SAME_LOWER
        opInfo.padt = opInfo.isConv2DLike ? 0 : padDepth / PAD_HALF_DIV;
        opInfo.padh = opInfo.isConv2DLike ? 0 : padDepth - opInfo.padt;
        opInfo.padd = padHeight / PAD_HALF_DIV;
        opInfo.padu = padHeight - opInfo.padd;
        opInfo.padr = padWidth / PAD_HALF_DIV;
        opInfo.padl = padWidth - opInfo.padr;
    }
}

// SPECIFIC/SAME/VALID for TF; SPECIFIC/VALID/SAME_UPPER/SAME_LOWER for ONNX;
template <typename T>
static bool GetPadModeAndUpdatePad(const T* context, ConvOpInfo& opInfo)
{
    if (opInfo.opType == ConvOptype::QUANT_CONV2D) {
        // QuantConv2D are not support pad mode
        return true;
    }

    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT_BOOL(context, attrs);
    const char* padMode = attrs->GetAttrPointer<char>(opInfo.paramIdx.padModeIdx);
    OPS_CHECK_NULL_WITH_CONTEXT_BOOL(context, padMode);
    opInfo.padMode = padMode;

    if constexpr (std::is_same<T, gert::InferShapeRangeContext>::value) {
        if (strcmp(opInfo.padMode, "SPECIFIC") == 0) {
        } else if (strcmp(opInfo.padMode, "VALID") == 0) {
            opInfo.padh = 0;
            opInfo.padt = 0;
            opInfo.padu = 0;
            opInfo.padd = 0;
            opInfo.padl = 0;
            opInfo.padr = 0;
        } else {
            // SAME, SAME_UPPER, SAME_LOWER
            opInfo.padh = -1;
            opInfo.padt = -1;
            opInfo.padu = -1;
            opInfo.padd = -1;
            opInfo.padl = -1;
            opInfo.padr = -1;
        }
        // invalid scene check already in InferShape
        return true;
    }

    if (strcmp(opInfo.padMode, "SPECIFIC") == 0) {
        return true;
    }

    if (strcmp(opInfo.padMode, "SAME") == 0 || strcmp(opInfo.padMode, "SAME_UPPER") == 0 ||
        strcmp(opInfo.padMode, "SAME_LOWER") == 0) {
        GetSamePadMode(opInfo);
    } else if (strcmp(opInfo.padMode, "VALID") == 0) {
        opInfo.padh = 0;
        opInfo.padt = 0;
        opInfo.padu = 0;
        opInfo.padd = 0;
        opInfo.padl = 0;
        opInfo.padr = 0;
    } else {
        OP_LOGE(context->GetNodeName(), "pad mode is invalid which is %s.", opInfo.padMode);
        return false;
    }
    return true;
}

template <typename T>
static bool GetConvPads(const T* context, ConvOpInfo& opInfo)
{
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT_BOOL(context, attrs);
    const gert::ContinuousVector* padsList = attrs->GetAttrPointer<gert::ContinuousVector>(opInfo.paramIdx.padIdx);
    OPS_CHECK_NULL_WITH_CONTEXT_BOOL(context, padsList);
    if (opInfo.isConv3DLike) {
        OP_LOGE_IF(padsList->GetSize() != CONV3DV2_PAD_SIZE_LIMIT, false, context->GetNodeName(),
            "The shape of the pads should be %zu, actual dim num: %zu",
            CONV3DV2_PAD_SIZE_LIMIT, padsList->GetSize());
    } else {
        OP_LOGE_IF(padsList->GetSize() != CONV2DV2_PAD_SIZE_LIMIT, false, context->GetNodeName(),
            "The shape of the pads should be %zu, actual dim num: %zu",
            CONV2DV2_PAD_SIZE_LIMIT, padsList->GetSize());
    }
    const int64_t* padsListData = static_cast<const int64_t*>(padsList->GetData());

    size_t idx = 0;
    if (opInfo.isConv3DLike) {
        opInfo.padh = padsListData[idx++];
        opInfo.padt = padsListData[idx++];
        opInfo.padu = padsListData[idx++];
        opInfo.padd = padsListData[idx++];
        opInfo.padl = padsListData[idx++];
        opInfo.padr = padsListData[idx++];
    } else {
        opInfo.padu = padsListData[idx++];
        opInfo.padd = padsListData[idx++];
        opInfo.padl = padsListData[idx++];
        opInfo.padr = padsListData[idx++];
    }

    OP_LOGD(context->GetNodeName(), "pads before pad_mode update, which is [%ld, %ld, %ld, %ld, %ld, %ld]. ",
        opInfo.padh, opInfo.padt, opInfo.padu, opInfo.padd, opInfo.padl, opInfo.padr);

    if (!GetPadModeAndUpdatePad(context, opInfo)) {
        return false;
    }

    OP_LOGD(context->GetNodeName(), "pads after pad_mode update, which is [%ld, %ld, %ld, %ld, %ld, %ld]. ",
        opInfo.padh, opInfo.padt, opInfo.padu, opInfo.padd, opInfo.padl, opInfo.padr);

    return CheckPositivePads(context->GetNodeName(), opInfo);
}

static bool CheckConvGroups(const InferShapeContext* context, ConvOpInfo& opInfo)
{
    bool unknownRank = opInfo.unknownRankX || opInfo.unknownRankW;
    if (unknownRank) { // unknown rank quit
        return true;
    }

    OPS_CHECK_NULL_WITH_CONTEXT_BOOL(context, context->GetAttrs());
    auto groupPtr = context->GetAttrs()->GetInt(opInfo.paramIdx.groupsIdx);
    OPS_CHECK_NULL_WITH_CONTEXT_BOOL(context, groupPtr);
    int64_t groups = *groupPtr;
    bool unknownShape = opInfo.unknownShapeX || opInfo.unknownShapeW;
    OP_CHECK((!unknownShape && groups <= 0),
        CUBE_INNER_ERR_REPORT(context->GetNodeName(), "groups %ld is illegal", groups), return false);

    bool icUpdateFlag = opInfo.ic < 0 && opInfo.kc > 0 && groups > 0;
    if (icUpdateFlag) {
        OP_LOGW(context->GetNodeName(), "input x channel is unknown, set in_channels[%ld] to kc * groups[%ld * %ld]",
            opInfo.ic, opInfo.kc, groups);
        opInfo.ic = opInfo.kc * groups;
    }

    if (opInfo.ic < 0 || opInfo.kc < 0 || groups < 0) {
        OP_LOGW(context->GetNodeName(), "Exist unknown shape, skip check: in_channels[%ld] = kc * groups[%ld * %ld]",
            opInfo.ic, opInfo.kc, groups);
        return true;
    }

    if (opInfo.ic != opInfo.kc * groups) {
        OP_LOGE(context->GetNodeName(),
                "The input channel should be equal to filter_channel*groups. input channel is %ld, "
                "filter channel is %ld, groups is: %ld.",
                opInfo.ic, opInfo.kc, groups);
        return false;
    }
    if (opInfo.kn % groups != 0) {
        OP_LOGE(context->GetNodeName(), "The output channels %ld should be divisible by groups %ld.",
                opInfo.kn, groups);
        return false;
    }
    return true;
}

static bool CheckConvInputWithPad(const InferShapeContext* context, const ConvOpInfo& opInfo)
{
    bool unknownRank = opInfo.unknownRankX || opInfo.unknownRankW;
    bool isDynamic = (opInfo.unknownShapeX && !opInfo.unknownRankX) || (opInfo.unknownShapeW && !opInfo.unknownRankW);
    if (isDynamic || unknownRank) {
        return true; // unknownRank or Dynamic skip all ZeroTensor check
    }
    int64_t idPad = opInfo.id + opInfo.padh + opInfo.padt - opInfo.dild * (opInfo.kd - 1) - 1;
    int64_t ihPad = opInfo.ih + opInfo.padu + opInfo.padd - opInfo.dilh * (opInfo.kh - 1) - 1;
    int64_t iwPad = opInfo.iw + opInfo.padl + opInfo.padr - opInfo.dilw * (opInfo.kw - 1) - 1;
    if (opInfo.isConv3DLike && (idPad < 0 || ihPad < 0 || iwPad < 0)) {
        std::stringstream convLogStr;
        convLogStr << "Fmap size(DHW) after padding should be greater than or equal to filtersize(DHW)."
                << "idPad " << idPad << ", ihPad " << ihPad << ", iwPad " << iwPad;
        OP_LOGE(context->GetNodeName(), "%s", convLogStr.str().c_str());
        return false;
    } else if (opInfo.isConv2DLike && (ihPad < 0 || iwPad < 0)) {
        OP_LOGE(context->GetNodeName(),
            "Fmap size(HW) after padding should be greater than or equal to filtersize(HW). ihPad %ld, iwPad %ld",
            ihPad, iwPad);
        return false;
    }

    return true;
}

static bool CheckOutputZeroTensor(const InferShapeContext* context, ConvOpInfo& opInfo)
{
    if (!opInfo.isInputZeroTensor) {
        return true;
    }
    bool conv2dNegFlag = opInfo.in < 0 || opInfo.kn < 0 || opInfo.oh < 0 || opInfo.ow < 0;
    bool conv2dNoneZeroFlag = opInfo.in != 0 && opInfo.kn != 0 && opInfo.oh != 0 && opInfo.ow != 0;
    std::stringstream convLogStr;
    if (opInfo.isConv2DLike) {
        if (conv2dNegFlag) {
            convLogStr << "Invalid input [in: "<< opInfo.in << ", ic: " << opInfo.ic << ", ih: "
                    << opInfo.ih << ", iw: " << opInfo.iw << "], output [on: " << opInfo.on << ", oc: "
                    << opInfo.oc << ", oh: " << opInfo.oh << ", ow: " << opInfo.ow << "] infers negative value!";
            OP_LOGE(context->GetNodeName(), "%s", convLogStr.str().c_str());
            return false;
        }
        if (conv2dNoneZeroFlag) {
            convLogStr << "zero tensor input [in: "<< opInfo.in << ", ic: " << opInfo.ic << ", ih: "
                    << opInfo.ih << ", iw: " << opInfo.iw << "] and non-zero tensor output [on: " << opInfo.on
                    << ", oc: " << opInfo.oc << ", oh: " << opInfo.oh << ", ow: " << opInfo.ow << "] are not supported!";
            OP_LOGE(context->GetNodeName(), "%s", convLogStr.str().c_str());
            return false;
        }
    } else if (opInfo.isConv3DLike) {
        if (conv2dNegFlag || opInfo.od < 0) {
            convLogStr << "Invalid input [in: "<< opInfo.in << ", ic: " << opInfo.ic << ", id: " << opInfo.id
                    << ", ih: " << opInfo.ih << ", iw: " << opInfo.iw << "], output [on: " << opInfo.on << ", oc: "
                    << opInfo.oc << ", od: " << opInfo.od << ", oh: " << opInfo.oh << ", ow: " << opInfo.ow
                    << "] infers negative value!";
            OP_LOGE(context->GetNodeName(), "%s", convLogStr.str().c_str());
            return false;
        }
        if (conv2dNoneZeroFlag && opInfo.od != 0) {
            convLogStr << "Zero tensor input [in: "<< opInfo.in << ", ic: " << opInfo.ic << ", id: " << opInfo.id
                    << ", ih: " << opInfo.ih << ", iw: " << opInfo.iw << "] and non-zero tensor output [on: "
                    << opInfo.on << ", oc: " << opInfo.oc << ", od: " << opInfo.od << ", oh: " << opInfo.oh
                    << ", ow: " << opInfo.ow << "] are not supported!";
            OP_LOGE(context->GetNodeName(), "%s", convLogStr.str().c_str());
            return false;
        }
    }
    opInfo.isOutputZeroTensor = true;
    if (opInfo.opType == ConvOptype::QUANT_CONV3D || opInfo.opType == ConvOptype::QUANT_CONV2D) {
        if (opInfo.isInputZeroTensor || opInfo.isOutputZeroTensor) {
            OP_LOGE(context->GetNodeName(), "zero tensor is not supported in quant scenario.");
        }
    }
    OP_LOGD(context->GetNodeName(), "Output is zero tensor.");
    return true;
}

static bool SetConvYDim(InferShapeContext* context, ConvOpInfo& opInfo, const ge::Format yFormat)
{
    gert::Shape* yShape = context->GetOutputShape(opInfo.paramIdx.outputIdx);
    OPS_CHECK_NULL_WITH_CONTEXT_BOOL(context, yShape);

    yShape->SetDimNum(0);
    if (yFormat == ge::Format::FORMAT_NCDHW) {
        yShape->AppendDim(opInfo.on);
        yShape->AppendDim(opInfo.oc);
        yShape->AppendDim(opInfo.od);
        yShape->AppendDim(opInfo.oh);
        yShape->AppendDim(opInfo.ow);
    } else if (yFormat == ge::Format::FORMAT_NDHWC) {
        yShape->AppendDim(opInfo.on);
        yShape->AppendDim(opInfo.od);
        yShape->AppendDim(opInfo.oh);
        yShape->AppendDim(opInfo.ow);
        yShape->AppendDim(opInfo.oc);
    } else if (yFormat == ge::Format::FORMAT_NHWC) {
        yShape->AppendDim(opInfo.on);
        yShape->AppendDim(opInfo.oh);
        yShape->AppendDim(opInfo.ow);
        yShape->AppendDim(opInfo.oc);
    } else if (yFormat == ge::Format::FORMAT_NCHW) {
        yShape->AppendDim(opInfo.on);
        yShape->AppendDim(opInfo.oc);
        yShape->AppendDim(opInfo.oh);
        yShape->AppendDim(opInfo.ow);
    } else {
        OP_LOGE(context->GetNodeName(), "The format of output y not support format %s.",
                ge::TypeUtils::FormatToSerialString(yFormat).c_str());
        return false;
    }

    OP_LOGD(context->GetNodeName(), "yShape: %s ", Ops::Base::ToString(*yShape).c_str());
    return true;
}

static bool SetConvYShape(InferShapeContext* context, ConvOpInfo& opInfo)
{
    const gert::CompileTimeTensorDesc* yDesc = context->GetOutputDesc(opInfo.paramIdx.outputIdx);
    OPS_CHECK_NULL_WITH_CONTEXT_BOOL(context, yDesc);
    const ge::Format yFormat = yDesc->GetOriginFormat();
    OP_LOGE_IF(yFormat == ge::Format::FORMAT_RESERVED, false, context->GetNodeName(),
        "get format failed: %d", yFormat);

    bool unknownRank = opInfo.unknownRankX || opInfo.unknownRankW;
    if (unknownRank) {
        opInfo.on = UNKNOWN_DIM_VALUE_;
        opInfo.oc = UNKNOWN_DIM_VALUE_;
        opInfo.od = UNKNOWN_DIM_VALUE_;
        opInfo.oh = UNKNOWN_DIM_VALUE_;
        opInfo.ow = UNKNOWN_DIM_VALUE_;
    } else {
        // Conv2DV2-like operator strd default is 1, no div 0 scence
        opInfo.on = opInfo.in;
        opInfo.oc = opInfo.kn;
        opInfo.od = (opInfo.id == UNKNOWN_DIM_VALUE_ || opInfo.kd == UNKNOWN_DIM_VALUE_) ? UNKNOWN_DIM_VALUE_ :
            (opInfo.id + opInfo.padh + opInfo.padt - opInfo.dild * (opInfo.kd - 1) - 1) / opInfo.strd + 1;
        opInfo.oh = (opInfo.ih == UNKNOWN_DIM_VALUE_ || opInfo.kh == UNKNOWN_DIM_VALUE_) ? UNKNOWN_DIM_VALUE_ :
            (opInfo.ih + opInfo.padu + opInfo.padd - opInfo.dilh * (opInfo.kh - 1) - 1) / opInfo.strh + 1;
        opInfo.ow = (opInfo.iw == UNKNOWN_DIM_VALUE_ || opInfo.kw == UNKNOWN_DIM_VALUE_) ? UNKNOWN_DIM_VALUE_ :
            (opInfo.iw + opInfo.padl + opInfo.padr - opInfo.dilw * (opInfo.kw - 1) - 1) / opInfo.strw + 1;
        if (!CheckOutputZeroTensor(context, opInfo)) {
            return false;
        }
    }

    return SetConvYDim(context, opInfo, yFormat);
}

static bool CheckQuantRoundMode(const InferShapeContext* context, ConvOpInfo& opInfo)
{
    if (opInfo.opType != ConvOptype::EXTEND_CONV2D && opInfo.opType != ConvOptype::QUANT_CONV2D &&
        opInfo.opType != ConvOptype::QUANT_CONV3D) {
        return true;
    }

    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT_BOOL(context, attrs);
    const char* roundMode = attrs->GetAttrPointer<char>(opInfo.paramIdx.roundModeIdx);
    OPS_CHECK_NULL_WITH_CONTEXT_BOOL(context, roundMode);
    const auto yDesc = context->GetOutputDesc(opInfo.paramIdx.outputIdx);
    OPS_CHECK_NULL_WITH_CONTEXT_BOOL(context, yDesc);
    const ge::DataType yDtype = yDesc->GetDataType();
    if (yDtype == ge::DT_HIFLOAT8) {
        if (strcmp(roundMode, "round") != 0) {
            OP_LOGE(context->GetNodeName(),
                    "round_mode only support round in hif8 dtype, now round_mode is %s.", roundMode);
            return false;
        }
    } else if (yDtype == ge::DT_FLOAT8_E4M3FN || yDtype == ge::DT_INT8) {
        if (strcmp(roundMode, "rint") != 0) {
            OP_LOGE(context->GetNodeName(),
                    "round_mode only support rint in not_hif8 dtype, now round_mode is %s.", roundMode);
            return false;
      }
    }
    return true;
}

static bool CheckDualOutputCase(const InferShapeContext* context, ConvOpInfo& opInfo)
{
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT_BOOL(context, attrs);
    const bool* dualOut = attrs->GetAttrPointer<bool>(opInfo.paramIdx.dualOutIdx);
    OPS_CHECK_NULL_WITH_CONTEXT_BOOL(context, dualOut);
    const bool* enableRelu1 = attrs->GetAttrPointer<bool>(opInfo.paramIdx.enableRelu1Idx);
    OPS_CHECK_NULL_WITH_CONTEXT_BOOL(context, enableRelu1);
    const auto scale1Desc = context->GetOptionalInputDesc(opInfo.paramIdx.scale1Idx);
    const auto reluWeight1Desc = context->GetOptionalInputDesc(opInfo.paramIdx.reluWeight1Idx);
    const auto clipValue1Desc = context->GetOptionalInputDesc(opInfo.paramIdx.clipValue1Idx);
    if (*dualOut) {
        OP_LOGE(context->GetNodeName(),
                "ExtendConv2D only support one output now, dualOut attr is %d.", *dualOut);
        return false;
    } else {
        if (scale1Desc != nullptr || reluWeight1Desc != nullptr || clipValue1Desc != nullptr || *enableRelu1) {
            OP_LOGE(context->GetNodeName(),
                    "when dualoutput is false, the second related input and attr is not allowed.");
            return false;
        }
    }
    return true;
}

static bool CheckSupportReluCase(const InferShapeContext* context, ConvOpInfo& opInfo)
{
    const auto reluWeight0Desc = context->GetOptionalInputDesc(opInfo.paramIdx.reluWeight0Idx);
    const auto clipValue0Desc = context->GetOptionalInputDesc(opInfo.paramIdx.clipValue0Idx);
    const auto reluWeight1Desc = context->GetOptionalInputDesc(opInfo.paramIdx.reluWeight1Idx);
    const auto clipValue1Desc = context->GetOptionalInputDesc(opInfo.paramIdx.clipValue1Idx);
    if (reluWeight0Desc != nullptr || clipValue0Desc != nullptr ||
        reluWeight1Desc != nullptr || clipValue1Desc != nullptr) {
        OP_LOGE(context->GetNodeName(),
                "ExtendConv2D only support normal relu now, reluweight/clipvalue should be null.");
        return false;
    }
    return true;
}

static bool CheckExtendConvScaleDtype(const InferShapeContext* context, const ConvOpInfo& opInfo)
{
    // check scale0 dtype if invalid
    const auto scale0Tensor = context->GetOptionalInputDesc(opInfo.paramIdx.scale0Idx);
    if (scale0Tensor != nullptr) {
        const ge::DataType scale0Dtype = scale0Tensor->GetDataType();
        if (scale0Dtype != ge::DT_UINT64 && scale0Dtype != ge::DT_INT64) {
            OP_LOGE(context->GetNodeName(),
                    "unSupported params data type [scale0]: [%s]", DTypeToStr(scale0Dtype).c_str());
            return false;
        }
    }
    const auto scale1Tensor = context->GetOptionalInputDesc(opInfo.paramIdx.scale1Idx);
    if (scale1Tensor != nullptr) {
        const ge::DataType scale1Dtype = scale1Tensor->GetDataType();
        if (scale1Dtype != ge::DT_UINT64 && scale1Dtype != ge::DT_INT64) {
            OP_LOGE(context->GetNodeName(),
                    "unSupported params data type [scale1]: [%s]", DTypeToStr(scale1Dtype).c_str());
            return false;
        }
    }
    return true;
}

static bool CheckExtendConv2DUnsupportCase(const InferShapeContext* context, ConvOpInfo& opInfo)
{
    return (!CheckDualOutputCase(context, opInfo) || !CheckSupportReluCase(context, opInfo) ||
            !CheckExtendConvScaleDtype(context, opInfo));
}

static ge::graphStatus CheckInputZeroTensor(const InferShapeContext* context, ConvOpInfo& opInfo)
{
    bool unknownRank = opInfo.unknownRankX || opInfo.unknownRankW;
    bool isDynamic = (opInfo.unknownShapeX && !opInfo.unknownRankX) || (opInfo.unknownShapeW && !opInfo.unknownRankW);
    if (isDynamic || unknownRank) {
        return ge::GRAPH_SUCCESS; // unknownRank or Dynamic skip all ZeroTensor check
    }

    bool conv2dInputZeroFlag = opInfo.ic == 0 || opInfo.in == 0 || opInfo.ih == 0 || opInfo.iw == 0 || opInfo.kn == 0;
    bool conv2dUnsupportZeroFlag = opInfo.kh == 0 || opInfo.kw == 0;
    if (opInfo.isConv2DLike) {
        if (conv2dUnsupportZeroFlag) {
            OP_LOGE(context->GetNodeName(),
                    "Input kh or kw can't be 0, [kn: %ld, kc: %ld, kh: %ld, kw: %ld] is not supported!",
                    opInfo.kn, opInfo.kc, opInfo.kh, opInfo.kw);
            return ge::GRAPH_FAILED;
        }
        if (conv2dInputZeroFlag) {
            opInfo.isInputZeroTensor = true;
            OP_LOGD(context->GetNodeName(), "Input is zero tensor.");
        }
        return ge::GRAPH_SUCCESS;
    }
    // Conv3DV2 and QuantConv3D
    if (conv2dUnsupportZeroFlag || opInfo.kd == 0) {
        OP_LOGE(context->GetNodeName(),
                "Input kd or kh or kw can't be 0, [kn: %ld, kc: %ld, kd: %ld, kh: %ld, kw: %ld] is not supported!",
                opInfo.kn, opInfo.kc, opInfo.kd, opInfo.kh, opInfo.kw);
        return ge::GRAPH_FAILED;
    }
    if (conv2dInputZeroFlag || opInfo.id == 0) {
        opInfo.isInputZeroTensor = true;
        OP_LOGD(context->GetNodeName(), "Input is zero tensor.");
    }

    return ge::GRAPH_SUCCESS;
}

static void GetConvOutShapeRange(int64_t minValue, int64_t maxValue, size_t idx, size_t dimNumLimit,
                                 std::vector<std::pair<int64_t, int64_t>>& outRange)
{
    int64_t dynamicRangeLowerBound = (minValue == ZERO_TENSOR_DYN_RANGE_LOWER_BOUND) ?
        ZERO_TENSOR_DYN_RANGE_LOWER_BOUND : DYN_RANGE_LOWER_BOUND;
    outRange[idx].first = std::max(minValue, dynamicRangeLowerBound);
    if (idx > dimNumLimit || dimNumLimit <= 0) {
        outRange[idx].second = UNKNOWN_DIM_VALUE_;
        return;
    }
    outRange[idx].second = maxValue;
}

static bool CheckConvInputRangeDim(const gert::InferShapeRangeContext* context, const size_t fmShapeRangeDimNum,
    const size_t wShapeRangeDimNum, const gert::Range<gert::Shape>* fmShapeRange,
    const gert::Range<gert::Shape>* wShapeRange)
{
    if (fmShapeRangeDimNum != wShapeRangeDimNum) {
        OP_LOGE(context->GetNodeName(), "fmap/weight ShapeRange Dims Num should be same.");
        return false;
    }
    size_t fmShapeRangeDimNumMin = fmShapeRange->GetMin()->GetDimNum();
    size_t wShapeRangeDimNumMin = wShapeRange->GetMin()->GetDimNum();
    if (fmShapeRangeDimNumMin != fmShapeRangeDimNum || wShapeRangeDimNum != wShapeRangeDimNumMin) {
        OP_LOGE(context->GetNodeName(), "fmap/weight Max ShapeRange Dims not equal to Min.");
        return false;
    }
    return true;
}

static void GetConvFmapShapeRange(const gert::InferShapeRangeContext* context, ConvOpInfo& opInfo,
    const size_t fmShapeRangeDimNum, std::vector<std::pair<int64_t, int64_t>>& fmRange,
    const gert::Range<gert::Shape>* fmShapeRange)
{
    OP_LOGD(context->GetNodeName(), "Get fmap shape range Min %s. ", Ops::Base::ToString(*fmShapeRange->GetMin()).c_str());
    OP_LOGD(context->GetNodeName(), "Get fmap shape range Max %s. ", Ops::Base::ToString(*fmShapeRange->GetMax()).c_str());

    for (size_t i = 0; i < fmShapeRangeDimNum; i++) {
        fmRange[i].first = fmShapeRange->GetMin()->GetDim(i);
        fmRange[i].second = fmShapeRange->GetMax()->GetDim(i);
        if (fmRange[i].first != fmRange[i].second && fmRange[i].first > 0 && fmRange[i].second > 0) {
            opInfo.unknownShapeX = true;
        }
        if (fmRange[i].first == UNKNOWN_RANK_DIM_VALUE_ || fmRange[i].second == UNKNOWN_RANK_DIM_VALUE_) {
            opInfo.unknownRankX = true;
        }
    }
}

static void GetConvWeightShapeRange(const gert::InferShapeRangeContext* context, ConvOpInfo& opInfo,
    const size_t wShapeRangeDimNum, std::vector<std::pair<int64_t, int64_t>>& weightRange,
    const gert::Range<gert::Shape>* wShapeRange)
{
    OP_LOGD(context->GetNodeName(), "Get weight shape range Min %s. ", Ops::Base::ToString(*wShapeRange->GetMin()).c_str());
    OP_LOGD(context->GetNodeName(), "Get weight shape range Max %s. ", Ops::Base::ToString(*wShapeRange->GetMax()).c_str());

    for (size_t i = 0; i < wShapeRangeDimNum; i++) {
        weightRange[i].first = wShapeRange->GetMin()->GetDim(i);
        weightRange[i].second = wShapeRange->GetMax()->GetDim(i);
        if (weightRange[i].first != weightRange[i].second && weightRange[i].first > 0 && weightRange[i].second > 0) {
            opInfo.unknownShapeW = true;
        }
        if (weightRange[i].first == UNKNOWN_RANK_DIM_VALUE_ || weightRange[i].second == UNKNOWN_RANK_DIM_VALUE_) {
            opInfo.unknownRankW = true;
        }
    }
}

static void SetConvOutShapeRange(const gert::InferShapeRangeContext* context, const size_t outShapeRangeDimNum,
    const std::vector<std::pair<int64_t, int64_t>>& outRange, gert::Range<gert::Shape>* outShapeRange)
{
    outShapeRange->GetMin()->SetDimNum(outShapeRangeDimNum);
    outShapeRange->GetMax()->SetDimNum(outShapeRangeDimNum);
    for (size_t i = 0; i < outShapeRangeDimNum; i++) {
        outShapeRange->GetMax()->SetDim(i, outRange[i].second);
        outShapeRange->GetMin()->SetDim(i, outRange[i].first);
    }
    OP_LOGD(context->GetNodeName(), "Get out shape range Min %s. ", Ops::Base::ToString(*outShapeRange->GetMin()).c_str());
    OP_LOGD(context->GetNodeName(), "Get out shape range Max %s. ", Ops::Base::ToString(*outShapeRange->GetMax()).c_str());
}

static bool CheckConvShapeRange(const gert::InferShapeRangeContext* context,
    const gert::Range<gert::Shape>* fmShapeRange, const gert::Range<gert::Shape>* wShapeRange,
    const gert::Range<gert::Shape>* outShapeRange)
{
    OPS_CHECK_NULL_WITH_CONTEXT_BOOL(context, fmShapeRange);
    OPS_CHECK_NULL_WITH_CONTEXT_BOOL(context, wShapeRange);
    OPS_CHECK_NULL_WITH_CONTEXT_BOOL(context, outShapeRange);
    OPS_CHECK_NULL_WITH_CONTEXT_BOOL(context, fmShapeRange->GetMax());
    OPS_CHECK_NULL_WITH_CONTEXT_BOOL(context, fmShapeRange->GetMin());
    OPS_CHECK_NULL_WITH_CONTEXT_BOOL(context, wShapeRange->GetMax());
    OPS_CHECK_NULL_WITH_CONTEXT_BOOL(context, wShapeRange->GetMin());
    return true;
}

static void GetConvOutShapeRangeNeedInfer(size_t dimIdx, ConvOpInfo& opInfo,
                                          std::pair<int64_t, int64_t>& kernel,
                                          const std::vector<std::pair<int64_t, int64_t>>& inRange,
                                          std::vector<std::pair<int64_t, int64_t>>& outRange)
{
    int64_t stride = 0;
    int64_t dilation = 0;
    int64_t pad = 0;
    if (dimIdx >= opInfo.xFormatIdx.size()) {
        OP_LOGE("GetConvOutShapeRangeNeedInfer", "dimIdx must in range of [0~%zu].", opInfo.xFormatIdx.size());
        return;
    }
    size_t idx = opInfo.xFormatIdx[dimIdx];
    if (dimIdx == IDX_LIST_H_IDX) {
        stride = opInfo.strh;
        dilation = opInfo.dilh;
        pad = opInfo.padu + opInfo.padd;
    } else if (dimIdx == IDX_LIST_W_IDX) {
        stride = opInfo.strw;
        dilation = opInfo.dilw;
        pad = opInfo.padl + opInfo.padr;
    } else { // D
        stride = opInfo.strd;
        dilation = opInfo.dild;
        pad = opInfo.padh + opInfo.padt;
    }

    if (stride <= 0) {
        OP_LOGE("GetConvOutShapeRangeNeedInfer", "diviser stride must > 0.");
        return;
    }

    int64_t low = inRange[idx].first;
    int64_t high = inRange[idx].second;
    if (kernel.first == 0) {
        outRange[idx].first = 0;
    }
    // rt1.0 only handles same, but essentially all pad type that cannot be directly calculated need to be calculated.
    if (strcmp(opInfo.padMode, "SAME") == 0 || strcmp(opInfo.padMode, "SAME_UPPER") == 0 ||
        strcmp(opInfo.padMode, "SAME_LOWER") == 0) {
        outRange[idx].first = (low + stride - 1) / stride;
        outRange[idx].second = (high + stride - 1) / stride;
    } else {
        outRange[idx].first = kernel.first != 0 ? (low + pad - dilation * (kernel.first - 1) - 1) / stride + 1 : 0;
        outRange[idx].second = kernel.second != 0 ? (high + pad - dilation * (kernel.second - 1) - 1) / stride + 1 : 0;
    }
    int64_t dynamicRangeLowerBound = (low == ZERO_TENSOR_DYN_RANGE_LOWER_BOUND) ?
            ZERO_TENSOR_DYN_RANGE_LOWER_BOUND : DYN_RANGE_LOWER_BOUND;
    outRange[idx].first = std::max(outRange[idx].first, dynamicRangeLowerBound);
    if (high == UNKNOWN_DIM_VALUE_) {
        outRange[idx].second = high;
    }
}

static bool InferConvOutShapeRange(gert::InferShapeRangeContext* context, ConvOpInfo& opInfo)
{
    OP_LOGD(context->GetNodeName(), "Begin dynamic shape set range. ");
    auto fmShapeRange = context->GetInputShapeRange(opInfo.paramIdx.fMapIdx);
    auto wShapeRange = context->GetInputShapeRange(opInfo.paramIdx.weightIdx);
    auto outShapeRange = context->GetOutputShapeRange(opInfo.paramIdx.outputIdx);
    OP_LOGE_IF(!CheckConvShapeRange(context, fmShapeRange, wShapeRange, outShapeRange), false, context->GetNodeName(),
        "Check conv shape range failed.");

    size_t fmShapeRangeDimNum = fmShapeRange->GetMax()->GetDimNum();
    size_t wShapeRangeDimNum = wShapeRange->GetMax()->GetDimNum();
    OP_LOGE_IF(!CheckConvInputRangeDim(context, fmShapeRangeDimNum, wShapeRangeDimNum, fmShapeRange, wShapeRange),
        false, context->GetNodeName(), "Check conv input shape range dim failed.");

    std::vector<std::pair<int64_t, int64_t>> fmRange(fmShapeRangeDimNum);
    GetConvFmapShapeRange(context, opInfo, fmShapeRangeDimNum, fmRange, fmShapeRange);
    std::vector<std::pair<int64_t, int64_t>> weightRange(wShapeRangeDimNum);
    GetConvWeightShapeRange(context, opInfo, wShapeRangeDimNum, weightRange, wShapeRange);
    size_t outShapeRangeDimNum = fmShapeRangeDimNum;
    std::vector<std::pair<int64_t, int64_t>> outRange(outShapeRangeDimNum);

    if (GetConvStrides(context, opInfo) && GetConvDilations(context, opInfo) &&
        GetConvPads(context, opInfo)) {
        OP_LOGD(context->GetNodeName(), "success to get inputs attr in infershape Range.");
    } else {
        CUBE_INNER_ERR_REPORT(context->GetNodeName(), "failed to infer attr in infershape Range.");
        return false;
    }

    GetConvOutShapeRange(fmRange[opInfo.xFormatIdx[IDX_LIST_N_IDX]].first, fmRange[opInfo.xFormatIdx[IDX_LIST_N_IDX]].second,
                         opInfo.xFormatIdx[IDX_LIST_N_IDX], fmShapeRangeDimNum, outRange);
    GetConvOutShapeRange(weightRange[opInfo.wFormatIdx[IDX_LIST_N_IDX]].first, weightRange[opInfo.wFormatIdx[IDX_LIST_N_IDX]].second,
                         opInfo.xFormatIdx[IDX_LIST_C_IDX], wShapeRangeDimNum, outRange);
    GetConvOutShapeRangeNeedInfer(IDX_LIST_H_IDX, opInfo, weightRange[opInfo.wFormatIdx[IDX_LIST_H_IDX]], fmRange, outRange);
    GetConvOutShapeRangeNeedInfer(IDX_LIST_W_IDX, opInfo, weightRange[opInfo.wFormatIdx[IDX_LIST_W_IDX]], fmRange, outRange);
    if (fmShapeRangeDimNum == CONV3DV2_SUPPORTED_DIM_NUM) {
        GetConvOutShapeRangeNeedInfer(IDX_LIST_D_IDX, opInfo, weightRange[opInfo.wFormatIdx[IDX_LIST_D_IDX]], fmRange,
            outRange);
    }

    if (opInfo.kn == UNKNOWN_DIM_VALUE_) {
        bool existWeightRange = !weightRange.empty() && (wShapeRangeDimNum == CONV2DV2_SUPPORTED_DIM_NUM ||
            wShapeRangeDimNum == CONV3DV2_SUPPORTED_DIM_NUM);
        // NOTE: RT1.0 sets weight c range to the output c range, and the fault will be fixed in this ascendc process.
        outRange[opInfo.xFormatIdx[IDX_LIST_C_IDX]] = existWeightRange ? weightRange[opInfo.wFormatIdx[IDX_LIST_N_IDX]] :
            std::make_pair(static_cast<int64_t>(1), static_cast<int64_t>(-1));
    }
    SetConvOutShapeRange(context, outShapeRangeDimNum, outRange, outShapeRange);
    return true;
}

static ge::graphStatus InferShapeForConvInner(InferShapeContext* context, OpParamIdx& opParamIdx, ConvOptype opType)
{
    OP_CHECK(context == nullptr, CUBE_INNER_ERR_REPORT("ConvolutionV2", "context is null"), return ge::GRAPH_FAILED);
    const auto opName = context->GetNodeName();
    OP_LOGD(context->GetNodeName(), "Enter shape infer. ");

    ConvOpInfo opInfo;
    opInfo.paramIdx = opParamIdx;
    opInfo.opType = opType;
    opInfo.isConv2DLike = (opType == ConvOptype::CONV2DV2 || opType == ConvOptype::QUANT_CONV2D ||
                           opType == ConvOptype::EXTEND_CONV2D);
    opInfo.isConv3DLike = (opType == ConvOptype::CONV3DV2 || opType == ConvOptype::QUANT_CONV3D);

    if (opInfo.opType == ConvOptype::QUANT_CONV2D &&
        context->GetInputDesc(OFFSET_IDX_QUANT_CONV) != nullptr) {
        OP_LOGE(context->GetNodeName(), "QuantConv2d don't support to set offset parameter.");
        return ge::GRAPH_FAILED;
    }
    const gert::CompileTimeTensorDesc* xDesc = context->GetInputDesc(opInfo.paramIdx.fMapIdx);
    OPS_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    const ge::Format xFormat = xDesc->GetOriginFormat();
    OP_LOGE_IF(xFormat == ge::Format::FORMAT_RESERVED, ge::GRAPH_FAILED,
        context->GetNodeName(), "Get x format failed: %d.", xFormat);
    const gert::CompileTimeTensorDesc* wDesc = context->GetInputDesc(opInfo.paramIdx.weightIdx);
    OPS_CHECK_NULL_WITH_CONTEXT(context, wDesc);
    const ge::Format wFormat = wDesc->GetOriginFormat();
    OP_LOGE_IF(wFormat == ge::Format::FORMAT_RESERVED, ge::GRAPH_FAILED,
        context->GetNodeName(), "Get w format failed: %d.", wFormat);
    GetConvShapeIdx(xFormat, opInfo.xFormatIdx);
    GetConvShapeIdx(wFormat, opInfo.wFormatIdx);

    if (GetConvXShape(context, opInfo) && GetConvFilterShape(context, opInfo) &&
        CheckConvBias(context, opInfo) && CheckConvGroups(context, opInfo) && GetConvStrides(context, opInfo) &&
        GetConvDilations(context, opInfo) && GetConvPads(context, opInfo) && CheckQuantRoundMode(context, opInfo)) {
        OP_LOGD(context->GetNodeName(), "success to get inputs shape and attrs.");
    } else {
        CUBE_INNER_ERR_REPORT(opName, "failed to infer shape for ConvolutionV2.");
        return ge::GRAPH_FAILED;
    }

    // extend conv2d check invalid cases
    if (opInfo.opType == ConvOptype::EXTEND_CONV2D && CheckExtendConv2DUnsupportCase(context, opInfo)) {
        CUBE_INNER_ERR_REPORT(opName, "failed to infer shape for ExtendConv2D.");
        return ge::GRAPH_FAILED;
    }

    if (CheckInputZeroTensor(context, opInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (!opInfo.isInputZeroTensor && !CheckConvInputWithPad(context, opInfo)) {
        return ge::GRAPH_FAILED;
    }

    if (!SetConvYShape(context, opInfo)) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeForConv2DV2(InferShapeContext *context)
{
    OpParamIdx convParamIdx = {X_IDX_CONV, W_IDX_CONV, BIAS_IDX_CONV, Y_IDX_CONV, STRIDES_IDX_CONV, PADS_IDX_CONV,
        DILATIONS_IDX_CONV, GROUPS_IDX_CONV, PAD_MODE_IDX_CONV, -1};
    ConvOptype convType = ConvOptype::CONV2DV2;
    return InferShapeForConvInner(context, convParamIdx, convType);
}

static ge::graphStatus InferShapeForConv3DV2(InferShapeContext *context)
{
    OpParamIdx convParamIdx = {X_IDX_CONV, W_IDX_CONV, BIAS_IDX_CONV, Y_IDX_CONV, STRIDES_IDX_CONV, PADS_IDX_CONV,
        DILATIONS_IDX_CONV, GROUPS_IDX_CONV, PAD_MODE_IDX_CONV, -1};
    ConvOptype convType = ConvOptype::CONV3DV2;
    return InferShapeForConvInner(context, convParamIdx, convType);
}

static ge::graphStatus InferShapeForQuantConv2D(InferShapeContext *context)
{
    OpParamIdx convParamIdx = {X_IDX_CONV, W_IDX_CONV, BIAS_IDX_QUANT_CONV, Y_IDX_CONV, STRIDES_IDX_QUANT_CONV,
        PADS_IDX_QUANT_CONV, DILATIONS_IDX_QUANT_CONV, GROUPS_IDX_QUANT_CONV, -1, ROUNDMODE_IDX_QUANT_CONV};
    ConvOptype convType = ConvOptype::QUANT_CONV2D;
    return InferShapeForConvInner(context, convParamIdx, convType);
}

static ge::graphStatus InferShapeForQuantConv3D(InferShapeContext *context)
{
    OpParamIdx convParamIdx = {X_IDX_CONV, W_IDX_CONV, BIAS_IDX_QUANT_CONV, Y_IDX_CONV, STRIDES_IDX_QUANT_CONV,
        PADS_IDX_QUANT_CONV, DILATIONS_IDX_QUANT_CONV, GROUPS_IDX_QUANT_CONV, PAD_MODE_IDX_QUANTCONV3D,
        ROUNDMODE_IDX_QUANT_CONV};
    ConvOptype convType = ConvOptype::QUANT_CONV3D;
    return InferShapeForConvInner(context, convParamIdx, convType);
}

static ge::graphStatus InferShapeForExtendConv2D(InferShapeContext *context)
{
    OpParamIdx convParamIdx = {X_IDX_CONV, W_IDX_CONV, BIAS_IDX_EXTEND_CONV, Y0_IDX_EXTEND_CONV,
        STRIDES_IDX_EXTEND_CONV, PADS_IDX_EXTEND_CONV, DILATIONS_IDX_EXTEND_CONV,
        GROUPS_IDX_EXTEND_CONV, PAD_MODE_IDX_EXTEND_CONV, ROUND_MODE_IDX_EXTEND_CONV,
        SCALE0_IDX_EXTEND_CONV, SCALE1_IDX_EXTEND_CONV, RELUWEIGHT0_IDX_EXTEND_CONV,
        RELUWEIGHT1_IDX_EXTEND_CONV, CLIPVALUE0_IDX_EXTEND_CONV, CLIPVALUE1_IDX_EXTEND_CONV,
        Y1_IDX_EXTEND_CONV, ENABLE_RELU0_IDX_EXTEND_CONV, ENABLE_RELU1_IDX_EXTEND_CONV,
        DUAL_OUT_IDX_EXTEND_CONV};
    ConvOptype convType = ConvOptype::EXTEND_CONV2D;
    return InferShapeForConvInner(context, convParamIdx, convType);
}

ge::graphStatus InferShapeRangeForConvInner(gert::InferShapeRangeContext* context, OpParamIdx& opParamIdx,
    ConvOptype opType)
{
    OP_CHECK(context == nullptr, CUBE_INNER_ERR_REPORT("ConvolutionV2", "context is null."), return ge::GRAPH_FAILED);
    OP_LOGI(context->GetNodeName(), "InferShapeRangeForConvInner running begin. ");

    ConvOpInfo opInfo;
    opInfo.paramIdx = opParamIdx;
    opInfo.opType = opType;
    opInfo.isConv2DLike = (opType == ConvOptype::CONV2DV2 || opType == ConvOptype::QUANT_CONV2D ||
                           opType == ConvOptype::EXTEND_CONV2D);
    opInfo.isConv3DLike = (opType == ConvOptype::CONV3DV2 || opType == ConvOptype::QUANT_CONV3D);

    if (opInfo.opType == ConvOptype::QUANT_CONV2D &&
        context->GetInputDesc(OFFSET_IDX_QUANT_CONV) != nullptr) {
        OP_LOGE(context->GetNodeName(), "QuantConv2d don't support to set offset parameter.");
        return ge::GRAPH_FAILED;
    }

    const auto xDesc = context->GetInputDesc(opInfo.paramIdx.fMapIdx);
    OPS_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    const auto xFormat = xDesc->GetOriginFormat();
    OP_LOGE_IF(xFormat == ge::Format::FORMAT_RESERVED, ge::GRAPH_FAILED, context->GetNodeName(),
        "Get x format failed: %d.", xFormat);
    const auto wDesc = context->GetInputDesc(opInfo.paramIdx.weightIdx);
    OPS_CHECK_NULL_WITH_CONTEXT(context, wDesc);
    const auto wFormat = wDesc->GetOriginFormat();
    OP_LOGE_IF(wFormat == ge::Format::FORMAT_RESERVED, ge::GRAPH_FAILED, context->GetNodeName(),
        "Get w format failed: %d.", wFormat);
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const char* padMode = attrs->GetAttrPointer<char>(opInfo.paramIdx.padModeIdx);
    OPS_CHECK_NULL_WITH_CONTEXT(context, padMode);
    opInfo.padMode = padMode;

    GetConvShapeIdx(xFormat, opInfo.xFormatIdx);
    GetConvShapeIdx(wFormat, opInfo.wFormatIdx);
    // NOTE: rt1.0 in dyn padding same sence, pads will set to all -1, but it is meaningless in AscendC compile process

    // set Conv output shape Range
    OP_LOGE_IF(!InferConvOutShapeRange(context, opInfo), ge::GRAPH_FAILED,
        context->GetNodeName(), "failed to set output shape range.");

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeRangeForQuantConv3D(gert::InferShapeRangeContext* context)
{
    OpParamIdx convParamIdx = {X_IDX_CONV, W_IDX_CONV, BIAS_IDX_QUANT_CONV, Y_IDX_CONV, STRIDES_IDX_QUANT_CONV,
        PADS_IDX_QUANT_CONV, DILATIONS_IDX_QUANT_CONV, GROUPS_IDX_QUANT_CONV, PAD_MODE_IDX_QUANTCONV3D,
        ROUNDMODE_IDX_QUANT_CONV};
    ConvOptype convType = ConvOptype::QUANT_CONV3D;
    return InferShapeRangeForConvInner(context, convParamIdx, convType);
}

static ge::graphStatus InferShapeRangeForExtendConv2D(gert::InferShapeRangeContext* context)
{
    OpParamIdx convParamIdx = {X_IDX_CONV, W_IDX_CONV, BIAS_IDX_EXTEND_CONV, Y0_IDX_EXTEND_CONV,
        STRIDES_IDX_EXTEND_CONV, PADS_IDX_EXTEND_CONV, DILATIONS_IDX_EXTEND_CONV,
        GROUPS_IDX_EXTEND_CONV, PAD_MODE_IDX_EXTEND_CONV, ROUND_MODE_IDX_EXTEND_CONV,
        SCALE0_IDX_EXTEND_CONV, SCALE1_IDX_EXTEND_CONV, RELUWEIGHT0_IDX_EXTEND_CONV,
        RELUWEIGHT1_IDX_EXTEND_CONV, CLIPVALUE0_IDX_EXTEND_CONV, CLIPVALUE1_IDX_EXTEND_CONV,
        Y1_IDX_EXTEND_CONV, ENABLE_RELU0_IDX_EXTEND_CONV, ENABLE_RELU1_IDX_EXTEND_CONV,
        DUAL_OUT_IDX_EXTEND_CONV};
    ConvOptype convType = ConvOptype::EXTEND_CONV2D;
    return InferShapeRangeForConvInner(context, convParamIdx, convType);
}

static ge::graphStatus InferDataTypeQuantConv3D(gert::InferDataTypeContext* context)
{
    OP_CHECK(context == nullptr, CUBE_INNER_ERR_REPORT("ConvolutionV2", "context is null."), return ge::GRAPH_FAILED);
    OP_LOGI(context->GetNodeName(), "InferDataTypeQuantConv3D run begin. ");

    const auto attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t *dtypeNum = attrs->GetAttrPointer<int64_t>(DTYPE_IDX_QUANT_CONV);
    ge::DataType outDtype = static_cast<ge::DataType>(*dtypeNum);
    context->SetOutputDataType(0, outDtype);
    OP_LOGD(context->GetNodeName(), "Set y dtype: %s success.", DTypeToStr(outDtype).c_str());
    OP_LOGI(context->GetNodeName(), "InferDataTypeQuantConv3D run success. ");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeExtendConv2D(gert::InferDataTypeContext* context)
{
    OP_CHECK(context == nullptr, CUBE_INNER_ERR_REPORT("ConvolutionV2", "context is null."), return ge::GRAPH_FAILED);
    OP_LOGI(context->GetNodeName(), "InferDataTypeExtendConv2D run begin. ");

    const auto attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t *dtype0Num = attrs->GetAttrPointer<int64_t>(DTYPE0_IDX_EXTEND_CONV);
    const int64_t *dtype1Num = attrs->GetAttrPointer<int64_t>(DTYPE1_IDX_EXTEND_CONV);
    auto xDtype = context->GetInputDataType(X_IDX_CONV);
    bool dtype0ValidFlag = (dtype0Num != nullptr) && (*dtype0Num != static_cast<int64_t>(-1));
    bool dtype1ValidFlag = (dtype1Num != nullptr) && (*dtype1Num != static_cast<int64_t>(-1));

    if (dtype0ValidFlag) {
        ge::DataType out0Dtype = static_cast<ge::DataType>(*dtype0Num);
        context->SetOutputDataType(Y0_IDX_EXTEND_CONV, out0Dtype);
    } else {
        if (xDtype == ge::DT_UNDEFINED) {
            OP_LOGE(context->GetNodeName(), "get x dtype: %s which is invalid when dtype0 is default value: -1.",
                DTypeToStr(xDtype).c_str());
            return ge::GRAPH_FAILED;
        }
        context->SetOutputDataType(Y0_IDX_EXTEND_CONV, xDtype);
    }
    OP_LOGD(context->GetNodeName(), "Set y0 dtype: %s success.",
        DTypeToStr(context->GetOutputDataType(Y0_IDX_EXTEND_CONV)).c_str());

    if (dtype1ValidFlag) {
        ge::DataType out1Dtype = static_cast<ge::DataType>(*dtype1Num);
        context->SetOutputDataType(Y1_IDX_EXTEND_CONV, out1Dtype);
    } else {
        if (xDtype == ge::DT_UNDEFINED) {
            OP_LOGE(context->GetNodeName(), "get x dtype: %s which is invalid when dtype1 is default value: -1.",
                    DTypeToStr(xDtype).c_str());
            return ge::GRAPH_FAILED;
        }
        context->SetOutputDataType(Y1_IDX_EXTEND_CONV, xDtype);
    }

    OP_LOGD(context->GetNodeName(), "Set y1 dtype: %s success.",
        DTypeToStr(context->GetOutputDataType(Y1_IDX_EXTEND_CONV)).c_str());
    OP_LOGI(context->GetNodeName(), "InferDataTypeExtendConv2D run success. ");
    return ge::GRAPH_SUCCESS;
}
}  // namespace Conv

IMPL_OP_INFERSHAPE(Conv2DV2)
    .InferShape(Ops::NN::Conv::InferShapeForConv2DV2);

IMPL_OP_INFERSHAPE(Conv3DV2)
    .InferShape(Ops::NN::Conv::InferShapeForConv3DV2);

IMPL_OP_INFERSHAPE(QuantConv2D)
    .InferShape(Ops::NN::Conv::InferShapeForQuantConv2D);

IMPL_OP_INFERSHAPE(QuantConv3D)
    .InferShape(Ops::NN::Conv::InferShapeForQuantConv3D)
    .InferShapeRange(Ops::NN::Conv::InferShapeRangeForQuantConv3D)
    .InferDataType(Ops::NN::Conv::InferDataTypeQuantConv3D);

IMPL_OP_INFERSHAPE(ExtendConv2D)
    .InferShape(Ops::NN::Conv::InferShapeForExtendConv2D)
    .InferShapeRange(Ops::NN::Conv::InferShapeRangeForExtendConv2D)
    .InferDataType(Ops::NN::Conv::InferDataTypeExtendConv2D);
}  // namespace NN
}  // namespace Ops
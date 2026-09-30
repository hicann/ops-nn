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
 * \file conv3d_infershape.cpp
 * \brief Conv3D 算子 RT2.0 InferShape / InferShapeRange 实现（自 canndev RT1.0 Conv3DInfer 迁移）
 */

#include <algorithm>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "conv/common/op_host/conv_common_cube_util.h"
#include "error_util.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "graph/utils/type_utils.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "util/shape_util.h"

#define OPS_CHECK_NULL_WITH_CONTEXT_BOOL(context, ptr)                                               \
    if ((ptr) == nullptr) {                                                                          \
        const char* name = ((context)->GetNodeName() == nullptr) ? "nil" : (context)->GetNodeName(); \
        OP_LOGE(name, "%s is nullptr!", #ptr);                                                       \
        REPORT_INNER_ERR_MSG("EZ9999", "op[%s], %s is nullptr!", name, #ptr);                        \
        return false;                                                                                \
    }

namespace ops {
namespace {
// 维度下标与 UNKNOWN_DIM_VALUE_ 复用 conv 家族共享头（常量统一来源，避免各算子重复定义）
using Ops::UNKNOWN_DIM_VALUE_;
using Ops::NN::Conv::kCDimDHWCNIdx;
using Ops::NN::Conv::kCDimNCDHWIdx;
using Ops::NN::Conv::kCDimNDHWCIdx;
using Ops::NN::Conv::kDDimDHWCNIdx;
using Ops::NN::Conv::kDDimNCDHWIdx;
using Ops::NN::Conv::kDDimNDHWCIdx;
using Ops::NN::Conv::kHDimDHWCNIdx;
using Ops::NN::Conv::kHDimNCDHWIdx;
using Ops::NN::Conv::kHDimNDHWCIdx;
using Ops::NN::Conv::kNDimDHWCNIdx;
using Ops::NN::Conv::kNDimNCDHWIdx;
using Ops::NN::Conv::kNDimNDHWCIdx;
using Ops::NN::Conv::kWDimDHWCNIdx;
using Ops::NN::Conv::kWDimNCDHWIdx;
using Ops::NN::Conv::kWDimNDHWCIdx;

// Conv3D 原型输入/输出/属性索引（与 conv3d_def.cpp 中注册顺序一致）
constexpr size_t CONV3D_X_IDX = 0;
constexpr size_t CONV3D_FILTER_IDX = 1;
constexpr size_t CONV3D_Y_IDX = 0;

constexpr size_t CONV3D_STRIDES_IDX = 0;
constexpr size_t CONV3D_PADS_IDX = 1;
constexpr size_t CONV3D_DILATIONS_IDX = 2;
constexpr size_t CONV3D_GROUPS_IDX = 3;
constexpr size_t CONV3D_DATA_FORMAT_IDX = 4;
// 私有属性 padding 由 IMPL_OP_INFERSHAPE 注册，排在第 6 位
constexpr size_t CONV3D_PADDING_IDX = 6;

constexpr size_t CONV3D_DIM_SIZE_LIMIT = 5;
constexpr size_t CONV3D_PADS_SIZE_LIMIT = 6;
// 兼容老 RT2.0（canndev cube_util GetConv3DPads）的 pads 长度 1/3 广播语义
constexpr size_t CONV3D_PADS_BROADCAST_SIZE_3 = 3;
constexpr size_t CONV3D_PADS_BROADCAST_SIZE_1 = 1;

// dilation 值域上界（对齐 conv3dv2 arch35 tiling ParseDilationLegal 的 MAX_DILATION_D/H/W_SHAPE）
constexpr int64_t CONV3D_MAX_DILATION_D = 1000000;
constexpr int64_t CONV3D_MAX_DILATION_H = 255;
constexpr int64_t CONV3D_MAX_DILATION_W = 255;

// N C D H W 在格式映射表中的下标
constexpr size_t IDX_LIST_N_IDX = 0;
constexpr size_t IDX_LIST_C_IDX = 1;
constexpr size_t IDX_LIST_D_IDX = 2;
constexpr size_t IDX_LIST_H_IDX = 3;
constexpr size_t IDX_LIST_W_IDX = 4;

// 对齐 RT1.0 nn_calculation_ops.cc 中 kDynamicRangeLowerBound / kDynamicRangeUpperBound
// （RT1.0 Conv3D 的 range 下界恒为 1，无零 Tensor 下界 0 特判；kZeroTensorDynamicRangeLowerBound
// 在 RT1.0 中仅 DepthwiseConv2D 等 2D 算子使用，不属于 Conv3D 语义，不迁移）
constexpr int64_t DYNAMIC_RANGE_LOWER_BOUND = 1;
constexpr int64_t DYNAMIC_RANGE_UPPER_BOUND = 4096;
// SAME padding 时 pad 总量前后均分（对齐 conv_forward_infershape.cpp 的 PAD_HALF_DIV 语义）
constexpr int64_t PAD_HALF_DIV = 2;

struct Conv3DInputShapes {
    int64_t in = 0;
    int64_t ic = 0;
    int64_t id = 0;
    int64_t ih = 0;
    int64_t iw = 0;

    int64_t kn = 0;
    int64_t kc = 0;
    int64_t kd = 0;
    int64_t kh = 0;
    int64_t kw = 0;

    bool unknownRankX = false;
    bool unknownShapeX = false;
    bool isInputZeroTensor = false;

    // 错误信息展示用的各维下标（随 x / filter 的 format 确定，对齐家族 ConvOpInfo.formatIdx 用法）
    size_t xDIdx = 1;
    size_t xHIdx = 2;
    size_t xWIdx = 3;
    size_t xCIdx = 4;
    size_t wDIdx = 0;
    size_t wHIdx = 1;
    size_t wWIdx = 2;
    size_t wCIdx = 3;
    size_t wNIdx = 4;
};

struct Conv3DAttrs {
    int64_t strd = 1;
    int64_t strh = 1;
    int64_t strw = 1;

    int64_t dild = 1;
    int64_t dilh = 1;
    int64_t dilw = 1;

    // padHead, padTail, padTop, padBottom, padLeft, padRight
    int64_t padh = -1;
    int64_t padt = -1;
    int64_t padu = -1;
    int64_t padd = -1;
    int64_t padl = -1;
    int64_t padr = -1;

    int64_t groups = 1;
    const char* padding = "";
};

inline std::string VectorToString(const std::vector<int64_t>& vec)
{
    std::string result = "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        result += std::to_string(vec[i]);
        if (i < vec.size() - 1) {
            result += ",";
        }
    }
    result += "]";
    return result;
}

inline std::string VectorsToString(const std::vector<std::vector<int64_t>>& vecs)
{
    std::string result = "[";
    for (size_t i = 0; i < vecs.size(); ++i) {
        result += VectorToString(vecs[i]);
        if (i < vecs.size() - 1) {
            result += ",";
        }
    }
    result += "]";
    return result;
}

// 输入 shape 解析（对应 RT1.0 NormalizeConv3dShape）
// 输入X的format解析，根据NCDHW和NDHWC
// 解析 x 各维（含错误信息展示下标与非法 format 拦截）
static bool UpdateConv3DXDims(const gert::InferShapeContext* context, ge::Format xFormat, const gert::Shape* xShape,
                              Conv3DInputShapes& shapes)
{
    if (xFormat == ge::Format::FORMAT_NCDHW) {
        shapes.xDIdx = kDDimNCDHWIdx;
        shapes.xHIdx = kHDimNCDHWIdx;
        shapes.xWIdx = kWDimNCDHWIdx;
        shapes.xCIdx = kCDimNCDHWIdx;
        shapes.in = xShape->GetDim(kNDimNCDHWIdx);
        shapes.ic = xShape->GetDim(kCDimNCDHWIdx);
        shapes.id = xShape->GetDim(kDDimNCDHWIdx);
        shapes.ih = xShape->GetDim(kHDimNCDHWIdx);
        shapes.iw = xShape->GetDim(kWDimNCDHWIdx);
    } else if (xFormat == ge::Format::FORMAT_NDHWC) {
        shapes.xDIdx = kDDimNDHWCIdx;
        shapes.xHIdx = kHDimNDHWCIdx;
        shapes.xWIdx = kWDimNDHWCIdx;
        shapes.xCIdx = kCDimNDHWCIdx;
        shapes.in = xShape->GetDim(kNDimNDHWCIdx);
        shapes.id = xShape->GetDim(kDDimNDHWCIdx);
        shapes.ih = xShape->GetDim(kHDimNDHWCIdx);
        shapes.iw = xShape->GetDim(kWDimNDHWCIdx);
        shapes.ic = xShape->GetDim(kCDimNDHWCIdx);
    } else {
        // x format 拦截校验
        std::string correctFormat = Ops::Base::ToString(ge::Format::FORMAT_NCDHW) + ", " +
                                    Ops::Base::ToString(ge::Format::FORMAT_NDHWC);
        OP_LOGE_FOR_INVALID_FORMAT(context->GetNodeName(), "x",
                                   ge::TypeUtils::FormatToAscendString(xFormat).GetString(), correctFormat.c_str());
        return false;
    }
    return true;
}

static bool GetConv3DXShape(const gert::InferShapeContext* context, ge::Format xFormat, Conv3DInputShapes& shapes)
{
    const gert::Shape* xShape = context->GetInputShape(CONV3D_X_IDX);
    OPS_CHECK_NULL_WITH_CONTEXT_BOOL(context, xShape);

    shapes.unknownShapeX = Ops::Base::IsUnknownShape(*xShape);
    shapes.unknownRankX = Ops::Base::IsUnknownRank(*xShape);
    OP_LOGD(context->GetNodeName(), "Conv3D unknownShapeX: %d, unknownRankX: %d.",
            static_cast<int32_t>(shapes.unknownShapeX), static_cast<int32_t>(shapes.unknownRankX));

    if (shapes.unknownRankX) {
        // 对齐 RT1.0：x 为 -2 时各维度按 -1 处理，输出除 C 维（来自 filter）外均为 -1
        shapes.in = UNKNOWN_DIM_VALUE_;
        shapes.ic = UNKNOWN_DIM_VALUE_;
        shapes.id = UNKNOWN_DIM_VALUE_;
        shapes.ih = UNKNOWN_DIM_VALUE_;
        shapes.iw = UNKNOWN_DIM_VALUE_;
        return true;
    }
    // x必须为5维shape或者shape为unknownrank(-2)
    if (xShape->GetDimNum() != CONV3D_DIM_SIZE_LIMIT) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "x", std::to_string(xShape->GetDimNum()).c_str(),
                                     std::to_string(CONV3D_DIM_SIZE_LIMIT).c_str());
        return false;
    }
    return UpdateConv3DXDims(context, xFormat, xShape, shapes);
}
// 输入filter的format解析，根据NCDHW和NDHWC和DHWCN
// 解析 filter 各维（含错误信息展示下标与非法 format 拦截）
static bool UpdateConv3DFilterDims(const gert::InferShapeContext* context, ge::Format wFormat,
                                   const gert::Shape* wShape, Conv3DInputShapes& shapes)
{
    if (wFormat == ge::Format::FORMAT_NCDHW) {
        shapes.wDIdx = kDDimNCDHWIdx;
        shapes.wHIdx = kHDimNCDHWIdx;
        shapes.wWIdx = kWDimNCDHWIdx;
        shapes.wCIdx = kCDimNCDHWIdx;
        shapes.wNIdx = kNDimNCDHWIdx;
        shapes.kn = wShape->GetDim(kNDimNCDHWIdx);
        shapes.kc = wShape->GetDim(kCDimNCDHWIdx);
        shapes.kd = wShape->GetDim(kDDimNCDHWIdx);
        shapes.kh = wShape->GetDim(kHDimNCDHWIdx);
        shapes.kw = wShape->GetDim(kWDimNCDHWIdx);
    } else if (wFormat == ge::Format::FORMAT_NDHWC) {
        shapes.wDIdx = kDDimNDHWCIdx;
        shapes.wHIdx = kHDimNDHWCIdx;
        shapes.wWIdx = kWDimNDHWCIdx;
        shapes.wCIdx = kCDimNDHWCIdx;
        shapes.wNIdx = kNDimNDHWCIdx;
        shapes.kn = wShape->GetDim(kNDimNDHWCIdx);
        shapes.kd = wShape->GetDim(kDDimNDHWCIdx);
        shapes.kh = wShape->GetDim(kHDimNDHWCIdx);
        shapes.kw = wShape->GetDim(kWDimNDHWCIdx);
        shapes.kc = wShape->GetDim(kCDimNDHWCIdx);
    } else if (wFormat == ge::Format::FORMAT_DHWCN) {
        shapes.wDIdx = kDDimDHWCNIdx;
        shapes.wHIdx = kHDimDHWCNIdx;
        shapes.wWIdx = kWDimDHWCNIdx;
        shapes.wCIdx = kCDimDHWCNIdx;
        shapes.wNIdx = kNDimDHWCNIdx;
        shapes.kd = wShape->GetDim(kDDimDHWCNIdx);
        shapes.kh = wShape->GetDim(kHDimDHWCNIdx);
        shapes.kw = wShape->GetDim(kWDimDHWCNIdx);
        shapes.kc = wShape->GetDim(kCDimDHWCNIdx);
        shapes.kn = wShape->GetDim(kNDimDHWCNIdx);
    } else {
        // format 非法拦截
        std::string correctFormat = Ops::Base::ToString(ge::Format::FORMAT_NCDHW) + ", " +
                                    Ops::Base::ToString(ge::Format::FORMAT_NDHWC) + ", " +
                                    Ops::Base::ToString(ge::Format::FORMAT_DHWCN);
        OP_LOGE_FOR_INVALID_FORMAT(context->GetNodeName(), "filter",
                                   ge::TypeUtils::FormatToAscendString(wFormat).GetString(), correctFormat.c_str());
        return false;
    }
    return true;
}

static bool GetConv3DFilterShape(const gert::InferShapeContext* context, Conv3DInputShapes& shapes)
{
    const gert::Shape* wShape = context->GetInputShape(CONV3D_FILTER_IDX);
    OPS_CHECK_NULL_WITH_CONTEXT_BOOL(context, wShape);

    // 对齐 RT1.0：filter 必须为 5 维（不支持 -2）
    if (wShape->GetDimNum() != CONV3D_DIM_SIZE_LIMIT) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "filter", std::to_string(wShape->GetDimNum()).c_str(),
                                     std::to_string(CONV3D_DIM_SIZE_LIMIT).c_str());
        return false;
    }

    const gert::CompileTimeTensorDesc* wDesc = context->GetInputDesc(CONV3D_FILTER_IDX);
    OPS_CHECK_NULL_WITH_CONTEXT_BOOL(context, wDesc);
    const ge::Format wFormat = wDesc->GetOriginFormat();
    if (!UpdateConv3DFilterDims(context, wFormat, wShape, shapes)) {
        return false;
    }

    // 对齐 Conv3DV2：filter 的 D/H/W 维不支持 0。属于空tensor校验的分支
    if (shapes.kd == 0 || shapes.kh == 0 || shapes.kw == 0) {
        std::string reason = "Shape[" + std::to_string(shapes.wDIdx) + "], shape[" + std::to_string(shapes.wHIdx) +
                             "] and shape[" + std::to_string(shapes.wWIdx) +
                             "] of this parameter must be greater than or equal to 1";
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "filter", Ops::Base::ToString(*wShape).c_str(),
                                              reason.c_str());
        return false;
    }
    return true;
}

// 属性解析（对应 RT1.0 VerifyConv3DDataFormat / VerifyConv3dStrides /
// VerifyConv3dDilations / GetPadConv3D 属性读取部分；InferShape 与 InferShapeRange 共用，
// 模板化双 context 对齐 conv_forward_infershape.cpp 惯例）
// 属性存在性 / 长度校验（strides/dilations/pads）
template <typename T>
static bool CheckConv3DAttrSize(const T* context, const gert::RuntimeAttrs* attrsPtr)
{
    // stride 长度必须为 5（对齐 RT1.0 VerifyConv3dStrides L7828）
    const gert::ContinuousVector* stridesList = attrsPtr->GetAttrPointer<gert::ContinuousVector>(CONV3D_STRIDES_IDX);
    OPS_CHECK_NULL_WITH_CONTEXT_BOOL(context, stridesList);
    if (stridesList->GetSize() != CONV3D_DIM_SIZE_LIMIT) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "strides", std::to_string(stridesList->GetSize()).c_str(),
                                     std::to_string(CONV3D_DIM_SIZE_LIMIT).c_str());
        return false;
    }

    // dilation 长度必须为 5。缺省值 [1,1,1,1,1] 由 conv3d_def.cpp 默认值承接
    // （RT1.0 提供缺省 setAttr，对齐 VerifyConv3dDilations L7800）
    const gert::ContinuousVector* dilationsList = attrsPtr->GetAttrPointer<gert::ContinuousVector>(
        CONV3D_DILATIONS_IDX);
    OPS_CHECK_NULL_WITH_CONTEXT_BOOL(context, dilationsList);
    if (dilationsList->GetSize() != CONV3D_DIM_SIZE_LIMIT) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "dilations",
                                     std::to_string(dilationsList->GetSize()).c_str(),
                                     std::to_string(CONV3D_DIM_SIZE_LIMIT).c_str());
        return false;
    }

    // pads 长度支持 1/3/6（兼容老 RT2.0 cube_util GetConv3DPads 语义，保证运行期兼容：
    // 非 950 平台图运行走本代码，原 pads=1/3 的用例不受影响）。RT1.0 仅支持 6，950 编译链路下
    // pads 非 6 会被下游 Conv3DV2 infershape/tiling 的 6 维约束拦截，报错阶段后移但拦截效果一致
    const gert::ContinuousVector* padsList = attrsPtr->GetAttrPointer<gert::ContinuousVector>(CONV3D_PADS_IDX);
    OPS_CHECK_NULL_WITH_CONTEXT_BOOL(context, padsList);
    const size_t padsSize = padsList->GetSize();
    if (padsSize != CONV3D_PADS_SIZE_LIMIT && padsSize != CONV3D_PADS_BROADCAST_SIZE_3 &&
        padsSize != CONV3D_PADS_BROADCAST_SIZE_1) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "pads", std::to_string(padsSize).c_str(),
                                     std::to_string(CONV3D_PADS_SIZE_LIMIT).c_str());
        return false;
    }
    return true;
}

// 属性枚举值校验（data_format、私有属性 padding）
template <typename T>
static bool CheckConv3DAttrValue(const T* context, const gert::RuntimeAttrs* attrsPtr, Conv3DAttrs& attrs)
{
    // data_format 校验（对齐 RT1.0 VerifyConv3DDataFormat L7860）
    const char* dataFormat = attrsPtr->GetAttrPointer<char>(CONV3D_DATA_FORMAT_IDX);
    OPS_CHECK_NULL_WITH_CONTEXT_BOOL(context, dataFormat);
    if (strcmp(dataFormat, "NDHWC") != 0 && strcmp(dataFormat, "NCDHW") != 0) {
        OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "data_format", dataFormat, "NDHWC, NCDHW");
        return false;
    }

    // 私有属性 padding（TF 前端场景由框架写入）：仅允许未设置（空）/SAME/VALID
    // （对齐 RT1.0 SetPadListConv3dForPadding L7892-7896 的非法值拦截，此前未迁移，补齐）
    const char* padding = attrsPtr->GetAttrPointer<char>(CONV3D_PADDING_IDX);
    OPS_CHECK_NULL_WITH_CONTEXT_BOOL(context, padding);
    if (padding[0] != '\0' && strcmp(padding, "SAME") != 0 && strcmp(padding, "VALID") != 0) {
        OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "padding", padding, "SAME, VALID");
        return false;
    }
    attrs.padding = padding;
    return true;
}

// pads 广播（对齐老 RT2.0 cube_util GetConv3DPads）：长度 1 全维同值；长度 3 按 [padD, padH, padW]
// 广播到各维前后；长度 6 显式逐维
static void UpdateConv3DPads(const gert::RuntimeAttrs* attrsPtr, Conv3DAttrs& attrs)
{
    const gert::ContinuousVector* padsList = attrsPtr->GetAttrPointer<gert::ContinuousVector>(CONV3D_PADS_IDX);
    const int64_t* pads = static_cast<const int64_t*>(padsList->GetData());
    const size_t padsSize = padsList->GetSize();
    if (padsSize == CONV3D_PADS_BROADCAST_SIZE_1) {
        attrs.padh = pads[0];
        attrs.padt = pads[0];
        attrs.padu = pads[0];
        attrs.padd = pads[0];
        attrs.padl = pads[0];
        attrs.padr = pads[0];
    } else if (padsSize == CONV3D_PADS_BROADCAST_SIZE_3) {
        attrs.padh = pads[0];
        attrs.padt = pads[0];
        attrs.padu = pads[1];
        attrs.padd = pads[1];
        attrs.padl = pads[2];
        attrs.padr = pads[2];
    } else {
        attrs.padh = pads[0];
        attrs.padt = pads[1];
        attrs.padu = pads[2];
        attrs.padd = pads[3];
        attrs.padl = pads[4];
        attrs.padr = pads[5];
    }
}

// 按 format 解析 D/H/W 维的 strides/dilations 值，并展开 pads 1/3/6 广播（纯解析，无失败路径）
static void UpdateConv3DDimAttrs(ge::Format xFormat, const gert::RuntimeAttrs* attrsPtr, Conv3DAttrs& attrs)
{
    const gert::ContinuousVector* stridesList = attrsPtr->GetAttrPointer<gert::ContinuousVector>(CONV3D_STRIDES_IDX);
    const gert::ContinuousVector* dilationsList = attrsPtr->GetAttrPointer<gert::ContinuousVector>(
        CONV3D_DILATIONS_IDX);
    const int64_t* strides = static_cast<const int64_t*>(stridesList->GetData());
    const int64_t* dilations = static_cast<const int64_t*>(dilationsList->GetData());
    if (xFormat == ge::Format::FORMAT_NCDHW) {
        attrs.strd = strides[kDDimNCDHWIdx];
        attrs.strh = strides[kHDimNCDHWIdx];
        attrs.strw = strides[kWDimNCDHWIdx];
        attrs.dild = dilations[kDDimNCDHWIdx];
        attrs.dilh = dilations[kHDimNCDHWIdx];
        attrs.dilw = dilations[kWDimNCDHWIdx];
    } else {
        // NDHWC，x format 已在 GetConv3DXShape 校验
        attrs.strd = strides[kDDimNDHWCIdx];
        attrs.strh = strides[kHDimNDHWCIdx];
        attrs.strw = strides[kWDimNDHWCIdx];
        attrs.dild = dilations[kDDimNDHWCIdx];
        attrs.dilh = dilations[kHDimNDHWCIdx];
        attrs.dilw = dilations[kWDimNDHWCIdx];
    }
    UpdateConv3DPads(attrsPtr, attrs);
}

// strides / dilations 值域校验
template <typename T>
static bool CheckConv3DStrideDilation(const T* context, const gert::RuntimeAttrs* attrsPtr, const Conv3DAttrs& attrs)
{
    // strides 必须为正数（对齐 RT1.0 GetAttrsConv3D L8023，canndev 老 RT2.0 仅拦截 0，此处恢复；
    // conv3dv2 tiling ParseStrideLegal 拦截 [1, MAX]，该收紧合理）
    if (attrs.strd <= 0 || attrs.strh <= 0 || attrs.strw <= 0) {
        std::string reason = "All dimensions of this parameter must be greater than or equal to 1";
        const gert::ContinuousVector* stridesList = attrsPtr->GetAttrPointer<gert::ContinuousVector>(
            CONV3D_STRIDES_IDX);
        const int64_t* strides = static_cast<const int64_t*>(stridesList->GetData());
        std::vector<int64_t> stridesVec(strides, strides + CONV3D_DIM_SIZE_LIMIT);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "strides", VectorToString(stridesVec).c_str(),
                                              reason.c_str());
        return false;
    }

    // dilation 值域（对齐 conv3dv2 arch35 tiling ParseDilationLegal：D 维 [1, 1000000]、H/W 维 [1, 255]，
    // 同时收口输出维公式乘加链的溢出触发域）
    if (attrs.dild < 1 || attrs.dild > CONV3D_MAX_DILATION_D || attrs.dilh < 1 || attrs.dilh > CONV3D_MAX_DILATION_H ||
        attrs.dilw < 1 || attrs.dilw > CONV3D_MAX_DILATION_W) {
        std::string reason = "The current value is not within the valid range. The valid range of D dim is [1, " +
                             std::to_string(CONV3D_MAX_DILATION_D) + "], and the valid range of H/W dim is [1, " +
                             std::to_string(CONV3D_MAX_DILATION_H) + "]";
        const gert::ContinuousVector* dilationsList = attrsPtr->GetAttrPointer<gert::ContinuousVector>(
            CONV3D_DILATIONS_IDX);
        const int64_t* dilations = static_cast<const int64_t*>(dilationsList->GetData());
        std::vector<int64_t> dilationsVec(dilations, dilations + CONV3D_DIM_SIZE_LIMIT);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "dilations", VectorToString(dilationsVec).c_str(),
                                              reason.c_str());
        return false;
    }
    return true;
}

template <typename T>
static bool ParseConv3DAttrs(const T* context, ge::Format xFormat, Conv3DAttrs& attrs)
{
    const gert::RuntimeAttrs* attrsPtr = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT_BOOL(context, attrsPtr);
    if (!CheckConv3DAttrSize(context, attrsPtr) || !CheckConv3DAttrValue(context, attrsPtr, attrs)) {
        return false;
    }
    UpdateConv3DDimAttrs(xFormat, attrsPtr, attrs);
    return CheckConv3DStrideDilation(context, attrsPtr, attrs);
}

// int64 溢出保护
// 卷积输出维公式 (in + padBefore + padAfter - dil * (kernel - 1) - 1) 的乘加链逐级溢出检查，
// 任一中间结果溢出返回 false，杜绝有符号整数溢出 UB
static bool CalcConv3DNumerator(int64_t inputSize, int64_t padBefore, int64_t padAfter, int64_t dilation,
                                int64_t kernel, int64_t& numerator)
{
    int64_t kernelSub1 = 0;
    int64_t dilatedKernelSub1 = 0;
    int64_t result = inputSize;
    if (__builtin_sub_overflow(kernel, 1, &kernelSub1) ||
        __builtin_mul_overflow(dilation, kernelSub1, &dilatedKernelSub1) ||
        __builtin_add_overflow(result, padBefore, &result) || __builtin_add_overflow(result, padAfter, &result) ||
        __builtin_sub_overflow(result, dilatedKernelSub1, &result) || __builtin_sub_overflow(result, 1, &result)) {
        return false;
    }
    numerator = result;
    return true;
}

// 输出维推导（动态维 -1 传播 + int64 溢出保护）：inputSize/kernel 任一为 -1 时输出 -1，
// 公式为 numerator / stride + 1（stride 已在 ParseConv3DAttrs 校验为正）
static bool CalcConv3DOutDim(int64_t inputSize, int64_t kernel, int64_t padBefore, int64_t padAfter, int64_t dilation,
                             int64_t stride, int64_t& outDim)
{
    if (inputSize == UNKNOWN_DIM_VALUE_ || kernel == UNKNOWN_DIM_VALUE_) {
        outDim = UNKNOWN_DIM_VALUE_;
        return true;
    }
    int64_t numerator = 0;
    return CalcConv3DNumerator(inputSize, padBefore, padAfter, dilation, kernel, numerator) &&
           __builtin_add_overflow(numerator / stride, 1, &outDim) == 0;
}

// 对齐 RT1.0 SetPadListConv3dForPadding 中 padding=SAME 的 pad 计算逻辑（动态维记为 -1）；
// dilateKernel 乘加链带溢出检查
static bool CalcSamePaddingForDim(int64_t inputSize, int64_t kernel, int64_t stride, int64_t dilation,
                                  int64_t& padBefore, int64_t& padAfter)
{
    if (inputSize < 0 || kernel < 0) {
        // 动态 shape 场景无法计算具体 pad，对齐 RT1.0 SetConv3dDynamicPads 记为 -1
        padBefore = UNKNOWN_DIM_VALUE_;
        padAfter = UNKNOWN_DIM_VALUE_;
        return true;
    }
    const int64_t tails = inputSize % stride;
    int64_t kernelSub1 = 0;
    int64_t dilateKernel = 0;
    if (__builtin_sub_overflow(kernel, 1, &kernelSub1) || __builtin_mul_overflow(dilation, kernelSub1, &dilateKernel) ||
        __builtin_add_overflow(dilateKernel, 1, &dilateKernel)) {
        return false;
    }
    const int64_t padNeeded = std::max((tails > 0 ? dilateKernel - tails : dilateKernel - stride),
                                       static_cast<int64_t>(0));
    padBefore = padNeeded / PAD_HALF_DIV;
    padAfter = padNeeded - padBefore;
    return true;
}

static bool CalcConv3DPads(const gert::InferShapeContext* context, const Conv3DInputShapes& shapes, Conv3DAttrs& attrs)
{
    // 私有属性 padding（TF 前端场景由框架写入），对齐 RT1.0 SetPadListConv3dForPadding：
    // SAME 按输入/卷积核动态计算，VALID 置 0（canndev 老 RT2.0 未处理 VALID，此处补齐）
    if (strcmp(attrs.padding, "SAME") == 0) {
        if (!CalcSamePaddingForDim(shapes.id, shapes.kd, attrs.strd, attrs.dild, attrs.padh, attrs.padt) ||
            !CalcSamePaddingForDim(shapes.ih, shapes.kh, attrs.strh, attrs.dilh, attrs.padu, attrs.padd) ||
            !CalcSamePaddingForDim(shapes.iw, shapes.kw, attrs.strw, attrs.dilw, attrs.padl, attrs.padr)) {
            std::vector<const gert::Shape*> xwShapes = {context->GetInputShape(CONV3D_X_IDX),
                                                        context->GetInputShape(CONV3D_FILTER_IDX)};
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                context->GetNodeName(), "x, filter", Ops::Base::ToString(xwShapes).c_str(),
                "Failed to calculate the SAME padding of Conv3D because of int64 overflow");
            return false;
        }
    } else if (strcmp(attrs.padding, "VALID") == 0) {
        attrs.padh = 0;
        attrs.padt = 0;
        attrs.padu = 0;
        attrs.padd = 0;
        attrs.padl = 0;
        attrs.padr = 0;
    }

    // 对齐 RT1.0 GetPadConv3D L7985-7991：静态 x 下 pads 不允许为负（动态 shape 放行）；
    // SAME 的 pads 为推导值（动态输入/卷积核维记 -1，对齐 RT1.0 SetConv3dDynamicPads 语义），同样放行
    bool isStaticX = !shapes.unknownShapeX && !shapes.unknownRankX;
    bool isSamePadding = strcmp(attrs.padding, "SAME") == 0;
    if (isStaticX && !isSamePadding &&
        (attrs.padh < 0 || attrs.padt < 0 || attrs.padu < 0 || attrs.padd < 0 || attrs.padl < 0 || attrs.padr < 0)) {
        std::string reason = "All dimensions of this parameter must be greater than or equal to 0";
        std::vector<int64_t> padsVec = {attrs.padh, attrs.padt, attrs.padu, attrs.padd, attrs.padl, attrs.padr};
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "pads", VectorToString(padsVec).c_str(),
                                              reason.c_str());
        return false;
    }

    // RT1.0 中基于 auto_pad 属性推导 pads 的功能点（SetPadListConv3dForAutoPad）未迁移：
    // onnx 5D 路径非 NOTSET/VALID 的 auto_pad 由前端插件（conv_onnx_plugin.cpp）直接拦截，
    // onnx 插件会将 VALID 场景转换为全 0 pads 透传；conv3d_proto.h 亦未注册 auto_pad 属性
    return true;
}

// 校验逻辑（对应 RT1.0 CheckInputZeroTensor / SetGroupsConv /
// CheckZeroTensorLegal / CheckConv3DInputWithPad）
static bool CheckConv3DGroups(const gert::InferShapeContext* context, const Conv3DInputShapes& shapes,
                              Conv3DAttrs& attrs)
{
    // 对齐 RT1.0：unknown rank 或 ic/kc 未知（动态或 0）时跳过 groups 校验
    if (shapes.unknownRankX || shapes.ic <= 0 || shapes.kc <= 0) {
        return true;
    }

    const gert::RuntimeAttrs* attrsPtr = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT_BOOL(context, attrsPtr);
    const int64_t* groupsPtr = attrsPtr->GetAttrPointer<int64_t>(CONV3D_GROUPS_IDX);
    OPS_CHECK_NULL_WITH_CONTEXT_BOOL(context, groupsPtr);
    attrs.groups = *groupsPtr;

    if (shapes.ic % shapes.kc != 0) {
        std::vector<const gert::Shape*> xwShapes = {context->GetInputShape(CONV3D_X_IDX),
                                                    context->GetInputShape(CONV3D_FILTER_IDX)};
        std::string reason = "Shape[" + std::to_string(shapes.xCIdx) + "] of x must be exactly divisible by Shape[" +
                             std::to_string(shapes.wCIdx) + "] of filter";
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "x, filter",
                                               Ops::Base::ToString(xwShapes).c_str(), reason.c_str());
        return false;
    }
    // 对齐 RT1.0 SetGroupsConv：groups 未显式设置（为 1）时按 ic / kc 隐式生效；
    // RT2.0 无法 SetAttr，隐式 groups 由 conv3dv2 tiling 中同等逻辑承接
    const int64_t implicitGroups = shapes.ic / shapes.kc;
    if (attrs.groups != 1 && attrs.groups != implicitGroups) {
        std::vector<const gert::Shape*> xwShapes = {context->GetInputShape(CONV3D_X_IDX),
                                                    context->GetInputShape(CONV3D_FILTER_IDX)};
        std::string reason = "Shape[" + std::to_string(shapes.xCIdx) + "] of x must be equal to shape[" +
                             std::to_string(shapes.wCIdx) + "] of filter multiplied by groups " +
                             std::to_string(attrs.groups);
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "x, filter",
                                               Ops::Base::ToString(xwShapes).c_str(), reason.c_str());
        return false;
    }
    // 隐式 groups（groups==1 且 ic 可被 kc 整除时按 ic/kc 生效）的推导已下沉到 conv3dv2 tiling
    return true;
}

static bool CheckConv3DZeroTensor(const Conv3DInputShapes& shapes, int64_t od, int64_t oh, int64_t ow)
{
    if (!shapes.isInputZeroTensor) {
        return true;
    }
    // 对齐 RT1.0 CheckZeroTensorLegal：零 Tensor 输入时输出不允许出现负数
    // （输出维公式在零输入下可能推负，此处防御性拦截）
    if (shapes.in < 0 || shapes.kn < 0 || od < 0 || oh < 0 || ow < 0) {
        return false;
    }
    // 零 Tensor 输入必须推导出零 Tensor 输出
    if (shapes.in != 0 && shapes.kn != 0 && od != 0 && oh != 0 && ow != 0) {
        return false;
    }
    return true;
}

// 对齐家族 ReportInputWithPadError 报错模式（conv_forward_infershape.cpp）：携带 x/filter/pads/dilations
static bool ReportConv3DInputWithPadError(const gert::InferShapeContext* context, const Conv3DAttrs& attrs,
                                          const std::string& reason)
{
    const gert::Shape* xShape = context->GetInputShape(CONV3D_X_IDX);
    OPS_CHECK_NULL_WITH_CONTEXT_BOOL(context, xShape);
    const gert::Shape* wShape = context->GetInputShape(CONV3D_FILTER_IDX);
    OPS_CHECK_NULL_WITH_CONTEXT_BOOL(context, wShape);
    const gert::RuntimeAttrs* attrsPtr = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT_BOOL(context, attrsPtr);
    const gert::ContinuousVector* dilationsList = attrsPtr->GetAttrPointer<gert::ContinuousVector>(
        CONV3D_DILATIONS_IDX);
    OPS_CHECK_NULL_WITH_CONTEXT_BOOL(context, dilationsList);
    const int64_t* dilationsPtr = static_cast<const int64_t*>(dilationsList->GetData());
    std::vector<int64_t> dilations(dilationsPtr, dilationsPtr + CONV3D_DIM_SIZE_LIMIT);
    std::vector<int64_t> pads = {attrs.padh, attrs.padt, attrs.padu, attrs.padd, attrs.padl, attrs.padr};
    std::string incorrectShapes = Ops::Base::ToString(*xShape) + Ops::Base::ToString(*wShape) + VectorToString(pads) +
                                  VectorToString(dilations);
    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "x, filter, pads, dilations",
                                           incorrectShapes.c_str(), reason.c_str());
    return false;
}

static bool CheckConv3DInputWithPad(const gert::InferShapeContext* context, const Conv3DInputShapes& shapes,
                                    const Conv3DAttrs& attrs)
{
    if (shapes.isInputZeroTensor) {
        return true;
    }
    // 对齐 RT1.0 CheckConv3DInputWithPad：padding 后的输入尺寸必须大于等于卷积核尺寸
    if (shapes.id > 0 && shapes.kd > 0 && shapes.ih > 0 && shapes.kh > 0 && shapes.iw > 0 && shapes.kw > 0) {
        int64_t idPad = 0;
        int64_t ihPad = 0;
        int64_t iwPad = 0;
        if (!CalcConv3DNumerator(shapes.id, attrs.padh, attrs.padt, attrs.dild, shapes.kd, idPad) ||
            !CalcConv3DNumerator(shapes.ih, attrs.padu, attrs.padd, attrs.dilh, shapes.kh, ihPad) ||
            !CalcConv3DNumerator(shapes.iw, attrs.padl, attrs.padr, attrs.dilw, shapes.kw, iwPad)) {
            std::vector<const gert::Shape*> xwShapes = {context->GetInputShape(CONV3D_X_IDX),
                                                        context->GetInputShape(CONV3D_FILTER_IDX)};
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                context->GetNodeName(), "x, filter", Ops::Base::ToString(xwShapes).c_str(),
                "Failed to calculate the padded input size of Conv3D because of int64 overflow");
            return false;
        }
        // 对齐家族 CheckInputWithPad 报错格式：x[idx] + pads[..] + pads[..] < dilations[idx] * (filter[idx] - 1) + 1
        if (idPad < 0) {
            std::stringstream reason;
            reason << "x[" + std::to_string(shapes.xDIdx) + "] + pads[0] + pads[1] < dilations[" +
                          std::to_string(shapes.xDIdx) + "] * (filter[" + std::to_string(shapes.wDIdx) + "] - 1) + 1, ";
            reason << "indicating that the filter is greater than the feature map and convolution compute "
                      "cannot be performed";
            return ReportConv3DInputWithPadError(context, attrs, reason.str());
        }
        if (ihPad < 0) {
            std::stringstream reason;
            reason << "x[" + std::to_string(shapes.xHIdx) + "] + pads[2] + pads[3] < dilations[" +
                          std::to_string(shapes.xHIdx) + "] * (filter[" + std::to_string(shapes.wHIdx) + "] - 1) + 1, ";
            reason << "indicating that the filter is greater than the feature map and convolution compute "
                      "cannot be performed";
            return ReportConv3DInputWithPadError(context, attrs, reason.str());
        }
        if (iwPad < 0) {
            std::stringstream reason;
            reason << "x[" + std::to_string(shapes.xWIdx) + "] + pads[4] + pads[5] < dilations[" +
                          std::to_string(shapes.xWIdx) + "] * (filter[" + std::to_string(shapes.wWIdx) + "] - 1) + 1, ";
            reason << "indicating that the filter is greater than the feature map and convolution compute "
                      "cannot be performed";
            return ReportConv3DInputWithPadError(context, attrs, reason.str());
        }
    }
    return true;
}

// dtype 一致性（对应 RT1.0 Conv3DVerify L8719-8730 的运行期兜底）
template <typename T>
static bool CheckConv3DDtypeConsistency(const T* context, const gert::CompileTimeTensorDesc* xDesc)
{
    const gert::CompileTimeTensorDesc* wDesc = context->GetInputDesc(CONV3D_FILTER_IDX);
    OPS_CHECK_NULL_WITH_CONTEXT_BOOL(context, wDesc);
    if (xDesc->GetDataType() == wDesc->GetDataType()) {
        return true;
    }
    OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context->GetNodeName(), "x and filter",
                                           (ge::TypeUtils::DataTypeToSerialString(xDesc->GetDataType()) + " and " +
                                            ge::TypeUtils::DataTypeToSerialString(wDesc->GetDataType()))
                                               .c_str(),
                                           "The dtypes of x and filter must be the same");
    return false;
}

// 输出 shape 推导（对应 RT1.0 Conv3DInfer 主流程）
// 计算 od / oh / ow，动态维度（-1）传播（对齐 Conv3DV2 rt2.0实现方式）；
// 乘加链带 int64 溢出保护，溢出直接拦截；unknownRankX 时保持 -1（对应 RT1.0 默认值）
static ge::graphStatus CalcConv3DOutDims(const gert::InferShapeContext* context, const Conv3DInputShapes& shapes,
                                         const Conv3DAttrs& attrs, int64_t& od, int64_t& oh, int64_t& ow)
{
    od = UNKNOWN_DIM_VALUE_;
    oh = UNKNOWN_DIM_VALUE_;
    ow = UNKNOWN_DIM_VALUE_;
    if (shapes.unknownRankX) {
        return ge::GRAPH_SUCCESS;
    }
    if (!CalcConv3DOutDim(shapes.id, shapes.kd, attrs.padh, attrs.padt, attrs.dild, attrs.strd, od) ||
        !CalcConv3DOutDim(shapes.ih, shapes.kh, attrs.padu, attrs.padd, attrs.dilh, attrs.strh, oh) ||
        !CalcConv3DOutDim(shapes.iw, shapes.kw, attrs.padl, attrs.padr, attrs.dilw, attrs.strw, ow)) {
        std::vector<const gert::Shape*> xwShapes = {context->GetInputShape(CONV3D_X_IDX),
                                                    context->GetInputShape(CONV3D_FILTER_IDX)};
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context->GetNodeName(), "x, filter", Ops::Base::ToString(xwShapes).c_str(),
            "Failed to calculate the output dim of Conv3D because of int64 overflow");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// 零 Tensor 输出合法性校验（对应 RT1.0 CheckZeroTensorLegal：零输入不允许推导出非零输出）
static ge::graphStatus CheckConv3DZeroTensorOutput(const gert::InferShapeContext* context,
                                                   const Conv3DInputShapes& shapes, int64_t od, int64_t oh, int64_t ow)
{
    if (CheckConv3DZeroTensor(shapes, od, oh, ow)) {
        return ge::GRAPH_SUCCESS;
    }
    // 对齐家族 CheckOutputZeroTensor 报错模式（conv_forward_infershape.cpp）
    std::string reason = "If any dimension of the shape of x or shape[" + std::to_string(shapes.wNIdx) +
                         "] of filter is 0, any dimension of the shape of y must be equal to 0";
    std::vector<std::vector<int64_t>> xyInferredShapes = {{shapes.in, shapes.ic, shapes.id, shapes.ih, shapes.iw},
                                                          {shapes.in, shapes.kn, od, oh, ow}};
    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "x, y(inferred)",
                                           VectorsToString(xyInferredShapes).c_str(), reason.c_str());
    return ge::GRAPH_FAILED;
}

// 按 y format 设置输出 shape
static ge::graphStatus SetConv3DOutputShape(gert::InferShapeContext* context, ge::Format yFormat,
                                            const Conv3DInputShapes& shapes, int64_t od, int64_t oh, int64_t ow)
{
    gert::Shape* yShape = context->GetOutputShape(CONV3D_Y_IDX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, yShape);
    yShape->SetDimNum(0);
    if (yFormat == ge::Format::FORMAT_NCDHW) {
        yShape->AppendDim(shapes.in);
        yShape->AppendDim(shapes.kn);
        yShape->AppendDim(od);
        yShape->AppendDim(oh);
        yShape->AppendDim(ow);
    } else if (yFormat == ge::Format::FORMAT_NDHWC) {
        yShape->AppendDim(shapes.in);
        yShape->AppendDim(od);
        yShape->AppendDim(oh);
        yShape->AppendDim(ow);
        yShape->AppendDim(shapes.kn);
    } else {
        // output format非法拦截
        std::string correctFormat = Ops::Base::ToString(ge::Format::FORMAT_NCDHW) + ", " +
                                    Ops::Base::ToString(ge::Format::FORMAT_NDHWC);
        OP_LOGE_FOR_INVALID_FORMAT(context->GetNodeName(), "y",
                                   ge::TypeUtils::FormatToAscendString(yFormat).GetString(), correctFormat.c_str());
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "Conv3D y shape: %s", Ops::Base::ToString(*yShape).c_str());
    return ge::GRAPH_SUCCESS;
}

// 解析并校验输入 shape / format / dtype / 属性 / groups（InferShape 主流程的输入准备）
static ge::graphStatus PrepareConv3DInputs(const gert::InferShapeContext* context,
                                           const gert::CompileTimeTensorDesc* xDesc, ge::Format xFormat,
                                           Conv3DInputShapes& shapes, Conv3DAttrs& attrs)
{
    // 解析输入 shape / format
    if (!GetConv3DXShape(context, xFormat, shapes) || !GetConv3DFilterShape(context, shapes)) {
        CUBE_INNER_ERR_REPORT(context->GetNodeName(), "failed to get input shape for Conv3D.");
        return ge::GRAPH_FAILED;
    }
    // x 与 filter dtype 一致性校验
    if (!CheckConv3DDtypeConsistency(context, xDesc)) {
        return ge::GRAPH_FAILED;
    }
    // 解析属性（data_format/strides/dilations/pads/padding）与 pads 展开
    if (!ParseConv3DAttrs(context, xFormat, attrs) || !CalcConv3DPads(context, shapes, attrs)) {
        CUBE_INNER_ERR_REPORT(context->GetNodeName(), "failed to get attrs for Conv3D.");
        return ge::GRAPH_FAILED;
    }
    // groups 校验（ic、kc 均已知时生效）
    if (!CheckConv3DGroups(context, shapes, attrs)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// 零 Tensor 识别（对齐 RT1.0：动态 shape 与 unknown rank 场景跳过）
static void UpdateConv3DZeroTensorFlag(const char* opName, Conv3DInputShapes& shapes, bool isDynamic)
{
    if (isDynamic) {
        return;
    }
    // 空入校验：x任意维=0 || filter的N/C维=0  -> 识别为空进
    if (shapes.in == 0 || shapes.ic == 0 || shapes.id == 0 || shapes.ih == 0 || shapes.iw == 0 || shapes.kn == 0 ||
        shapes.kc == 0) {
        shapes.isInputZeroTensor = true;
        OP_LOGD(opName, "Input is zero tensor.");
    }
}

static ge::graphStatus InferShapeForConv3D(gert::InferShapeContext* context)
{
    OP_CHECK(context == nullptr, CUBE_INNER_ERR_REPORT("Conv3D", "context is null."), return ge::GRAPH_FAILED);
    const auto opName = context->GetNodeName();
    OP_LOGD(opName, "Enter Conv3D InferShape.");

    const gert::CompileTimeTensorDesc* xDesc = context->GetInputDesc(CONV3D_X_IDX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    const ge::Format xFormat = xDesc->GetOriginFormat();
    OP_LOGE_IF(xFormat == ge::Format::FORMAT_RESERVED, ge::GRAPH_FAILED, opName, "Get x format failed: %d.", xFormat);

    const gert::CompileTimeTensorDesc* yDesc = context->GetOutputDesc(CONV3D_Y_IDX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, yDesc);
    const ge::Format yFormat = yDesc->GetOriginFormat();
    OP_LOGE_IF(yFormat == ge::Format::FORMAT_RESERVED, ge::GRAPH_FAILED, opName, "Get y format failed: %d.", yFormat);

    Conv3DInputShapes shapes;
    Conv3DAttrs attrs;
    if (PrepareConv3DInputs(context, xDesc, xFormat, shapes, attrs) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // 零 Tensor 识别 + 输出推导（动态 shape 与 unknown rank 场景跳过静态校验）
    const bool isDynamic = shapes.unknownShapeX && !shapes.unknownRankX;
    UpdateConv3DZeroTensorFlag(opName, shapes, isDynamic);

    // 计算 od / oh / ow（含 int64 溢出拦截）
    int64_t od = 0;
    int64_t oh = 0;
    int64_t ow = 0;
    if (CalcConv3DOutDims(context, shapes, attrs, od, oh, ow) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // 零 Tensor 输出合法性 + padding 后输入尺寸校验（静态 shape 场景）
    if (!isDynamic) {
        if (CheckConv3DZeroTensorOutput(context, shapes, od, oh, ow) != ge::GRAPH_SUCCESS ||
            !CheckConv3DInputWithPad(context, shapes, attrs)) {
            return ge::GRAPH_FAILED;
        }
    }

    // 按 y format 设置输出 shape
    return SetConv3DOutputShape(context, yFormat, shapes, od, oh, ow);
}

// InferShapeRange（对应 RT1.0 SetConv3dOutShapeRange）
static bool CheckConv3DInputRangeDim(const gert::InferShapeRangeContext* context, size_t fmDimNum, size_t wDimNum)
{
    // 输入fm_range和filter_range都必须为5维
    if (fmDimNum != CONV3D_DIM_SIZE_LIMIT || wDimNum != CONV3D_DIM_SIZE_LIMIT) {
        std::string incorrectDims = std::to_string(fmDimNum) + ", " + std::to_string(wDimNum);
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(context->GetNodeName(), "x_range_max, filter_range_max",
                                                  incorrectDims.c_str(),
                                                  "The dims of shape_range_max of x and filter must be 5");
        return false;
    }
    return true;
}

// 通用公式 (in + padBefore + padAfter - dilation * (kernel - 1) - 1) / stride + 1（对齐 RT1.0
// SetConv3dOutShapeDimRange L8074-8075；RT1.0 无 kernel 为 0 的零 Tensor 特判分支，kernel 为 0 的
// 场景已在 InferShape 零 Tensor 校验拦截）。乘加链经 CalcConv3DNumerator 逐级溢出检查，
// 与 InferShape 路径防护语义一致；上界为 -1（无信息）时透传 -1
static bool SetConv3DOutRangeForDim(int64_t padBefore, int64_t padAfter, int64_t stride, int64_t dilation,
                                    const std::pair<int64_t, int64_t>& kernel,
                                    const std::pair<int64_t, int64_t>& inRange, std::pair<int64_t, int64_t>& outRange)
{
    if (stride <= 0) {
        return true;
    }
    const int64_t low = inRange.first;
    const int64_t high = inRange.second;
    int64_t numerator = 0;
    if (!CalcConv3DNumerator(low, padBefore, padAfter, dilation, kernel.first, numerator) ||
        __builtin_add_overflow(numerator / stride, 1, &outRange.first) != 0) {
        return false;
    }
    if (high == UNKNOWN_DIM_VALUE_) {
        outRange.second = high;
        return true;
    }
    return CalcConv3DNumerator(high, padBefore, padAfter, dilation, kernel.second, numerator) &&
           __builtin_add_overflow(numerator / stride, 1, &outRange.second) == 0;
}

// SAME 场景输出维为 ceil(in / stride)（RT1.0 笔误缺陷的迁移修复，见 InferShapeRangeForConv3D 内
// 注释），in + stride - 1 加法带溢出检查；上界为 -1（无信息）时透传 -1
static bool SetConv3DSameOutRangeForDim(const std::pair<int64_t, int64_t>& inRange, int64_t stride,
                                        std::pair<int64_t, int64_t>& outRange)
{
    int64_t numerator = 0;
    if (__builtin_add_overflow(inRange.first, stride - 1, &numerator) != 0) {
        return false;
    }
    outRange.first = numerator / stride;
    if (inRange.second == UNKNOWN_DIM_VALUE_) {
        outRange.second = UNKNOWN_DIM_VALUE_;
        return true;
    }
    if (__builtin_add_overflow(inRange.second, stride - 1, &numerator) != 0) {
        return false;
    }
    outRange.second = numerator / stride;
    return true;
}

// Range 路径属性解析：复用 InferShape 的模板解析与校验，VALID 时 pads 置 0
// （对齐 RT1.0 range 公式的 pads 来自 GetPadConv3D 解析结果、VALID 已置 0，与 Conv3DV2 家族惯例
// conv_forward_infershape.cpp GetPadModeAndUpdatePad 的 InferShapeRangeContext 分支一致；
// SAME 由 ceil 分支承接，无需 pads 具体值）
static ge::graphStatus ParseConv3DRangeAttrs(const gert::InferShapeRangeContext* context, ge::Format xFormat,
                                             Conv3DAttrs& attrs)
{
    if (!ParseConv3DAttrs(context, xFormat, attrs)) {
        CUBE_INNER_ERR_REPORT(context->GetNodeName(), "failed to get attrs for Conv3D.");
        return ge::GRAPH_FAILED;
    }
    if (strcmp(attrs.padding, "VALID") == 0) {
        attrs.padh = 0;
        attrs.padt = 0;
        attrs.padu = 0;
        attrs.padd = 0;
        attrs.padl = 0;
        attrs.padr = 0;
    }
    return ge::GRAPH_SUCCESS;
}

// x / filter 各维（N C D H W）range 下标映射（含 filter format 拦截校验，
// 仅 NCDHW/NDHWC/DHWCN，对齐 InferShape 的 GetConv3DFilterShape）
static ge::graphStatus GetConv3DRangeDimIdx(const gert::InferShapeRangeContext* context, ge::Format xFormat,
                                            std::vector<size_t>& xIdx, std::vector<size_t>& wIdx)
{
    xIdx = {kNDimNDHWCIdx, kCDimNDHWCIdx, kDDimNDHWCIdx, kHDimNDHWCIdx, kWDimNDHWCIdx};
    wIdx = {kNDimNDHWCIdx, kCDimNDHWCIdx, kDDimNDHWCIdx, kHDimNDHWCIdx, kWDimNDHWCIdx};
    if (xFormat == ge::Format::FORMAT_NCDHW) {
        xIdx = {kNDimNCDHWIdx, kCDimNCDHWIdx, kDDimNCDHWIdx, kHDimNCDHWIdx, kWDimNCDHWIdx};
    }
    const gert::CompileTimeTensorDesc* wDesc = context->GetInputDesc(CONV3D_FILTER_IDX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, wDesc);
    const ge::Format wFormat = wDesc->GetOriginFormat();
    if (wFormat == ge::Format::FORMAT_NCDHW) {
        wIdx = {kNDimNCDHWIdx, kCDimNCDHWIdx, kDDimNCDHWIdx, kHDimNCDHWIdx, kWDimNCDHWIdx};
    } else if (wFormat == ge::Format::FORMAT_DHWCN) {
        wIdx = {kNDimDHWCNIdx, kCDimDHWCNIdx, kDDimDHWCNIdx, kHDimDHWCNIdx, kWDimDHWCNIdx};
    } else if (wFormat != ge::Format::FORMAT_NDHWC) {
        std::string correctFormat = Ops::Base::ToString(ge::Format::FORMAT_NCDHW) + ", " +
                                    Ops::Base::ToString(ge::Format::FORMAT_NDHWC) + ", " +
                                    Ops::Base::ToString(ge::Format::FORMAT_DHWCN);
        OP_LOGE_FOR_INVALID_FORMAT(context->GetNodeName(), "filter",
                                   ge::TypeUtils::FormatToAscendString(wFormat).GetString(), correctFormat.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// 推导输出各维 range（对应 RT1.0 SetConv3dOutShapeRange）：N 维原样继承 x 的 N 维 range；
// C 维继承 filter 的 N 维 range（kn 静态时即 (kn, kn)，动态时保留区间），下界为 -1（无 range
// 信息）时退化为 (1, -1)，不做 4096 封顶；D/H/W 维：SAME 场景输出为 ceil(in / stride)。
// RT1.0 该 SAME 分支为死代码（调用方 padding 局部变量 L8142 声明后未读取 padding 属性、恒为空串，
// 恒走 L8074-8075 通用公式，动态 SAME 下界因 as-if pads 失真；对照同文件 Conv2D/DepthwiseConv2D
// 的 GetConv2dOutShapeRange/GetDepthwiseConv2dOutShapeRange 均正确读取 padding 属性、SAME 走
// ceil 分支，确认为 RT1.0 笔误缺陷而非有意保守）。RT2.0 迁移时修复该缺陷，按 SAME 真实语义
// ceil 推导；非 SAME 场景为 (in + padSum - dilation * (kernel - 1) - 1) / stride + 1（对齐
// RT1.0 L8074-8075 通用公式），溢出拦截；上下界收敛仅对 D/H/W 维生效（对齐 RT1.0
// SetConv3dOutShapeDimRange L8078-8083：下界不小于 1（无零 Tensor 下界 0 特判），上界 -1 透传、
// 否则封顶 4096）
static ge::graphStatus CalcConv3DOutDimsRange(const gert::InferShapeRangeContext* context, const Conv3DAttrs& attrs,
                                              const gert::Range<gert::Shape>* fmShapeRange,
                                              const gert::Range<gert::Shape>* wShapeRange,
                                              const std::vector<std::pair<int64_t, int64_t>>& fmRange,
                                              const std::vector<std::pair<int64_t, int64_t>>& wRange,
                                              const std::vector<size_t>& xIdx, const std::vector<size_t>& wIdx,
                                              std::vector<std::pair<int64_t, int64_t>>& outRange)
{
    outRange[xIdx[IDX_LIST_N_IDX]] = fmRange[xIdx[IDX_LIST_N_IDX]];
    outRange[xIdx[IDX_LIST_C_IDX]] = wRange[wIdx[IDX_LIST_N_IDX]];
    if (outRange[xIdx[IDX_LIST_C_IDX]].first < 0) {
        outRange[xIdx[IDX_LIST_C_IDX]].first = DYNAMIC_RANGE_LOWER_BOUND;
    }
    bool isRangeCalcOk = true;
    if (strcmp(attrs.padding, "SAME") == 0) {
        isRangeCalcOk = SetConv3DSameOutRangeForDim(fmRange[xIdx[IDX_LIST_D_IDX]], attrs.strd,
                                                    outRange[xIdx[IDX_LIST_D_IDX]]) &&
                        SetConv3DSameOutRangeForDim(fmRange[xIdx[IDX_LIST_H_IDX]], attrs.strh,
                                                    outRange[xIdx[IDX_LIST_H_IDX]]) &&
                        SetConv3DSameOutRangeForDim(fmRange[xIdx[IDX_LIST_W_IDX]], attrs.strw,
                                                    outRange[xIdx[IDX_LIST_W_IDX]]);
    } else {
        isRangeCalcOk = SetConv3DOutRangeForDim(attrs.padh, attrs.padt, attrs.strd, attrs.dild,
                                                wRange[wIdx[IDX_LIST_D_IDX]], fmRange[xIdx[IDX_LIST_D_IDX]],
                                                outRange[xIdx[IDX_LIST_D_IDX]]) &&
                        SetConv3DOutRangeForDim(attrs.padu, attrs.padd, attrs.strh, attrs.dilh,
                                                wRange[wIdx[IDX_LIST_H_IDX]], fmRange[xIdx[IDX_LIST_H_IDX]],
                                                outRange[xIdx[IDX_LIST_H_IDX]]) &&
                        SetConv3DOutRangeForDim(attrs.padl, attrs.padr, attrs.strw, attrs.dilw,
                                                wRange[wIdx[IDX_LIST_W_IDX]], fmRange[xIdx[IDX_LIST_W_IDX]],
                                                outRange[xIdx[IDX_LIST_W_IDX]]);
    }
    if (!isRangeCalcOk) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context->GetNodeName(), "x, filter",
            (Ops::Base::ToString(*fmShapeRange->GetMax()) + ", " + Ops::Base::ToString(*wShapeRange->GetMax())).c_str(),
            "Failed to calculate the output range of Conv3D because of int64 overflow");
        return ge::GRAPH_FAILED;
    }
    const size_t dhwIdxList[] = {xIdx[IDX_LIST_D_IDX], xIdx[IDX_LIST_H_IDX], xIdx[IDX_LIST_W_IDX]};
    for (size_t idx : dhwIdxList) {
        outRange[idx].first = std::max(outRange[idx].first, DYNAMIC_RANGE_LOWER_BOUND);
        if (outRange[idx].second != UNKNOWN_DIM_VALUE_) {
            outRange[idx].second = std::min(outRange[idx].second, DYNAMIC_RANGE_UPPER_BOUND);
        }
    }
    return ge::GRAPH_SUCCESS;
}

// 写回输出 range
static ge::graphStatus SetConv3DOutputRange(gert::InferShapeRangeContext* context,
                                            gert::Range<gert::Shape>* outShapeRange,
                                            const std::vector<std::pair<int64_t, int64_t>>& outRange, size_t fmDimNum)
{
    outShapeRange->GetMin()->SetDimNum(fmDimNum);
    outShapeRange->GetMax()->SetDimNum(fmDimNum);
    for (size_t i = 0; i < fmDimNum; ++i) {
        outShapeRange->GetMin()->SetDim(i, outRange[i].first);
        outShapeRange->GetMax()->SetDim(i, outRange[i].second);
    }
    OP_LOGD(context->GetNodeName(), "Conv3D out range min: %s, max: %s",
            Ops::Base::ToString(*outShapeRange->GetMin()).c_str(),
            Ops::Base::ToString(*outShapeRange->GetMax()).c_str());
    return ge::GRAPH_SUCCESS;
}

// Range 入口校验：x format（仅 NCDHW/NDHWC，对齐 InferShape 的 GetConv3DXShape；Range 上下文
// 类型不同无法复用）与 x/filter dtype 一致性
static ge::graphStatus CheckConv3DRangeEntry(const gert::InferShapeRangeContext* context, ge::Format& xFormat)
{
    const gert::CompileTimeTensorDesc* xDesc = context->GetInputDesc(CONV3D_X_IDX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    xFormat = xDesc->GetOriginFormat();
    OP_LOGE_IF(xFormat == ge::Format::FORMAT_RESERVED, ge::GRAPH_FAILED, context->GetNodeName(),
               "Get x format failed: %d.", xFormat);
    if (xFormat != ge::Format::FORMAT_NCDHW && xFormat != ge::Format::FORMAT_NDHWC) {
        std::string correctFormat = Ops::Base::ToString(ge::Format::FORMAT_NCDHW) + ", " +
                                    Ops::Base::ToString(ge::Format::FORMAT_NDHWC);
        OP_LOGE_FOR_INVALID_FORMAT(context->GetNodeName(), "x",
                                   ge::TypeUtils::FormatToAscendString(xFormat).GetString(), correctFormat.c_str());
        return ge::GRAPH_FAILED;
    }
    return CheckConv3DDtypeConsistency(context, xDesc) ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED;
}

// Range 指针判空（含输出 range 内部 min/max，与输入侧对称）与输入维数校验
static ge::graphStatus CheckConv3DRangeShape(const gert::InferShapeRangeContext* context,
                                             const gert::Range<gert::Shape>* fmShapeRange,
                                             const gert::Range<gert::Shape>* wShapeRange,
                                             gert::Range<gert::Shape>* outShapeRange, size_t& fmDimNum, size_t& wDimNum)
{
    OPS_CHECK_NULL_WITH_CONTEXT(context, fmShapeRange);
    OPS_CHECK_NULL_WITH_CONTEXT(context, wShapeRange);
    OPS_CHECK_NULL_WITH_CONTEXT(context, outShapeRange);
    OPS_CHECK_NULL_WITH_CONTEXT(context, fmShapeRange->GetMin());
    OPS_CHECK_NULL_WITH_CONTEXT(context, fmShapeRange->GetMax());
    OPS_CHECK_NULL_WITH_CONTEXT(context, wShapeRange->GetMin());
    OPS_CHECK_NULL_WITH_CONTEXT(context, wShapeRange->GetMax());
    OPS_CHECK_NULL_WITH_CONTEXT(context, outShapeRange->GetMin());
    OPS_CHECK_NULL_WITH_CONTEXT(context, outShapeRange->GetMax());

    fmDimNum = fmShapeRange->GetMax()->GetDimNum();
    wDimNum = wShapeRange->GetMax()->GetDimNum();
    return CheckConv3DInputRangeDim(context, fmDimNum, wDimNum) ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED;
}

// 收集 x / filter 的 shape range
static void CollectConv3DShapeRange(const gert::Range<gert::Shape>* fmShapeRange,
                                    const gert::Range<gert::Shape>* wShapeRange, size_t fmDimNum, size_t wDimNum,
                                    std::vector<std::pair<int64_t, int64_t>>& fmRange,
                                    std::vector<std::pair<int64_t, int64_t>>& wRange)
{
    fmRange.resize(fmDimNum);
    for (size_t i = 0; i < fmDimNum; ++i) {
        fmRange[i].first = fmShapeRange->GetMin()->GetDim(i);
        fmRange[i].second = fmShapeRange->GetMax()->GetDim(i);
    }
    wRange.resize(wDimNum);
    for (size_t i = 0; i < wDimNum; ++i) {
        wRange[i].first = wShapeRange->GetMin()->GetDim(i);
        wRange[i].second = wShapeRange->GetMax()->GetDim(i);
    }
}

// 新增RT1.0功能：动态shape场景根据x_shape_range推导output的shape range
static ge::graphStatus InferShapeRangeForConv3D(gert::InferShapeRangeContext* context)
{
    OP_CHECK(context == nullptr, CUBE_INNER_ERR_REPORT("Conv3D", "context is null."), return ge::GRAPH_FAILED);
    const auto opName = context->GetNodeName();
    OP_LOGD(opName, "Enter Conv3D InferShapeRange.");

    ge::Format xFormat = ge::Format::FORMAT_RESERVED;
    if (CheckConv3DRangeEntry(context, xFormat) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    auto fmShapeRange = context->GetInputShapeRange(CONV3D_X_IDX);
    auto wShapeRange = context->GetInputShapeRange(CONV3D_FILTER_IDX);
    auto outShapeRange = context->GetOutputShapeRange(CONV3D_Y_IDX);
    size_t fmDimNum = 0;
    size_t wDimNum = 0;
    if (CheckConv3DRangeShape(context, fmShapeRange, wShapeRange, outShapeRange, fmDimNum, wDimNum) !=
        ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // 解析属性（与 InferShape 共用同一模板解析与校验，属性在两个 context 中只读一致），
    // padding=VALID 时按 pads=0 计算，忽略显式 pads
    Conv3DAttrs attrs;
    if (ParseConv3DRangeAttrs(context, xFormat, attrs) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // 收集 x / filter 的 shape range
    std::vector<std::pair<int64_t, int64_t>> fmRange;
    std::vector<std::pair<int64_t, int64_t>> wRange;
    CollectConv3DShapeRange(fmShapeRange, wShapeRange, fmDimNum, wDimNum, fmRange, wRange);

    // x / filter 各维（N C D H W）下标映射
    std::vector<size_t> xIdx;
    std::vector<size_t> wIdx;
    if (GetConv3DRangeDimIdx(context, xFormat, xIdx, wIdx) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // 推导输出各维 range（N/C 维继承 + D/H/W 维公式与上下界收敛，含 int64 溢出拦截）
    std::vector<std::pair<int64_t, int64_t>> outRange(fmDimNum, std::make_pair(UNKNOWN_DIM_VALUE_, UNKNOWN_DIM_VALUE_));
    if (CalcConv3DOutDimsRange(context, attrs, fmShapeRange, wShapeRange, fmRange, wRange, xIdx, wIdx, outRange) !=
        ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return SetConv3DOutputRange(context, outShapeRange, outRange, fmDimNum);
}
} // namespace

IMPL_OP_INFERSHAPE(Conv3D)
    .InferShape(InferShapeForConv3D)
    .InferShapeRange(InferShapeRangeForConv3D)
    .PrivateAttr("padding", "")
    .PrivateAttr("_op_impl_mode_enum", static_cast<int64_t>(0));
} // namespace ops

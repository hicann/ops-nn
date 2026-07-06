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
 * \file conv_backprop_filter_context_utils.cc
 * \brief
 */

#include <log/log.h>
#include <util/math_util.h>
#include "conv/common/op_host/op_tiling/conv_math_util.h"
#include "conv/common/op_host/op_tiling/conv_platform_util.h"
#include "conv_backprop_filter_context_utils.h"
#include <cstdarg>
#include "securec.h"
#include "error_util.h"
#include "conv/common/op_host/op_tiling/arch35/conv_base_numblocks_decision.h"

namespace {
constexpr size_t NCDHW_N_INDEX = 0;
constexpr size_t NCDHW_C_INDEX = 1;
constexpr size_t NCDHW_D_INDEX = 2;
constexpr size_t NCDHW_H_INDEX = 3;
constexpr size_t NCDHW_W_INDEX = 4;
constexpr size_t CONV_BACKPROP_SHAPE_DIM = 5;
constexpr size_t CONV_BACKPROP_PADS_DIM = 6;

// INDEX
constexpr size_t INPUT_DESC = 0;
constexpr size_t OUT_BACKPROP_DESC = 2;
constexpr size_t OUTPUT_DESC = 0;

constexpr size_t STRIDES_INDEX = 0;
constexpr size_t PADS_INDEX = 1;
constexpr size_t DIALTIONS_INDEX = 2;
constexpr size_t GROUPS_INDEX = 3;
constexpr size_t ENABLE_HF32_INDEX = 5;
constexpr size_t PADDING_INDEX = 6;

constexpr size_t DEFAULT_C0 = 16;
constexpr size_t DEFAULT_FP32_C0 = 8;
constexpr size_t BLOCK_CUBE = 16;
constexpr int32_t DILATION_LOWWER = 1;
constexpr int32_t DILATION_UPPER = 255;
constexpr int32_t STRIDE_LOWER = 1;
constexpr int32_t STRIDE_UPPER = INT32_MAX - 1;
constexpr int32_t PAD_LOWWER = 0;
int32_t PAD_UPPER = 255;
constexpr int32_t SHAPE_LOWER = 1;
constexpr int32_t SHAPE_UPPER = INT32_MAX - 1;
} // namespace

using namespace optiling;
using namespace optiling::conv_ops_tiling;
namespace Ops {
namespace NN {
namespace Conv {
template <typename T>
static bool GetNCDHWShape(const T& origin_shape, int64_t* ncdhw_shape, ge::Format origin_format)
{
    size_t idx = 0;
    if (origin_format == ge::FORMAT_NDHWC) {
        ncdhw_shape[idx++] = origin_shape[0]; // 0: N
        ncdhw_shape[idx++] = origin_shape[4]; // 4: C
        ncdhw_shape[idx++] = origin_shape[1]; // 1: D
        ncdhw_shape[idx++] = origin_shape[2]; // 2: H
        ncdhw_shape[idx++] = origin_shape[3]; // 3: W
        return true;
    } else if (origin_format == ge::FORMAT_NCDHW) {
        ncdhw_shape[idx++] = origin_shape[0]; // 0: N
        ncdhw_shape[idx++] = origin_shape[1]; // 1: C
        ncdhw_shape[idx++] = origin_shape[2]; // 2: D
        ncdhw_shape[idx++] = origin_shape[3]; // 3: H
        ncdhw_shape[idx++] = origin_shape[4]; // 4: W
        return true;
    } else if (origin_format == ge::FORMAT_DHWCN) {
        ncdhw_shape[idx++] = origin_shape[4]; // 4: N
        ncdhw_shape[idx++] = origin_shape[3]; // 3: C
        ncdhw_shape[idx++] = origin_shape[0]; // 0: D
        ncdhw_shape[idx++] = origin_shape[1]; // 1: H
        ncdhw_shape[idx++] = origin_shape[2]; // 2: W
        return true;
    } else {
        OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON("Conv3DBackpropFilterV2", "origin_format",
                                                ge::TypeUtils::FormatToSerialString(origin_format).c_str(),
                                                "{NCDHW, NDHWC, DHWCN}");
        return false;
    }
}

template <typename T>
static inline bool CheckRange(T val, T lower, T upper)
{
    return val >= lower && val <= upper;
}

namespace {
static bool IsArchAfter35(const gert::TilingContext* context)
{
    return context->GetCompileInfo<Ops::NN::Conv::Conv3DBackpropV2CompileInfo>()->npuArch == NpuArch::DAV_3510;
}

bool SetStridesAttr(const gert::TilingContext* context, Conv3dBpFilterV2RunInfo& runInfoV2)
{
    const auto op_name = (context->GetNodeName() == nullptr) ? "nil" : context->GetNodeName();
    const auto attrs = context->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE_WITH_INVALID_ATTR(op_name, "attrs", "null", "non_empty_value"), return false);

    const auto strides = attrs->GetAttrPointer<gert::ContinuousVector>(STRIDES_INDEX);
    OP_CHECK_IF(strides == nullptr, OP_LOGE_WITH_INVALID_ATTR(op_name, "strides", "null", "non_empty_value"),
                return false);
    OP_CHECK_IF(strides->GetSize() != CONV_BACKPROP_SHAPE_DIM,
                OP_LOGE_WITH_INVALID_ATTR_SIZE(op_name, "strides length", std::to_string(strides->GetSize()),
                                               std::to_string(CONV_BACKPROP_SHAPE_DIM)),
                return false);
    const int64_t* strides_data = static_cast<const int64_t*>(strides->GetData());
    std::vector<int64_t> normalized_strides(strides->GetSize(), 0);
    const ge::Format input_format = context->GetInputDesc(INPUT_DESC)->GetOriginFormat();
    OP_CHECK_IF(!GetNCDHWShape(strides_data, normalized_strides.data(), input_format),
                OP_LOGW(op_name, "GetNCDHWShape failed."), return false);

    // check param limit
    OP_CHECK_IF(normalized_strides[NCDHW_N_INDEX] != 1 || normalized_strides[NCDHW_C_INDEX] != 1,
                OP_LOGE_WITH_INVALID_ATTR_SIZE(op_name, "stride_n or stride_c",
                                               std::to_string(normalized_strides[NCDHW_N_INDEX]) + " or " +
                                                   std::to_string(normalized_strides[NCDHW_C_INDEX]),
                                               "1"),
                return false);
    OP_CHECK_IF(!CheckRange<int64_t>(normalized_strides[NCDHW_D_INDEX], static_cast<int64_t>(STRIDE_LOWER),
                                     static_cast<int64_t>(SHAPE_UPPER)),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    op_name, "stride_d", std::to_string(normalized_strides[NCDHW_D_INDEX]),
                    FormatString("The value of stride_d must be [%d, %d]", STRIDE_LOWER, SHAPE_UPPER).c_str()),
                return false);
    OP_CHECK_IF(!CheckRange<int64_t>(normalized_strides[NCDHW_H_INDEX], static_cast<int64_t>(STRIDE_LOWER),
                                     static_cast<int64_t>(STRIDE_UPPER)),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    op_name, "stride_h", std::to_string(normalized_strides[NCDHW_H_INDEX]),
                    FormatString("The value of stride_h must be [%d, %d]", STRIDE_LOWER, STRIDE_UPPER).c_str()),
                return false);
    OP_CHECK_IF(!CheckRange<int64_t>(normalized_strides[NCDHW_W_INDEX], static_cast<int64_t>(STRIDE_LOWER),
                                     static_cast<int64_t>(STRIDE_UPPER)),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    op_name, "stride_w", std::to_string(normalized_strides[NCDHW_W_INDEX]),
                    FormatString("The value of stride_w must be [%d, %d]", STRIDE_LOWER, STRIDE_UPPER).c_str()),
                return false);

    // stride attrs
    runInfoV2.stride_d = normalized_strides[NCDHW_D_INDEX];
    runInfoV2.stride_h = normalized_strides[NCDHW_H_INDEX];
    runInfoV2.stride_w = normalized_strides[NCDHW_W_INDEX];
    return true;
}

static bool ValidateDilationsBasicInfo(const char* op_name, const gert::TilingContext* context,
                                       const gert::ContinuousVector*& dilations)
{
    const auto attrs = context->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE_WITH_INVALID_ATTR(op_name, "attrs", "null", "non_empty_value"), return false);
    dilations = attrs->GetAttrPointer<gert::ContinuousVector>(DIALTIONS_INDEX);
    OP_CHECK_IF(dilations == nullptr, OP_LOGE_WITH_INVALID_ATTR(op_name, "dilations", "null", "non_empty_value"),
                return false);
    OP_CHECK_IF(dilations->GetSize() != CONV_BACKPROP_SHAPE_DIM,
                OP_LOGE_WITH_INVALID_ATTR_SIZE(op_name, "dilations length", std::to_string(dilations->GetSize()),
                                               std::to_string(CONV_BACKPROP_SHAPE_DIM)),
                return false);
    return true;
}

static bool CheckDilationsNC(const char* op_name, const std::vector<int64_t>& normalized_dilations)
{
    OP_CHECK_IF(normalized_dilations[NCDHW_N_INDEX] != 1 || normalized_dilations[NCDHW_C_INDEX] != 1,
                OP_LOGE_WITH_INVALID_ATTR_SIZE(op_name, "dilation_n or dilation_c",
                                               std::to_string(normalized_dilations[NCDHW_N_INDEX]) + " or " +
                                                   std::to_string(normalized_dilations[NCDHW_C_INDEX]),
                                               "1"),
                return false);
    return true;
}

static bool CheckDilationsRange(const char* op_name, const std::vector<int64_t>& normalized_dilations,
                                int32_t dilation_limit)
{
    OP_CHECK_IF(!CheckRange<int64_t>(normalized_dilations[NCDHW_D_INDEX], static_cast<int64_t>(DILATION_LOWWER),
                                     static_cast<int64_t>(SHAPE_UPPER)),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    op_name, "dilation_d", std::to_string(normalized_dilations[NCDHW_D_INDEX]),
                    FormatString("The value of dilation_d must be [%d, %d]", DILATION_LOWWER, SHAPE_UPPER).c_str()),
                return false);

    OP_CHECK_IF(!CheckRange<int64_t>(normalized_dilations[NCDHW_H_INDEX], static_cast<int64_t>(DILATION_LOWWER),
                                     static_cast<int64_t>(dilation_limit)),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    op_name, "dilation_h", std::to_string(normalized_dilations[NCDHW_H_INDEX]),
                    FormatString("The value of dilation_h must be [%d, %d]", DILATION_LOWWER, dilation_limit).c_str()),
                return false);
    OP_CHECK_IF(!CheckRange<int64_t>(normalized_dilations[NCDHW_W_INDEX], static_cast<int64_t>(DILATION_LOWWER),
                                     static_cast<int64_t>(dilation_limit)),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    op_name, "dilation_w", std::to_string(normalized_dilations[NCDHW_W_INDEX]),
                    FormatString("The value of dilation_w must be [%d, %d]", DILATION_LOWWER, dilation_limit).c_str()),
                return false);
    return true;
}

static void AdjustDilationByKernel(Conv3dBpFilterV2RunInfo& runInfoV2)
{
    if (runInfoV2.kd == 1) {
        runInfoV2.dilation_d = 1;
    }
    if (runInfoV2.kh == 1) {
        runInfoV2.dilation_h = 1;
    }
    if (runInfoV2.kw == 1) {
        runInfoV2.dilation_w = 1;
    }
}

bool SetDilationsAttr(const gert::TilingContext* context, Conv3dBpFilterV2RunInfo& runInfoV2)
{
    const auto op_name = (context->GetNodeName() == nullptr) ? "nil" : context->GetNodeName();
    const gert::ContinuousVector* dilations = nullptr;
    if (!ValidateDilationsBasicInfo(op_name, context, dilations)) {
        return false;
    }

    const int64_t* dilations_data = static_cast<const int64_t*>(dilations->GetData());
    std::vector<int64_t> normalized_dilations(dilations->GetSize(), 0);
    const ge::Format input_format = context->GetInputDesc(INPUT_DESC)->GetOriginFormat();
    OP_CHECK_IF(!GetNCDHWShape(dilations_data, normalized_dilations.data(), input_format),
                OP_LOGW(op_name, "GetNCDHWShape failed."), return false);

    if (!CheckDilationsNC(op_name, normalized_dilations)) {
        return false;
    }

    int32_t dilation_limit = IsArchAfter35(context) ? SHAPE_UPPER : DILATION_UPPER;
    if (!CheckDilationsRange(op_name, normalized_dilations, dilation_limit)) {
        return false;
    }

    runInfoV2.dilation_d = normalized_dilations[NCDHW_D_INDEX];
    runInfoV2.dilation_h = normalized_dilations[NCDHW_H_INDEX];
    runInfoV2.dilation_w = normalized_dilations[NCDHW_W_INDEX];
    AdjustDilationByKernel(runInfoV2);
    return true;
}

void SetHf32Flag(const char* op_name, const gert::RuntimeAttrs* attrs, Conv3dBpFilterV2RunInfo& runInfoV2)
{
    runInfoV2.hf32Flag = 0;
    if (runInfoV2.a_dtype == ge::DT_FLOAT && (attrs->GetAttrNum() > ENABLE_HF32_INDEX)) {
        auto enableHf32Ptr = attrs->GetBool(ENABLE_HF32_INDEX);
        if (enableHf32Ptr == nullptr) {
            OP_LOGE_WITH_INVALID_ATTR(op_name, "enable_hf32", "null", "non_empty_value");
            return;
        }
        bool enableHf32 = *enableHf32Ptr;
        OP_LOGD(op_name, "attr idx[%zu] enable_hf32 = %d", ENABLE_HF32_INDEX, enableHf32);
        runInfoV2.hf32Flag = static_cast<uint32_t>(enableHf32 ? 1 : 0);
    }
}

static bool ValidateGroupsAttr(const char* op_name, const gert::RuntimeAttrs* attrs, int32_t& groups)
{
    OP_CHECK_IF(attrs == nullptr, OP_LOGE_WITH_INVALID_ATTR(op_name, "attrs", "null", "non_empty_value"), return false);
    const auto groups_attr = attrs->GetAttrPointer<int64_t>(GROUPS_INDEX);
    OP_CHECK_IF(groups_attr == nullptr, OP_LOGE_WITH_INVALID_ATTR(op_name, "groups", "null", "non_empty_value"),
                return false);
    groups = static_cast<int32_t>(*groups_attr);
    OP_CHECK_IF(groups <= 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(op_name, "groups", std::to_string(groups),
                                                      "The value of groups must be greater than 0"),
                return false);
    return true;
}

static bool ValidateGroupsValue(const char* op_name, int32_t& groups, int32_t actual_groups,
                                const Conv3dBpFilterV2RunInfo& runInfoV2, int32_t filter_shape_c)
{
    if (groups == 1) {
        groups = actual_groups;
        OP_LOGD(op_name, "set groups=%d, x_channel(%d) / filter_channel(%d)", groups, runInfoV2.ci, filter_shape_c);
    } else if (actual_groups != static_cast<int64_t>(groups)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            op_name, "groups", std::to_string(groups),
            FormatString("x_channel(%d) / filter_channel(%d) != groups(%d)", runInfoV2.ci, filter_shape_c, groups)
                .c_str());
        return false;
    }
    if ((runInfoV2.co % groups) != 0) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            op_name, "groups", std::to_string(groups),
            FormatString("out_channels(%d) %% groups(%d) != 0", runInfoV2.co, groups).c_str());
        return false;
    }
    if (groups < 1 || groups > UINT16_MAX) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            op_name, "groups", std::to_string(groups),
            FormatString("The value of groups must be in range: [1, %d]", UINT16_MAX).c_str());
        return false;
    }
    return true;
}

static bool CalculateMagFactor(const char* op_name, int32_t groups, Conv3dBpFilterV2RunInfo& runInfoV2)
{
    int32_t mag_factor = 1;
    if (groups == 1) {
        runInfoV2.real_g = 1;
    } else {
        int64_t mag_factor0 = MathUtil::Lcm(runInfoV2.ci / groups, runInfoV2.k0) / (runInfoV2.ci / groups);
        int64_t mag_factor1 = MathUtil::Lcm(runInfoV2.co / groups, BLOCK_CUBE) / (runInfoV2.co / groups);
        mag_factor = std::min(MathUtil::Lcm(mag_factor0, mag_factor1), static_cast<int64_t>(groups));
        if (mag_factor <= 0) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(op_name, "mag_factor", std::to_string(mag_factor),
                                                  "The value of mag_factor must be greater than 0");
            return false;
        }
        runInfoV2.real_g = (groups + mag_factor - 1) / mag_factor;
    }
    runInfoV2.mag_factor = mag_factor;
    return true;
}

static void SetCin1Cout1G(Conv3dBpFilterV2RunInfo& runInfoV2, int32_t groups, int32_t mag_factor)
{
    if (runInfoV2.a_dtype == ge::DT_FLOAT) {
        runInfoV2.cin1_g = Ops::Base::CeilDiv(mag_factor * runInfoV2.ci / groups, static_cast<int32_t>(runInfoV2.k0));
        runInfoV2.cout1_g = Ops::Base::CeilAlign(mag_factor * runInfoV2.co / groups, static_cast<int32_t>(BLOCK_CUBE)) /
                            runInfoV2.k0;
    } else {
        runInfoV2.cin1_g = (mag_factor * runInfoV2.ci / groups + runInfoV2.k0 - 1) / runInfoV2.k0;
        runInfoV2.cout1_g = (mag_factor * runInfoV2.co / groups + runInfoV2.k0 - 1) / runInfoV2.k0;
    }
}

bool SetGroupsAttrs(const gert::TilingContext* context, Conv3dBpFilterV2RunInfo& runInfoV2, int32_t filter_shape_c)
{
    const auto op_name = (context->GetNodeName() == nullptr) ? "nil" : context->GetNodeName();
    int32_t groups = 0;
    if (!ValidateGroupsAttr(op_name, context->GetAttrs(), groups)) {
        return false;
    }

    int32_t actual_groups = static_cast<int32_t>(runInfoV2.ci) / filter_shape_c;
    if (!ValidateGroupsValue(op_name, groups, actual_groups, runInfoV2, filter_shape_c)) {
        return false;
    }

    runInfoV2.groups = groups;
    SetHf32Flag(op_name, context->GetAttrs(), runInfoV2);

    OP_CHECK_IF(IsArchAfter35(context), OP_LOGD(op_name, "DAV_3510 does not need to calculate factor in utils."),
                return true);

    if (!CalculateMagFactor(op_name, groups, runInfoV2)) {
        return false;
    }
    SetCin1Cout1G(runInfoV2, groups, runInfoV2.mag_factor);
    return true;
}

bool ValidateInputShapeRange(const char* op_name, const std::vector<int64_t>& shape)
{
    OP_CHECK_IF(!CheckRange<int64_t>(shape[NCDHW_N_INDEX], static_cast<int64_t>(SHAPE_LOWER),
                                     static_cast<int64_t>(SHAPE_UPPER)),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    op_name, "x_shape_batch", std::to_string(shape[NCDHW_N_INDEX]),
                    FormatString("x_shape_batch must be within the range [%d, %d]", SHAPE_LOWER, SHAPE_UPPER).c_str()),
                return false);
    OP_CHECK_IF(!CheckRange<int64_t>(shape[NCDHW_C_INDEX], static_cast<int64_t>(SHAPE_LOWER),
                                     static_cast<int64_t>(SHAPE_UPPER)),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    op_name, "x_shape_c", std::to_string(shape[NCDHW_C_INDEX]),
                    FormatString("x_shape_c must be within the range [%d, %d]", SHAPE_LOWER, SHAPE_UPPER).c_str()),
                return false);
    OP_CHECK_IF(!CheckRange<int64_t>(shape[NCDHW_D_INDEX], static_cast<int64_t>(SHAPE_LOWER),
                                     static_cast<int64_t>(SHAPE_UPPER)),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    op_name, "x_shape_d", std::to_string(shape[NCDHW_D_INDEX]),
                    FormatString("x_shape_d must be within the range [%d, %d]", SHAPE_LOWER, SHAPE_UPPER).c_str()),
                return false);
    OP_CHECK_IF(!CheckRange<int64_t>(shape[NCDHW_H_INDEX], static_cast<int64_t>(SHAPE_LOWER),
                                     static_cast<int64_t>(SHAPE_UPPER)),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    op_name, "x_shape_h", std::to_string(shape[NCDHW_H_INDEX]),
                    FormatString("x_shape_h must be within the range [%d, %d]", SHAPE_LOWER, SHAPE_UPPER).c_str()),
                return false);
    OP_CHECK_IF(!CheckRange<int64_t>(shape[NCDHW_W_INDEX], static_cast<int64_t>(SHAPE_LOWER),
                                     static_cast<int64_t>(SHAPE_UPPER)),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    op_name, "x_shape_w", std::to_string(shape[NCDHW_W_INDEX]),
                    FormatString("x_shape_w must be within the range [%d, %d]", SHAPE_LOWER, SHAPE_UPPER).c_str()),
                return false);
    return true;
}

bool SetFmapDesc(const gert::TilingContext* context, Conv3dBpFilterV2RunInfo& runInfoV2)
{
    const auto op_name = (context->GetNodeName() == nullptr) ? "nil" : context->GetNodeName();
    const gert::CompileTimeTensorDesc* inputDesc = context->GetInputDesc(INPUT_DESC);
    if (inputDesc == nullptr) {
        CUBE_INNER_ERR_REPORT(op_name, "failed to get x desc.");
        return false;
    }
    const ge::Format input_format = inputDesc->GetOriginFormat();
    const gert::StorageShape* inputShape = context->GetInputShape(INPUT_DESC);
    if (inputShape == nullptr) {
        CUBE_INNER_ERR_REPORT(op_name, "failed to get x shape.");
        return false;
    }
    OP_CHECK_IF(CONV_BACKPROP_SHAPE_DIM != inputShape->GetOriginShape().GetDimNum(),
                OP_LOGE_FOR_INVALID_SHAPEDIM(op_name, "x origin shape dims",
                                             std::to_string(inputShape->GetOriginShape().GetDimNum()),
                                             FormatString("%ld", CONV_BACKPROP_SHAPE_DIM).c_str()),
                return false);
    std::vector<int64_t> normalized_input_shape(CONV_BACKPROP_SHAPE_DIM, 0);
    OP_CHECK_IF(!GetNCDHWShape(inputShape->GetOriginShape(), normalized_input_shape.data(), input_format),
                OP_LOGW(op_name, "GetNCDHWShape failed."), return false);

    if (!ValidateInputShapeRange(op_name, normalized_input_shape)) {
        return false;
    }

    runInfoV2.batch = normalized_input_shape[NCDHW_N_INDEX];
    runInfoV2.ci = normalized_input_shape[NCDHW_C_INDEX];
    runInfoV2.di = normalized_input_shape[NCDHW_D_INDEX];
    runInfoV2.hi = normalized_input_shape[NCDHW_H_INDEX];
    runInfoV2.wi = normalized_input_shape[NCDHW_W_INDEX];

    runInfoV2.a_dtype = inputDesc->GetDataType();
    int32_t a_dtype_bytes = ge::GetSizeByDataType(runInfoV2.a_dtype);
    if (a_dtype_bytes == -1) {
        CUBE_INNER_ERR_REPORT(op_name, "failed to get x dtype size");
        return false;
    }
    runInfoV2.a_dtype_bytes = a_dtype_bytes;

    // Tiling parameters
    if (runInfoV2.a_dtype == ge::DT_FLOAT) {
        runInfoV2.k0 = DEFAULT_FP32_C0;
    } else {
        runInfoV2.k0 = DEFAULT_C0;
    }
    runInfoV2.ci1 = Ops::Base::CeilDiv(runInfoV2.ci, static_cast<int32_t>(runInfoV2.k0));
    runInfoV2.m0 = BLOCK_CUBE;
    runInfoV2.n0 = BLOCK_CUBE;
    return true;
}

static bool ValidateGradOutputDesc(const char* op_name, const gert::TilingContext* context,
                                   const gert::CompileTimeTensorDesc*& gradOutputDesc,
                                   const gert::StorageShape*& gradOutputShape)
{
    gradOutputDesc = context->GetInputDesc(OUT_BACKPROP_DESC);
    if (gradOutputDesc == nullptr) {
        CUBE_INNER_ERR_REPORT(op_name, "failed to get out_backprop desc.");
        return false;
    }
    gradOutputShape = context->GetInputShape(OUT_BACKPROP_DESC);
    if (gradOutputShape == nullptr) {
        CUBE_INNER_ERR_REPORT(op_name, "failed to get out_backprop shape.");
        return false;
    }
    if (CONV_BACKPROP_SHAPE_DIM != gradOutputShape->GetOriginShape().GetDimNum()) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(op_name, "out_backprop origin shape dims",
                                     std::to_string(gradOutputShape->GetOriginShape().GetDimNum()),
                                     FormatString("%ld", CONV_BACKPROP_SHAPE_DIM).c_str());
        return false;
    }
    return true;
}

static bool CheckGradOutputShapeRange(const char* op_name, const std::vector<int64_t>& normalized_dedy_shape)
{
    OP_CHECK_IF(
        !CheckRange<int64_t>(normalized_dedy_shape[NCDHW_N_INDEX], static_cast<int64_t>(SHAPE_LOWER),
                             static_cast<int64_t>(SHAPE_UPPER)),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            op_name, "out_backprop_batch", std::to_string(normalized_dedy_shape[NCDHW_N_INDEX]),
            FormatString("out_backprop_batch must be within the range [%d, %d]", SHAPE_LOWER, SHAPE_UPPER).c_str()),
        return false);
    OP_CHECK_IF(!CheckRange<int64_t>(normalized_dedy_shape[NCDHW_C_INDEX], static_cast<int64_t>(SHAPE_LOWER),
                                     static_cast<int64_t>(SHAPE_UPPER)),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    op_name, "out_backprop_c", std::to_string(normalized_dedy_shape[NCDHW_C_INDEX]),
                    FormatString("out_backprop_c must be within the range [%d, %d]", SHAPE_LOWER, SHAPE_UPPER).c_str()),
                return false);
    OP_CHECK_IF(!CheckRange<int64_t>(normalized_dedy_shape[NCDHW_D_INDEX], static_cast<int64_t>(SHAPE_LOWER),
                                     static_cast<int64_t>(SHAPE_UPPER)),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    op_name, "out_backprop_d", std::to_string(normalized_dedy_shape[NCDHW_D_INDEX]),
                    FormatString("out_backprop_d must be within the range [%d, %d]", SHAPE_LOWER, SHAPE_UPPER).c_str()),
                return false);
    OP_CHECK_IF(!CheckRange<int64_t>(normalized_dedy_shape[NCDHW_H_INDEX], static_cast<int64_t>(SHAPE_LOWER),
                                     static_cast<int64_t>(SHAPE_UPPER)),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    op_name, "out_backprop_h", std::to_string(normalized_dedy_shape[NCDHW_H_INDEX]),
                    FormatString("out_backprop_h must be within the range [%d, %d]", SHAPE_LOWER, SHAPE_UPPER).c_str()),
                return false);
    OP_CHECK_IF(!CheckRange<int64_t>(normalized_dedy_shape[NCDHW_W_INDEX], static_cast<int64_t>(SHAPE_LOWER),
                                     static_cast<int64_t>(SHAPE_UPPER)),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    op_name, "out_backprop_w", std::to_string(normalized_dedy_shape[NCDHW_W_INDEX]),
                    FormatString("out_backprop_w must be within the range [%d, %d]", SHAPE_LOWER, SHAPE_UPPER).c_str()),
                return false);
    return true;
}

static bool SetGradOutputDtypeBytes(const char* op_name, const gert::CompileTimeTensorDesc* gradOutputDesc,
                                    Conv3dBpFilterV2RunInfo& runInfoV2)
{
    runInfoV2.b_dtype = gradOutputDesc->GetDataType();
    int32_t b_dtype_bytes = ge::GetSizeByDataType(runInfoV2.b_dtype);
    if (b_dtype_bytes == -1) {
        CUBE_INNER_ERR_REPORT(op_name, "failed to get filter_shape dtype size");
        return false;
    }
    runInfoV2.b_dtype_bytes = b_dtype_bytes;
    return true;
}

bool SetGradOutputDesc(const gert::TilingContext* context, Conv3dBpFilterV2RunInfo& runInfoV2)
{
    const auto op_name = (context->GetNodeName() == nullptr) ? "nil" : context->GetNodeName();
    const gert::CompileTimeTensorDesc* gradOutputDesc = nullptr;
    const gert::StorageShape* gradOutputShape = nullptr;
    if (!ValidateGradOutputDesc(op_name, context, gradOutputDesc, gradOutputShape)) {
        return false;
    }

    const ge::Format gradOutputFormat = gradOutputDesc->GetOriginFormat();
    std::vector<int64_t> normalized_dedy_shape(CONV_BACKPROP_SHAPE_DIM, 0);
    OP_CHECK_IF(!GetNCDHWShape(gradOutputShape->GetOriginShape(), normalized_dedy_shape.data(), gradOutputFormat),
                OP_LOGW(op_name, "GetNCDHWShape failed."), return false);

    if (!CheckGradOutputShapeRange(op_name, normalized_dedy_shape)) {
        return false;
    }

    runInfoV2.batch = normalized_dedy_shape[NCDHW_N_INDEX];
    runInfoV2.co = normalized_dedy_shape[NCDHW_C_INDEX];
    runInfoV2.dout = normalized_dedy_shape[NCDHW_D_INDEX];
    runInfoV2.ho = normalized_dedy_shape[NCDHW_H_INDEX];
    runInfoV2.wo = normalized_dedy_shape[NCDHW_W_INDEX];

    return SetGradOutputDtypeBytes(op_name, gradOutputDesc, runInfoV2);
}

static bool ValidateOutputDescBasic(const char* op_name, const gert::TilingContext* context,
                                    const gert::CompileTimeTensorDesc*& outputDesc,
                                    const gert::StorageShape*& outputShape, ge::Format& output_format)
{
    outputDesc = context->GetOutputDesc(OUTPUT_DESC);
    if (outputDesc == nullptr) {
        CUBE_INNER_ERR_REPORT(op_name, "failed to get output desc.");
        return false;
    }
    output_format = outputDesc->GetOriginFormat();
    outputShape = context->GetOutputShape(OUTPUT_DESC);
    if (outputShape == nullptr) {
        CUBE_INNER_ERR_REPORT(op_name, "failed to get output shape.");
        return false;
    }
    if (CONV_BACKPROP_SHAPE_DIM != outputShape->GetOriginShape().GetDimNum()) {
        CUBE_INNER_ERR_REPORT(op_name, "output origin shape dims is invalid.");
        return false;
    }
    return true;
}

static bool ValidateFilterShapeC(const char* op_name, const Conv3dBpFilterV2RunInfo& runInfoV2,
                                 const std::vector<int64_t>& normalized_output_shape, int32_t& filter_shape_ci)
{
    OP_CHECK_IF(!CheckRange<int64_t>(normalized_output_shape[NCDHW_C_INDEX], static_cast<int64_t>(SHAPE_LOWER),
                                     static_cast<int64_t>(SHAPE_UPPER)),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    op_name, "filter_shape_c", std::to_string(normalized_output_shape[NCDHW_C_INDEX]),
                    FormatString("filter_shape_c must be within the range [%d, %d]", SHAPE_LOWER, SHAPE_UPPER).c_str()),
                return false);
    filter_shape_ci = normalized_output_shape[NCDHW_C_INDEX];
    OP_CHECK_IF(filter_shape_ci <= 0,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(op_name, "filter_shape_c", std::to_string(filter_shape_ci),
                                                      "filter_shape_c must be a positive number"),
                return false);
    OP_CHECK_IF(runInfoV2.ci % filter_shape_ci != 0,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    op_name, "x_channel and filter_shape_c",
                    (std::to_string(runInfoV2.ci) + " and " + std::to_string(filter_shape_ci)).c_str(),
                    FormatString("x_channel(%d) %% filter_channel(%d) != 0", runInfoV2.ci, filter_shape_ci).c_str()),
                return false);
    return true;
}

static bool CheckOutputShapeDHW(const char* op_name, const std::vector<int64_t>& normalized_output_shape)
{
    OP_CHECK_IF(!CheckRange<int64_t>(normalized_output_shape[NCDHW_D_INDEX], static_cast<int64_t>(SHAPE_LOWER),
                                     static_cast<int64_t>(SHAPE_UPPER)),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    op_name, "output_shape_d", std::to_string(normalized_output_shape[NCDHW_D_INDEX]),
                    FormatString("output_shape_d must be within the range [%d, %d]", SHAPE_LOWER, SHAPE_UPPER).c_str()),
                return false);
    OP_CHECK_IF(!CheckRange<int64_t>(normalized_output_shape[NCDHW_H_INDEX], static_cast<int64_t>(SHAPE_LOWER),
                                     static_cast<int64_t>(SHAPE_UPPER)),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    op_name, "output_shape_h", std::to_string(normalized_output_shape[NCDHW_H_INDEX]),
                    FormatString("output_shape_h must be within the range [%d, %d]", SHAPE_LOWER, SHAPE_UPPER).c_str()),
                return false);
    OP_CHECK_IF(!CheckRange<int64_t>(normalized_output_shape[NCDHW_W_INDEX], static_cast<int64_t>(SHAPE_LOWER),
                                     static_cast<int64_t>(SHAPE_UPPER)),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    op_name, "output_shape_w", std::to_string(normalized_output_shape[NCDHW_W_INDEX]),
                    FormatString("output_shape_w must be within the range [%d, %d]", SHAPE_LOWER, SHAPE_UPPER).c_str()),
                return false);
    return true;
}

static bool SetOutputDtypeBytes(const char* op_name, const gert::CompileTimeTensorDesc* outputDesc,
                                Conv3dBpFilterV2RunInfo& runInfoV2)
{
    runInfoV2.c_dtype = outputDesc->GetDataType();
    int32_t c_dtype_bytes = ge::GetSizeByDataType(runInfoV2.c_dtype);
    if (c_dtype_bytes == -1) {
        CUBE_INNER_ERR_REPORT(op_name, "failed to get output_shape dtype size");
        return false;
    }
    runInfoV2.c_dtype_bytes = c_dtype_bytes;
    return true;
}

bool SetOutputDesc(const gert::TilingContext* context, Conv3dBpFilterV2RunInfo& runInfoV2)
{
    const auto op_name = (context->GetNodeName() == nullptr) ? "nil" : context->GetNodeName();
    const gert::CompileTimeTensorDesc* outputDesc = nullptr;
    const gert::StorageShape* outputShape = nullptr;
    ge::Format output_format;
    if (!ValidateOutputDescBasic(op_name, context, outputDesc, outputShape, output_format)) {
        return false;
    }

    std::vector<int64_t> normalized_output_shape(CONV_BACKPROP_SHAPE_DIM, 0);
    OP_CHECK_IF(!GetNCDHWShape(outputShape->GetOriginShape(), normalized_output_shape.data(), output_format),
                OP_LOGW(op_name, "GetNCDHWShape failed."), return false);

    int32_t filter_shape_ci = 0;
    if (!ValidateFilterShapeC(op_name, runInfoV2, normalized_output_shape, filter_shape_ci)) {
        return false;
    }

    if (!CheckOutputShapeDHW(op_name, normalized_output_shape)) {
        return false;
    }

    runInfoV2.kd = normalized_output_shape[NCDHW_D_INDEX];
    runInfoV2.kh = normalized_output_shape[NCDHW_H_INDEX];
    runInfoV2.kw = normalized_output_shape[NCDHW_W_INDEX];

    if (!SetOutputDtypeBytes(op_name, outputDesc, runInfoV2)) {
        return false;
    }

    return SetGroupsAttrs(context, runInfoV2, filter_shape_ci);
}

void ReCalPaddings(Conv3dBpFilterV2RunInfo& runInfoV2, const char* padding)
{
    int32_t kernel_d_dilation = (runInfoV2.kd - 1) * runInfoV2.dilation_d + 1;
    int32_t kernel_h_dilation = (runInfoV2.kh - 1) * runInfoV2.dilation_h + 1;
    int32_t kernel_w_dilation = (runInfoV2.kw - 1) * runInfoV2.dilation_w + 1;
    // if pads is invalid and padding is SAME, calc pads ,considerate 2d to 3d ,exclude p_f and p_b
    if (padding != nullptr && padding[0] == 'S' && runInfoV2.pad_u == -1 && runInfoV2.pad_d == -1 &&
        runInfoV2.pad_l == -1 && runInfoV2.pad_r == -1) {
        int64_t tails_d = runInfoV2.di % runInfoV2.stride_d;
        int64_t tails_h = runInfoV2.hi % runInfoV2.stride_h;
        int64_t tails_w = runInfoV2.wi % runInfoV2.stride_w;
        int64_t pad_d = std::max((tails_d > 0 ? kernel_d_dilation - tails_d : kernel_d_dilation - runInfoV2.stride_d),
                                 0L);
        int64_t pad_h = std::max((tails_h > 0 ? kernel_h_dilation - tails_h : kernel_h_dilation - runInfoV2.stride_h),
                                 0L);
        int64_t pad_w = std::max((tails_w > 0 ? kernel_w_dilation - tails_w : kernel_w_dilation - runInfoV2.stride_w),
                                 0L);
        runInfoV2.pad_f = pad_d / 2; // 2 means pad_up is half size of pad_d
        runInfoV2.pad_b = pad_d - runInfoV2.pad_f;
        runInfoV2.pad_u = pad_h / 2; // 2 means pad_up is half size of pad_h
        runInfoV2.pad_d = pad_h - runInfoV2.pad_u;
        runInfoV2.pad_l = pad_w / 2; // 2 means pad_up is half size of pad_w
        runInfoV2.pad_r = pad_w - runInfoV2.pad_l;
    }
}

bool CheckGradOutputShape(const gert::TilingContext* context, Conv3dBpFilterV2RunInfo& runInfoV2)
{
    const auto op_name = (context->GetNodeName() == nullptr) ? "nil" : context->GetNodeName();
    int64_t do_expect = (static_cast<int64_t>(runInfoV2.di) + runInfoV2.pad_f + runInfoV2.pad_b -
                         runInfoV2.dilation_d * (runInfoV2.kd - 1) - 1) /
                            runInfoV2.stride_d +
                        1;
    int64_t ho_expect = (static_cast<int64_t>(runInfoV2.hi) + runInfoV2.pad_u + runInfoV2.pad_d -
                         runInfoV2.dilation_h * (runInfoV2.kh - 1) - 1) /
                            runInfoV2.stride_h +
                        1;
    int64_t wo_expect = (static_cast<int64_t>(runInfoV2.wi) + runInfoV2.pad_l + runInfoV2.pad_r -
                         runInfoV2.dilation_w * (runInfoV2.kw - 1) - 1) /
                            runInfoV2.stride_w +
                        1;
    OP_CHECK_IF(do_expect != runInfoV2.dout,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    op_name, "out_backprop_d", std::to_string(runInfoV2.dout),
                    FormatString("The out_backprop's D must be equal to the inferred D[%ld]", do_expect).c_str()),
                return false);
    OP_CHECK_IF(ho_expect != runInfoV2.ho,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    op_name, "out_backprop_h", std::to_string(runInfoV2.ho),
                    FormatString("The out_backprop's H must be equal to the inferred H[%ld]", ho_expect).c_str()),
                return false);
    OP_CHECK_IF(wo_expect != runInfoV2.wo,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    op_name, "out_backprop_w", std::to_string(runInfoV2.wo),
                    FormatString("The out_backprop's W must be equal to the inferred W[%ld]", wo_expect).c_str()),
                return false);
    return true;
}

static bool ValidatePadsAttr(const char* op_name, const gert::RuntimeAttrs* attrs, const gert::ContinuousVector*& pads)
{
    OP_CHECK_IF(attrs == nullptr, OP_LOGE_WITH_INVALID_ATTR(op_name, "attrs", "null", "non_empty_value"), return false);
    pads = attrs->GetAttrPointer<gert::ContinuousVector>(PADS_INDEX);
    OP_CHECK_IF(pads == nullptr, CUBE_INNER_ERR_REPORT(op_name, "failed to get pads from context fail."), return false);
    OP_CHECK_IF(pads->GetSize() != CONV_BACKPROP_PADS_DIM,
                OP_LOGE_WITH_INVALID_ATTR_SIZE(op_name, "pads size", std::to_string(pads->GetSize()),
                                               std::to_string(CONV_BACKPROP_PADS_DIM)),
                return false);
    return true;
}

static void SetPadsData(Conv3dBpFilterV2RunInfo& runInfoV2, const int64_t* pads_data)
{
    size_t pad_idx = 0;
    runInfoV2.pad_f = pads_data[pad_idx++];
    runInfoV2.pad_b = pads_data[pad_idx++];
    runInfoV2.pad_u = pads_data[pad_idx++];
    runInfoV2.pad_d = pads_data[pad_idx++];
    runInfoV2.pad_l = pads_data[pad_idx++];
    runInfoV2.pad_r = pads_data[pad_idx++];
}

static bool CheckPadsRange(const char* op_name, const Conv3dBpFilterV2RunInfo& runInfoV2)
{
    OP_CHECK_IF(!CheckRange(runInfoV2.pad_f, PAD_LOWWER, SHAPE_UPPER),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    op_name, "pad_f", std::to_string(runInfoV2.pad_f),
                    FormatString("The value of pad_f must be [%d, %d]", PAD_LOWWER, SHAPE_UPPER).c_str()),
                return false);
    OP_CHECK_IF(!CheckRange(runInfoV2.pad_b, PAD_LOWWER, SHAPE_UPPER),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    op_name, "pad_b", std::to_string(runInfoV2.pad_b),
                    FormatString("The value of pad_b must be [%d, %d]", PAD_LOWWER, SHAPE_UPPER).c_str()),
                return false);
    OP_CHECK_IF(!CheckRange(runInfoV2.pad_u, PAD_LOWWER, PAD_UPPER),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    op_name, "pad_u", std::to_string(runInfoV2.pad_u),
                    FormatString("The value of pad_u must be [%d, %d]", PAD_LOWWER, PAD_UPPER).c_str()),
                return false);
    OP_CHECK_IF(!CheckRange(runInfoV2.pad_d, PAD_LOWWER, PAD_UPPER),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    op_name, "pad_d", std::to_string(runInfoV2.pad_d),
                    FormatString("The value of pad_d must be [%d, %d]", PAD_LOWWER, PAD_UPPER).c_str()),
                return false);
    OP_CHECK_IF(!CheckRange(runInfoV2.pad_l, PAD_LOWWER, PAD_UPPER),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    op_name, "pad_l", std::to_string(runInfoV2.pad_l),
                    FormatString("The value of pad_l must be [%d, %d]", PAD_LOWWER, PAD_UPPER).c_str()),
                return false);
    OP_CHECK_IF(!CheckRange(runInfoV2.pad_r, PAD_LOWWER, PAD_UPPER),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    op_name, "pad_r", std::to_string(runInfoV2.pad_r),
                    FormatString("The value of pad_r must be [%d, %d]", PAD_LOWWER, PAD_UPPER).c_str()),
                return false);
    return true;
}

bool SetConvBackpropFilterAttrs(const gert::TilingContext* context, Conv3dBpFilterV2RunInfo& runInfoV2)
{
    const auto op_name = (context->GetNodeName() == nullptr) ? "nil" : context->GetNodeName();
    OP_CHECK_IF(!SetStridesAttr(context, runInfoV2), OP_LOGW(op_name, "failed to set strides attrs."), return false);
    OP_CHECK_IF(!SetDilationsAttr(context, runInfoV2), OP_LOGW(op_name, "failed to set dilation attrs."), return false);

    const gert::ContinuousVector* pads = nullptr;
    if (!ValidatePadsAttr(op_name, context->GetAttrs(), pads)) {
        return false;
    }

    const int64_t* pads_data = static_cast<const int64_t*>(pads->GetData());
    SetPadsData(runInfoV2, pads_data);

    const auto attrs = context->GetAttrs();
    if (attrs->GetAttrNum() <= PADDING_INDEX) {
        OP_LOGD(op_name, "no padding attr, skip calc and check");
        return true;
    }

    if (IsArchAfter35(context)) {
        PAD_UPPER = INT32_MAX - 1;
    }

    const char* padding = attrs->GetAttrPointer<char>(PADDING_INDEX);
    ReCalPaddings(runInfoV2, padding);
    if (!CheckPadsRange(op_name, runInfoV2)) {
        return false;
    }

    if (std::max({runInfoV2.pad_f, runInfoV2.pad_b, runInfoV2.pad_u, runInfoV2.pad_d, runInfoV2.pad_l,
                  runInfoV2.pad_r}) > 0) {
        OP_LOGD(op_name, "gradient calculation in padding regions is shape-dependent."
                         "certain kernel optimizations may skip computation and explicitly force these gradients to 0");
    }
    return CheckGradOutputShape(context, runInfoV2);
}
} // namespace

bool SetConv3dBpFilterV2RunInfo(const gert::TilingContext* context, Conv3dBpFilterV2RunInfo& runInfoV2)
{
    OP_CHECK_IF(!SetFmapDesc(context, runInfoV2), OP_LOGW("Conv3DBackpropFilterV2", "failed to get x desc."),
                return false);
    OP_CHECK_IF(!SetGradOutputDesc(context, runInfoV2),
                OP_LOGW("Conv3DBackpropFilterV2", "failed to get out_backprop desc."), return false);
    OP_CHECK_IF(!SetOutputDesc(context, runInfoV2), OP_LOGW("Conv3DBackpropFilterV2", "failed to get output desc."),
                return false);
    OP_CHECK_IF(!SetConvBackpropFilterAttrs(context, runInfoV2),
                OP_LOGW("Conv3DBackpropFilterV2", "failed to get attrs."), return false);
    return true;
}

} // namespace Conv
} // namespace NN
} // namespace Ops
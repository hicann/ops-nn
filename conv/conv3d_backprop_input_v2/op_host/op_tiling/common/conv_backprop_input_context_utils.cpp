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
 * \file conv_backprop_input_context_utils.cpp
 * \brief
 */
#include "conv_backprop_input_context_utils.h"
#include "conv_backprop_input_context_utils_internal.h"
#include <log/log.h>
#include <util/math_util.h>
#include <unordered_set>
#include <cstdarg>
#include "error_util.h"
#include "conv/common/op_host/op_tiling/conv_math_util.h"
#include "conv/common/op_host/op_tiling/conv_platform_util.h"
#include "securec.h"
#include "conv/common/op_host/op_tiling/arch35/conv_base_numblocks_decision.h"

using namespace optiling::conv_ops_tiling;
namespace Ops {
namespace NN {
namespace Conv {

bool CheckRangeInt64(int64_t value, int32_t value_low, int32_t value_up)
{
    if (value < value_low || value > value_up) {
        return false;
    }
    return true;
}

bool IsArchAfter35(const gert::TilingContext* context)
{
    return context->GetCompileInfo<Ops::NN::Conv::Conv3DBackpropV2CompileInfo>()->npuArch == NpuArch::DAV_3510;
}

bool IsSupportedDtypeForOutputPadding(const ge::DataType dtype)
{
    return dtype == ge::DT_BF16 || dtype == ge::DT_FLOAT16 || dtype == ge::DT_FLOAT || dtype == ge::DT_INT8;
}

bool ValidateConvBackpropContext(const gert::TilingContext* context)
{
    // 校验输入张量描述是否获取成功
    auto input_size_desc = context->GetInputDesc(INPUT_SIZE_INDEX);
    OP_CHECK_IF(input_size_desc == nullptr,
                CUBE_INNER_ERR_REPORT(context->GetNodeName(), "get input_size desc from context fail."), return false);
    auto filter_desc = context->GetInputDesc(FILTER_INDEX);
    OP_CHECK_IF(filter_desc == nullptr,
                CUBE_INNER_ERR_REPORT(context->GetNodeName(), "get filter desc from context fail."), return false);
    auto out_backprop_desc = context->GetInputDesc(OUT_BACKPROP_INDEX);
    OP_CHECK_IF(out_backprop_desc == nullptr,
                CUBE_INNER_ERR_REPORT(context->GetNodeName(), "get out_backprop desc from context fail."),
                return false);
    // 校验输出张量描述是否获取成功
    auto y_desc = context->GetOutputDesc(Y_INDEX);
    OP_CHECK_IF(y_desc == nullptr, CUBE_INNER_ERR_REPORT(context->GetNodeName(), "get y desc from context fail."),
                return false);
    // 校验属性参数获取是否成功
    auto attrs = context->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, CUBE_INNER_ERR_REPORT(context->GetNodeName(), "failed to get runtime attrs."),
                return false);

    return true;
}

bool CheckAttrRangeDilations(const gert::TilingContext* context, const int64_t* dilations)
{
    const auto op_name = context->GetNodeName();
    auto y_ori_format = context->GetOutputDesc(Y_INDEX)->GetOriginFormat();
    int32_t kDilationUpTmp = (IsArchAfter35(context) || IsSocVersionFuse(context)) ? kDimUp : kDilationUp;
    if (y_ori_format == ge::FORMAT_NCDHW) {
        OP_CHECK_IF(!CheckRangeInt64(dilations[K_N_DIM_NCDHW], K_DEFAULT_DILATIONS, K_DEFAULT_DILATIONS),
                    OP_LOGE_FOR_INVALID_VALUE(op_name, "dilation_n", std::to_string(dilations[K_N_DIM_NCDHW]).c_str(),
                                              std::to_string(K_DEFAULT_DILATIONS).c_str()),
                    return false);
        OP_CHECK_IF(!CheckRangeInt64(dilations[K_C_DIM_NCDHW], K_DEFAULT_DILATIONS, K_DEFAULT_DILATIONS),
                    OP_LOGE_FOR_INVALID_VALUE(op_name, "dilation_c", std::to_string(dilations[K_C_DIM_NCDHW]).c_str(),
                                              std::to_string(K_DEFAULT_DILATIONS).c_str()),
                    return false);
        OP_CHECK_IF(
            !CheckRangeInt64(dilations[K_D_DIM_NCDHW], kDilationLow, kDilationUpTmp),
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                op_name, "dilation_d", std::to_string(dilations[K_D_DIM_NCDHW]).c_str(),
                FormatString("The value of dilation_d must be range [%d, %d]", kDilationLow, kDilationUpTmp).c_str()),
            return false);
        OP_CHECK_IF(
            !CheckRangeInt64(dilations[K_H_DIM_NCDHW], kDilationLow, kDilationUpTmp),
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                op_name, "dilation_h", std::to_string(dilations[K_H_DIM_NCDHW]).c_str(),
                FormatString("The value of dilation_h must be range [%d, %d]", kDilationLow, kDilationUpTmp).c_str()),
            return false);
        OP_CHECK_IF(
            !CheckRangeInt64(dilations[K_W_DIM_NCDHW], kDilationLow, kDilationUpTmp),
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                op_name, "dilation_w", std::to_string(dilations[K_W_DIM_NCDHW]).c_str(),
                FormatString("The value of dilation_w must be range [%d, %d]", kDilationLow, kDilationUpTmp).c_str()),
            return false);
    } else {
        OP_CHECK_IF(!CheckRangeInt64(dilations[K_N_DIM_NDHWC], K_DEFAULT_DILATIONS, K_DEFAULT_DILATIONS),
                    OP_LOGE_FOR_INVALID_VALUE(op_name, "dilation_n", std::to_string(dilations[K_N_DIM_NDHWC]).c_str(),
                                              std::to_string(K_DEFAULT_DILATIONS).c_str()),
                    return false);
        OP_CHECK_IF(!CheckRangeInt64(dilations[K_C_DIM_NDHWC], K_DEFAULT_DILATIONS, K_DEFAULT_DILATIONS),
                    OP_LOGE_FOR_INVALID_VALUE(op_name, "dilation_c", std::to_string(dilations[K_C_DIM_NDHWC]).c_str(),
                                              std::to_string(K_DEFAULT_DILATIONS).c_str()),
                    return false);
        OP_CHECK_IF(
            !CheckRangeInt64(dilations[K_D_DIM_NDHWC], kDilationLow, kDilationUpTmp),
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                op_name, "dilation_d", std::to_string(dilations[K_D_DIM_NDHWC]).c_str(),
                FormatString("The value of dilation_d must be range [%d, %d]", kDilationLow, kDilationUpTmp).c_str()),
            return false);
        OP_CHECK_IF(
            !CheckRangeInt64(dilations[K_H_DIM_NDHWC], kDilationLow, kDilationUpTmp),
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                op_name, "dilation_h", std::to_string(dilations[K_H_DIM_NDHWC]).c_str(),
                FormatString("The value of dilation_h must be range [%d, %d]", kDilationLow, kDilationUpTmp).c_str()),
            return false);
        OP_CHECK_IF(
            !CheckRangeInt64(dilations[K_W_DIM_NDHWC], kDilationLow, kDilationUpTmp),
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                op_name, "dilation_w", std::to_string(dilations[K_W_DIM_NDHWC]).c_str(),
                FormatString("The value of dilation_w must be range [%d, %d]", kDilationLow, kDilationUpTmp).c_str()),
            return false);
    }

    return true;
}

bool CheckAttrRangeStrides(const gert::TilingContext* context, const int64_t* strides)
{
    const auto op_name = context->GetNodeName();
    auto y_ori_format = context->GetOutputDesc(Y_INDEX)->GetOriginFormat();
    if (y_ori_format == ge::FORMAT_NCDHW) {
        OP_CHECK_IF(!CheckRangeInt64(strides[K_N_DIM_NCDHW], K_DEFAULT_STRIDES, K_DEFAULT_STRIDES),
                    OP_LOGE_FOR_INVALID_VALUE(op_name, "stride_n", std::to_string(strides[K_N_DIM_NCDHW]).c_str(),
                                              std::to_string(K_DEFAULT_STRIDES).c_str()),
                    return false);
        OP_CHECK_IF(!CheckRangeInt64(strides[K_C_DIM_NCDHW], K_DEFAULT_STRIDES, K_DEFAULT_STRIDES),
                    OP_LOGE_FOR_INVALID_VALUE(op_name, "stride_c", std::to_string(strides[K_C_DIM_NCDHW]).c_str(),
                                              std::to_string(K_DEFAULT_STRIDES).c_str()),
                    return false);
        OP_CHECK_IF(!CheckRangeInt64(strides[K_D_DIM_NCDHW], kDimLow, kDimUp),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        op_name, "stride_d", std::to_string(strides[K_D_DIM_NCDHW]).c_str(),
                        FormatString("The value of stride_d must be range [%d, %d]", kDimLow, kDimUp).c_str()),
                    return false);
        OP_CHECK_IF(!CheckRangeInt64(strides[K_H_DIM_NCDHW], kDimLow, kDimUp),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        op_name, "stride_h", std::to_string(strides[K_H_DIM_NCDHW]).c_str(),
                        FormatString("The value of stride_h must be range [%d, %d]", kDimLow, kDimUp).c_str()),
                    return false);
        OP_CHECK_IF(!CheckRangeInt64(strides[K_W_DIM_NCDHW], kDimLow, kDimUp),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        op_name, "stride_w", std::to_string(strides[K_W_DIM_NCDHW]).c_str(),
                        FormatString("The value of stride_w must be range [%d, %d]", kDimLow, kDimUp).c_str()),
                    return false);
    } else {
        OP_CHECK_IF(!CheckRangeInt64(strides[K_N_DIM_NDHWC], K_DEFAULT_STRIDES, K_DEFAULT_STRIDES),
                    OP_LOGE_FOR_INVALID_VALUE(op_name, "stride_n", std::to_string(strides[K_N_DIM_NDHWC]).c_str(),
                                              std::to_string(K_DEFAULT_STRIDES).c_str()),
                    return false);
        OP_CHECK_IF(!CheckRangeInt64(strides[K_C_DIM_NDHWC], K_DEFAULT_STRIDES, K_DEFAULT_STRIDES),
                    OP_LOGE_FOR_INVALID_VALUE(op_name, "stride_c", std::to_string(strides[K_C_DIM_NDHWC]).c_str(),
                                              std::to_string(K_DEFAULT_STRIDES).c_str()),
                    return false);
        OP_CHECK_IF(!CheckRangeInt64(strides[K_D_DIM_NDHWC], kDimLow, kDimUp),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        op_name, "stride_d", std::to_string(strides[K_D_DIM_NDHWC]).c_str(),
                        FormatString("The value of stride_d must be range [%d, %d]", kDimLow, kDimUp).c_str()),
                    return false);
        OP_CHECK_IF(!CheckRangeInt64(strides[K_H_DIM_NDHWC], kDimLow, kDimUp),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        op_name, "stride_h", std::to_string(strides[K_H_DIM_NDHWC]).c_str(),
                        FormatString("The value of stride_h must be range [%d, %d]", kDimLow, kDimUp).c_str()),
                    return false);
        OP_CHECK_IF(!CheckRangeInt64(strides[K_W_DIM_NDHWC], kDimLow, kDimUp),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        op_name, "stride_w", std::to_string(strides[K_W_DIM_NDHWC]).c_str(),
                        FormatString("The value of stride_w must be range [%d, %d]", kDimLow, kDimUp).c_str()),
                    return false);
    }

    return true;
}

bool CheckAttrRangePads(const gert::TilingContext* context, const int64_t* pads)
{
    const auto op_name = context->GetNodeName();
    int32_t padHwUp = kPadUp;
    int32_t padDUp = kDimUp;

    if (IsSocVersionFuse(context)) {
        padDUp = kPadUp;
    }

    if (IsArchAfter35(context)) {
        padHwUp = kDimUp;
    }

    OP_CHECK_IF(!CheckRangeInt64(pads[K_CONV3D_PAD_HEAD_IDX], 0, padDUp),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    op_name, "pad_h", std::to_string(pads[K_CONV3D_PAD_HEAD_IDX]).c_str(),
                    FormatString("The value of pad_h must be range [%d, %d]", 0, padDUp).c_str()),
                return false);
    OP_CHECK_IF(!CheckRangeInt64(pads[K_CONV3D_PAD_TAIL_IDX], 0, padDUp),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    op_name, "pad_t", std::to_string(pads[K_CONV3D_PAD_TAIL_IDX]).c_str(),
                    FormatString("The value of pad_t must be range [%d, %d]", 0, padDUp).c_str()),
                return false);
    OP_CHECK_IF(!CheckRangeInt64(pads[K_CONV3D_PAD_UP_IDX], 0, padHwUp),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    op_name, "pad_u", std::to_string(pads[K_CONV3D_PAD_UP_IDX]).c_str(),
                    FormatString("The value of pad_u must be range [%d, %d]", 0, padHwUp).c_str()),
                return false);
    OP_CHECK_IF(!CheckRangeInt64(pads[K_CONV3D_PAD_DOWN_IDX], 0, padHwUp),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    op_name, "pad_d", std::to_string(pads[K_CONV3D_PAD_DOWN_IDX]).c_str(),
                    FormatString("The value of pad_d must be range [%d, %d]", 0, padHwUp).c_str()),
                return false);
    OP_CHECK_IF(!CheckRangeInt64(pads[K_CONV3D_PAD_LEFT_IDX], 0, padHwUp),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    op_name, "pad_l", std::to_string(pads[K_CONV3D_PAD_LEFT_IDX]).c_str(),
                    FormatString("The value of pad_l must be range [%d, %d]", 0, padHwUp).c_str()),
                return false);
    OP_CHECK_IF(!CheckRangeInt64(pads[K_CONV3D_PAD_RIGHT_IDX], 0, padHwUp),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    op_name, "pad_r", std::to_string(pads[K_CONV3D_PAD_RIGHT_IDX]).c_str(),
                    FormatString("The value of pad_r must be range [%d, %d]", 0, padHwUp).c_str()),
                return false);

    return true;
}

bool CheckAttrRange(gert::TilingContext* context, const int64_t* strides, const int64_t* pads, const int64_t* dilations,
                    const int64_t* groups)
{
    const auto op_name = context->GetNodeName();
    OP_CHECK_IF(!CheckAttrRangeDilations(context, dilations), OP_LOGE(op_name, "check dilations range failed"),
                return false);
    OP_CHECK_IF(!CheckAttrRangeStrides(context, strides), OP_LOGE(op_name, "check strides range failed"), return false);
    // Exclude (pad_u,pad_d,pad_l,pad_r) =-1 while paddings is "SAME"
    if (pads[K_CONV3D_PAD_UP_IDX] != -1 && pads[K_CONV3D_PAD_DOWN_IDX] != -1 && pads[K_CONV3D_PAD_LEFT_IDX] != -1 &&
        pads[K_CONV3D_PAD_RIGHT_IDX] != -1) {
        OP_CHECK_IF(!CheckAttrRangePads(context, pads), OP_LOGE(op_name, "check pads range failed"), return false);
    }

    // groups: [1, 2G - 1 ]
    if (groups != nullptr) {
        OP_CHECK_IF(!CheckRangeInt64(*groups, kDimLow, kDimUp),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        op_name, "group", std::to_string(*groups).c_str(),
                        FormatString("The value of group must be range [%d, %d]", kDimLow, kDimUp).c_str()),
                    return false);
    }

    return true;
}

bool CheckTransposeAttr(gert::TilingContext* context, OtherParams& otherParams)
{
    auto yDesc = context->GetOutputDesc(Y_INDEX);
    auto attrs = context->GetAttrs();
    auto outputPadding = attrs->GetAttrPointer<gert::ContinuousVector>(K_OUTPUT_PADDING_CONV3D_TRANSPOSE_IDX);
    OP_CHECK_IF(outputPadding == nullptr,
                CUBE_INNER_ERR_REPORT(context->GetNodeName(), "failed to get output_padding attrs."), return false);
    OP_CHECK_IF(outputPadding->GetSize() != K_ORI_SHAPE_DIM_3D,
                OP_LOGE_WITH_INVALID_ATTR_SIZE(context->GetNodeName(), "output_padding",
                                               std::to_string(outputPadding->GetSize()).c_str(), "5"),
                return false);
    const auto outputPaddingData = static_cast<const int64_t*>(outputPadding->GetData());
    if (yDesc->GetOriginFormat() == ge::FORMAT_NCDHW) {
        OP_CHECK_IF(outputPaddingData[K_N_DIM_NCDHW] != 0 || outputPaddingData[K_C_DIM_NCDHW] != 0,
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(context->GetNodeName(), "output_padding N and C",
                                                           (std::to_string(outputPaddingData[K_N_DIM_NCDHW]) + " and " +
                                                            std::to_string(outputPaddingData[K_C_DIM_NCDHW]))
                                                               .c_str(),
                                                           "The value of output_padding N and C must be 0"),
                    return false);
        otherParams.output_padding.output_padding_d = outputPaddingData[K_D_DIM_NCDHW];
        otherParams.output_padding.output_padding_h = outputPaddingData[K_H_DIM_NCDHW];
        otherParams.output_padding.output_padding_w = outputPaddingData[K_W_DIM_NCDHW];
    } else { // yDesc->GetOriginFormat() == ge::FORMAT_NDHWC
        OP_CHECK_IF(outputPaddingData[K_N_DIM_NDHWC] != 0 || outputPaddingData[K_C_DIM_NDHWC] != 0,
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(context->GetNodeName(), "output_padding N and C",
                                                           (std::to_string(outputPaddingData[K_N_DIM_NDHWC]) + " and " +
                                                            std::to_string(outputPaddingData[K_C_DIM_NDHWC]))
                                                               .c_str(),
                                                           "The value of output_padding N and C must be zero"),
                    return false);
        otherParams.output_padding.output_padding_d = outputPaddingData[K_D_DIM_NDHWC];
        otherParams.output_padding.output_padding_h = outputPaddingData[K_H_DIM_NDHWC];
        otherParams.output_padding.output_padding_w = outputPaddingData[K_W_DIM_NDHWC];
    }
    if (attrs->GetAttrNum() > K_OFFSET_X_CONV3D_TRANSPOSE_IDX) {
        const auto offsetX = attrs->GetAttrPointer<int64_t>(K_OFFSET_X_CONV3D_TRANSPOSE_IDX);
        OP_CHECK_IF(offsetX == nullptr, CUBE_INNER_ERR_REPORT(context->GetNodeName(), "failed to get offsetX attrs"),
                    return false);
        if (!IsArchAfter35(context) && !IsSocVersionFuse(context)) {
            OP_CHECK_IF(
                *offsetX != 0,
                OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "offsetX", std::to_string(*offsetX).c_str(), "0"),
                return false);
        }
    }
    if (IsSocVersionFuse(context)) {
        if (attrs->GetAttrNum() > K_FUSION_MODE_CONV3D_TRANSPOSE_IDX) {
            const auto fusion_mode = attrs->GetAttrPointer<int32_t>(K_FUSION_MODE_CONV3D_TRANSPOSE_IDX);
            OP_CHECK_IF(fusion_mode == nullptr,
                        CUBE_INNER_ERR_REPORT(context->GetNodeName(), "failed to get fusion_mode attrs"), return false);
            OP_CHECK_IF(*fusion_mode != 0 && *fusion_mode != 1,
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "fusion_mode",
                                                              std::to_string(*fusion_mode).c_str(),
                                                              "The value of fusion_mode must be in {0, 1}"),
                        return false);
        }
        if (attrs->GetAttrNum() > K_Y_QUANT_MODE_CONV3D_TRANSPOSE_IDX) {
            const auto y_quant_mode = attrs->GetAttrPointer<int32_t>(K_Y_QUANT_MODE_CONV3D_TRANSPOSE_IDX);
            OP_CHECK_IF(y_quant_mode == nullptr,
                        CUBE_INNER_ERR_REPORT(context->GetNodeName(), "failed to get quant_mode attrs"), return false);
            OP_CHECK_IF(*y_quant_mode != 0,
                        OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "y_quant_mode",
                                                  std::to_string(*y_quant_mode).c_str(), "0"),
                        return false);
        }
    }
    return true;
}

template <typename T>
void GetNCDHWShape(const T& origin_shape, Shape& ncdhw_shape, const ge::Format& origin_format)
{
    // caller already checked buffer size
    if (origin_format == ge::FORMAT_NDHWC) {
        ncdhw_shape.batch = origin_shape[0]; // 0: N
        ncdhw_shape.c = origin_shape[4];     // 4: C
        ncdhw_shape.d = origin_shape[1];     // 1: D
        ncdhw_shape.h = origin_shape[2];     // 2: H
        ncdhw_shape.w = origin_shape[3];     // 3: W
    } else if (origin_format == ge::FORMAT_NCDHW) {
        ncdhw_shape.batch = origin_shape[0]; // 0: N
        ncdhw_shape.c = origin_shape[1];     // 1: C
        ncdhw_shape.d = origin_shape[2];     // 2: D
        ncdhw_shape.h = origin_shape[3];     // 3: H
        ncdhw_shape.w = origin_shape[4];     // 4: W
    } else if (origin_format == ge::FORMAT_DHWCN) {
        ncdhw_shape.batch = origin_shape[4]; // 4: N
        ncdhw_shape.c = origin_shape[3];     // 3: C
        ncdhw_shape.d = origin_shape[0];     // 0: D
        ncdhw_shape.h = origin_shape[1];     // 1: H
        ncdhw_shape.w = origin_shape[2];     // 2: W
    } else if (origin_format == ge::FORMAT_NCHW) {
        // test
        ncdhw_shape.batch = origin_shape[1]; // 1: N
        ncdhw_shape.c = origin_shape[0];     // 0: C
        ncdhw_shape.d = 1;                   // 0: D
        ncdhw_shape.h = origin_shape[2];     // 2: H
        ncdhw_shape.w = origin_shape[3];     // 3: W
    }
}

bool CheckTransposeOutputdingRange(const gert::TilingContext* context, const Conv3dBpInputV2RunInfo& runInfoV2,
                                   const OtherParams& otherParams)
{
    // outputPadding值需要小于同维度dilation或stride
    OP_CHECK_IF((otherParams.output_padding.output_padding_d >= runInfoV2.stride_d &&
                 otherParams.output_padding.output_padding_d >= runInfoV2.dilation_d),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context->GetNodeName(), "output_padding_d",
                    std::to_string(otherParams.output_padding.output_padding_d).c_str(),
                    FormatString("The value of output_padding_d must be smaller than dilation_d[%d] or stride_d[%d]",
                                 runInfoV2.dilation_d, runInfoV2.stride_d)
                        .c_str()),
                return false);
    OP_CHECK_IF((otherParams.output_padding.output_padding_h >= runInfoV2.stride_h &&
                 otherParams.output_padding.output_padding_h >= runInfoV2.dilation_h),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context->GetNodeName(), "output_padding_h",
                    std::to_string(otherParams.output_padding.output_padding_h).c_str(),
                    FormatString("The value of output_padding_h must be smaller than dilation_h[%d] or stride_h[%d]",
                                 runInfoV2.dilation_h, runInfoV2.stride_h)
                        .c_str()),
                return false);
    OP_CHECK_IF((otherParams.output_padding.output_padding_w >= runInfoV2.stride_w &&
                 otherParams.output_padding.output_padding_w >= runInfoV2.dilation_w),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context->GetNodeName(), "output_padding_w",
                    std::to_string(otherParams.output_padding.output_padding_w).c_str(),
                    FormatString("The value of output_padding_w must be smaller than dilation_w[%d] or stride_w[%d]",
                                 runInfoV2.dilation_w, runInfoV2.stride_w)
                        .c_str()),
                return false);
    return true;
}

bool UpdateDtypeParams(const gert::TilingContext* context, Conv3dBpInputV2RunInfo& runInfoV2,
                       const optiling::OpTypeV2 op_type, OtherParams& otherParams)
{
    const auto op_name = context->GetNodeName();

    if (op_type == optiling::OpTypeV2::kConv3DTransposeV2 || op_type == optiling::OpTypeV2::kExtendConvTranspose) {
        // Conv3DTranspose, index of x is 1, index of filter is 2
        otherParams.a_dtype = context->GetInputDesc(FILTER_INDEX)->GetDataType();
        otherParams.b_dtype = context->GetInputDesc(OUT_BACKPROP_INDEX)->GetDataType();
        otherParams.c_dtype = context->GetOutputDesc(Y_INDEX)->GetDataType();
    } else {
        // Conv3dBackpropInput, index of out_backprop is 2, index of filter is 1
        otherParams.a_dtype = context->GetInputDesc(OUT_BACKPROP_INDEX)->GetDataType();
        otherParams.b_dtype = context->GetInputDesc(FILTER_INDEX)->GetDataType();
        otherParams.c_dtype = context->GetOutputDesc(Y_INDEX)->GetDataType();
    }

    if (otherParams.a_dtype == ge::DT_BF16 && otherParams.b_dtype == ge::DT_BF16 &&
        otherParams.c_dtype == ge::DT_BF16) {
        otherParams.a_dtype = ge::DT_FLOAT16;
        otherParams.b_dtype = ge::DT_FLOAT16;
        otherParams.c_dtype = ge::DT_FLOAT16;
    }

    bool isFp16Flag = otherParams.a_dtype == ge::DT_FLOAT16 && otherParams.b_dtype == ge::DT_FLOAT16 &&
                      otherParams.c_dtype == ge::DT_FLOAT16;
    bool isFp32Flag = otherParams.a_dtype == ge::DT_FLOAT && otherParams.b_dtype == ge::DT_FLOAT &&
                      otherParams.c_dtype == ge::DT_FLOAT;
    bool dtypeSupportFlag = isFp16Flag || isFp32Flag;
    string dtypeCheckLog = "fp16 and fp32";
    if (IsArchAfter35(context) || IsSocVersionFuse(context)) {
        bool isHiF8Flag = otherParams.a_dtype == ge::DT_HIFLOAT8 && otherParams.b_dtype == ge::DT_HIFLOAT8 &&
                          otherParams.c_dtype == ge::DT_HIFLOAT8;
        bool isFp8E4M3Flag = otherParams.a_dtype == ge::DT_FLOAT8_E4M3FN &&
                             otherParams.b_dtype == ge::DT_FLOAT8_E4M3FN && otherParams.c_dtype == ge::DT_FLOAT8_E4M3FN;
        bool isInt8Flag = otherParams.a_dtype == ge::DT_INT8 && otherParams.b_dtype == ge::DT_INT8 &&
                          (otherParams.c_dtype == ge::DT_FLOAT16 || otherParams.c_dtype == ge::DT_INT8);
        dtypeSupportFlag = isHiF8Flag || isFp8E4M3Flag || isFp16Flag || isFp32Flag || isInt8Flag;
        dtypeCheckLog = "hifloat8, float8_e4m3, int8, fp16 and fp32";
    }
    if (IsSocVersionFuse(context)) {
        bool isInt8Flag = otherParams.a_dtype == ge::DT_INT8 && otherParams.b_dtype == ge::DT_INT8 &&
                          (otherParams.c_dtype == ge::DT_INT8 || otherParams.c_dtype == ge::DT_FLOAT16);
        bool isA16W8Flag = otherParams.a_dtype == ge::DT_FLOAT16 && otherParams.b_dtype == ge::DT_INT8 &&
                           (otherParams.c_dtype == ge::DT_INT8 || otherParams.c_dtype == ge::DT_FLOAT16);
        bool isFp16Int8Flag = otherParams.a_dtype == ge::DT_FLOAT16 && otherParams.b_dtype == ge::DT_FLOAT16 &&
                              otherParams.c_dtype == ge::DT_INT8;
        dtypeSupportFlag = isFp16Flag || isInt8Flag || isA16W8Flag || isFp16Int8Flag;
        dtypeCheckLog = "fp16 and int8";
    }
    OP_CHECK_IF(!dtypeSupportFlag,
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                    op_name, "out_backprop, filter and y",
                    (ge::TypeUtils::DataTypeToSerialString(otherParams.a_dtype) + ", " +
                     ge::TypeUtils::DataTypeToSerialString(otherParams.b_dtype) + " and " +
                     ge::TypeUtils::DataTypeToSerialString(otherParams.c_dtype))
                        .c_str(),
                    ("The dtypes of out_backprop, filter and y must be within the range " + dtypeCheckLog).c_str()),
                return false);

    runInfoV2.a_dtype_bytes = ge::GetSizeByDataType(otherParams.a_dtype);
    OP_CHECK_IF(runInfoV2.a_dtype_bytes == -1,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    op_name, "out_backprop", ge::TypeUtils::DataTypeToSerialString(otherParams.a_dtype).c_str(),
                    "The dtype size of out_backprop must be not equal to -1"),
                return false);
    runInfoV2.b_dtype_bytes = ge::GetSizeByDataType(otherParams.b_dtype);
    OP_CHECK_IF(runInfoV2.b_dtype_bytes == -1,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    op_name, "filter", ge::TypeUtils::DataTypeToSerialString(otherParams.b_dtype).c_str(),
                    "The dtype size of filter must be not equal to -1"),
                return false);
    runInfoV2.c_dtype_bytes = ge::GetSizeByDataType(otherParams.c_dtype);
    OP_CHECK_IF(runInfoV2.c_dtype_bytes == -1,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    op_name, "y", ge::TypeUtils::DataTypeToSerialString(otherParams.c_dtype).c_str(),
                    "The dtype size of y must be not equal to -1"),
                return false);

    return true;
}

std::string FormatSetToString(const std::unordered_set<ge::Format>& format_set)
{
    std::string result;
    for (const auto& fmt : format_set) {
        if (!result.empty()) {
            result += "/";
        }
        result += ge::TypeUtils::FormatToSerialString(fmt);
    }
    return result;
}

bool CheckStorageFormat(const gert::TilingContext* context, size_t filter_input_index, size_t out_backprop_input_index,
                        optiling::OpTypeV2 op_type)
{
    // 获取输入输出的描述信息
    const auto out_backprop_desc = context->GetInputDesc(out_backprop_input_index);
    const auto filter_desc = context->GetInputDesc(filter_input_index);
    const auto y_desc = context->GetOutputDesc(Y_INDEX);
    // 获取输入输出的format信息
    auto out_backprop_format = static_cast<ge::Format>(ge::GetPrimaryFormat(out_backprop_desc->GetStorageFormat()));
    auto filter_format = static_cast<ge::Format>(ge::GetPrimaryFormat(filter_desc->GetStorageFormat()));
    auto y_format = static_cast<ge::Format>(ge::GetPrimaryFormat(y_desc->GetStorageFormat()));
    const auto op_name = context->GetNodeName();

    std::unordered_set<ge::Format> valid_out_bp_format;
    if ((IsArchAfter35(context) || IsSocVersionFuse(context)) && op_type == optiling::OpTypeV2::kExtendConvTranspose) {
        valid_out_bp_format = {ge::FORMAT_NCDHW};
    } else {
        valid_out_bp_format = {ge::FORMAT_NCDHW, ge::FORMAT_NDHWC};
    }

    std::unordered_set<ge::Format> valid_filter_format;
    if (IsSocVersionFuse(context)) {
        valid_filter_format = {ge::FORMAT_NDHWC, ge::FORMAT_FRACTAL_Z};
    } else if (op_type == optiling::OpTypeV2::kExtendConvTranspose) {
        valid_filter_format = {ge::FORMAT_NCDHW};
    } else {
        valid_filter_format = {ge::FORMAT_NCDHW, ge::FORMAT_NDHWC, ge::FORMAT_DHWCN};
    }

    std::unordered_set<ge::Format> valid_y_format;
    if ((IsArchAfter35(context) || IsSocVersionFuse(context)) && op_type == optiling::OpTypeV2::kExtendConvTranspose) {
        valid_y_format = {ge::FORMAT_NCDHW};
    } else {
        valid_y_format = {ge::FORMAT_NCDHW, ge::FORMAT_NDHWC};
    }

    bool invalid_tag = valid_out_bp_format.count(out_backprop_format) == 0 ||
                       valid_filter_format.count(filter_format) == 0 || valid_y_format.count(y_format) == 0;
    if (invalid_tag) {
        std::string formatsStr = ge::TypeUtils::FormatToSerialString(out_backprop_format) + ", " +
                                 ge::TypeUtils::FormatToSerialString(y_format) + " and " +
                                 ge::TypeUtils::FormatToSerialString(filter_format);
        OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(
            op_name, "out_backprop, y and filter", formatsStr.c_str(),
            ("The formats of out_backprop and y must be within the range {" + FormatSetToString(valid_out_bp_format) +
             "}, the format of filter must be within the range {" + FormatSetToString(valid_filter_format) + "}")
                .c_str());
        return false;
    }

    return true;
}

bool UpdateShapeParams(const gert::TilingContext* context, const Conv3dBpInputV2RunInfo& runInfoV2,
                       const Shape& out_backprop_shape_ncdhw, const Shape& filter_shape_ncdhw,
                       const Shape& y_shape_ncdhw, OtherParams& otherParams)
{
    const auto op_name = context->GetNodeName();
    OP_CHECK_IF(
        kDtypeBlockReduceMap.find(otherParams.a_dtype) == kDtypeBlockReduceMap.end() ||
            kDtypeBlockReduceMap.find(otherParams.c_dtype) == kDtypeBlockReduceMap.end() ||
            kDtypeBlockReduceMap.find(otherParams.b_dtype) == kDtypeBlockReduceMap.end(),
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(op_name, "out_backprop, filter and y",
                                               (ge::TypeUtils::DataTypeToSerialString(otherParams.a_dtype) + ", " +
                                                ge::TypeUtils::DataTypeToSerialString(otherParams.b_dtype) + " and " +
                                                ge::TypeUtils::DataTypeToSerialString(otherParams.c_dtype))
                                                   .c_str(),
                                               "The dtypes of out_backprop, filter and y must be within the range "
                                               "{DT_HIFLOAT8, DT_FLOAT8_E4M3FN, DT_FLOAT16, DT_FLOAT, DT_INT8}"),
        return false);
    // a_shape means out_backprop shape
    otherParams.a_shape.c = out_backprop_shape_ncdhw.c;
    otherParams.a_shape.d = out_backprop_shape_ncdhw.d;
    otherParams.a_shape.h = out_backprop_shape_ncdhw.h;
    otherParams.a_shape.w = out_backprop_shape_ncdhw.w;
    // only support fp16 now
    otherParams.a_shape.c0 = kDtypeBlockReduceMap.at(otherParams.a_dtype);
    otherParams.b_shape.c0 = kDtypeBlockReduceMap.at(otherParams.b_dtype);
    otherParams.a_shape.c1 = Ops::Base::CeilDiv(otherParams.a_shape.c, otherParams.b_shape.c0);
    otherParams.b_shape.c1 = Ops::Base::CeilDiv(otherParams.b_shape.c, otherParams.b_shape.c0);
    // c_shape means y shape
    otherParams.c_shape.c = y_shape_ncdhw.c;
    otherParams.c_shape.d = y_shape_ncdhw.d;
    otherParams.c_shape.h = y_shape_ncdhw.h;
    otherParams.c_shape.w = y_shape_ncdhw.w;
    otherParams.c_shape.c0 = kDtypeBlockReduceMap.at(otherParams.c_dtype);
    otherParams.c_shape.c1 = Ops::Base::CeilDiv(otherParams.c_shape.c, otherParams.c_shape.c0);
    // b_shape means filter shape
    otherParams.b_shape.batch = filter_shape_ncdhw.batch;
    otherParams.b_shape.c = filter_shape_ncdhw.c;
    otherParams.b_shape.d = filter_shape_ncdhw.d;
    otherParams.b_shape.h = static_cast<int32_t>(filter_shape_ncdhw.h);
    otherParams.b_shape.w = static_cast<int32_t>(filter_shape_ncdhw.w);
    otherParams.filter_d_dilation = (otherParams.b_shape.d - 1) * runInfoV2.dilation_d + 1;
    otherParams.filter_h_dilation = (otherParams.b_shape.h - 1) * runInfoV2.dilation_h + 1;
    otherParams.filter_w_dilation = (otherParams.b_shape.w - 1) * runInfoV2.dilation_w + 1;

    return true;
}

void ExtractStorageShapeInfo(const gert::TilingContext* context, size_t filter_input_index,
                             size_t out_backprop_input_index, const Conv3dBpInputV2RunInfo& runInfoV2,
                             OtherParams& otherParams)
{
    auto filter_shape = context->GetInputShape(filter_input_index);
    auto out_backprop_shape = context->GetInputShape(out_backprop_input_index);
    auto y_shape = context->GetOutputShape(Y_INDEX);

    otherParams.a_shape.batch = out_backprop_shape->GetStorageShape().GetDim(kNDimNDC1HWC0Idx);
    otherParams.c_shape.batch = y_shape->GetStorageShape().GetDim(kNDimNDC1HWC0Idx);
    otherParams.filter_gdkci1ghw = filter_shape->GetStorageShape().GetDim(kDkCin1HkWkFRACTALZ3DIdx);
    // NOTE only support shape of filter is (g'*dk*ci1_g'*hk*wk, co1', co0, ci0)
    otherParams.co1g = filter_shape->GetStorageShape().GetDim(kCo1FRACTALZ3DIdx);
    if (!IsArchAfter35(context) && !IsSocVersionFuse(context)) {
        otherParams.co1g = filter_shape->GetStorageShape().GetDim(kCo1FRACTALZ3DIdx);
        if (context->GetOutputDesc(Y_INDEX)->GetDataType() == ge::DT_FLOAT && runInfoV2.groups > 1) {
            otherParams.co1g *= 2; // 2: BLOCK_NUM / FP32_C0
        }
        otherParams.filter_co0 = filter_shape->GetStorageShape().GetDim(kCo0FRACTALZ3DIdx);
        otherParams.filter_ci0 = filter_shape->GetStorageShape().GetDim(kCin0FRACTALZ3DIdx);
    } else {
        otherParams.filter_co0 = BYTE_BLOCK / runInfoV2.b_dtype_bytes;
        otherParams.co1g = Ops::Base::CeilDiv(
            otherParams.multiple_extend * otherParams.b_shape.batch / runInfoV2.groups, otherParams.b_shape.c0);
        otherParams.filter_ci0 = kBlockSize;
    }
    otherParams.co1g_reduce = otherParams.co1g;
}

bool ValidateOriginShapeDims(const gert::TilingContext* context, const gert::Shape& out_backprop_ori_shape,
                             const gert::Shape& filter_ori_shape, const gert::Shape& y_ori_shape)
{
    const auto op_name = context->GetNodeName();
    OP_CHECK_IF(out_backprop_ori_shape.GetDimNum() != K_ORI_SHAPE_DIM_3D,
                OP_LOGE_FOR_INVALID_SHAPEDIM(op_name, "out_backprop",
                                             std::to_string(out_backprop_ori_shape.GetDimNum()).c_str(),
                                             std::to_string(K_ORI_SHAPE_DIM_3D).c_str()),
                return false);
    if (IsSocVersionFuse(context)) {
        OP_CHECK_IF(
            filter_ori_shape.GetDimNum() != K_ORI_SHAPE_DIM_3D && filter_ori_shape.GetDimNum() != K_ORI_SHAPE_DIM_2D,
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(op_name, "filter",
                                                     std::to_string(filter_ori_shape.GetDimNum()).c_str(),
                                                     "The shape dim of filter must be within the range {4, 5}"),
            return false);
    } else {
        OP_CHECK_IF(
            filter_ori_shape.GetDimNum() != K_ORI_SHAPE_DIM_3D,
            OP_LOGE_FOR_INVALID_SHAPEDIM(op_name, "filter", std::to_string(filter_ori_shape.GetDimNum()).c_str(),
                                         std::to_string(K_ORI_SHAPE_DIM_3D).c_str()),
            return false);
    }
    OP_CHECK_IF(y_ori_shape.GetDimNum() != K_ORI_SHAPE_DIM_3D,
                OP_LOGE_FOR_INVALID_SHAPEDIM(op_name, "y", std::to_string(y_ori_shape.GetDimNum()).c_str(),
                                             std::to_string(K_ORI_SHAPE_DIM_3D).c_str()),
                return false);
    return true;
}

bool CalShapeInfoFromDesc(const gert::TilingContext* context, size_t filter_input_index,
                          size_t out_backprop_input_index, const Conv3dBpInputV2RunInfo& runInfoV2,
                          OtherParams& otherParams)
{
    auto filter_desc = context->GetInputDesc(filter_input_index);
    auto out_backprop_desc = context->GetInputDesc(out_backprop_input_index);
    auto y_desc = context->GetOutputDesc(Y_INDEX);

    auto filter_shape = context->GetInputShape(filter_input_index);
    auto out_backprop_shape = context->GetInputShape(out_backprop_input_index);
    auto y_shape = context->GetOutputShape(Y_INDEX);

    ExtractStorageShapeInfo(context, filter_input_index, out_backprop_input_index, runInfoV2, otherParams);

    auto out_backprop_ori_format = out_backprop_desc->GetOriginFormat();
    auto filter_ori_format = filter_desc->GetOriginFormat();
    auto y_ori_format = y_desc->GetOriginFormat();

    const auto& out_backprop_ori_shape = out_backprop_shape->GetOriginShape();
    const auto& filter_ori_shape = filter_shape->GetOriginShape();
    const auto& y_ori_shape = y_shape->GetOriginShape();
    const auto op_name = context->GetNodeName();

    if (!ValidateOriginShapeDims(context, out_backprop_ori_shape, filter_ori_shape, y_ori_shape)) {
        return false;
    }

    Shape out_backprop_shape_ncdhw;
    Shape filter_shape_ncdhw;
    Shape y_shape_ncdhw;
    GetNCDHWShape(out_backprop_ori_shape, out_backprop_shape_ncdhw, out_backprop_ori_format);
    GetNCDHWShape(filter_ori_shape, filter_shape_ncdhw, filter_ori_format);
    GetNCDHWShape(y_ori_shape, y_shape_ncdhw, y_ori_format);

    OP_CHECK_IF(!UpdateShapeParams(context, runInfoV2, out_backprop_shape_ncdhw, filter_shape_ncdhw, y_shape_ncdhw,
                                   otherParams),
                OP_LOGE(op_name, "update shape params failed."), return false);
    return true;
}

bool GetShapeParams(gert::TilingContext* context, Conv3dBpInputV2RunInfo& runInfoV2, optiling::OpTypeV2 op_type,
                    bool isV2Impl, OtherParams& otherParams)
{
    const auto op_name = context->GetNodeName();
    size_t out_backprop_input_index = static_cast<size_t>(OUT_BACKPROP_INDEX);
    size_t filter_input_index = static_cast<size_t>(FILTER_INDEX);
    // 转置的话，filter和out_backprop进行交换
    if (op_type == optiling::OpTypeV2::kConv3DTransposeV2 || op_type == optiling::OpTypeV2::kExtendConvTranspose) {
        out_backprop_input_index = FILTER_INDEX;
        filter_input_index = OUT_BACKPROP_INDEX;
    }
    // 获取输入输出的描述信息
    const auto out_backprop_desc = context->GetInputDesc(out_backprop_input_index);
    const auto filter_desc = context->GetInputDesc(filter_input_index);
    const auto y_desc = context->GetOutputDesc(Y_INDEX);
    // 获取输入输出的shape信息
    const auto out_backprop_shape = context->GetInputShape(out_backprop_input_index);
    const auto filter_shape = context->GetInputShape(filter_input_index);
    const auto y_shape = context->GetOutputShape(Y_INDEX);
    OP_CHECK_IF(out_backprop_desc == nullptr || filter_desc == nullptr || y_desc == nullptr,
                CUBE_INNER_ERR_REPORT(op_name, "failed to get out_backprop/filter/y tensor desc from context."),
                return false);
    OP_CHECK_IF(out_backprop_shape == nullptr || filter_shape == nullptr || y_shape == nullptr,
                CUBE_INNER_ERR_REPORT(op_name, "failed to get out_backprop/filter/y shape from context."),
                return false);

    auto out_backprop_ori_format = out_backprop_desc->GetOriginFormat();
    auto filter_ori_format = filter_desc->GetOriginFormat();
    auto y_ori_format = y_desc->GetOriginFormat();
    OP_CHECK_IF(out_backprop_ori_format != y_ori_format,
                OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(op_name, "out_backprop and y",
                                                        (ge::TypeUtils::FormatToSerialString(out_backprop_ori_format) +
                                                         " and " + ge::TypeUtils::FormatToSerialString(y_ori_format))
                                                            .c_str(),
                                                        "The formats of out_backprop and y must be the same"),
                return false);
    OP_CHECK_IF(out_backprop_ori_format != ge::FORMAT_NDHWC && out_backprop_ori_format != ge::FORMAT_NCDHW,
                OP_LOGE_FOR_INVALID_FORMAT(op_name, "out_backprop",
                                           ge::TypeUtils::FormatToSerialString(out_backprop_ori_format).c_str(),
                                           "NDHWC or NCDHW"),
                return false);
    if (IsArchAfter35(context) || IsSocVersionFuse(context)) {
        OP_CHECK_IF(filter_ori_format != ge::FORMAT_NDHWC && filter_ori_format != ge::FORMAT_NCDHW &&
                        filter_ori_format != ge::FORMAT_DHWCN && filter_ori_format != ge::FORMAT_NCHW,
                    OP_LOGE_FOR_INVALID_FORMAT(op_name, "filter",
                                               ge::TypeUtils::FormatToSerialString(filter_ori_format).c_str(),
                                               "NDHWC or NCDHW or DHWCN or NCHW"),
                    return false);
        OP_CHECK_IF(!CheckStorageFormat(context, filter_input_index, out_backprop_input_index, op_type),
                    OP_LOGE(op_name, "Check storage format From Desc fail."), return false);
    } else {
        OP_CHECK_IF(filter_ori_format != ge::FORMAT_NDHWC && filter_ori_format != ge::FORMAT_NCDHW &&
                        filter_ori_format != ge::FORMAT_DHWCN,
                    OP_LOGE_FOR_INVALID_FORMAT(op_name, "filter",
                                               ge::TypeUtils::FormatToSerialString(filter_ori_format).c_str(),
                                               "NDHWC or NCDHW or DHWCN"),
                    return false);
        auto out_backprop_format = static_cast<ge::Format>(ge::GetPrimaryFormat(out_backprop_desc->GetStorageFormat()));
        auto filter_format = static_cast<ge::Format>(ge::GetPrimaryFormat(filter_desc->GetStorageFormat()));
        auto y_format = static_cast<ge::Format>(ge::GetPrimaryFormat(y_desc->GetStorageFormat()));
        OP_CHECK_IF(out_backprop_format != ge::FORMAT_NDC1HWC0 || filter_format != ge::FORMAT_FRACTAL_Z_3D,
                    OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(
                        op_name, "out_backprop and filter",
                        (ge::TypeUtils::FormatToSerialString(out_backprop_format) + " and " +
                         ge::TypeUtils::FormatToSerialString(filter_format))
                            .c_str(),
                        "The formats of out_backprop and filter must be NDC1HWC0 and FRACTAL_Z_3D"),
                    return false);
        if (!isV2Impl || op_type == optiling::OpTypeV2::kConv3DTransposeV2 ||
            op_type == optiling::OpTypeV2::kExtendConvTranspose) {
            OP_CHECK_IF(y_format != ge::FORMAT_NDC1HWC0,
                        OP_LOGE_FOR_INVALID_FORMAT(op_name, "y", ge::TypeUtils::FormatToSerialString(y_format).c_str(),
                                                   "NDC1HWC0"),
                        return false);
        } else {
            OP_CHECK_IF(y_format != ge::FORMAT_NDC1HWC0 && y_format != ge::FORMAT_NCDHW,
                        OP_LOGE_FOR_INVALID_FORMAT(op_name, "y", ge::TypeUtils::FormatToSerialString(y_format).c_str(),
                                                   "NDC1HWC0 or NCDHW"),
                        return false);
        }
    }

    OP_CHECK_IF(!CalShapeInfoFromDesc(context, filter_input_index, out_backprop_input_index, runInfoV2, otherParams),
                OP_LOGE(op_name, "Cal Shape Info From Desc fail."), return false);
    return true;
}

void ReCalDilation(const gert::TilingContext* context, Conv3dBpInputV2RunInfo& runInfoV2,
                   const OtherParams& otherParams)
{
    // if kernelD/H/W is equal to 1, dilationD/H/W should be set to 1
    if (otherParams.b_shape.d == 1) {
        OP_LOGD(context, "kernel_d is equal to 1, dilation_d will be set to 1");
        runInfoV2.dilation_d = 1;
    }

    if (otherParams.b_shape.h == 1) {
        OP_LOGD(context, "kernel_h is equal to 1, dilation_h will be set to 1");
        runInfoV2.dilation_h = 1;
    }

    if (otherParams.b_shape.w == 1) {
        OP_LOGD(context, "kernel_w is equal to 1, dilation_w will be set to 1");
        runInfoV2.dilation_w = 1;
    }
}

bool CalGroups(gert::TilingContext* context, OtherParams& otherParams, Conv3dBpInputV2RunInfo& runInfoV2)
{
    if (otherParams.b_shape.c == 0 || otherParams.c_shape.c % otherParams.b_shape.c != 0) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            context->GetNodeName(), "fmap_channel", std::to_string(otherParams.c_shape.c).c_str(),
            FormatString("The channel of fmap must be exactly divisible by filter_channel %ld", otherParams.b_shape.c)
                .c_str());
        return false;
    }

    int32_t groups = otherParams.c_shape.c / otherParams.b_shape.c;
    if (runInfoV2.groups == 1) {
        runInfoV2.groups = groups;
    } else if (groups != runInfoV2.groups) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context->GetNodeName(), "fmap_channel and filter_channel",
            (std::to_string(otherParams.c_shape.c) + " and " + std::to_string(otherParams.b_shape.c)).c_str(),
            FormatString("The channel of fmap must be equal to filter_channel multiplied by group %ld",
                         runInfoV2.groups)
                .c_str());
        return false;
    }

    OP_CHECK_IF(
        otherParams.a_shape.c % runInfoV2.groups != 0,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            context->GetNodeName(), "out_backprop's C", std::to_string(otherParams.a_shape.c).c_str(),
            FormatString("The C of out_backprop must be exactly divisible by group %ld", runInfoV2.groups).c_str()),
        return false);

    if (IsArchAfter35(context) || IsSocVersionFuse(context)) {
        bool invalidFilterFormat = runInfoV2.filterFormat != ge::FORMAT_NCDHW &&
                                   runInfoV2.filterFormat != ge::FORMAT_NDHWC &&
                                   runInfoV2.filterFormat != ge::FORMAT_DHWCN &&
                                   runInfoV2.filterFormat != ge::FORMAT_FRACTAL_Z;
        bool invalidOutBackpropFormat = runInfoV2.outBackpropFormat != ge::FORMAT_NCDHW &&
                                        runInfoV2.outBackpropFormat != ge::FORMAT_NDHWC;
        bool invalidYFormat = runInfoV2.yFormat != ge::FORMAT_NCDHW && runInfoV2.yFormat != ge::FORMAT_NDHWC;
        bool invalidFormat = invalidOutBackpropFormat || invalidFilterFormat || invalidYFormat;
        OP_CHECK_IF(runInfoV2.groups != 1 && invalidFormat,
                    CUBE_INNER_ERR_REPORT(context->GetNodeName(),
                                          "When groups(%d) > 1, out_backprop_format[%s] is limited to NCDHW/NDHWC, "
                                          "y_format[%s] is limited to NCDHW/NDHWC, "
                                          "filter_format[%s] are limited to NCDHW/NDHWC/DHWCN.",
                                          runInfoV2.groups,
                                          ge::TypeUtils::FormatToSerialString(runInfoV2.outBackpropFormat).c_str(),
                                          ge::TypeUtils::FormatToSerialString(runInfoV2.yFormat).c_str(),
                                          ge::TypeUtils::FormatToSerialString(runInfoV2.filterFormat).c_str()),
                    return false);
    }
    return true;
}

template <class T>
bool CheckAllZero(const T* tensor_data, size_t dim_size)
{
    if (tensor_data == nullptr) {
        // 获取不到data的场景，非onnx模型，input_size一定非零
        return false;
    }
    for (size_t idx = 0; idx < dim_size; ++idx) {
        if (tensor_data[idx] != 0) {
            return false;
        }
    }
    return true;
}

bool CheckInputSizeAllZero(const gert::TilingContext* context, bool& allzero)
{
    auto input_size = context->GetInputTensor(INPUT_SIZE_INDEX);
    OP_CHECK_IF(input_size == nullptr, CUBE_INNER_ERR_REPORT(context->GetNodeName(), "get input size fail"),
                return false);
    size_t input_size_dim_num = static_cast<size_t>(input_size->GetOriginShape().GetShapeSize());
    OP_CHECK_IF(input_size_dim_num != K_ORI_SHAPE_DIM_3D && input_size_dim_num != K_ORI_SHAPE_DIM_2D,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "input_size",
                                                         std::to_string(input_size_dim_num).c_str(),
                                                         "The shape dim of input size must be within the range {4, 5}"),
                return false);
    auto dtype = context->GetInputDesc(INPUT_SIZE_INDEX)->GetDataType();
    if (dtype == ge::DT_INT32) {
        auto tensor_data = input_size->GetData<int32_t>();
        allzero = CheckAllZero(tensor_data, input_size_dim_num);
    } else if (dtype == ge::DT_INT64) {
        auto tensor_data = input_size->GetData<int64_t>();
        allzero = CheckAllZero(tensor_data, input_size_dim_num);
    } else {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "input_size",
                                              ge::TypeUtils::DataTypeToSerialString(dtype).c_str(),
                                              "The dtype of input_size must be within the range {DT_INT32, DT_INT64}");
        return false;
    }
    return true;
}

/*
 * 该函数对 Conv3DTranspose 的处理包括两部分:
 * 1. 将 output_padding 属性计入 filter_[x]_dilation 统一处理,
 *    从 PyTorch 对 torch.nn.ConvTranspose3d 的介绍可知，这种处理在形式上是成立的
 * 2. ONNX 模型中的 Conv3DTranspose 算子 input_size 全为零，因此需要在此计算 c_shape
 */
bool HandleConv3DTranspose(gert::TilingContext* context, const Conv3dBpInputV2RunInfo& runInfoV2,
                           OtherParams& otherParams)
{
    otherParams.filter_d_dilation += otherParams.output_padding.output_padding_d;
    otherParams.filter_h_dilation += otherParams.output_padding.output_padding_h;
    otherParams.filter_w_dilation += otherParams.output_padding.output_padding_w;

    bool all_zero = false;
    if (CheckInputSizeAllZero(context, all_zero) && all_zero) {
        int64_t standard_d = runInfoV2.stride_d * (otherParams.a_shape.d - 1) +
                             ((otherParams.b_shape.d - 1) * runInfoV2.dilation_d + 1);
        otherParams.c_shape.d = standard_d + otherParams.output_padding.output_padding_d - runInfoV2.pad_h -
                                runInfoV2.pad_t;

        int64_t standard_h = runInfoV2.stride_h * (otherParams.a_shape.h - 1) +
                             ((otherParams.b_shape.h - 1) * runInfoV2.dilation_h + 1);
        otherParams.c_shape.h = standard_h + otherParams.output_padding.output_padding_h - runInfoV2.pad_u -
                                runInfoV2.pad_d;

        int64_t standard_w = runInfoV2.stride_w * (otherParams.a_shape.w - 1) +
                             ((otherParams.b_shape.w - 1) * runInfoV2.dilation_w + 1);
        otherParams.c_shape.w = standard_w + otherParams.output_padding.output_padding_w - runInfoV2.pad_l -
                                runInfoV2.pad_r;
        otherParams.c_shape.batch = otherParams.a_shape.batch;
        otherParams.c_shape.c = otherParams.b_shape.c;
        otherParams.c_shape.c1 = Ops::Base::CeilDiv(otherParams.c_shape.c, otherParams.c_shape.c0);
    }
    return true;
}

bool CheckCalPads(const gert::TilingContext* context, const Conv3dBpInputV2RunInfo& runInfoV2,
                  optiling::OpTypeV2 op_type, const OtherParams& otherParams)
{
    int64_t do_expect = (otherParams.c_shape.d + runInfoV2.pad_h + runInfoV2.pad_t - otherParams.filter_d_dilation) /
                            runInfoV2.stride_d +
                        1;
    int64_t ho_expect = (otherParams.c_shape.h + runInfoV2.pad_u + runInfoV2.pad_d - otherParams.filter_h_dilation) /
                            runInfoV2.stride_h +
                        1;
    int64_t wo_expect = (otherParams.c_shape.w + runInfoV2.pad_l + runInfoV2.pad_r - otherParams.filter_w_dilation) /
                            runInfoV2.stride_w +
                        1;
    std::string check_input_name = (op_type == optiling::OpTypeV2::kConv3DTransposeV2 ||
                                    op_type == optiling::OpTypeV2::kExtendConvTranspose) ?
                                       "x" :
                                       "out_backprop";
    OP_CHECK_IF(
        do_expect != otherParams.a_shape.d,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            context->GetNodeName(), (check_input_name + "'s D").c_str(), std::to_string(otherParams.a_shape.d).c_str(),
            FormatString("The value of %s's D must be equal to inferred D = %d", check_input_name.c_str(), do_expect)
                .c_str()),
        return false);
    OP_CHECK_IF(
        ho_expect != otherParams.a_shape.h,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            context->GetNodeName(), (check_input_name + "'s H").c_str(), std::to_string(otherParams.a_shape.h).c_str(),
            FormatString("The value of %s's H must be equal to inferred H = %d", check_input_name.c_str(), ho_expect)
                .c_str()),
        return false);
    OP_CHECK_IF(
        wo_expect != otherParams.a_shape.w,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            context->GetNodeName(), (check_input_name + "'s W").c_str(), std::to_string(otherParams.a_shape.w).c_str(),
            FormatString("The value of %s's W must be equal to inferred W = %d", check_input_name.c_str(), wo_expect)
                .c_str()),
        return false);
    return true;
}

bool CalPads(gert::TilingContext* context, Conv3dBpInputV2RunInfo& runInfoV2, optiling::OpTypeV2 op_type,
             OtherParams& otherParams)
{
    auto attrs = context->GetAttrs();
    size_t padding_attr_idx = kPaddingConv3dBpInputIdx;
    if (op_type == optiling::OpTypeV2::kConv3DTransposeV2 || op_type == optiling::OpTypeV2::kExtendConvTranspose) {
        if (IsSocVersionFuse(context)) {
            padding_attr_idx = kPaddingExtendConvTransposeIdx;
        } else {
            padding_attr_idx = kPaddingConv3dTransposeIdx;
        }
    }
    if (attrs->GetAttrNum() <= padding_attr_idx) {
        OP_LOGD(context, "no padding attr, skip calc and check");
        otherParams.filter_d_dilation += otherParams.output_padding.output_padding_d;
        otherParams.filter_h_dilation += otherParams.output_padding.output_padding_h;
        otherParams.filter_w_dilation += otherParams.output_padding.output_padding_w;
        return true;
    }

    auto padding = attrs->GetAttrPointer<char>(padding_attr_idx);
    if (padding != nullptr && (padding[0] == 'S')) {
        int32_t pad_d = std::max(Ops::Base::CeilAlign(otherParams.c_shape.d, static_cast<int64_t>(runInfoV2.stride_d)) -
                                     runInfoV2.stride_d + otherParams.filter_d_dilation - otherParams.c_shape.d,
                                 0L);
        int32_t pad_head = static_cast<int32_t>((static_cast<uint32_t>(pad_d) >> 1U));
        int32_t pad_tail = pad_d - pad_head;
        int32_t pad_h = std::max(Ops::Base::CeilAlign(otherParams.c_shape.h, static_cast<int64_t>(runInfoV2.stride_h)) -
                                     runInfoV2.stride_h + otherParams.filter_h_dilation - otherParams.c_shape.h,
                                 0L);
        int32_t pad_up = static_cast<int32_t>((static_cast<uint32_t>(pad_h) >> 1U));
        int32_t pad_down = pad_h - pad_up;
        int32_t pad_w = std::max(Ops::Base::CeilAlign(otherParams.c_shape.w, static_cast<int64_t>(runInfoV2.stride_w)) -
                                     runInfoV2.stride_w + otherParams.filter_w_dilation - otherParams.c_shape.w,
                                 0L);
        int32_t pad_left = static_cast<int32_t>((static_cast<uint32_t>(pad_w) >> 1U));
        int32_t pad_right = pad_w - pad_left;

        runInfoV2.pad_h = pad_head;
        runInfoV2.pad_t = pad_tail;
        runInfoV2.pad_u = pad_up;
        runInfoV2.pad_d = pad_down;
        runInfoV2.pad_l = pad_left;
        runInfoV2.pad_r = pad_right;
    }

    if (op_type == optiling::OpTypeV2::kConv3DTransposeV2 || op_type == optiling::OpTypeV2::kExtendConvTranspose) {
        OP_CHECK_IF(!HandleConv3DTranspose(context, runInfoV2, otherParams),
                    OP_LOGE(context, "Failed to process Conv3DTranspose."), return false);
    }

    return CheckCalPads(context, runInfoV2, op_type, otherParams);
}

int32_t CalFmapH(const gert::TilingContext* context, const Conv3dBpInputV2RunInfo& runInfoV2,
                 const OtherParams& otherParams)
{
    int32_t minBaseM = 1024;

    int64_t dedx_w = otherParams.c_shape.w;
    OP_CHECK_IF(dedx_w == 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "dedx_w", std::to_string(dedx_w).c_str(),
                                                      "The value of dedx_w cannot be 0"),
                return 0);

    constexpr int32_t FMAP_H_NUM = 2;
    int32_t hiCal;

    if (minBaseM % dedx_w == 0 || dedx_w % minBaseM == 0) {
        hiCal = Ops::Base::CeilDiv(static_cast<int64_t>(minBaseM), dedx_w);
    } else if (minBaseM > dedx_w) {
        hiCal = minBaseM / dedx_w + FMAP_H_NUM;
    } else {
        hiCal = FMAP_H_NUM;
    }
    // L1HKhk=1dilation
    int32_t khDilation = (otherParams.b_shape.h - 1) * runInfoV2.dilation_h + 1;
    int32_t hoCal = (hiCal - 1) + khDilation;
    int64_t hoExpand = static_cast<int64_t>(otherParams.a_shape.h - 1) * runInfoV2.stride_h + 1;
    return static_cast<int32_t>(std::min(static_cast<int64_t>(hoCal), hoExpand));
}

bool IsNeedTilingHkWk(gert::TilingContext* context, const Conv3dBpInputV2RunInfo& runInfoV2,
                      const OtherParams& otherParams)
{
    uint32_t minBaseN = 16;
    uint32_t dtypeByteL0a_ = runInfoV2.a_dtype_bytes;
    uint32_t dtypeByteL0b_ = runInfoV2.b_dtype_bytes;
    uint32_t blockSize_ = BYTE_BLOCK / dtypeByteL0b_;
    uint64_t l1_size = 0;

    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, CUBE_INNER_ERR_REPORT(context->GetNodeName(), "platformInfoPtr is null."),
                return false);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1_size);

    int64_t bpPadRight = otherParams.c_shape.w -
                         (static_cast<int64_t>(otherParams.a_shape.w - 1) * runInfoV2.stride_w + 1) +
                         (otherParams.b_shape.w - 1) * runInfoV2.dilation_w - runInfoV2.backprop_pad_l;
    int32_t curHo = CalFmapH(context, runInfoV2, otherParams);
    uint64_t minA1Size = static_cast<uint64_t>(dtypeByteL0a_) * curHo * otherParams.a_shape.w * runInfoV2.stride_w *
                         blockSize_;
    uint64_t lenHkWkC0 = otherParams.b_shape.h * otherParams.b_shape.w * blockSize_;
    uint64_t minB1Size = static_cast<uint64_t>(dtypeByteL0b_) * lenHkWkC0 * minBaseN;

    if ((minA1Size + minB1Size) <= l1_size && runInfoV2.backprop_pad_l <= PAD_DIM_UP && bpPadRight <= PAD_DIM_UP &&
        runInfoV2.backprop_pad_u <= PAD_DIM_UP && runInfoV2.backprop_pad_d <= PAD_DIM_UP &&
        runInfoV2.dilation_h <= PAD_DIM_UP && runInfoV2.dilation_w <= PAD_DIM_UP) {
        return false;
    }

    OP_LOGD(context, "need tiling hw wk");
    return true;
}

bool CalRealG(gert::TilingContext* context, Conv3dBpInputV2RunInfo& runInfoV2, OtherParams& otherParams)
{
    // calc real g and check shape
    int32_t dy_c_ori = otherParams.a_shape.c / runInfoV2.groups;
    OP_CHECK_IF(
        dy_c_ori == 0,
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            context->GetNodeName(), "out_backporp C and group",
            (std::to_string(otherParams.a_shape.c) + " and " + std::to_string(runInfoV2.groups)).c_str(),
            FormatString("The value of out_backporp of C must be at least group(%d)", runInfoV2.groups).c_str()),
        return false);
    int32_t dx_c_extend = MathUtil::Lcm(otherParams.b_shape.c, otherParams.c_shape.c0) / otherParams.b_shape.c;
    int32_t dy_c_extend = MathUtil::Lcm(dy_c_ori, kBlockSize) / dy_c_ori;
    otherParams.multiple_extend = std::min(MathUtil::Lcm(dx_c_extend, dy_c_extend),
                                           static_cast<int64_t>(runInfoV2.groups));
    runInfoV2.real_g = (static_cast<int64_t>(runInfoV2.groups) + otherParams.multiple_extend - 1) /
                       otherParams.multiple_extend;
    otherParams.ci1g = Ops::Base::CeilDiv(otherParams.multiple_extend * otherParams.b_shape.c, otherParams.c_shape.c0);
    int32_t co1g = (otherParams.multiple_extend * dy_c_ori + kBlockSize - 1) / kBlockSize;
    if (context->GetOutputDesc(Y_INDEX)->GetDataType() == ge::DT_FLOAT && runInfoV2.groups > 1) {
        co1g *= 2; // 2: BLOCK_NUM / FP32_C0
    }

    if (IsArchAfter35(context) || IsSocVersionFuse(context)) {
        otherParams.ci1g = Ops::Base::CeilDiv(otherParams.multiple_extend * otherParams.b_shape.c,
                                              static_cast<int64_t>(kBlockSize));
        otherParams.co1g = Ops::Base::CeilDiv(
            otherParams.multiple_extend * otherParams.b_shape.batch / runInfoV2.groups, otherParams.b_shape.c0);

        size_t filter_input_idx = static_cast<size_t>(FILTER_INDEX);
        size_t out_backprop_input_idx = static_cast<size_t>(OUT_BACKPROP_INDEX);
        const auto out_backprop_desc = context->GetInputDesc(out_backprop_input_idx);
        const auto filter_desc = context->GetInputDesc(filter_input_idx);

        constexpr uint32_t ENLARGE_BUFFER_NUM = 2;
        constexpr uint32_t REG_SIZE = 256;
        bool disableGroupEnlarge = static_cast<uint64_t>(ENLARGE_BUFFER_NUM) *
                                           (otherParams.co1g * otherParams.b_shape.h * otherParams.b_shape.w *
                                            otherParams.ci1g * kBlockSize * otherParams.b_shape.c0) *
                                           runInfoV2.b_dtype_bytes +
                                       REG_SIZE >
                                   context->GetCompileInfo<Ops::NN::Conv::Conv3DBackpropV2CompileInfo>()->ub_size;

        bool nonExtendedDtype = (filter_desc->GetDataType() == ge::DT_FLOAT8_E4M3FN ||
                                 filter_desc->GetDataType() == ge::DT_HIFLOAT8 ||
                                 filter_desc->GetDataType() == ge::DT_INT8) ||
                                (out_backprop_desc->GetDataType() == ge::DT_FLOAT8_E4M3FN ||
                                 out_backprop_desc->GetDataType() == ge::DT_HIFLOAT8 ||
                                 out_backprop_desc->GetDataType() == ge::DT_INT8);
        if (disableGroupEnlarge || nonExtendedDtype || IsNeedTilingHkWk(context, runInfoV2, otherParams)) {
            otherParams.multiple_extend = 1;
            runInfoV2.real_g = runInfoV2.groups;
            otherParams.ci1g = Ops::Base::CeilDiv(otherParams.b_shape.c, static_cast<int64_t>(kBlockSize));
            otherParams.co1g = Ops::Base::CeilDiv(otherParams.b_shape.batch / runInfoV2.groups, otherParams.b_shape.c0);
        }
    }

    if (!IsArchAfter35(context) && !IsSocVersionFuse(context)) {
        int64_t filterGDkCi1gHW = static_cast<int64_t>(runInfoV2.real_g) * otherParams.b_shape.d * otherParams.ci1g *
                                  otherParams.b_shape.h * otherParams.b_shape.w;

        OP_CHECK_IF(filterGDkCi1gHW != otherParams.filter_gdkci1ghw,
                    OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "the 1st dim of filter shape",
                                              std::to_string(otherParams.filter_gdkci1ghw).c_str(),
                                              std::to_string(filterGDkCi1gHW).c_str()),
                    return false);
        OP_CHECK_IF(co1g != otherParams.co1g,
                    OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "the 2nd dim of filter shape",
                                              std::to_string(otherParams.co1g).c_str(), std::to_string(co1g).c_str()),
                    return false);
    }
    return true;
}

int32_t CalBackpropPadBefore(int32_t filter, int32_t dilation, int32_t pad) { return (filter - 1) * dilation - pad; }

int64_t CalBackpropPadAfter(int64_t inputDim, int64_t outputDim, int32_t stride, int32_t pad)
{
    // orginal formula is inputDim = (outputDim * stride + 1) - padBefore + filterDilation, it can be simplified as
    // follow.
    return inputDim - outputDim * stride + pad;
}

bool IsOverflowInt32(int64_t value)
{
    if (value > INT32_MAX || value < INT32_MIN) {
        return true;
    }
    return false;
}

bool CheckRange(int32_t value, int32_t value_low, int32_t value_up)
{
    if (value < value_low || value > value_up) {
        return false;
    }
    return true;
}

bool CalModifyBackpropPadD(const gert::TilingContext* context, Conv3dBpInputV2RunInfo& runInfoV2,
                           OtherParams& otherParams)
{
    Shape& dedyShape = otherParams.a_shape;
    Shape& filterShape = otherParams.b_shape;
    Shape& dedxShape = otherParams.c_shape;

    otherParams.pad_head_before = CalBackpropPadBefore(filterShape.d, runInfoV2.dilation_d, runInfoV2.pad_h);
    int64_t pad_tail_after = CalBackpropPadAfter(dedxShape.d, dedyShape.d, runInfoV2.stride_d, runInfoV2.pad_h);
    OP_CHECK_IF(IsOverflowInt32(pad_tail_after) || !CheckRange(static_cast<int32_t>(pad_tail_after), -kDimUp, kDimUp),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context->GetNodeName(), "pad_tail_after = (inputD - outputD * strideD + padHead)",
                    std::to_string(pad_tail_after).c_str(),
                    FormatString("The value of pad_tail_after must be range [%d, %d]", -kDimUp, kDimUp).c_str()),
                return false);
    pad_tail_after = (pad_tail_after + abs(pad_tail_after)) / kNumTwo;
    otherParams.pad_tail_after = pad_tail_after;
    runInfoV2.backprop_pad_h = otherParams.pad_head_before;
    runInfoV2.backprop_pad_t = otherParams.pad_tail_after;
    return true;
}

bool CalModifyBackpropPadHW(gert::TilingContext* context, Conv3dBpInputV2RunInfo& runInfoV2, OtherParams& otherParams)
{
    Shape& dedyShape = otherParams.a_shape;
    Shape& filterShape = otherParams.b_shape;
    Shape& dedxShape = otherParams.c_shape;

    otherParams.pad_left_before = CalBackpropPadBefore(filterShape.w, runInfoV2.dilation_w, runInfoV2.pad_l);
    otherParams.pad_up_before = CalBackpropPadBefore(filterShape.h, runInfoV2.dilation_h, runInfoV2.pad_u);

    otherParams.shape_left_modify = (otherParams.pad_left_before - abs(otherParams.pad_left_before)) / kNumTwo;
    otherParams.shape_up_modify = (otherParams.pad_up_before - abs(otherParams.pad_up_before)) / kNumTwo;

    int64_t pad_right_after = CalBackpropPadAfter(dedxShape.w, dedyShape.w, runInfoV2.stride_w, runInfoV2.pad_l);
    int64_t pad_down_after;
    if (IsArchAfter35(context) || IsSocVersionFuse(context)) {
        otherParams.pad_head_before = CalBackpropPadBefore(filterShape.d, runInfoV2.dilation_d, runInfoV2.pad_h);
        pad_down_after = dedxShape.h - (static_cast<int64_t>(dedyShape.h - 1) * runInfoV2.stride_h + 1) +
                         (filterShape.h - 1) * runInfoV2.dilation_h - otherParams.pad_up_before;
    } else {
        pad_down_after = CalBackpropPadAfter(dedxShape.h, dedyShape.h, runInfoV2.stride_h, runInfoV2.pad_u);
    }

    int64_t shape_down_modify = (pad_down_after - abs(pad_down_after)) / kNumTwo;
    int64_t shape_right_modify = (pad_right_after - abs(pad_right_after)) / kNumTwo;

    otherParams.pad_right_after = pad_right_after;
    otherParams.pad_down_after = pad_down_after;
    otherParams.shape_right_modify = shape_right_modify;
    otherParams.shape_down_modify = shape_down_modify;

    runInfoV2.backprop_pad_u = otherParams.pad_up_before;
    runInfoV2.backprop_pad_d = otherParams.pad_down_after;
    runInfoV2.backprop_pad_l = otherParams.pad_left_before;
    runInfoV2.backprop_pad_r = otherParams.pad_right_after;
    return true;
}

bool CalModify(gert::TilingContext* context, Conv3dBpInputV2RunInfo& runInfoV2, OtherParams& otherParams)
{
    OP_CHECK_IF(!CalModifyBackpropPadD(context, runInfoV2, otherParams),
                OP_LOGE(context, "Calculate backprop pad d invalid"), return false);
    OP_CHECK_IF(!CalModifyBackpropPadHW(context, runInfoV2, otherParams),
                OP_LOGE(context, "Calculate backprop pad h,w invalid"), return false);
    return true;
}

bool CheckLowerBound(int32_t value, int32_t value_low) { return value >= value_low; }

bool CheckValue(int32_t value, int32_t value_temp) { return value == value_temp; }

bool CheckPadParamsWithLog(const Conv3dBpInputV2RunInfo& runInfoV2, const gert::TilingContext* context)
{
    const auto op_name = context->GetNodeName();
    int32_t kPadUpTmp = kDimUp;
    if (IsSocVersionFuse(context)) {
        kPadUpTmp = kPadUp;
    }
    OP_CHECK_IF(!CheckRange(runInfoV2.pad_h, 0, kPadUpTmp),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    op_name, "pad_h", std::to_string(runInfoV2.pad_h).c_str(),
                    FormatString("The value of pad_h must be range [%d, %d]", 0, kPadUpTmp).c_str()),
                return false);
    OP_CHECK_IF(!CheckRange(runInfoV2.pad_t, 0, kPadUpTmp),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    op_name, "pad_t", std::to_string(runInfoV2.pad_t).c_str(),
                    FormatString("The value of pad_t must be range [%d, %d]", 0, kPadUpTmp).c_str()),
                return false);
    OP_CHECK_IF(!CheckRange(runInfoV2.pad_u, 0, kPadUpTmp),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    op_name, "pad_u", std::to_string(runInfoV2.pad_u).c_str(),
                    FormatString("The value of pad_u must be range [%d, %d]", 0, kPadUpTmp).c_str()),
                return false);
    OP_CHECK_IF(!CheckRange(runInfoV2.pad_d, 0, kPadUpTmp),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    op_name, "pad_d", std::to_string(runInfoV2.pad_d).c_str(),
                    FormatString("The value of pad_d must be range [%d, %d]", 0, kPadUpTmp).c_str()),
                return false);
    OP_CHECK_IF(!CheckRange(runInfoV2.pad_l, 0, kPadUpTmp),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    op_name, "pad_l", std::to_string(runInfoV2.pad_l).c_str(),
                    FormatString("The value of pad_l must be range [%d, %d]", 0, kPadUpTmp).c_str()),
                return false);
    OP_CHECK_IF(!CheckRange(runInfoV2.pad_r, 0, kPadUpTmp),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    op_name, "pad_r", std::to_string(runInfoV2.pad_r).c_str(),
                    FormatString("The value of pad_r must be range [%d, %d]", 0, kPadUpTmp).c_str()),
                return false);
    return true;
}

bool CheckParamsWithLog(Conv3dBpInputV2RunInfo& runInfoV2, gert::TilingContext* context, OtherParams& otherParams)
{
    const auto op_name = context->GetNodeName();
    int32_t kStrideHWUpTmp = kDimUp;
    int32_t kStrideDUpTmp = kDimUp;
    int32_t kDilationUpTmp = kDilationUp;
    if (IsArchAfter35(context)) {
        kDilationUpTmp = kDimUp;
    }
    if (IsSocVersionFuse(context)) {
        kStrideHWUpTmp = kStrideHWUp;
    }
    OP_CHECK_IF(
        !CheckRange(runInfoV2.dilation_h, kDilationLow, kDilationUpTmp),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            op_name, "dilation_h", std::to_string(runInfoV2.dilation_h).c_str(),
            FormatString("The value of dilation_h must be range [%d, %d]", kDilationLow, kDilationUpTmp).c_str()),
        return false);
    OP_CHECK_IF(
        !CheckRange(runInfoV2.dilation_w, kDilationLow, kDilationUpTmp),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            op_name, "dilation_w", std::to_string(runInfoV2.dilation_w).c_str(),
            FormatString("The value of dilation_w must be range [%d, %d]", kDilationLow, kDilationUpTmp).c_str()),
        return false);
    OP_CHECK_IF(
        !CheckRange(runInfoV2.dilation_d, kDilationLow, kDilationUpTmp),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            op_name, "dilation_d", std::to_string(runInfoV2.dilation_d).c_str(),
            FormatString("The value of dilation_d must be range [%d, %d]", kDilationLow, kDilationUpTmp).c_str()),
        return false);
    OP_CHECK_IF(!CheckRange(runInfoV2.stride_h, kDimLow, kStrideHWUpTmp),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    op_name, "stride_h", std::to_string(runInfoV2.stride_h).c_str(),
                    FormatString("The value of stride_h must be range [%d, %d]", kDimLow, kStrideHWUpTmp).c_str()),
                return false);
    OP_CHECK_IF(!CheckRange(runInfoV2.stride_w, kDimLow, kStrideHWUpTmp),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    op_name, "stride_w", std::to_string(runInfoV2.stride_w).c_str(),
                    FormatString("The value of stride_w must be range [%d, %d]", kDimLow, kStrideHWUpTmp).c_str()),
                return false);
    OP_CHECK_IF(!CheckRange(runInfoV2.stride_d, kDimLow, kStrideDUpTmp),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    op_name, "stride_d", std::to_string(runInfoV2.stride_d).c_str(),
                    FormatString("The value of stride_d must be range [%d, %d]", kDimLow, kStrideDUpTmp).c_str()),
                return false);
    OP_CHECK_IF(!CheckPadParamsWithLog(runInfoV2, context), OP_LOGE(op_name, "check pad params failed"), return false);
    OP_CHECK_IF(
        (otherParams.filter_d_dilation > otherParams.fmap_d_padding),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            op_name, "filter_d_dilation = ((filter_d - 1) * dilation_d + 1)",
            std::to_string(otherParams.filter_d_dilation).c_str(),
            FormatString("The value of filter_d_dilation must be less than or equal to (fmap_d + pad_h + pad_t)=[%ld]",
                         otherParams.fmap_d_padding)
                .c_str()),
        return false);
    OP_CHECK_IF(
        (otherParams.filter_h_dilation > otherParams.fmap_h_padding),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            op_name, "filter_h_dilation = ((filter_h - 1) * dilation_h + 1)",
            std::to_string(otherParams.filter_h_dilation).c_str(),
            FormatString("The value of filter_h_dilation must be less than or equal to (fmap_h + pad_u + pad_d)=[%ld]",
                         otherParams.fmap_h_padding)
                .c_str()),
        return false);
    OP_CHECK_IF(
        (otherParams.filter_w_dilation > otherParams.fmap_w_padding),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            op_name, "filter_w_dilation = ((filter_w - 1) * dilation_w + 1)",
            std::to_string(otherParams.filter_w_dilation).c_str(),
            FormatString("The value of filter_w_dilation must be less than or equal to (fmap_w + pad_l + pad_r)=[%ld]",
                         otherParams.fmap_w_padding)
                .c_str()),
        return false);
    OP_CHECK_IF(!CheckRange(otherParams.a_shape.w * runInfoV2.stride_w, kDimLow, kDimUp),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    op_name, "out_backprop's W after expands",
                    std::to_string(otherParams.a_shape.w * runInfoV2.stride_w).c_str(),
                    FormatString("The value of out_backprop's W after expands must be range [%d, %d]", kDimLow, kDimUp)
                        .c_str()),
                return false);
    OP_CHECK_IF(!CheckRange(otherParams.a_shape.h * runInfoV2.stride_h, kDimLow, kDimUp),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    op_name, "out_backprop's H after expands",
                    std::to_string(otherParams.a_shape.h * runInfoV2.stride_h).c_str(),
                    FormatString("The value of out_backprop's H after expands must be range [%d, %d]", kDimLow, kDimUp)
                        .c_str()),
                return false);
    return true;
}

int64_t GetDfactorSdEqKd(const Conv3dBpInputV2RunInfo& runInfoV2, int32_t l0c_din, const OtherParams& otherParams)
{
    // ----[该函数用于计算StrideD=KerneD时 由Din和Dk 反推的Dout的大小]----
    if (otherParams.filter_d_dilation == 0 || l0c_din == 0) {
        return 1;
    }
    int64_t kernel_idx = runInfoV2.pad_h % otherParams.filter_d_dilation - 1;
    int64_t dedy_dout_used = 1;
    int64_t dedy_dout_used_max = 1;
    for (int dedx_idx = 0; dedx_idx < otherParams.c_shape.d; ++dedx_idx) {
        kernel_idx += 1;
        if (kernel_idx == otherParams.filter_d_dilation) {
            kernel_idx = 0;
            dedy_dout_used += 1;
        }
        if (dedx_idx % l0c_din == 0) {
            dedy_dout_used = 1;
        }
        if (dedy_dout_used > dedy_dout_used_max) {
            dedy_dout_used_max = dedy_dout_used;
        }
    }
    return dedy_dout_used_max;
}

template <typename T>
int64_t GetDfactor(T kd_factor, Conv3dBpInputV2RunInfo& runInfoV2, int32_t l0c_din, gert::TilingContext* context,
                   OtherParams& otherParams)
{
    int64_t estimate_d = static_cast<int64_t>(
        Ops::Base::CeilDiv(static_cast<int64_t>(kd_factor - 2 + l0c_din), static_cast<int64_t>(runInfoV2.stride_d)) +
        1);
    int64_t dout_factor = std::min(estimate_d, otherParams.a_shape.d);
    if (otherParams.filter_d_dilation == runInfoV2.stride_d) {
        dout_factor = (context->GetCompileInfo<Ops::NN::Conv::Conv3DBackpropV2CompileInfo>()
                           ->intrinsic_fix_pipe_l0c2out) ?
                          GetDfactorSdEqKd(runInfoV2, l0c_din, otherParams) :
                          std::max(dout_factor - 1, static_cast<int64_t>(1));
    };
    return dout_factor;
}

bool CheckL1SizeLimit(Conv3dBpInputV2RunInfo& runInfoV2, gert::TilingContext* context, OtherParams& otherParams)
{
    if (IsArchAfter35(context) || IsSocVersionFuse(context)) {
        return true;
    }
    int64_t w_value = otherParams.a_shape.w * runInfoV2.stride_w;
    int64_t h_value_max = (otherParams.filter_h_dilation - 1) + kBlockSize / otherParams.c_shape.w + 2;
    if (kBlockSize < otherParams.c_shape.w) {
        h_value_max = otherParams.filter_h_dilation + 1;
    } else if (kBlockSize % otherParams.c_shape.w == 0) {
        h_value_max = (otherParams.filter_h_dilation - 1) + kBlockSize / otherParams.c_shape.w;
    }

    h_value_max = std::min(h_value_max, otherParams.a_shape.h * runInfoV2.stride_h);

    // 计算最小载入时 b_l1_dk == 1, a_l1_d 需要根据载入情况反推
    int64_t b_l1_dk = 1;

    int64_t l1_real_dk_check = otherParams.a_shape.c1 == 1 ? otherParams.b_shape.d : 1;
    int64_t a_l1_d = GetDfactor(l1_real_dk_check, runInfoV2, 1, context, otherParams);

    int64_t a_l1_size = h_value_max * w_value * a_l1_d * otherParams.a_shape.c0 * runInfoV2.a_dtype_bytes;
    int64_t b_l1_size = b_l1_dk * otherParams.b_shape.h * otherParams.filter_co0 * otherParams.b_shape.w *
                        otherParams.filter_ci0 * runInfoV2.b_dtype_bytes;
    // 在stride_d > k_d，或者d方向需要额外补零时，v1算子需要在l1上预留一块大小为baseM*baseN的buffer，置零之后再写出去
    int64_t fill_zero_size = kBlockSize * kBlockSize * runInfoV2.b_dtype_bytes;
    if (!IsArchAfter35(context) && !IsSocVersionFuse(context)) {
        if (otherParams.b_dtype == ge::DT_FLOAT) {
            // fp32一定走v2，v2算子不需要预留buffer置零
            fill_zero_size = 0;
        }
    }

    bool w_size_limit = std::max(otherParams.c_shape.w, w_value) <= kDimWNormalUp;
    // 不支持切W
    if (a_l1_size + b_l1_size + fill_zero_size >
            static_cast<int64_t>(context->GetCompileInfo<Ops::NN::Conv::Conv3DBackpropV2CompileInfo>()->l1_size) ||
        !w_size_limit) {
        std::stringstream ss;
        ss << "Minimum load size may exceed L1 buffer, a_l1_size is " << a_l1_size << "B, b_l1_size is " << b_l1_size
           << "B, fill_zero_size is " << fill_zero_size << "B, L1size is "
           << context->GetCompileInfo<Ops::NN::Conv::Conv3DBackpropV2CompileInfo>()->l1_size
           << "B, width may exceed limits, actual width: " << std::max(otherParams.c_shape.w, w_value)
           << ", width limit: " << kDimWNormalUp;
        CUBE_INNER_ERR_REPORT(context->GetNodeName(), "Error msg: %s", ss.str().c_str());
        return false;
    }

    return true;
}

void SetConvAttrs(Conv3dBpInputV2RunInfo& runInfoV2, const int64_t* pads_data, Shape& strides_ncdhw,
                  Shape& dilations_ncdhw, const int64_t* groups, OtherParams& otherParams)
{
    size_t idx = 0;
    runInfoV2.pad_h = pads_data[idx++];
    runInfoV2.pad_t = pads_data[idx++];
    runInfoV2.pad_u = pads_data[idx++];
    runInfoV2.pad_d = pads_data[idx++];
    runInfoV2.pad_l = pads_data[idx++];
    runInfoV2.pad_r = pads_data[idx++];
    runInfoV2.stride_d = strides_ncdhw.d;
    runInfoV2.stride_h = strides_ncdhw.h;
    runInfoV2.stride_w = strides_ncdhw.w;
    runInfoV2.dilation_d = dilations_ncdhw.d;
    runInfoV2.dilation_h = dilations_ncdhw.h;
    runInfoV2.dilation_w = dilations_ncdhw.w;
    runInfoV2.groups = *groups;

    otherParams.stride_expand_flag = (runInfoV2.stride_h != 1 || runInfoV2.stride_w != 1 || runInfoV2.stride_d != 1);

    otherParams.dilation_d_gt_one_flag = dilations_ncdhw.d > 1 ? 1 : 0;
}

bool GetAttrAndDtypeParams(gert::TilingContext* context, Conv3dBpInputV2RunInfo& runInfoV2, optiling::OpTypeV2 op_type,
                           OtherParams& otherParams)
{
    auto attrs = context->GetAttrs();
    size_t idx = 0;
    const auto strides = attrs->GetAttrPointer<gert::ContinuousVector>(idx++);
    const auto pads = attrs->GetAttrPointer<gert::ContinuousVector>(idx++);
    const auto dilations = attrs->GetAttrPointer<gert::ContinuousVector>(idx++);
    const auto groups = attrs->GetAttrPointer<int64_t>(idx++);

    const auto op_name = context->GetNodeName();
    // stridesDim == 5
    OP_CHECK_IF(strides == nullptr, OP_LOGE_WITH_INVALID_ATTR(op_name, "strides", "null", "non_empty_value"),
                return false);
    OP_CHECK_IF(
        strides->GetSize() != kStridesDim,
        OP_LOGE_WITH_INVALID_ATTR(op_name, "strides", std::to_string(strides->GetSize()), std::to_string(kStridesDim)),
        return false);
    // padsDim == 6
    OP_CHECK_IF(pads == nullptr, OP_LOGE_WITH_INVALID_ATTR(op_name, "pads", "null", "non_empty_value"), return false);
    OP_CHECK_IF(pads->GetSize() != kPadsDim,
                OP_LOGE_WITH_INVALID_ATTR(op_name, "pads", std::to_string(pads->GetSize()), std::to_string(kPadsDim)),
                return false);
    // dilationsDim == 5
    OP_CHECK_IF(dilations == nullptr, OP_LOGE_WITH_INVALID_ATTR(op_name, "dilations", "null", "non_empty_value"),
                return false);
    OP_CHECK_IF(dilations->GetSize() != kDilationsDim,
                OP_LOGE_FOR_INVALID_SHAPEDIM(op_name, "dilations", std::to_string(dilations->GetSize()),
                                             std::to_string(kDilationsDim)),
                return false);
    OP_CHECK_IF(groups == nullptr, OP_LOGE_WITH_INVALID_ATTR(op_name, "groups", "null", "non_empty_value"),
                return false);

    const auto strides_data = static_cast<const int64_t*>(strides->GetData());
    const auto pads_data = static_cast<const int64_t*>(pads->GetData());
    const auto dilations_data = static_cast<const int64_t*>(dilations->GetData());
    OP_CHECK_IF(!CheckAttrRange(context, strides_data, pads_data, dilations_data, groups),
                OP_LOGW(context, "check attr range failed"), return false);

    Shape strides_ncdhw;
    Shape dilations_ncdhw;
    ge::Format data_format = context->GetInputDesc(OUT_BACKPROP_INDEX)->GetOriginFormat();
    if (op_type == optiling::OpTypeV2::kConv3DTransposeV2 || op_type == optiling::OpTypeV2::kExtendConvTranspose) {
        OP_CHECK_IF(!CheckTransposeAttr(context, otherParams), OP_LOGW(context, "check transpose attr failed"),
                    return false);
        data_format = context->GetInputDesc(FILTER_INDEX)->GetOriginFormat();
    }

    GetNCDHWShape(strides_data, strides_ncdhw, data_format);
    GetNCDHWShape(dilations_data, dilations_ncdhw, data_format);

    OP_CHECK_IF(strides_ncdhw.batch != 1,
                OP_LOGE_FOR_INVALID_VALUE(op_name, "strides N", std::to_string(strides_ncdhw.batch), "1"),
                return false);
    OP_CHECK_IF(strides_ncdhw.c != 1,
                OP_LOGE_FOR_INVALID_VALUE(op_name, "strides C", std::to_string(strides_ncdhw.c), "1"), return false);

    SetConvAttrs(runInfoV2, pads_data, strides_ncdhw, dilations_ncdhw, groups, otherParams);
    if (op_type == optiling::OpTypeV2::kConv3DTransposeV2 || op_type == optiling::OpTypeV2::kExtendConvTranspose) {
        OP_CHECK_IF(!CheckTransposeOutputdingRange(context, runInfoV2, otherParams),
                    OP_LOGW(context, "check transpose attr failed"), return false);
    }
    return UpdateDtypeParams(context, runInfoV2, op_type, otherParams);
}

bool GetInputOutputFormat(const gert::TilingContext* context, Conv3dBpInputV2RunInfo& runInfoV2,
                          const optiling::OpTypeV2 opType)
{
    size_t aMatrixIndex = OUT_BACKPROP_INDEX;
    size_t bMatrixIndex = FILTER_INDEX;
    const char* opName = context->GetNodeName(); // 日志打印用，允许为空

    if (opType == optiling::OpTypeV2::kConv3DTransposeV2 || opType == optiling::OpTypeV2::kExtendConvTranspose) {
        aMatrixIndex = FILTER_INDEX;
        bMatrixIndex = OUT_BACKPROP_INDEX;
    }

    const auto out_backprop_desc = context->GetInputDesc(aMatrixIndex);
    OP_CHECK_IF(out_backprop_desc == nullptr, CUBE_INNER_ERR_REPORT(opName, "out_backprop_desc is null"), return false);
    ge::Format outBackpropFormat = out_backprop_desc->GetStorageFormat();

    const auto filter_desc = context->GetInputDesc(bMatrixIndex);
    OP_CHECK_IF(filter_desc == nullptr, CUBE_INNER_ERR_REPORT(opName, "filter_desc is null"), return false);
    ge::Format filterFormat = filter_desc->GetStorageFormat();

    const auto y_desc = context->GetOutputDesc(Y_INDEX);
    OP_CHECK_IF(y_desc == nullptr, CUBE_INNER_ERR_REPORT(opName, "y_desc is null"), return false);
    ge::Format yFormat = y_desc->GetStorageFormat();

    runInfoV2.outBackpropFormat = outBackpropFormat;
    runInfoV2.filterFormat = filterFormat;
    runInfoV2.yFormat = yFormat;
    return true;
}

bool CalScale(const gert::TilingContext* context, const Conv3dBpInputV2RunInfo& runInfoV2,
              const OtherParams& otherParams)
{
    if (!IsSocVersionFuse(context)) {
        return true;
    }
    const char* opName = context->GetNodeName();
    if (otherParams.a_dtype == ge::DT_INT8) {
        if (static_cast<uint64_t>(runInfoV2.dedx_cin) * ge::GetSizeByDataType(ge::DT_INT64) > FB_BUFFER_SIZE) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                opName, "scale_size", std::to_string(runInfoV2.dedx_cin * ge::GetSizeByDataType(ge::DT_INT64)),
                FormatString("The value of scale_size must be less than or equal to FB_BUFFER_SIZE(%d)", FB_BUFFER_SIZE)
                    .c_str());
            return false;
        }
    }
    return true;
}

bool Conv3DBackpropInputParseFunc(gert::TilingContext* context, optiling::OpTypeV2 opType,
                                  Conv3dBpInputV2RunInfo& runInfoV2, OtherParams& otherParams, bool isV2Impl)
{
    const auto op_name = context->GetNodeName();
    OP_CHECK_IF(!GetAttrAndDtypeParams(context, runInfoV2, opType, otherParams),
                OP_LOGW(op_name, "Get attrs and dtype Failed."), return false);
    OP_CHECK_IF(!GetShapeParams(context, runInfoV2, opType, isV2Impl, otherParams),
                OP_LOGW(op_name, "Set shape params failed."), return false);
    ReCalDilation(context, runInfoV2, otherParams);
    OP_CHECK_IF(!GetInputOutputFormat(context, runInfoV2, opType),
                OP_LOGW(op_name, "failed to get input output format"), return false);
    OP_CHECK_IF(!CalGroups(context, otherParams, runInfoV2), OP_LOGW(op_name, "Calc groups failed."), return false);
    OP_CHECK_IF(!CalPads(context, runInfoV2, opType, otherParams), OP_LOGW(op_name, "Calc pads failed."), return false);
    OP_CHECK_IF(!CalModify(context, runInfoV2, otherParams), OP_LOGW(op_name, "Modify pad failed."), return false);
    OP_CHECK_IF(!CalRealG(context, runInfoV2, otherParams), OP_LOGW(op_name, "Calc real_g failed."), return false);
    OP_CHECK_IF(!CalScale(context, runInfoV2, otherParams), OP_LOGW(op_name, "Scale size too big, not support."),
                return false);
    return true;
}

bool IsSocVersionFuse(const gert::TilingContext* context)
{
    const auto op_name = context->GetNodeName();
    if (context == nullptr) {
        CUBE_INNER_ERR_REPORT(op_name, "context is null");
        return false;
    }
    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        CUBE_INNER_ERR_REPORT(op_name, "platformInfoPtr is null");
        return false;
    }

    std::string cube_vec_state_str;
    platformInfo->GetPlatformRes("SoCInfo", "cube_vector_combine", cube_vec_state_str);
    if (cube_vec_state_str.empty()) {
        CUBE_INNER_ERR_REPORT(op_name, "get cube_vector_combine failed");
        return false;
    }
    if (cube_vec_state_str == "fuse") {
        return true;
    }
    return false;
}

bool SetRunInfoToV2(gert::TilingContext* context, Conv3dBpInputV2RunInfo& runInfoV2, optiling::OpTypeV2 opType)
{
    OtherParams otherParams;
    OP_CHECK_IF(!ValidateConvBackpropContext(context), OP_LOGW(context->GetNodeName(), "failed to parse params"),
                return false);
    OP_CHECK_IF(!Conv3DBackpropInputParseFunc(context, opType, runInfoV2, otherParams, true),
                OP_LOGW(context->GetNodeName(), "failed to parse context"), return false);

    OP_CHECK_IF(!GetFusionMode(runInfoV2, context->GetNodeName(), context, opType),
                OP_LOGW(context, "failed to get enRelu flag"), return false);

    OP_CHECK_IF(!GetImplMode(runInfoV2, context->GetNodeName(), context, opType),
                OP_LOGW(context, "failed to get impl mode"), return false);

    if (!CheckCalPads(context, runInfoV2, opType, otherParams) || !CheckParams(runInfoV2, context, otherParams) ||
        !CheckAttrs(context, runInfoV2, context->GetNodeName(), otherParams) ||
        !CheckPadRange(context, runInfoV2, context->GetNodeName())) {
        OP_LOGW(context, "params is invalid");
        return false;
    }

    if ((opType == optiling::OpTypeV2::kConv3DTransposeV2 || opType == optiling::OpTypeV2::kExtendConvTranspose) &&
        (!CheckTranspose(context->GetNodeName(), context) || !CheckBiasParams(context, otherParams))) {
        OP_LOGW(context, "params is invalid");
        return false;
    }
    SetInitOutput(runInfoV2, opType, otherParams);
    // group扩维在单独的transdata算子中完成时，cin1_g和cout1_g为扩以后的属性，否则是常规值
    if (runInfoV2.real_g == 1 && !IsArchAfter35(context) && !IsSocVersionFuse(context)) {
        runInfoV2.dedy_cout1_g = otherParams.a_shape.c1;
        runInfoV2.dedx_cin1_g = otherParams.c_shape.c1;
    } else {
        runInfoV2.dedy_cout1_g = otherParams.co1g;
        runInfoV2.dedx_cin1_g = otherParams.ci1g;
    }
    runInfoV2.batch_n = otherParams.a_shape.batch;
    runInfoV2.dedx_d = otherParams.c_shape.d;
    runInfoV2.dedx_cin = otherParams.c_shape.c;
    runInfoV2.dedx_cin1 = otherParams.c_shape.c1;
    runInfoV2.dedx_h = otherParams.c_shape.h;
    runInfoV2.dedx_w = otherParams.c_shape.w;
    runInfoV2.dedy_d = otherParams.a_shape.d;
    runInfoV2.dedy_cout = otherParams.a_shape.c;
    runInfoV2.dedy_cout1 = otherParams.a_shape.c1;
    runInfoV2.dedy_h = otherParams.a_shape.h;
    runInfoV2.dedy_w = otherParams.a_shape.w;
    runInfoV2.kernel_d = otherParams.b_shape.d;
    runInfoV2.kernel_h = otherParams.b_shape.h;
    runInfoV2.kernel_w = otherParams.b_shape.w;

    runInfoV2.enlarge = otherParams.multiple_extend;                                                    // arch35
    runInfoV2.dedy_cout_g = otherParams.b_shape.batch / runInfoV2.groups * otherParams.multiple_extend; // arch35
    runInfoV2.dedx_cin_g = otherParams.b_shape.c * otherParams.multiple_extend;                         // arch35
    return true;
}

} // namespace Conv
} // namespace NN
} // namespace Ops

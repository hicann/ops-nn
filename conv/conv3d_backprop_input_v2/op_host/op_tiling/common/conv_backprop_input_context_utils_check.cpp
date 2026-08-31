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
 * \file conv_backprop_input_context_utils_check.cpp
 * \brief
 */
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

bool GetFusionMode(Conv3dBpInputV2RunInfo& runInfoV2, const char* opName, const gert::TilingContext* context,
                   optiling::OpTypeV2 opType)
{
    if (opType != optiling::OpTypeV2::kExtendConvTranspose && opType != optiling::OpTypeV2::kExtendConvTransposeV2) {
        return true;
    }
    auto attrs = context->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE_WITH_INVALID_ATTR(opName, "attrs", "null", "non_empty_value"), return false);
    size_t idx = K_FUSION_MODE_CONV3D_TRANSPOSE_IDX;
    if (idx < attrs->GetAttrNum()) {
        const int64_t* fusionMode = attrs->GetAttrPointer<int64_t>(idx);
        if (fusionMode != nullptr) {
            runInfoV2.enRelu0 = (*fusionMode & 0x1) ? 1 : 0;
            runInfoV2.enRelu1 = (*fusionMode & 0x2) ? 1 : 0;
        } else {
            OP_LOGW(opName, "relu flag is not support, so we set 0 as default");
            runInfoV2.enRelu0 = 0; // for extendConvTranspose fixpipe fusion pass, default value is 0
            runInfoV2.enRelu1 = 0;
        }
    }
    return true;
}

bool GetImplMode(Conv3dBpInputV2RunInfo& runInfoV2, const char* opName, const gert::TilingContext* context,
                 optiling::OpTypeV2 opType)
{
    if (opType == optiling::OpTypeV2::kExtendConvTranspose || opType == optiling::OpTypeV2::kExtendConvTransposeV2) {
        return true;
    }

    auto attrs = context->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE_WITH_INVALID_ATTR(opName, "attrs", "null", "non_empty_value"), return false);
    auto inputDesc = context->GetInputDesc(OUT_BACKPROP_INDEX);
    OP_CHECK_IF(inputDesc == nullptr, OP_LOGE_WITH_INVALID_ATTR(opName, "out_backprop", "null", "non_empty_value"),
                return false);
    ge::DataType aDtype = inputDesc->GetDataType();
    size_t enableHf32Index = 5U; // dx enablehf32 idx is 5
    if (opType == optiling::OpTypeV2::kConv3DTransposeV2) {
        enableHf32Index = 7U; // transpose enablehf32 idx is 7
    }
    if (aDtype == ge::DT_FLOAT && enableHf32Index < attrs->GetAttrNum()) {
        auto enableHf32Ptr = attrs->GetBool(enableHf32Index);
        if (enableHf32Ptr == nullptr) {
            OP_LOGE_WITH_INVALID_ATTR(opName, "enable_hf32", "null", "non_empty_value");
            return false;
        }
        bool enableHf32 = *enableHf32Ptr;
        OP_LOGD(opName, "attr idx[%zu] enable_hf32 = %d", enableHf32Index, enableHf32);
        runInfoV2.hf32_flag = static_cast<uint32_t>(enableHf32 ? 1 : 0);
    }
    return true;
}

bool CheckFilterShapeHW(const char* op_name, const OtherParams& otherParams, int32_t kFilterDimHWUpTmp)
{
    OP_CHECK_IF(
        !CheckRange(otherParams.b_shape.h, kDimLow, kFilterDimHWUpTmp),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            op_name, "filter_h", std::to_string(otherParams.b_shape.h),
            FormatString("the value of filter_h must be equal to [%d, %d]", kDimLow, kFilterDimHWUpTmp).c_str()),
        return false);
    OP_CHECK_IF(
        !CheckRange(otherParams.b_shape.w, kDimLow, kFilterDimHWUpTmp),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            op_name, "filter_w", std::to_string(otherParams.b_shape.w),
            FormatString("the value of filter_w must be equal to [%d, %d]", kDimLow, kFilterDimHWUpTmp).c_str()),
        return false);
    OP_CHECK_IF(!CheckRange(otherParams.b_shape.d, kDimLow, kDimBatchUp),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    op_name, "filter_d", std::to_string(otherParams.b_shape.d),
                    FormatString("the value of filter_d must be equal to [%d, %d]", kDimLow, kDimBatchUp).c_str()),
                return false);
    return true;
}

bool CheckAShapeParams(const char* op_name, const OtherParams& otherParams)
{
    OP_CHECK_IF(!CheckRange(otherParams.a_shape.batch, kDimLow, kDimBatchUp),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    op_name, "batch", std::to_string(otherParams.a_shape.batch),
                    FormatString("the value of batch must be equal to [%d, %d]", kDimLow, kDimBatchUp).c_str()),
                return false);
    OP_CHECK_IF(!CheckRange(otherParams.a_shape.d, kDimLow, kDimUp),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    op_name, "dout", std::to_string(otherParams.a_shape.d),
                    FormatString("the value of dout must be equal to [%d, %d]", kDimLow, kDimUp).c_str()),
                return false);
    OP_CHECK_IF(!CheckRange(otherParams.a_shape.h, kDimLow, kDimUp),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    op_name, "hout", std::to_string(otherParams.a_shape.h),
                    FormatString("the value of hout must be equal to [%d, %d]", kDimLow, kDimUp).c_str()),
                return false);
    OP_CHECK_IF(!CheckRange(otherParams.a_shape.w, kDimLow, kDimUp),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    op_name, "wout", std::to_string(otherParams.a_shape.w),
                    FormatString("the value of wout must be equal to [%d, %d]", kDimLow, kDimUp).c_str()),
                return false);
    return true;
}

bool CheckCShapeParams(const char* op_name, const OtherParams& otherParams)
{
    OP_CHECK_IF(!CheckLowerBound(otherParams.a_shape.c, kDimLow),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    op_name, "cout", std::to_string(otherParams.a_shape.c),
                    FormatString("the value of cout must be greater than or equal to [%d]", kDimLow).c_str()),
                return false);
    OP_CHECK_IF(!CheckLowerBound(otherParams.a_shape.c1, kDimLow),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    op_name, "cout1", std::to_string(otherParams.a_shape.c1),
                    FormatString("the value of cout1 must be greater than or equal to [%d]", kDimLow).c_str()),
                return false);
    OP_CHECK_IF(!CheckLowerBound(otherParams.c_shape.c1, kDimLow),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    op_name, "cin1", std::to_string(otherParams.c_shape.c1),
                    FormatString("the value of cin1 must be greater than or equal to [%d]", kDimLow).c_str()),
                return false);
    OP_CHECK_IF(!CheckLowerBound(otherParams.c_shape.c, kDimLow),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    op_name, "cin", std::to_string(otherParams.c_shape.c),
                    FormatString("the value of cin must be greater than or equal to [%d]", kDimLow).c_str()),
                return false);
    OP_CHECK_IF(!CheckLowerBound(otherParams.c_shape.d, kDimLow),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    op_name, "din", std::to_string(otherParams.c_shape.d),
                    FormatString("the value of din must be greater than or equal to [%d]", kDimLow).c_str()),
                return false);
    return true;
}

bool CheckShapeValidWithLog(const gert::TilingContext* context, const OtherParams& otherParams,
                            const Conv3dBpInputV2RunInfo& runInfoV2)
{
    const auto op_name = context->GetNodeName();
    int32_t kGroupUpTmp = kDimUp;
    int32_t kFilterDimHWUpTmp = kFilterDimHWUp;
    if (IsArchAfter35(context) || IsSocVersionFuse(context)) {
        kGroupUpTmp = kGroupUp;
        kFilterDimHWUpTmp = kDimUp;
    }
    if (!CheckRange(runInfoV2.groups, kDimLow, kGroupUpTmp)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            op_name, "groups", std::to_string(runInfoV2.groups),
            FormatString("the value of groups must be in range [%d, %d]", kDimLow, kGroupUpTmp).c_str());
        return false;
    }
    if (!CheckFilterShapeHW(op_name, otherParams, kFilterDimHWUpTmp)) {
        return false;
    }
    if (!CheckAShapeParams(op_name, otherParams)) {
        return false;
    }
    if (!CheckCShapeParams(op_name, otherParams)) {
        return false;
    }
    return true;
}

string IntToBinary(uint64_t& n)
{
    string ans = "";
    do {
        uint64_t t = n % 2UL;
        ans += (t + CHAR_0);
        n /= 2UL;
    } while (n != 0UL);
    return ans;
}

void CalculateAlignAndSize(const OtherParams& otherParams, const Conv3dBpInputV2RunInfo& runInfoV2,
                           int64_t& dedy_c_align, int64_t& dedx_c_align, int64_t& filter_c_align,
                           int64_t& filter_n_align, int64_t& dedy_size, int64_t& dedx_size, int64_t& filter_size)
{
    dedy_c_align = Ops::Base::CeilAlign(otherParams.a_shape.c, otherParams.a_shape.c0);
    dedx_c_align = Ops::Base::CeilAlign(otherParams.c_shape.c, otherParams.c_shape.c0);
    filter_c_align = Ops::Base::CeilAlign(otherParams.b_shape.c, static_cast<int64_t>(otherParams.filter_ci0));
    filter_n_align = Ops::Base::CeilAlign(otherParams.b_shape.batch, static_cast<int64_t>(otherParams.filter_co0));
    dedy_size = otherParams.a_shape.batch * dedy_c_align * otherParams.a_shape.d * otherParams.a_shape.w *
                otherParams.a_shape.h * runInfoV2.a_dtype_bytes;
    dedx_size = otherParams.a_shape.batch * dedx_c_align * otherParams.c_shape.d * otherParams.c_shape.w *
                otherParams.c_shape.h * runInfoV2.c_dtype_bytes;
    filter_size = filter_n_align * filter_c_align * otherParams.filter_d_dilation * otherParams.b_shape.w *
                  otherParams.b_shape.h * runInfoV2.b_dtype_bytes;
}

void SetFmapPaddingParams(OtherParams& otherParams, const Conv3dBpInputV2RunInfo& runInfoV2)
{
    otherParams.fmap_d_padding = otherParams.c_shape.d + runInfoV2.pad_h + runInfoV2.pad_t;
    otherParams.fmap_h_padding = otherParams.c_shape.h + runInfoV2.pad_u + runInfoV2.pad_d;
    otherParams.fmap_w_padding = otherParams.c_shape.w + runInfoV2.pad_l + runInfoV2.pad_r;
}

bool CheckGroupsAndFilterRange(const char* opName, const Conv3dBpInputV2RunInfo& runInfoV2,
                               const OtherParams& otherParams, int32_t kFilterDimHWUpTmp)
{
    OP_CHECK_IF(!CheckRange(runInfoV2.groups, kDimLow, kDimUp),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    opName, "groups", std::to_string(runInfoV2.groups),
                    FormatString("groups must be within the range [%d, %d]", kDimLow, kDimUp).c_str()),
                return false);
    OP_CHECK_IF(!CheckRange(otherParams.b_shape.h, kDimLow, kFilterDimHWUpTmp),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    opName, "filter_h", std::to_string(otherParams.b_shape.h),
                    FormatString("filter_h must be within the range [%d, %d]", kDimLow, kFilterDimHWUpTmp).c_str()),
                return false);
    OP_CHECK_IF(!CheckRange(otherParams.b_shape.w, kDimLow, kFilterDimHWUpTmp),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    opName, "filter_w", std::to_string(otherParams.b_shape.w),
                    FormatString("filter_w must be within the range [%d, %d]", kDimLow, kFilterDimHWUpTmp).c_str()),
                return false);
    OP_CHECK_IF(!CheckRange(otherParams.b_shape.d, kDimLow, kDimBatchUp),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    opName, "filter_d", std::to_string(otherParams.b_shape.d),
                    FormatString("filter_d must be within the range [%d, %d]", kDimLow, kDimBatchUp).c_str()),
                return false);
    return true;
}

bool CheckAShapeRangeInParams(const char* opName, const OtherParams& otherParams)
{
    OP_CHECK_IF(!CheckRange(otherParams.a_shape.batch, kDimLow, kDimBatchUp),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    opName, "batch", std::to_string(otherParams.a_shape.batch),
                    FormatString("batch must be within the range [%d, %d]", kDimLow, kDimBatchUp).c_str()),
                return false);
    OP_CHECK_IF(
        !CheckLowerBound(otherParams.a_shape.c1, kDimLow),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName, "co1", std::to_string(otherParams.a_shape.c1),
                                              FormatString("co1 must be greater than or equal to %d", kDimLow).c_str()),
        return false);
    OP_CHECK_IF(!CheckRange(otherParams.a_shape.d, kDimLow, kDimUp),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    opName, "dout", std::to_string(otherParams.a_shape.d),
                    FormatString("dout must be within the range [%d, %d]", kDimLow, kDimUp).c_str()),
                return false);
    OP_CHECK_IF(!CheckRange(otherParams.a_shape.h, kDimLow, kDimUp),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    opName, "hout", std::to_string(otherParams.a_shape.h),
                    FormatString("hout must be within the range [%d, %d]", kDimLow, kDimUp).c_str()),
                return false);
    OP_CHECK_IF(!CheckRange(otherParams.a_shape.w, kDimLow, kDimUp),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    opName, "wout", std::to_string(otherParams.a_shape.w),
                    FormatString("wout must be within the range [%d, %d]", kDimLow, kDimUp).c_str()),
                return false);
    OP_CHECK_IF(
        !CheckLowerBound(otherParams.a_shape.c, kDimLow),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName, "co", std::to_string(otherParams.a_shape.c),
                                              FormatString("co must be greater than or equal to %d", kDimLow).c_str()),
        return false);
    return true;
}

bool CheckCShapeRangeInParams(const char* opName, const OtherParams& otherParams)
{
    OP_CHECK_IF(
        !CheckLowerBound(otherParams.c_shape.c1, kDimLow),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName, "c1", std::to_string(otherParams.c_shape.c1),
                                              FormatString("c1 must be greater than or equal to %d", kDimLow).c_str()),
        return false);
    OP_CHECK_IF(
        !CheckLowerBound(otherParams.c_shape.c, kDimLow),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName, "cin", std::to_string(otherParams.c_shape.c),
                                              FormatString("cin must be greater than or equal to %d", kDimLow).c_str()),
        return false);
    OP_CHECK_IF(
        !CheckLowerBound(otherParams.c_shape.d, kDimLow),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName, "din", std::to_string(otherParams.c_shape.d),
                                              FormatString("din must be greater than or equal to %d", kDimLow).c_str()),
        return false);
    OP_CHECK_IF(!CheckRange(otherParams.c_shape.h, kDimLow, kDimUp),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    opName, "hin", std::to_string(otherParams.c_shape.h),
                    FormatString("the value of hin must be within the range [%d, %d]", kDimLow, kDimUp).c_str()),
                return false);
    OP_CHECK_IF(!CheckRange(otherParams.c_shape.w, kDimLow, kDimUp),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    opName, "win", std::to_string(otherParams.c_shape.w),
                    FormatString("the value of win must be within the range [%d, %d]", kDimLow, kDimUp).c_str()),
                return false);
    return true;
}

bool CheckDilationsRangeInParams(const char* opName, const Conv3dBpInputV2RunInfo& runInfoV2, int32_t kDilationUpTmp)
{
    OP_CHECK_IF(!CheckRange(runInfoV2.dilation_h, kDilationLow, kDilationUpTmp),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    opName, "h_dilation", std::to_string(runInfoV2.dilation_h),
                    FormatString("h_dilation must be within the range [%d, %d]", kDilationLow, kDilationUpTmp).c_str()),
                return false);
    OP_CHECK_IF(!CheckRange(runInfoV2.dilation_w, kDilationLow, kDilationUpTmp),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    opName, "w_dilation", std::to_string(runInfoV2.dilation_w),
                    FormatString("w_dilation must be within the range [%d, %d]", kDilationLow, kDilationUpTmp).c_str()),
                return false);
    OP_CHECK_IF(!CheckRange(runInfoV2.dilation_d, kDilationLow, kDilationUpTmp),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    opName, "d_dilation", std::to_string(runInfoV2.dilation_d),
                    FormatString("d_dilation must be within the range [%d, %d]", kDilationLow, kDilationUpTmp).c_str()),
                return false);
    return true;
}

bool CheckStridesExpandRange(const char* opName, const OtherParams& otherParams,
                             const Conv3dBpInputV2RunInfo& runInfoV2)
{
    OP_CHECK_IF(!CheckRange(otherParams.a_shape.h * runInfoV2.stride_h, kDimLow, kDimUp),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    opName, "hout", std::to_string(otherParams.a_shape.h * runInfoV2.stride_h),
                    FormatString("hout must be within the range [%d, %d]", kDimLow, kDimUp).c_str()),
                return false);
    OP_CHECK_IF(!CheckRange(otherParams.a_shape.w * runInfoV2.stride_w, kDimLow, kDimUp),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    opName, "wout", std::to_string(otherParams.a_shape.w * runInfoV2.stride_w),
                    FormatString("wout must be within the range [%d, %d]", kDimLow, kDimUp).c_str()),
                return false);
    OP_CHECK_IF(!CheckRange(otherParams.a_shape.d * runInfoV2.stride_d, kDimLow, kDimUp),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    opName, "dout", std::to_string(otherParams.a_shape.d * runInfoV2.stride_d),
                    FormatString("dout must be within the range [%d, %d]", kDimLow, kDimUp).c_str()),
                return false);
    return true;
}

bool CheckGroupsDivisibility(const char* opName, const OtherParams& otherParams,
                             const Conv3dBpInputV2RunInfo& runInfoV2)
{
    OP_CHECK_IF(!CheckValue(static_cast<int32_t>(otherParams.a_shape.c % static_cast<int64_t>(runInfoV2.groups)), 0),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName, "cout", std::to_string(otherParams.a_shape.c),
                                                      "cout must be exactly divisible by groups"),
                return false);
    OP_CHECK_IF(!CheckValue(static_cast<int32_t>(otherParams.c_shape.c % static_cast<int64_t>(runInfoV2.groups)), 0),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName, "cin", std::to_string(otherParams.c_shape.c),
                                                      "cin must be exactly divisible by groups"),
                return false);
    OP_CHECK_IF(!CheckValue(static_cast<int32_t>(otherParams.c_shape.c),
                            static_cast<int32_t>(otherParams.b_shape.c * static_cast<int64_t>(runInfoV2.groups))),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName, "cout", std::to_string(otherParams.c_shape.c),
                                                      "c dim of fmap must be equal with filter c multiplied groups"),
                return false);
    return true;
}

bool CheckBatchConsistency(const char* opName, const OtherParams& otherParams)
{
    OP_CHECK_IF(!CheckValue(otherParams.a_shape.c, otherParams.b_shape.batch),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    opName, "a_shape.c", std::to_string(otherParams.a_shape.c),
                    FormatString("a_shape.c must be equal to b_shape.batch[%d]", otherParams.b_shape.batch).c_str()),
                return false);
    OP_CHECK_IF(
        !CheckValue(otherParams.a_shape.batch, otherParams.c_shape.batch),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            opName, "a_shape.batch", std::to_string(otherParams.a_shape.batch),
            FormatString("a_shape.batch must be equal to c_shape.batch[%d]", otherParams.c_shape.batch).c_str()),
        return false);
    return true;
}

bool CheckFilterDilationVsPadding(const char* opName, const OtherParams& otherParams)
{
    OP_CHECK_IF(otherParams.filter_d_dilation > otherParams.fmap_d_padding,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    opName, "filter_d_dilation or fmap_d_padding",
                    std::to_string(otherParams.filter_d_dilation) + " or " + std::to_string(otherParams.fmap_d_padding),
                    FormatString("filter_d_dilation must be less than or equal to fmap_d_padding[%d]",
                                 otherParams.fmap_d_padding)
                        .c_str()),
                return false);
    OP_CHECK_IF(otherParams.filter_h_dilation > otherParams.fmap_h_padding,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    opName, "filter_h_dilation or fmap_h_padding",
                    std::to_string(otherParams.filter_h_dilation) + " or " + std::to_string(otherParams.fmap_h_padding),
                    FormatString("filter_h_dilation must be less than or equal to fmap_h_padding[%d]",
                                 otherParams.fmap_h_padding)
                        .c_str()),
                return false);
    OP_CHECK_IF(otherParams.filter_w_dilation > otherParams.fmap_w_padding,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    opName, "filter_w_dilation or fmap_w_padding",
                    std::to_string(otherParams.filter_w_dilation) + " or " + std::to_string(otherParams.fmap_w_padding),
                    FormatString("filter_w_dilation must be less than or equal to fmap_w_padding[%d]",
                                 otherParams.fmap_w_padding)
                        .c_str()),
                return false);
    return true;
}

bool CheckOutputShapeMatch(const char* opName, const OtherParams& otherParams, const Conv3dBpInputV2RunInfo& runInfoV2)
{
    int64_t do_temp = (otherParams.fmap_d_padding - otherParams.filter_d_dilation) / runInfoV2.stride_d + 1;
    int64_t ho_temp = (otherParams.fmap_h_padding - otherParams.filter_h_dilation) / runInfoV2.stride_h + 1;
    int64_t wo_temp = (otherParams.fmap_w_padding - otherParams.filter_w_dilation) / runInfoV2.stride_w + 1;
    OP_CHECK_IF(
        do_temp != otherParams.a_shape.d,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            opName, "dout", std::to_string(otherParams.a_shape.d),
            FormatString(
                "the value of dout must be equal to (fmap_d + pad_h + pad_t - filter_d_dilation) / stride_d + 1")
                .c_str()),
        return false);
    OP_CHECK_IF(
        ho_temp != otherParams.a_shape.h,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            opName, "hout", std::to_string(otherParams.a_shape.h),
            FormatString(
                "the value of hout must be equal to (fmap_h + pad_u + pad_d - filter_h_dilation) / stride_h + 1")
                .c_str()),
        return false);
    OP_CHECK_IF(
        wo_temp != otherParams.a_shape.w,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            opName, "wout", std::to_string(otherParams.a_shape.w),
            FormatString(
                "the value of wout must be equal to (fmap_w + pad_l + pad_r - filter_w_dilation) / stride_w + 1")
                .c_str()),
        return false);
    return true;
}

bool CheckAlignAndSizeLimits(const char* opName, int64_t dedy_c_align, int64_t dedx_c_align, int64_t filter_c_align,
                             int64_t filter_n_align, int64_t dedy_size, int64_t dedx_size, int64_t filter_size)
{
    OP_CHECK_IF(dedy_c_align == 0 || dedx_c_align == 0 || filter_c_align == 0 || filter_n_align == 0,
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                    opName, "{dedy_c_align, dedx_c_align, filter_c_align, filter_n_align}",
                    "{" + std::to_string(dedy_c_align) + ", " + std::to_string(dedx_c_align) + ", " +
                        std::to_string(filter_c_align) + ", " + std::to_string(filter_n_align) + "}",
                    FormatString("{dedy_c_align, dedx_c_align, filter_c_align, filter_n_align} cannot be 0").c_str()),
                return false);
    OP_CHECK_IF(
        dedy_size > kDataSizeMax,
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
            opName, "out_backprop_size", std::to_string(dedy_size),
            FormatString("the shape size of out_backprop must be less than or equal to %d", kDataSizeMax).c_str()),
        return false);
    OP_CHECK_IF(dedx_size > kDataSizeMax,
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                    opName, "fmap_size", std::to_string(dedx_size),
                    FormatString("the shape size of fmap_size must be less than or equal to %d", kDataSizeMax).c_str()),
                return false);
    OP_CHECK_IF(
        filter_size > kDataSizeMax,
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
            opName, "filter_size", std::to_string(filter_size),
            FormatString("the shape size of filter_size must be less than or equal to %d", kDataSizeMax).c_str()),
        return false);
    return true;
}

bool CheckBackpropPadRange(const char* opName, const OtherParams& otherParams, int32_t kPadUpTmp)
{
    OP_CHECK_IF(otherParams.pad_up_before > kPadUpTmp,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    opName, "pad_u", std::to_string(otherParams.pad_up_before),
                    FormatString("the value of pad_u must be equal to or less than %d", kPadUpTmp).c_str()),
                return false);
    OP_CHECK_IF(otherParams.pad_left_before > kPadUpTmp,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    opName, "pad_l", std::to_string(otherParams.pad_left_before),
                    FormatString("the value of pad_l must be equal to or less than %d", kPadUpTmp).c_str()),
                return false);
    OP_CHECK_IF(otherParams.pad_down_after > kPadUpTmp,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    opName, "pad_d", std::to_string(otherParams.pad_down_after),
                    FormatString("the value of pad_d must be equal to or less than %d", kPadUpTmp).c_str()),
                return false);
    OP_CHECK_IF(otherParams.pad_right_after > kPadUpTmp,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    opName, "pad_r", std::to_string(otherParams.pad_right_after),
                    FormatString("the value of pad_r must be equal to or less than %d", kPadUpTmp).c_str()),
                return false);
    return true;
}

bool CheckParams(Conv3dBpInputV2RunInfo& runInfoV2, gert::TilingContext* context, OtherParams& otherParams)
{
    int64_t dedy_c_align, dedx_c_align, filter_c_align, filter_n_align;
    int64_t dedy_size, dedx_size, filter_size;
    CalculateAlignAndSize(otherParams, runInfoV2, dedy_c_align, dedx_c_align, filter_c_align, filter_n_align, dedy_size,
                          dedx_size, filter_size);
    SetFmapPaddingParams(otherParams, runInfoV2);

    if (!CheckParamsWithLog(runInfoV2, context, otherParams) ||
        !CheckShapeValidWithLog(context, otherParams, runInfoV2)) {
        return false;
    }

    int32_t kFilterDimHWUpTmp = kFilterDimHWUp;
    int32_t kPadUpTmp = kPadUp;
    int32_t kDilationUpTmp = kDilationUp;
    if (IsArchAfter35(context)) {
        kFilterDimHWUpTmp = kDimUp;
        kPadUpTmp = kDimUp;
        kDilationUpTmp = kDimUp;
    }

    const char* opName = context->GetNodeName();
    if (!CheckGroupsAndFilterRange(opName, runInfoV2, otherParams, kFilterDimHWUpTmp)) {
        return false;
    }
    if (!CheckAShapeRangeInParams(opName, otherParams)) {
        return false;
    }
    if (!CheckCShapeRangeInParams(opName, otherParams)) {
        return false;
    }
    if (!CheckDilationsRangeInParams(opName, runInfoV2, kDilationUpTmp)) {
        return false;
    }
    if (!CheckStridesExpandRange(opName, otherParams, runInfoV2)) {
        return false;
    }
    if (!CheckGroupsDivisibility(opName, otherParams, runInfoV2)) {
        return false;
    }
    if (!CheckBatchConsistency(opName, otherParams)) {
        return false;
    }
    if (!CheckFilterDilationVsPadding(opName, otherParams)) {
        return false;
    }
    if (!CheckOutputShapeMatch(opName, otherParams, runInfoV2)) {
        return false;
    }
    if (!CheckAlignAndSizeLimits(opName, dedy_c_align, dedx_c_align, filter_c_align, filter_n_align, dedy_size,
                                 dedx_size, filter_size)) {
        return false;
    }
    OP_CHECK_IF(!CheckL1SizeLimit(runInfoV2, context, otherParams), OP_LOGW("this case may exceed size"), return false);
    if (!CheckBackpropPadRange(opName, otherParams, kPadUpTmp)) {
        return false;
    }

    return true;
}

bool CheckAttrs(const gert::TilingContext* context, Conv3dBpInputV2RunInfo& runInfoV2, const char* opName,
                OtherParams& otherParams)
{
    // kernel大于1时，才能有dilation属性, 前面代码已经做了兼容性属性重设，这里做二次check
    bool dilationDFlag = (runInfoV2.dilation_d != 1 && otherParams.b_shape.d == 1);
    bool dilationHFlag = (runInfoV2.dilation_h != 1 && otherParams.b_shape.h == 1);
    bool dilationWFlag = (runInfoV2.dilation_w != 1 && otherParams.b_shape.w == 1);

    int32_t strideHwUp = STRIDES_DIM_HW_UP;
    int32_t strideDUp = STRIDES_DIM_DEPTH_UP;
    if (IsArchAfter35(context)) {
        strideHwUp = kDimUp;
        strideDUp = kDimUp;
    }

    OP_CHECK_IF(dilationDFlag,
                CUBE_INNER_ERR_REPORT(opName, "cannot support dilation_d: [%s] != 1 while kernel_d: [%s] = 1",
                                      std::to_string(runInfoV2.dilation_d).c_str(),
                                      std::to_string(otherParams.b_shape.d).c_str()),
                return false);
    OP_CHECK_IF(dilationHFlag,
                CUBE_INNER_ERR_REPORT(opName, "cannot support dilation_h: [%s] != 1 while kernel_h: [%s] = 1",
                                      std::to_string(runInfoV2.dilation_h).c_str(),
                                      std::to_string(otherParams.b_shape.h).c_str()),
                return false);
    OP_CHECK_IF(dilationWFlag,
                CUBE_INNER_ERR_REPORT(opName, "cannot support dilation_w: [%s] != 1 while kernel_w: [%s] = 1",
                                      std::to_string(runInfoV2.dilation_w).c_str(),
                                      std::to_string(otherParams.b_shape.w).c_str()),
                return false);

    if (!IsArchAfter35(context)) {
        if (runInfoV2.stride_d > otherParams.b_shape.d) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                opName, "stride_d", std::to_string(runInfoV2.stride_d),
                FormatString("cannot support stride_d > kernel_d: %ld", otherParams.b_shape.d).c_str());
            return false;
        }
    }

    OP_CHECK_IF(CheckRange(runInfoV2.stride_h, DIM_LOW, strideHwUp) == false,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    opName, "stride_h", std::to_string(runInfoV2.stride_h),
                    FormatString("the value of stride_h must be in range [%d, %d]", DIM_LOW, strideHwUp).c_str()),
                return false);

    OP_CHECK_IF(CheckRange(runInfoV2.stride_w, DIM_LOW, strideHwUp) == false,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    opName, "stride_w", std::to_string(runInfoV2.stride_w),
                    FormatString("the value of stride_w must be in range [%d, %d]", DIM_LOW, strideHwUp).c_str()),
                return false);

    OP_CHECK_IF(CheckRange(runInfoV2.stride_d, DIM_LOW, strideDUp) == false,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    opName, "stride_d", std::to_string(runInfoV2.stride_d),
                    FormatString("the value of stride_d must be in range [%d, %d]", DIM_LOW, strideDUp).c_str()),
                return false);

    uint64_t curL0CDstStride = static_cast<uint64_t>(otherParams.c_shape.h) * otherParams.c_shape.w;
    OP_CHECK_IF(
        curL0CDstStride > UINT32_MAX,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            opName, "curL0CDstStride", std::to_string(curL0CDstStride),
            FormatString("the value of hi * wi=%lu must be less than or equal to %u", curL0CDstStride, UINT32_MAX)
                .c_str()),
        return false);

    OP_CHECK_IF(CheckRange(runInfoV2.groups, GROUPS_LOW, GROUPS_UP) == false,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    opName, "groups", std::to_string(runInfoV2.groups),
                    FormatString("the value of groups must be in range [%d, %d]", GROUPS_LOW, GROUPS_UP).c_str()),
                return false);
    return true;
}

bool CheckPadRange(const gert::TilingContext* context, Conv3dBpInputV2RunInfo& runInfoV2, const char* opName)
{
    int32_t padDimUp = PAD_DIM_UP;
    if (IsArchAfter35(context)) {
        padDimUp = kDimUp;
    }
    OP_CHECK_IF(CheckRange(runInfoV2.pad_h, PAD_DIM_LOW, padDimUp) == false,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    opName, "pad_h", std::to_string(runInfoV2.pad_h),
                    FormatString("the value of pad_h must be in range [%d, %d]", PAD_DIM_LOW, padDimUp).c_str()),
                return false);
    OP_CHECK_IF(CheckRange(runInfoV2.pad_t, PAD_DIM_LOW, padDimUp) == false,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    opName, "pad_t", std::to_string(runInfoV2.pad_t),
                    FormatString("the value of pad_t must be in range [%d, %d]", PAD_DIM_LOW, padDimUp).c_str()),
                return false);
    OP_CHECK_IF(CheckRange(runInfoV2.pad_u, PAD_DIM_LOW, padDimUp) == false,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    opName, "pad_u", std::to_string(runInfoV2.pad_u),
                    FormatString("the value of pad_u must be in range [%d, %d]", PAD_DIM_LOW, padDimUp).c_str()),
                return false);
    OP_CHECK_IF(CheckRange(runInfoV2.pad_d, PAD_DIM_LOW, padDimUp) == false,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    opName, "pad_d", std::to_string(runInfoV2.pad_d),
                    FormatString("the value of pad_d must be in range [%d, %d]", PAD_DIM_LOW, padDimUp).c_str()),
                return false);
    OP_CHECK_IF(CheckRange(runInfoV2.pad_l, PAD_DIM_LOW, padDimUp) == false,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    opName, "pad_l", std::to_string(runInfoV2.pad_l),
                    FormatString("the value of pad_l must be in range [%d, %d]", PAD_DIM_LOW, padDimUp).c_str()),
                return false);
    OP_CHECK_IF(CheckRange(runInfoV2.pad_r, PAD_DIM_LOW, padDimUp) == false,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    opName, "pad_r", std::to_string(runInfoV2.pad_r),
                    FormatString("the value of pad_r must be in range [%d, %d]", PAD_DIM_LOW, padDimUp).c_str()),
                return false);
    return true;
}

template <typename T>
std::string DebugString(const std::vector<T>& v)
{
    std::ostringstream oss;
    oss << "[";
    if (v.size() > 0) {
        for (size_t i = 0; i < v.size() - 1; ++i) {
            oss << v[i] << ", ";
        }
        oss << v[v.size() - 1];
    }
    oss << "]";
    return oss.str();
}

bool ValidateOutputPaddingData(const char* opName, const gert::ContinuousVector* outputPadding,
                               bool& outputPaddingAllzero, bool& outputPaddingAllNonNegative,
                               std::vector<int64_t>& outputPaddingValue)
{
    OP_CHECK_IF(outputPadding->GetData() == nullptr,
                OP_LOGE_WITH_INVALID_ATTR(opName, "output_padding GetData", "null", "non_empty_value"), return false);
    OP_CHECK_IF(outputPadding->GetSize() != OUTPUT_PADDING_DIM,
                OP_LOGE_WITH_INVALID_ATTR_SIZE(opName, "output_padding", std::to_string(outputPadding->GetSize()),
                                               std::to_string(OUTPUT_PADDING_DIM)),
                return false);
    const auto outputPaddingData = static_cast<const int64_t*>(outputPadding->GetData());
    outputPaddingAllzero = true;
    outputPaddingAllNonNegative = true;
    for (size_t index = 0; index < outputPadding->GetSize(); index++) {
        outputPaddingAllzero = outputPaddingData[index] != 0 ? false : outputPaddingAllzero;
        outputPaddingAllNonNegative = outputPaddingData[index] < 0 ? false : outputPaddingAllNonNegative;
        outputPaddingValue.push_back(outputPaddingData[index]);
    }
    return true;
}

bool CheckOutputPaddingDtype(const char* opName, const gert::TilingContext* context,
                             const std::vector<int64_t>& outputPaddingValue, bool outputPaddingAllzero,
                             bool outputPaddingAllNonNegative)
{
    OP_CHECK_IF(
        (!outputPaddingAllzero) &&
            (!IsSupportedDtypeForOutputPadding(context->GetInputDesc(FILTER_INDEX)->GetDataType()) ||
             !IsSupportedDtypeForOutputPadding(context->GetInputDesc(OUT_BACKPROP_INDEX)->GetDataType())),
        CUBE_INNER_ERR_REPORT(
            opName,
            "when output_padding[%s] is not all zero, op only supports bfloat16, float16, float32 and int8 for all "
            "inputs, get filter dtype[%s], output backprop dtype[%s]",
            DebugString(outputPaddingValue).c_str(),
            ge::TypeUtils::DataTypeToSerialString(context->GetInputDesc(FILTER_INDEX)->GetDataType()).c_str(),
            ge::TypeUtils::DataTypeToSerialString(context->GetInputDesc(OUT_BACKPROP_INDEX)->GetDataType()).c_str()),
        return false);
    OP_CHECK_IF(!outputPaddingAllNonNegative && IsArchAfter35(context),
                CUBE_INNER_ERR_REPORT(
                    opName, "output_padding[%s] contains negative values, op only supports all non-negative inputs.",
                    DebugString(outputPaddingValue).c_str()),
                return false);
    return true;
}

bool CheckOffsetXAndOffsetW(const char* opName, const gert::TilingContext* context, const gert::RuntimeAttrs* attrs)
{
    const auto offsetX = attrs->GetAttrPointer<int64_t>(OFFSET_X_INDEX);
    OP_CHECK_IF(offsetX != nullptr && *offsetX != 0,
                OP_LOGE_WITH_INVALID_ATTR(opName, "offset_x", "null", "non_empty_value"), return false);
    auto offsetWShape = context->GetOptionalInputShape(OFFSET_W_INDEX);
    OP_CHECK_IF(offsetWShape != nullptr && offsetWShape->GetStorageShape().GetShapeSize() != 0,
                OP_LOGE_WITH_INVALID_ATTR(opName, "offset_w", "null", "non_empty_value"), return false);
    return true;
}

bool CheckTranspose(const char* opName, const gert::TilingContext* context)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE_WITH_INVALID_ATTR(opName, "attrs", "null", "non_empty_value"), return false);

    auto outputPadding = attrs->GetAttrPointer<gert::ContinuousVector>(OUTPUT_PADDING_INDEX);
    if (outputPadding != nullptr) {
        bool outputPaddingAllzero = true;
        bool outputPaddingAllNonNegative = true;
        std::vector<int64_t> outputPaddingValue;
        if (!ValidateOutputPaddingData(opName, outputPadding, outputPaddingAllzero, outputPaddingAllNonNegative,
                                       outputPaddingValue)) {
            return false;
        }
        if (!CheckOutputPaddingDtype(opName, context, outputPaddingValue, outputPaddingAllzero,
                                     outputPaddingAllNonNegative)) {
            return false;
        }
    }

    if (!IsSocVersionFuse(context) && !IsArchAfter35(context)) {
        if (!CheckOffsetXAndOffsetW(opName, context, attrs)) {
            return false;
        }
        auto biasShape = context->GetOptionalInputShape(BAIS_INDEX);
        if (biasShape != nullptr && biasShape->GetStorageShape().GetShapeSize() != 0) {
            OP_LOGE_WITH_INVALID_ATTR(opName, "biasShape", "null", "non_empty_value");
            return false;
        }
    }
    return true;
}

bool CheckBiasParams(gert::TilingContext* context, const OtherParams& otherParams)
{
    if (!IsArchAfter35(context)) {
        return true;
    }

    auto biasShape = context->GetOptionalInputShape(BAIS_INDEX);
    if (biasShape == nullptr || biasShape->GetStorageShape().GetShapeSize() == 0) {
        return true;
    }

    const auto op_name = context->GetNodeName();
    auto biasDesc = context->GetOptionalInputDesc(BAIS_INDEX);
    if (biasDesc == nullptr) {
        CUBE_INNER_ERR_REPORT(op_name, "failed to get bias tensor desc from context");
        return false;
    }

    const auto& storageShape = biasShape->GetStorageShape();
    OP_CHECK_IF(
        storageShape.GetDimNum() != 1,
        OP_LOGE_FOR_INVALID_SHAPEDIM(
            op_name, "bias", std::to_string(storageShape.GetDimNum()),
            FormatString("bias has incorrect shape dim %zu, should be 1D tensor", storageShape.GetDimNum()).c_str()),
        return false);

    int64_t biasCin = storageShape.GetDim(0);
    OP_CHECK_IF(biasCin != otherParams.c_shape.c,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    op_name, "biasCin", std::to_string(biasCin),
                    FormatString("bias shape[0] should be equal to dedx_cin[%d]", otherParams.c_shape.c).c_str()),
                return false);

    ge::DataType biasDtype = biasDesc->GetDataType();
    // Bias的输入类型，在非量化和量化场景有差异，分别进行判断
    if (otherParams.a_dtype == ge::DT_INT8) {
        OP_CHECK_IF(
            biasDtype != ge::DT_INT32,
            OP_LOGE_FOR_INVALID_DTYPE(op_name, "biasDtype", ge::TypeUtils::DataTypeToSerialString(biasDtype).c_str(),
                                      FormatString("bias dtype has incorrect value %s, should be INT32 in quant mode",
                                                   ge::TypeUtils::DataTypeToSerialString(biasDtype).c_str())
                                          .c_str()),
            return false);
    } else {
        OP_CHECK_IF(
            biasDtype != ge::DT_FLOAT,
            OP_LOGE_FOR_INVALID_DTYPE(op_name, "biasDtype", ge::TypeUtils::DataTypeToSerialString(biasDtype).c_str(),
                                      FormatString("bias dtype has incorrect value %s, should be FP32",
                                                   ge::TypeUtils::DataTypeToSerialString(biasDtype).c_str())
                                          .c_str()),
            return false);
    }

    return true;
}

void SetInitOutput(Conv3dBpInputV2RunInfo& runInfoV2, const optiling::OpTypeV2 opType, const OtherParams& otherParams)
{
    int32_t doModulo = (otherParams.fmap_d_padding - otherParams.filter_d_dilation) % runInfoV2.stride_d;
    int32_t hoModulo = (otherParams.fmap_h_padding - otherParams.filter_h_dilation) % runInfoV2.stride_h;
    if (doModulo > runInfoV2.pad_t || hoModulo > runInfoV2.pad_d || runInfoV2.stride_h > otherParams.b_shape.h ||
        ((opType == optiling::OpTypeV2::kConv3DTransposeV2 || opType == optiling::OpTypeV2::kExtendConvTranspose ||
          opType == optiling::OpTypeV2::kExtendConvTransposeV2) &&
         (otherParams.output_padding.output_padding_d > 0 || otherParams.output_padding.output_padding_h > 0)) ||
        runInfoV2.dilation_d > 1) {
        // 1 is init output with l0C, 2 is init output with l1, defualt is 0 means not init output
        runInfoV2.initOutputFlag = 1;
    }
}
} // namespace Conv
} // namespace NN
} // namespace Ops

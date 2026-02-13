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
#include "conv/common/op_host/op_tiling/math_util.h"
#include "conv/common/op_host/op_tiling/platform_util.h"
#include "conv_backprop_filter_context_utils.h"
#include "conv/common/op_host/op_tiling/platform_util.h"

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
}

using namespace optiling;
namespace Ops {
namespace NN {
namespace Conv {
template <typename T>
static bool GetNCDHWShape(const T &origin_shape, int64_t *ncdhw_shape, ge::Format origin_format)
{
    size_t idx = 0;
    if (origin_format == ge::FORMAT_NDHWC) {
        ncdhw_shape[idx++] = origin_shape[0];  // 0: N
        ncdhw_shape[idx++] = origin_shape[4];  // 4: C
        ncdhw_shape[idx++] = origin_shape[1];  // 1: D
        ncdhw_shape[idx++] = origin_shape[2];  // 2: H
        ncdhw_shape[idx++] = origin_shape[3];  // 3: W
        return true;
    } else if (origin_format == ge::FORMAT_NCDHW) {
        ncdhw_shape[idx++] = origin_shape[0];  // 0: N
        ncdhw_shape[idx++] = origin_shape[1];  // 1: C
        ncdhw_shape[idx++] = origin_shape[2];  // 2: D
        ncdhw_shape[idx++] = origin_shape[3];  // 3: H
        ncdhw_shape[idx++] = origin_shape[4];  // 4: W
        return true;
    } else if (origin_format == ge::FORMAT_DHWCN) {
        ncdhw_shape[idx++] = origin_shape[4];  // 4: N
        ncdhw_shape[idx++] = origin_shape[3];  // 3: C
        ncdhw_shape[idx++] = origin_shape[0];  // 0: D
        ncdhw_shape[idx++] = origin_shape[1];  // 1: H
        ncdhw_shape[idx++] = origin_shape[2];  // 2: W
        return true;
    } else {
        OP_LOGE("Conv3DBackpropFilterV2", "this foramt %s is not supported.", ge::TypeUtils::FormatToSerialString(origin_format).c_str());
        return false;
    }
}

static inline bool CheckRange(int32_t val, int32_t lower, int32_t upper)
{
    return val >= lower && val <= upper;
}

namespace {
bool SetStridesAttr(const gert::TilingContext *context, Conv3dBpFilterV2RunInfo &runInfoV2)
{
    const auto op_name = (context->GetNodeName() == nullptr) ? "nil" : context->GetNodeName();
    const auto attrs = context->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(op_name, "failed to get attrs from context."), return false);

    const auto strides = attrs->GetAttrPointer<gert::ContinuousVector>(STRIDES_INDEX);
    OP_CHECK_IF(strides == nullptr, OP_LOGE(op_name, "get strides from context fail."), return false);
    OP_CHECK_IF(strides->GetSize() != CONV_BACKPROP_SHAPE_DIM, 
                OP_LOGE(op_name, "strides length = %zu must be 5.", strides->GetSize()), return false);
    const int64_t *strides_data = static_cast<const int64_t *>(strides->GetData());
    std::vector<int64_t> normalized_strides(strides->GetSize(), 0);
    const ge::Format input_format = context->GetInputDesc(INPUT_DESC)->GetOriginFormat();
    OP_CHECK_IF(!GetNCDHWShape(strides_data, normalized_strides.data(), input_format),
        OP_LOGE(op_name, "GetNCDHWShape failed."), return false);

    // check param limit
    OP_CHECK_IF(normalized_strides[NCDHW_N_INDEX] != 1 || normalized_strides[NCDHW_C_INDEX] != 1, OP_LOGE(op_name,
        "stride_n and stride_c's dim must be 1, current stride_n is %ld, stride_c is %ld.",
        normalized_strides[NCDHW_N_INDEX], normalized_strides[NCDHW_C_INDEX]), return false);
    OP_CHECK_IF(!CheckRange(normalized_strides[NCDHW_D_INDEX], STRIDE_LOWER, SHAPE_UPPER),
        OP_LOGE(op_name, "stride_d is invalid, current is %ld, stride_d support range [%d, %d]",
        normalized_strides[NCDHW_D_INDEX], STRIDE_LOWER, SHAPE_UPPER), return false);
    OP_CHECK_IF(!CheckRange(normalized_strides[NCDHW_H_INDEX], STRIDE_LOWER, STRIDE_UPPER),
        OP_LOGE(op_name, "stride_h is invalid, current is %ld, stride_h support range [%d, %d]",
        normalized_strides[NCDHW_H_INDEX], STRIDE_LOWER, STRIDE_UPPER), return false);
    OP_CHECK_IF(!CheckRange(normalized_strides[NCDHW_W_INDEX], STRIDE_LOWER, STRIDE_UPPER),
        OP_LOGE(op_name, "stride_w is invalid, current is %ld, stride_w support range [%d, %d]",
        normalized_strides[NCDHW_W_INDEX], STRIDE_LOWER, STRIDE_UPPER), return false);
    // stride attrs
    runInfoV2.stride_d = normalized_strides[NCDHW_D_INDEX];
    runInfoV2.stride_h = normalized_strides[NCDHW_H_INDEX];
    runInfoV2.stride_w = normalized_strides[NCDHW_W_INDEX];
    return true;
}

bool SetDilationsAttr(const gert::TilingContext *context, Conv3dBpFilterV2RunInfo &runInfoV2)
{
    const auto op_name = (context->GetNodeName() == nullptr) ? "nil" : context->GetNodeName();
    const auto attrs = context->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(op_name, "failed to get attrs from context."), return false);
    const auto dilations = attrs->GetAttrPointer<gert::ContinuousVector>(DIALTIONS_INDEX);
    OP_CHECK_IF(dilations == nullptr, OP_LOGE(op_name, "get dilations from context fail."), return false);
    OP_CHECK_IF(dilations->GetSize() != CONV_BACKPROP_SHAPE_DIM, 
                OP_LOGE(op_name, "dilations length = %zu must be 5.", dilations->GetSize()), return false);
    const int64_t *dilations_data = static_cast<const int64_t *>(dilations->GetData());
    std::vector<int64_t> normalized_dilations(dilations->GetSize(), 0);
    const ge::Format input_format = context->GetInputDesc(INPUT_DESC)->GetOriginFormat();
    OP_CHECK_IF(!GetNCDHWShape(dilations_data, normalized_dilations.data(), input_format),
        OP_LOGE(op_name, "GetNCDHWShape failed."), return false);

    // check param limit
    OP_CHECK_IF(normalized_dilations[NCDHW_N_INDEX] != 1 || normalized_dilations[NCDHW_C_INDEX] != 1, OP_LOGE(op_name,
        "dilation_n and dilation_c's dim must be 1, current dilation_n is %ld, dilation_c is %ld.",
        normalized_dilations[NCDHW_N_INDEX], normalized_dilations[NCDHW_C_INDEX]), return false);
    OP_CHECK_IF(!CheckRange(normalized_dilations[NCDHW_D_INDEX], DILATION_LOWWER, SHAPE_UPPER),
        OP_LOGE(op_name, "dilation_d is invalid, current is %ld, dilation_d support range [%d, %d]",
        normalized_dilations[NCDHW_D_INDEX], DILATION_LOWWER, SHAPE_UPPER), return false);
    int32_t dilation_limit =
        context->GetCompileInfo<Ops::NN::Conv::Conv3DBackpropV2CompileInfo>()->npuArch == NpuArch::DAV_3510 ? SHAPE_UPPER : DILATION_UPPER;
    OP_CHECK_IF(!CheckRange(normalized_dilations[NCDHW_H_INDEX], DILATION_LOWWER, dilation_limit),
        OP_LOGE(op_name, "dilation_h is invalid, current is %ld, dilation_h support range [%d, %d]",
        normalized_dilations[NCDHW_H_INDEX], DILATION_LOWWER, dilation_limit), return false);
    OP_CHECK_IF(!CheckRange(normalized_dilations[NCDHW_W_INDEX], DILATION_LOWWER, dilation_limit),
        OP_LOGE(op_name, "dilation_w is invalid, current is %ld, dilation_w support range [%d, %d]",
        normalized_dilations[NCDHW_W_INDEX], DILATION_LOWWER, dilation_limit), return false);
    // dilation attrs
    runInfoV2.dilation_d = normalized_dilations[NCDHW_D_INDEX];
    runInfoV2.dilation_h = normalized_dilations[NCDHW_H_INDEX];
    runInfoV2.dilation_w = normalized_dilations[NCDHW_W_INDEX];

    // recal dilation, if kernel d/h/w is equal to 1, dilation d/h/w should be set to 1
    if (runInfoV2.kd == 1) {
        runInfoV2.dilation_d = 1;
    }
    if (runInfoV2.kh == 1) {
        runInfoV2.dilation_h = 1;
    }
    if (runInfoV2.kw  == 1) {
        runInfoV2.dilation_w = 1;
    }
    return true;
}

bool SetGroupsAttrs(const gert::TilingContext *context, Conv3dBpFilterV2RunInfo &runInfoV2, int32_t filter_shape_c)
{
    const auto op_name = (context->GetNodeName() == nullptr) ? "nil" : context->GetNodeName();
    const auto attrs = context->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(op_name, "failed to get attrs from context."), return false);
    const auto groups_attr = attrs->GetAttrPointer<int64_t>(GROUPS_INDEX);
    OP_CHECK_IF(groups_attr == nullptr, OP_LOGE(op_name, "get groups from context fail."), return false);
    int32_t groups = static_cast<int32_t>(*groups_attr);
    OP_CHECK_IF(groups <= 0, OP_LOGE(op_name, "groups [%d] should be greater than 0.", groups), return false);

    int64_t actual_groups = static_cast<int64_t>(runInfoV2.ci) / filter_shape_c;
    if(groups == 1) {
        groups = static_cast<int32_t>(actual_groups);
        OP_LOGD(op_name, "set groups=%d, x_channel(%d) / filter_channel(%d)", groups, runInfoV2.ci, filter_shape_c);
    } else if (actual_groups != static_cast<int64_t>(groups)) {
        OP_LOGE(op_name, "x_channel(%d) / filter_channel(%d) != groups(%d)", runInfoV2.ci, filter_shape_c, groups);
        return false;
    }
    if (runInfoV2.co % groups != 0) {
        OP_LOGE(op_name, "out_channels(%d) %% groups(%d) != 0", runInfoV2.co , groups);
        return false;
    }
    // groups attrs
    int32_t mag_factor = 1;
    if (groups == 1) {
        runInfoV2.real_g = 1;
    } else {
        int64_t mag_factor0 = MathUtil::Lcm(runInfoV2.ci / groups, runInfoV2.k0) / (runInfoV2.ci / groups);
        int64_t mag_factor1 = MathUtil::Lcm(runInfoV2.co / groups, BLOCK_CUBE) / (runInfoV2.co / groups);
        mag_factor = std::min(MathUtil::Lcm(mag_factor0, mag_factor1), static_cast<int64_t>(groups));
        OP_CHECK_IF(mag_factor <= 0, OP_LOGE(op_name, "mag_factor [%d] should be greater than 0.", mag_factor), return false);
        runInfoV2.real_g = (groups + mag_factor - 1) / mag_factor;
    }

    OP_CHECK_IF(groups < 1 || groups > UINT16_MAX, OP_LOGE(op_name, "Groups [%d] is invalid, it should be in range: [1, %d]", groups, UINT16_MAX), return false);
    runInfoV2.groups = groups;
    runInfoV2.mag_factor = mag_factor;

    if (runInfoV2.a_dtype == ge::DT_FLOAT) {
        runInfoV2.cin1_g = Ops::Base::CeilDiv(mag_factor * runInfoV2.ci / groups, static_cast<int32_t>(runInfoV2.k0));
        runInfoV2.cout1_g = Ops::Base::CeilAlign(mag_factor * runInfoV2.co / groups, static_cast<int32_t>(BLOCK_CUBE)) / runInfoV2.k0;
    } else {
        runInfoV2.cin1_g = (mag_factor * runInfoV2.ci / groups + runInfoV2.k0 - 1) / runInfoV2.k0;
        runInfoV2.cout1_g = (mag_factor * runInfoV2.co / groups + runInfoV2.k0 - 1) / runInfoV2.k0;
    }
    runInfoV2.hf32Flag = 0;
    if (runInfoV2.a_dtype == ge::DT_FLOAT && (attrs->GetAttrNum() > ENABLE_HF32_INDEX)) {
        auto enableHf32Ptr = attrs->GetBool(ENABLE_HF32_INDEX);
        OP_CHECK_IF(enableHf32Ptr == nullptr, OP_LOGE(op_name, "get enable_hf32 attr is null"), return false);
        bool enableHf32 = *enableHf32Ptr;
        OP_LOGD(op_name, "attr idx[%zu] enable_hf32 = %d", ENABLE_HF32_INDEX, enableHf32);
        runInfoV2.hf32Flag = static_cast<uint32_t>(enableHf32 ? 1 : 0);
    }
    return true;
}

bool SetFmapDesc(const gert::TilingContext *context, Conv3dBpFilterV2RunInfo &runInfoV2)
{
    const auto op_name = (context->GetNodeName() == nullptr) ? "nil" : context->GetNodeName();
    const gert::CompileTimeTensorDesc *inputDesc = context->GetInputDesc(INPUT_DESC);
    OP_CHECK_IF(inputDesc == nullptr, OP_LOGE(op_name, "failed to get x desc."), return false);
    const ge::Format input_format = inputDesc->GetOriginFormat();
    const gert::StorageShape *inputShape = context->GetInputShape(INPUT_DESC);
    OP_CHECK_IF(inputShape == nullptr, OP_LOGE(op_name, "failed to get x shape."), return false);
    OP_CHECK_IF(CONV_BACKPROP_SHAPE_DIM != inputShape->GetOriginShape().GetDimNum(),
        OP_LOGE(op_name, "x origin shape dims = %zu must be 5", inputShape->GetOriginShape().GetDimNum()), return false);

    std::vector<int64_t> normalized_input_shape(CONV_BACKPROP_SHAPE_DIM, 0);
    OP_CHECK_IF(!GetNCDHWShape(inputShape->GetOriginShape(), normalized_input_shape.data(), input_format),
        OP_LOGE(op_name, "GetNCDHWShape failed."), return false);

    runInfoV2.batch = normalized_input_shape[NCDHW_N_INDEX];
    runInfoV2.ci = normalized_input_shape[NCDHW_C_INDEX];
    runInfoV2.di = normalized_input_shape[NCDHW_D_INDEX];
    runInfoV2.hi = normalized_input_shape[NCDHW_H_INDEX];
    runInfoV2.wi = normalized_input_shape[NCDHW_W_INDEX];
    OP_CHECK_IF(!CheckRange(runInfoV2.batch, SHAPE_LOWER, SHAPE_UPPER),
        OP_LOGE(op_name, "x_shape_batch is invalid, current is %d, x_shape_batch support range [%d, %d]",
        runInfoV2.batch, SHAPE_LOWER, SHAPE_UPPER), return false);
    OP_CHECK_IF(!CheckRange(runInfoV2.ci, SHAPE_LOWER, SHAPE_UPPER),
        OP_LOGE(op_name, "x_shape_c is invalid, current is %d, x_shape_c support range [%d, %d]", runInfoV2.ci, SHAPE_LOWER, SHAPE_UPPER), return false);
    OP_CHECK_IF(!CheckRange(runInfoV2.di, SHAPE_LOWER, SHAPE_UPPER),
        OP_LOGE(op_name, "x_shape_d is invalid, current is %d, x_shape_d support range [%d, %d]", runInfoV2.di, SHAPE_LOWER, SHAPE_UPPER), return false);
    OP_CHECK_IF(!CheckRange(runInfoV2.hi, SHAPE_LOWER, SHAPE_UPPER),
        OP_LOGE(op_name, "x_shape_h is invalid, current is %d, x_shape_h support range [%d, %d]", runInfoV2.hi, SHAPE_LOWER, SHAPE_UPPER), return false);
    OP_CHECK_IF(!CheckRange(runInfoV2.wi, SHAPE_LOWER, SHAPE_UPPER),
        OP_LOGE(op_name, "x_shape_w is invalid, current is %d, x_shape_w support range [%d, %d]", runInfoV2.wi, SHAPE_LOWER, SHAPE_UPPER), return false);

    runInfoV2.a_dtype = inputDesc->GetDataType();
    int32_t a_dtype_bytes = ge::GetSizeByDataType(runInfoV2.a_dtype);
    OP_CHECK_IF(a_dtype_bytes == -1, OP_LOGE(op_name, "x_shape dtype size is invalid"), return false);
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

bool SetGradOutputDesc(const gert::TilingContext *context, Conv3dBpFilterV2RunInfo &runInfoV2)
{
    const auto op_name = (context->GetNodeName() == nullptr) ? "nil" : context->GetNodeName();
    const gert::CompileTimeTensorDesc *gradOutputDesc = context->GetInputDesc(OUT_BACKPROP_DESC);
    OP_CHECK_IF(gradOutputDesc == nullptr, OP_LOGE(op_name, "failed to get out_backprop desc."), return false);
    const ge::Format gradOutputFormat = gradOutputDesc->GetOriginFormat();
    const gert::StorageShape *gradOutputShape = context->GetInputShape(OUT_BACKPROP_DESC);
    OP_CHECK_IF(gradOutputShape == nullptr, OP_LOGE(op_name, "failed to get out_backprop shape."), return false);
    OP_CHECK_IF(CONV_BACKPROP_SHAPE_DIM != gradOutputShape->GetOriginShape().GetDimNum(),
        OP_LOGE(op_name, "out_backprop origin shape dims = %zu must be 5", gradOutputShape->GetOriginShape().GetDimNum()), return false);

    std::vector<int64_t> normalized_dedy_shape(CONV_BACKPROP_SHAPE_DIM, 0);
    OP_CHECK_IF(!GetNCDHWShape(gradOutputShape->GetOriginShape(), normalized_dedy_shape.data(), gradOutputFormat),
        OP_LOGE(op_name, "GetNCDHWShape failed."), return false);
    runInfoV2.batch = normalized_dedy_shape[NCDHW_N_INDEX];
    runInfoV2.co = normalized_dedy_shape[NCDHW_C_INDEX];
    runInfoV2.dout = normalized_dedy_shape[NCDHW_D_INDEX];
    runInfoV2.ho = normalized_dedy_shape[NCDHW_H_INDEX];
    runInfoV2.wo = normalized_dedy_shape[NCDHW_W_INDEX];

    OP_CHECK_IF(!CheckRange(runInfoV2.batch, SHAPE_LOWER, SHAPE_UPPER),
        OP_LOGE(op_name, "out_backprop_batch is invalid, current is %d, out_backprop_batch support range [%d, %d]",
        runInfoV2.batch, SHAPE_LOWER, SHAPE_UPPER), return false);
    OP_CHECK_IF(!CheckRange(runInfoV2.co, SHAPE_LOWER, SHAPE_UPPER),
        OP_LOGE(op_name, "out_backprop_c is invalid, current is %d, out_backprop_c support range [%d, %d]", runInfoV2.co, SHAPE_LOWER, SHAPE_UPPER), return false);
    OP_CHECK_IF(!CheckRange(runInfoV2.dout, SHAPE_LOWER, SHAPE_UPPER),
        OP_LOGE(op_name, "out_backprop_d is invalid, current is %d, out_backprop_d support range [%d, %d]", runInfoV2.dout, SHAPE_LOWER, SHAPE_UPPER), return false);
    OP_CHECK_IF(!CheckRange(runInfoV2.ho, SHAPE_LOWER, SHAPE_UPPER),
        OP_LOGE(op_name, "out_backprop_h is invalid, current is %d, out_backprop_h support range [%d, %d]", runInfoV2.ho, SHAPE_LOWER, SHAPE_UPPER), return false);
    OP_CHECK_IF(!CheckRange(runInfoV2.wo, SHAPE_LOWER, SHAPE_UPPER),
        OP_LOGE(op_name, "out_backprop_w is invalid, current is %d, out_backprop_w support range [%d, %d]", runInfoV2.wo, SHAPE_LOWER, SHAPE_UPPER), return false);

    runInfoV2.b_dtype = gradOutputDesc->GetDataType();
    int32_t b_dtype_bytes = ge::GetSizeByDataType(runInfoV2.b_dtype);
    OP_CHECK_IF(b_dtype_bytes == -1, OP_LOGE(op_name, "filter_shape dtype size is invalid"), return false);
    runInfoV2.b_dtype_bytes = b_dtype_bytes;
    return true;
}

bool SetOutputDesc(const gert::TilingContext *context, Conv3dBpFilterV2RunInfo &runInfoV2)
{
    const auto op_name = (context->GetNodeName() == nullptr) ? "nil" : context->GetNodeName();
    const gert::CompileTimeTensorDesc *outputDesc = context->GetOutputDesc(OUTPUT_DESC);
    OP_CHECK_IF(outputDesc == nullptr, OP_LOGE(op_name, "failed to get output desc."), return false);
    const ge::Format output_format = outputDesc->GetOriginFormat();
    const gert::StorageShape *outputShape = context->GetOutputShape(OUTPUT_DESC);
    OP_CHECK_IF(outputShape == nullptr, OP_LOGE(op_name, "failed to get output shape."), return false);
    OP_CHECK_IF(CONV_BACKPROP_SHAPE_DIM != outputShape->GetOriginShape().GetDimNum(),
        OP_LOGE(op_name, "output origin shape dims = %zu must be 5 ", outputShape->GetOriginShape().GetDimNum()), return false);

    std::vector<int64_t> normalized_output_shape(CONV_BACKPROP_SHAPE_DIM, 0);
    OP_CHECK_IF(!GetNCDHWShape(outputShape->GetOriginShape(), normalized_output_shape.data(), output_format),
        OP_LOGE(op_name, "GetNCDHWShape failed."), return false);

    int32_t filter_shape_ci = normalized_output_shape[NCDHW_C_INDEX];
    OP_CHECK_IF(filter_shape_ci <= 0, OP_LOGE(op_name, "filter_shape_c(%d) should be greater than 0.", filter_shape_ci), return false);
    if (runInfoV2.ci % filter_shape_ci != 0) {
        OP_LOGE(op_name, "x_channel(%d) %% filter_channel(%d) != 0", runInfoV2.ci, filter_shape_ci);
        return false;
    }

    runInfoV2.kd = normalized_output_shape[NCDHW_D_INDEX];
    runInfoV2.kh = normalized_output_shape[NCDHW_H_INDEX];
    runInfoV2.kw = normalized_output_shape[NCDHW_W_INDEX];

    OP_CHECK_IF(!CheckRange(runInfoV2.kd, SHAPE_LOWER, SHAPE_UPPER),
        OP_LOGE(op_name, "output_shape_d is invalid, current is %d, output_shape_d support range [%d, %d]", runInfoV2.kd, SHAPE_LOWER, SHAPE_UPPER), return false);
    OP_CHECK_IF(!CheckRange(runInfoV2.kh, SHAPE_LOWER, SHAPE_UPPER),
        OP_LOGE(op_name, "output_shape_h is invalid, current is %d, output_shape_h support range [%d, %d]", runInfoV2.kh, SHAPE_LOWER, SHAPE_UPPER), return false);
    OP_CHECK_IF(!CheckRange(runInfoV2.kw, SHAPE_LOWER, SHAPE_UPPER),
        OP_LOGE(op_name, "output_shape_w is invalid, current is %d, output_shape_w support range [%d, %d]", runInfoV2.kw, SHAPE_LOWER, SHAPE_UPPER), return false);

    runInfoV2.c_dtype = outputDesc->GetDataType();
    int32_t c_dtype_bytes = ge::GetSizeByDataType(runInfoV2.c_dtype);
    OP_CHECK_IF(c_dtype_bytes == -1, OP_LOGE(op_name, "output_shape dtype size is invalid"), return false);
    runInfoV2.c_dtype_bytes = c_dtype_bytes;

    return SetGroupsAttrs(context, runInfoV2, filter_shape_ci);
}

void ReCalPaddings(Conv3dBpFilterV2RunInfo &runInfoV2, const char *padding)
{
    int32_t kernel_d_dilation = (runInfoV2.kd - 1) * runInfoV2.dilation_d + 1;
    int32_t kernel_h_dilation = (runInfoV2.kh - 1) * runInfoV2.dilation_h + 1;
    int32_t kernel_w_dilation = (runInfoV2.kw - 1) * runInfoV2.dilation_w + 1;
    // if pads is invalid and padding is SAME, calc pads ,considerate 2d to 3d ,exclude p_f and p_b
    if (padding != nullptr && padding[0] == 'S'
        && runInfoV2.pad_u == -1 && runInfoV2.pad_d == -1
        && runInfoV2.pad_l == -1 && runInfoV2.pad_r == -1) {
        int64_t tails_d = runInfoV2.di % runInfoV2.stride_d;
        int64_t tails_h = runInfoV2.hi % runInfoV2.stride_h;
        int64_t tails_w = runInfoV2.wi % runInfoV2.stride_w;
        int64_t pad_d = std::max((tails_d > 0 ? kernel_d_dilation - tails_d : kernel_d_dilation - runInfoV2.stride_d), 0L);
        int64_t pad_h = std::max((tails_h > 0 ? kernel_h_dilation - tails_h : kernel_h_dilation - runInfoV2.stride_h), 0L);
        int64_t pad_w = std::max((tails_w > 0 ? kernel_w_dilation - tails_w : kernel_w_dilation - runInfoV2.stride_w), 0L);
        runInfoV2.pad_f = pad_d / 2;  // 2 means pad_up is half size of pad_d
        runInfoV2.pad_b = pad_d - runInfoV2.pad_f;
        runInfoV2.pad_u = pad_h / 2;  // 2 means pad_up is half size of pad_h
        runInfoV2.pad_d = pad_h - runInfoV2.pad_u;
        runInfoV2.pad_l = pad_w / 2;  // 2 means pad_up is half size of pad_w
        runInfoV2.pad_r = pad_w - runInfoV2.pad_l;
    }
}

bool CheckGradOutputShape(const gert::TilingContext* context, Conv3dBpFilterV2RunInfo& runInfoV2)
{
    const auto op_name = (context->GetNodeName() == nullptr) ? "nil" : context->GetNodeName();
    int64_t do_expect =
        (static_cast<int64_t>(runInfoV2.di) + runInfoV2.pad_f + runInfoV2.pad_b - runInfoV2.dilation_d * (runInfoV2.kd - 1) - 1) /
        runInfoV2.stride_d + 1;
    int64_t ho_expect =
        (static_cast<int64_t>(runInfoV2.hi) + runInfoV2.pad_u + runInfoV2.pad_d - runInfoV2.dilation_h * (runInfoV2.kh - 1) - 1) /
        runInfoV2.stride_h + 1;
    int64_t wo_expect =
        (static_cast<int64_t>(runInfoV2.wi) + runInfoV2.pad_l + runInfoV2.pad_r - runInfoV2.dilation_w * (runInfoV2.kw - 1) - 1) /
        runInfoV2.stride_w + 1;
    OP_CHECK_IF(
        do_expect != runInfoV2.dout, OP_LOGE(op_name,
            "out_backprop's D = %ld is not equal with inferred D = %ld", runInfoV2.dout, do_expect), return false);
    OP_CHECK_IF(
        ho_expect != runInfoV2.ho, OP_LOGE(op_name,
            "out_backprop's H = %ld is not equal with inferred H = %ld", runInfoV2.ho, ho_expect), return false);
    OP_CHECK_IF(
        wo_expect != runInfoV2.wo, OP_LOGE(op_name,
            "out_backprop's W = %ld is not equal with inferred W = %ld", runInfoV2.wo, wo_expect), return false);
    return true;
}

static bool IsArchAfter35(const gert::TilingContext* context)
{
    return context->GetCompileInfo<Ops::NN::Conv::Conv3DBackpropV2CompileInfo>()->npuArch == 
                NpuArch::DAV_3510;
}

bool SetConvBackpropFilterAttrs(const gert::TilingContext *context, Conv3dBpFilterV2RunInfo &runInfoV2)
{
    const auto op_name = (context->GetNodeName() == nullptr) ? "nil" : context->GetNodeName();
    // set strides
    OP_CHECK_IF(!SetStridesAttr(context, runInfoV2), OP_LOGE(op_name, "failed to set strides attrs."), return false);
    // set dilation
    OP_CHECK_IF(!SetDilationsAttr(context, runInfoV2), OP_LOGE(op_name, "failed to set dilation attrs."), return false);
    // set pads
    const auto attrs = context->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(op_name, "failed to get attrs from context."), return false);

    const auto pads = attrs->GetAttrPointer<gert::ContinuousVector>(PADS_INDEX);
    OP_CHECK_IF(pads == nullptr, OP_LOGE(op_name, "get pads from context fail."), return false);
    OP_CHECK_IF(pads->GetSize() != CONV_BACKPROP_PADS_DIM, OP_LOGE(op_name, "pads of context len is invalid."), return false);
    const int64_t *pads_data = static_cast<const int64_t *>(pads->GetData());
    size_t pad_idx = 0;
    runInfoV2.pad_f = pads_data[pad_idx++];
    runInfoV2.pad_b = pads_data[pad_idx++];
    runInfoV2.pad_u = pads_data[pad_idx++];
    runInfoV2.pad_d = pads_data[pad_idx++];
    runInfoV2.pad_l = pads_data[pad_idx++];
    runInfoV2.pad_r = pads_data[pad_idx++];

    if (attrs->GetAttrNum() <= PADDING_INDEX) {
        OP_LOGD(op_name, "no padding attr, skip calc and check");
        return true;
    }

    if (IsArchAfter35(context)) {
        PAD_UPPER = INT32_MAX - 1;
    }

    const char *padding = attrs->GetAttrPointer<char>(PADDING_INDEX);
    ReCalPaddings(runInfoV2, padding);
    OP_CHECK_IF(!CheckRange(runInfoV2.pad_f, PAD_LOWWER, SHAPE_UPPER), OP_LOGE(op_name,
        "pad_f is invalid, current is %d, pad_f support range [%d, %d]", runInfoV2.pad_f, PAD_LOWWER, SHAPE_UPPER), return false);
    OP_CHECK_IF(!CheckRange(runInfoV2.pad_b, PAD_LOWWER, SHAPE_UPPER), OP_LOGE(op_name,
        "pad_b is invalid, current is %d, pad_b support range [%d, %d]", runInfoV2.pad_b, PAD_LOWWER, SHAPE_UPPER), return false);
    OP_CHECK_IF(!CheckRange(runInfoV2.pad_u, PAD_LOWWER, PAD_UPPER), OP_LOGE(op_name,
        "pad_u is invalid, current is %d, pad_u support range is [%d, %d]", runInfoV2.pad_u, PAD_LOWWER, PAD_UPPER), return false);
    OP_CHECK_IF(!CheckRange(runInfoV2.pad_d, PAD_LOWWER, PAD_UPPER), OP_LOGE(op_name,
        "pad_d is invalid, current is %d, pad_d support range is [%d, %d]", runInfoV2.pad_d, PAD_LOWWER, PAD_UPPER), return false);
    OP_CHECK_IF(!CheckRange(runInfoV2.pad_l, PAD_LOWWER, PAD_UPPER), OP_LOGE(op_name,
        "pad_l is invalid, current is %d, pad_l support range is [%d, %d]", runInfoV2.pad_l, PAD_LOWWER, PAD_UPPER), return false);
    OP_CHECK_IF(!CheckRange(runInfoV2.pad_r, PAD_LOWWER, PAD_UPPER), OP_LOGE(op_name,
        "pad_r is invalid, current is %d, pad_r support range is [%d, %d]", runInfoV2.pad_r, PAD_LOWWER, PAD_UPPER), return false);

    if (std::max({runInfoV2.pad_f, runInfoV2.pad_b,
         runInfoV2.pad_u, runInfoV2.pad_d,
         runInfoV2.pad_l, runInfoV2.pad_r}) > 0) {
        OP_LOGD(op_name,
            "gradient calculation in padding regions is shape-dependent."
            "certain kernel optimizations may skip computation and explicitly force these gradients to 0");
    }
    return CheckGradOutputShape(context, runInfoV2);
}
}

bool SetConv3dBpFilterV2RunInfo(const gert::TilingContext *context, Conv3dBpFilterV2RunInfo &runInfoV2)
{
    OP_CHECK_IF(!SetFmapDesc(context, runInfoV2), OP_LOGE("Conv3DBackpropFilterV2", "failed to get x desc."), return false);
    OP_CHECK_IF(!SetGradOutputDesc(context, runInfoV2), OP_LOGE("Conv3DBackpropFilterV2", "failed to get out_backprop desc."), return false);
    OP_CHECK_IF(!SetOutputDesc(context, runInfoV2), OP_LOGE("Conv3DBackpropFilterV2", "failed to get output desc."), return false);
    OP_CHECK_IF(!SetConvBackpropFilterAttrs(context, runInfoV2), OP_LOGE("Conv3DBackpropFilterV2","failed to get attrs."), return false);
    return true;
}

}
}
}
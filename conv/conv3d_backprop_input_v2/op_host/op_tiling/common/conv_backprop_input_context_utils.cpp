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
#include <log/log.h>
#include <util/math_util.h>
#include <unordered_set>
#include "error_util.h"
#include "conv/common/op_host/op_tiling/math_util.h"
#include "conv/common/op_host/op_tiling/platform_util.h"

namespace Ops {
namespace NN {
namespace Conv {

namespace {
// const ---> constexpr : 运行期 ---> 编译器
constexpr size_t kStridesDim = 5;
constexpr size_t kPadsDim = 6;
constexpr size_t kDilationsDim = 5;

// NCDHW
constexpr size_t K_N_DIM_NCDHW = 0;
constexpr size_t K_C_DIM_NCDHW = 1;
constexpr size_t K_D_DIM_NCDHW = 2;
constexpr size_t K_H_DIM_NCDHW = 3;
constexpr size_t K_W_DIM_NCDHW = 4;
// NDHWC
constexpr size_t K_N_DIM_NDHWC = 0;
constexpr size_t K_D_DIM_NDHWC = 1;
constexpr size_t K_H_DIM_NDHWC = 2;
constexpr size_t K_W_DIM_NDHWC = 3;
constexpr size_t K_C_DIM_NDHWC = 4;
// dilation
constexpr int32_t K_DEFAULT_DILATIONS = 1;
constexpr int32_t kDilationLow = 1;
constexpr int32_t kDilationUp = 255;
// stride
constexpr int32_t kDimUp = 2147483647; // 2G - 1
constexpr int32_t kStrideHWUp = 63;
constexpr int32_t kStrideDUp = 255;
constexpr int32_t K_DEFAULT_STRIDES = 1;
constexpr int32_t kDimLow = 1;
// pad
constexpr int32_t kPadUp = 255;
constexpr size_t K_CONV3D_PAD_HEAD_IDX = 0;
constexpr size_t K_CONV3D_PAD_TAIL_IDX = 1;
constexpr size_t K_CONV3D_PAD_UP_IDX = 2;
constexpr size_t K_CONV3D_PAD_DOWN_IDX = 3;
constexpr size_t K_CONV3D_PAD_LEFT_IDX = 4;
constexpr size_t K_CONV3D_PAD_RIGHT_IDX = 5;

// params index
constexpr size_t INPUT_SIZE_INDEX = 0;
constexpr size_t FILTER_INDEX = 1;
constexpr size_t OUT_BACKPROP_INDEX = 2;
constexpr size_t Y_INDEX = 0;
constexpr size_t K_OUTPUT_PADDING_CONV3D_TRANSPOSE_IDX = 5;
constexpr size_t K_ORI_SHAPE_DIM_2D = 4;
constexpr size_t K_ORI_SHAPE_DIM_3D = 5;
constexpr size_t K_OFFSET_X_CONV3D_TRANSPOSE_IDX = 6;
constexpr size_t K_FUSION_MODE_CONV3D_TRANSPOSE_IDX = 7;
constexpr size_t K_Y_QUANT_MODE_CONV3D_TRANSPOSE_IDX = 8;

// NDC1HWC0
constexpr size_t kNDimNDC1HWC0Idx = 0;
// FRACTAL_Z_3D
constexpr size_t kDkCin1HkWkFRACTALZ3DIdx = 0;
constexpr size_t kCo1FRACTALZ3DIdx = 1;
constexpr size_t kCo0FRACTALZ3DIdx = 2;
constexpr size_t kCin0FRACTALZ3DIdx = 3;
constexpr size_t kPaddingConv3dBpInputIdx = 5;
constexpr size_t kPaddingConv3dTransposeIdx = 7;
constexpr size_t kPaddingExtendConvTransposeIdx = 9;

constexpr int32_t kBlockSize = 16;
const int32_t BYTE_BLOCK = 32;
constexpr int32_t kBit8BlockReduce = 32;
constexpr int32_t kFP32BlockReduce = 8;
const std::map<int32_t, int32_t> kDtypeBlockReduceMap = {
  {ge::DT_HIFLOAT8, kBit8BlockReduce}, {ge::DT_FLOAT8_E4M3FN, kBit8BlockReduce},
  {ge::DT_FLOAT16, kBlockSize}, {ge::DT_FLOAT, kFP32BlockReduce}, {ge::DT_INT8, kBit8BlockReduce}
};
constexpr int32_t kNumTwo = 2;
constexpr int32_t kFilterDimHWUp = 511;
constexpr int32_t kDimBatchUp = ((1UL << 31) - 1);
constexpr int32_t kDimWNormalUp = 4096;
constexpr int32_t kGroupUp = 65535;
constexpr int64_t kDataSizeMax = ((1UL << 63) - 1);
constexpr int32_t DIM_LOW = 1;
constexpr int32_t STRIDES_DIM_HW_UP = 63;
constexpr int32_t STRIDES_DIM_DEPTH_UP = 255;
constexpr int32_t GROUPS_LOW = 1;
constexpr int32_t GROUPS_UP = 65535;
constexpr int32_t PAD_DIM_UP = 255;
constexpr int32_t PAD_DIM_LOW = 0;
constexpr size_t BAIS_INDEX = 3;
constexpr size_t OFFSET_W_INDEX = 4;
constexpr size_t OFFSET_X_INDEX = 6;
constexpr size_t OUTPUT_PADDING_INDEX = 5;
constexpr size_t OUTPUT_PADDING_DIM = 5;

constexpr uint64_t CHAR_0 = 48UL;

class Shape {
public:
    int64_t batch = 1;
    int64_t c = 16;
    int64_t d = 1;
    int64_t h = 1;
    int64_t w = 1;
    int64_t c1 = 1;
    int64_t c0 = 16;
};

// output_padding
struct OutputPadding
{
    int32_t output_padding_d = 0;
    int32_t output_padding_h = 0;
    int32_t output_padding_w = 0;
};
struct OtherParams {
    OutputPadding output_padding;
    int32_t groups = 1;
    int32_t stride_expand_flag = 0;
    int32_t dilation_d_gt_one_flag = 0;
    ge::DataType a_dtype = ge::DT_FLOAT16;
    ge::DataType b_dtype = ge::DT_FLOAT16;
    ge::DataType c_dtype = ge::DT_FLOAT16;
    Shape a_shape;
    Shape b_shape;
    Shape c_shape;
    int64_t filter_gdkci1ghw = 0;
    int32_t co1g = 0;
    int32_t ci1g = 0;
    int32_t filter_co0 = 16;  // co0 in fractal_z
    int32_t filter_ci0 = 16;  // cin0 in fractal_z
    int32_t co1g_reduce = 0;  // co1g calculated by block_reduce depend on dtype
    int64_t filter_d_dilation = 1;
    int64_t filter_h_dilation = 1;
    int64_t filter_w_dilation = 1;
    int32_t multiple_extend = 0;
    int32_t pad_head_before = 0;
    int32_t pad_up_before = 0;
    int32_t pad_left_before = 0;
    int32_t pad_tail_after = 0;
    int32_t pad_down_after = 0;
    int32_t pad_right_after = 0;
    int32_t shape_up_modify = 0;
    int32_t shape_left_modify = 0;
    int32_t shape_down_modify;
    int32_t shape_right_modify;
    int64_t fmap_d_padding = 0;
    int64_t fmap_h_padding = 0;
    int64_t fmap_w_padding = 0;
};

bool CheckRangeInt64(int64_t value, int32_t value_low, int32_t value_up) {
    if (value < value_low || value > value_up) {
        return false;
    }
    return true;
}

static bool IsArchAfter35(const gert::TilingContext *context) {
  return context->GetCompileInfo<Ops::NN::Conv::Conv3DBackpropV2CompileInfo>()->shortSocVersion ==
            platform_ascendc::SocVersion::ASCEND910_95;
}

bool ValidateConvBackpropContext(const gert::TilingContext *context) {
    // 校验输入张量描述是否获取成功
    auto input_size_desc = context->GetInputDesc(INPUT_SIZE_INDEX);
    OP_CHECK_IF(input_size_desc == nullptr, OP_LOGE(context->GetNodeName(), "get input_size desc from context fail."), return false);
    auto filter_desc = context->GetInputDesc(FILTER_INDEX);
    OP_CHECK_IF(filter_desc == nullptr, OP_LOGE(context->GetNodeName(), "get filter desc from context fail."), return false);
    auto out_backprop_desc = context->GetInputDesc(OUT_BACKPROP_INDEX);
    OP_CHECK_IF(out_backprop_desc == nullptr, OP_LOGE(context->GetNodeName(), "get out_backprop desc from context fail."), return false);
    // 校验输出张量描述是否获取成功
    auto y_desc = context->GetOutputDesc(Y_INDEX);
    OP_CHECK_IF(y_desc == nullptr, OP_LOGE(context->GetNodeName(), "get y desc from context fail."), return false);
    // 校验属性参数获取是否成功
    auto attrs = context->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(context->GetNodeName(), "failed to get runtime attrs."), return false);

    return true;
}

bool CheckAttrRangeDilations(const gert::TilingContext *context, const int64_t *dilations) {
    const auto op_name = context->GetNodeName();
    auto y_ori_format = context->GetOutputDesc(Y_INDEX)->GetOriginFormat();
    int32_t kDilationUpTmp = (IsArchAfter35(context) || IsSocVersionFuse(context)) ? kDimUp : kDilationUp;
    if (y_ori_format == ge::FORMAT_NCDHW) {
      OP_CHECK_IF(
        !CheckRangeInt64(dilations[K_N_DIM_NCDHW], K_DEFAULT_DILATIONS, K_DEFAULT_DILATIONS),
        OP_LOGE(op_name, "dilation_n value [%ld] is invalid, support range [%d, %d]",
                dilations[K_N_DIM_NCDHW], K_DEFAULT_DILATIONS, K_DEFAULT_DILATIONS),
        return false);
      OP_CHECK_IF(
        !CheckRangeInt64(dilations[K_C_DIM_NCDHW], K_DEFAULT_DILATIONS, K_DEFAULT_DILATIONS),
        OP_LOGE(op_name, "dilation_c value [%ld] is invalid, support range [%d, %d]",
                dilations[K_C_DIM_NCDHW], K_DEFAULT_DILATIONS, K_DEFAULT_DILATIONS),
        return false);
      OP_CHECK_IF(
        !CheckRangeInt64(dilations[K_D_DIM_NCDHW], kDilationLow, kDilationUpTmp),
        OP_LOGE(op_name, "dilation_d value [%ld] is invalid, support range [%d, %d]",
                dilations[K_D_DIM_NCDHW], kDilationLow, kDilationUpTmp),
        return false);
      OP_CHECK_IF(
        !CheckRangeInt64(dilations[K_H_DIM_NCDHW], kDilationLow, kDilationUpTmp),
        OP_LOGE(op_name, "dilation_h value [%ld] is invalid, support range [%d, %d]",
                dilations[K_H_DIM_NCDHW], kDilationLow, kDilationUpTmp),
        return false);
      OP_CHECK_IF(
        !CheckRangeInt64(dilations[K_W_DIM_NCDHW], kDilationLow, kDilationUpTmp),
        OP_LOGE(op_name, "dilation_w value [%ld] is invalid, support range [%d, %d]",
                dilations[K_W_DIM_NCDHW], kDilationLow, kDilationUpTmp),
        return false);
    } else {
      OP_CHECK_IF(
        !CheckRangeInt64(dilations[K_N_DIM_NDHWC], K_DEFAULT_DILATIONS, K_DEFAULT_DILATIONS),
        OP_LOGE(op_name, "dilation_n value [%ld] is invalid, support range [%d, %d]",
                dilations[K_N_DIM_NDHWC], K_DEFAULT_DILATIONS, K_DEFAULT_DILATIONS),
        return false);
      OP_CHECK_IF(
        !CheckRangeInt64(dilations[K_C_DIM_NDHWC], K_DEFAULT_DILATIONS, K_DEFAULT_DILATIONS),
        OP_LOGE(op_name, "dilation_c value [%ld] is invalid, support range [%d, %d]",
                dilations[K_C_DIM_NDHWC], K_DEFAULT_DILATIONS, K_DEFAULT_DILATIONS),
        return false);
      OP_CHECK_IF(
        !CheckRangeInt64(dilations[K_D_DIM_NDHWC], kDilationLow, kDilationUpTmp),
        OP_LOGE(op_name, "dilation_d value [%ld] is invalid, support range [%d, %d]",
                dilations[K_D_DIM_NDHWC], kDilationLow, kDilationUpTmp),
        return false);
      OP_CHECK_IF(
        !CheckRangeInt64(dilations[K_H_DIM_NDHWC], kDilationLow, kDilationUpTmp),
        OP_LOGE(op_name, "dilation_h value [%ld] is invalid, support range [%d, %d]",
                dilations[K_H_DIM_NDHWC], kDilationLow, kDilationUpTmp),
        return false);
      OP_CHECK_IF(
        !CheckRangeInt64(dilations[K_W_DIM_NDHWC], kDilationLow, kDilationUpTmp),
        OP_LOGE(op_name, "dilation_w value [%ld] is invalid, support range [%d, %d]",
                dilations[K_W_DIM_NDHWC], kDilationLow, kDilationUpTmp),
        return false);
    }

    return true;
}

bool CheckAttrRangeStrides(const gert::TilingContext *context, const int64_t *strides) {
    const auto op_name = context->GetNodeName();
    auto y_ori_format = context->GetOutputDesc(Y_INDEX)->GetOriginFormat();
    if (y_ori_format == ge::FORMAT_NCDHW) {
      OP_CHECK_IF(
        !CheckRangeInt64(strides[K_N_DIM_NCDHW], K_DEFAULT_STRIDES, K_DEFAULT_STRIDES),
        OP_LOGE(op_name, "stride_n value [%ld] is invalid, support range [%d, %d]",
                strides[K_N_DIM_NCDHW], K_DEFAULT_STRIDES, K_DEFAULT_STRIDES),
        return false);
      OP_CHECK_IF(
        !CheckRangeInt64(strides[K_C_DIM_NCDHW], K_DEFAULT_STRIDES, K_DEFAULT_STRIDES),
        OP_LOGE(op_name, "stride_c value [%ld] is invalid, support range [%d, %d]",
                strides[K_C_DIM_NCDHW], K_DEFAULT_STRIDES, K_DEFAULT_STRIDES),
        return false);
      OP_CHECK_IF(
        !CheckRangeInt64(strides[K_D_DIM_NCDHW], kDimLow, kDimUp),
        OP_LOGE(op_name, "stride_d value [%ld] is invalid, support range [%d, %d]",
                strides[K_D_DIM_NCDHW], kDimLow, kDimUp),
        return false);
      OP_CHECK_IF(
        !CheckRangeInt64(strides[K_H_DIM_NCDHW], kDimLow, kDimUp),
        OP_LOGE(op_name, "stride_h value [%ld] is invalid, support range [%d, %d]",
                strides[K_H_DIM_NCDHW], kDimLow, kDimUp),
        return false);
      OP_CHECK_IF(
        !CheckRangeInt64(strides[K_W_DIM_NCDHW], kDimLow, kDimUp),
        OP_LOGE(op_name, "stride_w value [%ld] is invalid, support range [%d, %d]",
                strides[K_W_DIM_NCDHW], kDimLow, kDimUp),
        return false);
    } else {
      OP_CHECK_IF(
        !CheckRangeInt64(strides[K_N_DIM_NDHWC], K_DEFAULT_STRIDES, K_DEFAULT_STRIDES),
        OP_LOGE(op_name, "stride_n value [%ld] is invalid, support range [%d, %d]",
                strides[K_N_DIM_NDHWC], K_DEFAULT_STRIDES, K_DEFAULT_STRIDES),
        return false);
      OP_CHECK_IF(
        !CheckRangeInt64(strides[K_C_DIM_NDHWC], K_DEFAULT_STRIDES, K_DEFAULT_STRIDES),
        OP_LOGE(op_name, "stride_c value [%ld] is invalid, support range [%d, %d]",
                strides[K_C_DIM_NDHWC], K_DEFAULT_STRIDES, K_DEFAULT_STRIDES),
        return false);
      OP_CHECK_IF(
        !CheckRangeInt64(strides[K_D_DIM_NDHWC], kDimLow, kDimUp),
        OP_LOGE(op_name, "stride_d value [%ld] is invalid, support range [%d, %d]",
                strides[K_D_DIM_NDHWC], kDimLow, kDimUp),
        return false);
      OP_CHECK_IF(
        !CheckRangeInt64(strides[K_H_DIM_NDHWC], kDimLow, kDimUp),
        OP_LOGE(op_name, "stride_h value [%ld] is invalid, support range [%d, %d]",
                strides[K_H_DIM_NDHWC], kDimLow, kDimUp),
        return false);
      OP_CHECK_IF(
        !CheckRangeInt64(strides[K_W_DIM_NDHWC], kDimLow, kDimUp),
        OP_LOGE(op_name, "stride_w value [%ld] is invalid, support range [%d, %d]",
                strides[K_W_DIM_NDHWC], kDimLow, kDimUp),
        return false);
    }

    return true;
}

bool CheckAttrRangePads(const gert::TilingContext *context, const int64_t *pads) {
    const auto op_name = context->GetNodeName();
    int32_t kPadUpTmp = kDimUp;
    if (IsArchAfter35(context) || IsSocVersionFuse(context)) {
      kPadUpTmp = kPadUp;
    }
    OP_CHECK_IF(
      !CheckRangeInt64(pads[K_CONV3D_PAD_HEAD_IDX], 0, kPadUpTmp),
      OP_LOGE(op_name, "pad_h value [%ld] is invalid, support range [%d, %d]",
              pads[K_CONV3D_PAD_HEAD_IDX], 0, kPadUpTmp),
      return false);
    OP_CHECK_IF(
      !CheckRangeInt64(pads[K_CONV3D_PAD_TAIL_IDX], 0, kPadUpTmp),
      OP_LOGE(op_name, "pad_t value [%ld] is invalid, support range [%d, %d]",
              pads[K_CONV3D_PAD_TAIL_IDX], 0, kPadUpTmp),
      return false);
    OP_CHECK_IF(
      !CheckRangeInt64(pads[K_CONV3D_PAD_UP_IDX], 0, kPadUp),
      OP_LOGE(op_name, "pad_u value [%ld] is invalid, support range [%d, %d]",
              pads[K_CONV3D_PAD_UP_IDX], 0, kPadUp),
      return false);
    OP_CHECK_IF(
      !CheckRangeInt64(pads[K_CONV3D_PAD_DOWN_IDX], 0, kPadUp),
      OP_LOGE(op_name, "pad_d value [%ld] is invalid, support range [%d, %d]",
              pads[K_CONV3D_PAD_DOWN_IDX], 0, kPadUp),
      return false);
    OP_CHECK_IF(
      !CheckRangeInt64(pads[K_CONV3D_PAD_LEFT_IDX], 0, kPadUp),
      OP_LOGE(op_name, "pad_l value [%ld] is invalid, support range [%d, %d]",
              pads[K_CONV3D_PAD_LEFT_IDX], 0, kPadUp),
      return false);
    OP_CHECK_IF(
      !CheckRangeInt64(pads[K_CONV3D_PAD_RIGHT_IDX], 0, kPadUp),
      OP_LOGE(op_name, "pad_r value [%ld] is invalid, support range [%d, %d]",
              pads[K_CONV3D_PAD_RIGHT_IDX], 0, kPadUp),
      return false);

    return true;
}

bool CheckAttrRange(gert::TilingContext *context,
                    const int64_t *strides,   const int64_t *pads,
                    const int64_t *dilations, const int64_t *groups) {
    const auto op_name = context->GetNodeName();
    OP_CHECK_IF(!CheckAttrRangeDilations(context, dilations),
      OP_LOGE(op_name, "check dilations range failed"),
      return false);
    OP_CHECK_IF(!CheckAttrRangeStrides(context, strides),
      OP_LOGE(op_name, "check strides range failed"),
      return false);
    //Exclude (pad_u,pad_d,pad_l,pad_r) =-1 while paddings is "SAME"
    if (pads[K_CONV3D_PAD_UP_IDX] !=-1 &&
        pads[K_CONV3D_PAD_DOWN_IDX] !=-1 &&
        pads[K_CONV3D_PAD_LEFT_IDX] !=-1 &&
        pads[K_CONV3D_PAD_RIGHT_IDX] != -1
    ) {
      OP_CHECK_IF(!CheckAttrRangePads(context, pads),
      OP_LOGE(op_name, "check pads range failed"),
      return false);
    }

    // groups: [1, 2G - 1 ]
    if (groups != nullptr) {
      OP_CHECK_IF(!CheckRangeInt64(*groups, kDimLow, kDimUp),
        OP_LOGE(op_name, "group value [%ld] is invalid, support range [%d, %d]", *groups, kDimLow, kDimUp),
        return false);
    }

    return true;
}

bool CheckTransposeAttr(gert::TilingContext *context, OtherParams& otherParams) {
    auto yDesc = context->GetOutputDesc(Y_INDEX);
    auto attrs = context->GetAttrs();
    auto outputPadding = attrs->GetAttrPointer<gert::ContinuousVector>(K_OUTPUT_PADDING_CONV3D_TRANSPOSE_IDX);
    OP_CHECK_IF(outputPadding == nullptr, OP_LOGE(context, "failed to get output_padding attrs"), return false);
    OP_CHECK_IF(outputPadding->GetSize() != K_ORI_SHAPE_DIM_3D,
      OP_LOGE(context, "The output_padding should be 5d, actual dim num: %zu", outputPadding->GetSize()), return false);
    const auto outputPaddingData = static_cast<const int64_t *>(outputPadding->GetData());
    if (yDesc->GetOriginFormat() == ge::FORMAT_NCDHW) {
      OP_CHECK_IF(outputPaddingData[K_N_DIM_NCDHW] != 0 || outputPaddingData[K_C_DIM_NCDHW] != 0,
        OP_LOGE(context, "N/C output_padding should be zero.\n"), return false);
      otherParams.output_padding.output_padding_d = outputPaddingData[K_D_DIM_NCDHW];
      otherParams.output_padding.output_padding_h = outputPaddingData[K_H_DIM_NCDHW];
      otherParams.output_padding.output_padding_w = outputPaddingData[K_W_DIM_NCDHW];
    } else {  // yDesc->GetOriginFormat() == ge::FORMAT_NDHWC
      OP_CHECK_IF(outputPaddingData[K_N_DIM_NDHWC] != 0 || outputPaddingData[K_C_DIM_NDHWC] != 0,
        OP_LOGE(context, "N/C output_padding should be zero.\n"), return false);
      otherParams.output_padding.output_padding_d = outputPaddingData[K_D_DIM_NDHWC];
      otherParams.output_padding.output_padding_h = outputPaddingData[K_H_DIM_NDHWC];
      otherParams.output_padding.output_padding_w = outputPaddingData[K_W_DIM_NDHWC];
    }
    if (attrs->GetAttrNum() > K_OFFSET_X_CONV3D_TRANSPOSE_IDX) {
      const auto offsetX = attrs->GetAttrPointer<int64_t>(K_OFFSET_X_CONV3D_TRANSPOSE_IDX);
      OP_CHECK_IF(offsetX == nullptr, OP_LOGE(context, "failed to get offsetX attrs"), return false);
      OP_CHECK_IF(*offsetX != 0, OP_LOGE(context, "offsetX:%ld is invalid, it should be 0", *offsetX), return false);
    }
    if (IsSocVersionFuse(context)) {
      if (attrs->GetAttrNum() > K_FUSION_MODE_CONV3D_TRANSPOSE_IDX) {
        const auto fusion_mode = attrs->GetAttrPointer<int32_t>(K_FUSION_MODE_CONV3D_TRANSPOSE_IDX);
        OP_LOGE_IF(fusion_mode == nullptr, false, context, "failed to get fusion_mode attrs");
        OP_LOGE_IF(*fusion_mode != 0 && *fusion_mode != 1, false, context, "fusion_mode:%d is invalid, it should be 0 or 1", *fusion_mode);
      }
      if (attrs->GetAttrNum() > K_Y_QUANT_MODE_CONV3D_TRANSPOSE_IDX) {
        const auto y_quant_mode = attrs->GetAttrPointer<int32_t>(K_Y_QUANT_MODE_CONV3D_TRANSPOSE_IDX);
        OP_LOGE_IF(y_quant_mode == nullptr, false, context, "failed to get quant_mode attrs");
        OP_LOGE_IF(*y_quant_mode != 0, false, context, "quant_mode:%d is invalid, it should be 0", *y_quant_mode);
      }
    }
    return true;
}

template <typename T>
static void GetNCDHWShape(const T &origin_shape, Shape &ncdhw_shape, const ge::Format &origin_format) {
  // caller already checked buffer size
  if (origin_format == ge::FORMAT_NDHWC) {
    ncdhw_shape.batch = origin_shape[0];  // 0: N
    ncdhw_shape.c = origin_shape[4];  // 4: C
    ncdhw_shape.d = origin_shape[1];  // 1: D
    ncdhw_shape.h = origin_shape[2];  // 2: H
    ncdhw_shape.w = origin_shape[3];  // 3: W
  } else if (origin_format == ge::FORMAT_NCDHW) {
    ncdhw_shape.batch = origin_shape[0];  // 0: N
    ncdhw_shape.c = origin_shape[1];  // 1: C
    ncdhw_shape.d = origin_shape[2];  // 2: D
    ncdhw_shape.h = origin_shape[3];  // 3: H
    ncdhw_shape.w = origin_shape[4];  // 4: W
  } else if (origin_format == ge::FORMAT_DHWCN) {
    ncdhw_shape.batch = origin_shape[4];  // 4: N
    ncdhw_shape.c = origin_shape[3];  // 3: C
    ncdhw_shape.d = origin_shape[0];  // 0: D
    ncdhw_shape.h = origin_shape[1];  // 1: H
    ncdhw_shape.w = origin_shape[2];  // 2: W
  }
}

static bool CheckTransposeOutputdingRange(gert::TilingContext *context, Conv3dBpInputV2RunInfo& runInfoV2, OtherParams& otherParams)
{
  // outputPadding值需要小于同维度dilation或stride
  OP_CHECK_IF(
    (otherParams.output_padding.output_padding_d >= runInfoV2.stride_d && otherParams.output_padding.output_padding_d >= runInfoV2.dilation_d),
    OP_LOGE(context, "output_padding_d value[%d] should smaller than dilation_d[%d] or stride_d[%d]",
            otherParams.output_padding.output_padding_d, runInfoV2.dilation_d, runInfoV2.stride_d),
    return false);
  OP_CHECK_IF(
    (otherParams.output_padding.output_padding_h >= runInfoV2.stride_h && otherParams.output_padding.output_padding_h >= runInfoV2.dilation_h),
    OP_LOGE(context, "output_padding_h value[%d] should smaller than dilation_h[%d] or stride_h[%d]",
            otherParams.output_padding.output_padding_h, runInfoV2.dilation_h, runInfoV2.stride_h),
    return false);
  OP_CHECK_IF(
    (otherParams.output_padding.output_padding_w >= runInfoV2.stride_w && otherParams.output_padding.output_padding_w >= runInfoV2.dilation_w),
    OP_LOGE(context, "output_padding_w value[%d] should smaller than dilation_w[%d] or stride_w[%d]",
            otherParams.output_padding.output_padding_w, runInfoV2.dilation_w, runInfoV2.stride_w),
    return false);

  return true;
}

static bool UpdateDtypeParams(const gert::TilingContext *context, Conv3dBpInputV2RunInfo& runInfoV2,
                              const optiling::OpTypeV2 op_type, OtherParams& otherParams) {
  const auto op_name = context->GetNodeName();

  if (op_type == optiling::OpTypeV2::kConv3DTransposeV2) {
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

  bool isFp16Flag = otherParams.a_dtype == ge::DT_FLOAT16 &&
    otherParams.b_dtype == ge::DT_FLOAT16 && otherParams.c_dtype == ge::DT_FLOAT16;
  bool isFp32Flag = otherParams.a_dtype == ge::DT_FLOAT &&
    otherParams.b_dtype == ge::DT_FLOAT && otherParams.c_dtype == ge::DT_FLOAT;
  bool dtypeSupportFlag = isFp16Flag || isFp32Flag;
  string dtypeCheckLog = "fp16 and fp32";
  if (IsArchAfter35(context) || IsSocVersionFuse(context)) {
    bool isHiF8Flag = otherParams.a_dtype == ge::DT_HIFLOAT8 &&
      otherParams.b_dtype == ge::DT_HIFLOAT8 && otherParams.c_dtype == ge::DT_HIFLOAT8;
    bool isFp8E4M3Flag = otherParams.a_dtype == ge::DT_FLOAT8_E4M3FN &&
      otherParams.b_dtype == ge::DT_FLOAT8_E4M3FN &&
      otherParams.c_dtype == ge::DT_FLOAT8_E4M3FN;
    dtypeSupportFlag = isHiF8Flag || isFp8E4M3Flag || isFp16Flag || isFp32Flag;
    dtypeCheckLog = "hifloat8, float8_e4m3, fp16 and fp32";
  }
  if (IsSocVersionFuse(context)) {
    bool isInt8Flag = otherParams.a_dtype == ge::DT_INT8 && otherParams.b_dtype == ge::DT_INT8 &&
      (otherParams.c_dtype == ge::DT_INT8 || otherParams.c_dtype == ge::DT_FLOAT16);
    bool isFp16Int8Flag = otherParams.a_dtype == ge::DT_FLOAT16 && otherParams.b_dtype == ge::DT_INT8 &&
      (otherParams.c_dtype == ge::DT_INT8 || otherParams.c_dtype == ge::DT_FLOAT16);
    dtypeSupportFlag = isFp16Flag || isInt8Flag || isFp16Int8Flag;
    dtypeCheckLog = "fp16 and int8";
  }
  OP_CHECK_IF(
    !dtypeSupportFlag,
    OP_LOGE(op_name, "out_backprop/fitler/y dtype only support %s now, get actual out_backprop dtype[%s] fitler dtype[%s] y dtype[%s].",
            dtypeCheckLog.c_str(),
            ge::TypeUtils::DataTypeToSerialString(otherParams.a_dtype).c_str(),
            ge::TypeUtils::DataTypeToSerialString(otherParams.b_dtype).c_str(),
            ge::TypeUtils::DataTypeToSerialString(otherParams.c_dtype).c_str()),
    return false);

  runInfoV2.a_dtype_bytes = ge::GetSizeByDataType(otherParams.a_dtype);
  OP_CHECK_IF(runInfoV2.a_dtype_bytes == -1,
    OP_LOGE(op_name, "out_backprop dtype size is invalid"), return false);
  runInfoV2.b_dtype_bytes = ge::GetSizeByDataType(otherParams.b_dtype);
  OP_CHECK_IF(runInfoV2.b_dtype_bytes == -1,
    OP_LOGE(op_name, "filter dtype size is invalid"), return false);
  runInfoV2.c_dtype_bytes = ge::GetSizeByDataType(otherParams.c_dtype);
  OP_CHECK_IF(runInfoV2.c_dtype_bytes == -1,
    OP_LOGE(op_name, "y dtype size is invalid"), return false);

  return true;
}

static bool CheckStorageFormat(const gert::TilingContext *context, size_t filter_input_index, size_t out_backprop_input_index) {
  // 获取输入输出的描述信息
  const auto out_backprop_desc = context->GetInputDesc(out_backprop_input_index);
  const auto filter_desc = context->GetInputDesc(filter_input_index);
  const auto y_desc = context->GetOutputDesc(Y_INDEX);
  // 获取输入输出的format信息
  auto out_backprop_format = static_cast<ge::Format>(ge::GetPrimaryFormat(out_backprop_desc->GetStorageFormat()));
  auto filter_format = static_cast<ge::Format>(ge::GetPrimaryFormat(filter_desc->GetStorageFormat()));
  auto y_format = static_cast<ge::Format>(ge::GetPrimaryFormat(y_desc->GetStorageFormat()));
  const auto op_name = context->GetNodeName();

  const std::unordered_set<ge::Format> valid_out_bp_format = {ge::FORMAT_NCDHW, ge::FORMAT_NDHWC};
  const std::unordered_set<ge::Format> valid_filter_format = {ge::FORMAT_NCDHW, ge::FORMAT_NDHWC, ge::FORMAT_DHWCN};
  const std::unordered_set<ge::Format> valid_y_format = {ge::FORMAT_NCDHW, ge::FORMAT_NDHWC};
  bool invalid_tag = valid_out_bp_format.count(out_backprop_format) == 0 || valid_filter_format.count(filter_format) == 0 ||
                     valid_y_format.count(y_format) == 0;
  if (IsArchAfter35(context) || IsSocVersionFuse(context)) {
    OP_CHECK_IF(invalid_tag,
                OP_LOGE(op_name, "out_backprop format[%s] and y format[%s] should be NCDHW/NDHWC, filter format[%s] should be NCDHW/NDHWC/DHWCN.",
                        ge::TypeUtils::FormatToSerialString(out_backprop_format).c_str(),
                        ge::TypeUtils::FormatToSerialString(y_format).c_str(),
                        ge::TypeUtils::FormatToSerialString(filter_format).c_str()),
                return false);
  }
  return true;
}

static bool UpdateShapeParams(const gert::TilingContext *context, const Conv3dBpInputV2RunInfo &runInfoV2,
                              const Shape &out_backprop_shape_ncdhw, const Shape &filter_shape_ncdhw,
                              const Shape &y_shape_ncdhw, OtherParams& otherParams) {
  const auto op_name = context->GetNodeName();
  OP_CHECK_IF(kDtypeBlockReduceMap.find(otherParams.a_dtype) == kDtypeBlockReduceMap.end() ||
              kDtypeBlockReduceMap.find(otherParams.c_dtype) == kDtypeBlockReduceMap.end() ||
              kDtypeBlockReduceMap.find(otherParams.b_dtype) == kDtypeBlockReduceMap.end(),
            OP_LOGE(op_name, "dtype is invalid!"),
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

static bool CalShapeInfoFromDesc(const gert::TilingContext *context, size_t filter_input_index,
                                 size_t out_backprop_input_index, Conv3dBpInputV2RunInfo &runInfoV2, OtherParams& otherParams) {
  auto filter_desc = context->GetInputDesc(filter_input_index);
  auto out_backprop_desc = context->GetInputDesc(out_backprop_input_index);
  auto y_desc = context->GetOutputDesc(Y_INDEX);

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
      otherParams.multiple_extend * otherParams.b_shape.batch / runInfoV2.groups,
      otherParams.b_shape.c0);
    otherParams.filter_ci0 = kBlockSize;
  }
  otherParams.co1g_reduce = otherParams.co1g;

  auto out_backprop_ori_format = out_backprop_desc->GetOriginFormat();
  auto filter_ori_format = filter_desc->GetOriginFormat();
  auto y_ori_format = y_desc->GetOriginFormat();

  const auto &out_backprop_ori_shape = out_backprop_shape->GetOriginShape();
  const auto &filter_ori_shape = filter_shape->GetOriginShape();
  const auto &y_ori_shape = y_shape->GetOriginShape();
  const auto op_name = context->GetNodeName();
  OP_CHECK_IF(out_backprop_ori_shape.GetDimNum() != K_ORI_SHAPE_DIM_3D,
              OP_LOGE(op_name, "out_backprop origin shape dim nums = %zu should be 5", out_backprop_ori_shape.GetDimNum()), return false);
  OP_CHECK_IF(filter_ori_shape.GetDimNum() != K_ORI_SHAPE_DIM_3D,
              OP_LOGE(op_name, "filter origin shape dim nums = %zu should be 5", filter_ori_shape.GetDimNum()), return false);
  OP_CHECK_IF(y_ori_shape.GetDimNum() != K_ORI_SHAPE_DIM_3D,
              OP_LOGE(op_name, "y origin shape dim nums = %zu should be 5", y_ori_shape.GetDimNum()), return false);

  Shape out_backprop_shape_ncdhw;
  Shape filter_shape_ncdhw;
  Shape y_shape_ncdhw;
  GetNCDHWShape(out_backprop_ori_shape, out_backprop_shape_ncdhw, out_backprop_ori_format);
  GetNCDHWShape(filter_ori_shape, filter_shape_ncdhw, filter_ori_format);
  GetNCDHWShape(y_ori_shape, y_shape_ncdhw, y_ori_format);

  OP_CHECK_IF(!UpdateShapeParams(context, runInfoV2, out_backprop_shape_ncdhw, filter_shape_ncdhw, y_shape_ncdhw, otherParams),
              OP_LOGE(op_name, "update shape params failed."), return false);
  return true;
}

static bool GetShapeParams(gert::TilingContext *context, Conv3dBpInputV2RunInfo &runInfoV2, optiling::OpTypeV2 op_type, bool isV2Impl, OtherParams& otherParams) {
  const auto op_name = context->GetNodeName();
  size_t out_backprop_input_index = static_cast<size_t>(OUT_BACKPROP_INDEX);
  size_t filter_input_index = static_cast<size_t>(FILTER_INDEX);
  // 转置的话，filter和out_backprop进行交换
  if (op_type == optiling::OpTypeV2::kConv3DTransposeV2) {
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
              OP_LOGE(op_name, "failed to get out_backprop/filter/y tensor desc from context."), return false);
  OP_CHECK_IF(out_backprop_shape == nullptr || filter_shape == nullptr || y_shape == nullptr,
              OP_LOGE(op_name, "failed to get out_backprop/filter/y shape from context."), return false);

  auto out_backprop_ori_format = out_backprop_desc->GetOriginFormat();
  auto filter_ori_format = filter_desc->GetOriginFormat();
  auto y_ori_format = y_desc->GetOriginFormat();
  OP_CHECK_IF(out_backprop_ori_format != y_ori_format,
              OP_LOGE(op_name, "y origin format[%s] should be same with out_backprop origin format[%s].",
                      ge::TypeUtils::FormatToSerialString(y_ori_format).c_str(),
                      ge::TypeUtils::FormatToSerialString(out_backprop_ori_format).c_str()),
              return false);
  OP_CHECK_IF(out_backprop_ori_format != ge::FORMAT_NDHWC && out_backprop_ori_format != ge::FORMAT_NCDHW,
              OP_LOGE(op_name, "out_backprop origin format[%s] should be NDHWC or NCDHW.",
                      ge::TypeUtils::FormatToSerialString(out_backprop_ori_format).c_str()),
              return false);
  if (IsArchAfter35(context) || IsSocVersionFuse(context)) {
    OP_CHECK_IF(filter_ori_format != ge::FORMAT_NDHWC && filter_ori_format != ge::FORMAT_NCDHW && filter_ori_format != ge::FORMAT_DHWCN,
                OP_LOGE(op_name, "filter origin format[%s] should be NDHWC/NCDHW/DHWCN.",
                        ge::TypeUtils::FormatToSerialString(filter_ori_format).c_str()),
                return false);
    OP_CHECK_IF(!CheckStorageFormat(context, filter_input_index, out_backprop_input_index),
                OP_LOGE(op_name, "Check storage format From Desc fail."), return false);
  } else {
    OP_CHECK_IF(filter_ori_format != ge::FORMAT_NDHWC && filter_ori_format != ge::FORMAT_NCDHW && filter_ori_format != ge::FORMAT_DHWCN,
                OP_LOGE(op_name, "filter origin format should be NDHWC or NCDHW or DHWCN."), return false);
    auto out_backprop_format = static_cast<ge::Format>(ge::GetPrimaryFormat(out_backprop_desc->GetStorageFormat()));
    auto filter_format = static_cast<ge::Format>(ge::GetPrimaryFormat(filter_desc->GetStorageFormat()));
    auto y_format = static_cast<ge::Format>(ge::GetPrimaryFormat(y_desc->GetStorageFormat()));
    OP_CHECK_IF(out_backprop_format != ge::FORMAT_NDC1HWC0 || filter_format != ge::FORMAT_FRACTAL_Z_3D,
                OP_LOGE(op_name, "out_backprop format should be NDC1HWC0, filter format should be FRACTAL_Z_3D."),
                return false);
    if (!isV2Impl || op_type == optiling::OpTypeV2::kConv3DTransposeV2) {
      OP_CHECK_IF(y_format != ge::FORMAT_NDC1HWC0,
                  OP_LOGE(op_name, "y format should be NDC1HWC0."), return false);
    } else {
      OP_CHECK_IF(y_format != ge::FORMAT_NDC1HWC0 && y_format != ge::FORMAT_NCDHW,
                  OP_LOGE(op_name, "y format should be NDC1HWC0 or NCDHW."), return false);
    }
  }

  OP_CHECK_IF(!CalShapeInfoFromDesc(context, filter_input_index, out_backprop_input_index, runInfoV2, otherParams),
              OP_LOGE(op_name, "Cal Shape Info From Desc fail."), return false);
  return true;
}

static void ReCalDilation(const gert::TilingContext *context, Conv3dBpInputV2RunInfo &runInfoV2, const OtherParams& otherParams) {
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

static bool CalGroups(gert::TilingContext *context, OtherParams& otherParams, Conv3dBpInputV2RunInfo &runInfoV2) {
  if (otherParams.b_shape.c == 0 || otherParams.c_shape.c % otherParams.b_shape.c != 0) {
    OP_LOGE(context, "fmap_channel(%ld) mod filter_channel(%ld) != 0", otherParams.c_shape.c,
            otherParams.b_shape.c);
    return false;
  }

  int32_t groups = otherParams.c_shape.c / otherParams.b_shape.c;
  if (runInfoV2.groups == 1) {
    runInfoV2.groups = groups;
  } else if (groups != runInfoV2.groups) {
    OP_LOGE(context, "fmap_channel(%ld) / filter_channel(%ld) != groups(%d)", otherParams.c_shape.c,
            otherParams.b_shape.c, runInfoV2.groups);
    return false;
  }

  OP_CHECK_IF(otherParams.a_shape.c % runInfoV2.groups != 0,
              OP_LOGE(context, "out_backprop's C(%ld) %% groups(%d) != 0", otherParams.a_shape.c, runInfoV2.groups),
              return false);

  if (IsArchAfter35(context) || IsSocVersionFuse(context)) {
    bool invalidFilterFormat = runInfoV2.filterFormat != ge::FORMAT_NCDHW &&
      runInfoV2.filterFormat != ge::FORMAT_NDHWC && runInfoV2.filterFormat != ge::FORMAT_DHWCN;
    bool invalidOutBackpropFormat = runInfoV2.outBackpropFormat != ge::FORMAT_NCDHW &&
        runInfoV2.outBackpropFormat != ge::FORMAT_NDHWC;
    bool invalidYFormat = runInfoV2.yFormat != ge::FORMAT_NCDHW && runInfoV2.yFormat != ge::FORMAT_NDHWC;
    bool invalidFormat = invalidOutBackpropFormat || invalidFilterFormat || invalidYFormat;
    OP_CHECK_IF(
        runInfoV2.groups != 1 && invalidFormat,
        CUBE_INNER_ERR_REPORT(context->GetNodeName(),
        "When groups(%d) > 1, out_backprop_format[%s] is limited to NCDHW/NDHWC, y_format[%s] is limited to NCDHW/NDHWC, "
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
static bool CheckAllZero(const T *tensor_data, size_t dim_size) {
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

static bool CheckInputSizeAllZero(gert::TilingContext *context, bool &allzero) {
  auto input_size = context->GetInputTensor(INPUT_SIZE_INDEX);
  OP_CHECK_IF(input_size == nullptr, OP_LOGE(context, "get input size fail"), return false);
  size_t input_size_dim_num = static_cast<size_t>(input_size->GetOriginShape().GetShapeSize());
  OP_CHECK_IF(input_size_dim_num != K_ORI_SHAPE_DIM_3D && input_size_dim_num != K_ORI_SHAPE_DIM_2D, OP_LOGE(context, "input_size_dim_num=[%zu], input_size must be 4d or 5d", input_size_dim_num), return false);
  auto dtype = context->GetInputDesc(INPUT_SIZE_INDEX)->GetDataType();
  if (dtype == ge::DT_INT32) {
    auto tensor_data = input_size->GetData<int32_t>();
    allzero = CheckAllZero(tensor_data, input_size_dim_num);
  } else if (dtype == ge::DT_INT64) {
    auto tensor_data = input_size->GetData<int64_t>();
    allzero = CheckAllZero(tensor_data, input_size_dim_num);
  } else {
    OP_LOGE(context, "input_size dtype only support int32 or int64");
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
static bool HandleConv3DTranspose(gert::TilingContext *context, const Conv3dBpInputV2RunInfo &runInfoV2, OtherParams& otherParams) {
  otherParams.filter_d_dilation += otherParams.output_padding.output_padding_d;
  otherParams.filter_h_dilation += otherParams.output_padding.output_padding_h;
  otherParams.filter_w_dilation += otherParams.output_padding.output_padding_w;

  bool all_zero = false;
  if (CheckInputSizeAllZero(context, all_zero) && all_zero) {
    int64_t standard_d = runInfoV2.stride_d * (otherParams.a_shape.d - 1) +
                         ((otherParams.b_shape.d - 1) * runInfoV2.dilation_d + 1);
    otherParams.c_shape.d = standard_d + otherParams.output_padding.output_padding_d -
                                            runInfoV2.pad_h - runInfoV2.pad_t;

    int64_t standard_h = runInfoV2.stride_h * (otherParams.a_shape.h - 1) +
                         ((otherParams.b_shape.h - 1) * runInfoV2.dilation_h + 1);
    otherParams.c_shape.h = standard_h + otherParams.output_padding.output_padding_h -
                                            runInfoV2.pad_u - runInfoV2.pad_d;

    int64_t standard_w = runInfoV2.stride_w * (otherParams.a_shape.w - 1) +
                         ((otherParams.b_shape.w - 1) * runInfoV2.dilation_w + 1);
    otherParams.c_shape.w = standard_w + otherParams.output_padding.output_padding_w -
                                            runInfoV2.pad_l - runInfoV2.pad_r;
    otherParams.c_shape.batch = otherParams.a_shape.batch;
    otherParams.c_shape.c = otherParams.b_shape.c;
    otherParams.c_shape.c1 = Ops::Base::CeilDiv(otherParams.c_shape.c, otherParams.c_shape.c0);
  }
  return true;
}

static bool CheckCalPads(const gert::TilingContext *context, const Conv3dBpInputV2RunInfo &runInfoV2, optiling::OpTypeV2 op_type, const OtherParams& otherParams) {
  int64_t do_expect = (otherParams.c_shape.d + runInfoV2.pad_h +
                      runInfoV2.pad_t - otherParams.filter_d_dilation) /
                      runInfoV2.stride_d + 1;
  int64_t ho_expect = (otherParams.c_shape.h + runInfoV2.pad_u +
                      runInfoV2.pad_d - otherParams.filter_h_dilation) /
                      runInfoV2.stride_h + 1;
  int64_t wo_expect = (otherParams.c_shape.w + runInfoV2.pad_l +
                      runInfoV2.pad_r - otherParams.filter_w_dilation) /
                      runInfoV2.stride_w + 1;
  std::string check_input_name = (op_type == optiling::OpTypeV2::kConv3DTransposeV2) ? "x" : "out_backprop";
  OP_CHECK_IF(do_expect != otherParams.a_shape.d, OP_LOGE(context->GetNodeName(), "%s's D = %ld is not equal with inferred D = %ld",
              check_input_name.c_str(), otherParams.a_shape.d, do_expect), return false);
  OP_CHECK_IF(ho_expect != otherParams.a_shape.h, OP_LOGE(context->GetNodeName(), "%s's H = %ld is not equal with inferred H = %ld",
              check_input_name.c_str(), otherParams.a_shape.h, ho_expect), return false);
  OP_CHECK_IF(wo_expect != otherParams.a_shape.w, OP_LOGE(context->GetNodeName(), "%s's W = %ld is not equal with inferred W = %ld",
              check_input_name.c_str(), otherParams.a_shape.w, wo_expect), return false);
  return true;
}

static bool CalPads(gert::TilingContext *context, Conv3dBpInputV2RunInfo &runInfoV2, optiling::OpTypeV2 op_type, OtherParams& otherParams) {
  auto attrs = context->GetAttrs();
  size_t padding_attr_idx = kPaddingConv3dBpInputIdx;
  if (op_type == optiling::OpTypeV2::kConv3DTransposeV2) {
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
                                 runInfoV2.stride_d + otherParams.filter_d_dilation -
                                 otherParams.c_shape.d,
                             0L);
    int32_t pad_head = static_cast<int32_t>((static_cast<uint32_t>(pad_d) >> 1U));
    int32_t pad_tail = pad_d - pad_head;
    int32_t pad_h = std::max(Ops::Base::CeilAlign(otherParams.c_shape.h, static_cast<int64_t>(runInfoV2.stride_h)) -
                                 runInfoV2.stride_h + otherParams.filter_h_dilation -
                                 otherParams.c_shape.h,
                             0L);
    int32_t pad_up = static_cast<int32_t>((static_cast<uint32_t>(pad_h) >> 1U));
    int32_t pad_down = pad_h - pad_up;
    int32_t pad_w = std::max(Ops::Base::CeilAlign(otherParams.c_shape.w, static_cast<int64_t>(runInfoV2.stride_w)) -
                                 runInfoV2.stride_w + otherParams.filter_w_dilation -
                                 otherParams.c_shape.w,
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

  if (op_type == optiling::OpTypeV2::kConv3DTransposeV2) {
    OP_CHECK_IF(!HandleConv3DTranspose(context, runInfoV2, otherParams), OP_LOGE(context, "Failed to process Conv3DTranspose."), return false);
  }

  return CheckCalPads(context, runInfoV2, op_type, otherParams);
}

static bool CalRealG(gert::TilingContext *context, Conv3dBpInputV2RunInfo &runInfoV2, OtherParams& otherParams) {
  // calc real g and check shape
  int32_t dy_c_ori = otherParams.a_shape.c / runInfoV2.groups;
  OP_CHECK_IF(dy_c_ori == 0,
              OP_LOGE(context, "Given groups %d , expected out_backporp to be at least %d at dimension 1, but got out_backporp of size %ld  instead",
                      runInfoV2.groups,
                      runInfoV2.groups,
                      otherParams.a_shape.c),
              return false);
  int32_t dx_c_extend = MathUtil::Lcm(otherParams.b_shape.c, otherParams.c_shape.c0) / otherParams.b_shape.c;
  int32_t dy_c_extend = MathUtil::Lcm(dy_c_ori, kBlockSize) / dy_c_ori;
  otherParams.multiple_extend = std::min(MathUtil::Lcm(dx_c_extend, dy_c_extend), static_cast<int64_t>(runInfoV2.groups));
  runInfoV2.real_g = (static_cast<int64_t>(runInfoV2.groups) + otherParams.multiple_extend - 1) / otherParams.multiple_extend;
  otherParams.ci1g = Ops::Base::CeilDiv(otherParams.multiple_extend * otherParams.b_shape.c, otherParams.c_shape.c0);
  int32_t co1g = (otherParams.multiple_extend * dy_c_ori + kBlockSize - 1) / kBlockSize;
  if (context->GetOutputDesc(Y_INDEX)->GetDataType() == ge::DT_FLOAT && runInfoV2.groups > 1) {
      co1g *= 2; // 2: BLOCK_NUM / FP32_C0
  }

  if (IsArchAfter35(context) || IsSocVersionFuse(context)) {
    otherParams.ci1g = Ops::Base::CeilDiv(
        otherParams.multiple_extend * otherParams.b_shape.c, static_cast<int64_t>(kBlockSize));
    otherParams.co1g = Ops::Base::CeilDiv(
        otherParams.multiple_extend * otherParams.b_shape.batch / runInfoV2.groups,
        otherParams.c_shape.c0);

    size_t filter_input_idx = static_cast<size_t>(FILTER_INDEX);
    size_t out_backprop_input_idx = static_cast<size_t>(OUT_BACKPROP_INDEX);
    const auto out_backprop_desc = context->GetInputDesc(out_backprop_input_idx);
    const auto filter_desc = context->GetInputDesc(filter_input_idx);

    constexpr uint32_t ENLARGE_BUFFER_NUM = 2;
    constexpr uint32_t REG_SIZE = 256;
    bool disableGroupEnlarge = static_cast<uint64_t>(ENLARGE_BUFFER_NUM) * (otherParams.co1g *
      otherParams.b_shape.h * otherParams.b_shape.w * otherParams.ci1g *
      kBlockSize * otherParams.b_shape.c0) * runInfoV2.b_dtype_bytes +
      REG_SIZE > context->GetCompileInfo<Ops::NN::Conv::Conv3DBackpropV2CompileInfo>()->ub_size;

    bool nonExtendedDtype = (filter_desc->GetDataType() == ge::DT_FLOAT8_E4M3FN || filter_desc->GetDataType() == ge::DT_HIFLOAT8 ||
      filter_desc->GetDataType() == ge::DT_INT8) || (out_backprop_desc->GetDataType() == ge::DT_FLOAT8_E4M3FN ||
      out_backprop_desc->GetDataType() == ge::DT_HIFLOAT8 || out_backprop_desc->GetDataType() == ge::DT_INT8);
    if (disableGroupEnlarge || nonExtendedDtype) {
      otherParams.multiple_extend = 1;
      runInfoV2.real_g = runInfoV2.groups;
      otherParams.ci1g = Ops::Base::CeilDiv(otherParams.b_shape.c, static_cast<int64_t>(kBlockSize));
      otherParams.co1g = Ops::Base::CeilDiv(otherParams.b_shape.batch / runInfoV2.groups, otherParams.c_shape.c0);
    }
  }

  if (!IsArchAfter35(context) && !IsSocVersionFuse(context)) {
    int64_t filterGDkCi1gHW = static_cast<int64_t>(runInfoV2.real_g) *
                               otherParams.b_shape.d * otherParams.ci1g *
                               otherParams.b_shape.h * otherParams.b_shape.w;

    OP_CHECK_IF(filterGDkCi1gHW != otherParams.filter_gdkci1ghw,
                OP_LOGE(context, "the 1st dim of filter shape should be %ld, which is %ld",
                        filterGDkCi1gHW, otherParams.filter_gdkci1ghw),
                return false);
    OP_CHECK_IF(co1g != otherParams.co1g,
                OP_LOGE(context, "the 2nd dim of filter shape should be %d, which is %d",
                        co1g, otherParams.co1g),
                return false);
  }
  return true;
}

static int32_t CalBackpropPadBefore(int32_t filter, int32_t dilation, int32_t pad) {
  return (filter - 1) * dilation - pad;
}

static int64_t CalBackpropPadAfter(int64_t inputDim, int64_t outputDim, int32_t stride, int32_t pad) {
  // orginal formula is inputDim = (outputDim * stride + 1) - padBefore + filterDilation, it can be simplified as follow.
  return inputDim - outputDim * stride + pad;
}

static inline bool IsOverflowInt32(int64_t value) {
  if (value > INT32_MAX || value < INT32_MIN) {
    return true;
  }
  return false;
}

static inline bool CheckRange(int32_t value, int32_t value_low, int32_t value_up) {
  if (value < value_low || value > value_up) {
    return false;
  }
  return true;
}

static bool CalModifyBackpropPadD(gert::TilingContext *context, Conv3dBpInputV2RunInfo &runInfoV2, OtherParams& otherParams) {
  Shape &dedyShape = otherParams.a_shape;
  Shape &filterShape = otherParams.b_shape;
  Shape &dedxShape = otherParams.c_shape;

  otherParams.pad_head_before = CalBackpropPadBefore(filterShape.d, runInfoV2.dilation_d, runInfoV2.pad_h);
  int64_t pad_tail_after = CalBackpropPadAfter(dedxShape.d, dedyShape.d, runInfoV2.stride_d, runInfoV2.pad_h);
  OP_CHECK_IF(IsOverflowInt32(pad_tail_after) || !CheckRange(static_cast<int32_t>(pad_tail_after), -kDimUp, kDimUp),
              OP_LOGE(context, "pad_tail_after = (inputD - outputD * strideD + padHead)=%ld is invalid, it should be in[%d, %d]",
                      pad_tail_after, -kDimUp, kDimUp),
              return false);
  pad_tail_after = (pad_tail_after + abs(pad_tail_after)) / kNumTwo;
  otherParams.pad_tail_after = pad_tail_after;
  runInfoV2.backprop_pad_h = otherParams.pad_head_before;
  runInfoV2.backprop_pad_t = otherParams.pad_tail_after;
  return true;
}

static bool CalModifyBackpropPadHW(gert::TilingContext *context, Conv3dBpInputV2RunInfo &runInfoV2, OtherParams& otherParams) {
  Shape &dedyShape = otherParams.a_shape;
  Shape &filterShape = otherParams.b_shape;
  Shape &dedxShape = otherParams.c_shape;

  otherParams.pad_left_before = CalBackpropPadBefore(filterShape.w, runInfoV2.dilation_w, runInfoV2.pad_l);
  otherParams.pad_up_before = CalBackpropPadBefore(filterShape.h, runInfoV2.dilation_h, runInfoV2.pad_u);
  int32_t kPadUpTmp = (IsArchAfter35(context) || IsSocVersionFuse(context)) ? kDimUp : kPadUp;

  OP_CHECK_IF(!CheckRange(otherParams.pad_left_before, 0, kPadUpTmp),
              OP_LOGE(context, "backprop_pad_left=((kw - 1) * dilation_w - pad_left)=[%d] is invalid, it should be in [%d, %d]",
                      otherParams.pad_left_before, 0 , kPadUpTmp),
              return false);
  OP_CHECK_IF(!CheckRange(otherParams.pad_up_before, 0, kPadUpTmp),
              OP_LOGE(context, "backprop_pad_up=((kh - 1) * dilation_h - pad_up)=[%d] is invalid, it should be in [%d, %d]",
                      otherParams.pad_up_before, 0 , kPadUpTmp),
              return false);
  if (IsArchAfter35(context) || IsSocVersionFuse(context)) {
    otherParams.pad_head_before = CalBackpropPadBefore(filterShape.d, runInfoV2.dilation_d, runInfoV2.pad_h);
    OP_CHECK_IF(!CheckRange(otherParams.pad_head_before, 0, kPadUpTmp),
                OP_LOGE(context, "backprop_pad_head=((kd - 1) * dilation_d - pad_head)=[%d] is invalid, it should be in [%d, %d]",
                        otherParams.pad_head_before, 0 , kPadUpTmp),
                return false);
  }

  otherParams.shape_left_modify = (otherParams.pad_left_before - abs(otherParams.pad_left_before)) / kNumTwo;
  otherParams.shape_up_modify = (otherParams.pad_up_before - abs(otherParams.pad_up_before)) / kNumTwo;

  int64_t pad_right_after = CalBackpropPadAfter(dedxShape.w, dedyShape.w, runInfoV2.stride_w, runInfoV2.pad_l);
  int64_t pad_down_after = CalBackpropPadAfter(dedxShape.h, dedyShape.h, runInfoV2.stride_h, runInfoV2.pad_u);

  OP_CHECK_IF(IsOverflowInt32(pad_right_after) || !CheckRange(static_cast<int32_t>(pad_right_after), -kPadUpTmp, kPadUpTmp),
              OP_LOGE(context, "backprop_right_pad = (inputW - outputW * strideW + padLeft)=%ld is invalid, it should be in[%d, %d]",
                      pad_right_after, -kPadUpTmp, kPadUpTmp),
              return false);

  OP_CHECK_IF(IsOverflowInt32(pad_down_after) || !CheckRange(static_cast<int32_t>(pad_down_after), -kPadUpTmp, kPadUpTmp),
              OP_LOGE(context, "backprop_down_pad = (inputH - outputH * strideH + padUp)=%ld is invalid, it should be in[%d, %d]",
                      pad_down_after, -kPadUpTmp, kPadUpTmp),
              return false);

  int64_t shape_down_modify = (pad_down_after - abs(pad_down_after)) / kNumTwo;
  int64_t shape_right_modify = (pad_right_after - abs(pad_right_after)) / kNumTwo;

  otherParams.pad_left_before = (otherParams.pad_left_before + abs(otherParams.pad_left_before)) / kNumTwo;
  pad_down_after = (pad_down_after + abs(pad_down_after)) / kNumTwo;
  pad_right_after = (pad_right_after + abs(pad_right_after)) / kNumTwo;

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

static bool CalModify(gert::TilingContext *context, Conv3dBpInputV2RunInfo &runInfoV2, OtherParams& otherParams) {
  OP_CHECK_IF(!CalModifyBackpropPadD(context, runInfoV2, otherParams), OP_LOGE(context, "Cal backprop pad d invalid"), return false);
  OP_CHECK_IF(!CalModifyBackpropPadHW(context, runInfoV2, otherParams), OP_LOGE(context, "Cal backprop pad h,w invalid"),return false);
  return true;
}

static inline bool CheckLowerBound(int32_t value, int32_t value_low) { return value >= value_low; }

static inline bool CheckValue(int32_t value, int32_t value_temp) { return value == value_temp; }

bool CheckPadParamsWithLog(const Conv3dBpInputV2RunInfo &runInfoV2, const gert::TilingContext *context) {
  const auto op_name = context->GetNodeName();
  int32_t kPadUpTmp = kDimUp;
  if (IsArchAfter35(context) || IsSocVersionFuse(context)) {
    kPadUpTmp = kPadUp;
  }
  OP_CHECK_IF(!CheckRange(runInfoV2.pad_h, 0, kPadUpTmp),
              OP_LOGE(op_name, "pad_h value [%d] is invalid, support range [%d, %d]", runInfoV2.pad_h, 0, kPadUpTmp),
              return false);
  OP_CHECK_IF(!CheckRange(runInfoV2.pad_t, 0, kPadUpTmp),
              OP_LOGE(op_name, "pad_t value [%d] is invalid, support range [%d, %d]", runInfoV2.pad_t, 0, kPadUpTmp),
              return false);
  OP_CHECK_IF(!CheckRange(runInfoV2.pad_u, 0, kPadUpTmp),
              OP_LOGE(op_name, "pad_u value [%d] is invalid, support range [%d, %d]", runInfoV2.pad_u, 0, kPadUpTmp),
              return false);
  OP_CHECK_IF(!CheckRange(runInfoV2.pad_d, 0, kPadUpTmp),
              OP_LOGE(op_name, "pad_d value [%d] is invalid, support range [%d, %d]", runInfoV2.pad_d, 0, kPadUpTmp),
              return false);
  OP_CHECK_IF(!CheckRange(runInfoV2.pad_l, 0, kPadUpTmp),
              OP_LOGE(op_name, "pad_l value [%d] is invalid, support range [%d, %d]", runInfoV2.pad_l, 0, kPadUpTmp),
              return false);
  OP_CHECK_IF(!CheckRange(runInfoV2.pad_r, 0, kPadUpTmp),
              OP_LOGE(op_name, "pad_r value [%d] is invalid, support range [%d, %d]", runInfoV2.pad_r, 0, kPadUpTmp),
              return false);
  return true;
}

bool CheckParamsWithLog(Conv3dBpInputV2RunInfo &runInfoV2, gert::TilingContext *context, OtherParams& otherParams) {
  const auto op_name = context->GetNodeName();
  int32_t kStrideHWUpTmp = kDimUp;
  int32_t kStrideDUpTmp = kDimUp;
  int32_t kDilationUpTmp = kDilationUp;
  if (IsArchAfter35(context) || IsSocVersionFuse(context)) {
    kDilationUpTmp = kDimUp;
  }
  OP_CHECK_IF(!CheckRange(runInfoV2.dilation_h, kDilationLow, kDilationUpTmp),
              OP_LOGE(op_name, "dilation_h value [%d] is invalid, support range [%d, %d]", runInfoV2.dilation_h, kDilationLow, kDilationUpTmp),
              return false);
  OP_CHECK_IF(!CheckRange(runInfoV2.dilation_w, kDilationLow, kDilationUpTmp),
              OP_LOGE(op_name, "dilation_w value [%d] is invalid, support range [%d, %d]", runInfoV2.dilation_w, kDilationLow, kDilationUpTmp),
              return false);
  OP_CHECK_IF(!CheckRange(runInfoV2.dilation_d, kDilationLow, kDilationUpTmp),
              OP_LOGE(op_name, "dilation_d value [%d] is invalid, support range [%d, %d]", runInfoV2.dilation_d, kDilationLow, kDilationUpTmp),
              return false);
  OP_CHECK_IF(!CheckRange(runInfoV2.stride_h, kDimLow, kStrideHWUpTmp),
              OP_LOGE(op_name, "stride_h value [%d] is invalid, support range [%d, %d]", runInfoV2.stride_h, kDimLow, kStrideHWUpTmp),
              return false);
  OP_CHECK_IF(!CheckRange(runInfoV2.stride_w, kDimLow, kStrideHWUpTmp),
              OP_LOGE(op_name, "stride_w value [%d] is invalid, support range [%d, %d]", runInfoV2.stride_w, kDimLow, kStrideHWUpTmp),
              return false);
  OP_CHECK_IF(!CheckRange(runInfoV2.stride_d, kDimLow, kStrideDUpTmp),
              OP_LOGE(op_name, "stride_d value [%d] is invalid, support range [%d, %d]", runInfoV2.stride_d, kDimLow, kStrideDUpTmp),
              return false);
  OP_CHECK_IF(!CheckPadParamsWithLog(runInfoV2, context),
              OP_LOGE(op_name, "check pad params failed"),
              return false);
  OP_CHECK_IF((otherParams.filter_d_dilation > otherParams.fmap_d_padding),
              OP_LOGE(op_name, "((filter_d - 1) * dilation_d + 1)=[%ld] must less than or equal to (fmap_d + pad_h + pad_t)=[%ld]",
                      otherParams.filter_d_dilation, otherParams.fmap_d_padding),
              return false);
  OP_CHECK_IF((otherParams.filter_h_dilation > otherParams.fmap_h_padding),
              OP_LOGE(op_name, "((filter_h - 1) * dilation_h + 1)=[%ld] must less than or equal to (fmap_h + pad_u + pad_d)=[%ld]",
                      otherParams.filter_h_dilation, otherParams.fmap_h_padding),
              return false);
  OP_CHECK_IF((otherParams.filter_w_dilation > otherParams.fmap_w_padding),
              OP_LOGE(op_name, "((filter_w - 1) * dilation_w + 1)=[%ld] must less than or equal to (fmap_w + pad_l + pad_r)=[%ld]",
                      otherParams.filter_w_dilation, otherParams.fmap_w_padding),
              return false);
  OP_CHECK_IF(!CheckRange(otherParams.a_shape.w * runInfoV2.stride_w, kDimLow, kDimUp),
              OP_LOGE(op_name, "out_backprop's W after expands [%ld] is invalid, support range [%d, %d]",
                      otherParams.a_shape.w * runInfoV2.stride_w, kDimLow, kDimUp),
              return false);
  OP_CHECK_IF(!CheckRange(otherParams.a_shape.h * runInfoV2.stride_h, kDimLow, kDimUp),
              OP_LOGE(op_name, "out_backprop's H after expands [%ld] is invalid, support range [%d, %d]",
                      otherParams.a_shape.h * runInfoV2.stride_h, kDimLow, kDimUp),
              return false);
  return true;
}

int64_t GetDfactorSdEqKd(const Conv3dBpInputV2RunInfo &runInfoV2, int32_t l0c_din, const OtherParams& otherParams) {
  // ----[该函数用于计算StrideD=KerneD时 由Din和Dk 反推的Dout的大小]----
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
static int64_t GetDfactor(T kd_factor, Conv3dBpInputV2RunInfo &runInfoV2, int32_t l0c_din, gert::TilingContext *context, OtherParams& otherParams) {
  int64_t estimate_d = static_cast<int64_t>(Ops::Base::CeilDiv(static_cast<int64_t>(kd_factor - 2 + l0c_din),
                                                              static_cast<int64_t>(runInfoV2.stride_d)) + 1);
  int64_t dout_factor = std::min(estimate_d, otherParams.a_shape.d);
  if (otherParams.filter_d_dilation == runInfoV2.stride_d) {
    dout_factor = (context->GetCompileInfo<Ops::NN::Conv::Conv3DBackpropV2CompileInfo>()->intrinsic_fix_pipe_l0c2out)
                      ? GetDfactorSdEqKd(runInfoV2, l0c_din, otherParams)
                      : std::max(dout_factor - 1, static_cast<int64_t>(1));
  };
  return dout_factor;
}

static bool CheckL1SizeLimit(Conv3dBpInputV2RunInfo &runInfoV2, gert::TilingContext *context, OtherParams& otherParams) {
  if (IsArchAfter35(context) || IsSocVersionFuse(context)) {
    return true;
  }
  int64_t w_value = otherParams.a_shape.w * runInfoV2.stride_w;
  int64_t h_value_max =
      (otherParams.filter_h_dilation - 1) + kBlockSize / otherParams.c_shape.w + 2;
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

  int64_t a_l1_size = h_value_max * w_value * a_l1_d * otherParams.a_shape.c0 *
                       runInfoV2.a_dtype_bytes;
  int64_t b_l1_size = b_l1_dk * otherParams.b_shape.h * otherParams.filter_co0 *
                       otherParams.b_shape.w * otherParams.filter_ci0 *
                       runInfoV2.b_dtype_bytes;
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
  if (a_l1_size + b_l1_size + fill_zero_size > static_cast<int64_t>(context->GetCompileInfo<Ops::NN::Conv::Conv3DBackpropV2CompileInfo>()->l1_size) ||
      !w_size_limit) {
    std::stringstream ss;
    ss << "Minimum load size may exceed L1 buffer, a_l1_size is " << a_l1_size << "B, b_l1_size is " << b_l1_size
      << "B, fill_zero_size is " << fill_zero_size << "B, L1size is " << context->GetCompileInfo<Ops::NN::Conv::Conv3DBackpropV2CompileInfo>()->l1_size
      << "B, width may exceed limits, actual width: " << std::max(otherParams.c_shape.w, w_value)
      << ", width limit: " << kDimWNormalUp;
    OP_LOGE(context->GetNodeName(), "Error msg: %s", ss.str().c_str());
    return false;
  }

  return true;
}

static void SetConvAttrs(Conv3dBpInputV2RunInfo& runInfoV2, const int64_t *pads_data, Shape &strides_ncdhw,
  Shape &dilations_ncdhw, const int64_t *groups, OtherParams& otherParams) {
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

  otherParams.stride_expand_flag = (runInfoV2.stride_h != 1 ||
    runInfoV2.stride_w != 1 || runInfoV2.stride_d != 1);

  otherParams.dilation_d_gt_one_flag = dilations_ncdhw.d > 1 ? 1: 0;
}

bool GetAttrAndDtypeParams(gert::TilingContext *context, Conv3dBpInputV2RunInfo& runInfoV2, optiling::OpTypeV2 op_type, OtherParams& otherParams) {
    auto attrs = context->GetAttrs();
    size_t idx = 0;
    const auto strides = attrs->GetAttrPointer<gert::ContinuousVector>(idx++);
    const auto pads = attrs->GetAttrPointer<gert::ContinuousVector>(idx++);
    const auto dilations = attrs->GetAttrPointer<gert::ContinuousVector>(idx++);
    const auto groups = attrs->GetAttrPointer<int64_t>(idx++);

    const auto op_name = context->GetNodeName();
    // stridesDim == 5
    OP_CHECK_IF(strides == nullptr, OP_LOGE(op_name, "get strides from context fail."), return false);
    OP_CHECK_IF(strides->GetSize() != kStridesDim, OP_LOGE(op_name, "strides of context dim len is invalid."), return false);
    // padsDim == 6
    OP_CHECK_IF(pads == nullptr, OP_LOGE(op_name, "get pads from context fail."), return false);
    OP_CHECK_IF(pads->GetSize() != kPadsDim, OP_LOGE(op_name, "pads of context dim len is invalid."), return false);
    // dilationsDim == 5
    OP_CHECK_IF(dilations == nullptr, OP_LOGE(op_name, "get dilations from context fail."), return false);
    OP_CHECK_IF(dilations->GetSize() != kDilationsDim, OP_LOGE(op_name, "dilations of context dim len is invalid."), return false);
    OP_CHECK_IF(groups == nullptr, OP_LOGE(op_name, "get groups from context fail."), return false);

    const auto strides_data = static_cast<const int64_t *>(strides->GetData());
    const auto pads_data = static_cast<const int64_t *>(pads->GetData());
    const auto dilations_data = static_cast<const int64_t *>(dilations->GetData());
    OP_CHECK_IF(!CheckAttrRange(context, strides_data, pads_data, dilations_data, groups),
      OP_LOGE(context, "check attr range failed"), return false);

    Shape strides_ncdhw;
    Shape dilations_ncdhw;
    ge::Format data_format = context->GetInputDesc(OUT_BACKPROP_INDEX)->GetOriginFormat();
    if (op_type == optiling::OpTypeV2::kConv3DTransposeV2) {
      OP_CHECK_IF(!CheckTransposeAttr(context, otherParams), OP_LOGE(context, "check transpose attr failed"), return false);
      data_format = context->GetInputDesc(FILTER_INDEX)->GetOriginFormat();
    }

    GetNCDHWShape(strides_data, strides_ncdhw, data_format);
    GetNCDHWShape(dilations_data, dilations_ncdhw, data_format);

    OP_CHECK_IF(strides_ncdhw.batch != 1,
      OP_LOGE(op_name, "strides N[%ld] is invalid, must be 1.", strides_ncdhw.batch), return false);
    OP_CHECK_IF(strides_ncdhw.c != 1,
      OP_LOGE(op_name, "strides C[%ld] is invalid, must be 1.", strides_ncdhw.c), return false);

    SetConvAttrs(runInfoV2, pads_data, strides_ncdhw, dilations_ncdhw, groups, otherParams);
    if (op_type == optiling::OpTypeV2::kConv3DTransposeV2) {
      OP_CHECK_IF(!CheckTransposeOutputdingRange(context, runInfoV2, otherParams), OP_LOGE(context, "check transpose attr failed"), return false);
    }
    return UpdateDtypeParams(context, runInfoV2, op_type, otherParams);
}

bool GetInputOutputFormat(const gert::TilingContext* context, Conv3dBpInputV2RunInfo& runInfoV2, const optiling::OpTypeV2 opType)
{
    size_t aMatrixIndex = OUT_BACKPROP_INDEX;
    size_t bMatrixIndex = FILTER_INDEX;
    const char* opName = context->GetNodeName(); // 日志打印用，允许为空

    if (opType == optiling::OpTypeV2::kConv3DTransposeV2) {
        aMatrixIndex = FILTER_INDEX;
        bMatrixIndex = OUT_BACKPROP_INDEX;
    }

    const auto out_backprop_desc = context->GetInputDesc(aMatrixIndex);
    OP_CHECK_IF(out_backprop_desc == nullptr, CUBE_INNER_ERR_REPORT(opName, "out_backprop_desc is null"),
        return false);
    ge::Format outBackpropFormat = out_backprop_desc->GetStorageFormat();

    const auto filter_desc = context->GetInputDesc(bMatrixIndex);
    OP_CHECK_IF(filter_desc == nullptr, CUBE_INNER_ERR_REPORT(opName, "filter_desc is null"),
        return false);
    ge::Format filterFormat = filter_desc->GetStorageFormat();

    const auto y_desc = context->GetOutputDesc(Y_INDEX);
    OP_CHECK_IF(y_desc == nullptr, CUBE_INNER_ERR_REPORT(opName, "y_desc is null"),
        return false);
    ge::Format yFormat = y_desc->GetStorageFormat();

    runInfoV2.outBackpropFormat = outBackpropFormat;
    runInfoV2.filterFormat = filterFormat;
    runInfoV2.yFormat = yFormat;
    return true;
}

bool CalScale(const gert::TilingContext *context, const Conv3dBpInputV2RunInfo &runInfoV2, const OtherParams& otherParams) {
	if (!IsSocVersionFuse(context)) {
		return true;
	}
    if (otherParams.a_dtype == ge::DT_INT8) {
        if (static_cast<uint64_t>(runInfoV2.dedx_cin) * ge::GetSizeByDataType(ge::DT_INT64) > FB_BUFFER_SIZE) {
            return false;
        }
    }
    return true;
}

bool Conv3DBackpropInputParseFunc(gert::TilingContext *context, optiling::OpTypeV2 opType,
                                  Conv3dBpInputV2RunInfo& runInfoV2, OtherParams& otherParams, bool isV2Impl) {
    const auto op_name = context->GetNodeName();
    OP_CHECK_IF(!GetAttrAndDtypeParams(context, runInfoV2, opType, otherParams), OP_LOGE(op_name, "Get attrs and dtype Failed."), return false);
    OP_CHECK_IF(!GetShapeParams(context, runInfoV2, opType, isV2Impl, otherParams), OP_LOGE(op_name, "Set shape params failed."), return false);
    ReCalDilation(context, runInfoV2, otherParams);
    OP_CHECK_IF(!GetInputOutputFormat(context, runInfoV2, opType),
                OP_LOGE(context, "failed to get input output format"), return false);
    OP_CHECK_IF(!CalGroups(context, otherParams, runInfoV2), OP_LOGE(op_name, "Calc groups failed."), return false);
    OP_CHECK_IF(!CalPads(context, runInfoV2, opType, otherParams), OP_LOGE(op_name, "Calc pads failed."), return false);
    OP_CHECK_IF(!CalRealG(context, runInfoV2, otherParams), OP_LOGE(op_name, "Calc real_g failed."), return false);
    OP_CHECK_IF(!CalModify(context, runInfoV2, otherParams), OP_LOGE(op_name, "Modify pad failed."), return false);
    OP_CHECK_IF(!CalScale(context, runInfoV2, otherParams), OP_LOGE(op_name, "Sacle size too big, not support."), return false);
    return true;
}

bool GetFusionMode(Conv3dBpInputV2RunInfo &runInfoV2, const char* opName,
    const gert::TilingContext* context, optiling::OpTypeV2 opType)
{
    if (!IsSocVersionFuse(context) || opType != optiling::OpTypeV2::kConv3DTransposeV2) {
        return true;
    }
    auto attrs = context->GetAttrs();
    OP_CHECK_IF(attrs == nullptr,
                OP_LOGE(context, "failed to get runtime attrs"),
                return false);
    size_t idx = K_FUSION_MODE_CONV3D_TRANSPOSE_IDX;
    if (idx < attrs->GetAttrNum()) {
        const int32_t *fusionMode = attrs->GetAttrPointer<int32_t>(idx);
        if (fusionMode != nullptr && *fusionMode == 1) {
            runInfoV2.enRelu = 1;
        } else {
            OP_LOGW(opName, "relu flag is not support, so we set 0 as default");
            runInfoV2.enRelu = 0; // for extendConvTranspose fixpipe fusion pass, default value is 0
        }
    }
    return true;
}

bool GetImplMode(Conv3dBpInputV2RunInfo &runInfoV2, const char* opName,
                 const gert::TilingContext* context, optiling::OpTypeV2 opType)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_IF(attrs == nullptr,
                OP_LOGE(context, "failed to get runtime attrs"),
                return false);
    auto inputDesc = context->GetInputDesc(OUT_BACKPROP_INDEX);
    OP_CHECK_IF(inputDesc == nullptr,
        OP_LOGE(opName, "failed to get out_backprop tensor desc from context"),
        return false
    );
    ge::DataType aDtype = inputDesc->GetDataType();
    size_t idx = 6; // dx impl mode idx is 6
    if (opType == optiling::OpTypeV2::kConv3DTransposeV2) {
        idx = 8U; // transpose impl mode idx is 8
    } else {
        idx = 6U; // dx impl mode idx is 6
    }
    if (aDtype == ge::DT_FLOAT && idx < attrs->GetAttrNum()) {
        const int32_t *precisionMode = attrs->GetAttrPointer<int32_t>(idx);
        if (precisionMode != nullptr && *precisionMode != -1) {
            runInfoV2.hf32_flag = ((static_cast<uint32_t>(*precisionMode) & static_cast<uint32_t>(0x40)) != 0U) ? 1 : 0; // 0x40: hf32 enable flag
        } else {
            OP_LOGW(opName, "impl mode is not support, so we set hf32 flag with 0 as default");
            runInfoV2.hf32_flag = 0;
        }
    }
    return true;
}

bool CheckShapeValidWithLog(const gert::TilingContext *context, const OtherParams& otherParams, const Conv3dBpInputV2RunInfo &runInfoV2) {
  const auto op_name = context->GetNodeName();
  int32_t kGroupUpTmp = kDimUp;
  int32_t kFilterDimHWUpTmp = kFilterDimHWUp;
  if (IsArchAfter35(context) || IsSocVersionFuse(context)) {
    kGroupUpTmp = kGroupUp;
    kFilterDimHWUpTmp = kDimUp;
  }
  OP_CHECK_IF(!CheckRange(runInfoV2.groups, kDimLow, kGroupUpTmp),
              OP_LOGE(op_name, "group value [%d] is invalid, support range [%d, %d]", runInfoV2.groups, kDimLow, kGroupUpTmp),
              return false);
  OP_CHECK_IF(!CheckRange(otherParams.b_shape.h, kDimLow, kFilterDimHWUpTmp),
              OP_LOGE(op_name, "the H dim of filter [%ld] is invalid, support range [%d, %d]", otherParams.b_shape.h, kDimLow, kFilterDimHWUpTmp),
              return false);
  OP_CHECK_IF(!CheckRange(otherParams.b_shape.w, kDimLow, kFilterDimHWUpTmp),
              OP_LOGE(op_name, "the W dim of filter [%ld] is invalid, support range [%d, %d]", otherParams.b_shape.w, kDimLow, kFilterDimHWUpTmp),
              return false);
  OP_CHECK_IF(!CheckRange(otherParams.b_shape.d, kDimLow, kDimBatchUp),
              OP_LOGE(op_name, "the D dim of filter [%ld] is invalid, support range [%d, %d]", otherParams.b_shape.d, kDimLow, kDimBatchUp),
              return false);
  OP_CHECK_IF(!CheckRange(otherParams.a_shape.batch, kDimLow, kDimBatchUp),
              OP_LOGE(op_name, "batch value [%ld] is invalid, support range [%d, %d]", otherParams.a_shape.batch, kDimLow, kDimBatchUp),
              return false);
  OP_CHECK_IF(!CheckRange(otherParams.a_shape.d, kDimLow, kDimUp),
              OP_LOGE(op_name, "dout value [%ld] is invalid, support range [%d, %d]", otherParams.a_shape.d, kDimLow, kDimUp),
              return false);
  OP_CHECK_IF(!CheckRange(otherParams.a_shape.h, kDimLow, kDimUp),
              OP_LOGE(op_name, "hout value [%ld] is invalid, support range [%d, %d]", otherParams.a_shape.h, kDimLow, kDimUp),
              return false);
  OP_CHECK_IF(!CheckRange(otherParams.a_shape.w, kDimLow, kDimUp),
              OP_LOGE(op_name, "wout value [%ld] is invalid, support range [%d, %d]", otherParams.a_shape.w, kDimLow, kDimUp),
              return false);
  OP_CHECK_IF(!CheckLowerBound(otherParams.a_shape.c, kDimLow),
              OP_LOGE(op_name, "cout value [%ld] is invalid, should not be less than [%d]", otherParams.a_shape.c, kDimLow),
              return false);
  OP_CHECK_IF(!CheckLowerBound(otherParams.a_shape.c1, kDimLow),
              OP_LOGE(op_name, "cout1 value [%ld] is invalid, should not be less than [%d]", otherParams.a_shape.c1, kDimLow),
              return false);
  OP_CHECK_IF(!CheckLowerBound(otherParams.c_shape.c1, kDimLow),
              OP_LOGE(op_name, "cin1 value [%ld] is invalid, should not be less than [%d]", otherParams.c_shape.c1, kDimLow),
              return false);
  OP_CHECK_IF(!CheckLowerBound(otherParams.c_shape.c, kDimLow),
              OP_LOGE(op_name, "cin value [%ld] is invalid, should not be less than [%d]", otherParams.c_shape.c, kDimLow),
              return false);
  OP_CHECK_IF(!CheckLowerBound(otherParams.c_shape.d, kDimLow),
              OP_LOGE(op_name, "din value [%ld] is invalid, should not be less than [%d]", otherParams.c_shape.d, kDimLow),
              return false);
  return true;
}

static inline string IntToBinary(uint64_t &n) {
  string ans = "";
  do {
    uint64_t t = n % 2UL;
    ans += (t + CHAR_0);
    n /= 2UL;
  } while (n != 0UL);
  return ans;
}

static inline void OutputErrorMsg(const vector<string> &error_info, string &error_flag) {
  string msg;
  for (size_t i = 0; i < error_flag.length(); i++) {
    if (error_flag[i] == '1' && i < error_info.size()) {
      msg = error_info[i];
      OP_LOGE("Conv3DBackpropInput", "Error msg is: %s", msg.c_str());
      break;
    }
  }
}

bool CheckParams(Conv3dBpInputV2RunInfo &runInfoV2, gert::TilingContext *context, OtherParams& otherParams) {
  int64_t dedy_c_align = Ops::Base::CeilAlign(otherParams.a_shape.c, otherParams.a_shape.c0);
  int64_t dedx_c_align = Ops::Base::CeilAlign(otherParams.c_shape.c, otherParams.c_shape.c0);
  int64_t filter_c_align = Ops::Base::CeilAlign(otherParams.b_shape.c, static_cast<int64_t>(otherParams.filter_ci0));
  int64_t filter_n_align = Ops::Base::CeilAlign(otherParams.b_shape.batch, static_cast<int64_t>(otherParams.filter_co0));

  int64_t dedy_size = otherParams.a_shape.batch * dedy_c_align * otherParams.a_shape.d *
                      otherParams.a_shape.w * otherParams.a_shape.h *
                      runInfoV2.a_dtype_bytes;
  int64_t dedx_size = otherParams.a_shape.batch * dedx_c_align * otherParams.c_shape.d *
                      otherParams.c_shape.w * otherParams.c_shape.h *
                      runInfoV2.c_dtype_bytes;
  int64_t filter_size = filter_n_align * filter_c_align * otherParams.filter_d_dilation *
                        otherParams.b_shape.w * otherParams.b_shape.h *
                        runInfoV2.b_dtype_bytes;
  otherParams.fmap_d_padding =
      otherParams.c_shape.d + runInfoV2.pad_h + runInfoV2.pad_t;
  otherParams.fmap_h_padding =
      otherParams.c_shape.h + runInfoV2.pad_u + runInfoV2.pad_d;
  otherParams.fmap_w_padding =
      otherParams.c_shape.w + runInfoV2.pad_l + runInfoV2.pad_r;

  if (!CheckParamsWithLog(runInfoV2, context, otherParams) || !CheckShapeValidWithLog(context, otherParams, runInfoV2)) {
    return false;
  }

  int32_t kFilterDimHWUpTmp = kFilterDimHWUp;
  int32_t kPadUpTmp = kPadUp;
  int32_t kDilationUpTmp = kDilationUp;
  if (IsArchAfter35(context) || IsSocVersionFuse(context)) {
    kFilterDimHWUpTmp = kDimUp;
    kPadUpTmp = kDimUp;
    kDilationUpTmp = kDimUp;
  }
  uint32_t shift = 0;
  uint64_t invalid = (static_cast<uint64_t>(!CheckRange(runInfoV2.groups, kDimLow, kDimUp)) << shift++);
  invalid = invalid + (static_cast<uint64_t>(!CheckRange(otherParams.b_shape.h, kDimLow, kFilterDimHWUpTmp)) << shift++);
  invalid = invalid + (static_cast<uint64_t>(!CheckRange(otherParams.b_shape.w, kDimLow, kFilterDimHWUpTmp)) << shift++);
  invalid = invalid + (static_cast<uint64_t>(!CheckRange(otherParams.b_shape.d, kDimLow, kDimBatchUp)) << shift++);
  invalid = invalid + (static_cast<uint64_t>(!CheckRange(otherParams.a_shape.batch, kDimLow, kDimBatchUp)) << shift++);
  invalid = invalid + (static_cast<uint64_t>(!CheckLowerBound(otherParams.a_shape.c1, kDimLow)) << shift++);
  invalid = invalid + (static_cast<uint64_t>(!CheckRange(otherParams.a_shape.d, kDimLow, kDimUp)) << shift++);
  invalid = invalid + (static_cast<uint64_t>(!CheckRange(otherParams.a_shape.h, kDimLow, kDimUp)) << shift++);
  invalid = invalid + (static_cast<uint64_t>(!CheckRange(otherParams.a_shape.w, kDimLow, kDimUp)) << shift++);
  invalid = invalid + (static_cast<uint64_t>(!CheckLowerBound(otherParams.a_shape.c, kDimLow)) << shift++);
  invalid = invalid + (static_cast<uint64_t>(!CheckLowerBound(otherParams.c_shape.c1, kDimLow)) << shift++);
  invalid = invalid + (static_cast<uint64_t>(!CheckLowerBound(otherParams.c_shape.c, kDimLow)) << shift++);
  invalid = invalid + (static_cast<uint64_t>(!CheckLowerBound(otherParams.c_shape.d, kDimLow)) << shift++);
  invalid = invalid + (static_cast<uint64_t>((!CheckRange(runInfoV2.dilation_h, kDilationLow, kDilationUpTmp) ||
                        !CheckRange(runInfoV2.dilation_w, kDilationLow, kDilationUpTmp) ||
                        !CheckRange(runInfoV2.dilation_d, kDilationLow, kDilationUpTmp)))
                       << shift++);
  invalid = invalid +
            (static_cast<uint64_t>(!CheckRange(otherParams.a_shape.h * runInfoV2.stride_h, kDimLow, kDimUp))
             << shift++);
  invalid = invalid +
            (static_cast<uint64_t>(!CheckRange(otherParams.a_shape.w * runInfoV2.stride_w, kDimLow, kDimUp))
             << shift++);
  // Co % g == 0
  invalid =
      invalid + (static_cast<uint64_t>(!CheckValue(static_cast<int32_t>(
          otherParams.a_shape.c % static_cast<int64_t>(runInfoV2.groups)
      ), 0)) << shift++);
  // Ci % g == 0
  invalid =
      invalid + (static_cast<uint64_t>(!CheckValue(static_cast<int32_t>(
          otherParams.c_shape.c % static_cast<int64_t>(runInfoV2.groups)
      ), 0)) << shift++);
  // Ci == Ci / g * g
  invalid = invalid + (static_cast<uint64_t>(!CheckValue(
          static_cast<int32_t>(otherParams.c_shape.c),
          static_cast<int32_t>(otherParams.b_shape.c * static_cast<int64_t>(runInfoV2.groups))
      )) << shift++);
  invalid = invalid +
            (static_cast<uint64_t>(!CheckValue(otherParams.a_shape.c, otherParams.b_shape.batch)) << shift++);
  invalid = invalid + (static_cast<uint64_t>(!CheckValue(otherParams.a_shape.batch, otherParams.c_shape.batch))
                       << shift++);
  invalid = invalid + (static_cast<uint64_t>(otherParams.filter_d_dilation > otherParams.fmap_d_padding) << shift++);
  invalid = invalid + (static_cast<uint64_t>(otherParams.filter_h_dilation > otherParams.fmap_h_padding) << shift++);
  invalid = invalid + (static_cast<uint64_t>(otherParams.filter_w_dilation > otherParams.fmap_w_padding) << shift++);
  invalid = invalid + (static_cast<uint64_t>(!CheckRange(otherParams.c_shape.h, kDimLow, kDimUp)) << shift++);
  invalid = invalid + (static_cast<uint64_t>(!CheckRange(otherParams.c_shape.w, kDimLow, kDimUp)) << shift++);
  int64_t do_temp = (otherParams.fmap_d_padding - otherParams.filter_d_dilation) /
                    runInfoV2.stride_d + 1;
  int64_t ho_temp = (otherParams.fmap_h_padding - otherParams.filter_h_dilation) /
                    runInfoV2.stride_h + 1;
  int64_t wo_temp = (otherParams.fmap_w_padding - otherParams.filter_w_dilation) /
                    runInfoV2.stride_w + 1;
  invalid = invalid + (static_cast<uint64_t>(do_temp != otherParams.a_shape.d) << shift++);
  invalid = invalid + (static_cast<uint64_t>(ho_temp != otherParams.a_shape.h) << shift++);
  invalid = invalid + (static_cast<uint64_t>(wo_temp != otherParams.a_shape.w) << shift++);
  invalid =
      invalid + (static_cast<uint64_t>(dedy_c_align == 0 || dedx_c_align == 0 || filter_c_align == 0 || filter_n_align == 0) << shift++);
  invalid = invalid + (static_cast<uint64_t>(dedy_size > kDataSizeMax) << shift++);
  invalid = invalid + (static_cast<uint64_t>((dedx_size > kDataSizeMax)) << shift++);
  // 左移flag超过int32范围，需要加int64强转
  invalid = invalid + (static_cast<uint64_t>((filter_size > kDataSizeMax)) << shift++);
  invalid = invalid + (static_cast<uint64_t>(!CheckL1SizeLimit(runInfoV2, context, otherParams)) << shift++);
  invalid = invalid +
            (static_cast<uint64_t>(otherParams.pad_up_before > kPadUpTmp || otherParams.pad_left_before > kPadUpTmp ||
                                  otherParams.pad_down_after > kPadUpTmp || otherParams.pad_right_after > kPadUpTmp)
             << shift++);
  if (invalid != 0) {
    vector<string> error_info = {"groups must be equal 1",
                                  "kh value invalid",
                                  "kw value invalid",
                                  "kd value invalid",
                                  "batch value invalid",
                                  "co1 value invalid",
                                  "dout value invalid",
                                  "ho value invalid",
                                  "wo value invalid",
                                  "co value invalid",
                                  "c1 value invalid",
                                  "cin value invalid",
                                  "din value invalid",
                                  "h, w, d dilations value invalid",
                                  "out_backprop's H after expands is invalid",
                                  "out_backprop's W after expands is invalid",
                                  "c dim of out_backprop must be div by groups",
                                  "c dim of fmap must be div by groups",
                                  "c dim of fmap must be equal with filter c multi groups",
                                  "c dim of out_backprop must be equal with filter n",
                                  "fmap batch not equal with out_backprop batch",
                                  "filter_d_dilation or fmap_d_padding invalid",
                                  "filter_h_dilation or fmap_h_padding invalid",
                                  "filter_w_dilation or fmap_w_padding invalid",
                                  "hin value invalid",
                                  "win value invalid",
                                  "fmap_d does not match out_backprop_d",
                                  "fmap_h does not match out_backprop_h",
                                  "fmap_w does not match out_backprop_w",
                                  "ci or co is invalid",
                                  "out_backprop size larger than int64",
                                  "fmap size larger than int64",
                                  "filter size larger than int64",
                                  "this case may exceed size",
                                  "backprop pad value invalid"};
    string error_flag = IntToBinary(invalid);
    OutputErrorMsg(error_info, error_flag);
    return false;
  }
  return true;
}

bool CheckAttrs(Conv3dBpInputV2RunInfo &runInfoV2, const char* opName, OtherParams& otherParams) {
  // kernel大于1时，才能有dilation属性, 前面代码已经做了兼容性属性重设，这里做二次check
  bool dilationDFlag = (runInfoV2.dilation_d != 1 && otherParams.b_shape.d == 1);
  bool dilationHFlag = (runInfoV2.dilation_h != 1 && otherParams.b_shape.h == 1);
  bool dilationWFlag = (runInfoV2.dilation_w != 1 && otherParams.b_shape.w == 1);

  OP_CHECK_IF(dilationDFlag,
      CUBE_INNER_ERR_REPORT(opName, "cannot support dilation_d: [%d] != 1 while kernel_d: [%ld] = 1",
      runInfoV2.dilation_d, otherParams.b_shape.d), return false);

  OP_CHECK_IF(dilationHFlag,
      CUBE_INNER_ERR_REPORT(opName, "cannot support dilation_h: [%d] != 1 while kernel_h: [%ld] = 1",
      runInfoV2.dilation_h, otherParams.b_shape.h), return false);

  OP_CHECK_IF(dilationWFlag,
      CUBE_INNER_ERR_REPORT(opName, "cannot support dilation_w: [%d] != 1 while kernel_w: [%ld] = 1",
      runInfoV2.dilation_w, otherParams.b_shape.w), return false);

  OP_CHECK_IF(runInfoV2.stride_d > otherParams.b_shape.d,
      OP_LOGE(opName, "cannot support stride_d: %d > kernel_d: %ld", runInfoV2.stride_d, otherParams.b_shape.d),
      return false);

  OP_CHECK_IF(CheckRange(runInfoV2.stride_h, DIM_LOW, STRIDES_DIM_HW_UP) == false,
      OP_LOGE(opName, "stride_h: %d is invalid, support range [%d, %d]", runInfoV2.stride_h, DIM_LOW, STRIDES_DIM_HW_UP),
      return false);

  OP_CHECK_IF(CheckRange(runInfoV2.stride_w, DIM_LOW, STRIDES_DIM_HW_UP) == false,
      OP_LOGE(opName, "stride_w: %d is invalid, support range [%d, %d]", runInfoV2.stride_w, DIM_LOW, STRIDES_DIM_HW_UP),
      return false);

  OP_CHECK_IF(CheckRange(runInfoV2.stride_d, DIM_LOW, STRIDES_DIM_DEPTH_UP) == false,
      OP_LOGE(opName, "stride_d: %d is invalid, support range [%d, %d]", runInfoV2.stride_d, DIM_LOW, STRIDES_DIM_DEPTH_UP),
      return false);
  uint64_t curL0CDstStride = static_cast<uint64_t>(otherParams.c_shape.h) * otherParams.c_shape.w;
  OP_CHECK_IF(curL0CDstStride > UINT32_MAX,
      OP_LOGE(opName, "cannot support hi * wi=%lu over %u", curL0CDstStride, UINT32_MAX),
      return false);

  OP_CHECK_IF(
      CheckRange(runInfoV2.groups, GROUPS_LOW, GROUPS_UP) == false,
      OP_LOGE(opName, "only support groups(%d) in range [%d, %d]", runInfoV2.groups, GROUPS_LOW, GROUPS_UP),
      return false);
  return true;
}

bool CheckPadRange(Conv3dBpInputV2RunInfo &runInfoV2, const char* opName, OtherParams& otherParams) {
    int32_t padHDimUp = std::min(PAD_DIM_UP, static_cast<int32_t>(otherParams.b_shape.d - 1));
    int32_t padUDimUp = std::min(PAD_DIM_UP, static_cast<int32_t>(otherParams.b_shape.h - 1));
    int32_t padLDimUp = std::min(PAD_DIM_UP, static_cast<int32_t>(otherParams.b_shape.w - 1));
    OP_CHECK_IF(CheckRange(runInfoV2.pad_h, PAD_DIM_LOW, padHDimUp) == false,
                OP_LOGE(opName, "pad head: %d invalid, it should be in [%d, %d]", runInfoV2.pad_h, PAD_DIM_LOW, padHDimUp),
                return false);
    OP_CHECK_IF(CheckRange(runInfoV2.pad_t, PAD_DIM_LOW, padHDimUp) == false,
                OP_LOGE(opName, "pad tail: %d invalid, it should be in [%d, %d]", runInfoV2.pad_t, PAD_DIM_LOW, padHDimUp),
                return false);
    OP_CHECK_IF(CheckRange(runInfoV2.pad_u, PAD_DIM_LOW, padUDimUp) == false,
                OP_LOGE(opName, "pad up: %d invalid, it should be in [%d, %d]", runInfoV2.pad_u, PAD_DIM_LOW, padUDimUp),
                return false);
    OP_CHECK_IF(CheckRange(runInfoV2.pad_d, PAD_DIM_LOW, padUDimUp) == false,
                OP_LOGE(opName, "pad down: %d invalid, it should be in [%d, %d]", runInfoV2.pad_d, PAD_DIM_LOW, padUDimUp),
                return false);
    OP_CHECK_IF(CheckRange(runInfoV2.pad_l, PAD_DIM_LOW, padLDimUp) == false,
                OP_LOGE(opName, "pad left: %d invalid, it should be in [%d, %d]", runInfoV2.pad_l, PAD_DIM_LOW, padLDimUp),
                return false);
    OP_CHECK_IF(CheckRange(runInfoV2.pad_r, PAD_DIM_LOW, padLDimUp) == false,
                OP_LOGE(opName, "pad right: %d invalid, it should be in [%d, %d]", runInfoV2.pad_r, PAD_DIM_LOW, padLDimUp),
                return false);
    return true;
}

template <typename T>
std::string DebugString(const std::vector<T>& v) {
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

bool CheckTranspose(const char* opName, const gert::TilingContext* context) {
    auto offsetWShape = context->GetOptionalInputShape(OFFSET_W_INDEX);

    auto attrs = context->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(opName, "attrs is null"), return false);

    const auto offsetX = attrs->GetAttrPointer<int64_t>(OFFSET_X_INDEX);
    auto outputPadding = attrs->GetAttrPointer<gert::ContinuousVector>(OUTPUT_PADDING_INDEX);

    if (outputPadding != nullptr) {
        OP_CHECK_IF(outputPadding->GetData() == nullptr,
                    OP_LOGE(opName, "output_padding GetData is null"), return false);
        OP_CHECK_IF(outputPadding->GetSize() != OUTPUT_PADDING_DIM,
                    OP_LOGE(opName, "the output padding:%zu is invalid, it should be 5d", outputPadding->GetSize()),
                    return false);
        const auto outputPaddingData = static_cast<const int64_t *>(outputPadding->GetData());
        bool outputPaddingAllzero = true;
        bool outputPaddingAllNonNegative = true;
        std::vector<int64_t> outputPaddingValue;
        for (size_t index = 0; index < outputPadding->GetSize(); index++) {
            outputPaddingAllzero = outputPaddingData[index] != 0 ? false : outputPaddingAllzero;
            outputPaddingAllNonNegative = outputPaddingData[index] < 0 ? false : outputPaddingAllNonNegative;
            outputPaddingValue.push_back(outputPaddingData[index]);
        }
        OP_CHECK_IF((!outputPaddingAllzero) &&
                ((context->GetInputDesc(FILTER_INDEX)->GetDataType() != ge::DT_BF16 &&
                context->GetInputDesc(FILTER_INDEX)->GetDataType() != ge::DT_FLOAT16 &&
                context->GetInputDesc(FILTER_INDEX)->GetDataType() != ge::DT_FLOAT) ||
                (context->GetInputDesc(OUT_BACKPROP_INDEX)->GetDataType() != ge::DT_BF16 &&
                context->GetInputDesc(OUT_BACKPROP_INDEX)->GetDataType() != ge::DT_FLOAT16 &&
                context->GetInputDesc(OUT_BACKPROP_INDEX)->GetDataType() != ge::DT_FLOAT)),
                CUBE_INNER_ERR_REPORT(opName,
                "when output_padding[%s] is not all zero, op can only support all input are bfloat16 or float16 or float32, get filter dtype[%s], output backprop dtype[%s]",
                DebugString(outputPaddingValue).c_str(),
                ge::TypeUtils::DataTypeToSerialString(context->GetInputDesc(FILTER_INDEX)->GetDataType()).c_str(),
                ge::TypeUtils::DataTypeToSerialString(context->GetInputDesc(OUT_BACKPROP_INDEX)->GetDataType()).c_str()),
                return false);
        OP_CHECK_IF(!outputPaddingAllNonNegative && IsArchAfter35(context),
                CUBE_INNER_ERR_REPORT(opName,
                "output_padding[%s] contains negative values, op can only support all non-negative inputs.",
                DebugString(outputPaddingValue).c_str()), return false);
    }
    OP_CHECK_IF(offsetX != nullptr && *offsetX != 0,
                OP_LOGE(opName, "cannot support offset_x attribute parameters"), return false);
    if (!IsSocVersionFuse(context)) {
        OP_CHECK_IF(offsetWShape != nullptr && offsetWShape->GetStorageShape().GetShapeSize() != 0,
                OP_LOGE(opName,"cannot support offset_w input parameters"), return false);
        auto biasShape = context->GetOptionalInputShape(BAIS_INDEX);
        OP_CHECK_IF(biasShape != nullptr && biasShape->GetStorageShape().GetShapeSize() != 0,
                OP_LOGE(opName, "cannot support bias"), return false);
    }

    return true;
}

void SetInitOutput(Conv3dBpInputV2RunInfo &runInfoV2, const optiling::OpTypeV2 opType, const OtherParams& otherParams) {
    int32_t doModulo = (otherParams.fmap_d_padding - otherParams.filter_d_dilation) %
                        runInfoV2.stride_d;
    int32_t hoModulo = (otherParams.fmap_h_padding - otherParams.filter_h_dilation) %
                        runInfoV2.stride_h;
    if (doModulo > runInfoV2.pad_t ||
        hoModulo > runInfoV2.pad_d ||
        runInfoV2.stride_h > otherParams.b_shape.h ||
        (opType == optiling::OpTypeV2::kConv3DTransposeV2 &&
          (otherParams.output_padding.output_padding_d > 0 || otherParams.output_padding.output_padding_h > 0)) ||
        runInfoV2.dilation_d > 1) {
        // 1 is init output with l0C, 2 is init output with l1, defualt is 0 means not init output
        runInfoV2.initOutputFlag = 1;
    }
}
}

bool IsSocVersionFuse(const gert::TilingContext *context)
{
    const auto op_name = context->GetNodeName();
    OP_CHECK_IF(context == nullptr, OP_LOGE(op_name, "context is null"), return false);
    fe::PlatFormInfos *platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OP_LOGE(op_name, "platformInfoPtr is null"), return false);

    std::string cube_vec_state_str;
    platformInfo->GetPlatformRes("SoCInfo", "cube_vector_combine", cube_vec_state_str);
    OP_CHECK_IF(cube_vec_state_str.empty(), OP_LOGE(op_name, "get cube_vector_combine failed"), return false);
    if (cube_vec_state_str == "fuse") {
      return true;
    }
    return false;
}

bool SetRunInfoToV2(gert::TilingContext* context, Conv3dBpInputV2RunInfo& runInfoV2,
                    optiling::OpTypeV2 opType) {
    OtherParams otherParams;
    OP_CHECK_IF(!ValidateConvBackpropContext(context),
                OP_LOGE(context->GetNodeName(), "failed to parse params"), return false);
    OP_CHECK_IF(!Conv3DBackpropInputParseFunc(context, opType, runInfoV2, otherParams, true),
                OP_LOGE(context->GetNodeName(), "failed to parse context"), return false);
    if (IsSocVersionFuse(context)) {
        OP_CHECK_IF(!GetFusionMode(runInfoV2, context->GetNodeName(), context, opType),
            OP_LOGE(context, "failed to get enRelu flag"), return false);
    } else {
        OP_CHECK_IF(!GetImplMode(runInfoV2, context->GetNodeName(), context, opType),
            OP_LOGE(context, "failed to get impl mode"), return false);
    }

    if (!CheckCalPads(context, runInfoV2, opType, otherParams) || !CheckParams(runInfoV2, context, otherParams) ||
        !CheckAttrs(runInfoV2, context->GetNodeName(), otherParams) || !CheckPadRange(runInfoV2, context->GetNodeName(), otherParams)) {
        OP_LOGE(context, "params is invalid");
        return false;
    }

    if (opType == optiling::OpTypeV2::kConv3DTransposeV2 && !CheckTranspose(context->GetNodeName(), context)) {
        OP_LOGE(context, "params is invalid");
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

    runInfoV2.enlarge = otherParams.multiple_extend; // arch35
    runInfoV2.dedy_cout_g = otherParams.b_shape.batch / runInfoV2.groups * otherParams.multiple_extend; // arch35
    runInfoV2.dedx_cin_g = otherParams.b_shape.c * otherParams.multiple_extend; // arch35
    return true;
}

} // namespace Conv
} // namespace NN
} // namespace Ops
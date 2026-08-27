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
 * \file conv_backprop_input_context_utils_internal.h
 * \brief
 */
#ifndef CONV_BACKPROP_INPUT_CONTEXT_UTILS_INTERNAL_H
#define CONV_BACKPROP_INPUT_CONTEXT_UTILS_INTERNAL_H

#include "conv_backprop_input_context_utils.h"
#include <log/log.h>
#include <util/math_util.h>
#include <unordered_set>
#include <cstdarg>
#include "error_util.h"
#include "conv/common/op_host/op_tiling/conv_math_util.h"
#include "conv/common/op_host/op_tiling/conv_platform_util.h"
#include "securec.h"
#include "conv/common/op_host/op_tiling/arch35/conv_base_numblocks_decision.h"

namespace Ops {
namespace NN {
namespace Conv {

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
constexpr size_t kPaddingConv3dBpInputIdx = 6;
constexpr size_t kPaddingConv3dTransposeIdx = 8;
constexpr size_t kPaddingExtendConvTransposeIdx = 9;

constexpr int32_t kBlockSize = 16;
const int32_t BYTE_BLOCK = 32;
constexpr int32_t kBit8BlockReduce = 32;
constexpr int32_t kFP32BlockReduce = 8;
const std::map<int32_t, int32_t> kDtypeBlockReduceMap = {{ge::DT_HIFLOAT8, kBit8BlockReduce},
                                                         {ge::DT_FLOAT8_E4M3FN, kBit8BlockReduce},
                                                         {ge::DT_FLOAT16, kBlockSize},
                                                         {ge::DT_FLOAT, kFP32BlockReduce},
                                                         {ge::DT_INT8, kBit8BlockReduce}};
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
struct OutputPadding {
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
    int32_t filter_co0 = 16; // co0 in fractal_z
    int32_t filter_ci0 = 16; // cin0 in fractal_z
    int32_t co1g_reduce = 0; // co1g calculated by block_reduce depend on dtype
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

bool CheckRangeInt64(int64_t value, int32_t value_low, int32_t value_up);
bool IsArchAfter35(const gert::TilingContext* context);
bool IsSupportedDtypeForOutputPadding(const ge::DataType dtype);
bool ValidateConvBackpropContext(const gert::TilingContext* context);
bool CheckAttrRangeDilations(const gert::TilingContext* context, const int64_t* dilations);
bool CheckAttrRangeStrides(const gert::TilingContext* context, const int64_t* strides);
bool CheckAttrRangePads(const gert::TilingContext* context, const int64_t* pads);
bool CheckAttrRange(gert::TilingContext* context, const int64_t* strides, const int64_t* pads, const int64_t* dilations,
                    const int64_t* groups);
bool CheckTransposeAttr(gert::TilingContext* context, OtherParams& otherParams);
template <typename T>
void GetNCDHWShape(const T& origin_shape, Shape& ncdhw_shape, const ge::Format& origin_format);
bool CheckTransposeOutputdingRange(const gert::TilingContext* context, const Conv3dBpInputV2RunInfo& runInfoV2,
                                   const OtherParams& otherParams);
bool UpdateDtypeParams(const gert::TilingContext* context, Conv3dBpInputV2RunInfo& runInfoV2,
                       const optiling::OpTypeV2 op_type, OtherParams& otherParams);
std::string FormatSetToString(const std::unordered_set<ge::Format>& format_set);
bool CheckStorageFormat(const gert::TilingContext* context, size_t filter_input_index, size_t out_backprop_input_index,
                        optiling::OpTypeV2 op_type);
bool UpdateShapeParams(const gert::TilingContext* context, const Conv3dBpInputV2RunInfo& runInfoV2,
                       const Shape& out_backprop_shape_ncdhw, const Shape& filter_shape_ncdhw,
                       const Shape& y_shape_ncdhw, OtherParams& otherParams);
void ExtractStorageShapeInfo(const gert::TilingContext* context, size_t filter_input_index,
                             size_t out_backprop_input_index, const Conv3dBpInputV2RunInfo& runInfoV2,
                             OtherParams& otherParams);
bool ValidateOriginShapeDims(const gert::TilingContext* context, const gert::Shape& out_backprop_ori_shape,
                             const gert::Shape& filter_ori_shape, const gert::Shape& y_ori_shape);
bool CalShapeInfoFromDesc(const gert::TilingContext* context, size_t filter_input_index,
                          size_t out_backprop_input_index, const Conv3dBpInputV2RunInfo& runInfoV2,
                          OtherParams& otherParams);
bool GetShapeParams(gert::TilingContext* context, Conv3dBpInputV2RunInfo& runInfoV2, optiling::OpTypeV2 op_type,
                    bool isV2Impl, OtherParams& otherParams);
void ReCalDilation(const gert::TilingContext* context, Conv3dBpInputV2RunInfo& runInfoV2,
                   const OtherParams& otherParams);
bool CalGroups(gert::TilingContext* context, OtherParams& otherParams, Conv3dBpInputV2RunInfo& runInfoV2);
template <class T>
bool CheckAllZero(const T* tensor_data, size_t dim_size);
bool CheckInputSizeAllZero(const gert::TilingContext* context, bool& allzero);
bool HandleConv3DTranspose(gert::TilingContext* context, const Conv3dBpInputV2RunInfo& runInfoV2,
                           OtherParams& otherParams);
bool CheckCalPads(const gert::TilingContext* context, const Conv3dBpInputV2RunInfo& runInfoV2,
                  optiling::OpTypeV2 op_type, const OtherParams& otherParams);
bool CalPads(gert::TilingContext* context, Conv3dBpInputV2RunInfo& runInfoV2, optiling::OpTypeV2 op_type,
             OtherParams& otherParams);
int32_t CalFmapH(const gert::TilingContext* context, const Conv3dBpInputV2RunInfo& runInfoV2,
                 const OtherParams& otherParams);
bool IsNeedTilingHkWk(gert::TilingContext* context, const Conv3dBpInputV2RunInfo& runInfoV2,
                      const OtherParams& otherParams);
bool CalRealG(gert::TilingContext* context, Conv3dBpInputV2RunInfo& runInfoV2, OtherParams& otherParams);
int32_t CalBackpropPadBefore(int32_t filter, int32_t dilation, int32_t pad);
int64_t CalBackpropPadAfter(int64_t inputDim, int64_t outputDim, int32_t stride, int32_t pad);
bool IsOverflowInt32(int64_t value);
bool CheckRange(int32_t value, int32_t value_low, int32_t value_up);
bool CalModifyBackpropPadD(const gert::TilingContext* context, Conv3dBpInputV2RunInfo& runInfoV2,
                           OtherParams& otherParams);
bool CalModifyBackpropPadHW(const gert::TilingContext* context, Conv3dBpInputV2RunInfo& runInfoV2,
                            OtherParams& otherParams);
bool CalModify(gert::TilingContext* context, Conv3dBpInputV2RunInfo& runInfoV2, OtherParams& otherParams);
bool CheckLowerBound(int32_t value, int32_t value_low);
bool CheckValue(int32_t value, int32_t value_temp);
bool CheckPadParamsWithLog(const Conv3dBpInputV2RunInfo& runInfoV2, const gert::TilingContext* context);
bool CheckParamsWithLog(Conv3dBpInputV2RunInfo& runInfoV2, gert::TilingContext* context, OtherParams& otherParams);
int64_t GetDfactorSdEqKd(const Conv3dBpInputV2RunInfo& runInfoV2, int32_t l0c_din, const OtherParams& otherParams);
template <typename T>
int64_t GetDfactor(T kd_factor, Conv3dBpInputV2RunInfo& runInfoV2, int32_t l0c_din, gert::TilingContext* context,
                   OtherParams& otherParams);
bool CheckL1SizeLimit(Conv3dBpInputV2RunInfo& runInfoV2, gert::TilingContext* context, OtherParams& otherParams);
void SetConvAttrs(Conv3dBpInputV2RunInfo& runInfoV2, const int64_t* pads_data, Shape& strides_ncdhw,
                  Shape& dilations_ncdhw, const int64_t* groups, OtherParams& otherParams);
bool GetAttrAndDtypeParams(gert::TilingContext* context, Conv3dBpInputV2RunInfo& runInfoV2, optiling::OpTypeV2 op_type,
                           OtherParams& otherParams);
bool GetInputOutputFormat(const gert::TilingContext* context, Conv3dBpInputV2RunInfo& runInfoV2,
                          const optiling::OpTypeV2 opType);
bool CalScale(const gert::TilingContext* context, const Conv3dBpInputV2RunInfo& runInfoV2,
              const OtherParams& otherParams);
bool Conv3DBackpropInputParseFunc(gert::TilingContext* context, optiling::OpTypeV2 opType,
                                  Conv3dBpInputV2RunInfo& runInfoV2, OtherParams& otherParams, bool isV2Impl);
bool GetFusionMode(Conv3dBpInputV2RunInfo& runInfoV2, const char* opName, const gert::TilingContext* context,
                   optiling::OpTypeV2 opType);
bool GetImplMode(Conv3dBpInputV2RunInfo& runInfoV2, const char* opName, const gert::TilingContext* context,
                 optiling::OpTypeV2 opType);
bool CheckFilterShapeHW(const char* op_name, const OtherParams& otherParams, int32_t kFilterDimHWUpTmp);
bool CheckAShapeParams(const char* op_name, const OtherParams& otherParams);
bool CheckCShapeParams(const char* op_name, const OtherParams& otherParams);
bool CheckShapeValidWithLog(const gert::TilingContext* context, const OtherParams& otherParams,
                            const Conv3dBpInputV2RunInfo& runInfoV2);
std::string IntToBinary(uint64_t& n);
void CalculateAlignAndSize(const OtherParams& otherParams, const Conv3dBpInputV2RunInfo& runInfoV2,
                           int64_t& dedy_c_align, int64_t& dedx_c_align, int64_t& filter_c_align,
                           int64_t& filter_n_align, int64_t& dedy_size, int64_t& dedx_size, int64_t& filter_size);
void SetFmapPaddingParams(OtherParams& otherParams, const Conv3dBpInputV2RunInfo& runInfoV2);
bool CheckGroupsAndFilterRange(const char* opName, const Conv3dBpInputV2RunInfo& runInfoV2,
                               const OtherParams& otherParams, int32_t kFilterDimHWUpTmp);
bool CheckAShapeRangeInParams(const char* opName, const OtherParams& otherParams);
bool CheckCShapeRangeInParams(const char* opName, const OtherParams& otherParams);
bool CheckDilationsRangeInParams(const char* opName, const Conv3dBpInputV2RunInfo& runInfoV2, int32_t kDilationUpTmp);
bool CheckStridesExpandRange(const char* opName, const OtherParams& otherParams,
                             const Conv3dBpInputV2RunInfo& runInfoV2);
bool CheckGroupsDivisibility(const char* opName, const OtherParams& otherParams,
                             const Conv3dBpInputV2RunInfo& runInfoV2);
bool CheckBatchConsistency(const char* opName, const OtherParams& otherParams);
bool CheckFilterDilationVsPadding(const char* opName, const OtherParams& otherParams);
bool CheckOutputShapeMatch(const char* opName, const OtherParams& otherParams, const Conv3dBpInputV2RunInfo& runInfoV2);
bool CheckAlignAndSizeLimits(const char* opName, int64_t dedy_c_align, int64_t dedx_c_align, int64_t filter_c_align,
                             int64_t filter_n_align, int64_t dedy_size, int64_t dedx_size, int64_t filter_size);
bool CheckBackpropPadRange(const char* opName, const OtherParams& otherParams, int32_t kPadUpTmp);
bool CheckParams(Conv3dBpInputV2RunInfo& runInfoV2, gert::TilingContext* context, OtherParams& otherParams);
bool CheckAttrs(const gert::TilingContext* context, Conv3dBpInputV2RunInfo& runInfoV2, const char* opName,
                OtherParams& otherParams);
bool CheckPadRange(const gert::TilingContext* context, Conv3dBpInputV2RunInfo& runInfoV2, const char* opName);
template <typename T>
std::string DebugString(const std::vector<T>& v);
bool ValidateOutputPaddingData(const char* opName, const gert::ContinuousVector* outputPadding,
                               bool& outputPaddingAllzero, bool& outputPaddingAllNonNegative,
                               std::vector<int64_t>& outputPaddingValue);
bool CheckOutputPaddingDtype(const char* opName, const gert::TilingContext* context,
                             const std::vector<int64_t>& outputPaddingValue, bool outputPaddingAllzero,
                             bool outputPaddingAllNonNegative);
bool CheckOffsetXAndOffsetW(const char* opName, const gert::TilingContext* context, const gert::RuntimeAttrs* attrs);
bool CheckTranspose(const char* opName, const gert::TilingContext* context);
bool CheckBiasParams(gert::TilingContext* context, const OtherParams& otherParams);
void SetInitOutput(Conv3dBpInputV2RunInfo& runInfoV2, const optiling::OpTypeV2 opType, const OtherParams& otherParams);

} // namespace Conv
} // namespace NN
} // namespace Ops

#endif

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_CONVOLUTION_BACKWARD_CHECKER_H_
#define OP_API_INC_CONVOLUTION_BACKWARD_CHECKER_H_

#include "aclnn_convolution_backward.h"
#include "convolutionbackward.h"

#include "matmul/common/op_host/op_api/matmul_util.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/transdata.h"
#include "../../../convolution_forward/op_host/op_api/convolution.h"
#include "pooling/avg_pool3_d_grad/op_host/op_api/dilation.h"
#include "level0/fill.h"
#include "level0/reduce_sum_op.h"
#include "level0/squeeze.h"
#include "level0/unsqueeze.h"
#include "level0/zero_op.h"
#include "../../../common/op_host/op_api/conv_cube_util.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "runtime/context.h"
#ifdef __cplusplus
extern "C" {
#endif

namespace {
const int kHDimNC1HWC0Idx = 2;
const int kWDimNC1HWC0Idx = 3;
const int kHDimNCHWIdx = 2;
const int kWDimNCHWIdx = 3;
const int nDimNCLIdx = 0;
const int cDimNCLIdx = 1;
const int lDimNCLIdx = 2;
const int coDimCoCiDHWIdx = 0;
const int ciDimCoCiDHWIdx = 1;
const int nDimNCDHWIdx = 0;
const int cDimNCDHWIdx = 1;
// 工具中-------------------------
const int dDimNCDHWIdx = 2;
const int hDimNCDHWIdx = 3;
const int wDimNCDHWIdx = 4;
const int32_t kHDimNDHWCIdx = 2;
const int32_t kWDimNDHWCIdx = 3;
// -----------------------------
const int kDILATIONHIdx = 0;
const int kDILATIONWIdx = 1;
const int kSTRIDEHIdx = 0;
const int kSTRIDEWIdx = 1;
const int kPADDINGUPIdx = 0;
const int kPADDINGLEFTIdx = 1;
const int kPadding4UpIdx = 0;
const int kPadding4DownIdx = 1;
const int kPadding4LeftIdx = 2;
const int kPadding4RightIdx = 3;
const int32_t CONV2D_ATTR_DIM = 2;
const int32_t CONV3D_ATTR_DIM = 3;
const int32_t CONV3D_PAD_DIM = 6;
const int32_t CONV3D_ATTR_D_IDX = 0;
const int32_t CONV3D_ATTR_H_IDX = 1;
const int32_t CONV3D_ATTR_W_IDX = 2;
const int32_t CONV3D_PAD_HEAD_IDX = 0;
const int32_t CONV3D_PAD_TAIL_IDX = 1;
const int32_t CONV3D_PAD_TOP_IDX = 2;
const int32_t CONV3D_PAD_BOTTOM_IDX = 3;
const int32_t CONV3D_PAD_LEFT_IDX = 4;
const int32_t CONV3D_PAD_RIGHT_IDX = 5;
const int32_t CONV2D_PAD_TOP_IDX = 0;
const int32_t CONV2D_PAD_BOTTOM_IDX = 1;
const int32_t CONV2D_PAD_LEFT_IDX = 2;
const int32_t CONV2D_PAD_RIGHT_IDX = 3;
const int32_t NCDHW_N_DIM = 0;
const int32_t NCDHW_C_DIM = 1;
const int32_t NCDHW_D_DIM = 2;
const int32_t NCDHW_H_DIM = 3;
const int32_t NCDHW_W_DIM = 4;
static std::map<ge::Format, std::string> g_formatToStrTab = {
    {ge::FORMAT_NCHW, "NCHW"},
    {ge::FORMAT_NHWC, "NHWC"},
    {ge::FORMAT_HWCN, "HWCN"},
    {ge::FORMAT_DHWNC, "DHWNC"},
    {ge::FORMAT_DHWCN, "DHWCN"},
    {ge::FORMAT_NDHWC, "NDHWC"},
    {ge::FORMAT_NCDHW, "NCDHW"},
    {ge::FORMAT_NC1HWC0, "NC1HWC0"},
    {ge::FORMAT_ND, "ND"},
    {ge::FORMAT_NDC1HWC0, "NDC1HWC0"},
    {ge::FORMAT_FRACTAL_Z_3D, "FRACTAL_Z_3D"}};
}

struct ConvolutionBackwardOutput {
  aclTensor *gradInput;
  aclTensor *gradWeight;
  aclTensor *gradBias;
};

struct ConvolutionBackwardResult {
  const aclTensor *gradInput;
  const aclTensor *gradWeight;
  const aclTensor *gradBias;
};

struct BatchMatmulInput {
  const aclTensor *leftData;
  const aclTensor *rightData;
  const aclTensor *outputData;
  bool isLeftTranspose;
  bool isRightTranspose;
};

enum class Conv3DBp2MmMode {
  CONV3D_BP_NO_MM = 0,
  CONV3D_BP_MM_1x1_KERNEL = 1,
  CONV3D_BP_MM_STRIDE_EQ_KERNEL = 2,
  CONV3D_BP_MM_FEATURE_MAP_EQ_KERNEL = 3,
};

struct ExpectValue {
  int64_t doExpect;
  int64_t hoExpect;
  int64_t woExpect;
};

bool CheckDtypeValid(const aclTensor *inputTensor, bool transposed = false);
bool CheckParamsValueAllZero(const aclIntArray *params);
bool CheckParamsValue(const aclIntArray *params, bool isPad);
// 需要调整
void GetChannleIndex(const op::Shape &shape, const op::Format &format, int64_t &channelIndex);
bool CheckResolutionGEKernelShape(const op::Shape &inputShape, const op::Shape &weightShape, const l0op::ConvolutionBackwardParams &params, int64_t dimIdx);
void GetInputShapeSize(const op::Format &format, const op::Shape &shape, int64_t &shapeDVal, int64_t &shapeHVal, int64_t &shapeWVal);
bool GetExpectValueDHW_95(const l0op::ConvolutionBackwardInputTensor &inputTensor, const l0op::ConvolutionBackwardParams &params, struct ExpectValue &expectValue, const op::Shape &inputShape, const op::Shape &weightShape);
bool CheckResolutionGEKernelShape_95(int64_t inputVal, int64_t weightVal, int64_t dimOrder, const l0op::ConvolutionBackwardParams &params);
int64_t GetExpectNum_95(int64_t inputVal, int64_t weightVal, int64_t dimOrder, const l0op::ConvolutionBackwardParams &params);
int64_t GetExpectNum(const op::Shape &inputShape, const op::Shape &weightShape, const l0op::ConvolutionBackwardParams &params, int64_t dimIdx);
bool CheckFormatValid(const aclTensor *inputTensor, const string &tensorName);
void GetWeightShapeSize(const op::Format &weightFormat, const op::Shape &weightShape, int64_t &weightDVal, int64_t &weightHVal, int64_t &weightWVal);
string AclarrayToString(const aclIntArray *array);
aclnnStatus CalculateConvolutionBackwardWithEmpty(l0op::ConvolutionBackwardInputTensor &inputTensor,
                                                  ConvolutionBackwardOutput &outputTensor,
                                                  l0op::ConvolutionBackwardParams &params, aclOpExecutor *executor);

namespace Ops {
namespace NN {
namespace Conv {

class ConvolutionBackwardChecker {
public:
ConvolutionBackwardChecker(const l0op::ConvolutionBackwardInputTensor &inputTensor, const ConvolutionBackwardOutput &outputTensor,
                           const l0op::ConvolutionBackwardParams &params, const op::SocVersion socVersion):
                           inputTensor_(inputTensor),
                           outputTensor_(outputTensor),
                           params_(params),
                           socVersion_(socVersion){}

public:

bool CheckDataTypeValidForGradInput();

bool CheckDataTypeValidForGradWeight();

bool CheckDataTypeValidForGradBias();

bool CheckDtypeValidFor8bit(const op::DataType& dType);

bool InterceptConvFor8bit();

bool IsConv8bit(const op::DataType& dType) const;

bool CheckDtypeValidForBpFilter8bit(const op::DataType& dType);

bool CheckParamsValidForBpFilter8bit();

bool CheckConvParams(size_t inputDim);
// 需要调整
inline op::DataType CalcPromoteType();

bool CheckCubeMathTypeConvBackward();

bool CheckConvShape();

bool CheckConvChannelAndGroup();

bool CheckConvShapePlus();

inline bool CheckNotNull();

aclnnStatus CheckParamsFor8Bit();

aclnnStatus CheckParams();

bool CheckParamsDim();

bool CheckEmptyTensor();

bool CheckParamsGroup();

bool CheckShapeTransposed();

bool CheckShape();

bool CheckShapeEmpty();

private:
const l0op::ConvolutionBackwardInputTensor inputTensor_;
const ConvolutionBackwardOutput outputTensor_;
const l0op::ConvolutionBackwardParams params_;
const op::SocVersion socVersion_;
};
} // namespace Conv
} // namespace NN
} // namespace Ops
#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_CONVOLUTION_BACKWARD_CHECKER_H_

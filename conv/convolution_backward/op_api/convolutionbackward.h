/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_OP_API_COMMON_INC_LEVEL0_OP_CONVOLUTIONBACKWARD_OP_H_
#define OP_API_OP_API_COMMON_INC_LEVEL0_OP_CONVOLUTIONBACKWARD_OP_H_

#include "opdev/op_executor.h"
namespace l0op {
const int CONV1DINPUTDIM = 3;
const int CONV2DINPUTDIM = 4;
const int CONV3DINPUTDIM = 5;

struct ConvBackpropParams {
  const aclTensor *input;
  const aclTensor *weight;
  const aclTensor *outBackprop;
  const aclIntArray *stride;
  const aclIntArray *padding;
  const aclIntArray *dilation;
  int groups;
};

struct ConvolutionBackwardInputTensor {
  const aclTensor *gradOutput;
  const aclTensor *input;
  const aclTensor *weight;
};

struct ConvolutionBackwardParams {
  const aclIntArray *biasSizes;
  const aclIntArray *stride;
  const aclIntArray *padding;
  const aclIntArray *dilation;
  const bool transposed;
  const aclIntArray *outputPadding;
  const int groups;
  const aclBoolArray *outputMask;
  const int8_t cubeMathType;
};

// Conv2dBackpropInput
// 5HD->FZ with Fp16
const aclTensor *Conv2DBackpropInputFp162Fp16(const aclTensor *input, const aclTensor *weight,
                                              const aclTensor *outBackprop, const aclIntArray *stride,
                                              const aclIntArray *padding, const aclIntArray *dilation, int groups,
                                              aclOpExecutor *executor);
// 1971 5HD->FZ with Fp32
const aclTensor *Conv2DBackpropInputFp322Fp32(const aclTensor *input, const aclTensor *weight,
                                              const aclTensor *outBackprop, const aclIntArray *stride,
                                              const aclIntArray *padding, const aclIntArray *dilation, int groups,
                                              aclOpExecutor *executor);
// 1971 5HD->FZ with Hf32
const aclTensor *Conv2DBackpropInputHf32(const aclTensor *input, const aclTensor *weight, const aclTensor *outBackprop,
                                         const aclIntArray *stride, const aclIntArray *padding,
                                         const aclIntArray *dilation, int groups, aclOpExecutor *executor);
// 1971 5HD->FZ with Bf16
const aclTensor *Conv2DBackpropInputBf162Bf16(const aclTensor *input, const aclTensor *weight,
                                              const aclTensor *outBackprop, const aclIntArray *stride,
                                              const aclIntArray *padding, const aclIntArray *dilation, int groups,
                                              aclOpExecutor *executor);
// 1971 5HD->FZ with input dataType and Hf32Flag.
const aclTensor *Conv2DBackpropInput(const aclTensor *input, const aclTensor *weight,
                                     const aclTensor *outBackprop, const aclIntArray *stride,
                                     const aclIntArray *padding, const aclIntArray *dilation, int groups,
                                     aclOpExecutor *executor, bool useHf32Flag, op::DataType dataType);
// Conv2dBackpropFilter
// 5HD->FZ with Fp16
const aclTensor *Conv2DBackpropFilterFp162Fp32(const aclTensor *input, const aclTensor *weight,
                                               const aclTensor *outBackprop, const aclIntArray *stride,
                                               const aclIntArray *padding, const aclIntArray *dilation, int groups,
                                               aclOpExecutor *executor);

// 1971 5HD->FZ with Fp32
const aclTensor *Conv2DBackpropFilterFp322Fp32(const aclTensor *input, const aclTensor *weight,
                                               const aclTensor *outBackprop, const aclIntArray *stride,
                                               const aclIntArray *padding, const aclIntArray *dilation, int groups,
                                               aclOpExecutor *executor);
// 1971 5HD->FZ with Hf32
const aclTensor *Conv2DBackpropFilterHf32(const aclTensor *input, const aclTensor *weight, const aclTensor *outBackprop,
                                          const aclIntArray *stride, const aclIntArray *padding,
                                          const aclIntArray *dilation, int groups, aclOpExecutor *executor);
// 1971 5HD->FZ with Bf16
const aclTensor *Conv2DBackpropFilterBf162Fp32(const aclTensor *input, const aclTensor *weight,
                                               const aclTensor *outBackprop, const aclIntArray *stride,
                                               const aclIntArray *padding, const aclIntArray *dilation, int groups,
                                               aclOpExecutor *executor);

// Conv3dBackpropFilter
// 6HD->FZ_3D with Fp16
const aclTensor *Conv3DBackpropFilterFp162Fp32(const aclTensor *input, const aclTensor *weight,
                                               const aclTensor *outBackprop, const aclIntArray *stride,
                                               const aclIntArray *padding, const aclIntArray *dilation, int groups,
                                               aclOpExecutor *executor);

// 1971 6HD->FZ_3D with Fp32
const aclTensor *Conv3DBackpropFilterFp322Fp32(const aclTensor *input, const aclTensor *weight,
                                               const aclTensor *outBackprop, const aclIntArray *stride,
                                               const aclIntArray *padding, const aclIntArray *dilation, int groups,
                                               aclOpExecutor *executor);

// 1971 6HD->FZ_3D with Bf16
const aclTensor *Conv3DBackpropFilterBf162Fp32(const aclTensor *input, const aclTensor *weight,
                                               const aclTensor *outBackprop, const aclIntArray *stride,
                                               const aclIntArray *padding, const aclIntArray *dilation, int groups,
                                               aclOpExecutor *executor);

// 1971 6HD->FZ_3D with Hf32
const aclTensor *Conv3DBackpropFilterHf32(const aclTensor *input, const aclTensor *weight, const aclTensor *outBackprop,
                                          const aclIntArray *stride, const aclIntArray *padding,
                                          const aclIntArray *dilation, int groups, aclOpExecutor *executor);

// Conv3dBackpropInput
// 6HD->FZ with Fp16
const aclTensor *Conv3DBackpropInputFp162Fp16(const aclTensor *input, const aclTensor *weight,
                                              const aclTensor *outBackprop, const aclIntArray *stride,
                                              const aclIntArray *padding, const aclIntArray *dilation, int groups,
                                              aclOpExecutor *executor);

// Ascend950: adapt Tensor between 3D and 2D. Other platform will return directly.
const aclTensor *AdapterTensor(const aclTensor *tensor, aclOpExecutor *executor, size_t inputDim, size_t targetDim);

// 1971 6HD->FZ with Fp32
const aclTensor *Conv3DBackpropInputFp322Fp32(const aclTensor *input, const aclTensor *weight,
                                              const aclTensor *outBackprop, const aclIntArray *stride,
                                              const aclIntArray *padding, const aclIntArray *dilation, int groups,
                                              aclOpExecutor *executor);
// 1971 6HD->FZ with Hf32
const aclTensor *Conv3DBackpropInputHf32(const aclTensor *input, const aclTensor *weight, const aclTensor *outBackprop,
                                         const aclIntArray *stride, const aclIntArray *padding,
                                         const aclIntArray *dilation, int groups, aclOpExecutor *executor);
// 1971 6HD->FZ with Bf16
const aclTensor *Conv3DBackpropInputBf162Bf16(const aclTensor *input, const aclTensor *weight,
                                              const aclTensor *outBackprop, const aclIntArray *stride,
                                              const aclIntArray *padding, const aclIntArray *dilation, int groups,
                                              aclOpExecutor *executor);
// Conv3dBackpropFilter
// 1971 6HD->FZ_3D with Bf16/FP16
// 1982 5HD->FZ_3D with Bf16/FP16

const aclTensor *Conv3DBackpropFilter(ConvolutionBackwardInputTensor &inputTensor, ConvolutionBackwardParams &params,
                                      aclOpExecutor *executor, bool use_hf32);
// Conv3dBackpropInput
// 1971 6HD->FZ with Bf16/FP16
// 91 5HD->FZ with Bf16/FP16
const aclTensor *Conv3DBackpropInput(ConvolutionBackwardInputTensor &inputTensor,
                                     ConvolutionBackwardParams &params,
                                     aclOpExecutor *executor, bool use_hf32);
bool IsConv3DBackpropInputV2(const ConvBackpropParams &params);

bool IsConv3DBackpropFilterV2(const ConvBackpropParams &params);

bool IsInputTransdataWhiteListCase(const ConvBackpropParams &params);

bool IsConv2DBackpropInputToCastCase(const ConvBackpropParams &params);

bool IsConv2DBackpropInputTo3DCase(const ConvBackpropParams &params);

bool IsConv2DBpFilterTo3Dcase(const ConvBackpropParams &params);
}  // namespace l0op

#endif  // OP_API_OP_API_COMMON_INC_LEVEL0_OP_CONVOLUTIONBACKWARD_OP_H_

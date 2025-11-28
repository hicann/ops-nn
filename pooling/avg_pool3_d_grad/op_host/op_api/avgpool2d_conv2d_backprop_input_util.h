/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef AVGPOOL2D_CONV2D_BACKPROP_INPUT_UTIL_H_
#define AVGPOOL2D_CONV2D_BACKPROP_INPUT_UTIL_H_
#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

namespace avgpool2d_conv2d_input_util {
struct ConvolutionBackwardInputTensorForAvgPool2d {
    const aclTensor* gradOutput;
    const aclTensor* input;
    const aclTensor* weight;
};

struct ConvolutionBackwardParamsForAvgPool2d {
    const aclIntArray* biasSizes;
    const aclIntArray* stride;
    const aclIntArray* padding;
    const aclIntArray* dilation;
    const bool transposed;
    const aclIntArray* outputPadding;
    const int64_t groups;
    const aclBoolArray* outputMask;
    const int8_t cubeMathType;
};

const aclTensor* CalculateConv2DBackpropInputForAvgPool2d(
    ConvolutionBackwardInputTensorForAvgPool2d& inputTensor, ConvolutionBackwardParamsForAvgPool2d& params, aclOpExecutor* executor);
} // namespace avgpool2d_conv2d_input_util

#ifdef __cplusplus
}
#endif
#endif
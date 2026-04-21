/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "deformable_conv2d_backward_checker.h"

namespace Ops {
namespace NN {
namespace Conv {
static const int64_t MAX_KERNEL_SIZE = 2048;
static const int64_t MAX_MATMUL_K = 65535;
static const int64_t THREE_NUM = 3;
static const int64_t INDEX_ZERO = 0;
static const int64_t INDEX_ONE = 1;
static const int64_t INDEX_TWO = 2;
static const int64_t INDEX_THREE = 3;
static const int32_t INT_MAX_VALUE = 2147483647;
static size_t KERNEL_ARRAY_DIM_SIZE = 2;
static size_t STRIDE_ARRAY_DIM_SIZE = 4;
static size_t PADDING_ARRAY_DIM_SIZE = 4;
static size_t DILATION_ARRAY_DIM_SIZE = 4;
static size_t DIM_FOUR = 4;
static size_t DIM_ONE = 1;

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16,
                                                                        op::DataType::DT_BF16};

bool DeformableConv2dBackwardChecker::CheckNotNull()
{
    OP_CHECK_NULL(inputTensor_.gradOutput, return false);
    OP_CHECK_NULL(inputTensor_.input, return false);
    OP_CHECK_NULL(inputTensor_.weight, return false);
    OP_CHECK_NULL(inputTensor_.offsetOut, return false);
    OP_CHECK_NULL(inputTensor_.offset, return false);

    OP_CHECK_NULL(outputTensor_.gradInput, return false);
    OP_CHECK_NULL(outputTensor_.gradWeight, return false);
    OP_CHECK_NULL(outputTensor_.gradBias, return false);
    OP_CHECK_NULL(outputTensor_.gradOffset, return false);

    OP_CHECK_NULL(params_.kernelSize, return false);
    OP_CHECK_NULL(params_.stride, return false);
    OP_CHECK_NULL(params_.padding, return false);
    OP_CHECK_NULL(params_.dilation, return false);
    return true;
}

bool DeformableConv2dBackwardChecker::CheckDtypeValid()
{
    OP_CHECK_DTYPE_NOT_SUPPORT(inputTensor_.input, DTYPE_SUPPORT_LIST, return false);
    
    //所有输入类型要求跟inputTensor_.input一致
    OP_CHECK_DTYPE_NOT_MATCH(outputTensor_.gradInput, inputTensor_.input->GetDataType(), return false);
    OP_CHECK_DTYPE_NOT_MATCH(outputTensor_.gradWeight, inputTensor_.input->GetDataType(), return false);
    OP_CHECK_DTYPE_NOT_MATCH(outputTensor_.gradOffset, inputTensor_.input->GetDataType(), return false);

    OP_CHECK_DTYPE_NOT_MATCH(inputTensor_.weight, inputTensor_.input->GetDataType(), return false);
    OP_CHECK_DTYPE_NOT_MATCH(inputTensor_.offsetOut, inputTensor_.input->GetDataType(), return false);
    OP_CHECK_DTYPE_NOT_MATCH(inputTensor_.offset, inputTensor_.input->GetDataType(), return false);
    OP_CHECK_DTYPE_NOT_MATCH(inputTensor_.gradOutput, inputTensor_.input->GetDataType(), return false);
    if (outputTensor_.gradBias != nullptr) {
        OP_CHECK_DTYPE_NOT_MATCH(outputTensor_.gradBias, inputTensor_.gradOutput->GetDataType(), return false);
    }
    
    return true;
}


bool DeformableConv2dBackwardChecker::CheckFormat()
{
    OP_CHECK(inputTensor_.gradOutput->GetStorageFormat() == Format::FORMAT_NCHW ||
             inputTensor_.gradOutput->GetStorageFormat() == Format::FORMAT_ND,
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gradOutput Format only support NCHW or ND."), return false);
    OP_CHECK(inputTensor_.input->GetStorageFormat() == Format::FORMAT_NCHW ||
            inputTensor_.input->GetStorageFormat() == Format::FORMAT_ND,
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "input Format only support NCHW or ND."), return false);
   
    OP_CHECK(inputTensor_.weight->GetStorageFormat() == inputTensor_.input->GetStorageFormat(),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format of weight and input should be equal."), return false);
    
    OP_CHECK(inputTensor_.offsetOut->GetStorageFormat() == inputTensor_.input->GetStorageFormat(),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format of offsetOut and input should be equal."), return false);
    
    OP_CHECK(inputTensor_.offset->GetStorageFormat() == inputTensor_.input->GetStorageFormat(),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format of offset and input should be equal."), return false);
    
    OP_CHECK(outputTensor_.gradInput->GetStorageFormat() == inputTensor_.input->GetStorageFormat(),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format of gradInput and input should be equal."), return false);
    
    OP_CHECK(outputTensor_.gradWeight->GetStorageFormat() == inputTensor_.weight->GetStorageFormat(),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format of gradWeight and weight should be equal."), return false);
    
    OP_CHECK(outputTensor_.gradOffset->GetStorageFormat() == inputTensor_.offsetOut->GetStorageFormat(),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format of gradOffset and offsetOut should be equal."), return false);
    
    if (outputTensor_.gradBias != nullptr) {
        OP_CHECK(outputTensor_.gradBias->GetStorageFormat() == Format::FORMAT_ND,
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gradBias Format only support ND."), return false);
    }
    
    return true;
}

bool DeformableConv2dBackwardChecker::CheckAttrs()
{
    OP_CHECK(params_.kernelSize->Size() == KERNEL_ARRAY_DIM_SIZE,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "kernelSize length should be 2."), return false);
    OP_CHECK(params_.stride->Size() == STRIDE_ARRAY_DIM_SIZE,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "stride length should be 4."), return false);
    OP_CHECK(params_.padding->Size() == PADDING_ARRAY_DIM_SIZE,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "padding length should be 4."), return false);
    OP_CHECK(params_.dilation->Size() == DILATION_ARRAY_DIM_SIZE,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "dilation length should be 4."), return false);
    
    int64_t kH = (*params_.kernelSize)[INDEX_ZERO];
    int64_t kW = (*params_.kernelSize)[INDEX_ONE];
    OP_CHECK(kH > 0 && kW > 0, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "kernelSize should greater than 0."), return false);
    //正向一致
    OP_CHECK(kH * kW <= MAX_KERNEL_SIZE, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "kH * kW should not exceed 2048."),
             return false);
    
    OP_CHECK((*params_.stride)[INDEX_ZERO] == 1 && (*params_.stride)[INDEX_ONE] == 1,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "stride[0]、stride[1] should be 1."), return false);
    OP_CHECK((*params_.stride)[INDEX_TWO] > 0 && (*params_.stride)[INDEX_THREE] > 0,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "stride[2]、stride[3] should greater than 0."), return false);
    
    OP_CHECK((*params_.dilation)[INDEX_ZERO] == 1 && (*params_.dilation)[INDEX_ONE] == 1,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "dilation[0]、dilation[1] should be 1."), return false);
    OP_CHECK((*params_.dilation)[INDEX_TWO] > 0 && (*params_.dilation)[INDEX_THREE] > 0,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "dilation[2]、dilation[3] should greater than 0."), return false);
    
    OP_CHECK(params_.groups > 0, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "groups should greater than 0."), return false);
    OP_CHECK(params_.deformableGroups > 0, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "deformableGroups should greater than 0."),
             return false);
    OP_CHECK(params_.modulated == true, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "modulated should be true."), return false);
    
    return true;
}

bool DeformableConv2dBackwardChecker::CheckDimension()
{
    OP_CHECK_WRONG_DIMENSION(inputTensor_.gradOutput, DIM_FOUR, return false);
    OP_CHECK_WRONG_DIMENSION(inputTensor_.input, DIM_FOUR, return false);
    OP_CHECK_WRONG_DIMENSION(inputTensor_.weight, DIM_FOUR, return false);
    OP_CHECK_WRONG_DIMENSION(inputTensor_.offsetOut, DIM_FOUR, return false);
    OP_CHECK_WRONG_DIMENSION(inputTensor_.offset, DIM_FOUR, return false);
    OP_CHECK_WRONG_DIMENSION(outputTensor_.gradInput, DIM_FOUR, return false);
    OP_CHECK_WRONG_DIMENSION(outputTensor_.gradWeight, DIM_FOUR, return false);
    OP_CHECK_WRONG_DIMENSION(outputTensor_.gradOffset, DIM_FOUR, return false);
    
    if (outputTensor_.gradBias != nullptr) {
        OP_CHECK_WRONG_DIMENSION(outputTensor_.gradBias, DIM_ONE, return false);
    }
    
    return true;
}

bool DeformableConv2dBackwardChecker::CheckShape()
{
    int64_t n = inputTensor_.input->GetViewShape()[INDEX_ZERO];
    int64_t inC = inputTensor_.input->GetViewShape()[INDEX_ONE];
    int64_t inH = inputTensor_.input->GetViewShape()[INDEX_TWO];
    int64_t inW = inputTensor_.input->GetViewShape()[INDEX_THREE];
    int64_t inSize = inH * inW;
    int64_t outC = inputTensor_.weight->GetViewShape()[INDEX_ZERO];
    OP_CHECK(inC % params_.deformableGroups == 0,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "inC needs to be divisible by deformable_groups."), return false);
    OP_CHECK(inC % params_.groups == 0 && outC % params_.groups == 0,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Both inC and outC needs to be divisible by groups."), return false);
    OP_CHECK(inSize <= INT_MAX_VALUE,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "inH multiplied by inW should not exceed 2147483647."), return false);

    int64_t kH = (*params_.kernelSize)[INDEX_ZERO];
    int64_t kW = (*params_.kernelSize)[INDEX_ONE];
    int64_t padTop = (*params_.padding)[INDEX_ZERO];
    int64_t padBottom = (*params_.padding)[INDEX_ONE];
    int64_t padLeft = (*params_.padding)[INDEX_TWO];
    int64_t padRight = (*params_.padding)[INDEX_THREE];
    int64_t dilationH = (*params_.dilation)[INDEX_TWO];
    int64_t dilationW = (*params_.dilation)[INDEX_THREE];
    int64_t strideH = (*params_.stride)[INDEX_TWO];
    int64_t strideW = (*params_.stride)[INDEX_THREE];
    int64_t outH = (inH + padTop + padBottom - (kH - 1) * dilationH - 1) / strideH + 1;
    int64_t outW = (inW + padLeft + padRight - (kW - 1) * dilationW - 1) / strideW + 1;

    op::Shape weightExpectShape = {outC, inC / params_.groups, kH, kW};
    op::Shape offsetExpectShape = {n, THREE_NUM * params_.deformableGroups * kH * kW, outH, outW};
    op::Shape outExpectShape = {n, outC, outH, outW};
    op::Shape deformOutExpectShape = {n, inC, outH * kH, outW * kW};
    op::Shape biasExpectShape = {outC};
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(inputTensor_.weight, weightExpectShape, return false);
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(inputTensor_.offset, offsetExpectShape, return false);
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(inputTensor_.gradOutput, outExpectShape, return false);
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(inputTensor_.offsetOut, deformOutExpectShape, return false);
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(outputTensor_.gradInput, inputTensor_.input->GetViewShape(), return false);
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(outputTensor_.gradWeight, inputTensor_.weight->GetViewShape(), return false);
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(outputTensor_.gradOffset, inputTensor_.offset->GetViewShape(), return false);
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(outputTensor_.gradBias, biasExpectShape, return false);
    
    int64_t matmulK = kH * kW * inC / params_.groups;
    OP_CHECK(matmulK <= MAX_MATMUL_K,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,"kH multiplied by kW multiplied by inC divided by groups should not exceed 65535."),return false);

    return true;
}

aclnnStatus DeformableConv2dBackwardChecker::CheckParams()
{   
    OP_CHECK(npuArch_ == NpuArch::DAV_3510,
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "current soc version not support."), return false);

    CHECK_COND(CheckNotNull(), ACLNN_ERR_PARAM_NULLPTR, "CheckNotNull failed!");
    CHECK_COND(CheckDtypeValid(), ACLNN_ERR_PARAM_INVALID, "CheckDtypeValid failed!");
    CHECK_COND(CheckFormat(), ACLNN_ERR_PARAM_INVALID, "CheckFormat failed!");
    CHECK_COND(CheckAttrs(), ACLNN_ERR_PARAM_INVALID, "CheckAttrs failed!");
    CHECK_COND(CheckDimension(), ACLNN_ERR_PARAM_INVALID, "CheckDimension failed!");
    CHECK_COND(CheckShape(), ACLNN_ERR_PARAM_INVALID, "CheckShape failed!");
    
    return ACLNN_SUCCESS;
}

} // namespace Conv
} // namespace NN
} // namespace Ops

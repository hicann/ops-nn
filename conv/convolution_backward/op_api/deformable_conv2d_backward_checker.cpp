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
#include "convolution_backward_checker.h"
#include "log/log.h"
#include "matmul/common/op_host/log_format_util.h"

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
static constexpr const char* ACLNN_DEFORMABLE_NAME = "aclnnDeformableConv2dBackwardGetWorkspaceSize";

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
    OP_CHECK(inputTensor_.gradOutput->GetStorageFormat() == Format::FORMAT_NCHW,
             OP_LOGE_FOR_INVALID_FORMAT(ACLNN_DEFORMABLE_NAME, "gradOutput",
                op::ToString(inputTensor_.gradOutput->GetStorageFormat()).GetString(),"NCHW"), return false);

    OP_CHECK(inputTensor_.input->GetStorageFormat() == Format::FORMAT_NCHW,
            OP_LOGE_FOR_INVALID_FORMAT(ACLNN_DEFORMABLE_NAME, "input",
                op::ToString(inputTensor_.input->GetStorageFormat()).GetString(),"NCHW"), return false);

    OP_CHECK(inputTensor_.weight->GetStorageFormat() == inputTensor_.input->GetStorageFormat(),
            OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(ACLNN_DEFORMABLE_NAME, "weight, input",
                FormatString("%s,%s", op::ToString(inputTensor_.weight->GetStorageFormat()).GetString(),
                op::ToString(inputTensor_.input->GetStorageFormat()).GetString()),
                 "The formats of weight and input must be the same"), return false);

    OP_CHECK(inputTensor_.offsetOut->GetStorageFormat() == inputTensor_.input->GetStorageFormat(),
             OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(ACLNN_DEFORMABLE_NAME, "offsetOut, input",
                FormatString("%s,%s", op::ToString(inputTensor_.offsetOut->GetStorageFormat()).GetString(),
                op::ToString(inputTensor_.input->GetStorageFormat()).GetString()),
                "The formats of offsetOut and input must be the same"), return false);
    
    OP_CHECK(inputTensor_.offset->GetStorageFormat() == inputTensor_.input->GetStorageFormat(),
             OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(ACLNN_DEFORMABLE_NAME, "offset, input",
                FormatString("%s,%s", op::ToString(inputTensor_.offset->GetStorageFormat()).GetString(),
                op::ToString(inputTensor_.input->GetStorageFormat()).GetString()),
                "The formats of offset and input must be the same"), return false);
    
    OP_CHECK(outputTensor_.gradInput->GetStorageFormat() == inputTensor_.input->GetStorageFormat(),
             OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(ACLNN_DEFORMABLE_NAME, "gradInput, input",
               FormatString("%s,%s", op::ToString(outputTensor_.gradInput->GetStorageFormat()).GetString(),
                op::ToString(inputTensor_.input->GetStorageFormat()).GetString()),
                    "The formats of gradInput and input must be the same"), return false);
    
    OP_CHECK(outputTensor_.gradWeight->GetStorageFormat() == inputTensor_.weight->GetStorageFormat(),
             OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(ACLNN_DEFORMABLE_NAME, "gradWeight, weight",
              FormatString("%s,%s", op::ToString(outputTensor_.gradWeight->GetStorageFormat()).GetString(),
                op::ToString(inputTensor_.weight->GetStorageFormat()).GetString()),
                "The formats of gradWeight and weight must be the same"), return false);
    
    OP_CHECK(outputTensor_.gradOffset->GetStorageFormat() == inputTensor_.offsetOut->GetStorageFormat(),
             OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(ACLNN_DEFORMABLE_NAME, "gradOffset, offsetOut",
              FormatString("%s,%s", op::ToString(outputTensor_.gradOffset->GetStorageFormat()).GetString(),
                op::ToString(inputTensor_.offsetOut->GetStorageFormat()).GetString()),
                "The formats of gradOffset and offsetOut must be the same"), return false);
    
    if (outputTensor_.gradBias != nullptr) {
        OP_CHECK(outputTensor_.gradBias->GetStorageFormat() == Format::FORMAT_ND,
            OP_LOGE_FOR_INVALID_FORMAT(ACLNN_DEFORMABLE_NAME, "gradBias",
            op::ToString(outputTensor_.gradBias->GetStorageFormat()).GetString(), "ND"), return false);
    }
    
    return true;
}

bool DeformableConv2dBackwardChecker::CheckAttrs()
{
    OP_CHECK(params_.kernelSize->Size() == KERNEL_ARRAY_DIM_SIZE,             
             OP_LOGE_FOR_INVALID_SHAPESIZE(ACLNN_DEFORMABLE_NAME, "kernelSize", std::to_string(params_.kernelSize->Size()).c_str()
             , std::to_string(KERNEL_ARRAY_DIM_SIZE).c_str()), return false);
    OP_CHECK(params_.stride->Size() == STRIDE_ARRAY_DIM_SIZE,
             OP_LOGE_WITH_INVALID_ATTR_SIZE(ACLNN_DEFORMABLE_NAME, "stride", std::to_string(params_.stride->Size()).c_str(), 
                std::to_string(STRIDE_ARRAY_DIM_SIZE).c_str()), return false);
    OP_CHECK(params_.padding->Size() == PADDING_ARRAY_DIM_SIZE,
             OP_LOGE_WITH_INVALID_ATTR_SIZE(ACLNN_DEFORMABLE_NAME, "padding", 
                std::to_string(params_.padding->Size()).c_str(), std::to_string(PADDING_ARRAY_DIM_SIZE).c_str()), return false);
    OP_CHECK(params_.dilation->Size() == DILATION_ARRAY_DIM_SIZE,
             OP_LOGE_WITH_INVALID_ATTR_SIZE(ACLNN_DEFORMABLE_NAME, "dilation", 
                std::to_string(params_.dilation->Size()).c_str(), std::to_string(DILATION_ARRAY_DIM_SIZE).c_str()), return false);
    
    int64_t kH = (*params_.kernelSize)[INDEX_ZERO];
    int64_t kW = (*params_.kernelSize)[INDEX_ONE];
    OP_CHECK(kH > 0 && kW > 0,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(ACLNN_DEFORMABLE_NAME, "kH,kW", 
        (std::to_string(kH) + "," + std::to_string(kW)).c_str(), "all axis of [kH, kW] must be greater than 0"), return false);
    OP_CHECK(kH * kW <= MAX_KERNEL_SIZE,
      OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(ACLNN_DEFORMABLE_NAME, "kH,kW", 
      (std::to_string(kH) + "," + std::to_string(kW)).c_str(), "all axis of [kH, kW] must be less than 2048"), return false);
    // stride >= 1
    OP_CHECK((*params_.stride)[INDEX_ZERO] == 1 && (*params_.stride)[INDEX_ONE] == 1 && CheckParamsValue(params_.stride, false),
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(ACLNN_DEFORMABLE_NAME,
        "stride[0],stride[1],stride[2],stride[3]", AclarrayToString(params_.stride).c_str(),
        "stride[0],stride[1] must be 1, stride[2], stride[3] must be greater than or equal to 1"
        ), return false);
    // padding >= 0
    OP_CHECK(CheckParamsValue(params_.padding, true),
       OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ACLNN_DEFORMABLE_NAME, "padding", 
       AclarrayToString(params_.padding).c_str(), "the value of padding must be greater than or equal to 0"), return false);
    // dilation >= 1
    OP_CHECK((*params_.dilation)[INDEX_ZERO] == 1 && (*params_.dilation)[INDEX_ONE] == 1 && CheckParamsValue(params_.dilation, false),
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(ACLNN_DEFORMABLE_NAME,
                "dilation", AclarrayToString(params_.dilation).c_str(),
                "dilation[0],dilation[1] must be 1, dilation[2],dilation[3] must be greater than or equal to 1"
            ), return false);
    OP_CHECK(params_.groups > 0,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ACLNN_DEFORMABLE_NAME, "groups", 
             std::to_string(params_.groups).c_str(), "the value of groups must be greater than 0"
             ), return false);
    OP_CHECK(params_.deformableGroups > 0,
             OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ACLNN_DEFORMABLE_NAME, "deformableGroups", 
             std::to_string(params_.deformableGroups).c_str(), "the value of deformableGroups must be greater than 0"
             ), return false);
    OP_CHECK(params_.modulated == true,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ACLNN_DEFORMABLE_NAME, "modulated", 
             std::to_string(params_.modulated).c_str(), "the value of modulated must be true"
             ), return false);
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
             OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(ACLNN_DEFORMABLE_NAME, "inC,deformableGroups", 
              (std::to_string(inC) +  "," + std::to_string(params_.deformableGroups)).c_str(), 
             "inC must be exactly divisible by deformableGroups"), return false);
    OP_CHECK(inC % params_.groups == 0 && outC % params_.groups == 0,
             OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(ACLNN_DEFORMABLE_NAME, "inC,outC,groups", 
             (std::to_string(inC) + "," + std::to_string(outC) + "," + std::to_string(params_.groups)).c_str(), 
             "All axis of [inC, outC] must be divisible by groups"), return false);
    OP_CHECK(inSize <= INT_MAX_VALUE, OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(ACLNN_DEFORMABLE_NAME, "inH,inW", 
             (std::to_string(inC) + "," + std::to_string(outC)).c_str(), 
             "inH * inW must be not greater than 2147483647"), return false);

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
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(ACLNN_DEFORMABLE_NAME, "(kH * kW * inC) / params_.groups", 
             (std::to_string(matmulK)).c_str(), 
             "kH * kW * inC / params_.groups should not greater than 65535"), return false);
    return true;
}

aclnnStatus DeformableConv2dBackwardChecker::CheckParams()
{   
    OP_CHECK(npuArch_ == NpuArch::DAV_3510,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ACLNN_DEFORMABLE_NAME, "npuarch",
                    std::to_string(static_cast<uint32_t>(npuArch_)).c_str(), "npuarch must not be 3510"),
            return false);

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

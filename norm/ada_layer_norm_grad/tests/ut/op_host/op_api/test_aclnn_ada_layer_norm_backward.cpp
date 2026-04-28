/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <vector>
#include "gtest/gtest.h"

#include "../../../op_host/op_api/aclnn_ada_layer_norm_backward.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace std;
using namespace op;

class l2_ada_layer_norm_backward_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "ada_layer_norm_backward_test SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "ada_layer_norm_backward_test TearDown" << std::endl;
    }
};

// nchw, float32
TEST_F(l2_ada_layer_norm_backward_test, ascend910B3_aclnnAdaLayerNormBackward_float32_nd_float32_nd)
{
    vector<int64_t> input_shape = {9, 2, 2, 3};
    vector<int64_t> norm_shape = {3};
    vector<int64_t> scale_shape = {9, 2, 3};
    vector<int64_t> shift_shape = {9, 2, 3};
    vector<int64_t> mean_shape = {9, 2, 2, 1};
    aclDataType dtype = ACL_FLOAT;
    aclFormat in_format = ACL_FORMAT_ND;
    auto gradOutDesc = TensorDesc(input_shape, dtype, in_format).ValueRange(-2, 2);
    auto inputDesc = gradOutDesc;
    auto normalizedShape = IntArrayDesc(norm_shape);
    auto meanDesc = TensorDesc(mean_shape, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto rstdDesc = TensorDesc(mean_shape, dtype, ACL_FORMAT_ND).ValueRange(1, 2);
    auto scaleDesc = TensorDesc(scale_shape, dtype, in_format).ValueRange(-2, 2);
    auto shiftDesc = TensorDesc(shift_shape, dtype, in_format).ValueRange(-2, 2);
    auto weightDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto biasDesc = weightDesc;

    auto gradInputOutDesc = TensorDesc(input_shape, dtype, in_format).Precision(0.01, 0.01);
    auto gradScaleOutDesc = TensorDesc(scale_shape, dtype, in_format).Precision(0.01, 0.01);
    auto gradShiftOutDesc = TensorDesc(shift_shape, dtype, in_format).Precision(0.01, 0.01);
    auto gradWeightOutDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).Precision(0.01, 0.01);
    auto gradBiasOutDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).Precision(0.01, 0.01);

    auto ut = OP_API_UT(
        aclnnAdaLayerNormBackward,
        INPUT(gradOutDesc, inputDesc, normalizedShape, rstdDesc, meanDesc, scaleDesc, shiftDesc, weightDesc, biasDesc),
        OUTPUT(gradInputOutDesc, gradScaleOutDesc, gradShiftOutDesc, gradWeightOutDesc, gradBiasOutDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// float64: kernel not support now
TEST_F(l2_ada_layer_norm_backward_test, ascend910B2_aclnnAdaLayerNormBackward_double_nd_double_nd)
{
    vector<int64_t> input_shape = {2, 6, 5, 6};
    vector<int64_t> norm_shape = {6};
    vector<int64_t> scale_shape = {2, 6, 6};
    vector<int64_t> shift_shape = {2, 6, 6};
    vector<int64_t> mean_shape = {2, 6, 5, 1};
    aclDataType dtype = ACL_DOUBLE;
    aclFormat in_format = ACL_FORMAT_NHWC;
    auto gradOutDesc = TensorDesc(input_shape, dtype, in_format).ValueRange(-2, 2);
    auto inputDesc = gradOutDesc;
    auto normalizedShape = IntArrayDesc(norm_shape);
    auto meanDesc = TensorDesc(mean_shape, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto rstdDesc = TensorDesc(mean_shape, dtype, ACL_FORMAT_ND).ValueRange(1, 2);
    auto scaleDesc = TensorDesc(scale_shape, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto shiftDesc = TensorDesc(shift_shape, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto weightDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto biasDesc = weightDesc;


    auto gradInputOutDesc = TensorDesc(input_shape, dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto gradScaleOutDesc = TensorDesc(scale_shape, dtype, ACL_FORMAT_ND).Precision(0.01, 0.01);
    auto gradShiftOutDesc = TensorDesc(shift_shape, dtype, ACL_FORMAT_ND).Precision(0.01, 0.01);
    auto gradWeightOutDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto gradBiasOutDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(
        aclnnAdaLayerNormBackward,
        INPUT(gradOutDesc, inputDesc, normalizedShape, rstdDesc, meanDesc, scaleDesc, shiftDesc, weightDesc, biasDesc),
        OUTPUT(gradInputOutDesc, gradScaleOutDesc, gradShiftOutDesc, gradWeightOutDesc, gradBiasOutDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// error input dtype
TEST_F(l2_ada_layer_norm_backward_test, ascend910B2_aclnnAdaLayerNormBackward_error_input_dtype)
{
    vector<int64_t> input_shape = {2, 6, 5, 6};
    vector<int64_t> norm_shape = {6};
    vector<int64_t> scale_shape = {2, 6, 6};
    vector<int64_t> shift_shape = {2, 6, 6};
    vector<int64_t> mean_shape = {2, 6, 5, 1};
    aclDataType dtype = ACL_FLOAT;
    aclFormat in_format = ACL_FORMAT_NHWC;
    auto gradOutDesc = TensorDesc(input_shape, ACL_UINT64, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto inputDesc = gradOutDesc;
    auto normalizedShape = IntArrayDesc(norm_shape);
    auto meanDesc = TensorDesc(mean_shape, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto rstdDesc = TensorDesc(mean_shape, dtype, ACL_FORMAT_ND).ValueRange(1, 2);
    auto scaleDesc = TensorDesc(scale_shape, dtype, ACL_FORMAT_ND).ValueRange(1, 2);
    auto shiftDesc = TensorDesc(shift_shape, dtype, ACL_FORMAT_ND).ValueRange(1, 2);
    auto weightDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto biasDesc = weightDesc;


    auto gradInputOutDesc = TensorDesc(input_shape, dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto gradScaleOutDesc = TensorDesc(scale_shape, dtype, ACL_FORMAT_ND).Precision(0.01, 0.01);
    auto gradShiftOutDesc = TensorDesc(shift_shape, dtype, ACL_FORMAT_ND).Precision(0.01, 0.01);
    auto gradWeightOutDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto gradBiasOutDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(
        aclnnAdaLayerNormBackward,
        INPUT(gradOutDesc, inputDesc, normalizedShape, rstdDesc, meanDesc, scaleDesc, shiftDesc, weightDesc, biasDesc),
        OUTPUT(gradInputOutDesc, gradScaleOutDesc, gradShiftOutDesc, gradWeightOutDesc, gradBiasOutDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// error out dtype
TEST_F(l2_ada_layer_norm_backward_test, ascend910B2_aclnnAdaLayerNormBackward_error_out_dtype)
{
    vector<int64_t> input_shape = {2, 6, 5, 6};
    vector<int64_t> norm_shape = {6};
    vector<int64_t> scale_shape = {2, 6, 6};
    vector<int64_t> shift_shape = {2, 6, 6};
    vector<int64_t> mean_shape = {2, 6, 5, 1};
    aclDataType dtype = ACL_FLOAT;
    aclFormat in_format = ACL_FORMAT_NHWC;
    auto gradOutDesc = TensorDesc(input_shape, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto inputDesc = gradOutDesc;
    auto normalizedShape = IntArrayDesc(norm_shape);
    auto meanDesc = TensorDesc(mean_shape, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto rstdDesc = TensorDesc(mean_shape, dtype, ACL_FORMAT_ND).ValueRange(1, 2);
    auto scaleDesc = TensorDesc(scale_shape, dtype, ACL_FORMAT_ND).ValueRange(1, 2);
    auto shiftDesc = TensorDesc(shift_shape, dtype, ACL_FORMAT_ND).ValueRange(1, 2);
    auto weightDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto biasDesc = weightDesc;


    auto gradInputOutDesc = TensorDesc(input_shape, ACL_UINT64, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto gradScaleOutDesc = TensorDesc(scale_shape, dtype, ACL_FORMAT_ND).Precision(0.01, 0.01);
    auto gradShiftOutDesc = TensorDesc(shift_shape, dtype, ACL_FORMAT_ND).Precision(0.01, 0.01);
    auto gradWeightOutDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto gradBiasOutDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(
        aclnnAdaLayerNormBackward,
        INPUT(gradOutDesc, inputDesc, normalizedShape, rstdDesc, meanDesc, scaleDesc, shiftDesc, weightDesc, biasDesc),
        OUTPUT(gradInputOutDesc, gradScaleOutDesc, gradShiftOutDesc, gradWeightOutDesc, gradBiasOutDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// error input len
TEST_F(l2_ada_layer_norm_backward_test, ascend910B2_aclnnAdaLayerNormBackward_error_input_len)
{
    vector<int64_t> input_shape = {1, 2, 1, 2, 1, 2, 1, 2, 2, 2, 2};
    vector<int64_t> norm_shape = {2, 1, 2, 1, 2, 1, 2, 2, 2, 2};
    vector<int64_t> scale_shape = {2, 6, 6};
    vector<int64_t> shift_shape = {2, 6, 6};
    vector<int64_t> mean_shape = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    aclDataType dtype = ACL_FLOAT;
    aclFormat in_format = ACL_FORMAT_ND;
    auto gradOutDesc = TensorDesc(input_shape, dtype, in_format).ValueRange(-2, 2);
    auto inputDesc = gradOutDesc;
    auto normalizedShape = IntArrayDesc(norm_shape);
    auto meanDesc = TensorDesc(mean_shape, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto rstdDesc = TensorDesc(mean_shape, dtype, ACL_FORMAT_ND).ValueRange(1, 2);
    auto scaleDesc = TensorDesc(scale_shape, dtype, ACL_FORMAT_ND).ValueRange(1, 2);
    auto shiftDesc = TensorDesc(shift_shape, dtype, ACL_FORMAT_ND).ValueRange(1, 2);
    auto weightDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto biasDesc = weightDesc;


    auto gradInputOutDesc = TensorDesc(input_shape, dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto gradScaleOutDesc = TensorDesc(scale_shape, dtype, ACL_FORMAT_ND).Precision(0.01, 0.01);
    auto gradShiftOutDesc = TensorDesc(shift_shape, dtype, ACL_FORMAT_ND).Precision(0.01, 0.01);
    auto gradWeightOutDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto gradBiasOutDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(
        aclnnAdaLayerNormBackward,
        INPUT(gradOutDesc, inputDesc, normalizedShape, rstdDesc, meanDesc, scaleDesc, shiftDesc, weightDesc, biasDesc),
        OUTPUT(gradInputOutDesc, gradScaleOutDesc, gradShiftOutDesc, gradWeightOutDesc, gradBiasOutDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// error normalizedShape len
TEST_F(l2_ada_layer_norm_backward_test, ascend910B2_aclnnAdaLayerNormBackward_error_normalizedShape_len)
{
    vector<int64_t> input_shape = {2, 6, 5, 6};
    vector<int64_t> norm_shape = {};
    vector<int64_t> scale_shape = {2, 6, 6};
    vector<int64_t> shift_shape = {2, 6, 6};
    vector<int64_t> mean_shape = {2, 6, 5, 1};
    aclDataType dtype = ACL_FLOAT;
    aclFormat in_format = ACL_FORMAT_ND;
    auto gradOutDesc = TensorDesc(input_shape, dtype, in_format).ValueRange(-2, 2);
    auto inputDesc = gradOutDesc;
    auto normalizedShape = IntArrayDesc(norm_shape);
    auto meanDesc = TensorDesc(mean_shape, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto rstdDesc = TensorDesc(mean_shape, dtype, ACL_FORMAT_ND).ValueRange(1, 2);
    auto scaleDesc = TensorDesc(scale_shape, dtype, ACL_FORMAT_ND).ValueRange(1, 2);
    auto shiftDesc = TensorDesc(shift_shape, dtype, ACL_FORMAT_ND).ValueRange(1, 2);
    auto weightDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto biasDesc = weightDesc;


    auto gradInputOutDesc = TensorDesc(input_shape, dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto gradScaleOutDesc = TensorDesc(scale_shape, dtype, ACL_FORMAT_ND).Precision(0.01, 0.01);
    auto gradShiftOutDesc = TensorDesc(shift_shape, dtype, ACL_FORMAT_ND).Precision(0.01, 0.01);
    auto gradWeightOutDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto gradBiasOutDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(
        aclnnAdaLayerNormBackward,
        INPUT(gradOutDesc, inputDesc, normalizedShape, rstdDesc, meanDesc, scaleDesc, shiftDesc, weightDesc, biasDesc),
        OUTPUT(gradInputOutDesc, gradScaleOutDesc, gradShiftOutDesc, gradWeightOutDesc, gradBiasOutDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// error weight len
TEST_F(l2_ada_layer_norm_backward_test, ascend910B2_aclnnAdaLayerNormBackward_error_weight_len)
{
    vector<int64_t> input_shape = {2, 6, 5, 6};
    vector<int64_t> norm_shape = {6};
    vector<int64_t> scale_shape = {2, 6, 6};
    vector<int64_t> shift_shape = {2, 6, 6};
    vector<int64_t> mean_shape = {2, 6, 5, 1};
    aclDataType dtype = ACL_FLOAT;
    aclFormat in_format = ACL_FORMAT_ND;
    auto gradOutDesc = TensorDesc(input_shape, dtype, in_format).ValueRange(-2, 2);
    auto inputDesc = gradOutDesc;
    auto normalizedShape = IntArrayDesc(norm_shape);
    auto meanDesc = TensorDesc(mean_shape, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto rstdDesc = TensorDesc(mean_shape, dtype, ACL_FORMAT_ND).ValueRange(1, 2);
    auto scaleDesc = TensorDesc(scale_shape, dtype, ACL_FORMAT_ND).ValueRange(1, 2);
    auto shiftDesc = TensorDesc(shift_shape, dtype, ACL_FORMAT_ND).ValueRange(1, 2);
    auto weightDesc = TensorDesc({1, 1, 1, 1, 1, 2}, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto biasDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);


    auto gradInputOutDesc = TensorDesc(input_shape, dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto gradScaleOutDesc = TensorDesc(scale_shape, dtype, ACL_FORMAT_ND).Precision(0.01, 0.01);
    auto gradShiftOutDesc = TensorDesc(shift_shape, dtype, ACL_FORMAT_ND).Precision(0.01, 0.01);
    auto gradWeightOutDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto gradBiasOutDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(
        aclnnAdaLayerNormBackward,
        INPUT(gradOutDesc, inputDesc, normalizedShape, rstdDesc, meanDesc, scaleDesc, shiftDesc, weightDesc, biasDesc),
        OUTPUT(gradInputOutDesc, gradScaleOutDesc, gradShiftOutDesc, gradWeightOutDesc, gradBiasOutDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// error bias len
TEST_F(l2_ada_layer_norm_backward_test, ascend910B2_aclnnAdaLayerNormBackward_error_bias_len)
{
    vector<int64_t> input_shape = {2, 6, 5, 6};
    vector<int64_t> norm_shape = {6};
    vector<int64_t> scale_shape = {2, 6, 6};
    vector<int64_t> shift_shape = {2, 6, 6};
    vector<int64_t> mean_shape = {2, 6, 5, 1};
    aclDataType dtype = ACL_FLOAT;
    aclFormat in_format = ACL_FORMAT_ND;
    auto gradOutDesc = TensorDesc(input_shape, dtype, in_format).ValueRange(-2, 2);
    auto inputDesc = gradOutDesc;
    auto normalizedShape = IntArrayDesc(norm_shape);
    auto meanDesc = TensorDesc(mean_shape, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto rstdDesc = TensorDesc(mean_shape, dtype, ACL_FORMAT_ND).ValueRange(1, 2);
    auto scaleDesc = TensorDesc(scale_shape, dtype, ACL_FORMAT_ND).ValueRange(1, 2);
    auto shiftDesc = TensorDesc(shift_shape, dtype, ACL_FORMAT_ND).ValueRange(1, 2);
    auto weightDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto biasDesc = TensorDesc({1, 1, 1, 1, 1, 2}, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);


    auto gradInputOutDesc = TensorDesc(input_shape, dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto gradScaleOutDesc = TensorDesc(scale_shape, dtype, ACL_FORMAT_ND).Precision(0.01, 0.01);
    auto gradShiftOutDesc = TensorDesc(shift_shape, dtype, ACL_FORMAT_ND).Precision(0.01, 0.01);
    auto gradWeightOutDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto gradBiasOutDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(
        aclnnAdaLayerNormBackward,
        INPUT(gradOutDesc, inputDesc, normalizedShape, rstdDesc, meanDesc, scaleDesc, shiftDesc, weightDesc, biasDesc),
        OUTPUT(gradInputOutDesc, gradScaleOutDesc, gradShiftOutDesc, gradWeightOutDesc, gradBiasOutDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// error gradInputOut len
TEST_F(l2_ada_layer_norm_backward_test, ascend910B2_aclnnAdaLayerNormBackward_error_grad_input_out_len)
{
    vector<int64_t> input_shape = {2, 6, 5, 6};
    vector<int64_t> norm_shape = {6};
    vector<int64_t> scale_shape = {2, 6, 6};
    vector<int64_t> shift_shape = {2, 6, 6};
    vector<int64_t> mean_shape = {2, 6, 5, 1};
    aclDataType dtype = ACL_FLOAT;
    aclFormat in_format = ACL_FORMAT_ND;
    auto gradOutDesc = TensorDesc(input_shape, dtype, in_format).ValueRange(-2, 2);
    auto inputDesc = gradOutDesc;
    auto normalizedShape = IntArrayDesc(norm_shape);
    auto meanDesc = TensorDesc(mean_shape, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto rstdDesc = TensorDesc(mean_shape, dtype, ACL_FORMAT_ND).ValueRange(1, 2);
    auto scaleDesc = TensorDesc(scale_shape, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto shiftDesc = TensorDesc(shift_shape, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto weightDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto biasDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);


    auto gradInputOutDesc = TensorDesc({1, 1, 1, 1, 1, 2}, dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto gradScaleOutDesc = TensorDesc(scale_shape, dtype, ACL_FORMAT_ND).Precision(0.01, 0.01);
    auto gradShiftOutDesc = TensorDesc(shift_shape, dtype, ACL_FORMAT_ND).Precision(0.01, 0.01);
    auto gradWeightOutDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto gradBiasOutDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(
        aclnnAdaLayerNormBackward,
        INPUT(gradOutDesc, inputDesc, normalizedShape, rstdDesc, meanDesc, scaleDesc, shiftDesc, weightDesc, biasDesc),
        OUTPUT(gradInputOutDesc, gradScaleOutDesc, gradShiftOutDesc, gradWeightOutDesc, gradBiasOutDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// error scale len
TEST_F(l2_ada_layer_norm_backward_test, ascend910B2_aclnnAdaLayerNormBackward_error_scale_len)
{
    auto gradOutDesc = TensorDesc({2, 6, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto inputDesc = gradOutDesc;
    auto normalizedShape = IntArrayDesc(vector<int64_t>{6});
    auto meanDesc = TensorDesc({2, 6, 5, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto rstdDesc = TensorDesc({2, 6, 5, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto scaleDesc = TensorDesc({2, 1, 6}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto shiftDesc = TensorDesc({2, 6, 6}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto weightDesc = TensorDesc({6}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto biasDesc = weightDesc;


    auto gradInputOutDesc = TensorDesc({2, 6, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto gradScaleOutDesc = TensorDesc({2, 6, 6}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.01, 0.01);
    auto gradShiftOutDesc = TensorDesc({2, 6, 6}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.01, 0.01);
    auto gradWeightOutDesc = TensorDesc({6}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto gradBiasOutDesc = TensorDesc({6}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(
        aclnnAdaLayerNormBackward,
        INPUT(gradOutDesc, inputDesc, normalizedShape, rstdDesc, meanDesc, scaleDesc, shiftDesc, weightDesc, biasDesc),
        OUTPUT(gradInputOutDesc, gradScaleOutDesc, gradShiftOutDesc, gradWeightOutDesc, gradBiasOutDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// error shift len
TEST_F(l2_ada_layer_norm_backward_test, ascend910B2_aclnnAdaLayerNormBackward_error_shift_len)
{
    auto gradOutDesc = TensorDesc({2, 6, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto inputDesc = gradOutDesc;
    auto normalizedShape = IntArrayDesc(vector<int64_t>{6});
    auto meanDesc = TensorDesc({2, 6, 5, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto rstdDesc = TensorDesc({2, 6, 5, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto scaleDesc = TensorDesc({2, 6, 6}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto shiftDesc = TensorDesc({2, 1, 6}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto weightDesc = TensorDesc({6}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto biasDesc = weightDesc;


    auto gradInputOutDesc = TensorDesc({2, 6, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto gradScaleOutDesc = TensorDesc({2, 6, 6}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.01, 0.01);
    auto gradShiftOutDesc = TensorDesc({2, 6, 6}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.01, 0.01);
    auto gradWeightOutDesc = TensorDesc({6}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto gradBiasOutDesc = TensorDesc({6}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(
        aclnnAdaLayerNormBackward,
        INPUT(gradOutDesc, inputDesc, normalizedShape, rstdDesc, meanDesc, scaleDesc, shiftDesc, weightDesc, biasDesc),
        OUTPUT(gradInputOutDesc, gradScaleOutDesc, gradShiftOutDesc, gradWeightOutDesc, gradBiasOutDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// error gradScaleOut len
TEST_F(l2_ada_layer_norm_backward_test, ascend910B2_aclnnAdaLayerNormBackward_error_grad_scale_out_len)
{
    auto gradOutDesc = TensorDesc({2, 6, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto inputDesc = gradOutDesc;
    auto normalizedShape = IntArrayDesc(vector<int64_t>{6});
    auto meanDesc = TensorDesc({2, 6, 5, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto rstdDesc = TensorDesc({2, 6, 5, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto scaleDesc = TensorDesc({2, 6, 6}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto shiftDesc = TensorDesc({2, 6, 6}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto weightDesc = TensorDesc({6}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto biasDesc = weightDesc;


    auto gradInputOutDesc = TensorDesc({2, 6, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto gradScaleOutDesc = TensorDesc({2, 1, 6}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.01, 0.01);
    auto gradShiftOutDesc = TensorDesc({2, 6, 6}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.01, 0.01);
    auto gradWeightOutDesc = TensorDesc({6}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto gradBiasOutDesc = TensorDesc({6}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(
        aclnnAdaLayerNormBackward,
        INPUT(gradOutDesc, inputDesc, normalizedShape, rstdDesc, meanDesc, scaleDesc, shiftDesc, weightDesc, biasDesc),
        OUTPUT(gradInputOutDesc, gradScaleOutDesc, gradShiftOutDesc, gradWeightOutDesc, gradBiasOutDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// error gradShiftOut len
TEST_F(l2_ada_layer_norm_backward_test, ascend910B2_aclnnAdaLayerNormBackward_error_grad_shift_out_len)
{
    auto gradOutDesc = TensorDesc({2, 6, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto inputDesc = gradOutDesc;
    auto normalizedShape = IntArrayDesc(vector<int64_t>{6});
    auto meanDesc = TensorDesc({2, 6, 5, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto rstdDesc = TensorDesc({2, 6, 5, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto scaleDesc = TensorDesc({2, 6, 6}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto shiftDesc = TensorDesc({2, 6, 6}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto weightDesc = TensorDesc({6}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto biasDesc = weightDesc;


    auto gradInputOutDesc = TensorDesc({2, 6, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto gradScaleOutDesc = TensorDesc({2, 6, 6}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.01, 0.01);
    auto gradShiftOutDesc = TensorDesc({2, 1, 6}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.01, 0.01);
    auto gradWeightOutDesc = TensorDesc({6}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto gradBiasOutDesc = TensorDesc({6}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(
        aclnnAdaLayerNormBackward,
        INPUT(gradOutDesc, inputDesc, normalizedShape, rstdDesc, meanDesc, scaleDesc, shiftDesc, weightDesc, biasDesc),
        OUTPUT(gradInputOutDesc, gradScaleOutDesc, gradShiftOutDesc, gradWeightOutDesc, gradBiasOutDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// error input and normalizedShape
TEST_F(l2_ada_layer_norm_backward_test, ascend910B2_aclnnAdaLayerNormBackward_diff_input_and_normalizedShape)
{
    auto gradOutDesc = TensorDesc({2, 6, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto inputDesc = gradOutDesc;
    auto normalizedShape = IntArrayDesc(vector<int64_t>{7, 8});
    auto meanDesc = TensorDesc({2, 6, 5, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto rstdDesc = TensorDesc({2, 6, 5, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto scaleDesc = TensorDesc({2, 6, 6}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto shiftDesc = TensorDesc({2, 6, 6}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto weightDesc = TensorDesc({7, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto biasDesc = weightDesc;


    auto gradInputOutDesc = TensorDesc({2, 6, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto gradScaleOutDesc = TensorDesc({2, 6, 6}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.01, 0.01);
    auto gradShiftOutDesc = TensorDesc({2, 6, 6}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.01, 0.01);
    auto gradWeightOutDesc = TensorDesc({7, 8}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto gradBiasOutDesc = TensorDesc({7, 8}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(
        aclnnAdaLayerNormBackward,
        INPUT(gradOutDesc, inputDesc, normalizedShape, rstdDesc, meanDesc, scaleDesc, shiftDesc, weightDesc, biasDesc),
        OUTPUT(gradInputOutDesc, gradScaleOutDesc, gradShiftOutDesc, gradWeightOutDesc, gradBiasOutDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// nullptr
TEST_F(l2_ada_layer_norm_backward_test, ascend910B2_aclnnAdaLayerNormBackward_nullptr)
{
    auto gradOutDesc = TensorDesc({7, 8, 9}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto inputDesc = TensorDesc({7, 8, 9}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto normalizedShape = IntArrayDesc(vector<int64_t>{9});
    auto meanDesc = TensorDesc({7, 8, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto rstdDesc = TensorDesc({7, 8, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto scaleDesc = TensorDesc({7, 9}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto shiftDesc = TensorDesc({7, 9}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto weightDesc = TensorDesc({9}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto biasDesc = weightDesc;


    auto gradInputOutDesc = TensorDesc({7, 8, 9}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto gradScaleOutDesc = TensorDesc({7, 9}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.01, 0.01);
    auto gradShiftOutDesc = TensorDesc({7, 9}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.01, 0.01);
    auto gradWeightOutDesc = TensorDesc({9}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto gradBiasOutDesc = TensorDesc({9}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);

    uint64_t workspaceSize = 0;

    // gradOut nullptr
    auto ut1 = OP_API_UT(
        aclnnAdaLayerNormBackward,
        INPUT((aclTensor*)nullptr, inputDesc, normalizedShape, rstdDesc, meanDesc, scaleDesc, shiftDesc, weightDesc, biasDesc),
        OUTPUT(gradInputOutDesc, gradScaleOutDesc, gradShiftOutDesc, gradWeightOutDesc, gradBiasOutDesc));
    // SAMPLE: only test GetWorkspaceSize
    aclnnStatus aclRet = ut1.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    // input nullptr
    auto ut2 = OP_API_UT(
        aclnnAdaLayerNormBackward,
        INPUT(gradOutDesc, (aclTensor*)nullptr, normalizedShape, rstdDesc, meanDesc, scaleDesc, shiftDesc, weightDesc, biasDesc),
        OUTPUT(gradInputOutDesc, gradScaleOutDesc, gradShiftOutDesc, gradWeightOutDesc, gradBiasOutDesc));
    // SAMPLE: only test GetWorkspaceSize
    aclRet = ut2.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    // normalizedShape nullptr
    auto ut3 = OP_API_UT(
        aclnnAdaLayerNormBackward,
        INPUT(gradOutDesc, inputDesc, (aclIntArray*)nullptr, rstdDesc, meanDesc, scaleDesc, shiftDesc, weightDesc, biasDesc),
        OUTPUT(gradInputOutDesc, gradScaleOutDesc, gradShiftOutDesc, gradWeightOutDesc, gradBiasOutDesc));
    // SAMPLE: only test GetWorkspaceSize
    aclRet = ut3.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    // rstdDesc nullptr
    auto ut4 = OP_API_UT(
        aclnnAdaLayerNormBackward,
        INPUT(gradOutDesc, inputDesc, normalizedShape, (aclTensor*)nullptr, meanDesc, scaleDesc, shiftDesc, weightDesc, biasDesc),
        OUTPUT(gradInputOutDesc, gradScaleOutDesc, gradShiftOutDesc, gradWeightOutDesc, gradBiasOutDesc));
    // SAMPLE: only test GetWorkspaceSize
    aclRet = ut4.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    // meanDesc nullptr
    auto ut5 = OP_API_UT(
        aclnnAdaLayerNormBackward,
        INPUT(gradOutDesc, inputDesc, normalizedShape, rstdDesc, (aclTensor*)nullptr, scaleDesc, shiftDesc, weightDesc, biasDesc),
        OUTPUT(gradInputOutDesc, gradScaleOutDesc, gradShiftOutDesc, gradWeightOutDesc, gradBiasOutDesc));
    // SAMPLE: only test GetWorkspaceSize
    aclRet = ut5.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    // shiftDesc nullptr
    auto ut6 = OP_API_UT(
        aclnnAdaLayerNormBackward,
        INPUT(
            gradOutDesc, inputDesc, normalizedShape, rstdDesc, meanDesc, scaleDesc, (aclTensor*)nullptr, weightDesc, biasDesc),
        OUTPUT(gradInputOutDesc, gradScaleOutDesc, gradShiftOutDesc, gradWeightOutDesc, gradBiasOutDesc));
    // SAMPLE: only test GetWorkspaceSize
    aclRet = ut6.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    // scaleDesc nullptr
    auto ut7 = OP_API_UT(
        aclnnAdaLayerNormBackward,
        INPUT(
            gradOutDesc, inputDesc, normalizedShape, rstdDesc, meanDesc, (aclTensor*)nullptr, shiftDesc, weightDesc, biasDesc),
        OUTPUT(gradInputOutDesc, gradScaleOutDesc, gradShiftOutDesc, gradWeightOutDesc, gradBiasOutDesc));
    // SAMPLE: only test GetWorkspaceSize
    aclRet = ut7.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// layer_norm_x_backprop_v3 float16 mix dtype
TEST_F(l2_ada_layer_norm_backward_test, ascend910B2_aclnnAdaLayerNormBackward_x_backprop_v3_float16_mix_dtype)
{

    vector<int64_t> input_shape = {2, 6, 5, 6};
    vector<int64_t> norm_shape = {6};
    vector<int64_t> scale_shape = {2, 6, 6};
    vector<int64_t> shift_shape = {2, 6, 6};
    vector<int64_t> mean_shape = {2, 6, 5, 1};

    aclDataType x_dtype = ACL_FLOAT16;
    aclDataType dtype = ACL_FLOAT;
    aclFormat in_format = ACL_FORMAT_ND;
    auto gradOutDesc = TensorDesc(input_shape, x_dtype, in_format).ValueRange(-2, 2);
    auto inputDesc = gradOutDesc;
    auto normalizedShape = IntArrayDesc(norm_shape);
    auto meanDesc = TensorDesc(mean_shape, x_dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto rstdDesc = TensorDesc(mean_shape, x_dtype, ACL_FORMAT_ND).ValueRange(1, 2);
    auto scaleDesc = TensorDesc(scale_shape, x_dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto shiftDesc = TensorDesc(shift_shape, x_dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto weightDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto biasDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);


    auto gradInputOutDesc = TensorDesc(input_shape, x_dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto gradScaleOutDesc = TensorDesc(scale_shape, x_dtype, ACL_FORMAT_ND).Precision(0.01, 0.01);
    auto gradShiftOutDesc = TensorDesc(shift_shape, x_dtype, ACL_FORMAT_ND).Precision(0.01, 0.01);
    auto gradWeightOutDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto gradBiasOutDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(
        aclnnAdaLayerNormBackward,
        INPUT(gradOutDesc, inputDesc, normalizedShape, rstdDesc, meanDesc, scaleDesc, shiftDesc, weightDesc, biasDesc),
        OUTPUT(gradInputOutDesc, gradScaleOutDesc, gradShiftOutDesc, gradWeightOutDesc, gradBiasOutDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// bf16 dtype test
TEST_F(l2_ada_layer_norm_backward_test, ascend910B2_aclnnAdaLayerNormBackward_bf16_nd)
{
    vector<int64_t> input_shape = {2, 6, 5, 6};
    vector<int64_t> norm_shape = {6};
    vector<int64_t> scale_shape = {2, 6, 6};
    vector<int64_t> shift_shape = {2, 6, 6};
    vector<int64_t> mean_shape = {2, 6, 5, 1};
    aclDataType dtype = ACL_BF16;
    aclFormat in_format = ACL_FORMAT_ND;
    auto gradOutDesc = TensorDesc(input_shape, dtype, in_format).ValueRange(-2, 2);
    auto inputDesc = gradOutDesc;
    auto normalizedShape = IntArrayDesc(norm_shape);
    auto meanDesc = TensorDesc(mean_shape, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto rstdDesc = TensorDesc(mean_shape, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto scaleDesc = TensorDesc(scale_shape, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto shiftDesc = TensorDesc(shift_shape, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto weightDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto biasDesc = weightDesc;

    auto gradInputOutDesc = TensorDesc(input_shape, dtype, ACL_FORMAT_ND).Precision(0.01, 0.01);
    auto gradScaleOutDesc = TensorDesc(scale_shape, dtype, ACL_FORMAT_ND).Precision(0.01, 0.01);
    auto gradShiftOutDesc = TensorDesc(shift_shape, dtype, ACL_FORMAT_ND).Precision(0.01, 0.01);
    auto gradWeightOutDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).Precision(0.01, 0.01);
    auto gradBiasOutDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).Precision(0.01, 0.01);

    auto ut = OP_API_UT(
        aclnnAdaLayerNormBackward,
        INPUT(gradOutDesc, inputDesc, normalizedShape, rstdDesc, meanDesc, scaleDesc, shiftDesc, weightDesc, biasDesc),
        OUTPUT(gradInputOutDesc, gradScaleOutDesc, gradShiftOutDesc, gradWeightOutDesc, gradBiasOutDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// M = 0, N > 0
TEST_F(l2_ada_layer_norm_backward_test, ascend910B2_aclnnAdaLayerNormBackward_m_zero_n_positive)
{
    vector<int64_t> input_shape = {0, 2, 7, 6};
    vector<int64_t> norm_shape = {6};
    vector<int64_t> scale_shape = {0, 2, 6};
    vector<int64_t> shift_shape = {0, 2, 6};
    vector<int64_t> mean_shape = {0, 2, 7, 1};
    aclDataType dtype = ACL_FLOAT;
    aclFormat in_format = ACL_FORMAT_ND;
    auto gradOutDesc = TensorDesc(input_shape, dtype, in_format).ValueRange(-2, 2);
    auto inputDesc = gradOutDesc;
    auto normalizedShape = IntArrayDesc(norm_shape);
    auto meanDesc = TensorDesc(mean_shape, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto rstdDesc = TensorDesc(mean_shape, dtype, ACL_FORMAT_ND).ValueRange(1, 2);
    auto scaleDesc = TensorDesc(scale_shape, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto shiftDesc = TensorDesc(shift_shape, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto weightDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto biasDesc = weightDesc;

    auto gradInputOutDesc = TensorDesc(input_shape, dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto gradScaleOutDesc = TensorDesc(scale_shape, dtype, ACL_FORMAT_ND).Precision(0.01, 0.01);
    auto gradShiftOutDesc = TensorDesc(shift_shape, dtype, ACL_FORMAT_ND).Precision(0.01, 0.01);
    auto gradWeightOutDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto gradBiasOutDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(
        aclnnAdaLayerNormBackward,
        INPUT(gradOutDesc, inputDesc, normalizedShape, rstdDesc, meanDesc, scaleDesc, shiftDesc, weightDesc, biasDesc),
        OUTPUT(gradInputOutDesc, gradScaleOutDesc, gradShiftOutDesc, gradWeightOutDesc, gradBiasOutDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 3D input [B, S, H]
TEST_F(l2_ada_layer_norm_backward_test, ascend910B2_aclnnAdaLayerNormBackward_3d_input)
{
    vector<int64_t> input_shape = {2, 3, 4};
    vector<int64_t> norm_shape = {4};
    vector<int64_t> scale_shape = {2, 4};
    vector<int64_t> shift_shape = {2, 4};
    vector<int64_t> mean_shape = {2, 3, 1};
    aclDataType dtype = ACL_FLOAT;
    aclFormat in_format = ACL_FORMAT_ND;
    auto gradOutDesc = TensorDesc(input_shape, dtype, in_format).ValueRange(-2, 2);
    auto inputDesc = gradOutDesc;
    auto normalizedShape = IntArrayDesc(norm_shape);
    auto meanDesc = TensorDesc(mean_shape, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto rstdDesc = TensorDesc(mean_shape, dtype, ACL_FORMAT_ND).ValueRange(1, 2);
    auto scaleDesc = TensorDesc(scale_shape, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto shiftDesc = TensorDesc(shift_shape, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto weightDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto biasDesc = weightDesc;

    auto gradInputOutDesc = TensorDesc(input_shape, dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto gradScaleOutDesc = TensorDesc(scale_shape, dtype, ACL_FORMAT_ND).Precision(0.01, 0.01);
    auto gradShiftOutDesc = TensorDesc(shift_shape, dtype, ACL_FORMAT_ND).Precision(0.01, 0.01);
    auto gradWeightOutDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto gradBiasOutDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(
        aclnnAdaLayerNormBackward,
        INPUT(gradOutDesc, inputDesc, normalizedShape, rstdDesc, meanDesc, scaleDesc, shiftDesc, weightDesc, biasDesc),
        OUTPUT(gradInputOutDesc, gradScaleOutDesc, gradShiftOutDesc, gradWeightOutDesc, gradBiasOutDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 5D input
TEST_F(l2_ada_layer_norm_backward_test, ascend910B2_aclnnAdaLayerNormBackward_5d_input)
{
    vector<int64_t> input_shape = {2, 2, 2, 2, 2};
    vector<int64_t> norm_shape = {2};
    vector<int64_t> scale_shape = {2, 2, 2, 2};
    vector<int64_t> shift_shape = {2, 2, 2, 2};
    vector<int64_t> mean_shape = {2, 2, 2, 2, 1};
    aclDataType dtype = ACL_FLOAT;
    aclFormat in_format = ACL_FORMAT_ND;
    auto gradOutDesc = TensorDesc(input_shape, dtype, in_format).ValueRange(-2, 2);
    auto inputDesc = gradOutDesc;
    auto normalizedShape = IntArrayDesc(norm_shape);
    auto meanDesc = TensorDesc(mean_shape, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto rstdDesc = TensorDesc(mean_shape, dtype, ACL_FORMAT_ND).ValueRange(1, 2);
    auto scaleDesc = TensorDesc(scale_shape, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto shiftDesc = TensorDesc(shift_shape, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto weightDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto biasDesc = weightDesc;

    auto gradInputOutDesc = TensorDesc(input_shape, dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto gradScaleOutDesc = TensorDesc(scale_shape, dtype, ACL_FORMAT_ND).Precision(0.01, 0.01);
    auto gradShiftOutDesc = TensorDesc(shift_shape, dtype, ACL_FORMAT_ND).Precision(0.01, 0.01);
    auto gradWeightOutDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto gradBiasOutDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(
        aclnnAdaLayerNormBackward,
        INPUT(gradOutDesc, inputDesc, normalizedShape, rstdDesc, meanDesc, scaleDesc, shiftDesc, weightDesc, biasDesc),
        OUTPUT(gradInputOutDesc, gradScaleOutDesc, gradShiftOutDesc, gradWeightOutDesc, gradBiasOutDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// scale/shift [B, 1, H] shape
TEST_F(l2_ada_layer_norm_backward_test, ascend910B2_aclnnAdaLayerNormBackward_scale_shift_b1h_shape)
{
    vector<int64_t> input_shape = {2, 6, 5, 6};
    vector<int64_t> norm_shape = {6};
    vector<int64_t> scale_shape = {2, 1, 6};
    vector<int64_t> shift_shape = {2, 1, 6};
    vector<int64_t> mean_shape = {2, 6, 5, 1};
    aclDataType dtype = ACL_FLOAT;
    aclFormat in_format = ACL_FORMAT_ND;
    auto gradOutDesc = TensorDesc(input_shape, dtype, in_format).ValueRange(-2, 2);
    auto inputDesc = gradOutDesc;
    auto normalizedShape = IntArrayDesc(norm_shape);
    auto meanDesc = TensorDesc(mean_shape, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto rstdDesc = TensorDesc(mean_shape, dtype, ACL_FORMAT_ND).ValueRange(1, 2);
    auto scaleDesc = TensorDesc(scale_shape, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto shiftDesc = TensorDesc(shift_shape, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto weightDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto biasDesc = weightDesc;

    auto gradInputOutDesc = TensorDesc(input_shape, dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto gradScaleOutDesc = TensorDesc(scale_shape, dtype, ACL_FORMAT_ND).Precision(0.01, 0.01);
    auto gradShiftOutDesc = TensorDesc(shift_shape, dtype, ACL_FORMAT_ND).Precision(0.01, 0.01);
    auto gradWeightOutDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto gradBiasOutDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(
        aclnnAdaLayerNormBackward,
        INPUT(gradOutDesc, inputDesc, normalizedShape, rstdDesc, meanDesc, scaleDesc, shiftDesc, weightDesc, biasDesc),
        OUTPUT(gradInputOutDesc, gradScaleOutDesc, gradShiftOutDesc, gradWeightOutDesc, gradBiasOutDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// weight nullptr
TEST_F(l2_ada_layer_norm_backward_test, ascend910B2_aclnnAdaLayerNormBackward_weight_nullptr)
{
    vector<int64_t> input_shape = {2, 6, 5, 6};
    vector<int64_t> norm_shape = {6};
    vector<int64_t> scale_shape = {2, 6, 6};
    vector<int64_t> shift_shape = {2, 6, 6};
    vector<int64_t> mean_shape = {2, 6, 5, 1};
    aclDataType dtype = ACL_FLOAT;
    aclFormat in_format = ACL_FORMAT_ND;
    auto gradOutDesc = TensorDesc(input_shape, dtype, in_format).ValueRange(-2, 2);
    auto inputDesc = gradOutDesc;
    auto normalizedShape = IntArrayDesc(norm_shape);
    auto meanDesc = TensorDesc(mean_shape, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto rstdDesc = TensorDesc(mean_shape, dtype, ACL_FORMAT_ND).ValueRange(1, 2);
    auto scaleDesc = TensorDesc(scale_shape, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto shiftDesc = TensorDesc(shift_shape, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto weightDesc = (aclTensor*)nullptr;
    auto biasDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);

    auto gradInputOutDesc = TensorDesc(input_shape, dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto gradScaleOutDesc = TensorDesc(scale_shape, dtype, ACL_FORMAT_ND).Precision(0.01, 0.01);
    auto gradShiftOutDesc = TensorDesc(shift_shape, dtype, ACL_FORMAT_ND).Precision(0.01, 0.01);
    auto gradWeightOutDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto gradBiasOutDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(
        aclnnAdaLayerNormBackward,
        INPUT(gradOutDesc, inputDesc, normalizedShape, rstdDesc, meanDesc, scaleDesc, shiftDesc, weightDesc, biasDesc),
        OUTPUT(gradInputOutDesc, gradScaleOutDesc, gradShiftOutDesc, gradWeightOutDesc, gradBiasOutDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// bias nullptr
TEST_F(l2_ada_layer_norm_backward_test, ascend910B2_aclnnAdaLayerNormBackward_bias_nullptr)
{
    vector<int64_t> input_shape = {2, 6, 5, 6};
    vector<int64_t> norm_shape = {6};
    vector<int64_t> scale_shape = {2, 6, 6};
    vector<int64_t> shift_shape = {2, 6, 6};
    vector<int64_t> mean_shape = {2, 6, 5, 1};
    aclDataType dtype = ACL_FLOAT;
    aclFormat in_format = ACL_FORMAT_ND;
    auto gradOutDesc = TensorDesc(input_shape, dtype, in_format).ValueRange(-2, 2);
    auto inputDesc = gradOutDesc;
    auto normalizedShape = IntArrayDesc(norm_shape);
    auto meanDesc = TensorDesc(mean_shape, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto rstdDesc = TensorDesc(mean_shape, dtype, ACL_FORMAT_ND).ValueRange(1, 2);
    auto scaleDesc = TensorDesc(scale_shape, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto shiftDesc = TensorDesc(shift_shape, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto weightDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto biasDesc = (aclTensor*)nullptr;

    auto gradInputOutDesc = TensorDesc(input_shape, dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto gradScaleOutDesc = TensorDesc(scale_shape, dtype, ACL_FORMAT_ND).Precision(0.01, 0.01);
    auto gradShiftOutDesc = TensorDesc(shift_shape, dtype, ACL_FORMAT_ND).Precision(0.01, 0.01);
    auto gradWeightOutDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto gradBiasOutDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(
        aclnnAdaLayerNormBackward,
        INPUT(gradOutDesc, inputDesc, normalizedShape, rstdDesc, meanDesc, scaleDesc, shiftDesc, weightDesc, biasDesc),
        OUTPUT(gradInputOutDesc, gradScaleOutDesc, gradShiftOutDesc, gradWeightOutDesc, gradBiasOutDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// weight and bias nullptr
TEST_F(l2_ada_layer_norm_backward_test, ascend910B2_aclnnAdaLayerNormBackward_weight_bias_nullptr)
{
    vector<int64_t> input_shape = {2, 6, 5, 6};
    vector<int64_t> norm_shape = {6};
    vector<int64_t> scale_shape = {2, 6, 6};
    vector<int64_t> shift_shape = {2, 6, 6};
    vector<int64_t> mean_shape = {2, 6, 5, 1};
    aclDataType dtype = ACL_FLOAT;
    aclFormat in_format = ACL_FORMAT_ND;
    auto gradOutDesc = TensorDesc(input_shape, dtype, in_format).ValueRange(-2, 2);
    auto inputDesc = gradOutDesc;
    auto normalizedShape = IntArrayDesc(norm_shape);
    auto meanDesc = TensorDesc(mean_shape, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto rstdDesc = TensorDesc(mean_shape, dtype, ACL_FORMAT_ND).ValueRange(1, 2);
    auto scaleDesc = TensorDesc(scale_shape, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto shiftDesc = TensorDesc(shift_shape, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto weightDesc = (aclTensor*)nullptr;
    auto biasDesc = (aclTensor*)nullptr;

    auto gradInputOutDesc = TensorDesc(input_shape, dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto gradScaleOutDesc = TensorDesc(scale_shape, dtype, ACL_FORMAT_ND).Precision(0.01, 0.01);
    auto gradShiftOutDesc = TensorDesc(shift_shape, dtype, ACL_FORMAT_ND).Precision(0.01, 0.01);
    auto gradWeightOutDesc = TensorDesc(norm_shape, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto gradBiasOutDesc = TensorDesc(norm_shape, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(
        aclnnAdaLayerNormBackward,
        INPUT(gradOutDesc, inputDesc, normalizedShape, rstdDesc, meanDesc, scaleDesc, shiftDesc, weightDesc, biasDesc),
        OUTPUT(gradInputOutDesc, gradScaleOutDesc, gradShiftOutDesc, gradWeightOutDesc, gradBiasOutDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// all output nullptr
TEST_F(l2_ada_layer_norm_backward_test, ascend910B2_aclnnAdaLayerNormBackward_all_output_nullptr)
{
    vector<int64_t> input_shape = {2, 6, 5, 6};
    vector<int64_t> norm_shape = {6};
    vector<int64_t> scale_shape = {2, 6, 6};
    vector<int64_t> shift_shape = {2, 6, 6};
    vector<int64_t> mean_shape = {2, 6, 5, 1};
    aclDataType dtype = ACL_FLOAT;
    aclFormat in_format = ACL_FORMAT_ND;
    auto gradOutDesc = TensorDesc(input_shape, dtype, in_format).ValueRange(-2, 2);
    auto inputDesc = gradOutDesc;
    auto normalizedShape = IntArrayDesc(norm_shape);
    auto meanDesc = TensorDesc(mean_shape, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto rstdDesc = TensorDesc(mean_shape, dtype, ACL_FORMAT_ND).ValueRange(1, 2);
    auto scaleDesc = TensorDesc(scale_shape, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto shiftDesc = TensorDesc(shift_shape, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto weightDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto biasDesc = weightDesc;

    auto ut = OP_API_UT(
        aclnnAdaLayerNormBackward,
        INPUT(gradOutDesc, inputDesc, normalizedShape, rstdDesc, meanDesc, scaleDesc, shiftDesc, weightDesc, biasDesc),
        OUTPUT((aclTensor*)nullptr, (aclTensor*)nullptr, (aclTensor*)nullptr, (aclTensor*)nullptr, (aclTensor*)nullptr));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}
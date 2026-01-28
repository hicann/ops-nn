/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <vector>
#include "gtest/gtest.h"

#include "../../../op_host/op_api/aclnn_layer_norm_backward.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace std;
using namespace op;

class l2_layer_norm_backward_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "layer_norm_backward_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "layer_norm_backward_test TearDown" << endl;
    }
};

// nchw, float32
TEST_F(l2_layer_norm_backward_test, ascend910B2_aclnnLayerNormBackward_float32_nd_float32_nd)
{
    vector<int64_t> input_shape = {9, 2, 2, 2};
    vector<int64_t> norm_shape = {2, 2};
    vector<int64_t> mean_shape = {9, 2, 1, 1};
    aclDataType dtype = ACL_FLOAT;
    aclFormat in_format = ACL_FORMAT_ND;
    auto gradOutDesc = TensorDesc(input_shape, dtype, in_format).ValueRange(-2, 2);
    auto inputDesc = gradOutDesc;
    auto normalizedShape = IntArrayDesc(norm_shape);
    auto meanDesc = TensorDesc(mean_shape, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto rstdDesc = TensorDesc(mean_shape, dtype, ACL_FORMAT_ND).ValueRange(1, 2);
    auto weightDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto biasDesc = weightDesc;
    auto outputmask = BoolArrayDesc(vector<bool>{true, true, true});

    auto gradInputOutDesc = TensorDesc(input_shape, dtype, in_format).Precision(0.01, 0.01);
    auto gradWeightOutDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).Precision(0.01, 0.01);
    auto gradBiasOutDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).Precision(0.01, 0.01);

    auto ut = OP_API_UT(
        aclnnLayerNormBackward,
        INPUT(gradOutDesc, inputDesc, normalizedShape, meanDesc, rstdDesc, weightDesc, biasDesc, outputmask),
        OUTPUT(gradInputOutDesc, gradWeightOutDesc, gradBiasOutDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// float64: kernel not support now
TEST_F(l2_layer_norm_backward_test, ascend910B2_aclnnLayerNormBackward_double_nd_double_nd)
{
    vector<int64_t> input_shape = {2, 6, 5, 6};
    vector<int64_t> norm_shape = {6, 5, 6};
    vector<int64_t> mean_shape = {2, 1, 1, 1};
    aclDataType dtype = ACL_DOUBLE;
    aclFormat in_format = ACL_FORMAT_NHWC;
    auto gradOutDesc = TensorDesc(input_shape, dtype, in_format).ValueRange(-2, 2);
    auto inputDesc = gradOutDesc;
    auto normalizedShape = IntArrayDesc(norm_shape);
    auto meanDesc = TensorDesc(mean_shape, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto rstdDesc = TensorDesc(mean_shape, dtype, ACL_FORMAT_ND).ValueRange(1, 2);
    auto weightDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto biasDesc = weightDesc;
    auto outputmask = BoolArrayDesc(vector<bool>{true, true, true});

    auto gradInputOutDesc = TensorDesc(input_shape, dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto gradWeightOutDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto gradBiasOutDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(
        aclnnLayerNormBackward,
        INPUT(gradOutDesc, inputDesc, normalizedShape, meanDesc, rstdDesc, weightDesc, biasDesc, outputmask),
        OUTPUT(gradInputOutDesc, gradWeightOutDesc, gradBiasOutDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// error input dtype
TEST_F(l2_layer_norm_backward_test, ascend910B2_aclnnLayerNormBackward_error_input_dtype)
{
    vector<int64_t> input_shape = {2, 6, 5, 6};
    vector<int64_t> norm_shape = {6, 5, 6};
    vector<int64_t> mean_shape = {2, 1, 1, 1};
    aclDataType dtype = ACL_FLOAT;
    aclFormat in_format = ACL_FORMAT_NHWC;
    auto gradOutDesc = TensorDesc(input_shape, ACL_UINT64, in_format).ValueRange(-2, 2);
    auto inputDesc = gradOutDesc;
    auto normalizedShape = IntArrayDesc(norm_shape);
    auto meanDesc = TensorDesc(mean_shape, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto rstdDesc = TensorDesc(mean_shape, dtype, ACL_FORMAT_ND).ValueRange(1, 2);
    auto weightDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto biasDesc = weightDesc;
    auto outputmask = BoolArrayDesc(vector<bool>{true, true, true});

    auto gradInputOutDesc = TensorDesc(input_shape, dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto gradWeightOutDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto gradBiasOutDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(
        aclnnLayerNormBackward,
        INPUT(gradOutDesc, inputDesc, normalizedShape, meanDesc, rstdDesc, weightDesc, biasDesc, outputmask),
        OUTPUT(gradInputOutDesc, gradWeightOutDesc, gradBiasOutDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// error out dtype
TEST_F(l2_layer_norm_backward_test, ascend910B2_aclnnLayerNormBackward_error_out_dtype)
{
    vector<int64_t> input_shape = {2, 6, 5, 6};
    vector<int64_t> norm_shape = {6, 5, 6};
    vector<int64_t> mean_shape = {2, 1, 1, 1};
    aclDataType dtype = ACL_FLOAT;
    aclFormat in_format = ACL_FORMAT_NHWC;
    auto gradOutDesc = TensorDesc(input_shape, dtype, in_format).ValueRange(-2, 2);
    auto inputDesc = gradOutDesc;
    auto normalizedShape = IntArrayDesc(norm_shape);
    auto meanDesc = TensorDesc(mean_shape, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto rstdDesc = TensorDesc(mean_shape, dtype, ACL_FORMAT_ND).ValueRange(1, 2);
    auto weightDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto biasDesc = weightDesc;
    auto outputmask = BoolArrayDesc(vector<bool>{true, true, true});

    auto gradInputOutDesc = TensorDesc(input_shape, ACL_UINT64, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto gradWeightOutDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto gradBiasOutDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(
        aclnnLayerNormBackward,
        INPUT(gradOutDesc, inputDesc, normalizedShape, meanDesc, rstdDesc, weightDesc, biasDesc, outputmask),
        OUTPUT(gradInputOutDesc, gradWeightOutDesc, gradBiasOutDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// ndhwc, float32
TEST_F(l2_layer_norm_backward_test, ascend910B2_aclnnLayerNormBackward_float32_ndhwc_float32_ndhwc)
{
    vector<int64_t> input_shape = {2, 3, 4, 5, 6};
    vector<int64_t> norm_shape = {3, 4, 5, 6};
    vector<int64_t> mean_shape = {2, 1, 1, 1, 1};
    aclDataType dtype = ACL_FLOAT;
    aclFormat in_format = ACL_FORMAT_NDHWC;
    auto gradOutDesc = TensorDesc(input_shape, dtype, in_format).ValueRange(-2, 2);
    auto inputDesc = gradOutDesc;
    auto normalizedShape = IntArrayDesc(norm_shape);
    auto meanDesc = TensorDesc(mean_shape, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto rstdDesc = TensorDesc(mean_shape, dtype, ACL_FORMAT_ND).ValueRange(1, 2);
    auto weightDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto biasDesc = weightDesc;
    auto outputmask = BoolArrayDesc(vector<bool>{true, true, true});

    auto gradInputOutDesc = TensorDesc(input_shape, dtype, ACL_FORMAT_ND).Precision(0.1, 0.1);
    auto gradWeightOutDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).Precision(0.1, 0.1);
    auto gradBiasOutDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).Precision(0.1, 0.1);

    auto ut = OP_API_UT(
        aclnnLayerNormBackward,
        INPUT(gradOutDesc, inputDesc, normalizedShape, meanDesc, rstdDesc, weightDesc, biasDesc, outputmask),
        OUTPUT(gradInputOutDesc, gradWeightOutDesc, gradBiasOutDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// ncdhw, float32
TEST_F(l2_layer_norm_backward_test, ascend910B2_aclnnLayerNormBackward_float32_ncdhw_float32_ncdhw)
{
    vector<int64_t> input_shape = {2, 2, 2, 2, 2};
    vector<int64_t> norm_shape = {2, 2};
    vector<int64_t> mean_shape = {2, 2, 2, 1, 1};
    aclDataType dtype = ACL_FLOAT;
    aclFormat in_format = ACL_FORMAT_NCDHW;
    auto gradOutDesc = TensorDesc(input_shape, dtype, in_format).ValueRange(-2, 2);
    auto inputDesc = gradOutDesc;
    auto normalizedShape = IntArrayDesc(norm_shape);
    auto meanDesc = TensorDesc(mean_shape, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto rstdDesc = TensorDesc(mean_shape, dtype, ACL_FORMAT_ND).ValueRange(1, 2);
    auto weightDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto biasDesc = weightDesc;
    auto outputmask = BoolArrayDesc(vector<bool>{true, true, true});

    auto gradInputOutDesc = TensorDesc(input_shape, dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto gradWeightOutDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto gradBiasOutDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(
        aclnnLayerNormBackward,
        INPUT(gradOutDesc, inputDesc, normalizedShape, meanDesc, rstdDesc, weightDesc, biasDesc, outputmask),
        OUTPUT(gradInputOutDesc, gradWeightOutDesc, gradBiasOutDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// M = 0, N > 0, output_mask second is true
TEST_F(l2_layer_norm_backward_test, ascend910B2_aclnnLayerNormBackward_m_positive_n_zero_dw_is_true)
{
    vector<int64_t> input_shape = {0, 2, 7, 18};
    vector<int64_t> norm_shape = {7, 18};
    vector<int64_t> mean_shape = {0, 2, 1, 1};
    aclDataType dtype = ACL_FLOAT;
    aclFormat in_format = ACL_FORMAT_ND;
    auto gradOutDesc = TensorDesc(input_shape, dtype, in_format).ValueRange(-2, 2);
    auto inputDesc = gradOutDesc;
    auto normalizedShape = IntArrayDesc(norm_shape);
    auto meanDesc = TensorDesc(mean_shape, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto rstdDesc = TensorDesc(mean_shape, dtype, ACL_FORMAT_ND).ValueRange(1, 2);
    auto weightDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto biasDesc = weightDesc;
    auto outputmask = BoolArrayDesc(vector<bool>{true, true, true});

    auto gradInputOutDesc = TensorDesc(input_shape, dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto gradWeightOutDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto gradBiasOutDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(
        aclnnLayerNormBackward,
        INPUT(gradOutDesc, inputDesc, normalizedShape, meanDesc, rstdDesc, weightDesc, biasDesc, outputmask),
        OUTPUT(gradInputOutDesc, gradWeightOutDesc, gradBiasOutDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// M = 0, N > 0, output_mask second is false
TEST_F(l2_layer_norm_backward_test, ascend910B2_aclnnLayerNormBackward_m_positive_n_zero_dw_is_false)
{
    vector<int64_t> input_shape = {0, 2, 7, 18};
    vector<int64_t> norm_shape = {7, 18};
    vector<int64_t> mean_shape = {0, 2, 1, 1};
    aclDataType dtype = ACL_FLOAT;
    aclFormat in_format = ACL_FORMAT_ND;
    auto gradOutDesc = TensorDesc(input_shape, dtype, in_format).ValueRange(-2, 2);
    auto inputDesc = gradOutDesc;
    auto normalizedShape = IntArrayDesc(norm_shape);
    auto meanDesc = TensorDesc(mean_shape, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto rstdDesc = TensorDesc(mean_shape, dtype, ACL_FORMAT_ND).ValueRange(1, 2);
    auto weightDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto biasDesc = weightDesc;
    auto outputmask = BoolArrayDesc(vector<bool>{true, false, true});

    auto gradInputOutDesc = TensorDesc(input_shape, dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto gradWeightOutDesc = (aclTensor*)nullptr;
    auto gradBiasOutDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(
        aclnnLayerNormBackward,
        INPUT(gradOutDesc, inputDesc, normalizedShape, meanDesc, rstdDesc, weightDesc, biasDesc, outputmask),
        OUTPUT(gradInputOutDesc, gradWeightOutDesc, gradBiasOutDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// M > 0, N = 0
TEST_F(l2_layer_norm_backward_test, ascend910B2_aclnnLayerNormBackward_5_0_3_float_nd_0_3_float_nd)
{
    vector<int64_t> input_shape = {5, 0, 3};
    vector<int64_t> norm_shape = {0, 3};
    vector<int64_t> mean_shape = {5, 1, 1};
    aclDataType dtype = ACL_FLOAT;
    aclFormat in_format = ACL_FORMAT_ND;
    auto gradOutDesc = TensorDesc(input_shape, dtype, in_format).ValueRange(-2, 2);
    auto inputDesc = gradOutDesc;
    auto normalizedShape = IntArrayDesc(norm_shape);
    auto meanDesc = TensorDesc(mean_shape, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto rstdDesc = TensorDesc(mean_shape, dtype, ACL_FORMAT_ND).ValueRange(1, 2);
    auto weightDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto biasDesc = weightDesc;
    auto outputmask = BoolArrayDesc(vector<bool>{true, true, true});

    auto gradInputOutDesc = TensorDesc(input_shape, dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto gradWeightOutDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto gradBiasOutDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(
        aclnnLayerNormBackward,
        INPUT(gradOutDesc, inputDesc, normalizedShape, meanDesc, rstdDesc, weightDesc, biasDesc, outputmask),
        OUTPUT(gradInputOutDesc, gradWeightOutDesc, gradBiasOutDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// error input len
TEST_F(l2_layer_norm_backward_test, ascend910B2_aclnnLayerNormBackward_error_input_len)
{
    vector<int64_t> input_shape = {1, 2, 1, 2, 1, 2, 1, 2, 2, 2, 2};
    vector<int64_t> norm_shape = {2, 1, 2, 1, 2, 1, 2, 2, 2, 2};
    vector<int64_t> mean_shape = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    aclDataType dtype = ACL_FLOAT;
    aclFormat in_format = ACL_FORMAT_ND;
    auto gradOutDesc = TensorDesc(input_shape, dtype, in_format).ValueRange(-2, 2);
    auto inputDesc = gradOutDesc;
    auto normalizedShape = IntArrayDesc(norm_shape);
    auto meanDesc = TensorDesc(mean_shape, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto rstdDesc = TensorDesc(mean_shape, dtype, ACL_FORMAT_ND).ValueRange(1, 2);
    auto weightDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto biasDesc = weightDesc;
    auto outputmask = BoolArrayDesc(vector<bool>{true, true, true});

    auto gradInputOutDesc = TensorDesc(input_shape, dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto gradWeightOutDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto gradBiasOutDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(
        aclnnLayerNormBackward,
        INPUT(gradOutDesc, inputDesc, normalizedShape, meanDesc, rstdDesc, weightDesc, biasDesc, outputmask),
        OUTPUT(gradInputOutDesc, gradWeightOutDesc, gradBiasOutDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// error normalizedShape len
TEST_F(l2_layer_norm_backward_test, ascend910B2_aclnnLayerNormBackward_error_normalizedShape_len)
{
    vector<int64_t> input_shape = {2, 6, 5, 6};
    vector<int64_t> norm_shape = {};
    vector<int64_t> mean_shape = {2, 6, 5, 6};
    aclDataType dtype = ACL_FLOAT;
    aclFormat in_format = ACL_FORMAT_ND;
    auto gradOutDesc = TensorDesc(input_shape, dtype, in_format).ValueRange(-2, 2);
    auto inputDesc = gradOutDesc;
    auto normalizedShape = IntArrayDesc(norm_shape);
    auto meanDesc = TensorDesc(mean_shape, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto rstdDesc = TensorDesc(mean_shape, dtype, ACL_FORMAT_ND).ValueRange(1, 2);
    auto weightDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto biasDesc = weightDesc;
    auto outputmask = BoolArrayDesc(vector<bool>{true, true, true});

    auto gradInputOutDesc = TensorDesc(input_shape, dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto gradWeightOutDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto gradBiasOutDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(
        aclnnLayerNormBackward,
        INPUT(gradOutDesc, inputDesc, normalizedShape, meanDesc, rstdDesc, weightDesc, biasDesc, outputmask),
        OUTPUT(gradInputOutDesc, gradWeightOutDesc, gradBiasOutDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// error weight len
TEST_F(l2_layer_norm_backward_test, ascend910B2_aclnnLayerNormBackward_error_weight_len)
{
    vector<int64_t> input_shape = {2, 3, 4};
    vector<int64_t> norm_shape = {3, 4};
    vector<int64_t> mean_shape = {2, 1, 1};
    aclDataType dtype = ACL_FLOAT;
    aclFormat in_format = ACL_FORMAT_ND;
    auto gradOutDesc = TensorDesc(input_shape, dtype, in_format).ValueRange(-2, 2);
    auto inputDesc = gradOutDesc;
    auto normalizedShape = IntArrayDesc(norm_shape);
    auto meanDesc = TensorDesc(mean_shape, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto rstdDesc = TensorDesc(mean_shape, dtype, ACL_FORMAT_ND).ValueRange(1, 2);
    auto weightDesc = TensorDesc({1, 2, 1, 2, 1, 2, 1, 2, 1}, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto biasDesc = weightDesc;
    auto outputmask = BoolArrayDesc(vector<bool>{true, true, true});

    auto gradInputOutDesc = TensorDesc(input_shape, dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto gradWeightOutDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto gradBiasOutDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(
        aclnnLayerNormBackward,
        INPUT(gradOutDesc, inputDesc, normalizedShape, meanDesc, rstdDesc, weightDesc, biasDesc, outputmask),
        OUTPUT(gradInputOutDesc, gradWeightOutDesc, gradBiasOutDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// error bias len
TEST_F(l2_layer_norm_backward_test, ascend910B2_aclnnLayerNormBackward_error_bias_len)
{
    vector<int64_t> input_shape = {2, 3, 4};
    vector<int64_t> norm_shape = {3, 4};
    vector<int64_t> mean_shape = {2, 1, 1};
    aclDataType dtype = ACL_FLOAT;
    aclFormat in_format = ACL_FORMAT_ND;
    auto gradOutDesc = TensorDesc(input_shape, dtype, in_format).ValueRange(-2, 2);
    auto inputDesc = gradOutDesc;
    auto normalizedShape = IntArrayDesc(norm_shape);
    auto meanDesc = TensorDesc(mean_shape, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto rstdDesc = TensorDesc(mean_shape, dtype, ACL_FORMAT_ND).ValueRange(1, 2);
    auto weightDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto biasDesc = TensorDesc({1, 2, 1, 2, 1, 2, 1, 2, 1}, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto outputmask = BoolArrayDesc(vector<bool>{true, true, true});

    auto gradInputOutDesc = TensorDesc(input_shape, dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto gradWeightOutDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto gradBiasOutDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(
        aclnnLayerNormBackward,
        INPUT(gradOutDesc, inputDesc, normalizedShape, meanDesc, rstdDesc, weightDesc, biasDesc, outputmask),
        OUTPUT(gradInputOutDesc, gradWeightOutDesc, gradBiasOutDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// error gradInputOut len
TEST_F(l2_layer_norm_backward_test, ascend910B2_aclnnLayerNormBackward_error_grad_input_out_len)
{
    vector<int64_t> input_shape = {2, 3, 4};
    vector<int64_t> norm_shape = {3, 4};
    vector<int64_t> mean_shape = {2, 1, 1};
    aclDataType dtype = ACL_FLOAT;
    aclFormat in_format = ACL_FORMAT_ND;
    auto gradOutDesc = TensorDesc(input_shape, dtype, in_format).ValueRange(-2, 2);
    auto inputDesc = gradOutDesc;
    auto normalizedShape = IntArrayDesc(norm_shape);
    auto meanDesc = TensorDesc(mean_shape, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto rstdDesc = TensorDesc(mean_shape, dtype, ACL_FORMAT_ND).ValueRange(1, 2);
    auto weightDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto biasDesc = weightDesc;
    auto outputmask = BoolArrayDesc(vector<bool>{true, true, true});

    auto gradInputOutDesc = TensorDesc({1, 2, 1, 2, 1, 2, 1, 2, 1}, dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto gradWeightOutDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto gradBiasOutDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(
        aclnnLayerNormBackward,
        INPUT(gradOutDesc, inputDesc, normalizedShape, meanDesc, rstdDesc, weightDesc, biasDesc, outputmask),
        OUTPUT(gradInputOutDesc, gradWeightOutDesc, gradBiasOutDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// error weight, bias and normalizedShape
TEST_F(l2_layer_norm_backward_test, ascend910B2_aclnnLayerNormBackward_diff_weight_and_normalizedShape)
{
    auto gradOutDesc = TensorDesc({2, 6, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto inputDesc = gradOutDesc;
    auto normalizedShape = IntArrayDesc(vector<int64_t>{5, 6});
    auto meanDesc = TensorDesc({2, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto rstdDesc = TensorDesc({2, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto weightDesc = TensorDesc({6, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto biasDesc = weightDesc;
    auto outputmask = BoolArrayDesc(vector<bool>{true, true, true});

    auto gradInputOutDesc = TensorDesc({2, 6, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto gradWeightOutDesc = TensorDesc({6, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto gradBiasOutDesc = TensorDesc({6, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(
        aclnnLayerNormBackward,
        INPUT(gradOutDesc, inputDesc, normalizedShape, meanDesc, rstdDesc, weightDesc, biasDesc, outputmask),
        OUTPUT(gradInputOutDesc, gradWeightOutDesc, gradBiasOutDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// error input and normalizedShape
TEST_F(l2_layer_norm_backward_test, ascend910B2_aclnnLayerNormBackward_diff_input_and_normalizedShape)
{
    auto gradOutDesc = TensorDesc({2, 6, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto inputDesc = gradOutDesc;
    auto normalizedShape = IntArrayDesc(vector<int64_t>{7, 8});
    auto meanDesc = TensorDesc({2, 6, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto rstdDesc = TensorDesc({2, 6, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto weightDesc = TensorDesc({7, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto biasDesc = weightDesc;
    auto outputmask = BoolArrayDesc(vector<bool>{true, true, true});

    auto gradInputOutDesc = TensorDesc({2, 6, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto gradWeightOutDesc = TensorDesc({7, 8}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto gradBiasOutDesc = TensorDesc({7, 8}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(
        aclnnLayerNormBackward,
        INPUT(gradOutDesc, inputDesc, normalizedShape, meanDesc, rstdDesc, weightDesc, biasDesc, outputmask),
        OUTPUT(gradInputOutDesc, gradWeightOutDesc, gradBiasOutDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// nullptr
TEST_F(l2_layer_norm_backward_test, ascend910B2_aclnnLayerNormBackward_nullptr)
{
    auto gradOutDesc = TensorDesc({7, 8, 9}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto inputDesc = TensorDesc({7, 8, 9}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto normalizedShape = IntArrayDesc(vector<int64_t>{8, 9});
    auto meanDesc = TensorDesc({7, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto rstdDesc = TensorDesc({7, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto weightDesc = TensorDesc({8, 9}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto biasDesc = weightDesc;
    auto outputmask = BoolArrayDesc(vector<bool>{true, true, true});

    auto gradInputOutDesc = TensorDesc({7, 8, 9}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto gradWeightOutDesc = TensorDesc({8, 9}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto gradBiasOutDesc = TensorDesc({8, 9}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);

    uint64_t workspaceSize = 0;

    // gradOut nullptr
    auto ut1 = OP_API_UT(
        aclnnLayerNormBackward,
        INPUT((aclTensor*)nullptr, inputDesc, normalizedShape, meanDesc, rstdDesc, weightDesc, biasDesc, outputmask),
        OUTPUT(gradInputOutDesc, gradWeightOutDesc, gradBiasOutDesc));
    // SAMPLE: only test GetWorkspaceSize
    aclnnStatus aclRet = ut1.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    // input nullptr
    auto ut2 = OP_API_UT(
        aclnnLayerNormBackward,
        INPUT(gradOutDesc, (aclTensor*)nullptr, normalizedShape, meanDesc, rstdDesc, weightDesc, biasDesc, outputmask),
        OUTPUT(gradInputOutDesc, gradWeightOutDesc, gradBiasOutDesc));
    // SAMPLE: only test GetWorkspaceSize
    aclRet = ut2.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    // normalizedShape nullptr
    auto ut3 = OP_API_UT(
        aclnnLayerNormBackward,
        INPUT(gradOutDesc, inputDesc, (aclIntArray*)nullptr, meanDesc, rstdDesc, weightDesc, biasDesc, outputmask),
        OUTPUT(gradInputOutDesc, gradWeightOutDesc, gradBiasOutDesc));
    // SAMPLE: only test GetWorkspaceSize
    aclRet = ut3.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    // meanDesc nullptr
    auto ut4 = OP_API_UT(
        aclnnLayerNormBackward,
        INPUT(gradOutDesc, inputDesc, normalizedShape, (aclTensor*)nullptr, rstdDesc, weightDesc, biasDesc, outputmask),
        OUTPUT(gradInputOutDesc, gradWeightOutDesc, gradBiasOutDesc));
    // SAMPLE: only test GetWorkspaceSize
    aclRet = ut4.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    // rstdDesc nullptr
    auto ut5 = OP_API_UT(
        aclnnLayerNormBackward,
        INPUT(gradOutDesc, inputDesc, normalizedShape, meanDesc, (aclTensor*)nullptr, weightDesc, biasDesc, outputmask),
        OUTPUT(gradInputOutDesc, gradWeightOutDesc, gradBiasOutDesc));
    // SAMPLE: only test GetWorkspaceSize
    aclRet = ut5.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    // outputmask nullptr
    auto ut8 = OP_API_UT(
        aclnnLayerNormBackward,
        INPUT(
            gradOutDesc, inputDesc, normalizedShape, meanDesc, rstdDesc, weightDesc, biasDesc, (aclBoolArray*)nullptr),
        OUTPUT(gradInputOutDesc, gradWeightOutDesc, gradBiasOutDesc));
    // SAMPLE: only test GetWorkspaceSize
    aclRet = ut8.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    // gradInputOutDesc nullptr
    auto ut9 = OP_API_UT(
        aclnnLayerNormBackward,
        INPUT(gradOutDesc, inputDesc, normalizedShape, meanDesc, rstdDesc, weightDesc, biasDesc, outputmask),
        OUTPUT((aclTensor*)nullptr, gradWeightOutDesc, gradBiasOutDesc));
    // SAMPLE: only test GetWorkspaceSize
    aclRet = ut9.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    // gradWeightOutDesc nullptr
    auto ut10 = OP_API_UT(
        aclnnLayerNormBackward,
        INPUT(gradOutDesc, inputDesc, normalizedShape, meanDesc, rstdDesc, weightDesc, biasDesc, outputmask),
        OUTPUT(gradInputOutDesc, (aclTensor*)nullptr, gradBiasOutDesc));
    // SAMPLE: only test GetWorkspaceSize
    aclRet = ut10.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    // gradBiasOutDesc nullptr
    auto ut11 = OP_API_UT(
        aclnnLayerNormBackward,
        INPUT(gradOutDesc, inputDesc, normalizedShape, meanDesc, rstdDesc, weightDesc, biasDesc, outputmask),
        OUTPUT(gradInputOutDesc, gradWeightOutDesc, (aclTensor*)nullptr));
    // SAMPLE: only test GetWorkspaceSize
    aclRet = ut11.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// outputmask[0] false
TEST_F(l2_layer_norm_backward_test, ascend910B2_aclnnLayerNormBackward_output_mask_first_false)
{
    auto gradOutDesc = TensorDesc({2, 6, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto inputDesc = gradOutDesc;
    auto normalizedShape = IntArrayDesc(vector<int64_t>{5, 6});
    auto meanDesc = TensorDesc({2, 6, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto rstdDesc = TensorDesc({2, 6, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto weightDesc = TensorDesc({5, 6}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto biasDesc = weightDesc;
    auto outputmask = BoolArrayDesc(vector<bool>{false, true, true});

    auto gradInputOutDesc = (aclTensor*)nullptr;
    auto gradWeightOutDesc = TensorDesc({5, 6}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto gradBiasOutDesc = TensorDesc({5, 6}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(
        aclnnLayerNormBackward,
        INPUT(gradOutDesc, inputDesc, normalizedShape, meanDesc, rstdDesc, weightDesc, biasDesc, outputmask),
        OUTPUT(gradInputOutDesc, gradWeightOutDesc, gradBiasOutDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// outputmask[1] false
TEST_F(l2_layer_norm_backward_test, ascend910B2_aclnnLayerNormBackward_output_mask_second_false)
{
    auto gradOutDesc = TensorDesc({2, 6, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto inputDesc = gradOutDesc;
    auto normalizedShape = IntArrayDesc(vector<int64_t>{5, 6});
    auto meanDesc = TensorDesc({2, 6, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto rstdDesc = TensorDesc({2, 6, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto weightDesc = TensorDesc({5, 6}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto biasDesc = weightDesc;
    auto outputmask = BoolArrayDesc(vector<bool>{true, false, true});

    auto gradInputOutDesc = TensorDesc({2, 6, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto gradWeightOutDesc = (aclTensor*)nullptr;
    auto gradBiasOutDesc = TensorDesc({5, 6}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(
        aclnnLayerNormBackward,
        INPUT(gradOutDesc, inputDesc, normalizedShape, meanDesc, rstdDesc, weightDesc, biasDesc, outputmask),
        OUTPUT(gradInputOutDesc, gradWeightOutDesc, gradBiasOutDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// outputmask[2] false
TEST_F(l2_layer_norm_backward_test, ascend910B2_aclnnLayerNormBackward_output_mask_third_false)
{
    auto gradOutDesc = TensorDesc({2, 6, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto inputDesc = gradOutDesc;
    auto normalizedShape = IntArrayDesc(vector<int64_t>{5, 6});
    auto meanDesc = TensorDesc({2, 6, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto rstdDesc = TensorDesc({2, 6, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto weightDesc = TensorDesc({5, 6}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto biasDesc = weightDesc;
    auto outputmask = BoolArrayDesc(vector<bool>{true, true, false});

    auto gradInputOutDesc = TensorDesc({2, 6, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto gradWeightOutDesc = TensorDesc({5, 6}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto gradBiasOutDesc = (aclTensor*)nullptr;

    auto ut = OP_API_UT(
        aclnnLayerNormBackward,
        INPUT(gradOutDesc, inputDesc, normalizedShape, meanDesc, rstdDesc, weightDesc, biasDesc, outputmask),
        OUTPUT(gradInputOutDesc, gradWeightOutDesc, gradBiasOutDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// outputmask all false
TEST_F(l2_layer_norm_backward_test, ascend910B2_aclnnLayerNormBackward_output_mask_all_false)
{
    auto gradOutDesc = TensorDesc({2, 6, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto inputDesc = gradOutDesc;
    auto normalizedShape = IntArrayDesc(vector<int64_t>{5, 6});
    auto meanDesc = TensorDesc({2, 6, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto rstdDesc = TensorDesc({2, 6, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto weightDesc = TensorDesc({5, 6}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto biasDesc = weightDesc;
    auto outputmask = BoolArrayDesc(vector<bool>{false, false, false});

    auto gradInputOutDesc = (aclTensor*)nullptr;
    auto gradWeightOutDesc = (aclTensor*)nullptr;
    auto gradBiasOutDesc = (aclTensor*)nullptr;

    auto ut = OP_API_UT(
        aclnnLayerNormBackward,
        INPUT(gradOutDesc, inputDesc, normalizedShape, meanDesc, rstdDesc, weightDesc, biasDesc, outputmask),
        OUTPUT(gradInputOutDesc, gradWeightOutDesc, gradBiasOutDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// error outputmask len
TEST_F(l2_layer_norm_backward_test, ascend910B2_aclnnLayerNormBackward_error_outputmask_len)
{
    auto gradOutDesc = TensorDesc({2, 6, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto inputDesc = gradOutDesc;
    auto normalizedShape = IntArrayDesc(vector<int64_t>{5, 6});
    auto meanDesc = TensorDesc({2, 6, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto rstdDesc = TensorDesc({2, 6, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto weightDesc = TensorDesc({5, 6}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto biasDesc = weightDesc;
    auto outputmask = BoolArrayDesc(vector<bool>{true, false});

    auto gradInputOutDesc = TensorDesc({2, 6, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto gradWeightOutDesc = (aclTensor*)nullptr;
    auto gradBiasOutDesc = TensorDesc({5, 6}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(
        aclnnLayerNormBackward,
        INPUT(gradOutDesc, inputDesc, normalizedShape, meanDesc, rstdDesc, weightDesc, biasDesc, outputmask),
        OUTPUT(gradInputOutDesc, gradWeightOutDesc, gradBiasOutDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// layer_norm_x_backprop_v3 float16 mix dtype
TEST_F(l2_layer_norm_backward_test, ascend910B2_aclnnLayerNormBackward_x_backprop_v3_float16_mix_dtype)
{
    vector<int64_t> input_shape = {2, 3, 4};
    vector<int64_t> norm_shape = {3, 4};
    vector<int64_t> mean_shape = {2, 1, 1};
    aclDataType x_dtype = ACL_FLOAT16;
    aclDataType dtype = ACL_FLOAT;
    aclFormat in_format = ACL_FORMAT_ND;
    auto gradOutDesc = TensorDesc(input_shape, x_dtype, in_format).ValueRange(-2, 2);
    auto inputDesc = gradOutDesc;
    auto normalizedShape = IntArrayDesc(norm_shape);
    auto meanDesc = TensorDesc(mean_shape, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto rstdDesc = TensorDesc(mean_shape, dtype, ACL_FORMAT_ND).ValueRange(1, 2);
    auto weightDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto biasDesc = weightDesc;
    auto outputmask = BoolArrayDesc(vector<bool>{true, true, true});

    auto gradInputOutDesc = TensorDesc(input_shape, dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto gradWeightOutDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto gradBiasOutDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(
        aclnnLayerNormBackward,
        INPUT(gradOutDesc, inputDesc, normalizedShape, meanDesc, rstdDesc, weightDesc, biasDesc, outputmask),
        OUTPUT(gradInputOutDesc, gradWeightOutDesc, gradBiasOutDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// layer_norm_x_backprop_v3 bfloat16 mix dtype
TEST_F(l2_layer_norm_backward_test, ascend910B2_aclnnLayerNormBackward_x_backprop_v3_bfloat16_mix_dtype)
{
    vector<int64_t> input_shape = {2, 3, 4};
    vector<int64_t> norm_shape = {3, 4};
    vector<int64_t> mean_shape = {2, 1, 1};
    aclDataType x_dtype = ACL_BF16;
    aclDataType dtype = ACL_FLOAT;
    aclFormat in_format = ACL_FORMAT_ND;
    auto gradOutDesc = TensorDesc(input_shape, x_dtype, in_format).ValueRange(-2, 2);
    auto inputDesc = gradOutDesc;
    auto normalizedShape = IntArrayDesc(norm_shape);
    auto meanDesc = TensorDesc(mean_shape, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto rstdDesc = TensorDesc(mean_shape, dtype, ACL_FORMAT_ND).ValueRange(1, 2);
    auto weightDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto biasDesc = weightDesc;
    auto outputmask = BoolArrayDesc(vector<bool>{true, true, true});

    auto gradInputOutDesc = TensorDesc(input_shape, x_dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto gradWeightOutDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto gradBiasOutDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(
        aclnnLayerNormBackward,
        INPUT(gradOutDesc, inputDesc, normalizedShape, meanDesc, rstdDesc, weightDesc, biasDesc, outputmask),
        OUTPUT(gradInputOutDesc, gradWeightOutDesc, gradBiasOutDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// layer_norm_x_backprop_v3 grad,input,mean,rstd is half, and weight is float mix dtype
TEST_F(l2_layer_norm_backward_test, ascend310P_aclnnLayerNormBackward_x_backprop_v3_half_float_mix_dtype)
{
    vector<int64_t> input_shape = {2, 3, 4};
    vector<int64_t> norm_shape = {3, 4};
    vector<int64_t> mean_shape = {2, 1, 1};
    aclDataType x_dtype = ACL_FLOAT16;
    aclDataType dtype = ACL_FLOAT;
    aclFormat in_format = ACL_FORMAT_ND;
    auto gradOutDesc = TensorDesc(input_shape, x_dtype, in_format).ValueRange(-2, 2);
    auto inputDesc = gradOutDesc;
    auto normalizedShape = IntArrayDesc(norm_shape);
    auto meanDesc = TensorDesc(mean_shape, x_dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto rstdDesc = TensorDesc(mean_shape, x_dtype, ACL_FORMAT_ND).ValueRange(1, 2);
    auto weightDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto biasDesc = weightDesc;
    auto outputmask = BoolArrayDesc(vector<bool>{true, true, true});

    auto gradInputOutDesc = TensorDesc(input_shape, x_dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto gradWeightOutDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto gradBiasOutDesc = TensorDesc(norm_shape, dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(
        aclnnLayerNormBackward,
        INPUT(gradOutDesc, inputDesc, normalizedShape, meanDesc, rstdDesc, weightDesc, biasDesc, outputmask),
        OUTPUT(gradInputOutDesc, gradWeightOutDesc, gradBiasOutDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// layer_norm_x_backprop_v3 grad, grad/input/mean/weight dtype is half, rstd dtype is float
TEST_F(l2_layer_norm_backward_test, ascend310P_aclnnLayerNormBackward_x_backprop_v3_half_float_mix_dtype2)
{
    vector<int64_t> input_shape = {2, 3, 4};
    vector<int64_t> norm_shape = {3, 4};
    vector<int64_t> mean_shape = {2, 1, 1};
    aclDataType x_dtype = ACL_FLOAT16;
    aclDataType dtype = ACL_FLOAT;
    aclFormat in_format = ACL_FORMAT_ND;
    auto gradOutDesc = TensorDesc(input_shape, x_dtype, in_format).ValueRange(-2, 2);
    auto inputDesc = gradOutDesc;
    auto normalizedShape = IntArrayDesc(norm_shape);
    auto meanDesc = TensorDesc(mean_shape, x_dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto rstdDesc = TensorDesc(mean_shape, dtype, ACL_FORMAT_ND).ValueRange(1, 2);
    auto weightDesc = TensorDesc(norm_shape, x_dtype, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto biasDesc = weightDesc;
    auto outputmask = BoolArrayDesc(vector<bool>{true, true, true});

    auto gradInputOutDesc = TensorDesc(input_shape, dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto gradWeightOutDesc = TensorDesc(norm_shape, x_dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto gradBiasOutDesc = TensorDesc(norm_shape, x_dtype, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(
        aclnnLayerNormBackward,
        INPUT(gradOutDesc, inputDesc, normalizedShape, meanDesc, rstdDesc, weightDesc, biasDesc, outputmask),
        OUTPUT(gradInputOutDesc, gradWeightOutDesc, gradBiasOutDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// gradWeight and gradBias need cast
TEST_F(l2_layer_norm_backward_test, ascend910D_aclnnLayerNormBackward_float32_nd_float16_nd)
{
    vector<int64_t> input_shape = {9, 2, 2, 2};
    vector<int64_t> norm_shape = {2, 2};
    vector<int64_t> mean_shape = {9, 2, 1, 1};
    aclDataType dtype32 = ACL_FLOAT;
    aclDataType dtype16 = ACL_FLOAT16;
    aclFormat in_format = ACL_FORMAT_ND;
    auto gradOutDesc = TensorDesc(input_shape, dtype32, in_format).ValueRange(-2, 2);
    auto inputDesc = gradOutDesc;
    auto normalizedShape = IntArrayDesc(norm_shape);
    auto meanDesc = TensorDesc(mean_shape, dtype32, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto rstdDesc = TensorDesc(mean_shape, dtype32, ACL_FORMAT_ND).ValueRange(1, 2);
    auto weightDesc = TensorDesc(norm_shape, dtype32, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto biasDesc = weightDesc;
    auto outputmask = BoolArrayDesc(vector<bool>{true, true, true});

    auto gradInputOutDesc = TensorDesc(input_shape, dtype32, in_format).Precision(0.01, 0.01);
    auto gradWeightOutDesc = TensorDesc(norm_shape, dtype16, ACL_FORMAT_ND).Precision(0.01, 0.01);
    auto gradBiasOutDesc = TensorDesc(norm_shape, dtype16, ACL_FORMAT_ND).Precision(0.01, 0.01);

    auto ut = OP_API_UT(
        aclnnLayerNormBackward,
        INPUT(gradOutDesc, inputDesc, normalizedShape, meanDesc, rstdDesc, weightDesc, biasDesc, outputmask),
        OUTPUT(gradInputOutDesc, gradWeightOutDesc, gradBiasOutDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// gradWeight and gradBias not need cast
TEST_F(l2_layer_norm_backward_test, ascend910D_aclnnLayerNormBackward_float32_nd_float32_nd)
{
    vector<int64_t> input_shape = {9, 2, 2, 2};
    vector<int64_t> norm_shape = {2, 2};
    vector<int64_t> mean_shape = {9, 2, 1, 1};
    aclDataType dtype32 = ACL_FLOAT;
    aclFormat in_format = ACL_FORMAT_ND;
    auto gradOutDesc = TensorDesc(input_shape, dtype32, in_format).ValueRange(-2, 2);
    auto inputDesc = gradOutDesc;
    auto normalizedShape = IntArrayDesc(norm_shape);
    auto meanDesc = TensorDesc(mean_shape, dtype32, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto rstdDesc = TensorDesc(mean_shape, dtype32, ACL_FORMAT_ND).ValueRange(1, 2);
    auto weightDesc = TensorDesc(norm_shape, dtype32, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto biasDesc = weightDesc;
    auto outputmask = BoolArrayDesc(vector<bool>{true, true, true});

    auto gradInputOutDesc = TensorDesc(input_shape, dtype32, in_format).Precision(0.01, 0.01);
    auto gradWeightOutDesc = TensorDesc(norm_shape, dtype32, ACL_FORMAT_ND).Precision(0.01, 0.01);
    auto gradBiasOutDesc = TensorDesc(norm_shape, dtype32, ACL_FORMAT_ND).Precision(0.01, 0.01);

    auto ut = OP_API_UT(
        aclnnLayerNormBackward,
        INPUT(gradOutDesc, inputDesc, normalizedShape, meanDesc, rstdDesc, weightDesc, biasDesc, outputmask),
        OUTPUT(gradInputOutDesc, gradWeightOutDesc, gradBiasOutDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// gradWeight and gradBias all mask
TEST_F(l2_layer_norm_backward_test, ascend910D_aclnnLayerNormBackward_float32_nd_float32_nd_all_mask)
{
    vector<int64_t> input_shape = {9, 2, 2, 2};
    vector<int64_t> norm_shape = {2, 2};
    vector<int64_t> mean_shape = {9, 2, 1, 1};
    aclDataType dtype32 = ACL_FLOAT;
    aclFormat in_format = ACL_FORMAT_ND;
    auto gradOutDesc = TensorDesc(input_shape, dtype32, in_format).ValueRange(-2, 2);
    auto inputDesc = gradOutDesc;
    auto normalizedShape = IntArrayDesc(norm_shape);
    auto meanDesc = TensorDesc(mean_shape, dtype32, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto rstdDesc = TensorDesc(mean_shape, dtype32, ACL_FORMAT_ND).ValueRange(1, 2);
    auto weightDesc = TensorDesc(norm_shape, dtype32, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto biasDesc = weightDesc;
    auto outputmask = BoolArrayDesc(vector<bool>{true, false, false});

    auto gradInputOutDesc = TensorDesc(input_shape, dtype32, in_format).Precision(0.01, 0.01);
    auto gradWeightOutDesc = TensorDesc(norm_shape, dtype32, ACL_FORMAT_ND).Precision(0.01, 0.01);
    auto gradBiasOutDesc = TensorDesc(norm_shape, dtype32, ACL_FORMAT_ND).Precision(0.01, 0.01);

    auto ut = OP_API_UT(
        aclnnLayerNormBackward,
        INPUT(gradOutDesc, inputDesc, normalizedShape, meanDesc, rstdDesc, weightDesc, biasDesc, outputmask),
        OUTPUT(gradInputOutDesc, gradWeightOutDesc, gradBiasOutDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// gradBias mask and gradWeight not need cast
TEST_F(l2_layer_norm_backward_test, ascend910D_aclnnLayerNormBackward_float32_nd_float16_nd_all_mask)
{
    vector<int64_t> input_shape = {9, 2, 2, 2};
    vector<int64_t> norm_shape = {2, 2};
    vector<int64_t> mean_shape = {9, 2, 1, 1};
    aclDataType dtype32 = ACL_FLOAT;
    aclDataType dtype16 = ACL_FLOAT16;
    aclFormat in_format = ACL_FORMAT_ND;
    auto gradOutDesc = TensorDesc(input_shape, dtype16, in_format).ValueRange(-2, 2);
    auto inputDesc = gradOutDesc;
    auto normalizedShape = IntArrayDesc(norm_shape);
    auto meanDesc = TensorDesc(mean_shape, dtype32, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto rstdDesc = TensorDesc(mean_shape, dtype32, ACL_FORMAT_ND).ValueRange(1, 2);
    auto weightDesc = TensorDesc(norm_shape, dtype16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto biasDesc = weightDesc;
    auto outputmask = BoolArrayDesc(vector<bool>{true, true, false});

    auto gradInputOutDesc = TensorDesc(input_shape, dtype32, in_format).Precision(0.01, 0.01);
    auto gradWeightOutDesc = TensorDesc(norm_shape, dtype16, ACL_FORMAT_ND).Precision(0.01, 0.01);
    auto gradBiasOutDesc = TensorDesc(norm_shape, dtype16, ACL_FORMAT_ND).Precision(0.01, 0.01);

    auto ut = OP_API_UT(
        aclnnLayerNormBackward,
        INPUT(gradOutDesc, inputDesc, normalizedShape, meanDesc, rstdDesc, weightDesc, biasDesc, outputmask),
        OUTPUT(gradInputOutDesc, gradWeightOutDesc, gradBiasOutDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

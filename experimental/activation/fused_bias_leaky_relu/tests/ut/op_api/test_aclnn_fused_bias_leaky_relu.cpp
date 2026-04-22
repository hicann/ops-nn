/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <vector>
#include <array>
#include "gtest/gtest.h"
#include "../../../op_api/aclnn_fused_bias_leaky_relu.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

class l2_fused_bias_leaky_relu_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "l2_fused_bias_leaky_relu_test SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "l2_fused_bias_leaky_relu_test TearDown" << endl;
    }

public:
    void CommonTest(
        const vector<int64_t>& xShape, const vector<int64_t>& biasShape, const vector<int64_t>& yShape,
        aclDataType xDtype, aclDataType biasDtype, aclDataType yDtype,
        double negativeSlope, double scale, aclnnStatus expectRet)
    {
        auto x = TensorDesc(xShape, xDtype, ACL_FORMAT_ND).ValueRange(-2, 2);
        auto bias = TensorDesc(biasShape, biasDtype, ACL_FORMAT_ND).ValueRange(-1, 1);
        auto y = TensorDesc(yShape, yDtype, ACL_FORMAT_ND);
        uint64_t workspace_size = 0;
        auto ut = OP_API_UT(aclnnFusedBiasLeakyRelu, INPUT(x, bias, negativeSlope, scale), OUTPUT(y));
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        EXPECT_EQ(aclRet, expectRet);
    }
};

TEST_F(l2_fused_bias_leaky_relu_test, ascend950_success_float32)
{
    CommonTest({1024}, {1024}, {1024}, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, 0.2, 1.414213562373, ACLNN_SUCCESS);
}

TEST_F(l2_fused_bias_leaky_relu_test, ascend950_success_float32_2d)
{
    CommonTest({2, 1024}, {2, 1024}, {2, 1024}, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, 0.2, 1.414213562373, ACLNN_SUCCESS);
}

TEST_F(l2_fused_bias_leaky_relu_test, ascend950_success_float32_4d)
{
    CommonTest({2, 4, 8, 16}, {2, 4, 8, 16}, {2, 4, 8, 16}, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, 0.2, 1.414213562373, ACLNN_SUCCESS);
}

TEST_F(l2_fused_bias_leaky_relu_test, ascend950_success_float16)
{
    CommonTest({1024}, {1024}, {1024}, ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16, 0.2, 1.414213562373, ACLNN_SUCCESS);
}

TEST_F(l2_fused_bias_leaky_relu_test, ascend950_success_float16_2d)
{
    CommonTest({8, 1024}, {8, 1024}, {8, 1024}, ACL_FLOAT16, ACL_FLOAT16, ACL_FLOAT16, 0.2, 1.414213562373, ACLNN_SUCCESS);
}

TEST_F(l2_fused_bias_leaky_relu_test, ascend950_success_negative_slope_zero)
{
    CommonTest({1024}, {1024}, {1024}, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, 0.0, 1.0, ACLNN_SUCCESS);
}

TEST_F(l2_fused_bias_leaky_relu_test, ascend950_success_scale_one)
{
    CommonTest({1024}, {1024}, {1024}, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, 0.2, 1.0, ACLNN_SUCCESS);
}

TEST_F(l2_fused_bias_leaky_relu_test, ascend950_success_large_negative_slope)
{
    CommonTest({1024}, {1024}, {1024}, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, 0.5, 1.414213562373, ACLNN_SUCCESS);
}

TEST_F(l2_fused_bias_leaky_relu_test, ascend950_success_empty_tensor)
{
    CommonTest({0}, {0}, {0}, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, 0.2, 1.414213562373, ACLNN_SUCCESS);
}

TEST_F(l2_fused_bias_leaky_relu_test, ascend950_param_invalid_dtype)
{
    CommonTest({1024}, {1024}, {1024}, ACL_INT32, ACL_INT32, ACL_INT32, 0.2, 1.414213562373, ACLNN_ERR_PARAM_INVALID);
    CommonTest({1024}, {1024}, {1024}, ACL_BF16, ACL_BF16, ACL_BF16, 0.2, 1.414213562373, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_fused_bias_leaky_relu_test, ascend950_param_invalid_dtype_mismatch)
{
    CommonTest({1024}, {1024}, {1024}, ACL_FLOAT, ACL_FLOAT16, ACL_FLOAT, 0.2, 1.414213562373, ACLNN_ERR_PARAM_INVALID);
    CommonTest({1024}, {1024}, {1024}, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16, 0.2, 1.414213562373, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_fused_bias_leaky_relu_test, ascend950_param_invalid_shape_mismatch)
{
    CommonTest({1024}, {512}, {1024}, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, 0.2, 1.414213562373, ACLNN_ERR_PARAM_INVALID);
    CommonTest({2, 1024}, {1024}, {2, 1024}, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, 0.2, 1.414213562373, ACLNN_ERR_PARAM_INVALID);
    CommonTest({1024}, {1024}, {512}, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, 0.2, 1.414213562373, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_fused_bias_leaky_relu_test, ascend950_param_nullptr)
{
    auto x = TensorDesc({1024}, ACL_FLOAT, ACL_FORMAT_ND);
    auto bias = TensorDesc({1024}, ACL_FLOAT, ACL_FORMAT_ND);
    auto y = TensorDesc({1024}, ACL_FLOAT, ACL_FORMAT_ND);
    uint64_t workspace_size = 0;

    auto ut1 = OP_API_UT(aclnnFusedBiasLeakyRelu, INPUT(nullptr, bias, 0.2, 1.414213562373), OUTPUT(y));
    EXPECT_EQ(ut1.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_NULLPTR);

    auto ut2 = OP_API_UT(aclnnFusedBiasLeakyRelu, INPUT(x, nullptr, 0.2, 1.414213562373), OUTPUT(y));
    EXPECT_EQ(ut2.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_NULLPTR);

    auto ut3 = OP_API_UT(aclnnFusedBiasLeakyRelu, INPUT(x, bias, 0.2, 1.414213562373), OUTPUT(nullptr));
    EXPECT_EQ(ut3.TestGetWorkspaceSize(&workspace_size), ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_fused_bias_leaky_relu_test, ascend950_max_dim)
{
    CommonTest({1, 2, 3, 4, 5, 6, 7, 8}, {1, 2, 3, 4, 5, 6, 7, 8}, {1, 2, 3, 4, 5, 6, 7, 8}, 
               ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, 0.2, 1.414213562373, ACLNN_SUCCESS);
}

TEST_F(l2_fused_bias_leaky_relu_test, ascend950_dim_overflow)
{
    CommonTest({1, 2, 3, 4, 5, 6, 7, 8, 9}, {1, 2, 3, 4, 5, 6, 7, 8, 9}, {1, 2, 3, 4, 5, 6, 7, 8, 9}, 
               ACL_FLOAT, ACL_FLOAT, ACL_FLOAT, 0.2, 1.414213562373, ACLNN_ERR_PARAM_INVALID);
}
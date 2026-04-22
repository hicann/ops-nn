/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/
#include <cfloat>

#include <array>
#include <vector>

#include "gtest/gtest.h"
#include "../../../op_api/aclnn_rotate_quant.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace op;
using namespace std;

class l2_rotate_quant_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "l2_rotate_quant_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "l2_rotate_quant_test TearDown" << endl;
    }
};

// x 为 nullptr
TEST_F(l2_rotate_quant_test, ascend910B1_rotate_quant_x_nullptr_fail)
{
    TensorDesc rot_desc = TensorDesc({64, 64}, ACL_BF16, ACL_FORMAT_ND);
    float alpha = 1.0f;
    TensorDesc y_desc = TensorDesc({4, 64}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnRotateQuant, INPUT((aclTensor*)nullptr, rot_desc, alpha), OUTPUT(y_desc, scale_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_NE(aclRet, ACLNN_SUCCESS);
}

// rot 为 nullptr
TEST_F(l2_rotate_quant_test, ascend910B1_rotate_quant_rot_nullptr_fail)
{
    TensorDesc x_desc = TensorDesc({4, 64}, ACL_BF16, ACL_FORMAT_ND);
    float alpha = 1.0f;
    TensorDesc y_desc = TensorDesc({4, 64}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnRotateQuant, INPUT(x_desc, (aclTensor*)nullptr, alpha), OUTPUT(y_desc, scale_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_NE(aclRet, ACLNN_SUCCESS);
}

// y 为 nullptr
TEST_F(l2_rotate_quant_test, ascend910B1_rotate_quant_y_nullptr_fail)
{
    TensorDesc x_desc = TensorDesc({4, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc rot_desc = TensorDesc({64, 64}, ACL_BF16, ACL_FORMAT_ND);
    float alpha = 1.0f;
    TensorDesc scale_desc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnRotateQuant, INPUT(x_desc, rot_desc, alpha), OUTPUT((aclTensor*)nullptr, scale_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_NE(aclRet, ACLNN_SUCCESS);
}

// scale 为 nullptr
TEST_F(l2_rotate_quant_test, ascend910B1_rotate_quant_scale_nullptr_fail)
{
    TensorDesc x_desc = TensorDesc({4, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc rot_desc = TensorDesc({64, 64}, ACL_BF16, ACL_FORMAT_ND);
    float alpha = 1.0f;
    TensorDesc y_desc = TensorDesc({4, 64}, ACL_INT8, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnRotateQuant, INPUT(x_desc, rot_desc, alpha), OUTPUT(y_desc, (aclTensor*)nullptr));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_NE(aclRet, ACLNN_SUCCESS);
}

// x 和 rot 数据类型不一致
TEST_F(l2_rotate_quant_test, ascend910B1_rotate_quant_dtype_mismatch_fail)
{
    TensorDesc x_desc = TensorDesc({4, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc rot_desc = TensorDesc({64, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    float alpha = 1.0f;
    TensorDesc y_desc = TensorDesc({4, 64}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnRotateQuant, INPUT(x_desc, rot_desc, alpha), OUTPUT(y_desc, scale_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_NE(aclRet, ACLNN_SUCCESS);
}

// x 数据类型不支持 (FP32)
TEST_F(l2_rotate_quant_test, ascend910B1_rotate_quant_x_dtype_unsupported_fail)
{
    TensorDesc x_desc = TensorDesc({4, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc rot_desc = TensorDesc({64, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    float alpha = 1.0f;
    TensorDesc y_desc = TensorDesc({4, 64}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnRotateQuant, INPUT(x_desc, rot_desc, alpha), OUTPUT(y_desc, scale_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_NE(aclRet, ACLNN_SUCCESS);
}

// y 数据类型不支持 (BF16)
TEST_F(l2_rotate_quant_test, ascend910B1_rotate_quant_y_dtype_unsupported_fail)
{
    TensorDesc x_desc = TensorDesc({4, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc rot_desc = TensorDesc({64, 64}, ACL_BF16, ACL_FORMAT_ND);
    float alpha = 1.0f;
    TensorDesc y_desc = TensorDesc({4, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnRotateQuant, INPUT(x_desc, rot_desc, alpha), OUTPUT(y_desc, scale_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_NE(aclRet, ACLNN_SUCCESS);
}

// scale 数据类型不支持 (BF16 instead of FP32)
TEST_F(l2_rotate_quant_test, ascend910B1_rotate_quant_scale_dtype_unsupported_fail)
{
    TensorDesc x_desc = TensorDesc({4, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc rot_desc = TensorDesc({64, 64}, ACL_BF16, ACL_FORMAT_ND);
    float alpha = 1.0f;
    TensorDesc y_desc = TensorDesc({4, 64}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({4}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnRotateQuant, INPUT(x_desc, rot_desc, alpha), OUTPUT(y_desc, scale_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_NE(aclRet, ACLNN_SUCCESS);
}

// rot 不是方阵
TEST_F(l2_rotate_quant_test, ascend910B1_rotate_quant_rot_not_square_fail)
{
    TensorDesc x_desc = TensorDesc({4, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc rot_desc = TensorDesc({64, 32}, ACL_BF16, ACL_FORMAT_ND);
    float alpha = 1.0f;
    TensorDesc y_desc = TensorDesc({4, 64}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnRotateQuant, INPUT(x_desc, rot_desc, alpha), OUTPUT(y_desc, scale_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_NE(aclRet, ACLNN_SUCCESS);
}

// x 维度不是2D
TEST_F(l2_rotate_quant_test, ascend910B1_rotate_quant_x_dim_fail)
{
    TensorDesc x_desc = TensorDesc({2, 4, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc rot_desc = TensorDesc({64, 64}, ACL_BF16, ACL_FORMAT_ND);
    float alpha = 1.0f;
    TensorDesc y_desc = TensorDesc({2, 4, 64}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnRotateQuant, INPUT(x_desc, rot_desc, alpha), OUTPUT(y_desc, scale_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_NE(aclRet, ACLNN_SUCCESS);
}

// rot 维度不是2D
TEST_F(l2_rotate_quant_test, ascend910B1_rotate_quant_rot_dim_fail)
{
    TensorDesc x_desc = TensorDesc({4, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc rot_desc = TensorDesc({1, 64, 64}, ACL_BF16, ACL_FORMAT_ND);
    float alpha = 1.0f;
    TensorDesc y_desc = TensorDesc({4, 64}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnRotateQuant, INPUT(x_desc, rot_desc, alpha), OUTPUT(y_desc, scale_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_NE(aclRet, ACLNN_SUCCESS);
}

// y shape 与 x shape 不一致
TEST_F(l2_rotate_quant_test, ascend910B1_rotate_quant_y_shape_fail)
{
    TensorDesc x_desc = TensorDesc({4, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc rot_desc = TensorDesc({64, 64}, ACL_BF16, ACL_FORMAT_ND);
    float alpha = 1.0f;
    TensorDesc y_desc = TensorDesc({4, 128}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnRotateQuant, INPUT(x_desc, rot_desc, alpha), OUTPUT(y_desc, scale_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_NE(aclRet, ACLNN_SUCCESS);
}

// scale shape 与 M 不一致
TEST_F(l2_rotate_quant_test, ascend910B1_rotate_quant_scale_shape_fail)
{
    TensorDesc x_desc = TensorDesc({4, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc rot_desc = TensorDesc({64, 64}, ACL_BF16, ACL_FORMAT_ND);
    float alpha = 1.0f;
    TensorDesc y_desc = TensorDesc({4, 64}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnRotateQuant, INPUT(x_desc, rot_desc, alpha), OUTPUT(y_desc, scale_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_NE(aclRet, ACLNN_SUCCESS);
}

// x N 不能被 rot K 整除
TEST_F(l2_rotate_quant_test, ascend910B1_rotate_quant_n_not_divisible_fail)
{
    TensorDesc x_desc = TensorDesc({4, 100}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc rot_desc = TensorDesc({64, 64}, ACL_BF16, ACL_FORMAT_ND);
    float alpha = 1.0f;
    TensorDesc y_desc = TensorDesc({4, 100}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnRotateQuant, INPUT(x_desc, rot_desc, alpha), OUTPUT(y_desc, scale_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_NE(aclRet, ACLNN_SUCCESS);
}

// empty tensor
TEST_F(l2_rotate_quant_test, ascend910B1_rotate_quant_empty_tensor)
{
    TensorDesc x_desc = TensorDesc({0, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc rot_desc = TensorDesc({64, 64}, ACL_BF16, ACL_FORMAT_ND);
    float alpha = 1.0f;
    TensorDesc y_desc = TensorDesc({0, 64}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnRotateQuant, INPUT(x_desc, rot_desc, alpha), OUTPUT(y_desc, scale_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// SoC 版本错误
TEST_F(l2_rotate_quant_test, ascend910B2_rotate_quant_bf16_int8_fail)
{
    TensorDesc x_desc = TensorDesc({4, 64}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc rot_desc = TensorDesc({64, 64}, ACL_BF16, ACL_FORMAT_ND);
    float alpha = 1.0f;
    TensorDesc y_desc = TensorDesc({4, 64}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnRotateQuant, INPUT(x_desc, rot_desc, alpha), OUTPUT(y_desc, scale_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}

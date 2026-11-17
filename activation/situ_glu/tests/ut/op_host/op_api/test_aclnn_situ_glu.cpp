/**
 * Copyright (c) 2026, Huawei Technologies Co., Ltd.
 * All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <gtest/gtest.h>
#include <cstdint>
#include <limits>
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "../../../../op_host/op_api/aclnn_situ_glu.h"

class l2_situ_glu_test : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

// 正常用例：fp32，dim=1
TEST_F(l2_situ_glu_test, Ascend910B_f32_normal_dim1)
{
    auto tensor_x = TensorDesc({2, 6}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dim = 1;
    double beta = 1.0;
    double linearBeta = 0.0;
    bool activateLeft = true;
    auto ut = OP_API_UT(aclnnSituGlu, INPUT(tensor_x, dim, beta, linearBeta, activateLeft), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常用例：fp16，dim 负数索引
TEST_F(l2_situ_glu_test, Ascend910B_f16_normal_neg_dim)
{
    auto tensor_x = TensorDesc({2, 4, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({2, 2, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
    int64_t dim = -2;
    double beta = 1.0;
    double linearBeta = 0.0;
    bool activateLeft = true;
    auto ut = OP_API_UT(aclnnSituGlu, INPUT(tensor_x, dim, beta, linearBeta, activateLeft), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常用例：bf16
TEST_F(l2_situ_glu_test, Ascend910B_bf16_normal)
{
    auto tensor_x = TensorDesc({4, 8}, ACL_BF16, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({4, 4}, ACL_BF16, ACL_FORMAT_ND);
    int64_t dim = -1;
    double beta = 1.0;
    double linearBeta = 0.0;
    bool activateLeft = true;
    auto ut = OP_API_UT(aclnnSituGlu, INPUT(tensor_x, dim, beta, linearBeta, activateLeft), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常用例：linearBeta=1.0、activateLeft=false
TEST_F(l2_situ_glu_test, Ascend910B_f32_normal_linear_beta)
{
    auto tensor_x = TensorDesc({2, 6}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dim = 1;
    double beta = 1.0;
    double linearBeta = 1.0;
    bool activateLeft = false;
    auto ut = OP_API_UT(aclnnSituGlu, INPUT(tensor_x, dim, beta, linearBeta, activateLeft), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常用例：x 为空 Tensor，out 同为空 Tensor
TEST_F(l2_situ_glu_test, Ascend910B_f32_normal_empty_x)
{
    auto tensor_x = TensorDesc({2, 0, 6}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({2, 0, 6}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dim = -2;
    double beta = 1.0;
    double linearBeta = 0.0;
    bool activateLeft = true;
    auto ut = OP_API_UT(aclnnSituGlu, INPUT(tensor_x, dim, beta, linearBeta, activateLeft), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常用例：x 非连续，out 非连续
TEST_F(l2_situ_glu_test, Ascend910B_f32_normal_noncontiguous)
{
    auto tensor_x = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2});
    auto out_tensor_desc = TensorDesc({1, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dim = 0;
    double beta = 1.0;
    double linearBeta = 0.0;
    bool activateLeft = true;
    auto ut = OP_API_UT(aclnnSituGlu, INPUT(tensor_x, dim, beta, linearBeta, activateLeft), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常用例：out 为非连续视图（行间隙），ViewCopy 按视图布局搬运
TEST_F(l2_situ_glu_test, Ascend910B_f32_normal_noncontiguous_out)
{
    auto tensor_x = TensorDesc({2, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND, {8, 1}, 0, {2, 8});
    int64_t dim = -1;
    double beta = 1.0;
    double linearBeta = 0.0;
    bool activateLeft = true;
    auto ut = OP_API_UT(aclnnSituGlu, INPUT(tensor_x, dim, beta, linearBeta, activateLeft), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 异常用例：x 为空指针
TEST_F(l2_situ_glu_test, Ascend910B_f32_abnormal_null_x)
{
    auto tensor_x = nullptr;
    auto out_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dim = 1;
    double beta = 1.0;
    double linearBeta = 0.0;
    bool activateLeft = true;
    auto ut = OP_API_UT(aclnnSituGlu, INPUT(tensor_x, dim, beta, linearBeta, activateLeft), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 异常用例：out 为空指针
TEST_F(l2_situ_glu_test, Ascend910B_f32_abnormal_null_out)
{
    auto tensor_x = TensorDesc({2, 6}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = nullptr;
    int64_t dim = 1;
    double beta = 1.0;
    double linearBeta = 0.0;
    bool activateLeft = true;
    auto ut = OP_API_UT(aclnnSituGlu, INPUT(tensor_x, dim, beta, linearBeta, activateLeft), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 异常用例：workspaceSize 为空指针
TEST_F(l2_situ_glu_test, Ascend910B_f32_abnormal_null_workspace_size)
{
    auto tensor_x = TensorDesc({2, 6}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dim = 1;
    double beta = 1.0;
    double linearBeta = 0.0;
    bool activateLeft = true;
    auto ut = OP_API_UT(aclnnSituGlu, INPUT(tensor_x, dim, beta, linearBeta, activateLeft), OUTPUT(out_tensor_desc));
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(nullptr);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 异常用例：x 数据类型不支持
TEST_F(l2_situ_glu_test, Ascend910B_abnormal_dtype_int64)
{
    auto tensor_x = TensorDesc({2, 6}, ACL_INT64, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND);
    int64_t dim = 1;
    double beta = 1.0;
    double linearBeta = 0.0;
    bool activateLeft = true;
    auto ut = OP_API_UT(aclnnSituGlu, INPUT(tensor_x, dim, beta, linearBeta, activateLeft), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常用例：out 数据类型与 x 不一致
TEST_F(l2_situ_glu_test, Ascend910B_f32_abnormal_dtype_mismatch)
{
    auto tensor_x = TensorDesc({2, 6}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    int64_t dim = 1;
    double beta = 1.0;
    double linearBeta = 0.0;
    bool activateLeft = true;
    auto ut = OP_API_UT(aclnnSituGlu, INPUT(tensor_x, dim, beta, linearBeta, activateLeft), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常用例：dim 超出取值范围
TEST_F(l2_situ_glu_test, Ascend910B_f32_abnormal_dim_out_of_range)
{
    auto tensor_x = TensorDesc({2, 6}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dim = 2;
    double beta = 1.0;
    double linearBeta = 0.0;
    bool activateLeft = true;
    auto ut = OP_API_UT(aclnnSituGlu, INPUT(tensor_x, dim, beta, linearBeta, activateLeft), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常用例：dim 维度大小不能被2整除
TEST_F(l2_situ_glu_test, Ascend910B_f32_abnormal_odd_split)
{
    auto tensor_x = TensorDesc({2, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({2, 4, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dim = 2;
    double beta = 1.0;
    double linearBeta = 0.0;
    bool activateLeft = true;
    auto ut = OP_API_UT(aclnnSituGlu, INPUT(tensor_x, dim, beta, linearBeta, activateLeft), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常用例：out shape 与期望 shape 不一致
TEST_F(l2_situ_glu_test, Ascend910B_f32_abnormal_out_shape_mismatch)
{
    auto tensor_x = TensorDesc({2, 6}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dim = 1;
    double beta = 1.0;
    double linearBeta = 0.0;
    bool activateLeft = true;
    auto ut = OP_API_UT(aclnnSituGlu, INPUT(tensor_x, dim, beta, linearBeta, activateLeft), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常用例：beta 为 0
TEST_F(l2_situ_glu_test, Ascend910B_f32_abnormal_beta_zero)
{
    auto tensor_x = TensorDesc({2, 6}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dim = 1;
    double beta = 0.0;
    double linearBeta = 0.0;
    bool activateLeft = true;
    auto ut = OP_API_UT(aclnnSituGlu, INPUT(tensor_x, dim, beta, linearBeta, activateLeft), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常用例：beta 为非有限值
TEST_F(l2_situ_glu_test, Ascend910B_f32_abnormal_beta_inf)
{
    auto tensor_x = TensorDesc({2, 6}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dim = 1;
    double beta = std::numeric_limits<double>::infinity();
    double linearBeta = 0.0;
    bool activateLeft = true;
    auto ut = OP_API_UT(aclnnSituGlu, INPUT(tensor_x, dim, beta, linearBeta, activateLeft), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常用例：linearBeta 为非有限值
TEST_F(l2_situ_glu_test, Ascend910B_f32_abnormal_linear_beta_inf)
{
    auto tensor_x = TensorDesc({2, 6}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t dim = 1;
    double beta = 1.0;
    double linearBeta = std::numeric_limits<double>::infinity();
    bool activateLeft = true;
    auto ut = OP_API_UT(aclnnSituGlu, INPUT(tensor_x, dim, beta, linearBeta, activateLeft), OUTPUT(out_tensor_desc));
    uint64_t workspace_size = 5;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

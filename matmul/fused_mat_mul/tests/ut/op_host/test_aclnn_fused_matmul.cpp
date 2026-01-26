/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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


#include "../../../op_host/op_api/aclnn_fused_matmul.h"
#include "op_api/op_api_def.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"


using namespace std;
using namespace op;

class l2_fusedmatmul_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    cout << "l2_fusedmatmul_test SetUp" << endl;
  }
  static void TearDownTestCase() {
    cout << "l2_fusedmatmul_test TearDown" << endl;
  }
};

TEST_F(l2_fusedmatmul_test, ascend910_95_test_middle_shape_fp16)
{
    TensorDesc x1_desc = TensorDesc({4, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({32, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc c_desc = TensorDesc({4, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({4, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    int8_t cubeMathType = 0;
    auto ut = OP_API_UT(aclnnFusedMatmul, INPUT(x1_desc, x2_desc, (aclTensor*)nullptr, c_desc, "add", cubeMathType),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_fusedmatmul_test, ascend910_95_test_hf32_bias)
{
    TensorDesc x1_desc = TensorDesc({128, 512}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({512, 512}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc bias_desc = TensorDesc({512}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({128, 512}, ACL_FLOAT, ACL_FORMAT_ND);
    int8_t cubeMathType = op::USE_HF32;
    auto ut = OP_API_UT(aclnnFusedMatmul, INPUT(x1_desc, x2_desc, bias_desc, (aclTensor*)nullptr, "", cubeMathType),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_fusedmatmul_test, ascend910_95_test_f32_relu)
{
    TensorDesc x1_desc = TensorDesc({128, 512}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({512, 512}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc bias_desc = TensorDesc({512}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({128, 512}, ACL_FLOAT, ACL_FORMAT_ND);
    int8_t cubeMathType = op::KEEP_DTYPE;
    auto ut = OP_API_UT(aclnnFusedMatmul, INPUT(x1_desc, x2_desc, bias_desc, (aclTensor*)nullptr, "relu", cubeMathType),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_fusedmatmul_test, ascend910_95_test_bias_f16_f32)
{
    TensorDesc x1_desc = TensorDesc({128, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({512, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias_desc = TensorDesc({512}, ACL_FLOAT, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({128, 512}, ACL_FLOAT16, ACL_FORMAT_ND);
    int8_t cubeMathType = 0;
    auto ut = OP_API_UT(aclnnFusedMatmul, INPUT(x1_desc, x2_desc, bias_desc, (aclTensor*)nullptr, "relu", cubeMathType),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_fusedmatmul_test, ascend910_95_test_middle_shape_fp16_failed_1)
{
    TensorDesc x1_desc = TensorDesc({4, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({32, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc c_desc = TensorDesc({4, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({4, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    int8_t cubeMathType = 6;
    auto ut = OP_API_UT(aclnnFusedMatmul, INPUT(x1_desc, x2_desc, (aclTensor*)nullptr, c_desc, "add", cubeMathType),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_fusedmatmul_test, ascend910_95_test_not_support_bias_failed)
{
    TensorDesc x1_desc = TensorDesc({4, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({32, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc bias_desc = TensorDesc({64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc c_desc = TensorDesc({4, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({4, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    int8_t cubeMathType = 0;
    auto ut = OP_API_UT(aclnnFusedMatmul, INPUT(x1_desc, x2_desc, bias_desc, c_desc, "add", cubeMathType),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_fusedmatmul_test, ascend910_95_test_middle_shape_fp16_failed_3)
{
    TensorDesc x1_desc = TensorDesc({4, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc x2_desc = TensorDesc({32, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc c_desc = TensorDesc({4, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc out_desc = TensorDesc({4, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    int8_t cubeMathType = 0;
    auto ut = OP_API_UT(aclnnFusedMatmul, INPUT(x1_desc, x2_desc, (aclTensor*)nullptr, c_desc, "notsupport", cubeMathType),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}




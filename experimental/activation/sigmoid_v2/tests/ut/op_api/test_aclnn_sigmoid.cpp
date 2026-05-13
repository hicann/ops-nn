/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"

#include "../../../op_host/op_api/aclnn_sigmoid.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace op;

class l2_sigmoid_test : public testing::Test {
 protected:
  static void SetUpTestCase()
  {
    std::cout << "sigmoid_test SetUp" << std::endl;
  }

  static void TearDownTestCase()
  {
    std::cout << "sigmoid_test TearDown" << std::endl;
  }
};

TEST_F(l2_sigmoid_test, case_001_float)
{
  auto selfDesc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-8, 8);
  auto outDesc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnSigmoid, INPUT(selfDesc), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACL_SUCCESS);
  ut.TestPrecision();
}

TEST_F(l2_sigmoid_test, case_002_float16)
{
  auto selfDesc = TensorDesc({2, 3, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-8, 8);
  auto outDesc = TensorDesc({2, 3, 4}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);

  auto ut = OP_API_UT(aclnnSigmoid, INPUT(selfDesc), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACL_SUCCESS);
  ut.TestPrecision();
}

TEST_F(l2_sigmoid_test, case_003_bfloat16)
{
  auto selfDesc = TensorDesc({2, 3, 4}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-8, 8);
  auto outDesc = TensorDesc({2, 3, 4}, ACL_BF16, ACL_FORMAT_ND).Precision(0.01, 0.01);

  auto ut = OP_API_UT(aclnnSigmoid, INPUT(selfDesc), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACL_SUCCESS);
  ut.TestPrecision();
}

TEST_F(l2_sigmoid_test, case_004_inplace_float16)
{
  auto selfDesc = TensorDesc({2, 3, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-8, 8);
  auto ut = OP_API_UT(aclnnInplaceSigmoid, INPUT(selfDesc), OUTPUT());
  uint64_t workspaceSize = 0;
  EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACL_SUCCESS);
  ut.TestPrecision();
}

TEST_F(l2_sigmoid_test, case_005_inplace_not_contiguous)
{
  auto selfDesc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(-8, 8);
  auto ut = OP_API_UT(aclnnInplaceSigmoid, INPUT(selfDesc), OUTPUT());
  uint64_t workspaceSize = 0;
  EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACL_SUCCESS);
  ut.TestPrecision();
}

TEST_F(l2_sigmoid_test, case_006_inplace_empty_tensor)
{
  auto selfDesc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnInplaceSigmoid, INPUT(selfDesc), OUTPUT());
  uint64_t workspaceSize = 0;
  EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACL_SUCCESS);
  ut.TestPrecision();
}

TEST_F(l2_sigmoid_test, case_007_empty_tensor)
{
  auto selfDesc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
  auto outDesc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnSigmoid, INPUT(selfDesc), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACL_SUCCESS);
  ut.TestPrecision();
}

TEST_F(l2_sigmoid_test, case_008_not_contiguous)
{
  auto selfDesc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(-8, 8);
  auto outDesc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnSigmoid, INPUT(selfDesc), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACL_SUCCESS);
  ut.TestPrecision();
}

TEST_F(l2_sigmoid_test, case_009_nullptr)
{
  auto tensorDesc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
  uint64_t workspaceSize = 0;

  auto ut1 = OP_API_UT(aclnnSigmoid, INPUT(nullptr), OUTPUT(tensorDesc));
  EXPECT_EQ(ut1.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_NULLPTR);

  auto ut2 = OP_API_UT(aclnnSigmoid, INPUT(tensorDesc), OUTPUT(nullptr));
  EXPECT_EQ(ut2.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_sigmoid_test, case_010_invalid_input_dtype)
{
  auto selfDesc = TensorDesc({2, 3, 4}, ACL_INT32, ACL_FORMAT_ND);
  auto outDesc = TensorDesc({2, 3, 4}, ACL_INT32, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnSigmoid, INPUT(selfDesc), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_sigmoid_test, case_011_dtype_mismatch)
{
  auto selfDesc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
  auto outDesc = TensorDesc({2, 3, 4}, ACL_FLOAT16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnSigmoid, INPUT(selfDesc), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_sigmoid_test, case_012_shape_mismatch)
{
  auto selfDesc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
  auto outDesc = TensorDesc({2, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnSigmoid, INPUT(selfDesc), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_sigmoid_test, case_013_max_dim)
{
  auto selfDesc = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8, 9}, ACL_FLOAT, ACL_FORMAT_ND);
  auto outDesc = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8, 9}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnSigmoid, INPUT(selfDesc), OUTPUT(outDesc));
  uint64_t workspaceSize = 0;
  EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_sigmoid_test, case_014_inplace_float)
{
  auto selfDesc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-8, 8);

  auto ut = OP_API_UT(aclnnInplaceSigmoid, INPUT(selfDesc), OUTPUT());
  uint64_t workspaceSize = 0;
  EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACL_SUCCESS);
  ut.TestPrecision();
}

TEST_F(l2_sigmoid_test, case_015_inplace_bfloat16)
{
  auto selfDesc = TensorDesc({2, 3, 4}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-8, 8);

  auto ut = OP_API_UT(aclnnInplaceSigmoid, INPUT(selfDesc), OUTPUT());
  uint64_t workspaceSize = 0;
  EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACL_SUCCESS);
  ut.TestPrecision();
}

TEST_F(l2_sigmoid_test, case_016_inplace_invalid_dtype)
{
  auto selfDesc = TensorDesc({2, 3, 4}, ACL_INT64, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnInplaceSigmoid, INPUT(selfDesc), OUTPUT());
  uint64_t workspaceSize = 0;
  EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_sigmoid_test, case_017_inplace_nullptr)
{
  uint64_t workspaceSize = 0;
  auto ut = OP_API_UT(aclnnInplaceSigmoid, INPUT(static_cast<aclTensor *>(nullptr)), OUTPUT());
  EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_sigmoid_test, case_018_inplace_max_dim)
{
  auto selfDesc = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8, 9}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnInplaceSigmoid, INPUT(selfDesc), OUTPUT());
  uint64_t workspaceSize = 0;
  EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

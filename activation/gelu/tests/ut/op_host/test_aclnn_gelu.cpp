/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "../../../op_host/op_api/aclnn_gelu.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"

class l2GeluTest : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "l2GeluTest SetUp" << std::endl;
  }

  static void TearDownTestCase() { std::cout << "l2GeluTest TearDown" << std::endl; }
};

// self为空
TEST_F(l2GeluTest, l2_gelu_test_001) {
  auto outDesc = TensorDesc({2, 4, 6, 7}, ACL_FLOAT16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnGelu, INPUT((aclTensor*)nullptr), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

// out为空
TEST_F(l2GeluTest, l2_gelu_test_002) {
  auto selfDesc = TensorDesc({2, 4, 6, 7}, ACL_FLOAT16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnGelu, INPUT(selfDesc), OUTPUT((aclTensor*)nullptr));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

// self和out的数据类型不一致
TEST_F(l2GeluTest, l2_gelu_test_003) {
  auto selfDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
  auto outDesc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnGelu, INPUT(selfDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// self数据类型不在支持范围内
TEST_F(l2GeluTest, l2_gelu_test_004) {
  auto selfDesc = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND);
  auto outDesc = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnGelu, INPUT(selfDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// self和out的shape不一致
TEST_F(l2GeluTest, l2_gelu_test_005) {
  auto selfDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
  auto outDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnGelu, INPUT(selfDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// self和out的shape不一致
TEST_F(l2GeluTest, l2_gelu_test_006) {
  auto selfDesc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
  auto outDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnGelu, INPUT(selfDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// self数据类型不在支持范围内
TEST_F(l2GeluTest, l2_gelu_test_007) {
  auto selfDesc = TensorDesc({2, 3}, ACL_DT_UNDEFINED, ACL_FORMAT_ND);
  auto outDesc = TensorDesc({2, 3}, ACL_DT_UNDEFINED, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnGelu, INPUT(selfDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// format NCHW、NHWC
TEST_F(l2GeluTest, l2_gelu_test_008) {
  auto selfDesc = TensorDesc({2, 4, 6, 7}, ACL_FLOAT, ACL_FORMAT_NCHW);
  auto outDesc = TensorDesc({2, 4, 6, 7}, ACL_FLOAT, ACL_FORMAT_NHWC)
          .Precision(0.01, 0.01);

  auto ut = OP_API_UT(aclnnGelu, INPUT(selfDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);

  ut.TestPrecision();
}

// format NDHWC、NCDHW
TEST_F(l2GeluTest, l2_gelu_test_009) {
  auto selfDesc = TensorDesc({2, 4, 6, 7, 8}, ACL_FLOAT16, ACL_FORMAT_NDHWC);
  auto outDesc = TensorDesc({2, 4, 6, 7, 8}, ACL_FLOAT16, ACL_FORMAT_NCDHW)
          .Precision(0.01, 0.01);

  auto ut = OP_API_UT(aclnnGelu, INPUT(selfDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);

  ut.TestPrecision();
}

// format HWCN
TEST_F(l2GeluTest, l2_gelu_test_010) {
  auto selfDesc = TensorDesc({2, 4, 6, 8}, ACL_FLOAT, ACL_FORMAT_HWCN);
  auto outDesc = TensorDesc({2, 4, 6, 8}, ACL_FLOAT, ACL_FORMAT_HWCN)
          .Precision(0.01, 0.01);

  auto ut = OP_API_UT(aclnnGelu, INPUT(selfDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);

  ut.TestPrecision();
}

// format self是私有格式
TEST_F(l2GeluTest, l2_gelu_test_011) {
  auto selfDesc = TensorDesc({2, 4, 6, 8, 5}, ACL_FLOAT, ACL_FORMAT_NC1HWC0);
  auto outDesc = TensorDesc({2, 4, 6, 8}, ACL_FLOAT, ACL_FORMAT_HWCN);

  auto ut = OP_API_UT(aclnnGelu, INPUT(selfDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// format out是私有格式
TEST_F(l2GeluTest, l2_gelu_test_012) {
  auto selfDesc = TensorDesc({2, 4, 6, 8}, ACL_FLOAT, ACL_FORMAT_HWCN);
  auto outDesc = TensorDesc({2, 4, 6, 8, 8}, ACL_FLOAT, ACL_FORMAT_NC1HWC0);

  auto ut = OP_API_UT(aclnnGelu, INPUT(selfDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// float16 正常路径
TEST_F(l2GeluTest, l2_gelu_test_013) {
  auto selfDesc = TensorDesc({2, 4, 6, 8}, ACL_FLOAT, ACL_FORMAT_ND);
  auto outDesc = TensorDesc({2, 4, 6, 8}, ACL_FLOAT, ACL_FORMAT_ND)
          .Precision(0.01, 0.01);

  auto ut = OP_API_UT(aclnnGelu, INPUT(selfDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);

  ut.TestPrecision();
}

// float32 正常路径
TEST_F(l2GeluTest, l2_gelu_test_014) {
  auto selfDesc = TensorDesc({2, 4, 6, 8, 4, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto outDesc = TensorDesc({2, 4, 6, 8, 4, 5}, ACL_FLOAT16, ACL_FORMAT_ND)
          .Precision(0.01, 0.01);

  auto ut = OP_API_UT(aclnnGelu, INPUT(selfDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);

  ut.TestPrecision();
}

// 数据范围[-1, 1] float16
TEST_F(l2GeluTest, l2_gelu_test_016) {
  auto selfDesc = TensorDesc({2, 4, 6, 8, 4, 5}, ACL_FLOAT16, ACL_FORMAT_ND)
          .ValueRange(-1, 1);
  auto outDesc = TensorDesc({2, 4, 6, 8, 4, 5}, ACL_FLOAT16, ACL_FORMAT_ND)
          .Precision(0.01, 0.01);

  auto ut = OP_API_UT(aclnnGelu, INPUT(selfDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);

  ut.TestPrecision();
}

// 数据范围[-1, 1] float32
TEST_F(l2GeluTest, l2_gelu_test_017) {
  auto selfDesc = TensorDesc({2, 4, 6, 8, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND)
          .ValueRange(-1, 1);
  auto outDesc = TensorDesc({2, 4, 6, 8, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND)
          .Precision(0.01, 0.01);

  auto ut = OP_API_UT(aclnnGelu, INPUT(selfDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);

  ut.TestPrecision();
}

// 非连续
TEST_F(l2GeluTest, l2_gelu_test_018) {
  auto selfDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2});
  auto outDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2})
          .Precision(0.01, 0.01);

  auto ut = OP_API_UT(aclnnGelu, INPUT(selfDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);

  ut.TestPrecision();
}
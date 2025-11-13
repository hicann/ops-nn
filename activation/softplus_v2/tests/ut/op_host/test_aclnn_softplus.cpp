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
#include "../../../op_host/op_api/aclnn_softplus.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

class l2_softplus_test : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "l2_softplus_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "l2_softplus_test TearDown" << std::endl;
  }
};

// 正常场景_float32_nd
TEST_F(l2_softplus_test, normal_dtype_float32_format_nd) {
  auto selfDesc = TensorDesc({10}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto betaDesc = ScalarDesc(1.5f);
  auto thresholdDesc = ScalarDesc(1.0f);
  auto outDesc = TensorDesc({10}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnSoftplus, INPUT(selfDesc, betaDesc, thresholdDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

TEST_F(l2_softplus_test, ascend910B2_normal_dtype_bf16_format_nd) {
  auto selfDesc = TensorDesc({10}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto betaDesc = ScalarDesc(1.5f);
  auto thresholdDesc = ScalarDesc(1.0f);
  auto outDesc = TensorDesc({10}, ACL_BF16, ACL_FORMAT_ND).Precision(0.004, 0.004);

  auto ut = OP_API_UT(aclnnSoftplus, INPUT(selfDesc, betaDesc, thresholdDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

// 正常场景_float32_nchw
TEST_F(l2_softplus_test, normal_dtype_float32_format_nchw) {
  auto selfDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-2, 2);
  auto betaDesc = ScalarDesc(1.5f);
  auto thresholdDesc = ScalarDesc(1.0f);
  auto outDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_NCHW).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnSoftplus, INPUT(selfDesc, betaDesc, thresholdDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

// 正常场景_float32_nhwc
TEST_F(l2_softplus_test, normal_dtype_float32_format_nhwc) {
  auto selfDesc = TensorDesc({4, 5, 6, 7}, ACL_FLOAT, ACL_FORMAT_NHWC).ValueRange(-2, 2);
  auto betaDesc = ScalarDesc(1.5f);
  auto thresholdDesc = ScalarDesc(1.0f);
  auto outDesc = TensorDesc({4, 5, 6, 7}, ACL_FLOAT, ACL_FORMAT_NHWC).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnSoftplus, INPUT(selfDesc, betaDesc, thresholdDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

// 正常场景_float32_hwcn
TEST_F(l2_softplus_test, normal_dtype_float32_format_hwcn) {
  auto selfDesc = TensorDesc({10}, ACL_FLOAT, ACL_FORMAT_HWCN).ValueRange(-2, 2);
  auto betaDesc = ScalarDesc(-1.5f);
  auto thresholdDesc = ScalarDesc(1.0f);
  auto outDesc = TensorDesc({10}, ACL_FLOAT, ACL_FORMAT_HWCN).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnSoftplus, INPUT(selfDesc, betaDesc, thresholdDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

// 正常场景_float32_ncdhw
TEST_F(l2_softplus_test, normal_dtype_float32_format_ncdhw) {
  auto selfDesc = TensorDesc({10}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-2, 2);
  auto betaDesc = ScalarDesc(1.5f);
  auto thresholdDesc = ScalarDesc(5.0f);
  auto outDesc = TensorDesc({10}, ACL_FLOAT, ACL_FORMAT_NCDHW).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnSoftplus, INPUT(selfDesc, betaDesc, thresholdDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

// 正常场景_float32_ndhwc
TEST_F(l2_softplus_test, normal_dtype_float32_format_ndhwc) {
  auto selfDesc = TensorDesc({4, 5, 9, 7}, ACL_FLOAT, ACL_FORMAT_NDHWC).ValueRange(-2, 2);
  auto betaDesc = ScalarDesc(2.2f);
  auto thresholdDesc = ScalarDesc(1.0f);
  auto outDesc = TensorDesc({4, 5, 9, 7}, ACL_FLOAT, ACL_FORMAT_NDHWC).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnSoftplus, INPUT(selfDesc, betaDesc, thresholdDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

// 正常场景_beta为1
TEST_F(l2_softplus_test, normal_beta_1) {
  auto selfDesc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto betaDesc = ScalarDesc(1.0f);
  auto thresholdDesc = ScalarDesc(1.0f);
  auto outDesc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnSoftplus, INPUT(selfDesc, betaDesc, thresholdDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

// 正常场景_8维输入
TEST_F(l2_softplus_test, normal_shape_dim_8) {
  auto selfDesc = TensorDesc({7, 9, 11, 3, 4, 6, 9, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto betaDesc = ScalarDesc(2.5f);
  auto thresholdDesc = ScalarDesc(4.0f);
  auto outDesc = TensorDesc({7, 9, 11, 3, 4, 6, 9, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnSoftplus, INPUT(selfDesc, betaDesc, thresholdDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

// 空tensor场景
TEST_F(l2_softplus_test, normal_empty_tensor) {
  auto selfDesc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
  auto betaDesc = ScalarDesc(1.0f);
  auto thresholdDesc = ScalarDesc(5.0f);
  auto outDesc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnSoftplus, INPUT(selfDesc, betaDesc, thresholdDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

// CheckNotNull_1
TEST_F(l2_softplus_test, abnormal_self_nullptr) {
  auto selfDesc = nullptr;
  auto betaDesc = ScalarDesc(1.0f);
  auto thresholdDesc = ScalarDesc(5.0f);
  auto outDesc = TensorDesc({10}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnSoftplus, INPUT(selfDesc, betaDesc, thresholdDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckNotNull_2
TEST_F(l2_softplus_test, abnormal_beta_nullptr) {
  auto selfDesc = TensorDesc({10}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto betaDesc = nullptr;
  auto thresholdDesc = ScalarDesc(1.0f);
  auto outDesc = TensorDesc({10}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnSoftplus, INPUT(selfDesc, betaDesc, thresholdDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckNotNull_3
TEST_F(l2_softplus_test, abnormal_threshold_nullptr) {
  auto selfDesc = TensorDesc({10}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto betaDesc = ScalarDesc(1.0f);
  auto thresholdDesc = nullptr;
  auto outDesc = TensorDesc({10}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnSoftplus, INPUT(selfDesc, betaDesc, thresholdDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckNotNull_4
TEST_F(l2_softplus_test, abnormal_out_nullptr) {
  auto selfDesc = TensorDesc({10}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto betaDesc = ScalarDesc(1.0f);
  auto thresholdDesc = ScalarDesc(1.0f);
  auto outDesc = nullptr;

  auto ut = OP_API_UT(aclnnSoftplus, INPUT(selfDesc, betaDesc, thresholdDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckDtypeValid_1
TEST_F(l2_softplus_test, abnormal_dtype_self_int32) {
  auto selfDesc = TensorDesc({10}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto betaDesc = ScalarDesc(1.0f);
  auto thresholdDesc = ScalarDesc(1.0f);
  auto outDesc = TensorDesc({10}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnSoftplus, INPUT(selfDesc, betaDesc, thresholdDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtypeValid_2
TEST_F(l2_softplus_test, abnormal_dtype_self_int64) {
  auto selfDesc = TensorDesc({10}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto betaDesc = ScalarDesc(1.0f);
  auto thresholdDesc = ScalarDesc(1.0f);
  auto outDesc = TensorDesc({10}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnSoftplus, INPUT(selfDesc, betaDesc, thresholdDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtypeValid_3
TEST_F(l2_softplus_test, abnormal_dtype_self_bool) {
  auto selfDesc = TensorDesc({10}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto betaDesc = ScalarDesc(1.0f);
  auto thresholdDesc = ScalarDesc(1.0f);
  auto outDesc = TensorDesc({10}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnSoftplus, INPUT(selfDesc, betaDesc, thresholdDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtypeValid_4
TEST_F(l2_softplus_test, abnormal_dtype_self_int8) {
  auto selfDesc = TensorDesc({10}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto betaDesc = ScalarDesc(1.0f);
  auto thresholdDesc = ScalarDesc(1.0f);
  auto outDesc = TensorDesc({10}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnSoftplus, INPUT(selfDesc, betaDesc, thresholdDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtypeValid_5
TEST_F(l2_softplus_test, abnormal_dtype_self_out_not_cast) {
  auto selfDesc = TensorDesc({10}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
  auto betaDesc = ScalarDesc(1.0f);
  auto thresholdDesc = ScalarDesc(1.0f);
  auto outDesc = TensorDesc({10}, ACL_INT32, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnSoftplus, INPUT(selfDesc, betaDesc, thresholdDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckShape_1
TEST_F(l2_softplus_test, abnormal_shape_dim_greater_than_threshold) {
  auto selfDesc = TensorDesc({7, 9, 11, 3, 4, 6, 9, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto betaDesc = ScalarDesc(1.0f);
  auto thresholdDesc = ScalarDesc(1.0f);
  auto outDesc = TensorDesc({7, 9, 11, 3, 4, 6, 9, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnSoftplus, INPUT(selfDesc, betaDesc, thresholdDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckShape_2
TEST_F(l2_softplus_test, abnormal_shape_self_out_not_equal) {
  auto selfDesc = TensorDesc({5, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto betaDesc = ScalarDesc(1.0f);
  auto thresholdDesc = ScalarDesc(1.0f);
  auto outDesc = TensorDesc({5, 3}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnSoftplus, INPUT(selfDesc, betaDesc, thresholdDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

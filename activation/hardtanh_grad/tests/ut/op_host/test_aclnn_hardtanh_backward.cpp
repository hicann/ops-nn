/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

#include "gtest/gtest.h"
#include "../../../op_host/op_api/aclnn_hardtanh_backward.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

class l2_hardtanh_backward_test : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "l2_hardtanh_backward_test SetUp" << std::endl;
  }

  static void TearDownTestCase() { std::cout << "l2_hardtanh_backward_test TearDown" << std::endl; }
};

// 正常场景_float32_nd
TEST_F(l2_hardtanh_backward_test, normal_dtype_float32_format_nd) {
  auto gradOutputDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
  auto selfDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
  auto minvalDesc = ScalarDesc(-1.0f);
  auto maxvalDesc = ScalarDesc(1.0f);
  auto outDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnHardtanhBackward, INPUT(gradOutputDesc, selfDesc, minvalDesc, maxvalDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

// 异常场景_int64_nd
TEST_F(l2_hardtanh_backward_test, abnormal_dtype_ACL_INT64_format_nd) {
  auto gradOutputDesc = TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND);
  auto selfDesc = TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND);
  auto minvalDesc = ScalarDesc(-1.0f);
  auto maxvalDesc = ScalarDesc(1.0f);
  auto outDesc = TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnHardtanhBackward, INPUT(gradOutputDesc, selfDesc, minvalDesc, maxvalDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景_int16_nd
TEST_F(l2_hardtanh_backward_test, abnormal_dtype_ACL_INT16_format_nd) {
  auto gradOutputDesc = TensorDesc({2, 3}, ACL_INT16, ACL_FORMAT_ND);
  auto selfDesc = TensorDesc({2, 3}, ACL_INT16, ACL_FORMAT_ND);
  auto minvalDesc = ScalarDesc(-1.0f);
  auto maxvalDesc = ScalarDesc(1.0f);
  auto outDesc = TensorDesc({2, 3}, ACL_INT16, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnHardtanhBackward, INPUT(gradOutputDesc, selfDesc, minvalDesc, maxvalDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景_int8_nd
TEST_F(l2_hardtanh_backward_test, abnormal_dtype_ACL_INT8_format_nd) {
  auto gradOutputDesc = TensorDesc({2, 3}, ACL_INT8, ACL_FORMAT_ND);
  auto selfDesc = TensorDesc({2, 3}, ACL_INT8, ACL_FORMAT_ND);
  auto minvalDesc = ScalarDesc(-1.0f);
  auto maxvalDesc = ScalarDesc(1.0f);
  auto outDesc = TensorDesc({2, 3}, ACL_INT8, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnHardtanhBackward, INPUT(gradOutputDesc, selfDesc, minvalDesc, maxvalDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景_uint8_nd
TEST_F(l2_hardtanh_backward_test, abnormal_dtype_ACL_UINT8_format_nd) {
  auto gradOutputDesc = TensorDesc({2, 3}, ACL_UINT8, ACL_FORMAT_ND);
  auto selfDesc = TensorDesc({2, 3}, ACL_UINT8, ACL_FORMAT_ND);
  auto minvalDesc = ScalarDesc(-1.0f);
  auto maxvalDesc = ScalarDesc(1.0f);
  auto outDesc = TensorDesc({2, 3}, ACL_UINT8, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnHardtanhBackward, INPUT(gradOutputDesc, selfDesc, minvalDesc, maxvalDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景_complex64_nd
TEST_F(l2_hardtanh_backward_test, abnormal_dtype_ACL_COMPLEX64_format_nd) {
  auto gradOutputDesc = TensorDesc({2, 3}, ACL_COMPLEX64, ACL_FORMAT_ND);
  auto selfDesc = TensorDesc({2, 3}, ACL_COMPLEX64, ACL_FORMAT_ND);
  auto minvalDesc = ScalarDesc(-1.0f);
  auto maxvalDesc = ScalarDesc(1.0f);
  auto outDesc = TensorDesc({2, 3}, ACL_COMPLEX64, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnHardtanhBackward, INPUT(gradOutputDesc, selfDesc, minvalDesc, maxvalDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景_complex128_nd
TEST_F(l2_hardtanh_backward_test, abnormal_dtype_ACL_COMPLEX128_format_nd) {
  auto gradOutputDesc = TensorDesc({2, 3}, ACL_COMPLEX128, ACL_FORMAT_ND);
  auto selfDesc = TensorDesc({2, 3}, ACL_COMPLEX128, ACL_FORMAT_ND);
  auto minvalDesc = ScalarDesc(-1.0f);
  auto maxvalDesc = ScalarDesc(1.0f);
  auto outDesc = TensorDesc({2, 3}, ACL_COMPLEX128, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnHardtanhBackward, INPUT(gradOutputDesc, selfDesc, minvalDesc, maxvalDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 正常场景_float16_nd
TEST_F(l2_hardtanh_backward_test, normal_dtype_float16_format_nd) {
  auto gradOutputDesc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto selfDesc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto minvalDesc = ScalarDesc(-1.0f);
  auto maxvalDesc = ScalarDesc(1.0f);
  auto outDesc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnHardtanhBackward, INPUT(gradOutputDesc, selfDesc, minvalDesc, maxvalDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

// 正常场景_bf16_nd
TEST_F(l2_hardtanh_backward_test, Ascend910B2_normal_dtype_bf16_format_nd) {
  auto gradOutputDesc = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND);
  auto selfDesc = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND);
  auto minvalDesc = ScalarDesc(-1.0f);
  auto maxvalDesc = ScalarDesc(1.0f);
  auto outDesc = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnHardtanhBackward, INPUT(gradOutputDesc, selfDesc, minvalDesc, maxvalDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // ut.TestPrecision();
}

// 正常场景_float32_nhwc
TEST_F(l2_hardtanh_backward_test, normal_dtype_float32_format_nhwc) {
  auto gradOutputDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_NHWC).ValueRange(-2, 2);
  auto selfDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_NHWC).ValueRange(-2, 2);
  auto minvalDesc = ScalarDesc(-1.0f);
  auto maxvalDesc = ScalarDesc(1.0f);
  auto outDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_NHWC).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnHardtanhBackward, INPUT(gradOutputDesc, selfDesc, minvalDesc, maxvalDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

// 正常场景_float32_nchw
TEST_F(l2_hardtanh_backward_test, normal_dtype_float32_format_nchw) {
  auto gradOutputDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-2, 2);
  auto selfDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-2, 2);
  auto minvalDesc = ScalarDesc(-1.0f);
  auto maxvalDesc = ScalarDesc(1.0f);
  auto outDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_NCHW).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnHardtanhBackward, INPUT(gradOutputDesc, selfDesc, minvalDesc, maxvalDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

// 正常场景_float32_hwcn
TEST_F(l2_hardtanh_backward_test, normal_dtype_float32_format_hwcn) {
  auto gradOutputDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_HWCN).ValueRange(-2, 2);
  auto selfDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_HWCN).ValueRange(-2, 2);
  auto minvalDesc = ScalarDesc(-1.0f);
  auto maxvalDesc = ScalarDesc(1.0f);
  auto outDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_HWCN).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnHardtanhBackward, INPUT(gradOutputDesc, selfDesc, minvalDesc, maxvalDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

// 正常场景_float32_ncdhw
TEST_F(l2_hardtanh_backward_test, normal_dtype_float32_format_ncdhw) {
  auto gradOutputDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-2, 2);
  auto selfDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-2, 2);
  auto minvalDesc = ScalarDesc(-1.0f);
  auto maxvalDesc = ScalarDesc(1.0f) ;

  auto outDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_NCDHW).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnHardtanhBackward, INPUT(gradOutputDesc, selfDesc, minvalDesc, maxvalDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

// 正常场景_float32_ndhwc
TEST_F(l2_hardtanh_backward_test, normal_dtype_float32_format_ndhwc) {
  auto gradOutputDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_NDHWC).ValueRange(-2, 2);
  auto selfDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_NDHWC).ValueRange(-2, 2);
  auto minvalDesc = ScalarDesc(-1.0f);
  auto maxvalDesc = ScalarDesc(1.0f);
  auto outDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_NDHWC).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnHardtanhBackward, INPUT(gradOutputDesc, selfDesc, minvalDesc, maxvalDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

// 空tensor场景
TEST_F(l2_hardtanh_backward_test, normal_empty_tensor) {
  auto gradOutputDesc = TensorDesc({0, 3}, ACL_FLOAT, ACL_FORMAT_ND);
  auto selfDesc = TensorDesc({0, 3}, ACL_FLOAT, ACL_FORMAT_ND);
  auto minvalDesc = ScalarDesc(-1.0f);
  auto maxvalDesc = ScalarDesc(1.0f);
  auto outDesc = TensorDesc({0, 3}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnHardtanhBackward, INPUT(gradOutputDesc, selfDesc, minvalDesc, maxvalDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

// CheckNotNull_1
TEST_F(l2_hardtanh_backward_test, abnormal_grad_output_nullptr) {
  auto gradOutputDesc = nullptr;
  auto selfDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
  auto minvalDesc = ScalarDesc(-1.0f);
  auto maxvalDesc = ScalarDesc(1.0f);
  auto outDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnHardtanhBackward, INPUT(gradOutputDesc, selfDesc, minvalDesc, maxvalDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckNotNull_2
TEST_F(l2_hardtanh_backward_test, abnormal_self_nullptr) {
  auto gradOutputDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
  auto selfDesc = nullptr;
  auto minvalDesc = ScalarDesc(-1.0f);
  auto maxvalDesc = ScalarDesc(1.0f);
  auto outDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnHardtanhBackward, INPUT(gradOutputDesc, selfDesc, minvalDesc, maxvalDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckNotNull_3
TEST_F(l2_hardtanh_backward_test, abnormal_out_nullptr) {
  auto gradOutputDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
  auto selfDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
  auto minvalDesc = ScalarDesc(-1.0f);
  auto maxvalDesc = ScalarDesc(1.0f);
  auto outDesc = nullptr;

  auto ut = OP_API_UT(aclnnHardtanhBackward, INPUT(gradOutputDesc, selfDesc, minvalDesc, maxvalDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckDtypeValid_1
TEST_F(l2_hardtanh_backward_test, abnormal_dtype_grad_output_int64) {
  auto gradOutputDesc = TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND);
  auto selfDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
  auto minvalDesc = ScalarDesc(-1.0f);
  auto maxvalDesc = ScalarDesc(1.0f);
  auto outDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnHardtanhBackward, INPUT(gradOutputDesc, selfDesc, minvalDesc, maxvalDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtypeValid_2
TEST_F(l2_hardtanh_backward_test, abnormal_dtype_self_int64) {
  auto gradOutputDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
  auto selfDesc = TensorDesc({2, 3}, ACL_INT64, ACL_FORMAT_ND);
  auto minvalDesc = ScalarDesc(-1.0f);
  auto maxvalDesc = ScalarDesc(1.0f);
  auto outDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnHardtanhBackward, INPUT(gradOutputDesc, selfDesc, minvalDesc, maxvalDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtypeValid_3
TEST_F(l2_hardtanh_backward_test, abnormal_dtype_grad_output_self_unequal) {
  auto gradOutputDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
  auto selfDesc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto minvalDesc = ScalarDesc(-1.0f);
  auto maxvalDesc = ScalarDesc(1.0f);
  auto outDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnHardtanhBackward, INPUT(gradOutputDesc, selfDesc, minvalDesc, maxvalDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckShape_1
TEST_F(l2_hardtanh_backward_test, abnormal_shape_grad_output_self_unequal) {
  auto gradOutputDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
  auto selfDesc = TensorDesc({1, 3}, ACL_FLOAT, ACL_FORMAT_ND);
  auto minvalDesc = ScalarDesc(-1.0f);
  auto maxvalDesc = ScalarDesc(1.0f);
  auto outDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnHardtanhBackward, INPUT(gradOutputDesc, selfDesc, minvalDesc, maxvalDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// CheckShape_2
TEST_F(l2_hardtanh_backward_test, abnormal_shape_dim_greater_than_threshold) {
  auto gradOutputDesc = TensorDesc({7, 9, 11, 3, 4, 6, 9, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto selfDesc = TensorDesc({7, 9, 11, 3, 4, 6, 9, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto minvalDesc = ScalarDesc(-1.0f);
  auto maxvalDesc = ScalarDesc(1.0f);
  auto outDesc = TensorDesc({7, 9, 11, 3, 4, 6, 9, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnHardtanhBackward, INPUT(gradOutputDesc, selfDesc, minvalDesc, maxvalDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

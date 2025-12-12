/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
#include "../../../op_host/op_api/aclnn_glu_backward.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace op;
using namespace std;

class l2_glu_backward_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "glu_backward_tensor_test SetUp" << std::endl;
  }

  static void TearDownTestCase() { std::cout << "glu_backward_tensor_test TearDown" << std::endl; }
};

// 正常场景 float dim正数
TEST_F(l2_glu_backward_test, test_glu_backward_normal_self_float_dim_positive) {
  auto grad_out = TensorDesc({2, 2, 6}, ACL_FLOAT, ACL_FORMAT_ND);
  auto tensor_self = TensorDesc({2, 4, 6}, ACL_FLOAT, ACL_FORMAT_ND);
  int64_t dim = 1;
  auto out_tensor_desc = TensorDesc({2, 4, 6}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnGluBackward, INPUT(grad_out, tensor_self, dim), OUTPUT(out_tensor_desc));
  // SAMPLE: precision simulate
  ut.TestPrecision();
}

// 正常场景 float dim负数
TEST_F(l2_glu_backward_test, test_glu_backward_normal_self_float_dim_nopositive) {
  auto grad_out = TensorDesc({2, 2, 6}, ACL_FLOAT, ACL_FORMAT_ND);
  auto tensor_self = TensorDesc({2, 4, 6}, ACL_FLOAT, ACL_FORMAT_ND);
  int64_t dim = -2;
  auto out_tensor_desc = TensorDesc({2, 4, 6}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnGluBackward, INPUT(grad_out, tensor_self, dim), OUTPUT(out_tensor_desc));
  // SAMPLE: precision simulate
  ut.TestPrecision();
}

// 正常场景 float16 dim负数,gpu支持，cpu不支持f16，所以不测试精度
TEST_F(l2_glu_backward_test, test_glu_backward_normal_self_float16_dim_nopositive) {
  auto grad_out = TensorDesc({2, 2, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto tensor_self = TensorDesc({2, 4, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
  int64_t dim = -2;
  auto out_tensor_desc = TensorDesc({2, 4, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnGluBackward, INPUT(grad_out, tensor_self, dim), OUTPUT(out_tensor_desc));
  // SAMPLE: precision simulate
  uint64_t workspace_size = 5;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景 double dim负数
TEST_F(l2_glu_backward_test, test_glu_backward_normal_self_double_dim_nopositive) {
  auto grad_out = TensorDesc({2, 2, 6}, ACL_DOUBLE, ACL_FORMAT_ND);
  auto tensor_self = TensorDesc({2, 4, 6}, ACL_DOUBLE, ACL_FORMAT_ND);
  int64_t dim = -2;
  auto out_tensor_desc = TensorDesc({2, 4, 6}, ACL_DOUBLE, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
  auto ut = OP_API_UT(aclnnGluBackward, INPUT(grad_out, tensor_self, dim), OUTPUT(out_tensor_desc));
  // SAMPLE: precision simulate
  ut.TestPrecision();
}

// 正常场景 空tensor
TEST_F(l2_glu_backward_test, test_glu_backward_normal_self_empty_tensor) {
  auto grad_out = TensorDesc({2, 0, 6}, ACL_FLOAT, ACL_FORMAT_ND);
  auto tensor_self = TensorDesc({2, 0, 6}, ACL_FLOAT, ACL_FORMAT_ND);
  int64_t dim = -2;
  auto out_tensor_desc = TensorDesc({2, 0, 6}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
  auto ut = OP_API_UT(aclnnGluBackward, INPUT(grad_out, tensor_self, dim), OUTPUT(out_tensor_desc));
  // SAMPLE: precision simulate
  ut.TestPrecision();
}

// 正常场景 float 8维tensor 02
TEST_F(l2_glu_backward_test, test_glu_backward_normal_self_float_8dim_tensor02) {
  auto grad_out = TensorDesc({4,8,8,8,8,8,8,8}, ACL_FLOAT, ACL_FORMAT_ND);
  auto tensor_self = TensorDesc({8,8,8,8,8,8,8,8}, ACL_FLOAT, ACL_FORMAT_ND);
  int64_t dim = -8;
  auto out_tensor_desc = TensorDesc({8,8,8,8,8,8,8,8}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnGluBackward, INPUT(grad_out, tensor_self, dim), OUTPUT(out_tensor_desc));
  // SAMPLE: precision simulate
  // ut.TestPrecision(); // comment bcz of timeout in model tests (482061 ms)
}

// 正常场景 float 8维tensor
TEST_F(l2_glu_backward_test, test_glu_backward_normal_self_float_8dim_tensor) {
  auto grad_out = TensorDesc({6,6,12,6,15,5,8,12}, ACL_FLOAT, ACL_FORMAT_ND);
  auto tensor_self = TensorDesc({12,6,12,6,15,5,8,12}, ACL_FLOAT, ACL_FORMAT_ND);
  int64_t dim = 0;
  auto out_tensor_desc = TensorDesc({12,6,12,6,15,5,8,12}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
  auto ut = OP_API_UT(aclnnGluBackward, INPUT(grad_out, tensor_self, dim), OUTPUT(out_tensor_desc));
  // SAMPLE: precision simulate
  // ut.TestPrecision(); // comment bcz of timeout in model tests (808654 ms)
}

// 正常场景 double 8维tensor
TEST_F(l2_glu_backward_test, test_glu_backward_normal_self_double_8dim_tensor) {
  auto grad_out = TensorDesc({6,6,12,6,15,5,8,12}, ACL_DOUBLE, ACL_FORMAT_ND);
  auto tensor_self = TensorDesc({12,6,12,6,15,5,8,12}, ACL_DOUBLE, ACL_FORMAT_ND);
  int64_t dim = 0;
  auto out_tensor_desc = TensorDesc({12,6,12,6,15,5,8,12}, ACL_DOUBLE, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
  auto ut = OP_API_UT(aclnnGluBackward, INPUT(grad_out, tensor_self, dim), OUTPUT(out_tensor_desc));
  // SAMPLE: precision simulate
  // ut.TestPrecision();
}

// 正常场景 float dim 为其他int类型
TEST_F(l2_glu_backward_test, test_glu_backward_normal_self_float_dim_int16) {
  auto grad_out = TensorDesc({2, 2, 6}, ACL_FLOAT, ACL_FORMAT_ND);
  auto tensor_self = TensorDesc({2, 4, 6}, ACL_FLOAT, ACL_FORMAT_ND);
  int16_t dim = -2;
  auto out_tensor_desc = TensorDesc({2, 4, 6}, ACL_FLOAT, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnGluBackward, INPUT(grad_out, tensor_self, dim), OUTPUT(out_tensor_desc));
  // SAMPLE: precision simulate
  ut.TestPrecision();
}

// 正常场景 float dim 为其他int类型
TEST_F(l2_glu_backward_test, test_glu_backward_normal_self_float_dim_int32) {
  auto grad_out = TensorDesc({2, 2, 6}, ACL_FLOAT, ACL_FORMAT_ND);
  auto tensor_self = TensorDesc({2, 4, 6}, ACL_FLOAT, ACL_FORMAT_ND);
  int32_t dim = -2;
  auto out_tensor_desc = TensorDesc({2, 4, 6}, ACL_FLOAT, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnGluBackward, INPUT(grad_out, tensor_self, dim), OUTPUT(out_tensor_desc));
  // SAMPLE: precision simulate
  ut.TestPrecision();
}

// 正常场景 float dim 为其他int类型
TEST_F(l2_glu_backward_test, test_glu_backward_normal_self_float_dim_int8) {
  auto grad_out = TensorDesc({2, 2, 6}, ACL_FLOAT, ACL_FORMAT_ND);
  auto tensor_self = TensorDesc({2, 4, 6}, ACL_FLOAT, ACL_FORMAT_ND);
  int8_t dim = -2;
  auto out_tensor_desc = TensorDesc({2, 4, 6}, ACL_FLOAT, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnGluBackward, INPUT(grad_out, tensor_self, dim), OUTPUT(out_tensor_desc));
  // SAMPLE: precision simulate
  ut.TestPrecision();
}

// 正常场景 float dim 为其他uint类型
TEST_F(l2_glu_backward_test, test_glu_backward_normal_self_float_dim_uint8) {
  auto grad_out = TensorDesc({2, 2, 6}, ACL_FLOAT, ACL_FORMAT_ND);
  auto tensor_self = TensorDesc({2, 4, 6}, ACL_FLOAT, ACL_FORMAT_ND);
  uint8_t dim = 1;
  auto out_tensor_desc = TensorDesc({2, 4, 6}, ACL_FLOAT, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnGluBackward, INPUT(grad_out, tensor_self, dim), OUTPUT(out_tensor_desc));
  // SAMPLE: precision simulate
  ut.TestPrecision();
}

// 正常场景 float dim 为其他uint类型
TEST_F(l2_glu_backward_test, test_glu_backward_normal_self_float_dim_uint16) {
  auto grad_out = TensorDesc({2, 2, 6}, ACL_FLOAT, ACL_FORMAT_ND);
  auto tensor_self = TensorDesc({2, 4, 6}, ACL_FLOAT, ACL_FORMAT_ND);
  uint16_t dim = 1;
  auto out_tensor_desc = TensorDesc({2, 4, 6}, ACL_FLOAT, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnGluBackward, INPUT(grad_out, tensor_self, dim), OUTPUT(out_tensor_desc));
  // SAMPLE: precision simulate
  ut.TestPrecision();
}

// 正常场景 float dim 为其他uint类型
TEST_F(l2_glu_backward_test, test_glu_backward_normal_self_float_dim_uint32) {
  auto grad_out = TensorDesc({2, 2, 6}, ACL_FLOAT, ACL_FORMAT_ND);
  auto tensor_self = TensorDesc({2, 4, 6}, ACL_FLOAT, ACL_FORMAT_ND);
  uint32_t dim = 1;
  auto out_tensor_desc = TensorDesc({2, 4, 6}, ACL_FLOAT, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnGluBackward, INPUT(grad_out, tensor_self, dim), OUTPUT(out_tensor_desc));
  // SAMPLE: precision simulate
  ut.TestPrecision();
}


// 正常场景 float dim 为其他uint类型
TEST_F(l2_glu_backward_test, test_glu_backward_normal_self_float_dim_uint64) {
  auto grad_out = TensorDesc({2, 2, 6}, ACL_FLOAT, ACL_FORMAT_ND);
  auto tensor_self = TensorDesc({2, 4, 6}, ACL_FLOAT, ACL_FORMAT_ND);
  uint64_t dim = 1;
  auto out_tensor_desc = TensorDesc({2, 4, 6}, ACL_FLOAT, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnGluBackward, INPUT(grad_out, tensor_self, dim), OUTPUT(out_tensor_desc));
  // SAMPLE: precision simulate
  ut.TestPrecision();
}

// 异常场景 gradout为null
TEST_F(l2_glu_backward_test, test_glu_backward_gradout_is_null) {
  auto grad_out = nullptr;
  auto tensor_self = TensorDesc({2, 4, 6}, ACL_FLOAT, ACL_FORMAT_ND);
  int64_t dim = -2;
  auto out_tensor_desc = TensorDesc({2, 4, 6}, ACL_FLOAT, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnGluBackward, INPUT(grad_out, tensor_self, dim), OUTPUT(out_tensor_desc));
  uint64_t workspace_size = 5;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 异常场景 self为null
TEST_F(l2_glu_backward_test, test_glu_backward_self_is_null) {
  auto grad_out = TensorDesc({2, 2, 6}, ACL_FLOAT, ACL_FORMAT_ND);
  auto tensor_self = nullptr;
  int64_t dim = -2;
  auto out_tensor_desc = TensorDesc({2, 2, 6}, ACL_FLOAT, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnGluBackward, INPUT(grad_out, tensor_self, dim), OUTPUT(out_tensor_desc));
  uint64_t workspace_size = 5;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 异常场景 out为null
TEST_F(l2_glu_backward_test, test_glu_backward_out_is_null) {
  auto grad_out = TensorDesc({2, 2, 6}, ACL_FLOAT, ACL_FORMAT_ND);
  auto tensor_self = TensorDesc({2, 4, 6}, ACL_FLOAT, ACL_FORMAT_ND);
  int64_t dim = -2;
  auto out_tensor_desc = nullptr;
  auto ut = OP_API_UT(aclnnGluBackward, INPUT(grad_out, tensor_self, dim), OUTPUT(out_tensor_desc));
  uint64_t workspace_size = 5;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 异常场景 self不在支持的数据类型范围内
TEST_F(l2_glu_backward_test, test_glu_backward_self_is_int64) {
  auto grad_out = TensorDesc({2, 2, 6}, ACL_FLOAT, ACL_FORMAT_ND);
  auto tensor_self = TensorDesc({2, 4, 6}, ACL_INT64, ACL_FORMAT_ND);
  int64_t dim = -2;
  auto out_tensor_desc = TensorDesc({2, 4, 6}, ACL_FLOAT, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnGluBackward, INPUT(grad_out, tensor_self, dim), OUTPUT(out_tensor_desc));
  uint64_t workspace_size = 5;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景 self不在支持的数据类型范围内
TEST_F(l2_glu_backward_test, test_glu_backward_self_is_bf16) {
  auto grad_out = TensorDesc({2, 2, 6}, ACL_FLOAT, ACL_FORMAT_ND);
  auto tensor_self = TensorDesc({2, 4, 6}, ACL_BF16, ACL_FORMAT_ND);
  int64_t dim = -2;
  auto out_tensor_desc = TensorDesc({2, 4, 6}, ACL_FLOAT, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnGluBackward, INPUT(grad_out, tensor_self, dim), OUTPUT(out_tensor_desc));
  uint64_t workspace_size = 5;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景 out不在支持的数据类型范围内,out的dtype必修与self一致
TEST_F(l2_glu_backward_test, test_glu_backward_out_is_int64) {
  auto grad_out = TensorDesc({2, 2, 6}, ACL_FLOAT, ACL_FORMAT_ND);
  auto tensor_self = TensorDesc({2, 4, 6}, ACL_FLOAT, ACL_FORMAT_ND);
  int64_t dim = -2;
  auto out_tensor_desc = TensorDesc({2, 4, 6}, ACL_INT64, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnGluBackward, INPUT(grad_out, tensor_self, dim), OUTPUT(out_tensor_desc));
  uint64_t workspace_size = 5;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景 gradout不在支持的数据类型范围内,gradout的dtype必修与self一致
TEST_F(l2_glu_backward_test, test_glu_backward_gradout_is_int64) {
  auto grad_out = TensorDesc({2, 2, 6}, ACL_INT64, ACL_FORMAT_ND);
  auto tensor_self = TensorDesc({2, 4, 6}, ACL_FLOAT, ACL_FORMAT_ND);
  int64_t dim = -2;
  auto out_tensor_desc = TensorDesc({2, 4, 6}, ACL_FLOAT, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnGluBackward, INPUT(grad_out, tensor_self, dim), OUTPUT(out_tensor_desc));
  uint64_t workspace_size = 5;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景 out不在支持的数据类型范围内,out的dtype必修与self一致
TEST_F(l2_glu_backward_test, test_glu_backward_out_is_bf16) {
  auto grad_out = TensorDesc({2, 2, 6}, ACL_FLOAT, ACL_FORMAT_ND);
  auto tensor_self = TensorDesc({2, 4, 6}, ACL_FLOAT, ACL_FORMAT_ND);
  int64_t dim = -2;
  auto out_tensor_desc = TensorDesc({2, 4, 6}, ACL_BF16, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnGluBackward, INPUT(grad_out, tensor_self, dim), OUTPUT(out_tensor_desc));
  uint64_t workspace_size = 5;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}


// 异常场景 入参dim超出了self的shape可选维度范围[-self.dim，self.dim-1]
TEST_F(l2_glu_backward_test, test_glu_backward_dim_is_less_than_left) {
  auto grad_out = TensorDesc({2, 2, 6}, ACL_FLOAT, ACL_FORMAT_ND);
  auto tensor_self = TensorDesc({2, 4, 6}, ACL_FLOAT, ACL_FORMAT_ND);
  int64_t dim = -4;
  auto out_tensor_desc = TensorDesc({2, 4, 6}, ACL_FLOAT, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnGluBackward, INPUT(grad_out, tensor_self, dim), OUTPUT(out_tensor_desc));
  uint64_t workspace_size = 5;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景 入参dim超出了self的shape可选维度范围[-self.dim，self.dim-1]
TEST_F(l2_glu_backward_test, test_glu_backward_dim_is_bigger_than_right) {
  auto grad_out = TensorDesc({2, 2, 6}, ACL_FLOAT, ACL_FORMAT_ND);
  auto tensor_self = TensorDesc({2, 4, 6}, ACL_FLOAT, ACL_FORMAT_ND);
  int64_t dim = 3;
  auto out_tensor_desc = TensorDesc({2, 4, 6}, ACL_FLOAT, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnGluBackward, INPUT(grad_out, tensor_self, dim), OUTPUT(out_tensor_desc));
  uint64_t workspace_size = 5;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景 入参self根据指定的dim所对应的维度不能整除2
TEST_F(l2_glu_backward_test, test_glu_backward_dim_of_self_not_even) {
  auto grad_out = TensorDesc({2, 4, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto tensor_self = TensorDesc({2, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND);
  int64_t dim = 2;
  auto out_tensor_desc = TensorDesc({2, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnGluBackward, INPUT(grad_out, tensor_self, dim), OUTPUT(out_tensor_desc));
  uint64_t workspace_size = 5;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景 gradout的shape不等于self根据dim拆分后的shape
TEST_F(l2_glu_backward_test, test_glu_backward_gradout_shape_is_error) {
  auto grad_out = TensorDesc({2, 3, 6}, ACL_FLOAT, ACL_FORMAT_ND);
  auto tensor_self = TensorDesc({2, 4, 6}, ACL_FLOAT, ACL_FORMAT_ND);
  int64_t dim = 1;
  auto out_tensor_desc = TensorDesc({2, 4, 6}, ACL_FLOAT, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnGluBackward, INPUT(grad_out, tensor_self, dim), OUTPUT(out_tensor_desc));
  uint64_t workspace_size = 5;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景 out的shape不等于self根据dim拆分后的shape
TEST_F(l2_glu_backward_test, test_glu_backward_out_shape_is_error) {
  auto grad_out = TensorDesc({2, 2, 6}, ACL_FLOAT, ACL_FORMAT_ND);
  auto tensor_self = TensorDesc({2, 4, 6}, ACL_FLOAT, ACL_FORMAT_ND);
  int64_t dim = 1;
  auto out_tensor_desc = TensorDesc({2, 4, 7}, ACL_FLOAT, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnGluBackward, INPUT(grad_out, tensor_self, dim), OUTPUT(out_tensor_desc));
  uint64_t workspace_size = 5;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景 gradout 的维度大于8
TEST_F(l2_glu_backward_test, test_glu_backward_gradout_dim_bigger_8) {
  auto grad_out = TensorDesc({2,2,2,2,2,2,2,2,2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto tensor_self = TensorDesc({2,2,2}, ACL_FLOAT, ACL_FORMAT_ND);
  int64_t dim = 0;
  auto out_tensor_desc = TensorDesc({2,2,2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnGluBackward, INPUT(grad_out, tensor_self, dim), OUTPUT(out_tensor_desc));
  uint64_t workspace_size = 5;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景 self的维度大于8
TEST_F(l2_glu_backward_test, test_glu_backward_self_dim_bigger_8) {
  auto grad_out = TensorDesc({2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto tensor_self = TensorDesc({2,2,2,2,2,2,2,2,2}, ACL_FLOAT, ACL_FORMAT_ND);
  int64_t dim = 0;
  auto out_tensor_desc = TensorDesc({2,2,2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnGluBackward, INPUT(grad_out, tensor_self, dim), OUTPUT(out_tensor_desc));
  uint64_t workspace_size = 5;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景 out的维度大于8
TEST_F(l2_glu_backward_test, test_glu_backward_out_dim_bigger_8) {
  auto grad_out = TensorDesc({2, 2, 6}, ACL_FLOAT, ACL_FORMAT_ND);
  auto tensor_self = TensorDesc({2,2,2}, ACL_FLOAT, ACL_FORMAT_ND);
  int64_t dim = 0;
  auto out_tensor_desc = TensorDesc({2,2,2,2,2,2,2,2,2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnGluBackward, INPUT(grad_out, tensor_self, dim), OUTPUT(out_tensor_desc));
  uint64_t workspace_size = 5;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景 self的维度小于等于0，标量tensor
TEST_F(l2_glu_backward_test, test_glu_backward_self_dim_is_0) {
  auto grad_out = TensorDesc({2, 2, 6}, ACL_FLOAT, ACL_FORMAT_ND);
  auto tensor_self = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
  int64_t dim = 0;
  auto out_tensor_desc = TensorDesc({2,2,2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnGluBackward, INPUT(grad_out, tensor_self, dim), OUTPUT(out_tensor_desc));
  uint64_t workspace_size = 5;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 异常场景 gradout的维度小于等于0，标量tensor
TEST_F(l2_glu_backward_test, test_glu_backward_gradout_dim_is_0) {
  auto grad_out = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
  auto tensor_self = TensorDesc({2,4,6}, ACL_FLOAT, ACL_FORMAT_ND);
  int64_t dim = 0;
  auto out_tensor_desc = TensorDesc({2,4,6}, ACL_FLOAT, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnGluBackward, INPUT(grad_out, tensor_self, dim), OUTPUT(out_tensor_desc));
  uint64_t workspace_size = 5;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非连续 self
TEST_F(l2_glu_backward_test, test_glu_backward_to_contiguous) {
  auto grad_out = TensorDesc({1, 4}, ACL_FLOAT, ACL_FORMAT_ND);
  auto tensor_self = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2});
  auto dim = 0;
  auto out_tensor_desc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnGluBackward, INPUT(grad_out, tensor_self, dim), OUTPUT(out_tensor_desc));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
  ut.TestPrecision();
}

// 非连续 self
TEST_F(l2_glu_backward_test, test_glu_backward_to_contiguous02) {
  auto grad_out = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
  auto tensor_self = TensorDesc({2, 6}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {6, 2});
  auto dim = -1;
  auto out_tensor_desc = TensorDesc({2, 6}, ACL_FLOAT, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnGluBackward, INPUT(grad_out, tensor_self, dim), OUTPUT(out_tensor_desc));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
  ut.TestPrecision();
}

// 非连续 gradout
TEST_F(l2_glu_backward_test, test_glu_backward_gradout_to_contiguous) {
  auto grad_out = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {3, 2});
  auto tensor_self = TensorDesc({2, 6}, ACL_FLOAT, ACL_FORMAT_ND);
  auto dim = -1;
  auto out_tensor_desc = TensorDesc({2, 6}, ACL_FLOAT, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnGluBackward, INPUT(grad_out, tensor_self, dim), OUTPUT(out_tensor_desc));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
  ut.TestPrecision();
}

// 非连续 gradout & self
TEST_F(l2_glu_backward_test, test_glu_backward_gradout_self_to_contiguous) {
  auto grad_out = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {3, 2});
  auto tensor_self = TensorDesc({2, 6}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {6, 2});
  auto dim = -1;
  auto out_tensor_desc = TensorDesc({2, 6}, ACL_FLOAT, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnGluBackward, INPUT(grad_out, tensor_self, dim), OUTPUT(out_tensor_desc));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
  ut.TestPrecision();
}

// 数据范围[-1，1]
TEST_F(l2_glu_backward_test, test_glu_backward_date_range_f1_1) {
  auto grad_out = TensorDesc({2, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto tensor_self = TensorDesc({2, 4, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto dim = 2;
  auto out_tensor_desc = TensorDesc({2, 4, 8}, ACL_FLOAT, ACL_FORMAT_ND);
  auto ut = OP_API_UT(aclnnGluBackward, INPUT(grad_out, tensor_self, dim), OUTPUT(out_tensor_desc));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_SUCCESS);
  ut.TestPrecision();
}

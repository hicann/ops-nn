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
#include "../../../op_api/aclnn_index_fill_tensor.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

class l2_inplace_index_fill_tensor_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    cout << "inpace_index_fill_tensor_test SetUp" << endl;
  }

  static void TearDownTestCase() { cout << "inpace_index_fill_tensor_test TearDown" << endl; }
};

// 正常场景 self:int32 fillVal:int32
TEST_F(l2_inplace_index_fill_tensor_test, test_index_fill_support_int32) {
  int64_t dim = 1;
  auto tensor_self = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND)
                         .ValueRange(-10, 10)
                         .Value(vector<int32_t>{3, 4, 9, 6, 7, 11});
  auto index = IntArrayDesc(vector<int64_t>{0, 2});
  auto fillVal = ScalarDesc(static_cast<int32_t>(1));

  auto ut = OP_API_UT(aclnnInplaceIndexFillTensor, INPUT(tensor_self, dim, index, fillVal), OUTPUT());

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_inplace_index_fill_tensor_test, test_index_fill_support_float16) {
  int64_t dim = 1;
  auto tensor_self = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND)
                         .ValueRange(-10, 10)
                         .Value(vector<float>{3, 4, 9, 6, 7, 11});
  auto index = IntArrayDesc(vector<int64_t>{0, 2});
  auto fillVal = ScalarDesc(static_cast<float>(1));

  auto ut = OP_API_UT(aclnnInplaceIndexFillTensor, INPUT(tensor_self, dim, index, fillVal), OUTPUT());

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_inplace_index_fill_tensor_test, test_index_fill_support_float32) {
  int64_t dim = 1;
  auto tensor_self = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND)
                         .ValueRange(-10, 10)
                         .Value(vector<float>{3, 4, 9, 6, 7, 11});
  auto index = IntArrayDesc(vector<int64_t>{0, 2});
  auto fillVal = ScalarDesc(static_cast<float>(1));

  auto ut = OP_API_UT(aclnnInplaceIndexFillTensor, INPUT(tensor_self, dim, index, fillVal), OUTPUT());

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常场景 self:int32 fillVal:float
TEST_F(l2_inplace_index_fill_tensor_test, test_index_fill_int32_float16_cast) {
  int64_t dim = 1;
  auto tensor_self = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND)
                         .ValueRange(-10, 10)
                         .Value(vector<int32_t>{3, 4, 9, 6, 7, 11});
  auto index = IntArrayDesc(vector<int64_t>{0, 2});
  auto fillVal = ScalarDesc(static_cast<float>(1));

  auto ut = OP_API_UT(aclnnInplaceIndexFillTensor, INPUT(tensor_self, dim, index, fillVal), OUTPUT());

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_inplace_index_fill_tensor_test, test_index_fill_int32_float32_cast) {
  int64_t dim = 1;
  auto tensor_self = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND)
                         .ValueRange(-10, 10)
                         .Value(vector<int32_t>{3, 4, 9, 6, 7, 11});
  auto index = IntArrayDesc(vector<int64_t>{0, 2});
  auto fillVal = ScalarDesc(static_cast<float>(1));

  auto ut = OP_API_UT(aclnnInplaceIndexFillTensor, INPUT(tensor_self, dim, index, fillVal), OUTPUT());

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_inplace_index_fill_tensor_test, test_index_fill_float16_int32_cast) {
  int64_t dim = 1;
  auto tensor_self = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND)
                         .ValueRange(-10, 10)
                         .Value(vector<float>{3, 4, 9, 6, 7, 11});
  auto index = IntArrayDesc(vector<int64_t>{0, 2});
  auto fillVal = ScalarDesc(static_cast<int32_t>(1));

  auto ut = OP_API_UT(aclnnInplaceIndexFillTensor, INPUT(tensor_self, dim, index, fillVal), OUTPUT());

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_inplace_index_fill_tensor_test, test_index_fill_float32_int32_cast) {
  int64_t dim = 1;
  auto tensor_self = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND)
                         .ValueRange(-10, 10)
                         .Value(vector<float>{3, 4, 9, 6, 7, 11});
  auto index = IntArrayDesc(vector<int64_t>{0, 2});
  auto fillVal = ScalarDesc(static_cast<int32_t>(1));

  auto ut = OP_API_UT(aclnnInplaceIndexFillTensor, INPUT(tensor_self, dim, index, fillVal), OUTPUT());

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// format test:HWCN
TEST_F(l2_inplace_index_fill_tensor_test, test_index_fill_support_hwcn) {
  int64_t dim = 1;
  auto tensor_self = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_HWCN)
                         .ValueRange(-10, 10)
                         .Value(vector<int32_t>{3, 4, 9, 6, 7, 11});
  auto index = IntArrayDesc(vector<int64_t>{0, 2});
  auto fillVal = ScalarDesc(static_cast<int32_t>(1));

  auto ut = OP_API_UT(aclnnInplaceIndexFillTensor, INPUT(tensor_self, dim, index, fillVal), OUTPUT());

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_inplace_index_fill_tensor_test, test_index_fill_support_ncdhw) {
  int64_t dim = 1;
  auto tensor_self = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_NCDHW)
                         .ValueRange(-10, 10)
                         .Value(vector<int32_t>{3, 4, 9, 6, 7, 11});
  auto index = IntArrayDesc(vector<int64_t>{0, 2});
  auto fillVal = ScalarDesc(static_cast<int32_t>(1));

  auto ut = OP_API_UT(aclnnInplaceIndexFillTensor, INPUT(tensor_self, dim, index, fillVal), OUTPUT());

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_inplace_index_fill_tensor_test, test_index_fill_support_ndhwc) {
  int64_t dim = 1;
  auto tensor_self = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_NDHWC)
                         .ValueRange(-10, 10)
                         .Value(vector<int32_t>{3, 4, 9, 6, 7, 11});
  auto index = IntArrayDesc(vector<int64_t>{0, 2});
  auto fillVal = ScalarDesc(static_cast<int32_t>(1));

  auto ut = OP_API_UT(aclnnInplaceIndexFillTensor, INPUT(tensor_self, dim, index, fillVal), OUTPUT());

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_inplace_index_fill_tensor_test, test_index_fill_support_nhwc) {
  int64_t dim = 1;
  auto tensor_self = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_NHWC)
                         .ValueRange(-10, 10)
                         .Value(vector<int32_t>{3, 4, 9, 6, 7, 11});
  auto index = IntArrayDesc(vector<int64_t>{0, 2});
  auto fillVal = ScalarDesc(static_cast<int32_t>(1));

  auto ut = OP_API_UT(aclnnInplaceIndexFillTensor, INPUT(tensor_self, dim, index, fillVal), OUTPUT());

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_inplace_index_fill_tensor_test, test_index_fill_support_nchw) {
  int64_t dim = 1;
  auto tensor_self = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_NCHW)
                         .ValueRange(-10, 10)
                         .Value(vector<int32_t>{3, 4, 9, 6, 7, 11});
  auto index = IntArrayDesc(vector<int64_t>{0, 2});
  auto fillVal = ScalarDesc(static_cast<int32_t>(1));

  auto ut = OP_API_UT(aclnnInplaceIndexFillTensor, INPUT(tensor_self, dim, index, fillVal), OUTPUT());

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// dim < 0
TEST_F(l2_inplace_index_fill_tensor_test, test_index_fill_support_minus_dim) {
  int64_t dim = -1;
  auto tensor_self = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_NCHW)
                         .ValueRange(-10, 10)
                         .Value(vector<int32_t>{3, 4, 9, 6, 7, 11});
  auto index = IntArrayDesc(vector<int64_t>{0, 2});
  auto fillVal = ScalarDesc(static_cast<int32_t>(1));

  auto ut = OP_API_UT(aclnnInplaceIndexFillTensor, INPUT(tensor_self, dim, index, fillVal), OUTPUT());

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// self.dim > 8
TEST_F(l2_inplace_index_fill_tensor_test, test_index_fill_max_dim_overflow) {
  int64_t dim = 1;
  auto tensor_self = TensorDesc({2,2,2,2,2,2,2,2,2,2}, ACL_INT32, ACL_FORMAT_ND)
                         .ValueRange(-10, 10);
  auto index = IntArrayDesc(vector<int64_t>{0, 2});
  auto fillVal = ScalarDesc(static_cast<int32_t>(1));

  auto ut = OP_API_UT(aclnnInplaceIndexFillTensor, INPUT(tensor_self, dim, index, fillVal), OUTPUT());

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_inplace_index_fill_tensor_test, test_index_fill_support_bool) {
  int64_t dim = 1;
  auto tensor_self = TensorDesc({2, 3}, ACL_BOOL, ACL_FORMAT_ND)
                         .Value(vector<bool>{false, true, false, true, false, false});
  auto index = IntArrayDesc(vector<int64_t>{0, 2});
  auto fillVal = ScalarDesc(static_cast<int32_t>(1));

  auto ut = OP_API_UT(aclnnInplaceIndexFillTensor, INPUT(tensor_self, dim, index, fillVal), OUTPUT());

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_inplace_index_fill_tensor_test, Ascend910B2_test_index_fill_support_bf16) {
  int64_t dim = 1;
  auto tensor_self = TensorDesc({2, 3}, ACL_BF16, ACL_FORMAT_ND)
                         .ValueRange(-10, 10);
  auto index = IntArrayDesc(vector<int64_t>{0, 2});
  auto fillVal = ScalarDesc(static_cast<int32_t>(1));

  auto ut = OP_API_UT(aclnnInplaceIndexFillTensor, INPUT(tensor_self, dim, index, fillVal), OUTPUT());

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_inplace_index_fill_tensor_test, test_index_fill_not_support_complex64) {
  int64_t dim = 1;
  auto tensor_self = TensorDesc({2, 3}, ACL_COMPLEX64, ACL_FORMAT_ND)
                         .ValueRange(-10, 10);
  auto index = IntArrayDesc(vector<int64_t>{0, 2});
  auto fillVal = ScalarDesc(static_cast<int32_t>(1));

  auto ut = OP_API_UT(aclnnInplaceIndexFillTensor, INPUT(tensor_self, dim, index, fillVal), OUTPUT());

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_inplace_index_fill_tensor_test, test_index_fill_not_support_complex128) {
  int64_t dim = 1;
  auto tensor_self = TensorDesc({2, 3}, ACL_COMPLEX128, ACL_FORMAT_ND)
                         .ValueRange(-10, 10);
  auto index = IntArrayDesc(vector<int64_t>{0, 2});
  auto fillVal = ScalarDesc(static_cast<int32_t>(1));

  auto ut = OP_API_UT(aclnnInplaceIndexFillTensor, INPUT(tensor_self, dim, index, fillVal), OUTPUT());

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 入参self为空tensor的场景
TEST_F(l2_inplace_index_fill_tensor_test, test_index_fill_support_self_empty) {
  int64_t dim = 1;
  auto tensor_self = TensorDesc({2, 0}, ACL_INT32, ACL_FORMAT_NCHW)
                         .ValueRange(-10, 10)
                         .Value(vector<int32_t>{3, 4, 9, 6, 7, 11});
  auto index = IntArrayDesc(vector<int64_t>{0, 2});
  auto fillVal = ScalarDesc(static_cast<int32_t>(1));

  auto ut = OP_API_UT(aclnnInplaceIndexFillTensor, INPUT(tensor_self, dim, index, fillVal), OUTPUT());

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_inplace_index_fill_tensor_test, test_index_fill_nullptr) {
  int64_t dim = 1;
  auto tensor_self = nullptr;
  auto index = IntArrayDesc(vector<int64_t>{0, 2});
  auto fillVal = ScalarDesc(static_cast<int32_t>(1));

  auto ut = OP_API_UT(aclnnInplaceIndexFillTensor, INPUT(tensor_self, dim, index, fillVal), OUTPUT());

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}
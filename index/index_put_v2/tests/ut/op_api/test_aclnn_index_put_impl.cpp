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
#include "gtest/gtest.h"
#include "../../../op_api/aclnn_index_put_impl.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;
class l2_index_put_impl_test : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "l2_index_put_impl_test SetUp" << std::endl;
  }

  static void TearDownTestCase() { std::cout << "l2_index_put_impl_test TearDown" << std::endl; }
};

// 空tensor
TEST_F(l2_index_put_impl_test, l2_index_put_impl_test_001) {
  bool unsafe = false;
  bool accumulate = true;
  vector<int64_t> self = {1, 1, 0, 0};
  vector<int64_t> value = {2, 2};
  vector<int64_t> indices = {3};
  vector<int64_t> out = {1, 1, 0, 0};

  auto self_desc = TensorDesc(self, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 4);
  auto value_desc = TensorDesc(self, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 3);
  auto indices_desc = TensorDesc(indices, ACL_INT64, ACL_FORMAT_ND).ValueRange(1, 1);
  auto tensor_list_desc = TensorListDesc({indices_desc,});
  auto output_desc = TensorDesc(out, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnIndexPutImpl, INPUT(self_desc, tensor_list_desc, value_desc,
                                               accumulate, unsafe),  // host api输入
                                               OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// // 正常路径，float32
TEST_F(l2_index_put_impl_test, l2_index_put_impl_test_002) {
  bool unsafe = false;
  bool accumulate = true;
  vector<int64_t> self = {300};
  vector<int64_t> value = {3};
  vector<int64_t> indices = {3};
  vector<int64_t> out = {{300}};

  auto self_desc = TensorDesc(self, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 4);
  auto value_desc = TensorDesc(value, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 3);
  auto indices_desc = TensorDesc(indices, ACL_INT64, ACL_FORMAT_ND).ValueRange(1, 1);
  auto tensor_list_desc = TensorListDesc({indices_desc,});
  auto output_desc = TensorDesc(out, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
  
  auto ut = OP_API_UT(aclnnIndexPutImpl, INPUT(self_desc, tensor_list_desc, value_desc,
                                               accumulate, unsafe),  // host api输入
                                               OUTPUT());
  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// // 正常路径，float16
TEST_F(l2_index_put_impl_test, l2_index_put_impl_test_fp16) {
  bool unsafe = false;
  bool accumulate = true;
  vector<int64_t> self = {300};
  vector<int64_t> value = {3};
  vector<int64_t> indices = {3};
  vector<int64_t> out = {{300}};

  auto self_desc = TensorDesc(self, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(1, 4);
  auto value_desc = TensorDesc(value, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(1, 3);
  auto indices_desc = TensorDesc(indices, ACL_INT64, ACL_FORMAT_ND).ValueRange(1, 1);
  auto tensor_list_desc = TensorListDesc({indices_desc,});
  auto output_desc = TensorDesc(out, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
  
  auto ut = OP_API_UT(aclnnIndexPutImpl, INPUT(self_desc, tensor_list_desc, value_desc,
                                               accumulate, unsafe),  // host api输入
                                               OUTPUT());
  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// // 正常路径，broadcast
TEST_F(l2_index_put_impl_test, l2_index_put_impl_test_broadcast) {
  bool unsafe = false;
  bool accumulate = true;
  vector<int64_t> self = {300,30};
  vector<int64_t> value = {100,1};
  vector<int64_t> indices = {100};
  vector<int64_t> out = {{300,30}};

  auto self_desc = TensorDesc(self, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 4);
  auto value_desc = TensorDesc(value, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 3);
  auto indices_desc = TensorDesc(indices, ACL_INT64, ACL_FORMAT_ND).ValueRange(1, 1);
  auto tensor_list_desc = TensorListDesc({indices_desc,});
  auto output_desc = TensorDesc(out, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
  
  auto ut = OP_API_UT(aclnnIndexPutImpl, INPUT(self_desc, tensor_list_desc, value_desc,
                                               accumulate, unsafe),  // host api输入
                                               OUTPUT());
  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// // 正常路径，float32aicpu
TEST_F(l2_index_put_impl_test, l2_index_put_impl_test_003) {
  bool unsafe = false;
  bool accumulate = true;
  vector<int64_t> self = {300,30};
  vector<int64_t> value = {3};
  vector<int64_t> indices = {3};
  vector<int64_t> out = {{300,30}};

  auto self_desc = TensorDesc(self, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 4);
  auto value_desc = TensorDesc(value, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 3);
  auto indices_desc = TensorDesc(indices, ACL_INT64, ACL_FORMAT_ND).ValueRange(1, 1);
  auto tensor_list_desc = TensorListDesc({indices_desc,indices_desc,});
  auto output_desc = TensorDesc(out, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnIndexPutImpl, INPUT(self_desc, tensor_list_desc, value_desc,
                                               accumulate, unsafe),  // host api输入
                                               OUTPUT());
  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// // 正常路径，float64aicpu
TEST_F(l2_index_put_impl_test, l2_index_put_impl_test_004) {
  bool unsafe = false;
  bool accumulate = true;
  vector<int64_t> self = {300,200};
  vector<int64_t> value = {3,200};
  vector<int64_t> indices = {3};
  vector<int64_t> out = {{300,200}};

  auto self_desc = TensorDesc(self, ACL_BOOL, ACL_FORMAT_ND).ValueRange(1, 4);
  auto value_desc = TensorDesc(value, ACL_BOOL, ACL_FORMAT_ND).ValueRange(1, 3);
  auto indices_desc = TensorDesc(indices, ACL_INT64, ACL_FORMAT_ND).ValueRange(1, 1);
  auto tensor_list_desc = TensorListDesc({indices_desc,});
  auto output_desc = TensorDesc(out, ACL_BOOL, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnIndexPutImpl, INPUT(self_desc, tensor_list_desc, value_desc,
                                               accumulate, unsafe),  // host api输入
                                               OUTPUT());
  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// // 正常路径，indices=0
TEST_F(l2_index_put_impl_test, l2_index_put_impl_test_008) {
  bool unsafe = false;
  bool accumulate = true;
  vector<int64_t> self = {300};
  vector<int64_t> value = {0};
  vector<int64_t> indices = {0};
  vector<int64_t> out = {{300}};

  auto self_desc = TensorDesc(self, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 4);
  auto value_desc = TensorDesc(value, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 3);
  auto indices_desc = TensorDesc(indices, ACL_INT64, ACL_FORMAT_ND).ValueRange(1, 1);
  auto tensor_list_desc = TensorListDesc({indices_desc,});
  auto output_desc = TensorDesc(out, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnIndexPutImpl, INPUT(self_desc, tensor_list_desc, value_desc,
                                               accumulate, unsafe),  // host api输入
                                               OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// // 正常路径
TEST_F(l2_index_put_impl_test, l2_index_put_impl_test_009) {
  bool unsafe = false;
  bool accumulate = true;
  vector<int64_t> self = {300,30,40};
  vector<int64_t> value = {3,30,40};
  vector<int64_t> indices = {3};
  vector<int64_t> out = {{300,30,40}};

  auto self_desc = TensorDesc(self, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 4);
  auto value_desc = TensorDesc(value, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 3);
  auto indices_desc = TensorDesc(indices, ACL_INT64, ACL_FORMAT_ND).ValueRange(1, 1);
  auto tensor_list_desc = TensorListDesc({indices_desc,});
  auto output_desc = TensorDesc(out, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnIndexPutImpl, INPUT(self_desc, tensor_list_desc, value_desc,
                                               accumulate, unsafe),  // host api输入
                                               OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// CheckDtypeEqual
TEST_F(l2_index_put_impl_test, l2_index_put_impl_test_dtype) {
  bool unsafe = false;
  bool accumulate = true;
  vector<int64_t> self = {300,30};
  vector<int64_t> value = {3};
  vector<int64_t> indices = {3};
  vector<int64_t> out = {{300,30}};

  auto self_desc = TensorDesc(self, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(1, 4);
  auto value_desc = TensorDesc(value, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 3);
  auto indices_desc = TensorDesc(indices, ACL_INT64, ACL_FORMAT_ND).ValueRange(1, 1);
  auto tensor_list_desc = TensorListDesc({indices_desc,indices_desc,});
  auto output_desc = TensorDesc(out, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnIndexPutImpl, INPUT(self_desc, tensor_list_desc, value_desc,
                                               accumulate, unsafe),  // host api输入
                                               OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// CheckNotNull_1
TEST_F(l2_index_put_impl_test, l2_index_put_impl_test_nullptr_self) {
  bool unsafe = false;
  bool accumulate = true;
  vector<int64_t> self = {300,30};
  vector<int64_t> value = {3};
  vector<int64_t> indices = {3};
  vector<int64_t> out = {{300,30}};

  auto self_desc = nullptr;
  auto value_desc = TensorDesc(value, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 3);
  auto indices_desc = TensorDesc(indices, ACL_INT64, ACL_FORMAT_ND).ValueRange(1, 1);
  auto tensor_list_desc = TensorListDesc({indices_desc,indices_desc,});
  auto output_desc = TensorDesc(out, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnIndexPutImpl, INPUT(self_desc, tensor_list_desc, value_desc,
                                               accumulate, unsafe),  // host api输入
                                               OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckNotNull_1
TEST_F(l2_index_put_impl_test, l2_index_put_impl_test_nullptr_value) {
  bool unsafe = false;
  bool accumulate = true;
  vector<int64_t> self = {300,30};
  vector<int64_t> value = {3};
  vector<int64_t> indices = {3};
  vector<int64_t> out = {{300,30}};

  auto self_desc = TensorDesc(self, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 4);
  auto value_desc = nullptr;
  auto indices_desc = TensorDesc(indices, ACL_INT64, ACL_FORMAT_ND).ValueRange(1, 1);
  auto tensor_list_desc = TensorListDesc({indices_desc,indices_desc,});
  auto output_desc = TensorDesc(out, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnIndexPutImpl, INPUT(self_desc, tensor_list_desc, value_desc,
                                               accumulate, unsafe),  // host api输入
                                               OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckDtypeValid
TEST_F(l2_index_put_impl_test, l2_index_put_impl_test_dtype_valid) {
  bool unsafe = false;
  bool accumulate = true;
  vector<int64_t> self = {300,30};
  vector<int64_t> value = {3};
  vector<int64_t> indices = {3};
  vector<int64_t> out = {{300,30}};

  auto self_desc = TensorDesc(self, ACL_COMPLEX64, ACL_FORMAT_ND).ValueRange(1, 4);
  auto value_desc = TensorDesc(value, ACL_COMPLEX64, ACL_FORMAT_ND).ValueRange(1, 3);
  auto indices_desc = TensorDesc(indices, ACL_INT64, ACL_FORMAT_ND).ValueRange(1, 1);
  auto tensor_list_desc = TensorListDesc({indices_desc,indices_desc,});
  auto output_desc = TensorDesc(out, ACL_COMPLEX64, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnIndexPutImpl, INPUT(self_desc, tensor_list_desc, value_desc,
                                               accumulate, unsafe),  // host api输入
                                               OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// CheckDtypeValid
TEST_F(l2_index_put_impl_test, l2_index_put_impl_test_dtype_valid_value) {
  bool unsafe = false;
  bool accumulate = true;
  vector<int64_t> self = {300,30};
  vector<int64_t> value = {3};
  vector<int64_t> indices = {3};
  vector<int64_t> out = {{300,30}};

  auto self_desc = TensorDesc(self, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 4);
  auto value_desc = TensorDesc(value, ACL_COMPLEX64, ACL_FORMAT_ND).ValueRange(1, 3);
  auto indices_desc = TensorDesc(indices, ACL_INT64, ACL_FORMAT_ND).ValueRange(1, 1);
  auto tensor_list_desc = TensorListDesc({indices_desc,indices_desc,});
  auto output_desc = TensorDesc(out, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnIndexPutImpl, INPUT(self_desc, tensor_list_desc, value_desc,
                                               accumulate, unsafe),  // host api输入
                                               OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// indices=0
TEST_F(l2_index_put_impl_test, l2_index_put_impl_test_none_indices) {
  bool unsafe = false;
  bool accumulate = true;
  vector<int64_t> self = {300,30};
  vector<int64_t> value = {300,3};
  vector<int64_t> indices = {0};
  vector<int64_t> out = {{300,30}};

  auto self_desc = TensorDesc(self, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 4);
  auto value_desc = TensorDesc(value, ACL_COMPLEX64, ACL_FORMAT_ND).ValueRange(1, 3);
  auto indices_desc = TensorDesc(indices, ACL_INT64, ACL_FORMAT_ND).ValueRange(1, 1);
  auto tensor_list_desc = TensorListDesc({indices_desc,});
  auto output_desc = TensorDesc(out, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnIndexPutImpl, INPUT(self_desc, tensor_list_desc, value_desc,
                                               accumulate, unsafe),  // host api输入
                                               OUTPUT());
  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// indices_1 is 0
TEST_F(l2_index_put_impl_test, l2_index_put_impl_test_none_indices_2) {
  bool unsafe = false;
  bool accumulate = true;
  vector<int64_t> self = {300,30};
  vector<int64_t> value = {300,3};
  vector<int64_t> indices = {3};
  vector<int64_t> indices_null = {0};
  vector<int64_t> out = {{300,30}};

  auto self_desc = TensorDesc(self, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 4);
  auto value_desc = TensorDesc(value, ACL_COMPLEX64, ACL_FORMAT_ND).ValueRange(1, 3);
  auto indices_desc = TensorDesc(indices, ACL_INT64, ACL_FORMAT_ND).ValueRange(1, 1);
  auto indices_null_desc = TensorDesc(indices_null, ACL_INT64, ACL_FORMAT_ND).ValueRange(1, 1);
  auto tensor_list_desc = TensorListDesc({indices_desc, indices_null_desc});
  auto output_desc = TensorDesc(out, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnIndexPutImpl, INPUT(self_desc, tensor_list_desc, value_desc,
                                               accumulate, unsafe),  // host api输入
                                               OUTPUT());
  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// // 正常路径
TEST_F(l2_index_put_impl_test, l2_index_put_impl_test_013) {
  bool unsafe = false;
  bool accumulate = true;
  vector<int64_t> self = {300,90};
  vector<int64_t> value = {60,90};
  vector<int64_t> indices = {60};
  vector<int64_t> out = {{300,90}};

  auto self_desc = TensorDesc(self, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 4);
  auto value_desc = TensorDesc(value, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 3);
  auto indices_desc = TensorDesc(indices, ACL_INT64, ACL_FORMAT_ND).ValueRange(1, 1);
  auto tensor_list_desc = TensorListDesc({indices_desc,});
  auto output_desc = TensorDesc(out, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnIndexPutImpl, INPUT(self_desc, tensor_list_desc, value_desc,
                                               accumulate, unsafe),  // host api输入
                                               OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_index_put_impl_test, l2_index_put_unsupport_dtype_for_self) {
  bool unsafe = false;
  bool accumulate = true;
  vector<int64_t> self = {300,30,40};
  vector<int64_t> value = {3,40};
  vector<int64_t> indices = {3};
  vector<int64_t> out = {{300,30,40}};

  auto self_desc = TensorDesc(self, ACL_BF16, ACL_FORMAT_ND).ValueRange(1, 4);
  auto value_desc = TensorDesc(value, ACL_INT16, ACL_FORMAT_ND).ValueRange(1, 3);
  auto indices_desc = TensorDesc(indices, ACL_INT64, ACL_FORMAT_ND).ValueRange(1, 1);
  auto indices_desc2 = TensorDesc(indices, ACL_BOOL, ACL_FORMAT_ND).ValueRange(1, 1);
  auto tensor_list_desc = TensorListDesc({indices_desc, indices_desc2,});
  auto output_desc = TensorDesc(out, ACL_INT16, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnIndexPutImpl, INPUT(self_desc, tensor_list_desc, value_desc,
                                               accumulate, unsafe),  // host api输入
                                               OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_index_put_impl_test, l2_index_put_unsupport_dtype_for_value) {
  bool unsafe = false;
  bool accumulate = true;
  vector<int64_t> self = {300,30,40};
  vector<int64_t> value = {3,40};
  vector<int64_t> indices = {3};
  vector<int64_t> out = {{300,30,40}};

  auto self_desc = TensorDesc(self, ACL_INT16, ACL_FORMAT_ND).ValueRange(1, 4);
  auto value_desc = TensorDesc(value, ACL_BF16, ACL_FORMAT_ND).ValueRange(1, 3);
  auto indices_desc = TensorDesc(indices, ACL_INT64, ACL_FORMAT_ND).ValueRange(1, 1);
  auto indices_desc2 = TensorDesc(indices, ACL_BOOL, ACL_FORMAT_ND).ValueRange(1, 1);
  auto tensor_list_desc = TensorListDesc({indices_desc, indices_desc2,});
  auto output_desc = TensorDesc(out, ACL_INT16, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnIndexPutImpl, INPUT(self_desc, tensor_list_desc, value_desc,
                                               accumulate, unsafe),  // host api输入
                                               OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 正常路径
TEST_F(l2_index_put_impl_test, ascend910B2_l2_index_put__non_head_broadcast) {
  bool unsafe = false;
  bool accumulate = false;
  vector<int64_t> self = {1, 4, 16, 32, 48};
  vector<int64_t> value = {4, 1, 32, 48};
  vector<int64_t> indices_null = {0};
  vector<int64_t> indices = {1};
  vector<int64_t> out = {{1, 4, 16, 32, 48}};

  auto self_desc = TensorDesc(self, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 4);
  auto value_desc = TensorDesc(value, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 3);
  auto indices_desc_null = TensorDesc(indices_null, ACL_INT64, ACL_FORMAT_ND).ValueRange(1, 1);
  auto indices_desc = TensorDesc(indices, ACL_INT64, ACL_FORMAT_ND).ValueRange(1, 1);
  auto tensor_list_desc = TensorListDesc({indices_desc_null, indices_desc_null, indices_desc,});
  auto output_desc = TensorDesc(out, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnIndexPutImpl, INPUT(self_desc, tensor_list_desc, value_desc,
                                               accumulate, unsafe),  // host api输入
                                               OUTPUT());

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// // 正常路径，int64，索引int32
TEST_F(l2_index_put_impl_test, l2_index_put_impl_test_int64_idx_int32) {
  bool unsafe = false;
  bool accumulate = true;
  vector<int64_t> self = {300};
  vector<int64_t> value = {3};
  vector<int64_t> indices = {3};
  vector<int64_t> out = {{300}};

  auto self_desc = TensorDesc(self, ACL_INT64, ACL_FORMAT_ND).ValueRange(1, 4);
  auto value_desc = TensorDesc(value, ACL_INT64, ACL_FORMAT_ND).ValueRange(1, 3);
  auto indices_desc = TensorDesc(indices, ACL_INT32, ACL_FORMAT_ND).ValueRange(1, 1);
  auto tensor_list_desc = TensorListDesc({indices_desc,});
  auto output_desc = TensorDesc(out, ACL_INT64, ACL_FORMAT_ND);
  
  auto ut = OP_API_UT(aclnnIndexPutImpl, INPUT(self_desc, tensor_list_desc, value_desc,
                                               accumulate, unsafe),  // host api输入
                                               OUTPUT());
  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// // 正常路径，int64，索引int64
TEST_F(l2_index_put_impl_test, l2_index_put_impl_test_int64_idx_int64) {
  bool unsafe = false;
  bool accumulate = true;
  vector<int64_t> self = {300};
  vector<int64_t> value = {3};
  vector<int64_t> indices = {3};
  vector<int64_t> out = {{300}};

  auto self_desc = TensorDesc(self, ACL_INT64, ACL_FORMAT_ND).ValueRange(1, 4);
  auto value_desc = TensorDesc(value, ACL_INT64, ACL_FORMAT_ND).ValueRange(1, 3);
  auto indices_desc = TensorDesc(indices, ACL_INT64, ACL_FORMAT_ND).ValueRange(1, 1);
  auto tensor_list_desc = TensorListDesc({indices_desc,});
  auto output_desc = TensorDesc(out, ACL_INT64, ACL_FORMAT_ND);
  
  auto ut = OP_API_UT(aclnnIndexPutImpl, INPUT(self_desc, tensor_list_desc, value_desc,
                                               accumulate, unsafe),  // host api输入
                                               OUTPUT());
  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// // 正常路径，int64，索引int64
TEST_F(l2_index_put_impl_test, Ascend910_9589_case_001) {
  bool unsafe = false;
  bool accumulate = true;
  vector<int64_t> self = {300};
  vector<int64_t> value = {3};
  vector<int64_t> indices = {3};
  vector<int64_t> out = {{300}};

  auto self_desc = TensorDesc(self, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 4);
  auto value_desc = TensorDesc(value, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 3);
  auto indices_desc = TensorDesc(indices, ACL_INT64, ACL_FORMAT_ND).ValueRange(1, 1);
  auto tensor_list_desc = TensorListDesc({indices_desc,});
  auto output_desc = TensorDesc(out, ACL_FLOAT, ACL_FORMAT_ND);
  
  auto ut = OP_API_UT(aclnnIndexPutImpl, INPUT(self_desc, tensor_list_desc, value_desc,
                                               accumulate, unsafe),  // host api输入
                                               OUTPUT());
  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// // 正常路径，int64，索引int64
TEST_F(l2_index_put_impl_test, Ascend910_9589_case_003_accFalse) {
  bool unsafe = false;
  bool accumulate = false;
  vector<int64_t> self = {300};
  vector<int64_t> value = {3};
  vector<int64_t> indices = {3};
  vector<int64_t> out = {{300}};

  auto self_desc = TensorDesc(self, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 4);
  auto value_desc = TensorDesc(value, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 3);
  auto indices_desc = TensorDesc(indices, ACL_INT64, ACL_FORMAT_ND).ValueRange(1, 1);
  auto tensor_list_desc = TensorListDesc({indices_desc,});
  auto output_desc = TensorDesc(out, ACL_FLOAT, ACL_FORMAT_ND);
  
  auto ut = OP_API_UT(aclnnIndexPutImpl, INPUT(self_desc, tensor_list_desc, value_desc,
                                               accumulate, unsafe),  // host api输入
                                               OUTPUT());
  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// // 正常路径，int64，索引int64
TEST_F(l2_index_put_impl_test, Ascend910_9589_case_006_accFalse) {
  bool unsafe = false;
  bool accumulate = false;
  vector<int64_t> self = {300, 300};
  vector<int64_t> value = {1, 1};
  vector<int64_t> indices = {3};
  vector<int64_t> out = {{300, 300}};

  auto self_desc = TensorDesc(self, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 4);
  auto value_desc = TensorDesc(value, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 3);
  auto indices_desc0 = TensorDesc({0}, ACL_INT64, ACL_FORMAT_ND).ValueRange(1, 1);
  auto indices_desc = TensorDesc(indices, ACL_INT64, ACL_FORMAT_ND).ValueRange(1, 1);
  auto tensor_list_desc = TensorListDesc({indices_desc, indices_desc0,});
  auto output_desc = TensorDesc(out, ACL_FLOAT, ACL_FORMAT_ND);
  
  auto ut = OP_API_UT(aclnnIndexPutImpl, INPUT(self_desc, tensor_list_desc, value_desc,
                                               accumulate, unsafe),  // host api输入
                                               OUTPUT());
  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// // 正常路径，int64，索引int64
TEST_F(l2_index_put_impl_test, Ascend910_9589_case_007_accTrue) {
  bool unsafe = false;
  bool accumulate = true;
  vector<int64_t> self = {300, 300};
  vector<int64_t> value = {1, 1};
  vector<int64_t> indices = {3};
  vector<int64_t> out = {{300, 300}};

  auto self_desc = TensorDesc(self, ACL_BOOL, ACL_FORMAT_ND).ValueRange(1, 4);
  auto value_desc = TensorDesc(value, ACL_BOOL, ACL_FORMAT_ND).ValueRange(1, 3);
  auto indices_desc0 = TensorDesc({0}, ACL_INT64, ACL_FORMAT_ND).ValueRange(1, 1);
  auto indices_desc = TensorDesc(indices, ACL_INT64, ACL_FORMAT_ND).ValueRange(1, 1);
  auto tensor_list_desc = TensorListDesc({indices_desc, indices_desc0,});
  auto output_desc = TensorDesc(out, ACL_BOOL, ACL_FORMAT_ND);
  
  auto ut = OP_API_UT(aclnnIndexPutImpl, INPUT(self_desc, tensor_list_desc, value_desc,
                                               accumulate, unsafe),  // host api输入
                                               OUTPUT());
  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// // 正常路径，int64，索引int64, 部分索引
TEST_F(l2_index_put_impl_test, Ascend910_9589_case_008_accTrue) {
  bool unsafe = false;
  bool accumulate = true;
  vector<int64_t> self = {300, 300};
  vector<int64_t> value = {3, 300};
  vector<int64_t> indices = {3};
  vector<int64_t> out = {{300, 300}};

  auto self_desc = TensorDesc(self, ACL_BOOL, ACL_FORMAT_ND).ValueRange(1, 4);
  auto value_desc = TensorDesc(value, ACL_BOOL, ACL_FORMAT_ND).ValueRange(1, 3);
  auto indices_desc = TensorDesc(indices, ACL_INT64, ACL_FORMAT_ND).ValueRange(1, 1);
  auto tensor_list_desc = TensorListDesc({indices_desc,});
  auto output_desc = TensorDesc(out, ACL_BOOL, ACL_FORMAT_ND);
  
  auto ut = OP_API_UT(aclnnIndexPutImpl, INPUT(self_desc, tensor_list_desc, value_desc,
                                               accumulate, unsafe),  // host api输入
                                               OUTPUT());
  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}
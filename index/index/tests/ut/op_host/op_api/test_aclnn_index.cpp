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
#include "level2/aclnn_index.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"

class l2_index_test : public testing::Test {
protected:
  static void SetUpTestCase() {
    std::cout << "l2_index_test SetUp" << std::endl;
  }

  static void TearDownTestCase() { std::cout << "l2_index_test TearDown" << std::endl; }
};

TEST_F(l2_index_test, l2_test_success_fp32) {
  vector<int64_t> self_shape = {30, 10};
  vector<int64_t> indices_shape = {5};
  vector<int64_t> out_shape = {{5, 10}};

  auto self_desc = TensorDesc(self_shape, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 4);
  auto indices_desc = TensorDesc(indices_shape, ACL_INT64, ACL_FORMAT_ND).ValueRange(1, 25);
  auto tensor_list_desc = TensorListDesc({indices_desc,});
  auto out_desc = TensorDesc(out_shape, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnIndex,
                      INPUT(self_desc, tensor_list_desc),  // host api输入
                      OUTPUT(out_desc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
  ut.TestPrecision();
}

TEST_F(l2_index_test, l2_test_self_empty_fp32) {
  vector<int64_t> self_shape = {0, 10};
  vector<int64_t> indices_shape = {5};
  vector<int64_t> out_shape = {{5, 10}};

  auto self_desc = TensorDesc(self_shape, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 4);
  auto indices_desc = TensorDesc(indices_shape, ACL_INT64, ACL_FORMAT_ND).ValueRange(1, 25);
  auto tensor_list_desc = TensorListDesc({indices_desc,});
  auto out_desc = TensorDesc(out_shape, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnIndex,
                      INPUT(self_desc, tensor_list_desc),  // host api输入
                      OUTPUT(out_desc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_index_test, l2_test_out_empty_fp32) {
  vector<int64_t> self_shape = {30, 10};
  vector<int64_t> indices_shape = {5};
  vector<int64_t> out_shape = {{5, 0}};

  auto self_desc = TensorDesc(self_shape, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 4);
  auto indices_desc = TensorDesc(indices_shape, ACL_INT64, ACL_FORMAT_ND).ValueRange(1, 25);
  auto tensor_list_desc = TensorListDesc({indices_desc,});
  auto out_desc = TensorDesc(out_shape, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnIndex,
                      INPUT(self_desc, tensor_list_desc),  // host api输入
                      OUTPUT(out_desc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_index_test, l2_test_success_multi_indices_aicore) {
  vector<int64_t> self_shape = {30, 30};
  vector<int64_t> indices_shape = {5};
  vector<int64_t> out_shape = {{5}};

  auto self_desc = TensorDesc(self_shape, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 4);
  auto indices_desc = TensorDesc(indices_shape, ACL_INT64, ACL_FORMAT_ND).ValueRange(1, 25);
  auto tensor_list_desc = TensorListDesc({indices_desc, indices_desc});
  auto out_desc = TensorDesc(out_shape, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnIndex,
                      INPUT(self_desc, tensor_list_desc),  // host api输入
                      OUTPUT(out_desc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
  ut.TestPrecision();
}

TEST_F(l2_index_test, l2_test_success_multi_indices_aicpu) {
  vector<int64_t> self_shape = {30, 30};
  vector<int64_t> indices_shape = {500001};
  vector<int64_t> out_shape = {{500001}};

  auto self_desc = TensorDesc(self_shape, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 4);
  auto indices_desc = TensorDesc(indices_shape, ACL_INT64, ACL_FORMAT_ND).ValueRange(1, 25);
  auto tensor_list_desc = TensorListDesc({indices_desc, indices_desc});
  auto out_desc = TensorDesc(out_shape, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnIndex,
                      INPUT(self_desc, tensor_list_desc),  // host api输入
                      OUTPUT(out_desc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_index_test, l2_test_success_fp16) {
  vector<int64_t> self_shape = {30, 10};
  vector<int64_t> indices_shape = {5};
  vector<int64_t> out_shape = {{5, 10}};

  auto self_desc = TensorDesc(self_shape, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(1, 4);
  auto indices_desc = TensorDesc(indices_shape, ACL_INT64, ACL_FORMAT_ND).ValueRange(1, 25);
  auto tensor_list_desc = TensorListDesc({indices_desc,});
  auto out_desc = TensorDesc(out_shape, ACL_FLOAT16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnIndex,
                      INPUT(self_desc, tensor_list_desc),  // host api输入
                      OUTPUT(out_desc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
  ut.TestPrecision();
}

TEST_F(l2_index_test, l2_test_success_int64) {
  vector<int64_t> self_shape = {30, 10};
  vector<int64_t> indices_shape = {5};
  vector<int64_t> out_shape = {{5, 10}};

  auto self_desc = TensorDesc(self_shape, ACL_INT64, ACL_FORMAT_ND).ValueRange(1, 4);
  auto indices_desc = TensorDesc(indices_shape, ACL_INT64, ACL_FORMAT_ND).ValueRange(1, 25);
  auto tensor_list_desc = TensorListDesc({indices_desc,});
  auto out_desc = TensorDesc(out_shape, ACL_INT64, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnIndex,
                      INPUT(self_desc, tensor_list_desc),  // host api输入
                      OUTPUT(out_desc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
  ut.TestPrecision();
}

TEST_F(l2_index_test, l2_test_success_int32) {
  vector<int64_t> self_shape = {30, 10};
  vector<int64_t> indices_shape = {5};
  vector<int64_t> out_shape = {{5, 10}};

  auto self_desc = TensorDesc(self_shape, ACL_INT32, ACL_FORMAT_ND).ValueRange(1, 4);
  auto indices_desc = TensorDesc(indices_shape, ACL_INT64, ACL_FORMAT_ND).ValueRange(1, 25);
  auto tensor_list_desc = TensorListDesc({indices_desc,});
  auto out_desc = TensorDesc(out_shape, ACL_INT32, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnIndex,
                      INPUT(self_desc, tensor_list_desc),  // host api输入
                      OUTPUT(out_desc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
  ut.TestPrecision();
}

TEST_F(l2_index_test, l2_test_success_int8) {
  vector<int64_t> self_shape = {30, 10};
  vector<int64_t> indices_shape = {5};
  vector<int64_t> out_shape = {{5, 10}};

  auto self_desc = TensorDesc(self_shape, ACL_INT8, ACL_FORMAT_ND).ValueRange(1, 4);
  auto indices_desc = TensorDesc(indices_shape, ACL_INT64, ACL_FORMAT_ND).ValueRange(1, 25);
  auto tensor_list_desc = TensorListDesc({indices_desc,});
  auto out_desc = TensorDesc(out_shape, ACL_INT8, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnIndex,
                      INPUT(self_desc, tensor_list_desc),  // host api输入
                      OUTPUT(out_desc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
  ut.TestPrecision();
}

TEST_F(l2_index_test, l2_test_success_bool) {
  vector<int64_t> self_shape = {30, 10};
  vector<int64_t> indices_shape = {5};
  vector<int64_t> out_shape = {{5, 10}};

  auto self_desc = TensorDesc(self_shape, ACL_BOOL, ACL_FORMAT_ND).ValueRange(1, 4);
  auto indices_desc = TensorDesc(indices_shape, ACL_INT64, ACL_FORMAT_ND).ValueRange(1, 25);
  auto tensor_list_desc = TensorListDesc({indices_desc,});
  auto out_desc = TensorDesc(out_shape, ACL_BOOL, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnIndex,
                      INPUT(self_desc, tensor_list_desc),  // host api输入
                      OUTPUT(out_desc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
  ut.TestPrecision();
}

TEST_F(l2_index_test, l2_test_success_uint8) {
  vector<int64_t> self_shape = {30, 10};
  vector<int64_t> indices_shape = {5};
  vector<int64_t> out_shape = {{5, 10}};

  auto self_desc = TensorDesc(self_shape, ACL_UINT8, ACL_FORMAT_ND).ValueRange(1, 4);
  auto indices_desc = TensorDesc(indices_shape, ACL_INT64, ACL_FORMAT_ND).ValueRange(1, 25);
  auto tensor_list_desc = TensorListDesc({indices_desc,});
  auto out_desc = TensorDesc(out_shape, ACL_UINT8, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnIndex,
                      INPUT(self_desc, tensor_list_desc),  // host api输入
                      OUTPUT(out_desc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
  ut.TestPrecision();
}

TEST_F(l2_index_test, l2_test_failed_nullptr) {
  vector<int64_t> self_shape = {30, 10};
  vector<int64_t> indices_shape = {5};
  vector<int64_t> out_shape = {{5, 10}};

  auto self_desc = TensorDesc(self_shape, ACL_INT32, ACL_FORMAT_ND).ValueRange(1, 4);
  auto indices_desc = TensorDesc(indices_shape, ACL_INT64, ACL_FORMAT_ND).ValueRange(1, 25);
  auto tensor_list_desc = TensorListDesc({indices_desc,});
  auto out_desc = TensorDesc(out_shape, ACL_INT32, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnIndex,
                      INPUT(nullptr, tensor_list_desc),  // host api输入
                      OUTPUT(out_desc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_index_test, l2_test_failed_unmatch_dtype) {
  vector<int64_t> self_shape = {30, 10};
  vector<int64_t> indices_shape = {5};
  vector<int64_t> out_shape = {{5, 10}};

  auto self_desc = TensorDesc(self_shape, ACL_INT8, ACL_FORMAT_ND).ValueRange(1, 4);
  auto indices_desc = TensorDesc(indices_shape, ACL_INT64, ACL_FORMAT_ND).ValueRange(1, 25);
  auto tensor_list_desc = TensorListDesc({indices_desc,});
  auto out_desc = TensorDesc(out_shape, ACL_INT32, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnIndex,
                      INPUT(self_desc, tensor_list_desc),  // host api输入
                      OUTPUT(out_desc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_index_test, l2_test_success_self_double_aicpu) {
  vector<int64_t> self_shape = {30, 10};
  vector<int64_t> indices_shape = {5};
  vector<int64_t> out_shape = {{5, 10}};

  auto self_desc = TensorDesc(self_shape, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(1, 4);
  auto indices_desc = TensorDesc(indices_shape, ACL_INT64, ACL_FORMAT_ND).ValueRange(1, 25);
  auto tensor_list_desc = TensorListDesc({indices_desc,});
  auto out_desc = TensorDesc(out_shape, ACL_DOUBLE, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnIndex,
                      INPUT(self_desc, tensor_list_desc),  // host api输入
                      OUTPUT(out_desc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
  // ut.TestPrecision();
}

TEST_F(l2_index_test, l2_test_success_indices_bool_aicpu) {
  vector<int64_t> self_shape = {30, 10};
  vector<int64_t> indices_shape = {5};
  vector<int64_t> out_shape = {{5, 10}};

  auto self_desc = TensorDesc(self_shape, ACL_INT8, ACL_FORMAT_ND).ValueRange(1, 4);
  auto indices_desc = TensorDesc(indices_shape, ACL_BOOL, ACL_FORMAT_ND).ValueRange(1, 25);
  auto tensor_list_desc = TensorListDesc({indices_desc,});
  auto out_desc = TensorDesc(out_shape, ACL_INT8, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnIndex,
                      INPUT(self_desc, tensor_list_desc),  // host api输入
                      OUTPUT(out_desc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
  // ut.TestPrecision();
}

TEST_F(l2_index_test, l2_test_success_single_indices_int_aicore) {
  vector<int64_t> self_shape = {600000, 80};
  vector<int64_t> indices_shape = {5000};
  vector<int64_t> out_shape = {{5000, 80}};

  auto self_desc = TensorDesc(self_shape, ACL_INT8, ACL_FORMAT_ND).ValueRange(1, 4);
  auto indices_desc = TensorDesc(indices_shape, ACL_INT64, ACL_FORMAT_ND).ValueRange(1, 25);
  auto tensor_list_desc = TensorListDesc({indices_desc,});
  auto out_desc = TensorDesc(out_shape, ACL_INT8, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnIndex,
                      INPUT(self_desc, tensor_list_desc),  // host api输入
                      OUTPUT(out_desc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
  ut.TestPrecision();
}

TEST_F(l2_index_test, l2_test_failed_unsupport_dtype) {
  vector<int64_t> self_shape = {600000, 80};
  vector<int64_t> indices_shape = {5000};
  vector<int64_t> out_shape = {{5000, 80}};

  auto self_desc = TensorDesc(self_shape, ACL_BF16, ACL_FORMAT_ND).ValueRange(1, 4);
  auto indices_desc = TensorDesc(indices_shape, ACL_INT64, ACL_FORMAT_ND).ValueRange(1, 25);
  auto tensor_list_desc = TensorListDesc({indices_desc,});
  auto out_desc = TensorDesc(out_shape, ACL_BF16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnIndex,
                      INPUT(self_desc, tensor_list_desc),  // host api输入
                      OUTPUT(out_desc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_index_test, ascend910B2_l2_test_success_bf16_aicore) {
  vector<int64_t> self_shape = {600000, 80};
  vector<int64_t> indices_shape = {5000};
  vector<int64_t> out_shape = {{5000, 80}};

  auto self_desc = TensorDesc(self_shape, ACL_BF16, ACL_FORMAT_ND).ValueRange(1, 4);
  auto indices_desc = TensorDesc(indices_shape, ACL_INT64, ACL_FORMAT_ND).ValueRange(1, 25);
  auto tensor_list_desc = TensorListDesc({indices_desc,});
  auto out_desc = TensorDesc(out_shape, ACL_BF16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnIndex,
                      INPUT(self_desc, tensor_list_desc),  // host api输入
                      OUTPUT(out_desc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
  ut.TestPrecision();
}

TEST_F(l2_index_test, ascend910B2_l2_test_double_aicpu) {
  vector<int64_t> self_shape = {600000, 80};
  vector<int64_t> indices_shape = {5000};
  vector<int64_t> out_shape = {{5000, 80}};

  auto self_desc = TensorDesc(self_shape, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(1, 4);
  auto indices_desc = TensorDesc(indices_shape, ACL_INT64, ACL_FORMAT_ND).ValueRange(1, 25);
  auto tensor_list_desc = TensorListDesc({indices_desc,});
  auto out_desc = TensorDesc(out_shape, ACL_DOUBLE, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnIndex,
                      INPUT(self_desc, tensor_list_desc),  // host api输入
                      OUTPUT(out_desc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_index_test, ascend910B2_l2_test_partial_continous) {
  vector<int64_t> self_shape = {120, 196, 7};
  vector<int64_t> indices_shape = {9000};
  vector<int64_t> out_shape = {{9000, 7}};

  auto self_desc = TensorDesc(self_shape, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 4);
  auto indices_desc = TensorDesc(indices_shape, ACL_INT64, ACL_FORMAT_ND).ValueRange(1, 25);
  auto tensor_list_desc = TensorListDesc({indices_desc, indices_desc});
  auto out_desc = TensorDesc(out_shape, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnIndex,
                      INPUT(self_desc, tensor_list_desc),  // host api输入
                      OUTPUT(out_desc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_index_test, ascend910B2_l2_test_non_first_axis) {
  vector<int64_t> self_shape = {120, 197, 4096};
  vector<int64_t> indices_shape = {1};
  vector<int64_t> empty_shape = {0};
  vector<int64_t> out_shape = {{120, 1, 4096}};

  auto self_desc = TensorDesc(self_shape, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 4);
  auto empty_indices_desc = TensorDesc(empty_shape, ACL_INT64, ACL_FORMAT_ND).ValueRange(1, 25);
  auto indices_desc = TensorDesc(indices_shape, ACL_INT64, ACL_FORMAT_ND).ValueRange(1, 25);
  auto tensor_list_desc = TensorListDesc({empty_indices_desc, indices_desc});
  auto out_desc = TensorDesc(out_shape, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnIndex,
                      INPUT(self_desc, tensor_list_desc),  // host api输入
                      OUTPUT(out_desc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_index_test, ascend910_95_l2_test_non_first_axis) {
  vector<int64_t> self_shape = {120, 197, 4096};
  vector<int64_t> indices_shape = {1};
  vector<int64_t> empty_shape = {0};
  vector<int64_t> out_shape = {{120, 1, 4096}};

  auto self_desc = TensorDesc(self_shape, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(1, 4);
  auto empty_indices_desc = TensorDesc(empty_shape, ACL_INT64, ACL_FORMAT_ND).ValueRange(1, 25);
  auto indices_desc = TensorDesc(indices_shape, ACL_INT64, ACL_FORMAT_ND).ValueRange(1, 25);
  auto tensor_list_desc = TensorListDesc({empty_indices_desc, indices_desc});
  auto out_desc = TensorDesc(out_shape, ACL_DOUBLE, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnIndex,
                      INPUT(self_desc, tensor_list_desc),  // host api输入
                      OUTPUT(out_desc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_index_test, ascend910B2_l2_test_non_first_axis_small_tail) {
  vector<int64_t> self_shape = {512, 26, 16};
  vector<int64_t> indices_shape = {2496};
  vector<int64_t> empty_shape = {0};
  vector<int64_t> out_shape = {{512, 2496}};

  auto self_desc = TensorDesc(self_shape, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 4);
  auto empty_indices_desc = TensorDesc(empty_shape, ACL_INT64, ACL_FORMAT_ND).ValueRange(1, 25);
  auto indices_desc = TensorDesc(indices_shape, ACL_INT64, ACL_FORMAT_ND).ValueRange(1, 25);
  auto tensor_list_desc = TensorListDesc({empty_indices_desc, indices_desc, indices_desc});
  auto out_desc = TensorDesc(out_shape, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnIndex,
                      INPUT(self_desc, tensor_list_desc),  // host api输入
                      OUTPUT(out_desc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_index_test, l2_test_failed_unsupport_format) {
  vector<int64_t> self_shape = {30, 10};
  vector<int64_t> indices_shape = {5};
  vector<int64_t> out_shape = {{5, 10}};

  auto self_desc = TensorDesc(self_shape, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 4);
  auto indices_desc = TensorDesc(indices_shape, ACL_INT64, ACL_FORMAT_FRACTAL_NZ).ValueRange(1, 25);
  auto tensor_list_desc = TensorListDesc({indices_desc,});
  auto out_desc = TensorDesc(out_shape, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnIndex,
                      INPUT(self_desc, tensor_list_desc),  // host api输入
                      OUTPUT(out_desc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_index_test, ascend910_95_l2_test_success_fp32) {
  vector<int64_t> self_shape = {30, 10};
  vector<int64_t> indices_shape = {5};
  vector<int64_t> out_shape = {{5, 10}};

  auto self_desc = TensorDesc(self_shape, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 4);
  auto indices_desc = TensorDesc(indices_shape, ACL_INT64, ACL_FORMAT_ND).ValueRange(1, 25);
  auto tensor_list_desc = TensorListDesc({indices_desc,});
  auto out_desc = TensorDesc(out_shape, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnIndex,
                      INPUT(self_desc, tensor_list_desc),  // host api输入
                      OUTPUT(out_desc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
  ut.TestPrecision();
}
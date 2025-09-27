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

#include "level2/aclnn_max_unpool2d_backward.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace op;
using namespace std;

class l2_max_unpool2d_backward_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "l2_max_unpool2d_backward_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "l2_max_unpool2d_backward_test TearDown" << std::endl;
  }
};

TEST_F(l2_max_unpool2d_backward_test, case_001) {
  auto grad_desc = TensorDesc({2, 16, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
  auto self_desc = TensorDesc({2, 16, 5, 5}, ACL_FLOAT, ACL_FORMAT_NCHW);
  auto indices_desc = TensorDesc({2, 16, 5, 5}, ACL_INT64, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto output_szie = IntArrayDesc(vector<int64_t>{2, 3});
  auto out_desc = TensorDesc(self_desc);

  auto ut =
      OP_API_UT(aclnnMaxUnpool2dBackward, INPUT(grad_desc, self_desc, indices_desc, output_szie), OUTPUT(out_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

TEST_F(l2_max_unpool2d_backward_test, case_002) {
  auto grad_desc = TensorDesc({7, 2, 3}, ACL_FLOAT16, ACL_FORMAT_NCHW);
  auto self_desc = TensorDesc({7, 5, 5}, ACL_FLOAT16, ACL_FORMAT_NCHW);
  auto indices_desc = TensorDesc({7, 5, 5}, ACL_INT64, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto output_szie = IntArrayDesc(vector<int64_t>{2, 3});
  auto out_desc = TensorDesc(self_desc).Precision(0.0001, 0.0001);

  auto ut =
      OP_API_UT(aclnnMaxUnpool2dBackward, INPUT(grad_desc, self_desc, indices_desc, output_szie), OUTPUT(out_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

TEST_F(l2_max_unpool2d_backward_test, case_003) {
  auto grad_desc = TensorDesc({7, 2, 3}, ACL_INT8, ACL_FORMAT_NCHW);
  auto self_desc = TensorDesc({7, 5, 5}, ACL_INT8, ACL_FORMAT_NCHW);
  auto indices_desc = TensorDesc({7, 5, 5}, ACL_INT64, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto output_szie = IntArrayDesc(vector<int64_t>{2, 3});
  auto out_desc = TensorDesc(self_desc);

  auto ut =
      OP_API_UT(aclnnMaxUnpool2dBackward, INPUT(grad_desc, self_desc, indices_desc, output_szie), OUTPUT(out_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

TEST_F(l2_max_unpool2d_backward_test, case_004) {
  auto grad_desc = TensorDesc({3, 7, 2, 3}, ACL_INT32, ACL_FORMAT_NCHW);
  auto self_desc = TensorDesc({3, 7, 5, 5}, ACL_INT32, ACL_FORMAT_NCHW);
  auto indices_desc = TensorDesc({3, 7, 5, 5}, ACL_INT64, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto output_szie = IntArrayDesc(vector<int64_t>{2, 3});
  auto out_desc = TensorDesc(self_desc);

  auto ut =
      OP_API_UT(aclnnMaxUnpool2dBackward, INPUT(grad_desc, self_desc, indices_desc, output_szie), OUTPUT(out_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

// check shape
TEST_F(l2_max_unpool2d_backward_test, case_005) {
  auto grad_desc = TensorDesc({7, 2, 3}, ACL_INT16, ACL_FORMAT_NCHW);
  auto self_desc = TensorDesc({7, 6, 5}, ACL_INT16, ACL_FORMAT_NCHW);
  auto indices_desc = TensorDesc({7, 5, 5}, ACL_INT64, ACL_FORMAT_NCHW);
  auto output_szie = IntArrayDesc(vector<int64_t>{2, 3});
  auto out_desc = TensorDesc(self_desc);

  auto ut =
      OP_API_UT(aclnnMaxUnpool2dBackward, INPUT(grad_desc, self_desc, indices_desc, output_szie), OUTPUT(out_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// check shape
TEST_F(l2_max_unpool2d_backward_test, case_006) {
  auto grad_desc = TensorDesc({1, 6, 2, 3}, ACL_UINT8, ACL_FORMAT_NCHW);
  auto self_desc = TensorDesc({2, 7, 5, 5}, ACL_UINT8, ACL_FORMAT_NCHW);
  auto indices_desc = TensorDesc({2, 7, 5, 5}, ACL_INT32, ACL_FORMAT_NCHW);
  auto output_szie = IntArrayDesc(vector<int64_t>{2, 3});
  auto out_desc = TensorDesc(self_desc);

  auto ut =
      OP_API_UT(aclnnMaxUnpool2dBackward, INPUT(grad_desc, self_desc, indices_desc, output_szie), OUTPUT(out_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// check shape
TEST_F(l2_max_unpool2d_backward_test, case_007) {
  auto grad_desc = TensorDesc({7, 2, 3}, ACL_INT8, ACL_FORMAT_NCHW);
  auto self_desc = TensorDesc({7, 5, 5}, ACL_INT8, ACL_FORMAT_NCHW);
  auto indices_desc = TensorDesc({7, 5, 5}, ACL_INT64, ACL_FORMAT_NCHW);
  auto output_szie = IntArrayDesc(vector<int64_t>{2, 3, 4});
  auto out_desc = TensorDesc(self_desc);

  auto ut =
      OP_API_UT(aclnnMaxUnpool2dBackward, INPUT(grad_desc, self_desc, indices_desc, output_szie), OUTPUT(out_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// check dtype
TEST_F(l2_max_unpool2d_backward_test, case_008) {
  auto grad_desc = TensorDesc({7, 2, 3}, ACL_DOUBLE, ACL_FORMAT_NCHW);
  auto self_desc = TensorDesc({7, 5, 5}, ACL_DOUBLE, ACL_FORMAT_NCHW);
  auto indices_desc = TensorDesc({7, 5, 5}, ACL_INT64, ACL_FORMAT_NCHW);
  auto output_szie = IntArrayDesc(vector<int64_t>{2, 3, 4});
  auto out_desc = TensorDesc(self_desc);

  auto ut =
      OP_API_UT(aclnnMaxUnpool2dBackward, INPUT(grad_desc, self_desc, indices_desc, output_szie), OUTPUT(out_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// cast
TEST_F(l2_max_unpool2d_backward_test, case_009) {
  auto grad_desc = TensorDesc({7, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
  auto self_desc = TensorDesc({7, 5, 5}, ACL_FLOAT, ACL_FORMAT_NCHW);
  auto indices_desc = TensorDesc({7, 5, 5}, ACL_INT64, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto output_szie = IntArrayDesc(vector<int64_t>{2, 3});
  auto out_desc = TensorDesc({7, 5, 5}, ACL_FLOAT16, ACL_FORMAT_NCHW).Precision(0.001, 0.001);

  auto ut =
      OP_API_UT(aclnnMaxUnpool2dBackward, INPUT(grad_desc, self_desc, indices_desc, output_szie), OUTPUT(out_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

// nullptr
TEST_F(l2_max_unpool2d_backward_test, case_010) {
  auto grad_desc = TensorDesc({7, 2, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
  auto self_desc = TensorDesc({7, 5, 5}, ACL_FLOAT, ACL_FORMAT_NCHW);
  auto indices_desc = TensorDesc({7, 5, 5}, ACL_INT64, ACL_FORMAT_NCHW);
  auto output_szie = IntArrayDesc(vector<int64_t>{2, 3});
  auto out_desc = TensorDesc({7, 5, 5}, ACL_FLOAT16, ACL_FORMAT_NCHW).Precision(0.001, 0.001);

  auto ut = OP_API_UT(aclnnMaxUnpool2dBackward, INPUT((aclTensor*)nullptr, self_desc, indices_desc, output_szie),
                      OUTPUT(out_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

  auto ut2 = OP_API_UT(aclnnMaxUnpool2dBackward, INPUT(grad_desc, (aclTensor*)nullptr, indices_desc, output_szie),
                       OUTPUT(out_desc));

  // SAMPLE: only test GetWorkspaceSize
  workspace_size = 0;
  aclRet = ut2.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

  auto ut3 = OP_API_UT(aclnnMaxUnpool2dBackward, INPUT(grad_desc, self_desc, (aclTensor*)nullptr, output_szie),
                       OUTPUT(out_desc));

  // SAMPLE: only test GetWorkspaceSize
  workspace_size = 0;
  aclRet = ut3.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

  auto ut4 = OP_API_UT(aclnnMaxUnpool2dBackward, INPUT(grad_desc, self_desc, indices_desc, (aclIntArray*)nullptr),
                       OUTPUT(out_desc));

  // SAMPLE: only test GetWorkspaceSize
  workspace_size = 0;
  aclRet = ut4.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

  auto ut5 = OP_API_UT(aclnnMaxUnpool2dBackward, INPUT(grad_desc, self_desc, indices_desc, output_szie),
                       OUTPUT((aclTensor*)nullptr));

  // SAMPLE: only test GetWorkspaceSize
  workspace_size = 0;
  aclRet = ut5.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// bool
TEST_F(l2_max_unpool2d_backward_test, case_011) {
  auto grad_desc = TensorDesc({7, 2, 3}, ACL_BOOL, ACL_FORMAT_NCHW);
  auto self_desc = TensorDesc({7, 4, 5}, ACL_BOOL, ACL_FORMAT_NCHW, {25, 1, 5}, 0, {7, 5, 4});
  auto indices_desc = TensorDesc({7, 4, 5}, ACL_INT64, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto output_szie = IntArrayDesc(vector<int64_t>{2, 3});
  auto out_desc = TensorDesc({7, 5, 5}, ACL_BOOL, ACL_FORMAT_NCHW).Precision(0.001, 0.001);

  auto ut =
      OP_API_UT(aclnnMaxUnpool2dBackward, INPUT(grad_desc, self_desc, indices_desc, output_szie), OUTPUT(out_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

}

TEST_F(l2_max_unpool2d_backward_test, case_012) {
  auto grad_desc = TensorDesc({7, 0, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
  auto self_desc = TensorDesc({7, 5, 5}, ACL_FLOAT, ACL_FORMAT_NCHW);
  auto indices_desc = TensorDesc({7, 5, 5}, ACL_INT64, ACL_FORMAT_NCHW).ValueRange(0, 2);
  auto output_szie = IntArrayDesc(vector<int64_t>{2, 3});
  auto out_desc = TensorDesc({7, 5, 5}, ACL_FLOAT16, ACL_FORMAT_NCHW).Precision(0.001, 0.001);

  auto ut =
      OP_API_UT(aclnnMaxUnpool2dBackward, INPUT(grad_desc, self_desc, indices_desc, output_szie), OUTPUT(out_desc));

  // SAMPLE: only test GetWorkspaceSize
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

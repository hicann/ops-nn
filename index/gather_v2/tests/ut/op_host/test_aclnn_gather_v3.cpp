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

#include "../../../op_host/op_api/aclnn_gather_v3.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class l2GatherV3Test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    cout << "gather_v3_test SetUp" << endl;
  }

  static void TearDownTestCase() { cout << "gather_v3_test TearDown" << endl; }
};


TEST_F(l2GatherV3Test, case_1) {
  auto tensorWeightDesc = TensorDesc({10, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto tensorIndicesDesc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND)
      .ValueRange(0, 5)
      .Value(vector<int>{1,2,3,4});
  auto tensorOutDesc = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND);
  int64_t dim = 0;
  int64_t batchDims = 0;
  int64_t mode = 1;

  auto ut = OP_API_UT(aclnnGatherV3,  // host api第二段接口名称
                      INPUT(tensorWeightDesc, dim, tensorIndicesDesc, batchDims, mode),   // host api输入
                      OUTPUT(tensorOutDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}

// indice nullptr
TEST_F(l2GatherV3Test, case_2) {
  auto tensorWeightDesc = TensorDesc({10, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto tensorIndicesDesc = nullptr;
  auto tensorOutDesc = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND);

  int64_t dim = 0;
  int64_t batchDims = 0;
  int64_t mode = 1;

  auto ut = OP_API_UT(aclnnGatherV3,  // host api第二段接口名称
                      INPUT(tensorWeightDesc, dim, tensorIndicesDesc, batchDims, mode),   // host api输入
                      OUTPUT(tensorOutDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// weight nullptr
TEST_F(l2GatherV3Test, case_3) {
  auto tensorWeightDesc = nullptr;
  auto tensorIndicesDesc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND)
      .ValueRange(0, 5)
      .Value(vector<int>{1,2,3,4});
  auto tensorOutDesc = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND);

  int64_t dim = 0;
  int64_t batchDims = 0;
  int64_t mode = 1;

  auto ut = OP_API_UT(aclnnGatherV3,  // host api第二段接口名称
                      INPUT(tensorWeightDesc, dim, tensorIndicesDesc, batchDims, mode),   // host api输入
                      OUTPUT(tensorOutDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// out nullptr
TEST_F(l2GatherV3Test, case_4) {
  auto tensorWeightDesc = TensorDesc({10, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto tensorIndicesDesc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND)
      .ValueRange(0, 5)
      .Value(vector<int>{1,2,3,4});
  auto tensorOutDesc = nullptr;

  int64_t dim = 0;
  int64_t batchDims = 0;
  int64_t mode = 1;

  auto ut = OP_API_UT(aclnnGatherV3,  // host api第二段接口名称
                      INPUT(tensorWeightDesc, dim, tensorIndicesDesc, batchDims, mode),   // host api输入
                      OUTPUT(tensorOutDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// indices support int64, weight support float16
TEST_F(l2GatherV3Test, case_5) {
  auto tensorWeightDesc = TensorDesc({10, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto tensorIndicesDesc = TensorDesc({4}, ACL_INT64, ACL_FORMAT_ND)
      .ValueRange(0, 5)
      .Value(vector<int>{1,2,3,4});
  auto tensorOutDesc = TensorDesc({4, 3}, ACL_FLOAT16, ACL_FORMAT_ND);

  int64_t dim = 0;
  int64_t batchDims = 0;
  int64_t mode = 1;

  auto ut = OP_API_UT(aclnnGatherV3,  // host api第二段接口名称
                      INPUT(tensorWeightDesc, dim, tensorIndicesDesc, batchDims, mode),   // host api输入
                      OUTPUT(tensorOutDesc));


  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  // SAMPLE: precision simulate
  ut.TestPrecision();
}


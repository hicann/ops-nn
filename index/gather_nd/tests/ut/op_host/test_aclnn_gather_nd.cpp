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

#include "../../../op_host/op_api/aclnn_gather_nd.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class l2GatherNdTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    cout << "gather_nd_test SetUp" << endl;
  }

  static void TearDownTestCase() { cout << "gather_nd_test TearDown" << endl; }
};

TEST_F(l2GatherNdTest, l2_gather_nd_case_nullptr_01) {
  auto outputDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  bool negativeIndexSupport = false;

  auto ut = OP_API_UT(aclnnGatherNd, INPUT((aclTensor*)nullptr, (aclTensor*)nullptr, negativeIndexSupport), OUTPUT(outputDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2GatherNdTest, l2_gather_nd_case_nullptr_02) {
  auto indexDesc = TensorDesc({2, 2}, ACL_INT32, ACL_FORMAT_ND);
  bool negativeIndexSupport = false;
  auto outputDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnGatherNd, INPUT((aclTensor*)nullptr, indexDesc, negativeIndexSupport), OUTPUT(outputDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2GatherNdTest, l2_gather_nd_case_nullptr_03) {
  auto selfDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  bool negativeIndexSupport = false;
  auto outputDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnGatherNd, INPUT(selfDesc, (aclTensor*)nullptr, negativeIndexSupport), OUTPUT(outputDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2GatherNdTest, l2_gather_nd_case_dtype_inconsistency_01) {
  auto selfDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
  auto indexDesc = TensorDesc({2, 2}, ACL_INT32, ACL_FORMAT_ND);
  bool negativeIndexSupport = false;
  auto outputDesc = TensorDesc({2}, ACL_FLOAT16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnGatherNd, INPUT(selfDesc, indexDesc, negativeIndexSupport), OUTPUT(outputDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2GatherNdTest, l2_gather_nd_case_dtype_inconsistency_02) {
  auto selfDesc = TensorDesc({2, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto indexDesc = TensorDesc({2, 2}, ACL_INT32, ACL_FORMAT_ND);
  bool negativeIndexSupport = false;
  auto outputDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnGatherNd, INPUT(selfDesc, indexDesc, negativeIndexSupport), OUTPUT(outputDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2GatherNdTest, l2_gather_nd_case_dtype_not_support_01) {
  auto selfDesc = TensorDesc({2, 2}, ACL_UINT32, ACL_FORMAT_ND);
  auto indexDesc = TensorDesc({2, 2}, ACL_INT32, ACL_FORMAT_ND);
  bool negativeIndexSupport = false;
  auto outputDesc = TensorDesc({2}, ACL_UINT32, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnGatherNd, INPUT(selfDesc, indexDesc, negativeIndexSupport), OUTPUT(outputDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2GatherNdTest, l2_gather_nd_case_shape_not_support_01) {
  auto selfDesc = TensorDesc({2, 2}, ACL_INT64, ACL_FORMAT_ND);
  auto indexDesc = TensorDesc({2, 2}, ACL_INT32, ACL_FORMAT_ND);
  bool negativeIndexSupport = false;
  auto outputDesc = TensorDesc({2, 2}, ACL_INT64, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnGatherNd, INPUT(selfDesc, indexDesc, negativeIndexSupport), OUTPUT(outputDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2GatherNdTest, l2_gather_nd_case_shape_not_support_02) {
  auto selfDesc = TensorDesc({2, 2}, ACL_INT64, ACL_FORMAT_ND);
  auto indexDesc = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND);
  bool negativeIndexSupport = false;
  auto outputDesc = TensorDesc({2}, ACL_INT64, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnGatherNd, INPUT(selfDesc, indexDesc, negativeIndexSupport), OUTPUT(outputDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2GatherNdTest, l2_gather_nd_case_dimension_not_support_01) {
  auto selfDesc = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2}, ACL_INT64, ACL_FORMAT_ND);
  auto indexDesc = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_INT32, ACL_FORMAT_ND);
  bool negativeIndexSupport = false;
  auto outputDesc = TensorDesc({2}, ACL_INT64, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnGatherNd, INPUT(selfDesc, indexDesc, negativeIndexSupport), OUTPUT(outputDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2GatherNdTest, l2_gather_nd_case_dimension_not_support_02) {
  auto selfDesc = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_INT64, ACL_FORMAT_ND);
  auto indexDesc = TensorDesc({2, 2}, ACL_INT32, ACL_FORMAT_ND);
  bool negativeIndexSupport = false;
  auto outputDesc = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2}, ACL_INT64, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnGatherNd, INPUT(selfDesc, indexDesc, negativeIndexSupport), OUTPUT(outputDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2GatherNdTest, l2_gather_nd_case_negativeIndexSupport_true_001) {
  auto selfDesc = TensorDesc({2, 2}, ACL_INT64, ACL_FORMAT_ND);
  auto indexDesc = TensorDesc({2, 2}, ACL_INT32, ACL_FORMAT_ND);
  bool negativeIndexSupport = true;
  auto outputDesc = TensorDesc({2}, ACL_INT64, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnGatherNd, INPUT(selfDesc, indexDesc, negativeIndexSupport), OUTPUT(outputDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2GatherNdTest, l2_gather_nd_case_negativeIndexSupport_true_002) {
  auto selfDesc = TensorDesc({2, 2}, ACL_INT64, ACL_FORMAT_ND);
  auto indexDesc = TensorDesc({2, 1}, ACL_INT32, ACL_FORMAT_ND);
  bool negativeIndexSupport = true;
  auto outputDesc = TensorDesc({2, 2}, ACL_INT64, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnGatherNd, INPUT(selfDesc, indexDesc, negativeIndexSupport), OUTPUT(outputDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2GatherNdTest, l2_gather_nd_case_int64_001) {
  auto selfDesc = TensorDesc({2, 2}, ACL_INT64, ACL_FORMAT_ND).Value(vector<int64_t>{0, 1, 2, 3});
  auto indexDesc = TensorDesc({2, 2}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int64_t>{0, 0, 1, 1});
  bool negativeIndexSupport = false;
  auto outputDesc = TensorDesc({2}, ACL_INT64, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnGatherNd, INPUT(selfDesc, indexDesc, negativeIndexSupport), OUTPUT(outputDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

TEST_F(l2GatherNdTest, l2_gather_nd_case_int64_002) {
  auto selfDesc = TensorDesc({2, 2}, ACL_INT64, ACL_FORMAT_ND).Value(vector<int64_t>{0, 1, 2, 3});
  auto indexDesc = TensorDesc({2, 1}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{1, 0});
  bool negativeIndexSupport = false;
  auto outputDesc = TensorDesc({2, 2}, ACL_INT64, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnGatherNd, INPUT(selfDesc, indexDesc, negativeIndexSupport), OUTPUT(outputDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

TEST_F(l2GatherNdTest, l2_gather_nd_case_int32_001) {
  auto selfDesc = TensorDesc({2, 2}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0, 1, 2, 3});
  auto indexDesc = TensorDesc({2, 2}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0, 0, 1, 1});
  bool negativeIndexSupport = false;
  auto outputDesc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnGatherNd, INPUT(selfDesc, indexDesc, negativeIndexSupport), OUTPUT(outputDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

TEST_F(l2GatherNdTest, l2_gather_nd_case_int32_002) {
  auto selfDesc = TensorDesc({2, 2}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0, 1, 2, 3});
  auto indexDesc = TensorDesc({2, 1}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0, 1});
  bool negativeIndexSupport = false;
  auto outputDesc = TensorDesc({2, 2}, ACL_INT32, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnGatherNd, INPUT(selfDesc, indexDesc, negativeIndexSupport), OUTPUT(outputDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

TEST_F(l2GatherNdTest, l2_gather_nd_case_int8_001) {
  auto selfDesc = TensorDesc({2, 2}, ACL_INT8, ACL_FORMAT_ND).Value(vector<int8_t>{0, 1, 2, 3});
  auto indexDesc = TensorDesc({2, 2}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0, 0, 1, 1});
  bool negativeIndexSupport = false;
  auto outputDesc = TensorDesc({2}, ACL_INT8, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnGatherNd, INPUT(selfDesc, indexDesc, negativeIndexSupport), OUTPUT(outputDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

TEST_F(l2GatherNdTest, l2_gather_nd_case_int8_002) {
  auto selfDesc = TensorDesc({2, 2}, ACL_INT8, ACL_FORMAT_ND).Value(vector<int8_t>{0, 1, 2, 3});
  auto indexDesc = TensorDesc({2, 1}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{1, 0});
  bool negativeIndexSupport = false;
  auto outputDesc = TensorDesc({2, 2}, ACL_INT8, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnGatherNd, INPUT(selfDesc, indexDesc, negativeIndexSupport), OUTPUT(outputDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

TEST_F(l2GatherNdTest, l2_gather_nd_case_uint8_001) {
  auto selfDesc = TensorDesc({2, 2}, ACL_UINT8, ACL_FORMAT_ND).Value(vector<uint8_t>{0, 1, 2, 3});
  auto indexDesc = TensorDesc({2, 2}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0, 0, 1, 1});
  bool negativeIndexSupport = false;
  auto outputDesc = TensorDesc({2}, ACL_UINT8, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnGatherNd, INPUT(selfDesc, indexDesc, negativeIndexSupport), OUTPUT(outputDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

TEST_F(l2GatherNdTest, l2_gather_nd_case_uint8_002) {
  auto selfDesc = TensorDesc({2, 2}, ACL_UINT8, ACL_FORMAT_ND).Value(vector<uint8_t>{0, 1, 2, 3});
  auto indexDesc = TensorDesc({2, 1}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{1, 0});
  bool negativeIndexSupport = false;
  auto outputDesc = TensorDesc({2, 2}, ACL_UINT8, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnGatherNd, INPUT(selfDesc, indexDesc, negativeIndexSupport), OUTPUT(outputDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

TEST_F(l2GatherNdTest, l2_gather_nd_case_bool_001) {
  auto selfDesc = TensorDesc({2, 2}, ACL_BOOL, ACL_FORMAT_ND).Value(vector<bool>{true, true, false, false});
  auto indexDesc = TensorDesc({2, 2}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0, 0, 1, 1});
  bool negativeIndexSupport = false;
  auto outputDesc = TensorDesc({2}, ACL_BOOL, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnGatherNd, INPUT(selfDesc, indexDesc, negativeIndexSupport), OUTPUT(outputDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

TEST_F(l2GatherNdTest, l2_gather_nd_case_bool_002) {
  auto selfDesc = TensorDesc({2, 2}, ACL_BOOL, ACL_FORMAT_ND).Value(vector<bool>{true, true, false, false});
  auto indexDesc = TensorDesc({2, 1}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0, 1});
  bool negativeIndexSupport = false;
  auto outputDesc = TensorDesc({2, 2}, ACL_BOOL, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnGatherNd, INPUT(selfDesc, indexDesc, negativeIndexSupport), OUTPUT(outputDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

TEST_F(l2GatherNdTest, l2_gather_nd_case_float_001) {
  auto selfDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1.1, 1.2, 1.3, 1.4});
  auto indexDesc = TensorDesc({2, 2}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0, 0, 1, 1});
  bool negativeIndexSupport = false;
  auto outputDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnGatherNd, INPUT(selfDesc, indexDesc, negativeIndexSupport), OUTPUT(outputDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

TEST_F(l2GatherNdTest, l2_gather_nd_case_float_002) {
  auto selfDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1.1, 1.2, 1.3, 1.4});
  auto indexDesc = TensorDesc({2, 1}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0, 1});
  bool negativeIndexSupport = false;
  auto outputDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnGatherNd, INPUT(selfDesc, indexDesc, negativeIndexSupport), OUTPUT(outputDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

TEST_F(l2GatherNdTest, l2_gather_nd_case_float16_001) {
  auto selfDesc = TensorDesc({2, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto indexDesc = TensorDesc({2, 2}, ACL_INT32, ACL_FORMAT_ND);
  bool negativeIndexSupport = false;
  auto outputDesc = TensorDesc({2}, ACL_FLOAT16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnGatherNd, INPUT(selfDesc, indexDesc, negativeIndexSupport), OUTPUT(outputDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2GatherNdTest, l2_gather_nd_case_float16_002) {
  auto selfDesc = TensorDesc({2, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
  auto indexDesc = TensorDesc({2, 1}, ACL_INT32, ACL_FORMAT_ND);
  bool negativeIndexSupport = false;
  auto outputDesc = TensorDesc({2, 2}, ACL_FLOAT16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnGatherNd, INPUT(selfDesc, indexDesc, negativeIndexSupport), OUTPUT(outputDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2GatherNdTest, ascend910B2_l2_gather_nd_case_bfloat16_001) {
  auto selfDesc = TensorDesc({2, 2}, ACL_BF16, ACL_FORMAT_ND);
  auto indexDesc = TensorDesc({2, 2}, ACL_INT32, ACL_FORMAT_ND);
  bool negativeIndexSupport = false;
  auto outputDesc = TensorDesc({2}, ACL_BF16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnGatherNd, INPUT(selfDesc, indexDesc, negativeIndexSupport), OUTPUT(outputDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2GatherNdTest, ascend910B2_l2_gather_nd_case_bfloat16_002) {
  auto selfDesc = TensorDesc({2, 2}, ACL_BF16, ACL_FORMAT_ND);
  auto indexDesc = TensorDesc({2, 1}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0, 0, 1, 1});
  bool negativeIndexSupport = false;
  auto outputDesc = TensorDesc({2, 2}, ACL_BF16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnGatherNd, INPUT(selfDesc, indexDesc, negativeIndexSupport), OUTPUT(outputDesc));
  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

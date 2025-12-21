/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \\file test_index_copy.cpp
 * \\brief
 */
#include <array>
#include <vector>
#include "gtest/gtest.h"

#include "../../../op_host/op_api/aclnn_index_copy.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include <cstdlib>
#include <ctime>

using namespace op;
using namespace std;

class l2_index_copy_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "index_copy_test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "index_copy_test TearDown" << std::endl;
  }
};

// 基础用例
TEST_F(l2_index_copy_test, Ascend910_9598_aclnnIndexCopy_base_case_1) {
  // self input
  const vector<int64_t>& selfShape = {5, 3};
  aclDataType selfDtype = ACL_INT32;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // index input
  const vector<int64_t>& indexShape = {3};
  aclDataType indexDtype = ACL_INT64;
  aclFormat indexFormat = ACL_FORMAT_ND;
  // source input
  const vector<int64_t>& sourceShape = {3, 3};
  aclDataType sourceDtype = selfDtype;
  aclFormat sourceFormat = ACL_FORMAT_ND;

  auto selfTensorDesc =
      TensorDesc(selfShape, selfDtype, selfFormat).Value(vector<int32_t>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
  auto indexTensorDesc = TensorDesc(indexShape, indexDtype, indexFormat).Value(vector<int32_t>{0, 4, 2});
  auto sourceTensorDesc =
      TensorDesc(sourceShape, sourceDtype, sourceFormat).Value(vector<int32_t>{1, 2, 3, 4, 5, 6, 7, 8, 9});
  auto outTensorDesc =
      TensorDesc(selfShape, selfDtype, selfFormat).Value(vector<int32_t>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});

  auto ut = OP_API_UT(aclnnIndexCopy, INPUT(selfTensorDesc, 0, indexTensorDesc, sourceTensorDesc), OUTPUT(outTensorDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

TEST_F(l2_index_copy_test, aclnnIndexCopy_base_case_2) {
  // self input
  const vector<int64_t>& selfShape = {3, 5};
  aclDataType selfDtype = ACL_FLOAT;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // index input
  const vector<int64_t>& indexShape = {3};
  aclDataType indexDtype = ACL_INT64;
  aclFormat indexFormat = ACL_FORMAT_ND;
  // source input
  const vector<int64_t>& sourceShape = {3, 3};
  aclDataType sourceDtype = ACL_FLOAT;
  aclFormat sourceFormat = ACL_FORMAT_ND;

  auto selfTensorDesc =
      TensorDesc(selfShape, selfDtype, selfFormat)
          .Value(vector<float>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
  auto indexTensorDesc = TensorDesc(indexShape, indexDtype, indexFormat).Value(vector<int32_t>{0, 4, 2});
  auto sourceTensorDesc = TensorDesc(sourceShape, sourceDtype, sourceFormat)
                              .Value(vector<float>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0});
  auto outTensorDesc =
      TensorDesc(selfShape, selfDtype, selfFormat)
          .Value(vector<float>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});

  auto ut = OP_API_UT(aclnnIndexCopy, INPUT(selfTensorDesc, 1, indexTensorDesc, sourceTensorDesc), OUTPUT(outTensorDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}

// self类型和out不一致
TEST_F(l2_index_copy_test, aclnnIndexCopy_self_and_out_dtype_non_consistent) {
  // self input
  const vector<int64_t>& selfShape = {2, 4};
  aclDataType selfDtype = ACL_FLOAT;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // index input
  const vector<int64_t>& indexShape = {4};
  aclDataType indexDtype = ACL_INT64;
  aclFormat indexFormat = ACL_FORMAT_ND;
  // source input
  const vector<int64_t>& sourceShape = selfShape;
  aclDataType sourceDtype = ACL_FLOAT16;
  aclFormat sourceFormat = ACL_FORMAT_ND;

  aclDataType outDtype = ACL_FLOAT16;

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto indexTensorDesc = TensorDesc(indexShape, indexDtype, indexFormat).Value(vector<int32_t>{3, 2, 1, 0});
  auto sourceTensorDesc = TensorDesc(sourceShape, sourceDtype, sourceFormat);
  auto outTensorDesc = TensorDesc(selfShape, outDtype, selfFormat);

  auto ut = OP_API_UT(aclnnIndexCopy, INPUT(selfTensorDesc, 1, indexTensorDesc, sourceTensorDesc), OUTPUT(outTensorDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// self形状和out不一致
TEST_F(l2_index_copy_test, aclnnIndexCopy_self_and_out_shape_non_consistent) {
  // self input
  const vector<int64_t>& selfShape = {2, 4};
  aclDataType selfDtype = ACL_FLOAT;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // index input
  const vector<int64_t>& indexShape = {4};
  aclDataType indexDtype = ACL_INT64;
  aclFormat indexFormat = ACL_FORMAT_ND;
  // source input
  const vector<int64_t>& sourceShape = selfShape;
  aclDataType sourceDtype = ACL_FLOAT16;
  aclFormat sourceFormat = ACL_FORMAT_ND;

  const vector<int64_t>& outShape = {4, 4};

  auto selfTensorDesc = TensorDesc(selfShape, selfDtype, selfFormat);
  auto indexTensorDesc = TensorDesc(indexShape, indexDtype, indexFormat).Value(vector<int32_t>{3, 2, 1, 0});
  auto sourceTensorDesc = TensorDesc(sourceShape, sourceDtype, sourceFormat);
  auto outTensorDesc = TensorDesc(outShape, selfDtype, selfFormat);

  auto ut = OP_API_UT(aclnnIndexCopy, INPUT(selfTensorDesc, 1, indexTensorDesc, sourceTensorDesc), OUTPUT(outTensorDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_index_copy_test, Ascend910_9598_aclnnIndexCopy_base_case_2) {
  // self input
  const vector<int64_t>& selfShape = {3, 5};
  aclDataType selfDtype = ACL_FLOAT;
  aclFormat selfFormat = ACL_FORMAT_ND;
  // index input
  const vector<int64_t>& indexShape = {3};
  aclDataType indexDtype = ACL_INT64;
  aclFormat indexFormat = ACL_FORMAT_ND;
  // source input
  const vector<int64_t>& sourceShape = {3, 3};
  aclDataType sourceDtype = ACL_FLOAT;
  aclFormat sourceFormat = ACL_FORMAT_ND;

  auto selfTensorDesc =
      TensorDesc(selfShape, selfDtype, selfFormat)
          .Value(vector<float>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
  auto indexTensorDesc = TensorDesc(indexShape, indexDtype, indexFormat).Value(vector<int32_t>{0, 4, 2});
  auto sourceTensorDesc = TensorDesc(sourceShape, sourceDtype, sourceFormat)
                              .Value(vector<float>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0});
  auto outTensorDesc =
      TensorDesc(selfShape, selfDtype, selfFormat)
          .Value(vector<float>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});

  auto ut = OP_API_UT(aclnnIndexCopy, INPUT(selfTensorDesc, 1, indexTensorDesc, sourceTensorDesc), OUTPUT(outTensorDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(aclRet, ACL_SUCCESS);

  ut.TestPrecision();
}
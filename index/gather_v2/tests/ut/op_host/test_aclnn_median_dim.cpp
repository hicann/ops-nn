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

#include "../../../op_host/op_api/aclnn_median.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/inner/types.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class l2_median_dim_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "l2_median_dim_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "l2_median_dim_test TearDown" << endl;
    }
};

TEST_F(l2_median_dim_test, aclnnMedianDim_3_9_2_complex64_nd_1_nd) {
  const int64_t dim = 1;
  const bool keepdim = false;

  auto self_tensor_desc = TensorDesc({3,9,2}, ACL_COMPLEX64, ACL_FORMAT_ND).ValueRange(0, 2);
  auto indices_tensor_desc = TensorDesc({3,2}, ACL_INT64, ACL_FORMAT_ND).Precision(0.001, 0.001);
  auto out_tensor_desc = TensorDesc({3,2}, ACL_COMPLEX64, ACL_FORMAT_ND).Precision(0.001, 0.001);

  auto ut = OP_API_UT(aclnnMedianDim, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc, indices_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_median_dim_test, aclnnMedianDim_3_9_2_complex128_nd_1_nd) {
  const int64_t dim = 1;
  const bool keepdim = false;

  auto self_tensor_desc = TensorDesc({3,9,2}, ACL_COMPLEX128, ACL_FORMAT_ND).ValueRange(0, 2);
  auto indices_tensor_desc = TensorDesc({3,2}, ACL_INT64, ACL_FORMAT_ND).Precision(0.001, 0.001);
  auto out_tensor_desc = TensorDesc({3,2}, ACL_COMPLEX128, ACL_FORMAT_ND).Precision(0.001, 0.001);

  auto ut = OP_API_UT(aclnnMedianDim, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc, indices_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_median_dim_test, aclnnMedianDim_3_9_2_string_nd_1_nd) {
  const int64_t dim = 1;
  const bool keepdim = false;

  auto self_tensor_desc = TensorDesc({3,9,2}, ACL_STRING, ACL_FORMAT_ND).ValueRange(0, 2);
  auto indices_tensor_desc = TensorDesc({3,2}, ACL_INT64, ACL_FORMAT_ND).Precision(0.001, 0.001);
  auto out_tensor_desc = TensorDesc({3,2}, ACL_STRING, ACL_FORMAT_ND).Precision(0.001, 0.001);

  auto ut = OP_API_UT(aclnnMedianDim, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc, indices_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_median_dim_test, aclnnMedianDim_0_4_float_nd_0_nd) {
  const int64_t dim = 0;
  const bool keepdim = false;

  auto self_tensor_desc = TensorDesc({0,4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
  auto indices_tensor_desc = TensorDesc({4}, ACL_INT64, ACL_FORMAT_ND).Precision(0.001, 0.001);
  auto out_tensor_desc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);

  auto ut = OP_API_UT(aclnnMedianDim, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc, indices_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_median_dim_test, aclnnMedianDim_9D_float_nd_0_nd) {
  const int64_t dim = 0;
  const bool keepdim = false;

  auto self_tensor_desc = TensorDesc({7, 9, 11, 3, 4, 6, 9, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
  auto indices_tensor_desc = TensorDesc({9, 11, 3, 4, 6, 9, 2, 2}, ACL_INT64, ACL_FORMAT_ND).Precision(0.001, 0.001);
  auto out_tensor_desc = TensorDesc({9, 11, 3, 4, 6, 9, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);

  auto ut = OP_API_UT(aclnnMedianDim, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc, indices_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_median_dim_test, aclnnMedianDim_3_4_float_nd_0_nd_error_shape) {
  const int64_t dim = 0;
  const bool keepdim = false;

  auto self_tensor_desc = TensorDesc({3,4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
  auto indices_tensor_desc = TensorDesc({4}, ACL_INT64, ACL_FORMAT_ND).Precision(0.001, 0.001);
  auto out_tensor_desc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);

  auto ut = OP_API_UT(aclnnMedianDim, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc, indices_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  // EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_median_dim_test, aclnnMedianDim_3_4_float_nd_0_nd_self_is_null) {
  const int64_t dim = 0;
  const bool keepdim = false;

  auto self_tensor_desc = nullptr;
  auto indices_tensor_desc = TensorDesc({4}, ACL_INT64, ACL_FORMAT_ND).Precision(0.001, 0.001);
  auto out_tensor_desc = TensorDesc({4,3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);

  auto ut = OP_API_UT(aclnnMedianDim, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc, indices_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_median_dim_test, aclnnMedianDim_3_4_float_nd_0_nd_indices_is_null) {
  const int64_t dim = 0;
  const bool keepdim = false;

  auto self_tensor_desc = TensorDesc({3,4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
  auto indices_tensor_desc = nullptr;
  auto out_tensor_desc = TensorDesc({4,3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);

  auto ut = OP_API_UT(aclnnMedianDim, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc, indices_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_median_dim_test, aclnnMedianDim_3_4_float_nd_0_nd_out_is_null) {
  const int64_t dim = 0;
  const bool keepdim = false;

  auto self_tensor_desc = TensorDesc({3,4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
  auto indices_tensor_desc = TensorDesc({4}, ACL_INT64, ACL_FORMAT_ND).Precision(0.001, 0.001);
  auto out_tensor_desc = nullptr;

  auto ut = OP_API_UT(aclnnMedianDim, INPUT(self_tensor_desc, dim, keepdim), OUTPUT(out_tensor_desc, indices_tensor_desc));

  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}


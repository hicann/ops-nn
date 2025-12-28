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
 * \file test_aclnn_grouped_dynamic_mx_quant.cpp
 * \brief
 */
#include <float.h>

#include <array>
#include <vector>

#include "gtest/gtest.h"
#include "../../../op_host/op_api/aclnn_grouped_dynamic_mx_quant.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class l2_grouped_dynamic_mx_quant_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    cout << "l2_grouped_dynamic_mx_quant_test SetUp" << endl;
  }

  static void TearDownTestCase() {
    cout << "l2_grouped_dynamic_mx_quant_test TearDown" << endl;
  }
};

TEST_F(l2_grouped_dynamic_mx_quant_test, ascend910_95_grouped_dynamic_mx_quant_bf16_E4M3) {
  TensorDesc x_desc = TensorDesc({64, 5}, ACL_BF16, ACL_FORMAT_ND);
  TensorDesc group_index_desc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
  TensorDesc y_desc = TensorDesc({64, 5}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
  TensorDesc mxscale_desc = TensorDesc({3, 5, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
  int64_t dstType = static_cast<int64_t>(ACL_FLOAT8_E4M3FN);
  int64_t blocksize = 32;
  const char* roundMode = "rint";
  class SocVersionManager testSocVersion(SocVersion::ASCEND910_95);

  auto ut = OP_API_UT(aclnnGroupedDynamicMxQuant, INPUT(x_desc, group_index_desc, roundMode, dstType, blocksize),
                      OUTPUT(y_desc, mxscale_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_grouped_dynamic_mx_quant_test, ascend910_95_grouped_dynamic_mx_quant_bf16_E5M2) {
  TensorDesc x_desc = TensorDesc({64, 5}, ACL_BF16, ACL_FORMAT_ND);
  TensorDesc group_index_desc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
  TensorDesc y_desc = TensorDesc({64, 5}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
  TensorDesc mxscale_desc = TensorDesc({3, 5, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
  int64_t dstType = static_cast<int64_t>(ACL_FLOAT8_E5M2);
  int64_t blocksize = 32;
  const char* roundMode = "rint";

  class SocVersionManager testSocVersion(SocVersion::ASCEND910_95);
  auto ut = OP_API_UT(aclnnGroupedDynamicMxQuant, INPUT(x_desc, group_index_desc, roundMode, dstType, blocksize),
                      OUTPUT(y_desc, mxscale_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_grouped_dynamic_mx_quant_test, ascend910_95_grouped_dynamic_mx_quant_fp16_E4M3) {
  TensorDesc x_desc = TensorDesc({64, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc group_index_desc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
  TensorDesc y_desc = TensorDesc({64, 5}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
  TensorDesc mxscale_desc = TensorDesc({3, 5, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
  int64_t dstType = static_cast<int64_t>(ACL_FLOAT8_E4M3FN);
  int64_t blocksize = 32;
  const char* roundMode = "rint";

  class SocVersionManager testSocVersion(SocVersion::ASCEND910_95);
  auto ut = OP_API_UT(aclnnGroupedDynamicMxQuant, INPUT(x_desc, group_index_desc, roundMode, dstType, blocksize),
                      OUTPUT(y_desc, mxscale_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_grouped_dynamic_mx_quant_test, ascend910_95_grouped_dynamic_mx_quant_fp16_E5M2) {
  TensorDesc x_desc = TensorDesc({64, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc group_index_desc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
  TensorDesc y_desc = TensorDesc({64, 5}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
  TensorDesc mxscale_desc = TensorDesc({3, 5, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
  int64_t dstType = static_cast<int64_t>(ACL_FLOAT8_E5M2);
  int64_t blocksize = 32;
  const char* roundMode = "rint";

  class SocVersionManager testSocVersion(SocVersion::ASCEND910_95);
  auto ut = OP_API_UT(aclnnGroupedDynamicMxQuant, INPUT(x_desc, group_index_desc, roundMode, dstType, blocksize),
                      OUTPUT(y_desc, mxscale_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_grouped_dynamic_mx_quant_test, ascend910_95_grouped_dynamic_mx_quant_fp16_E5M2_x_Dim_fail) {
  TensorDesc x_desc = TensorDesc({64, 5, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc group_index_desc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
  TensorDesc y_desc = TensorDesc({64, 5}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
  TensorDesc mxscale_desc = TensorDesc({3, 5, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
  int64_t dstType = static_cast<int64_t>(ACL_FLOAT8_E5M2);
  int64_t blocksize = 32;
  const char* roundMode = "rint";

  class SocVersionManager testSocVersion(SocVersion::ASCEND910_95);
  auto ut = OP_API_UT(aclnnGroupedDynamicMxQuant, INPUT(x_desc, group_index_desc, roundMode, dstType, blocksize),
                      OUTPUT(y_desc, mxscale_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_grouped_dynamic_mx_quant_test, ascend910_95_grouped_dynamic_mx_quant_fp16_E5M2_group_Dim_fail) {
  TensorDesc x_desc = TensorDesc({64, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc group_index_desc = TensorDesc({2, 4}, ACL_INT32, ACL_FORMAT_ND);
  TensorDesc y_desc = TensorDesc({64, 5}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
  TensorDesc mxscale_desc = TensorDesc({3, 5, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
  int64_t dstType = static_cast<int64_t>(ACL_FLOAT8_E5M2);
  int64_t blocksize = 32;
  const char* roundMode = "rint";

  class SocVersionManager testSocVersion(SocVersion::ASCEND910_95);
  auto ut = OP_API_UT(aclnnGroupedDynamicMxQuant, INPUT(x_desc, group_index_desc, roundMode, dstType, blocksize),
                      OUTPUT(y_desc, mxscale_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_grouped_dynamic_mx_quant_test, ascend910_95_grouped_dynamic_mx_quant_fp16_E5M2_y_Dim_fail) {
  TensorDesc x_desc = TensorDesc({64, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc group_index_desc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
  TensorDesc y_desc = TensorDesc({64, 5, 5}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
  TensorDesc mxscale_desc = TensorDesc({3, 5, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
  int64_t dstType = static_cast<int64_t>(ACL_FLOAT8_E5M2);
  int64_t blocksize = 32;
  const char* roundMode = "rint";

  class SocVersionManager testSocVersion(SocVersion::ASCEND910_95);
  auto ut = OP_API_UT(aclnnGroupedDynamicMxQuant, INPUT(x_desc, group_index_desc, roundMode, dstType, blocksize),
                      OUTPUT(y_desc, mxscale_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_grouped_dynamic_mx_quant_test, ascend910_95_grouped_dynamic_mx_quant_fp16_E5M2_mxscale_Dim_fail) {
  TensorDesc x_desc = TensorDesc({64, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc group_index_desc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
  TensorDesc y_desc = TensorDesc({64, 5}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
  TensorDesc mxscale_desc = TensorDesc({3}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
  int64_t dstType = static_cast<int64_t>(ACL_FLOAT8_E5M2);
  int64_t blocksize = 32;
  const char* roundMode = "rint";

  class SocVersionManager testSocVersion(SocVersion::ASCEND910_95);
  auto ut = OP_API_UT(aclnnGroupedDynamicMxQuant, INPUT(x_desc, group_index_desc, roundMode, dstType, blocksize),
                      OUTPUT(y_desc, mxscale_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_grouped_dynamic_mx_quant_test, ascend910_95_grouped_dynamic_mx_quant_fp16_E5M2_y_shape_fail) {
  TensorDesc x_desc = TensorDesc({64, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc group_index_desc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
  TensorDesc y_desc = TensorDesc({64, 10}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
  TensorDesc mxscale_desc = TensorDesc({3, 5, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
  int64_t dstType = static_cast<int64_t>(ACL_FLOAT8_E5M2);
  int64_t blocksize = 32;
  const char* roundMode = "rint";

  class SocVersionManager testSocVersion(SocVersion::ASCEND910_95);
  auto ut = OP_API_UT(aclnnGroupedDynamicMxQuant, INPUT(x_desc, group_index_desc, roundMode, dstType, blocksize),
                      OUTPUT(y_desc, mxscale_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_grouped_dynamic_mx_quant_test, ascend910_95_grouped_dynamic_mx_quant_fp16_E5M2_mxscale_shape_fail) {
  TensorDesc x_desc = TensorDesc({64, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc group_index_desc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
  TensorDesc y_desc = TensorDesc({64, 5}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
  TensorDesc mxscale_desc = TensorDesc({5, 5, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
  int64_t dstType = static_cast<int64_t>(ACL_FLOAT8_E5M2);
  int64_t blocksize = 32;
  const char* roundMode = "rint";

  class SocVersionManager testSocVersion(SocVersion::ASCEND910_95);
  auto ut = OP_API_UT(aclnnGroupedDynamicMxQuant, INPUT(x_desc, group_index_desc, roundMode, dstType, blocksize),
                      OUTPUT(y_desc, mxscale_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_grouped_dynamic_mx_quant_test, ascend910_95_grouped_dynamic_mx_quant_x_dtype_fail) {
  TensorDesc x_desc = TensorDesc({64, 5}, ACL_FLOAT, ACL_FORMAT_ND);
  TensorDesc group_index_desc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
  TensorDesc y_desc = TensorDesc({64, 5}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
  TensorDesc mxscale_desc = TensorDesc({3, 5, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
  int64_t dstType = static_cast<int64_t>(ACL_FLOAT8_E5M2);
  int64_t blocksize = 32;
  const char* roundMode = "rint";

  class SocVersionManager testSocVersion(SocVersion::ASCEND910_95);
  auto ut = OP_API_UT(aclnnGroupedDynamicMxQuant, INPUT(x_desc, group_index_desc, roundMode, dstType, blocksize),
                      OUTPUT(y_desc, mxscale_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_grouped_dynamic_mx_quant_test, ascend910_95_grouped_dynamic_mx_quant_group_dtype_fail) {
  TensorDesc x_desc = TensorDesc({64, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc group_index_desc = TensorDesc({2}, ACL_INT64, ACL_FORMAT_ND);
  TensorDesc y_desc = TensorDesc({64, 5}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
  TensorDesc mxscale_desc = TensorDesc({3, 5, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
  int64_t dstType = static_cast<int64_t>(ACL_FLOAT8_E5M2);
  int64_t blocksize = 32;
  const char* roundMode = "rint";

  class SocVersionManager testSocVersion(SocVersion::ASCEND910_95);
  auto ut = OP_API_UT(aclnnGroupedDynamicMxQuant, INPUT(x_desc, group_index_desc, roundMode, dstType, blocksize),
                      OUTPUT(y_desc, mxscale_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_grouped_dynamic_mx_quant_test, ascend910_95_grouped_dynamic_mx_quant_y_dtype_fail) {
  TensorDesc x_desc = TensorDesc({64, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc group_index_desc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
  TensorDesc y_desc = TensorDesc({64, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc mxscale_desc = TensorDesc({3, 5, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
  int64_t dstType = static_cast<int64_t>(ACL_FLOAT8_E5M2);
  int64_t blocksize = 32;
  const char* roundMode = "rint";

  class SocVersionManager testSocVersion(SocVersion::ASCEND910_95);
  auto ut = OP_API_UT(aclnnGroupedDynamicMxQuant, INPUT(x_desc, group_index_desc, roundMode, dstType, blocksize),
                      OUTPUT(y_desc, mxscale_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_grouped_dynamic_mx_quant_test, ascend910_95_grouped_dynamic_mx_quant_mxscale_dtype_fail) {
  TensorDesc x_desc = TensorDesc({64, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc group_index_desc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
  TensorDesc y_desc = TensorDesc({64, 5}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
  TensorDesc mxscale_desc = TensorDesc({3, 5, 2}, ACL_INT32, ACL_FORMAT_ND);
  int64_t dstType = static_cast<int64_t>(ACL_FLOAT8_E5M2);
  int64_t blocksize = 32;
  const char* roundMode = "rint";

  class SocVersionManager testSocVersion(SocVersion::ASCEND910_95);
  auto ut = OP_API_UT(aclnnGroupedDynamicMxQuant, INPUT(x_desc, group_index_desc, roundMode, dstType, blocksize),
                      OUTPUT(y_desc, mxscale_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_grouped_dynamic_mx_quant_test, ascend910_95_grouped_dynamic_mx_quant_roundmode_fail) {
  TensorDesc x_desc = TensorDesc({64, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc group_index_desc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
  TensorDesc y_desc = TensorDesc({64, 5}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
  TensorDesc mxscale_desc = TensorDesc({3, 5, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
  int64_t dstType = static_cast<int64_t>(ACL_FLOAT8_E5M2);
  int64_t blocksize = 32;
  const char* roundMode = "floor";
  auto ut = OP_API_UT(aclnnGroupedDynamicMxQuant, INPUT(x_desc, group_index_desc, roundMode, dstType, blocksize),
                      OUTPUT(y_desc, mxscale_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_grouped_dynamic_mx_quant_test, ascend910_95_grouped_dynamic_mx_quant_blocksize_fail) {
  TensorDesc x_desc = TensorDesc({64, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc group_index_desc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
  TensorDesc y_desc = TensorDesc({64, 5}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
  TensorDesc mxscale_desc = TensorDesc({3, 5, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
  int64_t dstType = static_cast<int64_t>(ACL_FLOAT8_E5M2);
  int64_t blocksize = 64;
  const char* roundMode = "rint";

  class SocVersionManager testSocVersion(SocVersion::ASCEND910_95);
  auto ut = OP_API_UT(aclnnGroupedDynamicMxQuant, INPUT(x_desc, group_index_desc, roundMode, dstType, blocksize),
                      OUTPUT(y_desc, mxscale_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_grouped_dynamic_mx_quant_test, ascend910_95_grouped_dynamic_mx_quant_dstType_fail) {
  TensorDesc x_desc = TensorDesc({64, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc group_index_desc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
  TensorDesc y_desc = TensorDesc({64, 5}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
  TensorDesc mxscale_desc = TensorDesc({3, 5, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
  int64_t dstType = static_cast<int64_t>(ACL_FLOAT16);
  int64_t blocksize = 32;
  const char* roundMode = "rint";

  class SocVersionManager testSocVersion(SocVersion::ASCEND910_95);
  auto ut = OP_API_UT(aclnnGroupedDynamicMxQuant, INPUT(x_desc, group_index_desc, roundMode, dstType, blocksize),
                      OUTPUT(y_desc, mxscale_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_grouped_dynamic_mx_quant_test, ascend910B2_grouped_dynamic_mx_quant_soc_fail) {
  TensorDesc x_desc = TensorDesc({64, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
  TensorDesc group_index_desc = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
  TensorDesc y_desc = TensorDesc({64, 5}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
  TensorDesc mxscale_desc = TensorDesc({3, 5, 2}, ACL_FLOAT8_E8M0, ACL_FORMAT_ND);
  int64_t dstType = static_cast<int64_t>(ACL_FLOAT8_E5M2);
  int64_t blocksize = 64;
  const char* roundMode = "rint";

  class SocVersionManager testSocVersion(SocVersion::ASCEND910_95);
  auto ut = OP_API_UT(aclnnGroupedDynamicMxQuant, INPUT(x_desc, group_index_desc, roundMode, dstType, blocksize),
                      OUTPUT(y_desc, mxscale_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
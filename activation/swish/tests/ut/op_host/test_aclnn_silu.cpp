/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/
#include <float.h>

#include <array>
#include <iostream>
#include <vector>

#include "gtest/gtest.h"
#include "../../../op_host/op_api/aclnn_silu.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;

class silu_test : public testing::Test {
 protected:
  static void SetUpTestCase() { cout << "sign_test SetUp" << endl; }

  static void TearDownTestCase() { cout << "sign_test TeastDown" << endl; }
};

TEST_F(silu_test, test_silu_dataType_error) {
  vector<aclDataType> ValidList = {ACL_DT_UNDEFINED, ACL_INT8,   ACL_INT16,     ACL_INT32,      ACL_INT64,
                                   ACL_UINT8,        ACL_UINT16, ACL_UINT32,    ACL_UINT64,     ACL_DOUBLE,
                                   ACL_BOOL,         ACL_STRING, ACL_COMPLEX64, ACL_COMPLEX128, ACL_BF16};

  int length = ValidList.size();
  vector<int64_t> input_dim = {2, 16, 32, 16};
  vector<int64_t> result_dim = {2, 16, 32, 16};

  for (int i = 0; i < length; i++) {
    auto inputDesc = TensorDesc(input_dim, ValidList[i], ACL_FORMAT_ND).ValueRange(-2, 2);
    auto outDesc = TensorDesc(result_dim, ValidList[i], ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnSilu, INPUT(inputDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
  }
}

TEST_F(silu_test, test_silu_format) {
  vector<aclFormat> ValidList = {
      ACL_FORMAT_UNDEFINED, ACL_FORMAT_NCHW,        ACL_FORMAT_NHWC,  ACL_FORMAT_ND,    ACL_FORMAT_NC1HWC0,
      ACL_FORMAT_FRACTAL_Z, ACL_FORMAT_NC1HWC0_C04, ACL_FORMAT_HWCN,  ACL_FORMAT_NDHWC, ACL_FORMAT_FRACTAL_NZ,
      ACL_FORMAT_NCDHW,     ACL_FORMAT_NDC1HWC0,    ACL_FRACTAL_Z_3D, ACL_FORMAT_NC,    ACL_FORMAT_NCL};

  int length = ValidList.size();
  vector<int64_t> input_dim = {2, 16, 32, 16};
  vector<int64_t> result_dim = {2, 16, 32, 16};

  for (int i = 0; i < length; i++) {
    auto inputDesc = TensorDesc(input_dim, ACL_FLOAT, ValidList[i]).ValueRange(-1, 1);
    auto outDesc = TensorDesc(result_dim, ACL_FLOAT, ValidList[i]).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnSilu, INPUT(inputDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    ut.TestPrecision();
  }
}

TEST_F(silu_test, test_silu_inconsistent_shape) {
  auto inputDesc = TensorDesc({2, 16, 32, 16}, ACL_FLOAT, ACL_FORMAT_ND);
  auto outDesc = TensorDesc({2, 16, 32, 18}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnSilu, INPUT(inputDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceout = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceout, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(silu_test, test_silu_inconsistent_dtype) {
  auto inputDesc = TensorDesc({2, 16, 32, 16}, ACL_FLOAT, ACL_FORMAT_ND);
  auto outDesc = TensorDesc({2, 16, 32, 16}, ACL_FLOAT16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnSilu, INPUT(inputDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceout = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceout, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(silu_test, test_silu_empty_input) {
  auto inputDesc = TensorDesc({2, 16, 32, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto outDesc = TensorDesc({2, 16, 32, 16}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnSilu, INPUT(inputDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceout = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceout, ACLNN_SUCCESS);
  ut.TestPrecision();
}

TEST_F(silu_test, test_silu_nullptr_input) {
  auto outDesc = TensorDesc({2, 16, 32, 16}, ACL_FLOAT, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnSilu, INPUT((aclTensor *)nullptr), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceout = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceout, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(silu_test, test_silu_nullptr_out) {
  auto inputDesc = TensorDesc({2, 16, 32, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

  auto ut = OP_API_UT(aclnnSilu, INPUT(inputDesc), OUTPUT((aclTensor *)nullptr));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceout = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceout, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(silu_test, test_silu_FP32) {
  auto inputDesc = TensorDesc({2, 16, 32, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto outDesc = TensorDesc({2, 16, 32, 16}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnSilu, INPUT(inputDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceout = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceout, ACLNN_SUCCESS);
  ut.TestPrecision();
}

TEST_F(silu_test, test_silu_FP16) {
  auto inputDesc = TensorDesc({2, 16, 32, 16}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
  auto outDesc = TensorDesc({2, 16, 32, 16}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnSilu, INPUT(inputDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceout = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceout, ACLNN_SUCCESS);
  ut.TestPrecision();
}

TEST_F(silu_test, test_silu_uncontiguous) {
  auto inputDesc = TensorDesc({2, 16}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {16, 2}).ValueRange(-1, 1);
  auto outDesc = TensorDesc({2, 16}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {16, 2}).Precision(0.0001, 0.0001);

  auto ut = OP_API_UT(aclnnSilu, INPUT(inputDesc), OUTPUT(outDesc));

  uint64_t workspaceSize = 0;
  aclnnStatus getWorkspaceout = ut.TestGetWorkspaceSize(&workspaceSize);
  EXPECT_EQ(getWorkspaceout, ACLNN_SUCCESS);
  ut.TestPrecision();
}

TEST_F(silu_test, test_silu_shape_larger_8) {
  vector<aclFormat> ValidList = {
      ACL_FORMAT_UNDEFINED, ACL_FORMAT_NCHW,        ACL_FORMAT_NHWC,  ACL_FORMAT_ND,    ACL_FORMAT_NC1HWC0,
      ACL_FORMAT_FRACTAL_Z, ACL_FORMAT_NC1HWC0_C04, ACL_FORMAT_HWCN,  ACL_FORMAT_NDHWC, ACL_FORMAT_FRACTAL_NZ,
      ACL_FORMAT_NCDHW,     ACL_FORMAT_NDC1HWC0,    ACL_FRACTAL_Z_3D, ACL_FORMAT_NC,    ACL_FORMAT_NCL};

  int length = ValidList.size();

  for (int i = 0; i < length; i++) {
    auto inputDesc = TensorDesc({1, 1, 1, 1, 2, 1, 1, 1, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto resultDesc = TensorDesc({1, 1, 1, 1, 2, 1, 1, 1, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnSilu, INPUT(inputDesc), OUTPUT(resultDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    // ut.TestPrecision();  // comment bcz of timeout in model tests (986327 ms)
  }
}

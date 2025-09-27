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
 #include <float.h>
 #include <gmock/gmock.h>
 #include "gtest/gtest.h"
 #include "../../../op_host/op_api/aclnn_transpose_batch_mat_mul.h"

 #include "op_api_ut_common/array_desc.h"
 #include "op_api_ut_common/inner/types.h"
 #include "op_api_ut_common/op_api_ut.h"
 #include "op_api_ut_common/scalar_desc.h"
 #include "op_api_ut_common/tensor_desc.h"
 #include "opdev/platform.h"

 using namespace op;
 using namespace std;

 class l2_transpose_batch_mat_mul_test : public testing::Test {
  protected:
    static void SetUpTestCase() {
        cout << "l2_transpose_batch_mat_mul_test SetUp" << endl;
    }

    static void TearDownTestCase() {
        cout << "l2_transpose_batch_mat_mul_test TearDown" << endl;
    }
 };

 TEST_F(l2_transpose_batch_mat_mul_test, ascend910B2_tbmm_case_01) {
   int64_t M = 32;
   int64_t K = 512;
   int64_t N = 128;
   int64_t Batch = 16;
   TensorDesc x1_desc = TensorDesc({M, Batch, K}, ACL_FLOAT16, ACL_FORMAT_ND);
   TensorDesc x2_desc = TensorDesc({Batch, K, N}, ACL_FLOAT16, ACL_FORMAT_ND);
   vector<int64_t> perm_x1 = {1, 0, 2};
   vector<int64_t> perm_x2 = {0, 1, 2};
   vector<int64_t> perm_y = {1, 0, 2};

   auto perm_x1_desc = IntArrayDesc(perm_x1);
   auto perm_x2_desc = IntArrayDesc(perm_x2);
   auto perm_y_desc = IntArrayDesc(perm_y);
   int8_t cubeMathType = 1;
   int32_t batch_split_factor = 1;
   TensorDesc out_desc = TensorDesc({M, Batch, N}, ACL_FLOAT16, ACL_FORMAT_ND);

   auto ut = OP_API_UT(aclnnTransposeBatchMatMul, INPUT(x1_desc, x2_desc, nullptr, nullptr,
                       perm_x1_desc, perm_x2_desc, perm_y_desc, cubeMathType, batch_split_factor),
                       OUTPUT(out_desc));
   uint64_t workspace_size = 0;
   aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
 }

 TEST_F(l2_transpose_batch_mat_mul_test, ascend910B2_tbmm_case_02) {
   int64_t M = 32;
   int64_t K = 512;
   int64_t N = 128;
   int64_t Batch = 16;
   TensorDesc x1_desc = TensorDesc({0}, ACL_FLOAT16, ACL_FORMAT_ND);
   TensorDesc x2_desc = TensorDesc({Batch, K, N}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
   vector<int64_t> perm_x1 = {1, 0, 2};
   vector<int64_t> perm_x2 = {0, 1, 2};
   vector<int64_t> perm_y = {1, 0, 2};

   auto perm_x1_desc = IntArrayDesc(perm_x1);
   auto perm_x2_desc = IntArrayDesc(perm_x2);
   auto perm_y_desc = IntArrayDesc(perm_y);
   int8_t cubeMathType = 1;
   int32_t batch_split_factor = 1;
   TensorDesc out_desc = TensorDesc({M, Batch, N}, ACL_FLOAT16, ACL_FORMAT_ND);

   auto ut = OP_API_UT(aclnnTransposeBatchMatMul, INPUT(x1_desc, x2_desc, nullptr, nullptr,
                       perm_x1_desc, perm_x2_desc, perm_y_desc, cubeMathType, batch_split_factor),
                       OUTPUT(out_desc));
   uint64_t workspace_size = 0;
   aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
 }

 TEST_F(l2_transpose_batch_mat_mul_test, ascend910B2_tbmm_case_03) {
   int64_t M = 32;
   int64_t K = 512;
   int64_t N = 128;
   int64_t Batch = 16;
   TensorDesc x1_desc = TensorDesc({M, Batch, K}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
   TensorDesc x2_desc = TensorDesc({0}, ACL_FLOAT16, ACL_FORMAT_ND);
   vector<int64_t> perm_x1 = {1, 0, 2};
   vector<int64_t> perm_x2 = {0, 1, 2};
   vector<int64_t> perm_y = {1, 0, 2};

   auto perm_x1_desc = IntArrayDesc(perm_x1);
   auto perm_x2_desc = IntArrayDesc(perm_x2);
   auto perm_y_desc = IntArrayDesc(perm_y);
   int8_t cubeMathType = 1;
   int32_t batch_split_factor = 1;
   TensorDesc out_desc = TensorDesc({M, Batch, N}, ACL_FLOAT16, ACL_FORMAT_ND);

   auto ut = OP_API_UT(aclnnTransposeBatchMatMul, INPUT(x1_desc, x2_desc, nullptr, nullptr,
                       perm_x1_desc, perm_x2_desc, perm_y_desc, cubeMathType, batch_split_factor),
                       OUTPUT(out_desc));
   uint64_t workspace_size = 0;
   aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
 }

 TEST_F(l2_transpose_batch_mat_mul_test, ascend910B2_tbmm_case_04) {
   int64_t M = 32;
   int64_t K = 512;
   int64_t N = 128;
   int64_t Batch = 16;
   TensorDesc x1_desc = TensorDesc({M, Batch, K}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
   TensorDesc x2_desc = TensorDesc({Batch, K, N}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
   TensorDesc bias_desc = TensorDesc({M, Batch, N}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
   vector<int64_t> perm_x1 = {1, 0, 2};
   vector<int64_t> perm_x2 = {0, 1, 2};
   vector<int64_t> perm_y = {1, 0, 2};

   auto perm_x1_desc = IntArrayDesc(perm_x1);
   auto perm_x2_desc = IntArrayDesc(perm_x2);
   auto perm_y_desc = IntArrayDesc(perm_y);
   int8_t cubeMathType = 1;
   int32_t batch_split_factor = 1;
   TensorDesc out_desc = TensorDesc({M, Batch, N}, ACL_FLOAT16, ACL_FORMAT_ND);

   auto ut = OP_API_UT(aclnnTransposeBatchMatMul, INPUT(x1_desc, x2_desc, bias_desc, nullptr,
                       perm_x1_desc, perm_x2_desc, perm_y_desc, cubeMathType, batch_split_factor),
                       OUTPUT(out_desc));
   uint64_t workspace_size = 0;
   aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
 }

 TEST_F(l2_transpose_batch_mat_mul_test, ascend910B2_tbmm_case_05) {
   int64_t M = 32;
   int64_t K = 512;
   int64_t N = 128;
   int64_t Batch = 16;
   TensorDesc x1_desc = TensorDesc({M, Batch, K}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
   TensorDesc x2_desc = TensorDesc({Batch, K, N}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
   vector<int64_t> perm_x1 = {1, 0, 2, 2, 2, 2, 2, 2};
   vector<int64_t> perm_x2 = {0, 1, 2};
   vector<int64_t> perm_y = {1, 0, 2};

   auto perm_x1_desc = IntArrayDesc(perm_x1);
   auto perm_x2_desc = IntArrayDesc(perm_x2);
   auto perm_y_desc = IntArrayDesc(perm_y);
   int8_t cubeMathType = 1;
   int32_t batch_split_factor = 1;
   TensorDesc out_desc = TensorDesc({M, Batch, N}, ACL_FLOAT16, ACL_FORMAT_ND);

   auto ut = OP_API_UT(aclnnTransposeBatchMatMul, INPUT(x1_desc, x2_desc, nullptr, nullptr,
                       perm_x1_desc, perm_x2_desc, perm_y_desc, cubeMathType, batch_split_factor),
                       OUTPUT(out_desc));
   uint64_t workspace_size = 0;
   aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
 }

 TEST_F(l2_transpose_batch_mat_mul_test, ascend910B2_tbmm_case_06) {
   int64_t M = 32;
   int64_t K = 512;
   int64_t N = 128;
   int64_t Batch = 16;
   TensorDesc x1_desc = TensorDesc({M, Batch, K}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
   TensorDesc x2_desc = TensorDesc({Batch, K, N}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
   vector<int64_t> perm_x1 = {1, 0, 2};
   vector<int64_t> perm_x2 = {0, 1, 2, 2, 2, 2, 2};
   vector<int64_t> perm_y = {1, 0, 2};

   auto perm_x1_desc = IntArrayDesc(perm_x1);
   auto perm_x2_desc = IntArrayDesc(perm_x2);
   auto perm_y_desc = IntArrayDesc(perm_y);
   int8_t cubeMathType = 1;
   int32_t batch_split_factor = 1;
   TensorDesc out_desc = TensorDesc({M, Batch, N}, ACL_FLOAT16, ACL_FORMAT_ND);

   auto ut = OP_API_UT(aclnnTransposeBatchMatMul, INPUT(x1_desc, x2_desc, nullptr, nullptr,
                       perm_x1_desc, perm_x2_desc, perm_y_desc, cubeMathType, batch_split_factor),
                       OUTPUT(out_desc));
   uint64_t workspace_size = 0;
   aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
 }

 TEST_F(l2_transpose_batch_mat_mul_test, ascend910B2_tbmm_case_07) {
   int64_t M = 32;
   int64_t K = 512;
   int64_t N = 128;
   int64_t Batch = 16;
   TensorDesc x1_desc = TensorDesc({M, Batch, K}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
   TensorDesc x2_desc = TensorDesc({Batch, K, N}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
   vector<int64_t> perm_x1 = {1, 0, 2};
   vector<int64_t> perm_x2 = {0, 1, 2};
   vector<int64_t> perm_y = {1, 0, 2, 2, 2, 2, 2, 2};

   auto perm_x1_desc = IntArrayDesc(perm_x1);
   auto perm_x2_desc = IntArrayDesc(perm_x2);
   auto perm_y_desc = IntArrayDesc(perm_y);
   int8_t cubeMathType = 1;
   int32_t batch_split_factor = 1;
   TensorDesc out_desc = TensorDesc({M, Batch, N}, ACL_FLOAT16, ACL_FORMAT_ND);

   auto ut = OP_API_UT(aclnnTransposeBatchMatMul, INPUT(x1_desc, x2_desc, nullptr, nullptr,
                       perm_x1_desc, perm_x2_desc, perm_y_desc, cubeMathType, batch_split_factor),
                       OUTPUT(out_desc));
   uint64_t workspace_size = 0;
   aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
 }

 TEST_F(l2_transpose_batch_mat_mul_test, ascend910B2_tbmm_case_08) {
   int64_t M = 32;
   int64_t K = 512;
   int64_t N = 128;
   int64_t Batch = 16;
   TensorDesc x1_desc = TensorDesc({M, Batch, K+1}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
   TensorDesc x2_desc = TensorDesc({Batch, K, N}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
   vector<int64_t> perm_x1 = {1, 0, 2};
   vector<int64_t> perm_x2 = {0, 1, 2};
   vector<int64_t> perm_y = {1, 0, 2};

   auto perm_x1_desc = IntArrayDesc(perm_x1);
   auto perm_x2_desc = IntArrayDesc(perm_x2);
   auto perm_y_desc = IntArrayDesc(perm_y);
   int8_t cubeMathType = 1;
   int32_t batch_split_factor = 1;
   TensorDesc out_desc = TensorDesc({M, Batch, N}, ACL_FLOAT16, ACL_FORMAT_ND);

   auto ut = OP_API_UT(aclnnTransposeBatchMatMul, INPUT(x1_desc, x2_desc, nullptr, nullptr,
                       perm_x1_desc, perm_x2_desc, perm_y_desc, cubeMathType, batch_split_factor),
                       OUTPUT(out_desc));
   uint64_t workspace_size = 0;
   aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
 }

 TEST_F(l2_transpose_batch_mat_mul_test, ascend910B2_tbmm_case_09) {
   int64_t M = 32;
   int64_t K = 512;
   int64_t N = 128;
   int64_t Batch = 16;
   TensorDesc x1_desc = TensorDesc({M, Batch, K}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
   TensorDesc x2_desc = TensorDesc({Batch, K, N}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
   TensorDesc scale_desc = TensorDesc({Batch * N}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);
   vector<int64_t> perm_x1 = {1, 0, 2};
   vector<int64_t> perm_x2 = {0, 1, 2};
   vector<int64_t> perm_y = {1, 0, 2};

   auto perm_x1_desc = IntArrayDesc(perm_x1);
   auto perm_x2_desc = IntArrayDesc(perm_x2);
   auto perm_y_desc = IntArrayDesc(perm_y);
   int8_t cubeMathType = 1;
   int32_t batch_split_factor = 1;
   TensorDesc out_desc = TensorDesc({M, 1, Batch * N}, ACL_INT8, ACL_FORMAT_ND);

   auto ut = OP_API_UT(aclnnTransposeBatchMatMul, INPUT(x1_desc, x2_desc, nullptr, scale_desc,
                       perm_x1_desc, perm_x2_desc, perm_y_desc, cubeMathType, batch_split_factor),
                       OUTPUT(out_desc));
   uint64_t workspace_size = 0;
   aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
 }

 TEST_F(l2_transpose_batch_mat_mul_test, ascend910B2_tbmm_case_10) {
   int64_t M = 32;
   int64_t K = 512;
   int64_t N = 128;
   int64_t Batch = 16;
   TensorDesc x1_desc = TensorDesc({M, Batch, K}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
   TensorDesc x2_desc = TensorDesc({Batch, K, N}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
   vector<int64_t> perm_x1 = {3, 3, 3};
   vector<int64_t> perm_x2 = {0, 1, 2};
   vector<int64_t> perm_y = {1, 0, 2};

   auto perm_x1_desc = IntArrayDesc(perm_x1);
   auto perm_x2_desc = IntArrayDesc(perm_x2);
   auto perm_y_desc = IntArrayDesc(perm_y);
   int8_t cubeMathType = 1;
   int32_t batch_split_factor = 1;
   TensorDesc out_desc = TensorDesc({M, Batch, N}, ACL_FLOAT16, ACL_FORMAT_ND);

   auto ut = OP_API_UT(aclnnTransposeBatchMatMul, INPUT(x1_desc, x2_desc, nullptr, nullptr,
                       perm_x1_desc, perm_x2_desc, perm_y_desc, cubeMathType, batch_split_factor),
                       OUTPUT(out_desc));
   uint64_t workspace_size = 0;
   aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
 }

 TEST_F(l2_transpose_batch_mat_mul_test, ascend910B2_tbmm_case_11) {
    int64_t M = 32;
    int64_t K = 512;
    int64_t N = 128;
    int64_t Batch = 16;
    TensorDesc x1_desc = TensorDesc({M, Batch, K}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc x2_desc = TensorDesc({Batch, K, N}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    TensorDesc scale_desc = TensorDesc({Batch}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);
    vector<int64_t> perm_x1 = {1, 0, 2};
    vector<int64_t> perm_x2 = {0, 1, 2};
    vector<int64_t> perm_y = {1, 0, 2};

    auto perm_x1_desc = IntArrayDesc(perm_x1);
    auto perm_x2_desc = IntArrayDesc(perm_x2);
    auto perm_y_desc = IntArrayDesc(perm_y);
    int8_t cubeMathType = 1;
    int32_t batch_split_factor = 1;
    TensorDesc out_desc = TensorDesc({M, 1, Batch * N}, ACL_INT8, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnTransposeBatchMatMul, INPUT(x1_desc, x2_desc, nullptr, scale_desc,
                        perm_x1_desc, perm_x2_desc, perm_y_desc, cubeMathType, batch_split_factor),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
  }

 TEST_F(l2_transpose_batch_mat_mul_test, ascend910B2_tbmm_case_12) {
   int64_t M = 32;
   int64_t K = 512;
   int64_t N = 128;
   int64_t Batch = 16;
   TensorDesc x1_desc = TensorDesc({M, Batch, K}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
   TensorDesc x2_desc = TensorDesc({Batch, K, N}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
   TensorDesc scale_desc = TensorDesc({Batch * N}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);
   vector<int64_t> perm_x1 = {1, 0, 2};
   vector<int64_t> perm_x2 = {0, 1, 2};
   vector<int64_t> perm_y = {1, 0, 2};

   auto perm_x1_desc = IntArrayDesc(perm_x1);
   auto perm_x2_desc = IntArrayDesc(perm_x2);
   auto perm_y_desc = IntArrayDesc(perm_y);
   int8_t cubeMathType = 1;
   int32_t batch_split_factor = 7;
   TensorDesc out_desc = TensorDesc({M, 1, Batch * N}, ACL_INT8, ACL_FORMAT_ND);

   auto ut = OP_API_UT(aclnnTransposeBatchMatMul, INPUT(x1_desc, x2_desc, nullptr, scale_desc,
                       perm_x1_desc, perm_x2_desc, perm_y_desc, cubeMathType, batch_split_factor),
                       OUTPUT(out_desc));
   uint64_t workspace_size = 0;
   aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
 }

 TEST_F(l2_transpose_batch_mat_mul_test, ascend910B2_tbmm_case_13) {
   int64_t M = 32;
   int64_t K = 512;
   int64_t N = 128;
   int64_t Batch = 16;
   TensorDesc x1_desc = TensorDesc({M, Batch, K}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
   TensorDesc x2_desc = TensorDesc({Batch, K, N}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
   vector<int64_t> perm_x1 = {1, 0, 2};
   vector<int64_t> perm_x2 = {0, 1, 2};
   vector<int64_t> perm_y = {1, 0, 2};

   auto perm_x1_desc = IntArrayDesc(perm_x1);
   auto perm_x2_desc = IntArrayDesc(perm_x2);
   auto perm_y_desc = IntArrayDesc(perm_y);
   int8_t cubeMathType = 1;
   int32_t batch_split_factor = 4;
   TensorDesc out_desc = TensorDesc({4, M, Batch * N / 4}, ACL_FLOAT16, ACL_FORMAT_ND);

   auto ut = OP_API_UT(aclnnTransposeBatchMatMul, INPUT(x1_desc, x2_desc, nullptr, nullptr,
                       perm_x1_desc, perm_x2_desc, perm_y_desc, cubeMathType, batch_split_factor),
                       OUTPUT(out_desc));
   uint64_t workspace_size = 0;
   aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
 }

 TEST_F(l2_transpose_batch_mat_mul_test, ascend910B2_tbmm_case_14) {
   int64_t M = 32;
   int64_t K = 512;
   int64_t N = 128;
   int64_t Batch = 16;
   TensorDesc x1_desc = TensorDesc({M, Batch, K}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
   TensorDesc x2_desc = TensorDesc({Batch, K, N}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
   vector<int64_t> perm_x1 = {1, 0, 2};
   vector<int64_t> perm_x2 = {0, 1, 2};
   vector<int64_t> perm_y = {1, 0, 2};

   auto perm_x1_desc = IntArrayDesc(perm_x1);
   auto perm_x2_desc = IntArrayDesc(perm_x2);
   auto perm_y_desc = IntArrayDesc(perm_y);
   int8_t cubeMathType = 1;
   int32_t batch_split_factor = 5;
   TensorDesc out_desc = TensorDesc({5, M, Batch * N / 5}, ACL_FLOAT16, ACL_FORMAT_ND);

   auto ut = OP_API_UT(aclnnTransposeBatchMatMul, INPUT(x1_desc, x2_desc, nullptr, nullptr,
                       perm_x1_desc, perm_x2_desc, perm_y_desc, cubeMathType, batch_split_factor),
                       OUTPUT(out_desc));
   uint64_t workspace_size = 0;
   aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
 }
 TEST_F(l2_transpose_batch_mat_mul_test, ascend910B2_tbmm_case_15) {
  int64_t M = 32;
  int64_t K = 512;
  int64_t N = 128;
  int64_t Batch = 16;
  TensorDesc x1_desc = TensorDesc({M, Batch, K}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
  TensorDesc x2_desc = TensorDesc({Batch, K, N}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
  TensorDesc scale_desc = TensorDesc({Batch * N}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-1, 1);
  vector<int64_t> perm_x1 = {1, 0, 2};
  vector<int64_t> perm_x2 = {0, 1, 2};
  vector<int64_t> perm_y = {1, 0, 2};

  auto perm_x1_desc = IntArrayDesc(perm_x1);
  auto perm_x2_desc = IntArrayDesc(perm_x2);
  auto perm_y_desc = IntArrayDesc(perm_y);
  int8_t cubeMathType = 1;
  int32_t batch_split_factor = 1;
  TensorDesc out_desc = TensorDesc({M, 1, Batch * N}, ACL_INT8, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnTransposeBatchMatMul, INPUT(x1_desc, x2_desc, nullptr, scale_desc,
                      perm_x1_desc, perm_x2_desc, perm_y_desc, cubeMathType, batch_split_factor),
                      OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}
TEST_F(l2_transpose_batch_mat_mul_test, ascend910B2_tbmm_case_16) {
  int64_t M = 32;
  int64_t K = 512;
  int64_t N = 128;
  int64_t Batch = 16;
  TensorDesc x1_desc = TensorDesc({M, Batch, K}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
  TensorDesc x2_desc = TensorDesc({Batch, K, N}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
  vector<int64_t> perm_x1 = {1, 0, 2};
  vector<int64_t> perm_x2 = {0, 1, 2};
  vector<int64_t> perm_y = {1, 0, 2};

  auto perm_x1_desc = IntArrayDesc(perm_x1);
  auto perm_x2_desc = IntArrayDesc(perm_x2);
  auto perm_y_desc = IntArrayDesc(perm_y);
  int8_t cubeMathType = 1;
  int32_t batch_split_factor = 8;
  TensorDesc out_desc = TensorDesc({8, M, Batch * N / 8}, ACL_FLOAT16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnTransposeBatchMatMul, INPUT(x1_desc, x2_desc, nullptr, nullptr,
                      perm_x1_desc, perm_x2_desc, perm_y_desc, cubeMathType, batch_split_factor),
                      OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}
TEST_F(l2_transpose_batch_mat_mul_test, ascend910B2_tbmm_case_17) {
  int64_t M = 32;
  int64_t K = 512;
  int64_t N = 128;
  int64_t Batch = 256;
  TensorDesc x1_desc = TensorDesc({M, Batch, K}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
  TensorDesc x2_desc = TensorDesc({Batch, K, N}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
  vector<int64_t> perm_x1 = {1, 0, 2};
  vector<int64_t> perm_x2 = {0, 1, 2};
  vector<int64_t> perm_y = {1, 0, 2};

  auto perm_x1_desc = IntArrayDesc(perm_x1);
  auto perm_x2_desc = IntArrayDesc(perm_x2);
  auto perm_y_desc = IntArrayDesc(perm_y);
  int8_t cubeMathType = 1;
  int32_t batch_split_factor = 8;
  TensorDesc out_desc = TensorDesc({8, M, Batch * N / 8}, ACL_FLOAT16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnTransposeBatchMatMul, INPUT(x1_desc, x2_desc, nullptr, nullptr,
                      perm_x1_desc, perm_x2_desc, perm_y_desc, cubeMathType, batch_split_factor),
                      OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}
TEST_F(l2_transpose_batch_mat_mul_test, ascend910B2_tbmm_case_18) {
  int64_t M = 32;
  int64_t K = 512;
  int64_t N = 65536;
  int64_t Batch = 16;
  TensorDesc x1_desc = TensorDesc({Batch, M, K}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
  TensorDesc x2_desc = TensorDesc({Batch, K, N}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
  vector<int64_t> perm_x1 = {0, 1, 2};
  vector<int64_t> perm_x2 = {0, 1, 2};
  vector<int64_t> perm_y = {1, 0, 2};

  auto perm_x1_desc = IntArrayDesc(perm_x1);
  auto perm_x2_desc = IntArrayDesc(perm_x2);
  auto perm_y_desc = IntArrayDesc(perm_y);
  int8_t cubeMathType = 1;
  int32_t batch_split_factor = 8;
  TensorDesc out_desc = TensorDesc({8, M, Batch * N / 8}, ACL_FLOAT16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnTransposeBatchMatMul, INPUT(x1_desc, x2_desc, nullptr, nullptr,
                      perm_x1_desc, perm_x2_desc, perm_y_desc, cubeMathType, batch_split_factor),
                      OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}
TEST_F(l2_transpose_batch_mat_mul_test, ascend910B2_tbmm_case_19) {
  int64_t M = 32;
  int64_t K = 512;
  int64_t N = 128;
  int64_t Batch = 16;
  TensorDesc x1_desc = TensorDesc({M, Batch, K}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
  TensorDesc x2_desc = TensorDesc({Batch, K, N}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
  vector<int64_t> perm_x1 = {3, 3, 3};
  vector<int64_t> perm_x2 = {0, 1, 2};
  vector<int64_t> perm_y = {1, 0, 2};

  auto perm_x1_desc = IntArrayDesc(perm_x1);
  auto perm_x2_desc = IntArrayDesc(perm_x2);
  auto perm_y_desc = IntArrayDesc(perm_y);
  int8_t cubeMathType = 1;
  int32_t batch_split_factor = -1;
  TensorDesc out_desc = TensorDesc({M, Batch, N}, ACL_FLOAT16, ACL_FORMAT_ND);

  auto ut = OP_API_UT(aclnnTransposeBatchMatMul, INPUT(x1_desc, x2_desc, nullptr, nullptr,
                      perm_x1_desc, perm_x2_desc, perm_y_desc, cubeMathType, batch_split_factor),
                      OUTPUT(out_desc));
  uint64_t workspace_size = 0;
  aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}
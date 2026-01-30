/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

 #include <array>
 #include <vector>
 #include <float.h>
 #include <gmock/gmock.h>
 #include "gtest/gtest.h"
 #include "../../../op_api/aclnn_transpose_quant_batch_mat_mul.h"

 #include "op_api_ut_common/array_desc.h"
 #include "op_api_ut_common/inner/types.h"
 #include "op_api_ut_common/op_api_ut.h"
 #include "op_api_ut_common/scalar_desc.h"
 #include "op_api_ut_common/tensor_desc.h"
 #include "opdev/platform.h"

 using namespace op;
 using namespace std;

 class l2_transpose_quant_batch_mat_mul_test : public testing::Test {
  protected:
    static void SetUpTestCase() {
        cout << "l2_transpose_quant_batch_mat_mul_test SetUp" << endl;
    }

    static void TearDownTestCase() {
        cout << "l2_transpose_quant_batch_mat_mul_test TearDown" << endl;
    }
 };

 TEST_F(l2_transpose_quant_batch_mat_mul_test, ascend91095_tqbmm_case_01) {
  op::SocVersionManager versionManager(op::SocVersion::ASCEND310);
   int64_t M = 32;
   int64_t K = 512;
   int64_t N = 128;
   int64_t Batch = 16;
   // 1 0 2
   TensorDesc x1_desc = TensorDesc({M, Batch, K}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
   // 0 1 2
   TensorDesc x2_desc = TensorDesc({Batch, K, N}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
   vector<int64_t> perm_x1 = {1, 0, 2};
   vector<int64_t> perm_x2 = {0, 1, 2};
   vector<int64_t> perm_y = {1, 0, 2};
   TensorDesc x1Scale_desc = TensorDesc({32, }, ACL_FLOAT, ACL_FORMAT_ND);
   TensorDesc x2Scale_desc = TensorDesc({128, }, ACL_FLOAT, ACL_FORMAT_ND);
   auto perm_x1_desc = IntArrayDesc(perm_x1);
   auto perm_x2_desc = IntArrayDesc(perm_x2);
   auto perm_y_desc = IntArrayDesc(perm_y);
   int32_t groupSize = 0;
   int32_t dtype = 1; // FP16
   int32_t batch_split_factor = 1;
   TensorDesc out_desc = TensorDesc({M, Batch, N}, ACL_FLOAT16, ACL_FORMAT_ND);

   auto ut = OP_API_UT(
       aclnnTransposeQuantBatchMatMul,
       INPUT(
           x1_desc, x2_desc, nullptr, x1Scale_desc, x2Scale_desc, dtype, groupSize, perm_x1_desc, perm_x2_desc,
           perm_y_desc, batch_split_factor),
       OUTPUT(out_desc));
   uint64_t workspace_size = 0;
   aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
 }


 TEST_F(l2_transpose_quant_batch_mat_mul_test, ascend91095_tqbmm_case_02) {
  op::SocVersionManager versionManager(op::SocVersion::ASCEND310);
   int64_t M = 32;
   int64_t K = 512;
   int64_t N = 128;
   int64_t Batch = 16;
   // 1 0 2
   TensorDesc x1_desc = TensorDesc({M, Batch, K}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
   // 0 1 2
   TensorDesc x2_desc = TensorDesc({Batch, K, N}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
   vector<int64_t> perm_x1 = {1, 0, 2};
   vector<int64_t> perm_x2 = {0, 1, 2};
   vector<int64_t> perm_y = {1, 0, 2};
   TensorDesc x1Scale_desc = TensorDesc({32, }, ACL_FLOAT, ACL_FORMAT_ND);
   TensorDesc x2Scale_desc = TensorDesc({128, }, ACL_FLOAT, ACL_FORMAT_ND);
   auto perm_x1_desc = IntArrayDesc(perm_x1);
   auto perm_x2_desc = IntArrayDesc(perm_x2);
   auto perm_y_desc = IntArrayDesc(perm_y);
   int32_t groupSize = 0;
   int32_t dtype = 1; // FP16
   int32_t batch_split_factor = 1;
   TensorDesc out_desc = TensorDesc({M, Batch, N}, ACL_FLOAT16, ACL_FORMAT_ND);

   auto ut = OP_API_UT(
       aclnnTransposeQuantBatchMatMul,
       INPUT(
           x1_desc, x2_desc, nullptr, x1Scale_desc, x2Scale_desc, dtype, groupSize, perm_x1_desc, perm_x2_desc,
           perm_y_desc, batch_split_factor),
       OUTPUT(out_desc));
   uint64_t workspace_size = 0;
   aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
 }

 TEST_F(l2_transpose_quant_batch_mat_mul_test, ascend91095_tqbmm_case_03) {
  op::SocVersionManager versionManager(op::SocVersion::ASCEND310);
   int64_t M = 32;
   int64_t K = 512;
   int64_t N = 128;
   int64_t Batch = 16;
   // 1 0 2
   TensorDesc x1_desc = TensorDesc({M, Batch, K}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
   // 0 1 2
   TensorDesc x2_desc = TensorDesc({Batch, K, N}, ACL_FLOAT8_E5M2, ACL_FORMAT_ND);
   vector<int64_t> perm_x1 = {1, 0, 2};
   vector<int64_t> perm_x2 = {0, 1, 2};
   vector<int64_t> perm_y = {1, 0, 2};
   TensorDesc x1Scale_desc = TensorDesc({32, }, ACL_FLOAT, ACL_FORMAT_ND);
   TensorDesc x2Scale_desc = TensorDesc({128, }, ACL_FLOAT, ACL_FORMAT_ND);
   auto perm_x1_desc = IntArrayDesc(perm_x1);
   auto perm_x2_desc = IntArrayDesc(perm_x2);
   auto perm_y_desc = IntArrayDesc(perm_y);
   int32_t groupSize = 0;
   int32_t dtype = 27; // BF16
   int32_t batch_split_factor = 1;
   TensorDesc out_desc = TensorDesc({M, Batch, N}, ACL_BF16, ACL_FORMAT_ND);

   auto ut = OP_API_UT(
       aclnnTransposeQuantBatchMatMul,
       INPUT(
           x1_desc, x2_desc, nullptr, x1Scale_desc, x2Scale_desc, dtype, groupSize, perm_x1_desc, perm_x2_desc,
           perm_y_desc, batch_split_factor),
       OUTPUT(out_desc));
   uint64_t workspace_size = 0;
   aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
 }

 TEST_F(l2_transpose_quant_batch_mat_mul_test, ascend91095_tqbmm_case_04) {
  op::SocVersionManager versionManager(op::SocVersion::ASCEND310);
   int64_t M = 32;
   int64_t K = 512;
   int64_t N = 128;
   int64_t Batch = 16;
   // 1 0 2
   TensorDesc x1_desc = TensorDesc({M, Batch, K}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
   // 0 1 2
   TensorDesc x2_desc = TensorDesc({Batch, K, N}, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND);
   vector<int64_t> perm_x1 = {1, 0, 2};
   vector<int64_t> perm_x2 = {0, 1, 2};
   vector<int64_t> perm_y = {1, 0, 2};
   TensorDesc x1Scale_desc = TensorDesc({32, }, ACL_FLOAT, ACL_FORMAT_ND);
   TensorDesc x2Scale_desc = TensorDesc({128, }, ACL_FLOAT, ACL_FORMAT_ND);
   auto perm_x1_desc = IntArrayDesc(perm_x1);
   auto perm_x2_desc = IntArrayDesc(perm_x2);
   auto perm_y_desc = IntArrayDesc(perm_y);
   int32_t groupSize = 0;
   int32_t dtype = 27; // BF16
   int32_t batch_split_factor = 1;
   TensorDesc out_desc = TensorDesc({M, Batch, N}, ACL_BF16, ACL_FORMAT_ND);

   auto ut = OP_API_UT(
       aclnnTransposeQuantBatchMatMul,
       INPUT(
           x1_desc, x2_desc, nullptr, x1Scale_desc, x2Scale_desc, dtype, groupSize, perm_x1_desc, perm_x2_desc,
           perm_y_desc, batch_split_factor),
       OUTPUT(out_desc));
   uint64_t workspace_size = 0;
   aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
 }
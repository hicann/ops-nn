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
#include <thread>
#include <gmock/gmock.h>
#include <vector>
#include <array>
#include "gtest/gtest.h"

#include "../../../op_host/op_api/aclnn_fused_quant_matmul.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace std;
using namespace op;

class l2_FusedQuantMatmul_test : public testing::Test {
 protected:
  static void SetUpTestCase() { cout << "l2_FusedQuantMatmul_test SetUp" << endl; }

  static void TearDownTestCase() { cout << "l2_FusedQuantMatmul_test TearDown" << endl; }
};


TEST_F(l2_FusedQuantMatmul_test, ascend910B1_test_case_A8W8_1)
{   
    // A8W8_pertoken_perchannel_gelutanh_out_fp16
    TensorDesc x1_desc = TensorDesc({27, 8}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1,1);
    TensorDesc x2_desc = TensorDesc({8, 64}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1,1);
    TensorDesc x1Scale_desc = TensorDesc({27}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1,1);
    TensorDesc x2Scale_desc = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1,1);
    TensorDesc out_desc = TensorDesc({27, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnFusedQuantMatmul, INPUT(x1_desc, x2_desc, x1Scale_desc, x2Scale_desc, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 
                            "gelu_tanh", false, false, 0),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_FusedQuantMatmul_test, ascend910B1_test_case_A8W8_2)
{   
    // A8W8_pertoken_perchannel_gelutanh_out_bf16
    TensorDesc x1_desc = TensorDesc({27, 8}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1,1);
    TensorDesc x2_desc = TensorDesc({8, 64}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1,1);
    TensorDesc x1Scale_desc = TensorDesc({27}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1,1);
    TensorDesc x2Scale_desc = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1,1);
    TensorDesc out_desc = TensorDesc({27, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnFusedQuantMatmul, INPUT(x1_desc, x2_desc, x1Scale_desc, x2Scale_desc, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 
                            "gelu_tanh", false, false, 0),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_FusedQuantMatmul_test, ascend910B1_test_case_A8W8_3)
{   
    // A8W8_weightnz_pertoken_perchannel_geluerf_out_bf16
    TensorDesc x1_desc = TensorDesc({4, 32}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-1,1);
    TensorDesc x2_desc = TensorDesc({32, 64}, ACL_INT8, ACL_FORMAT_FRACTAL_NZ, {}, 0, {2, 2, 16, 32});
    TensorDesc x1Scale_desc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1,1);
    TensorDesc x2Scale_desc = TensorDesc({64}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1,1);
    TensorDesc out_desc = TensorDesc({4, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnFusedQuantMatmul, INPUT(x1_desc, x2_desc, x1Scale_desc, x2Scale_desc, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 
                            "gelu_erf", false, false, 0),
                        OUTPUT(out_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}
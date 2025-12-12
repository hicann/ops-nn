/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/
#include <array>
#include <vector>
#include "gtest/gtest.h"

#include "../../../../op_host/op_api/aclnn_prelu.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;

class l2_prelu_test : public testing::Test {
 protected:
  static void SetUpTestCase() { cout << "prelu_test SetUp" << endl; }

  static void TearDownTestCase() { cout << "prelu_test TearDown" << endl; }
};

TEST_F(l2_prelu_test, case_1) {
    auto self_tensor_desc = TensorDesc({1, 16, 4, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-1, 1);
    auto weight_tensor_desc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({1, 16, 4, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnPrelu, INPUT(self_tensor_desc, weight_tensor_desc), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_prelu_test, case_NHWC_1) {
    auto self_tensor_desc = TensorDesc({1, 16, 4, 16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto weight_tensor_desc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({1, 16, 4, 16}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnPrelu, INPUT(self_tensor_desc, weight_tensor_desc), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_prelu_test, case_NCHW) {
    auto self_tensor_desc = TensorDesc({1, 8, 4, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-1, 1);
    auto weight_tensor_desc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto out_desc = TensorDesc({1, 8, 4, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnPrelu, INPUT(self_tensor_desc, weight_tensor_desc), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_prelu_test, case_HWCN_1) {
    auto self_tensor_desc = TensorDesc({1, 16, 4, 4}, ACL_FLOAT, ACL_FORMAT_HWCN).ValueRange(-1, 1);
    auto weight_tensor_desc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_HWCN);
    auto out_desc = TensorDesc({1, 16, 4, 4}, ACL_FLOAT, ACL_FORMAT_HWCN).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnPrelu, INPUT(self_tensor_desc, weight_tensor_desc), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_prelu_test, case_HWCN_2) {
    auto self_tensor_desc = TensorDesc({1, 4, 16, 4}, ACL_FLOAT, ACL_FORMAT_HWCN).ValueRange(-1, 1);
    auto weight_tensor_desc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_HWCN);
    auto out_desc = TensorDesc({1, 4, 16, 4}, ACL_FLOAT, ACL_FORMAT_HWCN).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnPrelu, INPUT(self_tensor_desc, weight_tensor_desc), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_prelu_test, case_NCDHW) {
    auto self_tensor_desc = TensorDesc({1, 16, 4, 4, 7}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
    auto weight_tensor_desc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto out_desc = TensorDesc({1, 16, 4, 4, 7}, ACL_FLOAT, ACL_FORMAT_NCDHW).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnPrelu, INPUT(self_tensor_desc, weight_tensor_desc), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_prelu_test, case_empty) {
    auto self_tensor_desc = TensorDesc({1, 16, 0, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-1, 1);
    auto weight_tensor_desc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto out_desc = TensorDesc({1, 16, 0, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnPrelu, INPUT(self_tensor_desc, weight_tensor_desc), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    // ut.TestPrecision();
}

TEST_F(l2_prelu_test, case_2) {
    auto self_tensor_desc = TensorDesc({1, 8, 4, 4}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto weight_tensor_desc = TensorDesc({8}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({1, 8, 4, 4}, ACL_FLOAT16, ACL_FORMAT_NCHW).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnPrelu, INPUT(self_tensor_desc, weight_tensor_desc), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // fp16 CPU不支持
    // ut.TestPrecision();
}

TEST_F(l2_prelu_test, case_4) {
    auto self_tensor_desc = TensorDesc({4, 8, 2, 4}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto weight_tensor_desc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({4, 8, 2, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnPrelu, INPUT(self_tensor_desc, weight_tensor_desc), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_prelu_test, case_5) {
    auto self_tensor_desc = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_NDHWC);
    auto weight_tensor_desc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT, ACL_FORMAT_NDHWC).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnPrelu, INPUT(self_tensor_desc, weight_tensor_desc), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_prelu_test, case_6) {
    auto self_tensor_desc = TensorDesc({4, 8, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto weight_tensor_desc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({4, 8, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnPrelu, INPUT(self_tensor_desc, weight_tensor_desc), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_prelu_test, ascend910B2_case_bf16) {
    auto self_tensor_desc = TensorDesc({4, 8, 4}, ACL_BF16, ACL_FORMAT_ND);
    auto weight_tensor_desc = TensorDesc({8}, ACL_BF16, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({4, 8, 4}, ACL_BF16, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnPrelu, INPUT(self_tensor_desc, weight_tensor_desc), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    //ut.TestPrecision();
}

// CheckNotNull
TEST_F(l2_prelu_test, case_null) {
    auto self_tensor_desc = TensorDesc({1, 8, 4, 4}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto weight_tensor_desc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({1, 8, 4, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnPrelu, INPUT(nullptr, weight_tensor_desc), OUTPUT(out_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut_2 = OP_API_UT(aclnnPrelu, INPUT(self_tensor_desc, nullptr), OUTPUT(out_desc));
    // SAMPLE: only test GetWorkspaceSize
    aclRet = ut_2.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut_3 = OP_API_UT(aclnnPrelu, INPUT(self_tensor_desc, weight_tensor_desc), OUTPUT(nullptr));
    // SAMPLE: only test GetWorkspaceSize
    aclRet = ut_3.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_prelu_test, case_nine_dim) {
    auto self_tensor_desc = TensorDesc({10, 24, 3, 5, 10, 22, 42, 30, 24}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto weight_tensor_desc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({10, 24, 3, 5, 10, 22, 42, 30, 24}, ACL_FLOAT, ACL_FORMAT_NHWC).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnPrelu, INPUT(self_tensor_desc, weight_tensor_desc), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_prelu_test, case_weight_1_selfdim_4) {
    auto self_tensor_desc = TensorDesc({2, 4, 3, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto weight_tensor_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({2, 4, 3, 5}, ACL_FLOAT, ACL_FORMAT_NHWC).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnPrelu, INPUT(self_tensor_desc, weight_tensor_desc), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_prelu_test, case_weight_gt_1_selfdim_1) {
    auto self_tensor_desc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto weight_tensor_desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_NHWC).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnPrelu, INPUT(self_tensor_desc, weight_tensor_desc), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_prelu_test, case_weight_gt_1_selfdim_gt_1) {
    auto self_tensor_desc = TensorDesc({2, 6}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto weight_tensor_desc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({2, 6}, ACL_FLOAT, ACL_FORMAT_NHWC).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(aclnnPrelu, INPUT(self_tensor_desc, weight_tensor_desc), OUTPUT(out_desc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    // SAMPLE: precision simulate
    ut.TestPrecision();
}

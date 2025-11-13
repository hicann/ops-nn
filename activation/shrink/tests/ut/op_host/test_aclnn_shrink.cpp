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
#include <gtest/gtest.h>

#include "../../../op_host/op_api/aclnn_shrink.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class l2_shrink_test : public testing::Test {
protected:
    static void SetUpTestCase() {std::cout << "shrink_test Setup" << std::endl;}
    static void TearDownTestCase() {std::cout << "shrink_test TearDown" << std::endl;}
};

TEST_F(l2_shrink_test, case_1){
    auto tensor_desc = TensorDesc({5, 6, 7}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-20, 20);
    auto lambd = ScalarDesc(1.0f);
    auto bias = ScalarDesc(0.0f);
    auto out_tensor_desc = TensorDesc({5, 6, 7}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnShrink, INPUT(tensor_desc, lambd, bias), OUTPUT(out_tensor_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_shrink_test, case_2){
    auto tensor_desc = TensorDesc({5, 6, 7}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-20, 20);
    auto lambd = ScalarDesc(1.0f);
    auto bias = ScalarDesc(0.0f);
    auto out_tensor_desc = TensorDesc({5, 6, 7}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnShrink, INPUT(tensor_desc, lambd, bias), OUTPUT(out_tensor_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_shrink_test, case_3){
    auto tensor_desc = TensorDesc({5, 6, 7}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-20, 20);
    auto lambd = ScalarDesc(1.0f);
    auto bias = ScalarDesc(0.0f);
    auto out_tensor_desc = TensorDesc({5, 6, 7}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnShrink, INPUT(tensor_desc, lambd, bias), OUTPUT(out_tensor_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_shrink_test, case_4){
    auto tensor_desc = TensorDesc({5, 6, 7}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-20, 20);
    auto lambd = ScalarDesc(1.0f);
    auto bias = ScalarDesc(0.0f);
    auto out_tensor_desc = TensorDesc({5, 6, 7}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnShrink, INPUT(tensor_desc, lambd, bias), OUTPUT(out_tensor_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_shrink_test, case_5)
{
    auto tensor_desc = TensorDesc({5, 6, 7}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-20, 20);
    auto lambd = ScalarDesc(1.0f);
    auto bias = ScalarDesc(0.0f);
    auto out_tensor_desc = TensorDesc({5, 6, 7}, ACL_DOUBLE, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnShrink, INPUT(tensor_desc, lambd, bias), OUTPUT(out_tensor_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_shrink_test, case_6)
{
    auto tensor_desc = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-20, 20);
    auto lambd = ScalarDesc(1.0f);
    auto bias = ScalarDesc(0.0f);
    auto out_tensor_desc = TensorDesc({5, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnShrink, INPUT(nullptr, lambd, bias), OUTPUT(out_tensor_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}
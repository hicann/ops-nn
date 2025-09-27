/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <vector>
#include <array>
#include "gtest/gtest.h"

#include "../../../op_host/op_api/aclnn_instance_norm.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

class l2_instance_norm_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "instance_norm_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "instance_norm_test TearDown" << endl;
    }
};

// 正常路径，nchw格式
TEST_F(l2_instance_norm_test, ascend310P_instance_norm_test_nchw)
{
    auto xDesc = TensorDesc({2, 8, 8, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gammaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto betaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);

    auto yDesc = TensorDesc({2, 8, 8, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto meanDesc = TensorDesc({2, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto varDesc = TensorDesc({2, 8, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    const char* dataFormat = "NCHW";
    double eps = 1e-5;

    auto ut = OP_API_UT(
        aclnnInstanceNorm, INPUT(xDesc, gammaDesc, betaDesc, dataFormat, eps), OUTPUT(yDesc, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    // EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 正常路径，nchw格式
TEST_F(l2_instance_norm_test, ascend310P_instance_norm_test_nhwc)
{
    auto xDesc = TensorDesc({2, 8, 16, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gammaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto betaDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);

    auto yDesc = TensorDesc({2, 8, 16, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto meanDesc = TensorDesc({2, 1, 1, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto varDesc = TensorDesc({2, 1, 1, 8}, ACL_FLOAT, ACL_FORMAT_ND);

    const char* dataFormat = "NHWC";
    double eps = 1e-5;

    auto ut = OP_API_UT(
        aclnnInstanceNorm, INPUT(xDesc, gammaDesc, betaDesc, dataFormat, eps), OUTPUT(yDesc, meanDesc, varDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    // EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

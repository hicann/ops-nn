/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "gtest/gtest.h"
#include <array>
#include <vector>

#include "../../../op_host/op_api/aclnn_mish.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

#include <unistd.h>
#include "opdev/platform.h"

using namespace op;
using namespace std;

class l2_mish_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "mish_test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "mish_test TearDown" << std::endl;
    }
};

// 测试合法数据类型float
TEST_F(l2_mish_test, case_001_float)
{
    auto selfDesc = TensorDesc({2, 2, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto outDesc = TensorDesc({2, 2, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnMish, INPUT(selfDesc), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 测试合法数据类型float16
TEST_F(l2_mish_test, case_002_float16)
{
    auto selfDesc = TensorDesc({2, 2, 3, 2}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto outDesc = TensorDesc({2, 2, 3, 2}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnMish, INPUT(selfDesc), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 测试合法数据类型bfloat16
TEST_F(l2_mish_test, ascend910B2_case_003_support_bf16)
{
    auto selfDesc = TensorDesc({2, 2, 3, 2}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto outDesc = TensorDesc({2, 2, 3, 2}, ACL_BF16, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnMish, INPUT(selfDesc), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 测试非法数据类型double
TEST_F(l2_mish_test, case_004_float64)
{
    auto selfDesc = TensorDesc({2, 2, 3, 2}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto outDesc = TensorDesc({2, 2, 3, 2}, ACL_DOUBLE, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnMish, INPUT(selfDesc), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 测试非法数据类型int
TEST_F(l2_mish_test, case_005_int32)
{
    auto selfDesc = TensorDesc({2, 2, 3, 2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);

    auto outDesc = TensorDesc({2, 2, 3, 2}, ACL_INT32, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnMish, INPUT(selfDesc), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 测试空Tensor
TEST_F(l2_mish_test, case_006_empty_tensor)
{
    auto selfDesc = TensorDesc({1, 1, 0, 2}, ACL_FLOAT, ACL_FORMAT_ND);

    auto outDesc = TensorDesc({1, 1, 0, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnMish, INPUT(selfDesc), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 测试非连续
TEST_F(l2_mish_test, case_007_not_contiguous)
{
    auto selfDesc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(-2, 2);

    auto outDesc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnMish, INPUT(selfDesc), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 测试空指针
TEST_F(l2_mish_test, case_008_nullptr)
{
    auto tensorDesc = TensorDesc({10, 5}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut_1 = OP_API_UT(aclnnMish, INPUT(nullptr), OUTPUT(tensorDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut_1.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut_2 = OP_API_UT(aclnnMish, INPUT(tensorDesc), OUTPUT(nullptr));
    // SAMPLE: only test GetWorkspaceSize
    aclRet = ut_2.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// 测试维度大于8
TEST_F(l2_mish_test, case_009_max_dim)
{
    auto selfDesc = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8, 9}, ACL_FLOAT, ACL_FORMAT_ND);

    auto outDesc = TensorDesc({1, 1, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnMish, INPUT(selfDesc), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
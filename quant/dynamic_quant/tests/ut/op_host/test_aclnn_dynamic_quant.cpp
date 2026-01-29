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
#include <vector>

#include "gtest/gtest.h"
#include <gmock/gmock.h>
#include "../../../op_host/op_api/aclnn_dynamic_quant.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

// IFA aclnn ut for 910b has error in UT environment. Deleted.
class l2_dynamic_quant_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "l2_dynamic_quant_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "l2_dynamic_quant_test TearDown" << endl;
    }
};

TEST_F(l2_dynamic_quant_test, ascend910B2_dynamic_quant_fp16)
{
    TensorDesc x_desc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc smooth_desc = TensorDesc({32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({16, 32}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnDynamicQuant, INPUT(x_desc, smooth_desc), OUTPUT(y_desc, scale_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}

TEST_F(l2_dynamic_quant_test, ascend910B2_dynamic_quant_bf16)
{
    TensorDesc x_desc = TensorDesc({16, 32}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc smooth_desc = TensorDesc({32}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({16, 32}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnDynamicQuant, INPUT(x_desc, smooth_desc), OUTPUT(y_desc, scale_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}

// TEST_F(l2_dynamic_quant_test, case_for_fault_injection_01)
// {
//     (void)setenv("FOR_FAULT_INJECTION", "1", 1);
//     TensorDesc x_desc = TensorDesc({16, 32}, ACL_BF16, ACL_FORMAT_ND);
//     TensorDesc smooth_desc = TensorDesc({32}, ACL_BF16, ACL_FORMAT_ND);
//     TensorDesc y_desc = TensorDesc({16, 32}, ACL_INT8, ACL_FORMAT_ND);
//     TensorDesc scale_desc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND);

//     auto ut = OP_API_UT(aclnnDynamicQuant, INPUT(x_desc, smooth_desc), OUTPUT(y_desc, scale_desc));
//     uint64_t workspace_size = 0;
//     aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
// }

TEST_F(l2_dynamic_quant_test, ascend950PR_9589_dynamic_quant_fp16_int8_testcase_01)
{
    TensorDesc x_desc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc smooth_desc = TensorDesc({32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({16, 32}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnDynamicQuant, INPUT(x_desc, smooth_desc), OUTPUT(y_desc, scale_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_dynamic_quant_test, ascend310P3_dynamic_quant_fp16_int8_testcase_01)
{
    TensorDesc x_desc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc smooth_desc = TensorDesc({32}, ACL_FLOAT16, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({16, 32}, ACL_INT8, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnDynamicQuant, INPUT(x_desc, smooth_desc), OUTPUT(y_desc, scale_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
}

// 910D平台测试 bf16进 int4出
TEST_F(l2_dynamic_quant_test, ascend950PR_9589_dynamic_quant_bf16_int4_testcase_fail_01)
{
    TensorDesc x_desc = TensorDesc({16, 31}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc smooth_desc = TensorDesc({31}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({16, 31}, ACL_INT4, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnDynamicQuant, INPUT(x_desc, smooth_desc), OUTPUT(y_desc, scale_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 910D平台测试 bf16进 int32出
TEST_F(l2_dynamic_quant_test, ascend950PR_9589_dynamic_quant_bf16_int32_testcase_03)
{
    TensorDesc x_desc = TensorDesc({16, 32}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc smooth_desc = TensorDesc({32}, ACL_BF16, ACL_FORMAT_ND);
    TensorDesc y_desc = TensorDesc({16, 4}, ACL_INT32, ACL_FORMAT_ND);
    TensorDesc scale_desc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnDynamicQuant, INPUT(x_desc, smooth_desc), OUTPUT(y_desc, scale_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}
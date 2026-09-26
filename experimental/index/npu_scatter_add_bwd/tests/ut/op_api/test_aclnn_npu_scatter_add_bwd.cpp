/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
#include "../../../op_api/aclnn_npu_scatter_add_bwd.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace op;
using namespace std;

class l2_npu_scatter_add_bwd_test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "l2_npu_scatter_add_bwd_test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "l2_npu_scatter_add_bwd_test TearDown" << std::endl; }
};

// y_grad为空指针
TEST_F(l2_npu_scatter_add_bwd_test, case_y_grad_nullptr)
{
    auto x_desc = TensorDesc({8, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto s_desc = TensorDesc({8}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto indices_desc = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 4);
    auto x_grad_desc = TensorDesc({8, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto s_grad_desc = TensorDesc({8}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnNpuScatterAddBwd, INPUT(nullptr, x_desc, s_desc, indices_desc, x_grad_desc, s_grad_desc),
                        OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// x为空指针
TEST_F(l2_npu_scatter_add_bwd_test, case_x_nullptr)
{
    auto y_grad_desc = TensorDesc({4, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto s_desc = TensorDesc({8}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto indices_desc = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 4);
    auto x_grad_desc = TensorDesc({8, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto s_grad_desc = TensorDesc({8}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnNpuScatterAddBwd,
                        INPUT(y_grad_desc, nullptr, s_desc, indices_desc, x_grad_desc, s_grad_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// bf16常规场景
TEST_F(l2_npu_scatter_add_bwd_test, case_bfloat16_success)
{
    auto y_grad_desc = TensorDesc({4, 16}, ACL_BF16, ACL_FORMAT_ND);
    auto x_desc = TensorDesc({8, 16}, ACL_BF16, ACL_FORMAT_ND);
    auto s_desc = TensorDesc({8}, ACL_BF16, ACL_FORMAT_ND);
    auto indices_desc = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 4);
    auto x_grad_desc = TensorDesc({8, 16}, ACL_BF16, ACL_FORMAT_ND);
    auto s_grad_desc = TensorDesc({8}, ACL_BF16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnNpuScatterAddBwd,
                        INPUT(y_grad_desc, x_desc, s_desc, indices_desc, x_grad_desc, s_grad_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// fp16常规场景
TEST_F(l2_npu_scatter_add_bwd_test, case_float16_success)
{
    auto y_grad_desc = TensorDesc({4, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto x_desc = TensorDesc({8, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto s_desc = TensorDesc({8}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto indices_desc = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 4);
    auto x_grad_desc = TensorDesc({8, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto s_grad_desc = TensorDesc({8}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnNpuScatterAddBwd,
                        INPUT(y_grad_desc, x_desc, s_desc, indices_desc, x_grad_desc, s_grad_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// x为fp32类型，不支持
TEST_F(l2_npu_scatter_add_bwd_test, case_x_float32_invalid)
{
    auto y_grad_desc = TensorDesc({4, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto x_desc = TensorDesc({8, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto s_desc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto indices_desc = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 4);
    auto x_grad_desc = TensorDesc({8, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto s_grad_desc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnNpuScatterAddBwd,
                        INPUT(y_grad_desc, x_desc, s_desc, indices_desc, x_grad_desc, s_grad_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// indices为int64类型，不支持
TEST_F(l2_npu_scatter_add_bwd_test, case_indices_int64_invalid)
{
    auto y_grad_desc = TensorDesc({4, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto x_desc = TensorDesc({8, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto s_desc = TensorDesc({8}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto indices_desc = TensorDesc({8}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 4);
    auto x_grad_desc = TensorDesc({8, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto s_grad_desc = TensorDesc({8}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnNpuScatterAddBwd,
                        INPUT(y_grad_desc, x_desc, s_desc, indices_desc, x_grad_desc, s_grad_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// x为3维张量，不支持
TEST_F(l2_npu_scatter_add_bwd_test, case_x_dim3_invalid)
{
    auto y_grad_desc = TensorDesc({4, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto x_desc = TensorDesc({8, 4, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto s_desc = TensorDesc({8}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto indices_desc = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 4);
    auto x_grad_desc = TensorDesc({8, 4, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto s_grad_desc = TensorDesc({8}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnNpuScatterAddBwd,
                        INPUT(y_grad_desc, x_desc, s_desc, indices_desc, x_grad_desc, s_grad_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// y_grad与x的dim[1]不一致
TEST_F(l2_npu_scatter_add_bwd_test, case_dim1_mismatch_invalid)
{
    auto y_grad_desc = TensorDesc({4, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto x_desc = TensorDesc({8, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto s_desc = TensorDesc({8}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto indices_desc = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 4);
    auto x_grad_desc = TensorDesc({8, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto s_grad_desc = TensorDesc({8}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnNpuScatterAddBwd,
                        INPUT(y_grad_desc, x_desc, s_desc, indices_desc, x_grad_desc, s_grad_desc), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

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
#include "../../../op_api/aclnn_npu_scatter_add.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace op;
using namespace std;

class l2_npu_scatter_add_test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "l2_npu_scatter_add_test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "l2_npu_scatter_add_test TearDown" << std::endl; }
};

// x为空指针
TEST_F(l2_npu_scatter_add_test, case_x_nullptr)
{
    auto y_desc = TensorDesc({4, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto s_desc = TensorDesc({8}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto indices_desc = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 4);
    auto sort_idx_desc = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 8);

    auto ut = OP_API_UT(aclnnNpuScatterAdd, INPUT(nullptr, y_desc, s_desc, indices_desc, sort_idx_desc, nullptr, false),
                        OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// y为空指针
TEST_F(l2_npu_scatter_add_test, case_y_nullptr)
{
    auto x_desc = TensorDesc({8, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto s_desc = TensorDesc({8}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto indices_desc = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 4);
    auto sort_idx_desc = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 8);

    auto ut = OP_API_UT(aclnnNpuScatterAdd, INPUT(x_desc, nullptr, s_desc, indices_desc, sort_idx_desc, nullptr, false),
                        OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// indices为空指针
TEST_F(l2_npu_scatter_add_test, case_indices_nullptr)
{
    auto x_desc = TensorDesc({8, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto y_desc = TensorDesc({4, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto s_desc = TensorDesc({8}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto sort_idx_desc = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 8);

    auto ut = OP_API_UT(aclnnNpuScatterAdd, INPUT(x_desc, y_desc, s_desc, nullptr, sort_idx_desc, nullptr, false),
                        OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// sort_idx为空指针
TEST_F(l2_npu_scatter_add_test, case_sort_idx_nullptr)
{
    auto x_desc = TensorDesc({8, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto y_desc = TensorDesc({4, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto s_desc = TensorDesc({8}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto indices_desc = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 4);

    auto ut = OP_API_UT(aclnnNpuScatterAdd, INPUT(x_desc, y_desc, s_desc, indices_desc, nullptr, nullptr, false),
                        OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// bf16常规场景（带缩放因子）
TEST_F(l2_npu_scatter_add_test, case_bfloat16_with_scale)
{
    auto x_desc = TensorDesc({8, 16}, ACL_BF16, ACL_FORMAT_ND);
    auto y_desc = TensorDesc({4, 16}, ACL_BF16, ACL_FORMAT_ND);
    auto s_desc = TensorDesc({8}, ACL_BF16, ACL_FORMAT_ND);
    auto indices_desc = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 4);
    auto sort_idx_desc = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 8);

    auto ut = OP_API_UT(aclnnNpuScatterAdd, INPUT(x_desc, y_desc, s_desc, indices_desc, sort_idx_desc, nullptr, false),
                        OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// fp16常规场景（无缩放因子、带valid_token_num、高精度模式）
TEST_F(l2_npu_scatter_add_test, case_float16_no_scale_with_valid)
{
    auto x_desc = TensorDesc({8, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto y_desc = TensorDesc({4, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto indices_desc = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 4);
    auto sort_idx_desc = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 8);
    auto valid_desc = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnNpuScatterAdd,
                        INPUT(x_desc, y_desc, nullptr, indices_desc, sort_idx_desc, valid_desc, true), OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// x为fp32类型，不支持
TEST_F(l2_npu_scatter_add_test, case_x_float32_invalid)
{
    auto x_desc = TensorDesc({8, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto y_desc = TensorDesc({4, 16}, ACL_FLOAT, ACL_FORMAT_ND);
    auto indices_desc = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 4);
    auto sort_idx_desc = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 8);

    auto ut = OP_API_UT(aclnnNpuScatterAdd, INPUT(x_desc, y_desc, nullptr, indices_desc, sort_idx_desc, nullptr, false),
                        OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// indices为int64类型，不支持
TEST_F(l2_npu_scatter_add_test, case_indices_int64_invalid)
{
    auto x_desc = TensorDesc({8, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto y_desc = TensorDesc({4, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto indices_desc = TensorDesc({8}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 4);
    auto sort_idx_desc = TensorDesc({8}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 8);

    auto ut = OP_API_UT(aclnnNpuScatterAdd, INPUT(x_desc, y_desc, nullptr, indices_desc, sort_idx_desc, nullptr, false),
                        OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// x为3维张量，不支持
TEST_F(l2_npu_scatter_add_test, case_x_dim3_invalid)
{
    auto x_desc = TensorDesc({8, 4, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto y_desc = TensorDesc({4, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto indices_desc = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 4);
    auto sort_idx_desc = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 8);

    auto ut = OP_API_UT(aclnnNpuScatterAdd, INPUT(x_desc, y_desc, nullptr, indices_desc, sort_idx_desc, nullptr, false),
                        OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// x与y的dim[1]不一致
TEST_F(l2_npu_scatter_add_test, case_dim1_mismatch_invalid)
{
    auto x_desc = TensorDesc({8, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto y_desc = TensorDesc({4, 32}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto indices_desc = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 4);
    auto sort_idx_desc = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 8);

    auto ut = OP_API_UT(aclnnNpuScatterAdd, INPUT(x_desc, y_desc, nullptr, indices_desc, sort_idx_desc, nullptr, false),
                        OUTPUT());

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

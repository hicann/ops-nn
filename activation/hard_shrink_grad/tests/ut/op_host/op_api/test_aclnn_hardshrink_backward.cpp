/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <vector>
#include <array>
#include "gtest/gtest.h"

#include "level2/aclnn_hardshrink_backward.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

class l2_hardshrink_backward_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "Hardshrink Backward Test Setup" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "Hardshrink Backward Test TearDown" << endl;
    }
};

TEST_F(l2_hardshrink_backward_test, case_normal)
{
    auto gradOutput_desc = TensorDesc({1, 1, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND)
                               .Value(vector<float>{1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2});
    auto self_desc = TensorDesc({1, 1, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND)
                         .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto lambd_desc = ScalarDesc(8.0f);
    auto gradInput_desc = TensorDesc(self_desc).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnHardshrinkBackward, INPUT(gradOutput_desc, self_desc, lambd_desc), OUTPUT(gradInput_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// check nullptr
TEST_F(l2_hardshrink_backward_test, case_nullptr)
{
    auto gradOutput_desc = TensorDesc({1, 1, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND)
                               .Value(vector<float>{1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2});
    auto self_desc = TensorDesc({1, 1, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND)
                         .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto lambd_desc = ScalarDesc(8.0f);
    auto gradInput_desc = TensorDesc(self_desc).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnHardshrinkBackward, INPUT(gradOutput_desc, self_desc, lambd_desc), OUTPUT(nullptr));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut2 = OP_API_UT(aclnnHardshrinkBackward, INPUT(gradOutput_desc, self_desc, nullptr), OUTPUT(gradInput_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize2 = 0;
    aclnnStatus aclRet2 = ut2.TestGetWorkspaceSize(&workspaceSize2);
    EXPECT_EQ(aclRet2, ACLNN_ERR_PARAM_NULLPTR);

    auto ut3 = OP_API_UT(aclnnHardshrinkBackward, INPUT(gradOutput_desc, nullptr, lambd_desc), OUTPUT(gradInput_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize3 = 0;
    aclnnStatus aclRet3 = ut.TestGetWorkspaceSize(&workspaceSize3);
    EXPECT_EQ(aclRet3, ACLNN_ERR_PARAM_NULLPTR);

    auto ut4 = OP_API_UT(aclnnHardshrinkBackward, INPUT(nullptr, self_desc, lambd_desc), OUTPUT(gradInput_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize4 = 0;
    aclnnStatus aclRet4 = ut4.TestGetWorkspaceSize(&workspaceSize4);
    EXPECT_EQ(aclRet4, ACLNN_ERR_PARAM_NULLPTR);
}

// check dtype
TEST_F(l2_hardshrink_backward_test, case_dtype_float16)
{
    auto gradOutput_desc = TensorDesc({1, 1, 4, 4}, ACL_FLOAT16, ACL_FORMAT_ND)
                               .Value(vector<float>{1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2});
    auto self_desc = TensorDesc({1, 1, 4, 4}, ACL_FLOAT16, ACL_FORMAT_ND)
                         .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto lambd_desc = ScalarDesc(8.0f);
    auto gradInput_desc = TensorDesc(self_desc).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnHardshrinkBackward, INPUT(gradOutput_desc, self_desc, lambd_desc), OUTPUT(gradInput_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_hardshrink_backward_test, case_dtype_double)
{
    auto gradOutput_desc = TensorDesc({1, 1, 4, 4}, ACL_DOUBLE, ACL_FORMAT_ND)
                               .Value(vector<float>{1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2});
    auto self_desc = TensorDesc({1, 1, 4, 4}, ACL_DOUBLE, ACL_FORMAT_ND)
                         .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto lambd_desc = ScalarDesc(8.0f);
    auto gradInput_desc = TensorDesc(self_desc).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnHardshrinkBackward, INPUT(gradOutput_desc, self_desc, lambd_desc), OUTPUT(gradInput_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// check promote dtype
TEST_F(l2_hardshrink_backward_test, case_promote_dtype)
{
    auto gradOutput_desc = TensorDesc({1, 1, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND)
                               .Value(vector<float>{1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2});
    auto self_desc = TensorDesc({1, 1, 4, 4}, ACL_FLOAT16, ACL_FORMAT_ND)
                         .Value(vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    auto lambd_desc = ScalarDesc(8.0f);
    auto gradInput_desc = TensorDesc(self_desc).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnHardshrinkBackward, INPUT(gradOutput_desc, self_desc, lambd_desc), OUTPUT(gradInput_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// check broadcast succ
TEST_F(l2_hardshrink_backward_test, case_broadcast_succ)
{
    auto gradOutput_desc = TensorDesc({1, 1, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto self_desc = TensorDesc({1, 1, 1, 4}, ACL_FLOAT16, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4});
    auto lambd_desc = ScalarDesc(8.0f);
    auto gradInput_desc = TensorDesc(gradOutput_desc).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnHardshrinkBackward, INPUT(gradOutput_desc, self_desc, lambd_desc), OUTPUT(gradInput_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// check broadcast fail
TEST_F(l2_hardshrink_backward_test, case_broadcast_fail)
{
    auto gradOutput_desc = TensorDesc({1, 1, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND)
                               .Value(vector<float>{1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2});
    auto self_desc = TensorDesc({1, 1, 1, 3}, ACL_FLOAT16, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3});
    auto lambd_desc = ScalarDesc(8.0f);
    auto gradInput_desc = TensorDesc(self_desc).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnHardshrinkBackward, INPUT(gradOutput_desc, self_desc, lambd_desc), OUTPUT(gradInput_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// check empty tensor
TEST_F(l2_hardshrink_backward_test, case_empty_tensor)
{
    auto gradOutput_desc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto self_desc = TensorDesc({1, 1, 1, 4}, ACL_FLOAT16, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4});
    auto lambd_desc = ScalarDesc(8.0f);
    auto gradInput_desc = TensorDesc(self_desc).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnHardshrinkBackward, INPUT(gradOutput_desc, self_desc, lambd_desc), OUTPUT(gradInput_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// not contiguous
TEST_F(l2_hardshrink_backward_test, case_not_contiguous)
{
    auto gradOutput_desc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2});
    auto self_desc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2});
    auto lambd_desc = ScalarDesc(8.0f);
    auto gradInput_desc = TensorDesc(self_desc).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnHardshrinkBackward, INPUT(gradOutput_desc, self_desc, lambd_desc), OUTPUT(gradInput_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}
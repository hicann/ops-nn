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

#include "../../../op_host/op_api/aclnn_leaky_relu_backward.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

class l2_leaky_relu_backward_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "leaky_relu_backward_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "leaky_relu_backward_test TearDown" << endl;
    }
};

TEST_F(l2_leaky_relu_backward_test, case_null_tensor)
{
    auto tensor_desc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scalar_desc = ScalarDesc(0.01f);

    auto ut =
        OP_API_UT(aclnnLeakyReluBackward, INPUT(tensor_desc, tensor_desc, scalar_desc, true), OUTPUT(tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // ut.TestPrecision();
}

TEST_F(l2_leaky_relu_backward_test, case_nullptr)
{
    auto tensor_desc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scalar_desc = ScalarDesc(0.01f);

    auto ut = OP_API_UT(aclnnLeakyReluBackward, INPUT(nullptr, tensor_desc, scalar_desc, true), OUTPUT(tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut1 = OP_API_UT(aclnnLeakyReluBackward, INPUT(tensor_desc, nullptr, scalar_desc, true), OUTPUT(tensor_desc));
    aclRet = ut1.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut2 = OP_API_UT(aclnnLeakyReluBackward, INPUT(tensor_desc, tensor_desc, nullptr, true), OUTPUT(tensor_desc));
    aclRet = ut2.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut3 = OP_API_UT(aclnnLeakyReluBackward, INPUT(tensor_desc, tensor_desc, scalar_desc, true), OUTPUT(nullptr));
    aclRet = ut3.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_leaky_relu_backward_test, case_dim1_Float_ND)
{
    auto tensor_desc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scalar_desc = ScalarDesc(0.01f);

    auto ut =
        OP_API_UT(aclnnLeakyReluBackward, INPUT(tensor_desc, tensor_desc, scalar_desc, true), OUTPUT(tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // ut.TestPrecision();
}

TEST_F(l2_leaky_relu_backward_test, case_dim2_Float16_NCHW)
{
    auto tensor_desc = TensorDesc({2, 2}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto scalar_desc = ScalarDesc(0.01f, ACL_FLOAT16);

    auto ut =
        OP_API_UT(aclnnLeakyReluBackward, INPUT(tensor_desc, tensor_desc, scalar_desc, true), OUTPUT(tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // ut.TestPrecision();
}

TEST_F(l2_leaky_relu_backward_test, case_dim3_FLoat16_NHWC)
{
    auto tensor_desc = TensorDesc({2, 2, 2}, ACL_FLOAT16, ACL_FORMAT_NHWC);
    auto scalar_desc = ScalarDesc(0.01);

    auto ut =
        OP_API_UT(aclnnLeakyReluBackward, INPUT(tensor_desc, tensor_desc, scalar_desc, true), OUTPUT(tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // ut.TestPrecision();
}

TEST_F(l2_leaky_relu_backward_test, case_dim3_Double_NHWC)
{
    auto tensor_desc = TensorDesc({2, 2, 2, 2}, ACL_DOUBLE, ACL_FORMAT_NHWC);
    auto scalar_desc = ScalarDesc(0.01);

    auto ut =
        OP_API_UT(aclnnLeakyReluBackward, INPUT(tensor_desc, tensor_desc, scalar_desc, true), OUTPUT(tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // ut.TestPrecision();
}

TEST_F(l2_leaky_relu_backward_test, case_error_dtype)
{
    auto tensor_desc = TensorDesc({2, 2, 2}, ACL_UINT32, ACL_FORMAT_ND);
    auto scalar_desc = ScalarDesc(1);

    auto ut =
        OP_API_UT(aclnnLeakyReluBackward, INPUT(tensor_desc, tensor_desc, scalar_desc, true), OUTPUT(tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_leaky_relu_backward_test, case_different_dtype)
{
    auto grad = TensorDesc({2, 2, 2, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto self = TensorDesc({2, 2, 2, 2}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto scalar_desc = ScalarDesc(0.01f);
    auto out = TensorDesc({2, 2, 2, 2}, ACL_DOUBLE, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnLeakyReluBackward, INPUT(grad, self, scalar_desc, true), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // ut.TestPrecision();
}

TEST_F(l2_leaky_relu_backward_test, case_error_shape)
{
    auto tensor_desc = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scalar_desc = ScalarDesc(0.01f);

    auto ut =
        OP_API_UT(aclnnLeakyReluBackward, INPUT(tensor_desc, tensor_desc, scalar_desc, true), OUTPUT(tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_leaky_relu_backward_test, case_different_shape)
{
    auto grad = TensorDesc({2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto self = TensorDesc({2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scalar_desc = ScalarDesc(0.01f);
    auto out = TensorDesc({2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnLeakyReluBackward, INPUT(grad, self, scalar_desc, true), OUTPUT(out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_leaky_relu_backward_test, case_dim4_Float16_HWCN)
{
    auto tensor_desc = TensorDesc({2, 2, 2, 2}, ACL_FLOAT16, ACL_FORMAT_HWCN);
    auto scalar_desc = ScalarDesc(0.01);

    auto ut =
        OP_API_UT(aclnnLeakyReluBackward, INPUT(tensor_desc, tensor_desc, scalar_desc, true), OUTPUT(tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // ut.TestPrecision();
}

TEST_F(l2_leaky_relu_backward_test, case_dim5_Float_NDHWC)
{
    auto tensor_desc = TensorDesc({2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_NDHWC);
    auto scalar_desc = ScalarDesc(0.01f);

    auto ut =
        OP_API_UT(aclnnLeakyReluBackward, INPUT(tensor_desc, tensor_desc, scalar_desc, true), OUTPUT(tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // ut.TestPrecision();
}

TEST_F(l2_leaky_relu_backward_test, case_dim5_Float_NCDHW)
{
    auto tensor_desc = TensorDesc({2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_NCDHW);
    auto scalar_desc = ScalarDesc(0.01f);

    auto ut =
        OP_API_UT(aclnnLeakyReluBackward, INPUT(tensor_desc, tensor_desc, scalar_desc, true), OUTPUT(tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // ut.TestPrecision();
}

TEST_F(l2_leaky_relu_backward_test, case_dim5_Float16_NC1HWC0)
{
    auto tensor_desc = TensorDesc({2, 2, 2, 2, 2}, ACL_FLOAT16, ACL_FORMAT_NC1HWC0);
    auto scalar_desc = ScalarDesc(0.01f, ACL_FLOAT16);

    auto ut =
        OP_API_UT(aclnnLeakyReluBackward, INPUT(tensor_desc, tensor_desc, scalar_desc, true), OUTPUT(tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // ut.TestPrecision();
}

TEST_F(l2_leaky_relu_backward_test, case_dim8_Float16_ND)
{
    auto tensor_desc = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto scalar_desc = ScalarDesc(0.01f, ACL_FLOAT16);

    auto ut =
        OP_API_UT(aclnnLeakyReluBackward, INPUT(tensor_desc, tensor_desc, scalar_desc, true), OUTPUT(tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // ut.TestPrecision();
}

TEST_F(l2_leaky_relu_backward_test, case_dim6_Float_ND)
{
    auto tensor_desc = TensorDesc({2, 2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scalar_desc = ScalarDesc(0.1f);

    auto ut =
        OP_API_UT(aclnnLeakyReluBackward, INPUT(tensor_desc, tensor_desc, scalar_desc, true), OUTPUT(tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // ut.TestPrecision();
}

TEST_F(l2_leaky_relu_backward_test, case_dim7_range)
{
    auto tensor_desc = TensorDesc({2, 2, 2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto scalar_desc = ScalarDesc(0.5f);

    auto ut =
        OP_API_UT(aclnnLeakyReluBackward, INPUT(tensor_desc, tensor_desc, scalar_desc, false), OUTPUT(tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // ut.TestPrecision();
}

TEST_F(l2_leaky_relu_backward_test, case_not_contiguous)
{
    auto tensor_desc = TensorDesc({3, 5, 6}, ACL_FLOAT, ACL_FORMAT_ND, {30, 1, 5}, 0, {3, 6, 5});
    auto scalar_desc = ScalarDesc(0.1f);

    auto ut =
        OP_API_UT(aclnnLeakyReluBackward, INPUT(tensor_desc, tensor_desc, scalar_desc, true), OUTPUT(tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // ut.TestPrecision();
}

TEST_F(l2_leaky_relu_backward_test, case_error_inplace)
{
    auto tensor_desc = TensorDesc({2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scalar_desc = ScalarDesc(-0.1f);

    auto ut =
        OP_API_UT(aclnnLeakyReluBackward, INPUT(tensor_desc, tensor_desc, scalar_desc, true), OUTPUT(tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_leaky_relu_backward_test, ascend910B2_case_dim3_FLoat16_NHWC)
{
    auto tensor_desc = TensorDesc({2, 2, 2}, ACL_BF16, ACL_FORMAT_NHWC);
    auto scalar_desc = ScalarDesc(0.01);

    auto ut =
        OP_API_UT(aclnnLeakyReluBackward, INPUT(tensor_desc, tensor_desc, scalar_desc, true), OUTPUT(tensor_desc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // ut.TestPrecision();
}
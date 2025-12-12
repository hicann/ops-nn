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

#include "../../../op_host/op_api/aclnn_geglu.h"

#include "op_api_ut_common/inner/types.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

#include "opdev/platform.h"

using namespace op;
using namespace std;

class l2_geglu_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "l2_geglu_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "l2_geglu_test TearDown" << endl;
    }
};

TEST_F(l2_geglu_test, geglu_self_nullptr)
{
    int64_t dim = -1;
    int64_t approximate = 1;
    auto ut = OP_API_UT(
        aclnnGeGlu, INPUT((aclTensor*)nullptr, dim, approximate), OUTPUT((aclTensor*)nullptr, (aclTensor*)nullptr));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_geglu_test, geglu_out_nullptr)
{
    int64_t dim = -1;
    int64_t approximate = 1;
    auto self_tensor_desc = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-20, 20);
    auto ut = OP_API_UT(
        aclnnGeGlu, INPUT(self_tensor_desc, dim, approximate), OUTPUT((aclTensor*)nullptr, (aclTensor*)nullptr));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_geglu_test, geglu_out_gelu_nullptr)
{
    int64_t dim = -1;
    int64_t approximate = 1;
    auto self_tensor_desc = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-20, 20);
    auto out_tensor_desc = TensorDesc({4, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnGeGlu, INPUT(self_tensor_desc, dim, approximate), OUTPUT(out_tensor_desc, (aclTensor*)nullptr));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_geglu_test, geglu_self_wrong_dtype)
{
    auto self_tensor_desc = TensorDesc({4, 4}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-20, 20);
    auto out_tensor_desc = TensorDesc({4, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_gelu_tensor_desc = TensorDesc({4, 2}, ACL_FLOAT, ACL_FORMAT_ND);

    int64_t dim = -1;
    int64_t approximate = 1;
    auto ut =
        OP_API_UT(aclnnGeGlu, INPUT(self_tensor_desc, dim, approximate), OUTPUT(out_tensor_desc, out_gelu_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_geglu_test, geglu_out_wrong_dtype)
{
    auto self_tensor_desc = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-20, 20);
    auto out_tensor_desc = TensorDesc({4, 2}, ACL_INT32, ACL_FORMAT_ND);
    auto out_gelu_tensor_desc = TensorDesc({4, 2}, ACL_FLOAT, ACL_FORMAT_ND);

    int64_t dim = -1;
    int64_t approximate = 1;
    auto ut =
        OP_API_UT(aclnnGeGlu, INPUT(self_tensor_desc, dim, approximate), OUTPUT(out_tensor_desc, out_gelu_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_geglu_test, geglu_out_gelu_wrong_dtype)
{
    auto self_tensor_desc = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-20, 20);
    auto out_tensor_desc = TensorDesc({4, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_gelu_tensor_desc = TensorDesc({4, 2}, ACL_INT32, ACL_FORMAT_ND);

    int64_t dim = -1;
    int64_t approximate = 1;
    auto ut =
        OP_API_UT(aclnnGeGlu, INPUT(self_tensor_desc, dim, approximate), OUTPUT(out_tensor_desc, out_gelu_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_geglu_test, geglu_out_different_dtype)
{
    auto self_tensor_desc = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-20, 20);
    auto out_tensor_desc = TensorDesc({4, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out_gelu_tensor_desc = TensorDesc({4, 2}, ACL_FLOAT, ACL_FORMAT_ND);

    int64_t dim = -1;
    int64_t approximate = 1;
    auto ut =
        OP_API_UT(aclnnGeGlu, INPUT(self_tensor_desc, dim, approximate), OUTPUT(out_tensor_desc, out_gelu_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_geglu_test, geglu_out_gelu_different_dtype)
{
    auto self_tensor_desc = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-20, 20);
    auto out_tensor_desc = TensorDesc({4, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_gelu_tensor_desc = TensorDesc({4, 2}, ACL_FLOAT16, ACL_FORMAT_ND);

    int64_t dim = -1;
    int64_t approximate = 1;
    auto ut =
        OP_API_UT(aclnnGeGlu, INPUT(self_tensor_desc, dim, approximate), OUTPUT(out_tensor_desc, out_gelu_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_geglu_test, geglu_self_bigDim)
{
    auto self_tensor_desc = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-20, 20);
    auto out_tensor_desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_gelu_tensor_desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);

    int64_t dim = -1;
    int64_t approximate = 1;
    auto ut =
        OP_API_UT(aclnnGeGlu, INPUT(self_tensor_desc, dim, approximate), OUTPUT(out_tensor_desc, out_gelu_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_geglu_test, geglu_out_bigDim)
{
    auto self_tensor_desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-20, 20);
    auto out_tensor_desc = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_gelu_tensor_desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);

    int64_t dim = -1;
    int64_t approximate = 1;
    auto ut =
        OP_API_UT(aclnnGeGlu, INPUT(self_tensor_desc, dim, approximate), OUTPUT(out_tensor_desc, out_gelu_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_geglu_test, geglu_out_gelu_bigDim)
{
    auto self_tensor_desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-20, 20);
    auto out_tensor_desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_gelu_tensor_desc = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);

    int64_t dim = -1;
    int64_t approximate = 1;
    auto ut =
        OP_API_UT(aclnnGeGlu, INPUT(self_tensor_desc, dim, approximate), OUTPUT(out_tensor_desc, out_gelu_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_geglu_test, geglu_wrong_dim)
{
    auto self_tensor_desc = TensorDesc({4, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-20, 20);
    auto out_tensor_desc = TensorDesc({4, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out_gelu_tensor_desc = TensorDesc({4, 2}, ACL_FLOAT16, ACL_FORMAT_ND);

    int64_t dim = 10;
    int64_t approximate = 1;
    auto ut =
        OP_API_UT(aclnnGeGlu, INPUT(self_tensor_desc, dim, approximate), OUTPUT(out_tensor_desc, out_gelu_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_geglu_test, geglu_wrong_out_shape)
{
    auto self_tensor_desc = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-20, 20);
    auto out_tensor_desc = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_gelu_tensor_desc = TensorDesc({4, 2}, ACL_FLOAT, ACL_FORMAT_ND);

    int64_t dim = -1;
    int64_t approximate = 1;
    auto ut =
        OP_API_UT(aclnnGeGlu, INPUT(self_tensor_desc, dim, approximate), OUTPUT(out_tensor_desc, out_gelu_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_geglu_test, geglu_wrong_out_gelu_shape)
{
    auto self_tensor_desc = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-20, 20);
    auto out_tensor_desc = TensorDesc({4, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_gelu_tensor_desc = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    int64_t dim = -1;
    int64_t approximate = 1;
    auto ut =
        OP_API_UT(aclnnGeGlu, INPUT(self_tensor_desc, dim, approximate), OUTPUT(out_tensor_desc, out_gelu_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_geglu_test, geglu_wrong_slice_dim)
{
    auto self_tensor_desc = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-20, 20);
    auto out_tensor_desc = TensorDesc({4, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_gelu_tensor_desc = TensorDesc({4, 2}, ACL_FLOAT, ACL_FORMAT_ND);

    int64_t dim = -1;
    int64_t approximate = 1;
    auto ut =
        OP_API_UT(aclnnGeGlu, INPUT(self_tensor_desc, dim, approximate), OUTPUT(out_tensor_desc, out_gelu_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_geglu_test, geglu_wrong_approximate)
{
    auto self_tensor_desc = TensorDesc({4, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-20, 20);
    auto out_tensor_desc = TensorDesc({4, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out_gelu_tensor_desc = TensorDesc({4, 2}, ACL_FLOAT16, ACL_FORMAT_ND);

    int64_t dim = -1;
    int64_t approximate = 10;
    auto ut =
        OP_API_UT(aclnnGeGlu, INPUT(self_tensor_desc, dim, approximate), OUTPUT(out_tensor_desc, out_gelu_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_geglu_test, geglu_precision)
{
    auto self_tensor_desc = TensorDesc({4, 4, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-20, 20);
    auto out_tensor_desc = TensorDesc({4, 4, 2}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto out_gelu_tensor_desc = TensorDesc({4, 4, 2}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);

    int64_t dim = -1;
    int64_t approximate = 1;
    auto ut =
        OP_API_UT(aclnnGeGlu, INPUT(self_tensor_desc, dim, approximate), OUTPUT(out_tensor_desc, out_gelu_tensor_desc));
    // uint64_t workspace_size = 0;
    // aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    // EXPECT_EQ(aclRet, ACL_SUCCESS);
    // ut.TestPrecision();
}

TEST_F(l2_geglu_test, geglu_empty_tensor_1)
{
    auto self_tensor_desc = TensorDesc({0}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-20, 20);
    auto out_tensor_desc = TensorDesc({0}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out_gelu_tensor_desc = TensorDesc({0}, ACL_FLOAT16, ACL_FORMAT_ND);

    int64_t dim = -1;
    int64_t approximate = 1;
    auto ut =
        OP_API_UT(aclnnGeGlu, INPUT(self_tensor_desc, dim, approximate), OUTPUT(out_tensor_desc, out_gelu_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    ut.TestPrecision();
}

TEST_F(l2_geglu_test, geglu_empty_tensor_2)
{
    auto self_tensor_desc = TensorDesc({4, 0, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-20, 20);
    auto out_tensor_desc = TensorDesc({4, 0, 2}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto out_gelu_tensor_desc = TensorDesc({4, 0, 2}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);

    int64_t dim = -1;
    int64_t approximate = 1;
    auto ut =
        OP_API_UT(aclnnGeGlu, INPUT(self_tensor_desc, dim, approximate), OUTPUT(out_tensor_desc, out_gelu_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    ut.TestPrecision();
}

TEST_F(l2_geglu_test, ascend910B2_geglu_non_contiguous)
{
    auto self_tensor_desc = TensorDesc({2, 8}, ACL_FLOAT16, ACL_FORMAT_ND, {1, 2}, 0, {8, 2}).ValueRange(-20, 20);
    auto out_tensor_desc = TensorDesc({2, 4}, ACL_FLOAT16, ACL_FORMAT_ND, {1, 2}, 0, {4, 2}).Precision(0.001, 0.001);
    auto out_gelu_tensor_desc =
        TensorDesc({2, 4}, ACL_FLOAT16, ACL_FORMAT_ND, {1, 2}, 0, {4, 2}).Precision(0.001, 0.001);

    int64_t dim = -1;
    int64_t approximate = 1;
    auto ut =
        OP_API_UT(aclnnGeGlu, INPUT(self_tensor_desc, dim, approximate), OUTPUT(out_tensor_desc, out_gelu_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    // ut.TestPrecision();
}

// new
TEST_F(l2_geglu_test, gegluv_self_nullptr)
{
    int64_t dim = -1;
    int64_t approximate = 1;
    bool activateLefts[2] = {true, false};
    for (auto activateLeft : activateLefts) {
        auto ut = OP_API_UT(
            aclnnGeGluV3, INPUT((aclTensor*)nullptr, dim, approximate, activateLeft),
            OUTPUT((aclTensor*)nullptr, (aclTensor*)nullptr));
        uint64_t workspace_size = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
    }
}

TEST_F(l2_geglu_test, gegluv_out_nullptr)
{
    int64_t dim = -1;
    int64_t approximate = 1;
    auto self_tensor_desc = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-20, 20);
    bool activateLefts[2] = {true, false};
    for (auto activateLeft : activateLefts) {
        auto ut = OP_API_UT(
            aclnnGeGluV3, INPUT(self_tensor_desc, dim, approximate, activateLeft),
            OUTPUT((aclTensor*)nullptr, (aclTensor*)nullptr));
        uint64_t workspaceSize = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
        EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
    }
}

TEST_F(l2_geglu_test, gegluv_out_gelu_nullptr)
{
    int64_t dim = -1;
    int64_t approximate = 1;
    auto self_tensor_desc = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-20, 20);
    auto out_tensor_desc = TensorDesc({4, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    bool activateLefts[2] = {true, false};
    for (auto activateLeft : activateLefts) {
        auto ut = OP_API_UT(
            aclnnGeGluV3, INPUT(self_tensor_desc, dim, approximate, activateLeft),
            OUTPUT(out_tensor_desc, (aclTensor*)nullptr));
        uint64_t workspaceSize = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
        EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
    }
}

TEST_F(l2_geglu_test, gegluv_self_wrong_dtype)
{
    auto self_tensor_desc = TensorDesc({4, 4}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-20, 20);
    auto out_tensor_desc = TensorDesc({4, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_gelu_tensor_desc = TensorDesc({4, 2}, ACL_FLOAT, ACL_FORMAT_ND);

    int64_t dim = -1;
    int64_t approximate = 1;
    bool activateLefts[2] = {true, false};
    for (auto activateLeft : activateLefts) {
        auto ut = OP_API_UT(
            aclnnGeGluV3, INPUT(self_tensor_desc, dim, approximate, activateLeft),
            OUTPUT(out_tensor_desc, out_gelu_tensor_desc));
        uint64_t workspace_size = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    }
}

TEST_F(l2_geglu_test, gegluv_out_wrong_dtype)
{
    auto self_tensor_desc = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-20, 20);
    auto out_tensor_desc = TensorDesc({4, 2}, ACL_INT32, ACL_FORMAT_ND);
    auto out_gelu_tensor_desc = TensorDesc({4, 2}, ACL_FLOAT, ACL_FORMAT_ND);

    int64_t dim = -1;
    int64_t approximate = 1;
    bool activateLefts[2] = {true, false};
    for (auto activateLeft : activateLefts) {
        auto ut = OP_API_UT(
            aclnnGeGluV3, INPUT(self_tensor_desc, dim, approximate, activateLeft),
            OUTPUT(out_tensor_desc, out_gelu_tensor_desc));
        uint64_t workspace_size = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    }
}

TEST_F(l2_geglu_test, gegluv_out_gelu_wrong_dtype)
{
    auto self_tensor_desc = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-20, 20);
    auto out_tensor_desc = TensorDesc({4, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_gelu_tensor_desc = TensorDesc({4, 2}, ACL_INT32, ACL_FORMAT_ND);

    int64_t dim = -1;
    int64_t approximate = 1;
    bool activateLefts[2] = {true, false};
    for (auto activateLeft : activateLefts) {
        auto ut = OP_API_UT(
            aclnnGeGluV3, INPUT(self_tensor_desc, dim, approximate, activateLeft),
            OUTPUT(out_tensor_desc, out_gelu_tensor_desc));
        uint64_t workspace_size = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    }
}

TEST_F(l2_geglu_test, gegluv_out_different_dtype)
{
    auto self_tensor_desc = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-20, 20);
    auto out_tensor_desc = TensorDesc({4, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out_gelu_tensor_desc = TensorDesc({4, 2}, ACL_FLOAT, ACL_FORMAT_ND);

    int64_t dim = -1;
    int64_t approximate = 1;
    bool activateLefts[2] = {true, false};
    for (auto activateLeft : activateLefts) {
        auto ut = OP_API_UT(
            aclnnGeGluV3, INPUT(self_tensor_desc, dim, approximate, activateLeft),
            OUTPUT(out_tensor_desc, out_gelu_tensor_desc));
        uint64_t workspace_size = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    }
}

TEST_F(l2_geglu_test, gegluv_out_gelu_different_dtype)
{
    auto self_tensor_desc = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-20, 20);
    auto out_tensor_desc = TensorDesc({4, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_gelu_tensor_desc = TensorDesc({4, 2}, ACL_FLOAT16, ACL_FORMAT_ND);

    int64_t dim = -1;
    int64_t approximate = 1;
    bool activateLefts[2] = {true, false};
    for (auto activateLeft : activateLefts) {
        auto ut = OP_API_UT(
            aclnnGeGluV3, INPUT(self_tensor_desc, dim, approximate, activateLeft),
            OUTPUT(out_tensor_desc, out_gelu_tensor_desc));
        uint64_t workspace_size = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    }
}

TEST_F(l2_geglu_test, gegluv_self_bigDim)
{
    auto self_tensor_desc = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-20, 20);
    auto out_tensor_desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_gelu_tensor_desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);

    int64_t dim = -1;
    int64_t approximate = 1;
    bool activateLefts[2] = {true, false};
    for (auto activateLeft : activateLefts) {
        auto ut = OP_API_UT(
            aclnnGeGluV3, INPUT(self_tensor_desc, dim, approximate, activateLeft),
            OUTPUT(out_tensor_desc, out_gelu_tensor_desc));
        uint64_t workspace_size = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    }
}

TEST_F(l2_geglu_test, gegluv_out_bigDim)
{
    auto self_tensor_desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-20, 20);
    auto out_tensor_desc = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_gelu_tensor_desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);

    int64_t dim = -1;
    int64_t approximate = 1;
    bool activateLefts[2] = {true, false};
    for (auto activateLeft : activateLefts) {
        auto ut = OP_API_UT(
            aclnnGeGluV3, INPUT(self_tensor_desc, dim, approximate, activateLeft),
            OUTPUT(out_tensor_desc, out_gelu_tensor_desc));
        uint64_t workspace_size = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    }
}

TEST_F(l2_geglu_test, gegluv_out_gelu_bigDim)
{
    auto self_tensor_desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-20, 20);
    auto out_tensor_desc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_gelu_tensor_desc = TensorDesc({2, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);

    int64_t dim = -1;
    int64_t approximate = 1;
    bool activateLefts[2] = {true, false};
    for (auto activateLeft : activateLefts) {
        auto ut = OP_API_UT(
            aclnnGeGluV3, INPUT(self_tensor_desc, dim, approximate, activateLeft),
            OUTPUT(out_tensor_desc, out_gelu_tensor_desc));
        uint64_t workspace_size = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    }
}

TEST_F(l2_geglu_test, gegluv_wrong_dim)
{
    auto self_tensor_desc = TensorDesc({4, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-20, 20);
    auto out_tensor_desc = TensorDesc({4, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out_gelu_tensor_desc = TensorDesc({4, 2}, ACL_FLOAT16, ACL_FORMAT_ND);

    int64_t dim = 10;
    int64_t approximate = 1;
    bool activateLefts[2] = {true, false};
    for (auto activateLeft : activateLefts) {
        auto ut = OP_API_UT(
            aclnnGeGluV3, INPUT(self_tensor_desc, dim, approximate, activateLeft),
            OUTPUT(out_tensor_desc, out_gelu_tensor_desc));
        uint64_t workspace_size = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    }
}

TEST_F(l2_geglu_test, gegluv_wrong_out_shape)
{
    auto self_tensor_desc = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-20, 20);
    auto out_tensor_desc = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_gelu_tensor_desc = TensorDesc({4, 2}, ACL_FLOAT, ACL_FORMAT_ND);

    int64_t dim = -1;
    int64_t approximate = 1;
    bool activateLefts[2] = {true, false};
    for (auto activateLeft : activateLefts) {
        auto ut = OP_API_UT(
            aclnnGeGluV3, INPUT(self_tensor_desc, dim, approximate, activateLeft),
            OUTPUT(out_tensor_desc, out_gelu_tensor_desc));
        uint64_t workspace_size = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    }
}

TEST_F(l2_geglu_test, gegluv_wrong_out_gelu_shape)
{
    auto self_tensor_desc = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-20, 20);
    auto out_tensor_desc = TensorDesc({4, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_gelu_tensor_desc = TensorDesc({4, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    int64_t dim = -1;
    int64_t approximate = 1;
    bool activateLefts[2] = {true, false};
    for (auto activateLeft : activateLefts) {
        auto ut = OP_API_UT(
            aclnnGeGluV3, INPUT(self_tensor_desc, dim, approximate, activateLeft),
            OUTPUT(out_tensor_desc, out_gelu_tensor_desc));
        uint64_t workspace_size = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    }
}

TEST_F(l2_geglu_test, gegluv_wrong_approximate)
{
    auto self_tensor_desc = TensorDesc({4, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-20, 20);
    auto out_tensor_desc = TensorDesc({4, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out_gelu_tensor_desc = TensorDesc({4, 2}, ACL_FLOAT16, ACL_FORMAT_ND);

    int64_t dim = -1;
    int64_t approximate = 10;
    bool activateLefts[2] = {true, false};
    for (auto activateLeft : activateLefts) {
        auto ut = OP_API_UT(
            aclnnGeGluV3, INPUT(self_tensor_desc, dim, approximate, activateLeft),
            OUTPUT(out_tensor_desc, out_gelu_tensor_desc));
        uint64_t workspace_size = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    }
}

TEST_F(l2_geglu_test, gegluv_precision)
{
    auto self_tensor_desc = TensorDesc({4, 4, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-20, 20);
    auto out_tensor_desc = TensorDesc({4, 4, 2}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto out_gelu_tensor_desc = TensorDesc({4, 4, 2}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);

    int64_t dim = -1;
    int64_t approximate = 1;
    bool activateLefts[2] = {true, false};
    for (auto activateLeft : activateLefts) {
        auto ut = OP_API_UT(
            aclnnGeGluV3, INPUT(self_tensor_desc, dim, approximate, activateLeft),
            OUTPUT(out_tensor_desc, out_gelu_tensor_desc));
        // uint64_t workspace_size = 0;
        // aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        // EXPECT_EQ(aclRet, ACL_SUCCESS);
        // ut.TestPrecision();
    }
}

TEST_F(l2_geglu_test, gegluv_empty_tensor_1)
{
    auto self_tensor_desc = TensorDesc({0}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-20, 20);
    auto out_tensor_desc = TensorDesc({0}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto out_gelu_tensor_desc = TensorDesc({0}, ACL_FLOAT16, ACL_FORMAT_ND);

    int64_t dim = -1;
    int64_t approximate = 1;
    bool activateLefts[2] = {true, false};
    for (auto activateLeft : activateLefts) {
        auto ut = OP_API_UT(
            aclnnGeGluV3, INPUT(self_tensor_desc, dim, approximate, activateLeft),
            OUTPUT(out_tensor_desc, out_gelu_tensor_desc));
        uint64_t workspace_size = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        EXPECT_EQ(aclRet, ACL_SUCCESS);
        ut.TestPrecision();
    }
}

TEST_F(l2_geglu_test, gegluv_empty_tensor_2)
{
    auto self_tensor_desc = TensorDesc({4, 0, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-20, 20);
    auto out_tensor_desc = TensorDesc({4, 0, 2}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto out_gelu_tensor_desc = TensorDesc({4, 0, 2}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);

    int64_t dim = -1;
    int64_t approximate = 1;

    bool activateLefts[2] = {true, false};
    for (auto activateLeft : activateLefts) {
        auto ut = OP_API_UT(
            aclnnGeGluV3, INPUT(self_tensor_desc, dim, approximate, activateLeft),
            OUTPUT(out_tensor_desc, out_gelu_tensor_desc));
        uint64_t workspace_size = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        EXPECT_EQ(aclRet, ACL_SUCCESS);
        ut.TestPrecision();
    }
}

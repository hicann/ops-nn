/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

#include "gtest/gtest.h"
#include "../../../op_host/op_api/aclnn_erfinv.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"

class l2ErfinvTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "l2ErfinvTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "l2ErfinvTest TearDown" << std::endl;
    }
};

// self的数据类型不在支持范围内
TEST_F(l2ErfinvTest, 001_dtype_uint16)
{
    auto selfDesc = TensorDesc({2, 3}, ACL_UINT16, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnErfinv, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// dtype undefined
TEST_F(l2ErfinvTest, 002_dtype_undefined)
{
    auto selfDesc = TensorDesc({2, 4, 3, 3}, ACL_DT_UNDEFINED, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 4, 3, 3}, ACL_DT_UNDEFINED, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnErfinv, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// dtype bfloat16
TEST_F(l2ErfinvTest, 003_dtype_bfloat16)
{
    auto selfDesc = TensorDesc({2, 4, 3, 3}, ACL_BF16, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 4, 3, 3}, ACL_BF16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnErfinv, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    //  EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// dtype int64
TEST_F(l2ErfinvTest, 004_dtype_int64)
{
    auto selfDesc = TensorDesc({2, 4, 3, 3}, ACL_INT64, ACL_FORMAT_NDHWC);
    auto outDesc = TensorDesc({2, 4, 3, 3}, ACL_FLOAT, ACL_FORMAT_NDHWC);

    auto ut = OP_API_UT(aclnnErfinv, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    ut.TestPrecision();
}

// dtype bool
TEST_F(l2ErfinvTest, 005_dtype_bool)
{
    auto selfDesc = TensorDesc({2, 4}, ACL_BOOL, ACL_FORMAT_ND).Value(vector<int>{1, 0, 3, 0, 5, 6, 7, 0});
    auto outDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnErfinv, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    ut.TestPrecision();
}

// 正常路径，float32
TEST_F(l2ErfinvTest, 006_dtype_float)
{
    auto selfDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnErfinv, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    ut.TestPrecision();
}

// 正常路径，float
TEST_F(l2ErfinvTest, 007_dtype_float16)
{
    auto selfDesc = TensorDesc({2, 4, 3, 3}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 4, 3, 3}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnErfinv, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    ut.TestPrecision();
}

// self和out的shape不一致
TEST_F(l2ErfinvTest, 008_self_out_different_shape)
{
    auto selfDesc = TensorDesc({2, 3, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnErfinv, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 空tensor
TEST_F(l2ErfinvTest, 009_self_empty)
{
    auto selfDesc = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({2, 0}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnErfinv, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    ut.TestPrecision();
}

// self空指针
TEST_F(l2ErfinvTest, 010_self_nullptr)
{
    auto outDesc = TensorDesc({2, 4}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnErfinv, INPUT(nullptr), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// out空指针
TEST_F(l2ErfinvTest, 011_out_nullptr)
{
    auto selfDesc = TensorDesc({2, 4}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnErfinv, INPUT(selfDesc), OUTPUT(nullptr));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// format NCHW、NHWC
TEST_F(l2ErfinvTest, 012_format)
{
    auto selfDesc = TensorDesc({2, 4, 3, 3}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto outDesc = TensorDesc({2, 4, 3, 3}, ACL_FLOAT, ACL_FORMAT_NHWC);

    auto ut = OP_API_UT(aclnnErfinv, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    ut.TestPrecision();
}

// format self是私有格式
TEST_F(l2ErfinvTest, 013_format_private)
{
    auto selfDesc = TensorDesc({2, 4, 3, 3, 5}, ACL_FLOAT, ACL_FORMAT_NC1HWC0);
    auto outDesc = TensorDesc({2, 4, 3, 3, 5}, ACL_FLOAT, ACL_FORMAT_NC1HWC0);

    auto ut = OP_API_UT(aclnnErfinv, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    //  ut.TestPrecision();
}

// 正常路径，float32，inplace
TEST_F(l2ErfinvTest, 014_inplace)
{
    auto selfDesc = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND)
                        .Value(vector<float>{1, -3, 0, 0.4, -0.5, 1.6, 0.7, 2.8, 9, 1, 1, -2});

    auto ut = OP_API_UT(aclnnInplaceErfinv, INPUT(selfDesc), OUTPUT());

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    ut.TestPrecision();
}

// 非连续
TEST_F(l2ErfinvTest, 015_not_contiguous)
{
    auto selfDesc = TensorDesc({2, 5}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {5, 2})
                        .Value(vector<float>{0.683, 0.954, 0.997, -1.6, 4, 0, 1, -2, 3, -4});
    auto outDesc = TensorDesc({2, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnErfinv, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);

    ut.TestPrecision();
}

// self.dim > 8
TEST_F(l2ErfinvTest, 016_self_dim_gt_8)
{
    auto selfDesc = TensorDesc({1, 2, 3, 1, 2, 3, 1, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({1, 2, 3, 1, 2, 3, 1, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnErfinv, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 910A，bf16
TEST_F(l2ErfinvTest, 017_ascend910A_error_support_bf16)
{
    auto selfDesc = TensorDesc({4, 4}, ACL_BF16, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({4, 4}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnErfinv, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 910B，bf16
TEST_F(l2ErfinvTest, 018_ascend910B_support_bf16)
{
    auto selfDesc = TensorDesc({4, 4}, ACL_BF16, ACL_FORMAT_ND);
    auto outDesc = TensorDesc({4, 4}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnErfinv, INPUT(selfDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

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
#include "../../../op_host/op_api/aclnn_gelu_v2.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"

class l2GeluV2Test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "l2GeluV2Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "l2GeluV2Test TearDown" << std::endl;
    }
};

// x为空
TEST_F(l2GeluV2Test, l2_gelu_v2_test_001)
{
    auto yDesc = TensorDesc({2, 4, 6, 7}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnGeluV2, INPUT((aclTensor*)nullptr, 0), OUTPUT(yDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

// y为空
TEST_F(l2GeluV2Test, l2_gelu_v2_test_002)
{
    auto xDesc = TensorDesc({2, 4, 6, 7}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnGeluV2, INPUT(xDesc, 0), OUTPUT((aclTensor*)nullptr));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

// x和y的数据类型不一致
TEST_F(l2GeluV2Test, l2_gelu_v2_test_003)
{
    auto xDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({2, 3}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnGeluV2, INPUT(xDesc, 0), OUTPUT(yDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// x数据类型不在支持范围内
TEST_F(l2GeluV2Test, l2_gelu_v2_test_004)
{
    auto xDesc = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({2, 3}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnGeluV2, INPUT(xDesc, 0), OUTPUT(yDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// x和y的shape不一致
TEST_F(l2GeluV2Test, l2_gelu_v2_test_005)
{
    auto xDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnGeluV2, INPUT(xDesc, 0), OUTPUT(yDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// x和y的shape不一致
TEST_F(l2GeluV2Test, l2_gelu_v2_test_006)
{
    auto xDesc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnGeluV2, INPUT(xDesc, 0), OUTPUT(yDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// x数据类型不在支持范围内
TEST_F(l2GeluV2Test, l2_gelu_v2_test_007)
{
    auto xDesc = TensorDesc({2, 3}, ACL_DT_UNDEFINED, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({2, 3}, ACL_DT_UNDEFINED, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnGeluV2, INPUT(xDesc, 0), OUTPUT(yDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// format NCHW、NHWC
TEST_F(l2GeluV2Test, l2_gelu_v2_test_008)
{
    auto xDesc = TensorDesc({2, 4, 6, 7}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto yDesc = TensorDesc({2, 4, 6, 7}, ACL_FLOAT, ACL_FORMAT_NHWC).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnGeluV2, INPUT(xDesc, 0), OUTPUT(yDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// format NDHWC、NCDHW
TEST_F(l2GeluV2Test, l2_gelu_v2_test_009)
{
    auto xDesc = TensorDesc({2, 4, 6, 7, 8}, ACL_FLOAT16, ACL_FORMAT_NDHWC);
    auto yDesc = TensorDesc({2, 4, 6, 7, 8}, ACL_FLOAT16, ACL_FORMAT_NCDHW).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnGeluV2, INPUT(xDesc, 0), OUTPUT(yDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// format HWCN
TEST_F(l2GeluV2Test, l2_gelu_v2_test_010)
{
    auto xDesc = TensorDesc({2, 4, 6, 8}, ACL_FLOAT, ACL_FORMAT_HWCN);
    auto yDesc = TensorDesc({2, 4, 6, 8}, ACL_FLOAT, ACL_FORMAT_HWCN).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnGeluV2, INPUT(xDesc, 0), OUTPUT(yDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// format self是私有格式
TEST_F(l2GeluV2Test, l2_gelu_v2_test_011)
{
    auto xDesc = TensorDesc({2, 4, 6, 8, 5}, ACL_FLOAT, ACL_FORMAT_NC1HWC0);
    auto yDesc = TensorDesc({2, 4, 6, 8}, ACL_FLOAT, ACL_FORMAT_HWCN);

    auto ut = OP_API_UT(aclnnGeluV2, INPUT(xDesc, 0), OUTPUT(yDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// format out是私有格式
TEST_F(l2GeluV2Test, l2_gelu_v2_test_012)
{
    auto xDesc = TensorDesc({2, 4, 6, 8}, ACL_FLOAT, ACL_FORMAT_HWCN);
    auto yDesc = TensorDesc({2, 4, 6, 8, 8}, ACL_FLOAT, ACL_FORMAT_NC1HWC0);

    auto ut = OP_API_UT(aclnnGeluV2, INPUT(xDesc, 0), OUTPUT(yDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// float16 正常路径
TEST_F(l2GeluV2Test, l2_gelu_v2_test_013)
{
    auto xDesc = TensorDesc({2, 4, 6, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({2, 4, 6, 8}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnGeluV2, INPUT(xDesc, 0), OUTPUT(yDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// float32 正常路径
TEST_F(l2GeluV2Test, l2_gelu_v2_test_014)
{
    auto xDesc = TensorDesc({2, 4, 6, 8, 4, 5}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({2, 4, 6, 8, 4, 5}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnGeluV2, INPUT(xDesc, 0), OUTPUT(yDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 数据范围[-1, 1] float16
TEST_F(l2GeluV2Test, l2_gelu_v2_test_015)
{
    auto xDesc = TensorDesc({2, 4, 6, 8, 4, 5}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto yDesc = TensorDesc({2, 4, 6, 8, 4, 5}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnGeluV2, INPUT(xDesc, 0), OUTPUT(yDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 数据范围[-1, 1] float32
TEST_F(l2GeluV2Test, l2_gelu_v2_test_016)
{
    auto xDesc = TensorDesc({2, 4, 6, 8, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto yDesc = TensorDesc({2, 4, 6, 8, 4, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnGeluV2, INPUT(xDesc, 0), OUTPUT(yDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

// 非连续
TEST_F(l2GeluV2Test, l2_gelu_v2_test_017)
{
    auto xDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2});
    auto yDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2}).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnGeluV2, INPUT(xDesc, 0), OUTPUT(yDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}
// 非连续 "tanh"
TEST_F(l2GeluV2Test, l2_gelu_v2_test_018)
{
    auto xDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2});
    auto yDesc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2}).Precision(0.01, 0.01);

    auto ut = OP_API_UT(aclnnGeluV2, INPUT(xDesc, 1), OUTPUT(yDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}
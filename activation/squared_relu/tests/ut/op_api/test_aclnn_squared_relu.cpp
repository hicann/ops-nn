/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <vector>
#include "gtest/gtest.h"

#include "../../../op_api/aclnn_squared_relu.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class l2_squared_relu_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "SquaredRelu Test Setup" << endl; }
    static void TearDownTestCase() { cout << "SquaredRelu Test TearDown" << endl; }
};

// 正常路径
TEST_F(l2_squared_relu_test, case_1)
{
    auto tensor_desc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc(tensor_desc);

    auto ut = OP_API_UT(aclnnSquaredRelu, INPUT(tensor_desc), OUTPUT(out_tensor_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 空tensor
TEST_F(l2_squared_relu_test, case_2)
{
    auto self_tensor_desc = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_tensor_desc = TensorDesc(self_tensor_desc);
    auto ut = OP_API_UT(aclnnSquaredRelu, INPUT(self_tensor_desc), OUTPUT(out_tensor_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// CheckNotNull input
TEST_F(l2_squared_relu_test, case_3)
{
    auto tensor_desc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnSquaredRelu, INPUT(nullptr), OUTPUT(tensor_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// CheckNotNull out
TEST_F(l2_squared_relu_test, case_4)
{
    auto tensor_desc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnSquaredRelu, INPUT(tensor_desc), OUTPUT(nullptr));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// input dtype不支持
TEST_F(l2_squared_relu_test, case_5)
{
    auto self_desc = TensorDesc({2, 4}, ACL_INT16, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnSquaredRelu, INPUT(self_desc), OUTPUT(out_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// out dtype不支持
TEST_F(l2_squared_relu_test, case_6)
{
    auto self_desc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({2, 4}, ACL_INT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnSquaredRelu, INPUT(self_desc), OUTPUT(out_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// out dtype与input不一致
TEST_F(l2_squared_relu_test, case_7)
{
    auto self_desc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({2, 4}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnSquaredRelu, INPUT(self_desc), OUTPUT(out_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// out与input的shape不一致（issue #7756）
TEST_F(l2_squared_relu_test, case_8)
{
    auto self_desc = TensorDesc({4, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_desc = TensorDesc({4, 4}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnSquaredRelu, INPUT(self_desc), OUTPUT(out_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 维度数超过8
TEST_F(l2_squared_relu_test, case_9)
{
    auto self_desc = TensorDesc({1, 2, 2, 2, 2, 2, 2, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out_desc = TensorDesc(self_desc);

    auto ut = OP_API_UT(aclnnSquaredRelu, INPUT(self_desc), OUTPUT(out_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// 非连续input（input支持非连续）。stub不模拟kernel执行，仅校验首段通过；
// 非连续场景的端到端写正确性由真机验证覆盖（910B3 repro：连续/非连续逻辑视图结果均正确）
TEST_F(l2_squared_relu_test, case_10)
{
    auto self_desc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(-1000, 1000);
    auto out_desc = TensorDesc(self_desc);

    auto ut = OP_API_UT(aclnnSquaredRelu, INPUT(self_desc), OUTPUT(out_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 非连续out（计算结果按out逻辑视图回写）。stub不模拟kernel执行，仅校验首段通过；
// 端到端写正确性由真机验证覆盖（910B3 repro：strides [8,1] 视图写后逐元素比对正确）
TEST_F(l2_squared_relu_test, case_11)
{
    auto self_desc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    // out声明shape [2,4]、strides [8,1]、底层storage [2,8]，为非连续视图
    auto out_desc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND, {8, 1}, 0, {2, 8});

    auto ut = OP_API_UT(aclnnSquaredRelu, INPUT(self_desc), OUTPUT(out_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// bf16，平台判据与L1实现保持一致（DAV_2201或regbase架构支持bf16；
// 当前SDK的NpuArch枚举仅含DAV_3510，为regbase架构代表）
TEST_F(l2_squared_relu_test, case_12)
{
    auto tensor_desc = TensorDesc({2, 4}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_tensor_desc = TensorDesc(tensor_desc);

    auto ut = OP_API_UT(aclnnSquaredRelu, INPUT(tensor_desc), OUTPUT(out_tensor_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    if (curArch == NpuArch::DAV_2201 || curArch == NpuArch::DAV_3510) {
        EXPECT_EQ(aclRet, ACL_SUCCESS);
    } else {
        EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    }
}

// 正常路径，8维边界
TEST_F(l2_squared_relu_test, case_13)
{
    auto tensor_desc = TensorDesc({1, 2, 3, 4, 5, 6, 7, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out_tensor_desc = TensorDesc(tensor_desc);

    auto ut = OP_API_UT(aclnnSquaredRelu, INPUT(tensor_desc), OUTPUT(out_tensor_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常路径，fp16
TEST_F(l2_squared_relu_test, case_14)
{
    auto tensor_desc = TensorDesc({2, 4, 6, 8}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-100, 100);
    auto out_tensor_desc = TensorDesc(tensor_desc);

    auto ut = OP_API_UT(aclnnSquaredRelu, INPUT(tensor_desc), OUTPUT(out_tensor_desc));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 输出参数workspaceSize为空指针（可靠性检视意见：应返回错误码而非崩溃）
TEST_F(l2_squared_relu_test, case_15)
{
    auto self_desc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    aclTensor* in = DescToAclContainer(self_desc);
    aclTensor* out = DescToAclContainer(self_desc);
    aclOpExecutor* ex = nullptr;

    aclnnStatus aclRet = aclnnSquaredReluGetWorkspaceSize(in, out, nullptr, &ex);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
    EXPECT_EQ(ex, nullptr);
    Release(in);
    Release(out);
}

// 输出参数executor为空指针
TEST_F(l2_squared_relu_test, case_16)
{
    auto self_desc = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    aclTensor* in = DescToAclContainer(self_desc);
    aclTensor* out = DescToAclContainer(self_desc);
    uint64_t workspaceSize = 0;

    aclnnStatus aclRet = aclnnSquaredReluGetWorkspaceSize(in, out, &workspaceSize, nullptr);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
    Release(in);
    Release(out);
}

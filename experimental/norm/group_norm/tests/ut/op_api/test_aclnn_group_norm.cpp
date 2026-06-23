/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../../../op_api/aclnn_group_norm_experimental.h"
#include <gtest/gtest.h>
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace op;

class TestAclnnGroupNorm : public testing::Test {};

TEST_F(TestAclnnGroupNorm, group_norm_normal_float16)
{
    auto self = TensorDesc({2, 8, 4, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto gamma = TensorDesc({8}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto beta = TensorDesc({8}, ACL_FLOAT16, ACL_FORMAT_ND);
    int64_t n = 2;
    int64_t c = 8;
    int64_t hxw = 16;
    int64_t group = 4;
    double eps = 0.00001;
    auto out = TensorDesc({2, 8, 4, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto mean = TensorDesc({2, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto rstd = TensorDesc({2, 4}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnGroupNorm, INPUT(self, gamma, beta, n, c, hxw, group, eps), OUTPUT(out, mean, rstd));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
}

TEST_F(TestAclnnGroupNorm, group_norm_normal_float32_without_gamma_beta)
{
    auto self = TensorDesc({1, 4, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t n = 1;
    int64_t c = 4;
    int64_t hxw = 4;
    int64_t group = 2;
    double eps = 0.00001;
    auto out = TensorDesc({1, 4, 2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto mean = TensorDesc({1, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto rstd = TensorDesc({1, 2}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnGroupNorm,
        INPUT(self, static_cast<aclTensor*>(nullptr), static_cast<aclTensor*>(nullptr), n, c, hxw, group, eps),
        OUTPUT(out, mean, rstd));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
}

TEST_F(TestAclnnGroupNorm, group_norm_null_self)
{
    auto gamma = TensorDesc({8}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto beta = TensorDesc({8}, ACL_FLOAT16, ACL_FORMAT_ND);
    int64_t n = 2;
    int64_t c = 8;
    int64_t hxw = 16;
    int64_t group = 4;
    double eps = 0.00001;
    auto out = TensorDesc({2, 8, 4, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto mean = TensorDesc({2, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto rstd = TensorDesc({2, 4}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnGroupNorm, INPUT(static_cast<aclTensor*>(nullptr), gamma, beta, n, c, hxw, group, eps),
        OUTPUT(out, mean, rstd));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(TestAclnnGroupNorm, group_norm_invalid_dtype)
{
    auto self = TensorDesc({2, 8, 4, 4}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto gamma = TensorDesc({8}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto beta = TensorDesc({8}, ACL_DOUBLE, ACL_FORMAT_ND);
    int64_t n = 2;
    int64_t c = 8;
    int64_t hxw = 16;
    int64_t group = 4;
    double eps = 0.00001;
    auto out = TensorDesc({2, 8, 4, 4}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto mean = TensorDesc({2, 4}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto rstd = TensorDesc({2, 4}, ACL_DOUBLE, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnGroupNorm, INPUT(self, gamma, beta, n, c, hxw, group, eps), OUTPUT(out, mean, rstd));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(TestAclnnGroupNorm, group_norm_invalid_group)
{
    auto self = TensorDesc({2, 8, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gamma = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto beta = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t n = 2;
    int64_t c = 8;
    int64_t hxw = 16;
    int64_t group = 3;
    double eps = 0.00001;
    auto out = TensorDesc({2, 8, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto mean = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto rstd = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnGroupNorm, INPUT(self, gamma, beta, n, c, hxw, group, eps), OUTPUT(out, mean, rstd));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(TestAclnnGroupNorm, group_norm_invalid_mean_shape)
{
    auto self = TensorDesc({2, 8, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gamma = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto beta = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t n = 2;
    int64_t c = 8;
    int64_t hxw = 16;
    int64_t group = 4;
    double eps = 0.00001;
    auto out = TensorDesc({2, 8, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto mean = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto rstd = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnGroupNorm, INPUT(self, gamma, beta, n, c, hxw, group, eps), OUTPUT(out, mean, rstd));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

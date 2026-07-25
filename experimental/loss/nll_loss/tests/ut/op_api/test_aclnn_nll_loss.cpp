/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_nll_loss.cpp
 * \brief
 */
#include <vector>
#include <array>
#include "gtest/gtest.h"

#include "opdev/platform.h"
#include "../../../op_api/aclnn_nll_loss.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

class l2_nll_loss_test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "l2_nll_loss_test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "l2_nll_loss_test TearDown" << std::endl; }
};

// reduction: 0 = none, 1 = mean, 2 = sum
TEST_F(l2_nll_loss_test, case_01_float32_mean)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);

    auto selfDesc = TensorDesc({4, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto targetDesc = TensorDesc({4}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 7);
    auto weightDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
    auto outDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto totalWeightDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t reduction = 1;
    int64_t ignoreIndex = -100;

    auto ut = OP_API_UT(aclnnNLLLoss, INPUT(selfDesc, targetDesc, weightDesc, reduction, ignoreIndex),
                        OUTPUT(outDesc, totalWeightDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_nll_loss_test, case_02_float16_sum)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);

    auto selfDesc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto targetDesc = TensorDesc({16}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 31);
    auto weightDesc = TensorDesc({32}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);
    auto outDesc = TensorDesc({1}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto totalWeightDesc = TensorDesc({1}, ACL_FLOAT16, ACL_FORMAT_ND);
    int64_t reduction = 2;
    int64_t ignoreIndex = -100;

    auto ut = OP_API_UT(aclnnNLLLoss, INPUT(selfDesc, targetDesc, weightDesc, reduction, ignoreIndex),
                        OUTPUT(outDesc, totalWeightDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

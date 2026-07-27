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
 * \file test_aclnn_sync_batch_norm_backward_reduce.cpp
 * \brief
 */
#include <vector>
#include <array>
#include "gtest/gtest.h"

#include "opdev/platform.h"
#include "../../../op_api/aclnn_batch_norm_backward_reduce.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

class l2_sync_batch_norm_backward_reduce_test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "l2_sync_batch_norm_backward_reduce_test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "l2_sync_batch_norm_backward_reduce_test TearDown" << std::endl; }
};

TEST_F(l2_sync_batch_norm_backward_reduce_test, case_01_float32)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);

    auto gradOutDesc = TensorDesc({4, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto inputDesc = TensorDesc({4, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto meanDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto invstdDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
    auto weightDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
    bool inputG = true;
    bool weightG = true;
    bool biasG = true;
    auto sumDyDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sumDyXmuDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradWeightDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradBiasDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnBatchNormReduceBackward,
                        INPUT(gradOutDesc, inputDesc, meanDesc, invstdDesc, weightDesc, inputG, weightG, biasG),
                        OUTPUT(sumDyDesc, sumDyXmuDesc, gradWeightDesc, gradBiasDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_sync_batch_norm_backward_reduce_test, case_02_float16)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);

    auto gradOutDesc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto inputDesc = TensorDesc({16, 32}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto meanDesc = TensorDesc({32}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto invstdDesc = TensorDesc({32}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);
    auto weightDesc = TensorDesc({32}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(0, 1);
    bool inputG = true;
    bool weightG = true;
    bool biasG = true;
    auto sumDyDesc = TensorDesc({32}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto sumDyXmuDesc = TensorDesc({32}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto gradWeightDesc = TensorDesc({32}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto gradBiasDesc = TensorDesc({32}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnBatchNormReduceBackward,
                        INPUT(gradOutDesc, inputDesc, meanDesc, invstdDesc, weightDesc, inputG, weightG, biasG),
                        OUTPUT(sumDyDesc, sumDyXmuDesc, gradWeightDesc, gradBiasDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

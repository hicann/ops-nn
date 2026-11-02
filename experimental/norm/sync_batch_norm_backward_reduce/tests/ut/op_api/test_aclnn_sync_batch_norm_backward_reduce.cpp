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

// *** issue 7105: mean/invstd shape校验（长度必须与input的channel轴一致） ***
TEST_F(l2_sync_batch_norm_backward_reduce_test, case_03_mean_shape_less_than_c)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);

    auto gradOutDesc = TensorDesc({4, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto inputDesc = TensorDesc({4, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto meanDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
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
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_sync_batch_norm_backward_reduce_test, case_04_mean_shape_greater_than_c)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);

    auto gradOutDesc = TensorDesc({4, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto inputDesc = TensorDesc({4, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto meanDesc = TensorDesc({9}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
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
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_sync_batch_norm_backward_reduce_test, case_05_invstd_shape_not_equal_c)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);

    auto gradOutDesc = TensorDesc({4, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto inputDesc = TensorDesc({4, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto meanDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto invstdDesc = TensorDesc({7}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
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
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// *** issue 7106: weight可选参数校验（非空时dtype/format/shape必须合法） ***
TEST_F(l2_sync_batch_norm_backward_reduce_test, case_06_weight_invalid_dtype)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);

    auto gradOutDesc = TensorDesc({4, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto inputDesc = TensorDesc({4, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto meanDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto invstdDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
    auto weightDesc = TensorDesc({8}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-8, 8);
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
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_sync_batch_norm_backward_reduce_test, case_07_weight_shape_less_than_c)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);

    auto gradOutDesc = TensorDesc({4, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto inputDesc = TensorDesc({4, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto meanDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto invstdDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
    auto weightDesc = TensorDesc({7}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
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
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_sync_batch_norm_backward_reduce_test, case_08_weight_shape_greater_than_c)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);

    auto gradOutDesc = TensorDesc({4, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto inputDesc = TensorDesc({4, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto meanDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto invstdDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
    auto weightDesc = TensorDesc({9}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
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
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_sync_batch_norm_backward_reduce_test, case_09_weight_nullptr_valid)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);

    auto gradOutDesc = TensorDesc({4, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto inputDesc = TensorDesc({4, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto meanDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto invstdDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
    bool inputG = true;
    bool weightG = true;
    bool biasG = true;
    auto sumDyDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sumDyXmuDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradWeightDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradBiasDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnBatchNormReduceBackward,
        INPUT(gradOutDesc, inputDesc, meanDesc, invstdDesc, (aclTensor*)nullptr, inputG, weightG, biasG),
        OUTPUT(sumDyDesc, sumDyXmuDesc, gradWeightDesc, gradBiasDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

TEST_F(l2_sync_batch_norm_backward_reduce_test, case_10_weight_invalid_format)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);

    auto gradOutDesc = TensorDesc({4, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto inputDesc = TensorDesc({4, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto meanDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto invstdDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 1);
    auto weightDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_NC1HWC0).ValueRange(0, 1);
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
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// *** issue 7104: 空Tensor输出填零（所有架构一致，不再依赖IsRegbase） ***
TEST_F(l2_sync_batch_norm_backward_reduce_test, case_11_empty_input_fill_zero)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);

    auto gradOutDesc = TensorDesc({0, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto inputDesc = TensorDesc({0, 8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
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

TEST_F(l2_sync_batch_norm_backward_reduce_test, case_12_empty_mid_dim_fill_zero)
{
    op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);

    auto gradOutDesc = TensorDesc({2, 8, 0, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto inputDesc = TensorDesc({2, 8, 0, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
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

/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <array>
#include "gtest/gtest.h"
#include "opdev/platform.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"
#include "norm/batch_norm_grad_v3/op_host/op_api/aclnn_batch_norm_backward.h"
#include "norm/batch_norm_grad_v3/op_host/op_api/aclnn_fast_batch_norm_backward.h"

class BatchNormGradV3MaskWorkspaceUt : public testing::TestWithParam<std::array<bool, 2>> {
protected:
    void CheckMask(bool fast, op::SocVersion soc = op::SocVersion::ASCEND950, bool training = true)
    {
        op::SocVersionManager platform(soc);
        auto expectedOutputs = GetParam();
        auto dy = TensorDesc({2, 16, 8, 8}, ACL_FLOAT, ACL_FORMAT_NCHW);
        auto x = TensorDesc({2, 16, 8, 8}, ACL_FLOAT, ACL_FORMAT_NCHW);
        auto weight = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND);
        auto mean = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND);
        auto variance = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND);
        auto savedMean = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND);
        auto savedRstd = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND);
        auto dx = TensorDesc({2, 16, 8, 8}, ACL_FLOAT, ACL_FORMAT_NCHW);
        // Disabled L2 outputs remain non-null to verify mask takes precedence.
        auto dweight = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND);
        auto dbias = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND);
        auto mask = BoolArrayDesc(vector<bool>{true, expectedOutputs[0], expectedOutputs[1]});
        uint64_t workspace = 0;
        if (fast) {
            auto ut = OP_API_UT(aclnnFastBatchNormBackward,
                                INPUT(dy, x, weight, mean, variance, savedMean, savedRstd, training, 1e-5, mask, 0),
                                OUTPUT(dx, dweight, dbias));
            EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace), ACLNN_SUCCESS);
        } else {
            auto ut = OP_API_UT(aclnnBatchNormBackward,
                                INPUT(dy, x, weight, mean, variance, savedMean, savedRstd, training, 1e-5, mask),
                                OUTPUT(dx, dweight, dbias));
            EXPECT_EQ(ut.TestGetWorkspaceSize(&workspace), ACLNN_SUCCESS);
        }
    }
};

TEST_P(BatchNormGradV3MaskWorkspaceUt, AclnnBatchNormBackward) { CheckMask(false); }
// Compatibility: A2/A3 keep their original L0 attributes and L2 mask handling.
TEST_P(BatchNormGradV3MaskWorkspaceUt, AclnnFastBatchNormBackwardA2) { CheckMask(true, op::SocVersion::ASCEND910B); }
TEST_P(BatchNormGradV3MaskWorkspaceUt, AclnnFastBatchNormBackwardA3) { CheckMask(true, op::SocVersion::ASCEND910_93); }

TEST_P(BatchNormGradV3MaskWorkspaceUt, AclnnFastBatchNormBackwardA2Infer)
{
    CheckMask(true, op::SocVersion::ASCEND910B, false);
}
TEST_P(BatchNormGradV3MaskWorkspaceUt, AclnnFastBatchNormBackwardA3Infer)
{
    CheckMask(true, op::SocVersion::ASCEND910_93, false);
}
TEST_P(BatchNormGradV3MaskWorkspaceUt, AclnnBatchNormBackwardA2) { CheckMask(false, op::SocVersion::ASCEND910B); }
TEST_P(BatchNormGradV3MaskWorkspaceUt, AclnnBatchNormBackwardA3) { CheckMask(false, op::SocVersion::ASCEND910_93); }
TEST_P(BatchNormGradV3MaskWorkspaceUt, AclnnBatchNormBackwardA2Infer)
{
    CheckMask(false, op::SocVersion::ASCEND910B, false);
}
TEST_P(BatchNormGradV3MaskWorkspaceUt, AclnnBatchNormBackwardA3Infer)
{
    CheckMask(false, op::SocVersion::ASCEND910_93, false);
}

INSTANTIATE_TEST_SUITE_P(AllMasks, BatchNormGradV3MaskWorkspaceUt,
                         testing::Values(std::array<bool, 2>{true, true}, std::array<bool, 2>{true, false},
                                         std::array<bool, 2>{false, true}, std::array<bool, 2>{false, false}));

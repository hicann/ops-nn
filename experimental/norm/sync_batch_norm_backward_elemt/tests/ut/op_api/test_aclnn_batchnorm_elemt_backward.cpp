/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "../../../op_host/op_api/aclnn_batch_norm_elemt_backward.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"
#include "op_api/op_api_def.h"

class l2BatchNormElemtBackwardTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "l2l2BatchNormElemtBackwardTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "l2l2BatchNormElemtBackwardTest TearDown" << std::endl; }
};

TEST_F(l2BatchNormElemtBackwardTest, ascend910B2_batch_norm_elemt_backward_bf16)
{
    auto gradOutDesc = TensorDesc({2, 3, 1, 4}, ACL_BF16, ACL_FORMAT_NCHW).ValueRange(1, 1);
    auto selfDesc = TensorDesc({2, 3, 1, 4}, ACL_BF16, ACL_FORMAT_NCHW).ValueRange(1, 1);

    auto meanDesc = TensorDesc({3}, ACL_BF16, ACL_FORMAT_ND).Value(vector<float>{8, 5, 9});
    auto invstdDesc = TensorDesc({3}, ACL_BF16, ACL_FORMAT_ND).Value(vector<float>{2, 1, 2});
    auto weightDesc = TensorDesc({3}, ACL_BF16, ACL_FORMAT_ND).Value(vector<float>{1, 1, 4});
    auto sumDyDesc = TensorDesc({3}, ACL_BF16, ACL_FORMAT_ND).Value(vector<float>{2, 2, 6});
    auto sumDyXmnDesc = TensorDesc({3}, ACL_BF16, ACL_FORMAT_ND).Value(vector<float>{2, 3, 11});
    auto counterDesc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int>{5, 5, 5});
    auto gradInputDesc = TensorDesc(selfDesc);

    auto ut = OP_API_UT(
        aclnnBatchNormElemtBackward,
        INPUT(gradOutDesc, selfDesc, meanDesc, invstdDesc, weightDesc, sumDyDesc, sumDyXmnDesc, counterDesc),
        OUTPUT(gradInputDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    if (op::GetCurrentPlatformInfo().GetSocVersion() == op::SocVersion::ASCEND910B) {
        EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
    } else {
        // EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
    }
}

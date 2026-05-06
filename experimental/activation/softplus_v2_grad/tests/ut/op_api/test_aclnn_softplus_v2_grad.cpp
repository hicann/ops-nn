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
#include <vector>

#include "../../../op_host/op_api/aclnn_softplus_backward.h"
#include "gtest/gtest.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace std;

class l2_softplus_backward_test : public testing::Test {
   protected:
    static void SetUpTestCase() { std::cout << "l2_softplus_backward_test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "l2_softplus_backward_test TearDown" << std::endl; }
};

TEST_F(l2_softplus_backward_test, case_01_float) {
    auto gradOutputDesc =
        TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{-1.023, 2023.08, 3.14, 10987654321.0});
    auto selfDesc =
        TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{-1.023, 2023.08, 3.14, 10987654321.0});
    auto outDesc = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto betaDesc = ScalarDesc(0.0f);
    auto threholdDesc = ScalarDesc(20.0f);
    auto ut =
        OP_API_UT(aclnnSoftplusBackward, INPUT(gradOutputDesc, selfDesc, betaDesc, threholdDesc), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

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
#include "gtest/gtest.h"
#include "opdev/op_log.h"
#include "level2/aclnn_gelu_backward_v2.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace op;
using namespace std;

class l2_gelu_backward_v2_test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "Gelu Backward V2 Test Setup" << std::endl; }
    static void TearDownTestCase() { std::cout << "Gelu Backward V2 Test TearDown" << std::endl; }
};

TEST_F(l2_gelu_backward_v2_test, gelu_backward_v2_testcase_001_normal_float32)
{
    auto gradOutputDesc = TensorDesc({2, 5}, ACL_FLOAT, ACL_FORMAT_ND)
                              .Value(vector<float>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
    auto selfDesc = TensorDesc({2, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto gradInputDesc = TensorDesc(gradOutputDesc).Precision(0.001, 0.001);
    char* approximate = "none";
    auto ut = OP_API_UT(aclnnGeluBackwardV2, INPUT(gradOutputDesc, selfDesc, approximate), OUTPUT(gradInputDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    // ut.TestPrecision();
}

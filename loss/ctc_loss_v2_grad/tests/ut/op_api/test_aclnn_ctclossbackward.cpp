/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
#include <vector>
#include <array>
#include "gtest/gtest.h"
#include "../../../op_api/aclnn_ctc_loss_backward.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class l2_ctc_loss_backward_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "ctc_loss_backward_test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "ctc_loss_backward_test TearDown" << std::endl;
    }
};

int64_t TT = 12;
int64_t NN = 4;
int64_t TEMP_NN = NN;
int64_t CC = 5;
int64_t SS = 7;
int64_t CC_V3 = 20000;

// 正常情況aicore target is bool
TEST_F(l2_ctc_loss_backward_test, test_ctc_loss_backward_float_target_is_bool)
{
    auto gradOut =
        TensorDesc({NN}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10).Value(vector<float>{1.0, 1.0, 1.0, 1.0});
    auto logProbs = TensorDesc({TT, NN, CC}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto targets = TensorDesc({NN, SS}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto inputLengths = IntArrayDesc(vector<int64_t>{TT, TT, TT, TT});
    auto targetLengths = IntArrayDesc(vector<int64_t>{SS, SS, SS, SS});
    auto negLogLikelihood = TensorDesc({NN}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto logAlpha = TensorDesc({NN, TT, 15}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    int64_t blank = 0;
    bool zeroInfinity = false;
    auto out = TensorDesc({TT, NN, CC}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnCtcLossBackward,
        INPUT(gradOut, logProbs, targets, inputLengths, targetLengths, negLogLikelihood, logAlpha, blank, zeroInfinity),
        OUTPUT(out));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

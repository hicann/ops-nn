/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "level2/aclnn_scaled_masked_softmax.h"

#include <array>
#include <vector>
#include "gtest/gtest.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace op;
using namespace std;

class l2_scaled_masked_softmax_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        setenv("ASCEND_SLOG_PRINT_TO_STDOUT", "1", true);
        setenv("ASCEND_GLOBAL_LOG_LEVEL", "0", true);
        cout << "scaled_masked_softmax_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        unsetenv("ASCEND_SLOG_PRINT_TO_STDOUT");
        unsetenv("ASCEND_GLOBAL_LOG_LEVEL");
        cout << "scaled_masked_softmax_test TearDown" << endl;
    }
};

TEST_F(l2_scaled_masked_softmax_test, Ascend910B_aclnnScaledMaskedSoftmax_dimNum_not_4)
{
    auto xTensor = TensorDesc({1, 2, 2, 128, 128}, ACL_BF16, ACL_FORMAT_ND);
    auto maskTensor = TensorDesc({1, 2, 2, 128, 128}, ACL_BOOL, ACL_FORMAT_ND);
    auto outTensor = TensorDesc({1, 2, 2, 128, 128}, ACL_BF16, ACL_FORMAT_ND);
    double scale = 0.8;
    bool fixTriu = false;
    auto ut = OP_API_UT(aclnnScaledMaskedSoftmax, INPUT(xTensor, maskTensor, scale, fixTriu), OUTPUT(outTensor));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_scaled_masked_softmax_test, Ascend910B_aclnnScaledMaskedSoftmax_dim4_out_of_range)
{
    auto xTensor = TensorDesc({2, 2, 128, 5000}, ACL_BF16, ACL_FORMAT_ND);
    auto maskTensor = TensorDesc({1, 2, 128, 5000}, ACL_BOOL, ACL_FORMAT_ND);
    auto outTensor = TensorDesc({2, 2, 128, 5000}, ACL_BF16, ACL_FORMAT_ND);
    double scale = 0.8;
    bool fixTriu = false;
    auto ut = OP_API_UT(aclnnScaledMaskedSoftmax, INPUT(xTensor, maskTensor, scale, fixTriu), OUTPUT(outTensor));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_scaled_masked_softmax_test, Ascend910B_aclnnScaledMaskedSoftmax_broadcast_failed)
{
    auto xTensor = TensorDesc({2, 4, 128, 128}, ACL_BF16, ACL_FORMAT_ND);
    auto maskTensor = TensorDesc({1, 2, 128, 128}, ACL_BOOL, ACL_FORMAT_ND);
    auto outTensor = TensorDesc({2, 2, 128, 128}, ACL_BF16, ACL_FORMAT_ND);
    double scale = 0.8;
    bool fixTriu = false;
    auto ut = OP_API_UT(aclnnScaledMaskedSoftmax, INPUT(xTensor, maskTensor, scale, fixTriu), OUTPUT(outTensor));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
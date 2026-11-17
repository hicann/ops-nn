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
#include "../../../op_api/aclnn_chunk_loss.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class l2_chunk_loss_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "chunk_loss_test SetUp" << endl; }

    static void TearDownTestCase() { cout << "chunk_loss_test TearDown" << endl; }
};

TEST_F(l2_chunk_loss_test, Ascend910B_case_fp16_normal)
{
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B) {
        auto hidden = TensorDesc({4, 8}, ACL_FLOAT16, ACL_FORMAT_ND);
        auto weight = TensorDesc({16, 8}, ACL_FLOAT16, ACL_FORMAT_ND);
        auto labels = TensorDesc({4}, ACL_INT64, ACL_FORMAT_ND);
        auto loss = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
        auto maxOut = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
        auto sumOut = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
        auto dH = TensorDesc({4, 8}, ACL_FLOAT16, ACL_FORMAT_ND);
        double alpha = 4.0;
        int64_t ignoreIndex = -100;

        auto ut = OP_API_UT(aclnnChunkLoss, INPUT(hidden, weight, labels, alpha, ignoreIndex),
                            OUTPUT(loss, maxOut, sumOut, dH));

        uint64_t workspaceSize = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
        EXPECT_EQ(aclRet, ACL_SUCCESS);
    }
}

TEST_F(l2_chunk_loss_test, Ascend910B_case_bf16_normal)
{
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B) {
        auto hidden = TensorDesc({8, 16}, ACL_BF16, ACL_FORMAT_ND);
        auto weight = TensorDesc({32, 16}, ACL_BF16, ACL_FORMAT_ND);
        auto labels = TensorDesc({8}, ACL_INT64, ACL_FORMAT_ND);
        auto loss = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
        auto maxOut = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
        auto sumOut = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND);
        auto dH = TensorDesc({8, 16}, ACL_BF16, ACL_FORMAT_ND);
        double alpha = 1.0;
        int64_t ignoreIndex = -100;

        auto ut = OP_API_UT(aclnnChunkLoss, INPUT(hidden, weight, labels, alpha, ignoreIndex),
                            OUTPUT(loss, maxOut, sumOut, dH));

        uint64_t workspaceSize = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
        EXPECT_EQ(aclRet, ACL_SUCCESS);
    }
}

TEST_F(l2_chunk_loss_test, Ascend910B_case_nullptr)
{
    auto weight = TensorDesc({16, 8}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto labels = TensorDesc({4}, ACL_INT64, ACL_FORMAT_ND);
    auto loss = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    auto maxOut = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sumOut = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dH = TensorDesc({4, 8}, ACL_FLOAT16, ACL_FORMAT_ND);
    double alpha = 4.0;
    int64_t ignoreIndex = -100;

    auto ut = OP_API_UT(aclnnChunkLoss, INPUT((aclTensor*)nullptr, weight, labels, alpha, ignoreIndex),
                        OUTPUT(loss, maxOut, sumOut, dH));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_chunk_loss_test, Ascend910B_case_invalid_alpha)
{
    auto hidden = TensorDesc({4, 8}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto weight = TensorDesc({16, 8}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto labels = TensorDesc({4}, ACL_INT64, ACL_FORMAT_ND);
    auto loss = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    auto maxOut = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sumOut = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dH = TensorDesc({4, 8}, ACL_FLOAT16, ACL_FORMAT_ND);
    double alpha = 0.0;
    int64_t ignoreIndex = -100;

    auto ut = OP_API_UT(aclnnChunkLoss, INPUT(hidden, weight, labels, alpha, ignoreIndex),
                        OUTPUT(loss, maxOut, sumOut, dH));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_chunk_loss_test, Ascend910B_case_invalid_dtype)
{
    auto hidden = TensorDesc({4, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto weight = TensorDesc({16, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto labels = TensorDesc({4}, ACL_INT64, ACL_FORMAT_ND);
    auto loss = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    auto maxOut = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sumOut = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dH = TensorDesc({4, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    double alpha = 4.0;
    int64_t ignoreIndex = -100;

    auto ut = OP_API_UT(aclnnChunkLoss, INPUT(hidden, weight, labels, alpha, ignoreIndex),
                        OUTPUT(loss, maxOut, sumOut, dH));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_chunk_loss_test, Ascend910B_case_mix_dtype_invalid)
{
    auto hidden = TensorDesc({4, 8}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto weight = TensorDesc({16, 8}, ACL_BF16, ACL_FORMAT_ND);
    auto labels = TensorDesc({4}, ACL_INT64, ACL_FORMAT_ND);
    auto loss = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    auto maxOut = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sumOut = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dH = TensorDesc({4, 8}, ACL_FLOAT16, ACL_FORMAT_ND);
    double alpha = 4.0;
    int64_t ignoreIndex = -100;

    auto ut = OP_API_UT(aclnnChunkLoss, INPUT(hidden, weight, labels, alpha, ignoreIndex),
                        OUTPUT(loss, maxOut, sumOut, dH));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_chunk_loss_test, Ascend910B_case_shape_mismatch_invalid)
{
    auto hidden = TensorDesc({4, 8}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto weight = TensorDesc({16, 8}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto labels = TensorDesc({3}, ACL_INT64, ACL_FORMAT_ND);
    auto loss = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    auto maxOut = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sumOut = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dH = TensorDesc({4, 8}, ACL_FLOAT16, ACL_FORMAT_ND);
    double alpha = 4.0;
    int64_t ignoreIndex = -100;

    auto ut = OP_API_UT(aclnnChunkLoss, INPUT(hidden, weight, labels, alpha, ignoreIndex),
                        OUTPUT(loss, maxOut, sumOut, dH));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_chunk_loss_test, Ascend910B_case_output_shape_mismatch_invalid)
{
    auto hidden = TensorDesc({4, 8}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto weight = TensorDesc({16, 8}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto labels = TensorDesc({4}, ACL_INT64, ACL_FORMAT_ND);
    auto loss = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND);
    auto maxOut = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sumOut = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dH = TensorDesc({4, 8}, ACL_FLOAT16, ACL_FORMAT_ND);
    double alpha = 4.0;
    int64_t ignoreIndex = -100;

    auto ut = OP_API_UT(aclnnChunkLoss, INPUT(hidden, weight, labels, alpha, ignoreIndex),
                        OUTPUT(loss, maxOut, sumOut, dH));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_chunk_loss_test, Ascend910B_case_loss_not_scalar_invalid)
{
    auto hidden = TensorDesc({4, 8}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto weight = TensorDesc({16, 8}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto labels = TensorDesc({4}, ACL_INT64, ACL_FORMAT_ND);
    auto loss = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto maxOut = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sumOut = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dH = TensorDesc({4, 8}, ACL_FLOAT16, ACL_FORMAT_ND);
    double alpha = 4.0;
    int64_t ignoreIndex = -100;

    auto ut = OP_API_UT(aclnnChunkLoss, INPUT(hidden, weight, labels, alpha, ignoreIndex),
                        OUTPUT(loss, maxOut, sumOut, dH));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

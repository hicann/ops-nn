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
#include "../../../op_api/aclnn_ctc_loss.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class l2_ctc_loss_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "ctc_loss_test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "ctc_loss_test TearDown" << std::endl;
    }
};

int64_t T = 12;
int64_t N = 4;
int64_t TEMP_N = N;
int64_t C = 5;
int64_t S = 7;
int64_t LOGALPHA_X = ((2 * S + 1) + 7) / 8 * 8;
// int64_t CC_V3 = 20000;

// logProbs is empty Tensor 空tensor
TEST_F(l2_ctc_loss_test, test_ctc_loss_logprobs_is_empty_tensor_normal)
{
    N = 0;
    S = 0;
    LOGALPHA_X = ((2 * S + 1) + 7) / 8 * 8;
    auto logProbs = TensorDesc({T, N, C}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto targets = TensorDesc({N, S}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto inputLengths = IntArrayDesc(vector<int64_t>{});
    auto targetLengths = IntArrayDesc(vector<int64_t>{});
    int64_t blank = 0;
    bool zeroInfinity = false;
    auto negLogLikelihoodOut = TensorDesc({N}, ACL_DOUBLE, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto logAlphaOut = TensorDesc({N, T, LOGALPHA_X}, ACL_DOUBLE, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    auto ut = OP_API_UT(
        aclnnCtcLoss, INPUT(logProbs, targets, inputLengths, targetLengths, blank, zeroInfinity),
        OUTPUT(negLogLikelihoodOut, logAlphaOut));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    N = TEMP_N;
    S = 7;
    LOGALPHA_X = ((2 * S + 1) + 7) / 8 * 8;
    ut.TestPrecision();
}

// ============以下为异常拦截场景
// output is null
TEST_F(l2_ctc_loss_test, test_ctc_loss_output_is_null)
{
    auto logProbs = TensorDesc({T, N, C}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto targets = TensorDesc({N, S}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto inputLengths = IntArrayDesc(vector<int64_t>{T, T, T, T});
    auto targetLengths = IntArrayDesc(vector<int64_t>{S, S, S, S});
    int64_t blank = 0;
    bool zeroInfinity = false;
    auto negLogLikelihoodOut = nullptr;
    auto logAlphaOut = TensorDesc({N, T, LOGALPHA_X}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnCtcLoss, INPUT(logProbs, targets, inputLengths, targetLengths, blank, zeroInfinity),
        OUTPUT(negLogLikelihoodOut, logAlphaOut));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// input is null
TEST_F(l2_ctc_loss_test, test_ctc_loss_input_is_null)
{
    auto logProbs = nullptr;
    auto targets = TensorDesc({N, S}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto inputLengths = IntArrayDesc(vector<int64_t>{T, T, T, T});
    auto targetLengths = IntArrayDesc(vector<int64_t>{S, S, S, S});
    int64_t blank = 0;
    bool zeroInfinity = false;
    auto negLogLikelihoodOut = TensorDesc({N}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto logAlphaOut = TensorDesc({N, T, LOGALPHA_X}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnCtcLoss, INPUT(logProbs, targets, inputLengths, targetLengths, blank, zeroInfinity),
        OUTPUT(negLogLikelihoodOut, logAlphaOut));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// logProbs dtype不在范围内
TEST_F(l2_ctc_loss_test, test_ctc_loss_logprobs_dtype_error)
{
    auto logProbs = TensorDesc({T, N, C}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto targets = TensorDesc({N, S}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto inputLengths = IntArrayDesc(vector<int64_t>{T, T, T, T});
    auto targetLengths = IntArrayDesc(vector<int64_t>{S, S, S, S});
    int64_t blank = 0;
    bool zeroInfinity = false;
    auto negLogLikelihoodOut = TensorDesc({N}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto logAlphaOut = TensorDesc({N, T, LOGALPHA_X}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnCtcLoss, INPUT(logProbs, targets, inputLengths, targetLengths, blank, zeroInfinity),
        OUTPUT(negLogLikelihoodOut, logAlphaOut));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// negLogLikelihood dtype不在范围内
TEST_F(l2_ctc_loss_test, test_ctc_loss_negLogLikelihood_dtype_error)
{
    auto logProbs = TensorDesc({T, N, C}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto targets = TensorDesc({N, S}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto inputLengths = IntArrayDesc(vector<int64_t>{T, T, T, T});
    auto targetLengths = IntArrayDesc(vector<int64_t>{S, S, S, S});
    int64_t blank = 0;
    bool zeroInfinity = false;
    auto negLogLikelihoodOut = TensorDesc({N}, ACL_INT8, ACL_FORMAT_ND);
    auto logAlphaOut = TensorDesc({N, T, LOGALPHA_X}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnCtcLoss, INPUT(logProbs, targets, inputLengths, targetLengths, blank, zeroInfinity),
        OUTPUT(negLogLikelihoodOut, logAlphaOut));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// logAlpha dtype不在范围内
TEST_F(l2_ctc_loss_test, test_ctc_loss_logAlpha_dtype_error)
{
    auto logProbs = TensorDesc({T, N, C}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto targets = TensorDesc({N, S}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto inputLengths = IntArrayDesc(vector<int64_t>{T, T, T, T});
    auto targetLengths = IntArrayDesc(vector<int64_t>{S, S, S, S});
    int64_t blank = 0;
    bool zeroInfinity = false;
    auto negLogLikelihoodOut = TensorDesc({N}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto logAlphaOut = TensorDesc({N, T, LOGALPHA_X}, ACL_INT8, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnCtcLoss, INPUT(logProbs, targets, inputLengths, targetLengths, blank, zeroInfinity),
        OUTPUT(negLogLikelihoodOut, logAlphaOut));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// targets dtype不在范围内
TEST_F(l2_ctc_loss_test, test_ctc_loss_targets_dtype_error)
{
    auto logProbs = TensorDesc({T, N, C}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto targets = TensorDesc({N, S}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto inputLengths = IntArrayDesc(vector<int64_t>{T, T, T, T});
    auto targetLengths = IntArrayDesc(vector<int64_t>{S, S, S, S});
    int64_t blank = 0;
    bool zeroInfinity = false;
    auto negLogLikelihoodOut = TensorDesc({N}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto logAlphaOut = TensorDesc({N, T, LOGALPHA_X}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnCtcLoss, INPUT(logProbs, targets, inputLengths, targetLengths, blank, zeroInfinity),
        OUTPUT(negLogLikelihoodOut, logAlphaOut));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// logprobs first Dim is 0
TEST_F(l2_ctc_loss_test, test_ctc_loss_logprobs_first_dim_0_error)
{
    auto logProbs = TensorDesc({0, N, C}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto targets = TensorDesc({N, S}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto inputLengths = IntArrayDesc(vector<int64_t>{T, T, T, T});
    auto targetLengths = IntArrayDesc(vector<int64_t>{S, S, S, S});
    int64_t blank = 0;
    bool zeroInfinity = false;
    auto negLogLikelihoodOut = TensorDesc({N}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto logAlphaOut = TensorDesc({N, T, LOGALPHA_X}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnCtcLoss, INPUT(logProbs, targets, inputLengths, targetLengths, blank, zeroInfinity),
        OUTPUT(negLogLikelihoodOut, logAlphaOut));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// logProbs DimNum is not 3
TEST_F(l2_ctc_loss_test, test_ctc_loss_logProbs_dimnum_not_3_error)
{
    auto logProbs = TensorDesc({T, N, C, 2}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto targets = TensorDesc({N, S}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto inputLengths = IntArrayDesc(vector<int64_t>{T, T, T, T});
    auto targetLengths = IntArrayDesc(vector<int64_t>{S, S, S, S});
    int64_t blank = 0;
    bool zeroInfinity = false;
    auto negLogLikelihoodOut = TensorDesc({N}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto logAlphaOut = TensorDesc({N, T, LOGALPHA_X}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnCtcLoss, INPUT(logProbs, targets, inputLengths, targetLengths, blank, zeroInfinity),
        OUTPUT(negLogLikelihoodOut, logAlphaOut));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// blank is greater than C
TEST_F(l2_ctc_loss_test, test_ctc_loss_blank_greater_than_C_error)
{
    auto logProbs = TensorDesc({T, N, C}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto targets = TensorDesc({N, S}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto inputLengths = IntArrayDesc(vector<int64_t>{T, T, T, T});
    auto targetLengths = IntArrayDesc(vector<int64_t>{S, S, S, S});
    int64_t blank = C + 1;
    bool zeroInfinity = false;
    auto negLogLikelihoodOut = TensorDesc({N}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto logAlphaOut = TensorDesc({N, T, LOGALPHA_X}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnCtcLoss, INPUT(logProbs, targets, inputLengths, targetLengths, blank, zeroInfinity),
        OUTPUT(negLogLikelihoodOut, logAlphaOut));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// inputLengths's max value is greater than T
TEST_F(l2_ctc_loss_test, test_ctc_loss_inputLengths_maxvalue_greater_than_T_error)
{
    auto logProbs = TensorDesc({T, N, C}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto targets = TensorDesc({N, S}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto inputLengths = IntArrayDesc(vector<int64_t>{T + 1, T, T, T});
    auto targetLengths = IntArrayDesc(vector<int64_t>{S, S, S, S});
    int64_t blank = 0;
    bool zeroInfinity = false;
    auto negLogLikelihoodOut = TensorDesc({N}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto logAlphaOut = TensorDesc({N, T, LOGALPHA_X}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnCtcLoss, INPUT(logProbs, targets, inputLengths, targetLengths, blank, zeroInfinity),
        OUTPUT(negLogLikelihoodOut, logAlphaOut));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// targetLengths's size is not N
TEST_F(l2_ctc_loss_test, test_ctc_loss_targetLengths_size_not_N_error)
{
    auto logProbs = TensorDesc({T, N, C}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto targets = TensorDesc({N, S}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto inputLengths = IntArrayDesc(vector<int64_t>{T, T, T, T});
    auto targetLengths = IntArrayDesc(vector<int64_t>{S, S, S, S, S});
    int64_t blank = 0;
    bool zeroInfinity = false;
    auto negLogLikelihoodOut = TensorDesc({N}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto logAlphaOut = TensorDesc({N, T, LOGALPHA_X}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnCtcLoss, INPUT(logProbs, targets, inputLengths, targetLengths, blank, zeroInfinity),
        OUTPUT(negLogLikelihoodOut, logAlphaOut));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// targets's shape [N,S] , S < targetMaxLength
TEST_F(l2_ctc_loss_test, test_ctc_loss_target_second_dim_less_than_targetMaxLength_error)
{
    auto logProbs = TensorDesc({T, N, C}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto targets = TensorDesc({N, S}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto inputLengths = IntArrayDesc(vector<int64_t>{T, T, T, T});
    auto targetLengths = IntArrayDesc(vector<int64_t>{S + 1, S, S, S});
    int64_t blank = 0;
    bool zeroInfinity = false;
    auto negLogLikelihoodOut = TensorDesc({N}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto logAlphaOut = TensorDesc({N, T, LOGALPHA_X}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnCtcLoss, INPUT(logProbs, targets, inputLengths, targetLengths, blank, zeroInfinity),
        OUTPUT(negLogLikelihoodOut, logAlphaOut));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// targets's shape [X] , X not equal sum(targetLengths)
TEST_F(l2_ctc_loss_test, test_ctc_loss_target_1_dim_size_not_sumtargetlengths_error)
{
    auto logProbs = TensorDesc({T, N, C}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto targets = TensorDesc({4 * S + 1}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto inputLengths = IntArrayDesc(vector<int64_t>{T, T, T, T});
    auto targetLengths = IntArrayDesc(vector<int64_t>{S, S, S, S});
    int64_t blank = 0;
    bool zeroInfinity = false;
    auto negLogLikelihoodOut = TensorDesc({N}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto logAlphaOut = TensorDesc({N, T, LOGALPHA_X}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnCtcLoss, INPUT(logProbs, targets, inputLengths, targetLengths, blank, zeroInfinity),
        OUTPUT(negLogLikelihoodOut, logAlphaOut));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// targets's shape is not 1 or 2
TEST_F(l2_ctc_loss_test, test_ctc_loss_target_shape_isnot_1or2_error)
{
    auto logProbs = TensorDesc({T, N, C}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto targets = TensorDesc({N, S, 2}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto inputLengths = IntArrayDesc(vector<int64_t>{T, T, T, T});
    auto targetLengths = IntArrayDesc(vector<int64_t>{S, S, S, S});
    int64_t blank = 0;
    bool zeroInfinity = false;
    auto negLogLikelihoodOut = TensorDesc({N}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto logAlphaOut = TensorDesc({N, T, LOGALPHA_X}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnCtcLoss, INPUT(logProbs, targets, inputLengths, targetLengths, blank, zeroInfinity),
        OUTPUT(negLogLikelihoodOut, logAlphaOut));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// negLogLikelihoodOut's shape is not N
TEST_F(l2_ctc_loss_test, test_ctc_loss_outNegLogLikelihood_shape_not_N_error)
{
    auto logProbs = TensorDesc({T, N, C}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto targets = TensorDesc({N, S}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto inputLengths = IntArrayDesc(vector<int64_t>{T, T, T, T});
    auto targetLengths = IntArrayDesc(vector<int64_t>{S, S, S, S});
    int64_t blank = 0;
    bool zeroInfinity = false;
    auto negLogLikelihoodOut = TensorDesc({N + 1}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto logAlphaOut = TensorDesc({N, T, LOGALPHA_X}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnCtcLoss, INPUT(logProbs, targets, inputLengths, targetLengths, blank, zeroInfinity),
        OUTPUT(negLogLikelihoodOut, logAlphaOut));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// logAlphaOut's shape is not [N, T, LOGALPHA_X]
TEST_F(l2_ctc_loss_test, test_ctc_loss_outLogAlpha_shape_error)
{
    auto logProbs = TensorDesc({T, N, C}, ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto targets = TensorDesc({N, S}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto inputLengths = IntArrayDesc(vector<int64_t>{T, T, T, T});
    auto targetLengths = IntArrayDesc(vector<int64_t>{S, S, S, S});
    int64_t blank = 0;
    bool zeroInfinity = false;
    auto negLogLikelihoodOut = TensorDesc({N}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto logAlphaOut = TensorDesc({N, T, LOGALPHA_X + 1}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnCtcLoss, INPUT(logProbs, targets, inputLengths, targetLengths, blank, zeroInfinity),
        OUTPUT(negLogLikelihoodOut, logAlphaOut));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

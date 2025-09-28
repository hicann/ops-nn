/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <vector>
#include <array>
#include "gtest/gtest.h"
#include "level2/aclnn_ctc_loss_backward.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class l2_ctc_loss_backward_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ctc_loss_backward_test SetUp" << std::endl;
  }

  static void TearDownTestCase() { std::cout << "ctc_loss_backward_test TearDown" << std::endl; }
};

int64_t TT = 12;
int64_t NN = 4;
int64_t TEMP_NN = NN;
int64_t CC = 5;
int64_t SS = 7;
int64_t CC_V3 = 20000;

// 正常情況aicore target is bool
TEST_F(l2_ctc_loss_backward_test, test_ctc_loss_backward_float_target_is_bool) {
    auto gradOut = TensorDesc({NN}, ACL_FLOAT, ACL_FORMAT_ND)
                            .ValueRange(-10, 10)
                            .Value(vector<float>{1.0, 1.0, 1.0, 1.0});
    auto logProbs = TensorDesc({TT, NN, CC}, ACL_FLOAT, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);
    auto targets = TensorDesc({NN, SS}, ACL_BOOL, ACL_FORMAT_ND)
                            .ValueRange(-1, 1);
    auto inputLengths = IntArrayDesc(vector<int64_t>{TT, TT, TT, TT});
    auto targetLengths = IntArrayDesc(vector<int64_t>{SS, SS, SS, SS});
    auto negLogLikelihood = TensorDesc({NN}, ACL_FLOAT, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);
    auto logAlpha = TensorDesc({NN, TT, 15}, ACL_FLOAT, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);
    int64_t blank = 0;
    bool zeroInfinity = false;
    auto out = TensorDesc({TT, NN, CC}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnCtcLossBackward, INPUT(gradOut, logProbs, targets, inputLengths, targetLengths, negLogLikelihood, logAlpha, blank, zeroInfinity), OUTPUT(out));
    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// logProbs is empty Tensor
TEST_F(l2_ctc_loss_backward_test, test_ctc_loss_backward_logprobs_is_empty_tensor_normal) {
    NN = 0;
    SS = 0;
    auto gradOut = TensorDesc({NN}, ACL_DOUBLE, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);
    auto logProbs = TensorDesc({TT, NN, CC}, ACL_DOUBLE, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);
    auto targets = TensorDesc({NN, SS}, ACL_INT64, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);
    auto inputLengths = IntArrayDesc(vector<int64_t>{});
    auto targetLengths = IntArrayDesc(vector<int64_t>{});
    auto negLogLikelihood = TensorDesc({NN}, ACL_DOUBLE, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);
    auto logAlpha = TensorDesc({NN, TT, 15}, ACL_DOUBLE, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);
    int64_t blank = 0;
    bool zeroInfinity = false;
    auto out = TensorDesc({TT, NN, CC}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnCtcLossBackward, INPUT(gradOut, logProbs, targets, inputLengths, targetLengths, negLogLikelihood, logAlpha, blank, zeroInfinity), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    NN = TEMP_NN;
    SS = 7;
}

// 正常情況float
TEST_F(l2_ctc_loss_backward_test, test_ctc_loss_backward_float_all_normal) {
    auto gradOut = TensorDesc({NN}, ACL_FLOAT, ACL_FORMAT_ND)
                            .ValueRange(-10, 10)
                            .Value(vector<float>{1.0, 1.0, 1.0, 1.0});

    auto logProbs = TensorDesc({TT, NN, CC}, ACL_FLOAT, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);

    auto targets = TensorDesc({NN, SS}, ACL_INT64, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);

    auto inputLengths = IntArrayDesc(vector<int64_t>{TT, TT, TT, TT});

    auto targetLengths = IntArrayDesc(vector<int64_t>{SS, SS, SS, SS});

    auto negLogLikelihood = TensorDesc({NN}, ACL_FLOAT, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);

    auto logAlpha = TensorDesc({NN, TT, 15}, ACL_FLOAT, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);

    int64_t blank = 0;
    bool zeroInfinity = false;

    auto out = TensorDesc({TT, NN, CC}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCtcLossBackward, INPUT(gradOut, logProbs, targets, inputLengths, targetLengths, negLogLikelihood, logAlpha, blank, zeroInfinity), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    // SAMPLE: precision simulate
    // ut.TestPrecision();
}

// 正常情況v3 float
TEST_F(l2_ctc_loss_backward_test, ascend910B2_test_ctc_loss_backward_float_v3_all_normal) {
    auto gradOut = TensorDesc({NN}, ACL_FLOAT, ACL_FORMAT_ND)
                            .ValueRange(-10, 10)
                            .Value(vector<float>{1.0, 1.0, 1.0, 1.0});

    auto logProbs = TensorDesc({TT, NN, CC_V3}, ACL_FLOAT, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);

    auto targets = TensorDesc({NN, SS}, ACL_INT64, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);

    auto inputLengths = IntArrayDesc(vector<int64_t>{TT, TT, TT, TT});

    auto targetLengths = IntArrayDesc(vector<int64_t>{SS, SS, SS, SS});

    auto negLogLikelihood = TensorDesc({NN}, ACL_FLOAT, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);

    auto logAlpha = TensorDesc({NN, TT, 15}, ACL_FLOAT, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);

    int64_t blank = 0;
    bool zeroInfinity = false;

    auto out = TensorDesc({TT, NN, CC_V3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCtcLossBackward, INPUT(gradOut, logProbs, targets, inputLengths, targetLengths, negLogLikelihood, logAlpha, blank, zeroInfinity), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    // SAMPLE: precision simulate
    // ut.TestPrecision();
}

// 正常情況double
TEST_F(l2_ctc_loss_backward_test, test_ctc_loss_backward_double_all_normal) {
    auto gradOut = TensorDesc({NN}, ACL_DOUBLE, ACL_FORMAT_ND)
                            .ValueRange(-10, 10)
                            .Value(vector<float>{1.0, 1.0, 1.0, 1.0});

    auto logProbs = TensorDesc({TT, NN, CC}, ACL_DOUBLE, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);

    auto targets = TensorDesc({NN, SS}, ACL_INT64, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);

    auto inputLengths = IntArrayDesc(vector<int64_t>{TT, TT, TT, TT});

    auto targetLengths = IntArrayDesc(vector<int64_t>{SS, SS, SS, SS});

    auto negLogLikelihood = TensorDesc({NN}, ACL_DOUBLE, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);

    auto logAlpha = TensorDesc({NN, TT, 15}, ACL_DOUBLE, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);

    int64_t blank = 0;
    bool zeroInfinity = false;

    auto out = TensorDesc({TT, NN, CC}, ACL_DOUBLE, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCtcLossBackward, INPUT(gradOut, logProbs, targets, inputLengths, targetLengths, negLogLikelihood, logAlpha, blank, zeroInfinity), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
    // SAMPLE: precision simulate
    // ut.TestPrecision();
}

// output is null
TEST_F(l2_ctc_loss_backward_test, test_ctc_loss_backward_output_is_null) {
    auto gradOut = TensorDesc({NN}, ACL_DOUBLE, ACL_FORMAT_ND)
                            .ValueRange(-10, 10)
                            .Value(vector<float>{1.0, 1.0, 1.0, 1.0});

    auto logProbs = TensorDesc({TT, NN, CC}, ACL_DOUBLE, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);

    auto targets = TensorDesc({NN, SS}, ACL_INT64, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);

    auto inputLengths = IntArrayDesc(vector<int64_t>{TT, TT, TT, TT});

    auto targetLengths = IntArrayDesc(vector<int64_t>{SS, SS, SS, SS});

    auto negLogLikelihood = TensorDesc({NN}, ACL_DOUBLE, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);

    auto logAlpha = TensorDesc({NN, TT, 15}, ACL_DOUBLE, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);

    int64_t blank = 0;
    bool zeroInfinity = false;

    auto out = nullptr;

    auto ut = OP_API_UT(aclnnCtcLossBackward, INPUT(gradOut, logProbs, targets, inputLengths, targetLengths, negLogLikelihood, logAlpha, blank, zeroInfinity), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// input is null
TEST_F(l2_ctc_loss_backward_test, test_ctc_loss_backward_input_is_null) {
    auto gradOut = nullptr;

    auto logProbs = TensorDesc({TT, NN, CC}, ACL_DOUBLE, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);

    auto targets = TensorDesc({NN, SS}, ACL_INT64, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);

    auto inputLengths = IntArrayDesc(vector<int64_t>{TT, TT, TT, TT});

    auto targetLengths = IntArrayDesc(vector<int64_t>{SS, SS, SS, SS});

    auto negLogLikelihood = TensorDesc({NN}, ACL_DOUBLE, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);

    auto logAlpha = TensorDesc({NN, TT, 15}, ACL_DOUBLE, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);

    int64_t blank = 0;
    bool zeroInfinity = false;

    auto out = TensorDesc({TT, NN, CC}, ACL_DOUBLE, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCtcLossBackward, INPUT(gradOut, logProbs, targets, inputLengths, targetLengths, negLogLikelihood, logAlpha, blank, zeroInfinity), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// gradout dtype不在范围内
TEST_F(l2_ctc_loss_backward_test, test_ctc_loss_backward_gradout_dtype_error) {
    auto gradOut = TensorDesc({NN}, ACL_INT8, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);

    auto logProbs = TensorDesc({TT, NN, CC}, ACL_DOUBLE, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);

    auto targets = TensorDesc({NN, SS}, ACL_INT64, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);

    auto inputLengths = IntArrayDesc(vector<int64_t>{TT, TT, TT, TT});

    auto targetLengths = IntArrayDesc(vector<int64_t>{SS, SS, SS, SS});

    auto negLogLikelihood = TensorDesc({NN}, ACL_DOUBLE, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);

    auto logAlpha = TensorDesc({NN, TT, 15}, ACL_DOUBLE, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);

    int64_t blank = 0;
    bool zeroInfinity = false;

    auto out = TensorDesc({TT, NN, CC}, ACL_DOUBLE, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCtcLossBackward, INPUT(gradOut, logProbs, targets, inputLengths, targetLengths, negLogLikelihood, logAlpha, blank, zeroInfinity), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// logProbs dtype不在范围内
TEST_F(l2_ctc_loss_backward_test, test_ctc_loss_backward_logprobs_dtype_error) {
    auto gradOut = TensorDesc({NN}, ACL_DOUBLE, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);

    auto logProbs = TensorDesc({TT, NN, CC}, ACL_INT8, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);

    auto targets = TensorDesc({NN, SS}, ACL_INT64, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);

    auto inputLengths = IntArrayDesc(vector<int64_t>{TT, TT, TT, TT});

    auto targetLengths = IntArrayDesc(vector<int64_t>{SS, SS, SS, SS});

    auto negLogLikelihood = TensorDesc({NN}, ACL_DOUBLE, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);

    auto logAlpha = TensorDesc({NN, TT, 15}, ACL_DOUBLE, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);

    int64_t blank = 0;
    bool zeroInfinity = false;

    auto out = TensorDesc({TT, NN, CC}, ACL_DOUBLE, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCtcLossBackward, INPUT(gradOut, logProbs, targets, inputLengths, targetLengths, negLogLikelihood, logAlpha, blank, zeroInfinity), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// negLogLikelihood dtype不在范围内
TEST_F(l2_ctc_loss_backward_test, test_ctc_loss_backward_negLogLikelihood_dtype_error) {
    auto gradOut = TensorDesc({NN}, ACL_DOUBLE, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);

    auto logProbs = TensorDesc({TT, NN, CC}, ACL_DOUBLE, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);

    auto targets = TensorDesc({NN, SS}, ACL_INT64, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);

    auto inputLengths = IntArrayDesc(vector<int64_t>{TT, TT, TT, TT});

    auto targetLengths = IntArrayDesc(vector<int64_t>{SS, SS, SS, SS});

    auto negLogLikelihood = TensorDesc({NN}, ACL_INT8, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);

    auto logAlpha = TensorDesc({NN, TT, 15}, ACL_DOUBLE, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);

    int64_t blank = 0;
    bool zeroInfinity = false;

    auto out = TensorDesc({TT, NN, CC}, ACL_DOUBLE, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCtcLossBackward, INPUT(gradOut, logProbs, targets, inputLengths, targetLengths, negLogLikelihood, logAlpha, blank, zeroInfinity), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// logAlpha dtype不在范围内
TEST_F(l2_ctc_loss_backward_test, test_ctc_loss_backward_logAlpha_dtype_error) {
    auto gradOut = TensorDesc({NN}, ACL_DOUBLE, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);

    auto logProbs = TensorDesc({TT, NN, CC}, ACL_DOUBLE, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);

    auto targets = TensorDesc({NN, SS}, ACL_INT64, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);

    auto inputLengths = IntArrayDesc(vector<int64_t>{TT, TT, TT, TT});

    auto targetLengths = IntArrayDesc(vector<int64_t>{SS, SS, SS, SS});

    auto negLogLikelihood = TensorDesc({NN}, ACL_DOUBLE, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);

    auto logAlpha = TensorDesc({NN, TT, 15}, ACL_INT8, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);

    int64_t blank = 0;
    bool zeroInfinity = false;

    auto out = TensorDesc({TT, NN, CC}, ACL_DOUBLE, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCtcLossBackward, INPUT(gradOut, logProbs, targets, inputLengths, targetLengths, negLogLikelihood, logAlpha, blank, zeroInfinity), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// out dtype不在范围内
TEST_F(l2_ctc_loss_backward_test, test_ctc_loss_backward_out_dtype_error) {
    auto gradOut = TensorDesc({NN}, ACL_DOUBLE, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);

    auto logProbs = TensorDesc({TT, NN, CC}, ACL_DOUBLE, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);

    auto targets = TensorDesc({NN, SS}, ACL_INT64, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);

    auto inputLengths = IntArrayDesc(vector<int64_t>{TT, TT, TT, TT});

    auto targetLengths = IntArrayDesc(vector<int64_t>{SS, SS, SS, SS});

    auto negLogLikelihood = TensorDesc({NN}, ACL_DOUBLE, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);

    auto logAlpha = TensorDesc({NN, TT, 15}, ACL_DOUBLE, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);

    int64_t blank = 0;
    bool zeroInfinity = false;

    auto out = TensorDesc({TT, NN, CC}, ACL_INT8, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCtcLossBackward, INPUT(gradOut, logProbs, targets, inputLengths, targetLengths, negLogLikelihood, logAlpha, blank, zeroInfinity), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// targets dtype不在范围内
TEST_F(l2_ctc_loss_backward_test, test_ctc_loss_backward_targets_dtype_error) {
    auto gradOut = TensorDesc({NN}, ACL_DOUBLE, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);

    auto logProbs = TensorDesc({TT, NN, CC}, ACL_DOUBLE, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);

    auto targets = TensorDesc({NN, SS}, ACL_INT8, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);

    auto inputLengths = IntArrayDesc(vector<int64_t>{TT, TT, TT, TT});

    auto targetLengths = IntArrayDesc(vector<int64_t>{SS, SS, SS, SS});

    auto negLogLikelihood = TensorDesc({NN}, ACL_DOUBLE, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);

    auto logAlpha = TensorDesc({NN, TT, 15}, ACL_DOUBLE, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);

    int64_t blank = 0;
    bool zeroInfinity = false;

    auto out = TensorDesc({TT, NN, CC}, ACL_DOUBLE, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCtcLossBackward, INPUT(gradOut, logProbs, targets, inputLengths, targetLengths, negLogLikelihood, logAlpha, blank, zeroInfinity), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// input dtype not same
TEST_F(l2_ctc_loss_backward_test, test_ctc_loss_backward_input_dtype_not_same) {
    auto gradOut = TensorDesc({NN}, ACL_FLOAT, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);

    auto logProbs = TensorDesc({TT, NN, CC}, ACL_DOUBLE, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);

    auto targets = TensorDesc({NN, SS}, ACL_INT32, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);

    auto inputLengths = IntArrayDesc(vector<int64_t>{TT, TT, TT, TT});

    auto targetLengths = IntArrayDesc(vector<int64_t>{SS, SS, SS, SS});

    auto negLogLikelihood = TensorDesc({NN}, ACL_DOUBLE, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);

    auto logAlpha = TensorDesc({NN, TT, 15}, ACL_DOUBLE, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);

    int64_t blank = 0;
    bool zeroInfinity = false;

    auto out = TensorDesc({TT, NN, CC}, ACL_DOUBLE, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCtcLossBackward, INPUT(gradOut, logProbs, targets, inputLengths, targetLengths, negLogLikelihood, logAlpha, blank, zeroInfinity), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// logprobs first Dim is 0
TEST_F(l2_ctc_loss_backward_test, test_ctc_loss_backward_logprobs_first_dim_0_error) {
    auto gradOut = TensorDesc({NN}, ACL_DOUBLE, ACL_FORMAT_ND)
                            .ValueRange(-10, 10)
                            .Value(vector<float>{1.0, 1.0, 1.0, 1.0});

    auto logProbs = TensorDesc({0, NN, CC}, ACL_DOUBLE, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);

    auto targets = TensorDesc({NN, SS}, ACL_INT64, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);

    auto inputLengths = IntArrayDesc(vector<int64_t>{TT, TT, TT, TT});

    auto targetLengths = IntArrayDesc(vector<int64_t>{SS, SS, SS, SS});

    auto negLogLikelihood = TensorDesc({NN}, ACL_DOUBLE, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);

    auto logAlpha = TensorDesc({NN, TT, 15}, ACL_DOUBLE, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);

    int64_t blank = 0;
    bool zeroInfinity = false;

    auto out = TensorDesc({TT, NN, CC}, ACL_DOUBLE, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnCtcLossBackward, INPUT(gradOut, logProbs, targets, inputLengths, targetLengths, negLogLikelihood, logAlpha, blank, zeroInfinity), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// gradOut DimNum is not 1
TEST_F(l2_ctc_loss_backward_test, test_ctc_loss_backward_gradout_dimnum_not_1_error) {
    auto gradOut = TensorDesc({NN,2}, ACL_DOUBLE, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);
    auto logProbs = TensorDesc({TT, NN, CC}, ACL_DOUBLE, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);
    auto targets = TensorDesc({NN, SS}, ACL_INT64, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);
    auto inputLengths = IntArrayDesc(vector<int64_t>{TT, TT, TT, TT});
    auto targetLengths = IntArrayDesc(vector<int64_t>{SS, SS, SS, SS});
    auto negLogLikelihood = TensorDesc({NN}, ACL_DOUBLE, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);
    auto logAlpha = TensorDesc({NN, TT, 15}, ACL_DOUBLE, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);
    int64_t blank = 0;
    bool zeroInfinity = false;
    auto out = TensorDesc({TT, NN, CC}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnCtcLossBackward, INPUT(gradOut, logProbs, targets, inputLengths, targetLengths, negLogLikelihood, logAlpha, blank, zeroInfinity), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// negLogLikelihood DimNum is not 1
TEST_F(l2_ctc_loss_backward_test, test_ctc_loss_backward_negLogLikelihood_dimnum_not_1_error) {
    auto gradOut = TensorDesc({NN}, ACL_DOUBLE, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);
    auto logProbs = TensorDesc({TT, NN, CC}, ACL_DOUBLE, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);
    auto targets = TensorDesc({NN, SS}, ACL_INT64, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);
    auto inputLengths = IntArrayDesc(vector<int64_t>{TT, TT, TT, TT});
    auto targetLengths = IntArrayDesc(vector<int64_t>{SS, SS, SS, SS});
    auto negLogLikelihood = TensorDesc({NN,2}, ACL_DOUBLE, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);
    auto logAlpha = TensorDesc({NN, TT, 15}, ACL_DOUBLE, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);
    int64_t blank = 0;
    bool zeroInfinity = false;
    auto out = TensorDesc({TT, NN, CC}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnCtcLossBackward, INPUT(gradOut, logProbs, targets, inputLengths, targetLengths, negLogLikelihood, logAlpha, blank, zeroInfinity), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// logProbs DimNum is not 3
TEST_F(l2_ctc_loss_backward_test, test_ctc_loss_backward_logProbs_dimnum_not_3_error) {
    auto gradOut = TensorDesc({NN}, ACL_DOUBLE, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);
    auto logProbs = TensorDesc({TT, NN, CC, 2}, ACL_DOUBLE, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);
    auto targets = TensorDesc({NN, SS}, ACL_INT64, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);
    auto inputLengths = IntArrayDesc(vector<int64_t>{TT, TT, TT, TT});
    auto targetLengths = IntArrayDesc(vector<int64_t>{SS, SS, SS, SS});
    auto negLogLikelihood = TensorDesc({NN}, ACL_DOUBLE, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);
    auto logAlpha = TensorDesc({NN, TT, 15}, ACL_DOUBLE, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);
    int64_t blank = 0;
    bool zeroInfinity = false;
    auto out = TensorDesc({TT, NN, CC}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnCtcLossBackward, INPUT(gradOut, logProbs, targets, inputLengths, targetLengths, negLogLikelihood, logAlpha, blank, zeroInfinity), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// logAlpha DimNum is not 3
TEST_F(l2_ctc_loss_backward_test, test_ctc_loss_backward_logAlpha_dimnum_not_3_error) {
    auto gradOut = TensorDesc({NN}, ACL_DOUBLE, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);
    auto logProbs = TensorDesc({TT, NN, CC}, ACL_DOUBLE, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);
    auto targets = TensorDesc({NN, SS}, ACL_INT64, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);
    auto inputLengths = IntArrayDesc(vector<int64_t>{TT, TT, TT, TT});
    auto targetLengths = IntArrayDesc(vector<int64_t>{SS, SS, SS, SS});
    auto negLogLikelihood = TensorDesc({NN}, ACL_DOUBLE, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);
    auto logAlpha = TensorDesc({NN, TT, 15, 4}, ACL_DOUBLE, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);
    int64_t blank = 0;
    bool zeroInfinity = false;
    auto out = TensorDesc({TT, NN, CC}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnCtcLossBackward, INPUT(gradOut, logProbs, targets, inputLengths, targetLengths, negLogLikelihood, logAlpha, blank, zeroInfinity), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// blank is greater than CC
TEST_F(l2_ctc_loss_backward_test, test_ctc_loss_backward_blank_greater_than_C_error) {
    auto gradOut = TensorDesc({NN}, ACL_DOUBLE, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);
    auto logProbs = TensorDesc({TT, NN, CC}, ACL_DOUBLE, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);
    auto targets = TensorDesc({NN, SS}, ACL_INT64, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);
    auto inputLengths = IntArrayDesc(vector<int64_t>{TT, TT, TT, TT});
    auto targetLengths = IntArrayDesc(vector<int64_t>{SS, SS, SS, SS});
    auto negLogLikelihood = TensorDesc({NN}, ACL_DOUBLE, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);
    auto logAlpha = TensorDesc({NN, TT, 15}, ACL_DOUBLE, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);
    int64_t blank = CC + 1;
    bool zeroInfinity = false;
    auto out = TensorDesc({TT, NN, CC}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnCtcLossBackward, INPUT(gradOut, logProbs, targets, inputLengths, targetLengths, negLogLikelihood, logAlpha, blank, zeroInfinity), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// inputLengths's max value is greater than TT
TEST_F(l2_ctc_loss_backward_test, test_ctc_loss_backward_inputLengths_maxvalue_greater_than_T_error) {
    auto gradOut = TensorDesc({NN}, ACL_DOUBLE, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);
    auto logProbs = TensorDesc({TT, NN, CC}, ACL_DOUBLE, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);
    auto targets = TensorDesc({NN, SS}, ACL_INT64, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);
    auto inputLengths = IntArrayDesc(vector<int64_t>{TT + 1, TT, TT, TT});
    auto targetLengths = IntArrayDesc(vector<int64_t>{SS, SS, SS, SS});
    auto negLogLikelihood = TensorDesc({NN}, ACL_DOUBLE, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);
    auto logAlpha = TensorDesc({NN, TT, 15}, ACL_DOUBLE, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);
    int64_t blank = 0;
    bool zeroInfinity = false;
    auto out = TensorDesc({TT, NN, CC}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnCtcLossBackward, INPUT(gradOut, logProbs, targets, inputLengths, targetLengths, negLogLikelihood, logAlpha, blank, zeroInfinity), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// targetLengths's size is not NN
TEST_F(l2_ctc_loss_backward_test, test_ctc_loss_backward_targetLengths_size_not_N_error) {
    auto gradOut = TensorDesc({NN}, ACL_DOUBLE, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);
    auto logProbs = TensorDesc({TT, NN, CC}, ACL_DOUBLE, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);
    auto targets = TensorDesc({NN, SS}, ACL_INT64, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);
    auto inputLengths = IntArrayDesc(vector<int64_t>{TT, TT, TT, TT});
    auto targetLengths = IntArrayDesc(vector<int64_t>{SS, SS, SS, SS, SS});
    auto negLogLikelihood = TensorDesc({NN}, ACL_DOUBLE, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);
    auto logAlpha = TensorDesc({NN, TT, 15}, ACL_DOUBLE, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);
    int64_t blank = 0;
    bool zeroInfinity = false;
    auto out = TensorDesc({TT, NN, CC}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnCtcLossBackward, INPUT(gradOut, logProbs, targets, inputLengths, targetLengths, negLogLikelihood, logAlpha, blank, zeroInfinity), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}


// targets's shape [NN,SS] , SS < targetMaxLength
TEST_F(l2_ctc_loss_backward_test, test_ctc_loss_backward_target_second_dim_less_than_targetMaxLength_error) {
    auto gradOut = TensorDesc({NN}, ACL_DOUBLE, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);
    auto logProbs = TensorDesc({TT, NN, CC}, ACL_DOUBLE, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);
    auto targets = TensorDesc({NN, SS}, ACL_INT64, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);
    auto inputLengths = IntArrayDesc(vector<int64_t>{TT, TT, TT, TT});
    auto targetLengths = IntArrayDesc(vector<int64_t>{SS+1, SS, SS, SS});
    auto negLogLikelihood = TensorDesc({NN}, ACL_DOUBLE, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);
    auto logAlpha = TensorDesc({NN, TT, 15}, ACL_DOUBLE, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);
    int64_t blank = 0;
    bool zeroInfinity = false;
    auto out = TensorDesc({TT, NN, CC}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnCtcLossBackward, INPUT(gradOut, logProbs, targets, inputLengths, targetLengths, negLogLikelihood, logAlpha, blank, zeroInfinity), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// targets's shape [X] , X not equal sum(targetLengths)
TEST_F(l2_ctc_loss_backward_test, test_ctc_loss_backward_target_1_dim_size_not_sumtargetlengths_error) {
    auto gradOut = TensorDesc({NN}, ACL_DOUBLE, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);
    auto logProbs = TensorDesc({TT, NN, CC}, ACL_DOUBLE, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);
    auto targets = TensorDesc({4 * SS + 1}, ACL_INT64, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);
    auto inputLengths = IntArrayDesc(vector<int64_t>{TT, TT, TT, TT});
    auto targetLengths = IntArrayDesc(vector<int64_t>{SS, SS, SS, SS});
    auto negLogLikelihood = TensorDesc({NN}, ACL_DOUBLE, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);
    auto logAlpha = TensorDesc({NN, TT, 15}, ACL_DOUBLE, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);
    int64_t blank = 0;
    bool zeroInfinity = false;
    auto out = TensorDesc({TT, NN, CC}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnCtcLossBackward, INPUT(gradOut, logProbs, targets, inputLengths, targetLengths, negLogLikelihood, logAlpha, blank, zeroInfinity), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// targets's shape is not 1 or 2
TEST_F(l2_ctc_loss_backward_test, test_ctc_loss_backward_target_shape_isnot_1or2_error) {
    auto gradOut = TensorDesc({NN}, ACL_DOUBLE, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);
    auto logProbs = TensorDesc({TT, NN, CC}, ACL_DOUBLE, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);
    auto targets = TensorDesc({NN, SS, 2}, ACL_INT64, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);
    auto inputLengths = IntArrayDesc(vector<int64_t>{TT, TT, TT, TT});
    auto targetLengths = IntArrayDesc(vector<int64_t>{SS, SS, SS, SS});
    auto negLogLikelihood = TensorDesc({NN}, ACL_DOUBLE, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);
    auto logAlpha = TensorDesc({NN, TT, 15}, ACL_DOUBLE, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);
    int64_t blank = 0;
    bool zeroInfinity = false;
    auto out = TensorDesc({TT, NN, CC}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnCtcLossBackward, INPUT(gradOut, logProbs, targets, inputLengths, targetLengths, negLogLikelihood, logAlpha, blank, zeroInfinity), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// out's shape is not equal logProbs's shape
TEST_F(l2_ctc_loss_backward_test, test_ctc_loss_backward_out_shape_isnot_equal_logprobs_error) {
    auto gradOut = TensorDesc({NN}, ACL_DOUBLE, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);
    auto logProbs = TensorDesc({TT, NN, CC}, ACL_DOUBLE, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);
    auto targets = TensorDesc({NN, SS}, ACL_INT64, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);
    auto inputLengths = IntArrayDesc(vector<int64_t>{TT, TT, TT, TT});
    auto targetLengths = IntArrayDesc(vector<int64_t>{SS, SS, SS, SS});
    auto negLogLikelihood = TensorDesc({NN}, ACL_DOUBLE, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);
    auto logAlpha = TensorDesc({NN, TT, 15}, ACL_DOUBLE, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);
    int64_t blank = 0;
    bool zeroInfinity = false;
    auto out = TensorDesc({TT, NN, NN}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnCtcLossBackward, INPUT(gradOut, logProbs, targets, inputLengths, targetLengths, negLogLikelihood, logAlpha, blank, zeroInfinity), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// check T shape
TEST_F(l2_ctc_loss_backward_test, test_ctc_loss_backward_out_shape_t_isnot_equal_logalpha_t_error) {
    auto gradOut = TensorDesc({NN}, ACL_DOUBLE, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);
    auto logProbs = TensorDesc({TT, NN, CC}, ACL_DOUBLE, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);
    auto targets = TensorDesc({NN, SS}, ACL_INT64, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);
    auto inputLengths = IntArrayDesc(vector<int64_t>{TT, TT, TT, TT});
    auto targetLengths = IntArrayDesc(vector<int64_t>{SS, SS, SS, SS});
    auto negLogLikelihood = TensorDesc({NN}, ACL_DOUBLE, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);
    auto logAlpha = TensorDesc({NN, TT - 1, 15}, ACL_DOUBLE, ACL_FORMAT_ND)
                            .ValueRange(-10, 10);
    int64_t blank = 0;
    bool zeroInfinity = false;
    auto out = TensorDesc({TT, NN, NN}, ACL_DOUBLE, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnCtcLossBackward, INPUT(gradOut, logProbs, targets, inputLengths, targetLengths, negLogLikelihood, logAlpha, blank, zeroInfinity), OUTPUT(out));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

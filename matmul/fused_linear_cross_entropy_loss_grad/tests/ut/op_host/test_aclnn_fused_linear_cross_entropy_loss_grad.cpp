/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "gtest/gtest.h"
#include "../../../op_host/op_api/aclnn_fused_linear_cross_entropy_loss_grad.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;

class l2_fused_linear_cross_entropy_loss_grad_test : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "l2_fused_linear_cross_entropy_loss_grad_test SetUp" << std::endl;
    }

    static void TearDownTestCase() { std::cout << "l2_fused_linear_cross_entropy_loss_grad_test TearDown" << std::endl; }
};

TEST_F(l2_fused_linear_cross_entropy_loss_grad_test, l2_fused_linear_cross_entropy_loss_grad_test_input_empty_success) {
    int64_t BT = 0;
    int64_t H = 4096;
    int64_t V = 19936;
    vector<int64_t> grad = {BT};
    vector<int64_t> input = {BT, H};
    vector<int64_t> weight = {V, H};
    vector<int64_t> targetMask = {BT};
    vector<int64_t> maskedTarget = {BT};
    vector<int64_t> logitsMaxOptional = {BT};
    vector<int64_t> sumExpLogitsOptional = {BT};
    vector<int64_t> softmaxOptional = {BT, V};
    vector<int64_t> inputGradOut = {BT, H};
    vector<int64_t> weightGradOut = {V, H};

    auto grad_desc = TensorDesc(grad, ACL_FLOAT, ACL_FORMAT_ND);
    auto input_desc = TensorDesc(input, ACL_FLOAT16, ACL_FORMAT_ND);
    auto weight_desc = TensorDesc(weight, ACL_FLOAT16, ACL_FORMAT_ND);
    auto targetMask_desc = TensorDesc(targetMask, ACL_BOOL, ACL_FORMAT_ND);
    auto maskedTarget_desc = TensorDesc(maskedTarget, ACL_INT32, ACL_FORMAT_ND);
    auto logitsMaxOptional_desc = TensorDesc(logitsMaxOptional, ACL_FLOAT, ACL_FORMAT_ND);
    auto sumExpLogitsOptional_desc = TensorDesc(sumExpLogitsOptional, ACL_FLOAT, ACL_FORMAT_ND);
    auto softmaxOptional_desc = TensorDesc(softmaxOptional, ACL_FLOAT, ACL_FORMAT_ND);
    auto inputGradOut_desc = TensorDesc(inputGradOut, ACL_FLOAT16, ACL_FORMAT_ND);
    auto weightGradOut_desc = TensorDesc(weightGradOut, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnFusedLinearCrossEntropyLossGrad,
        INPUT(grad_desc, input_desc, weight_desc, targetMask_desc, maskedTarget_desc, 0.0,
              logitsMaxOptional_desc, sumExpLogitsOptional_desc, softmaxOptional_desc),
        OUTPUT(inputGradOut_desc, weightGradOut_desc)
    );

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B or \
        GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93) {
        EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
    } else {
        // 空tensor直接返回成功，未及平台校验
        EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
    }
}

TEST_F(l2_fused_linear_cross_entropy_loss_grad_test, l2_fused_linear_cross_entropy_loss_grad_test_platform_failed) {
    int64_t BT = 8192;
    int64_t H = 4096;
    int64_t V = 19392;
    vector<int64_t> grad = {BT};
    vector<int64_t> input = {BT, H};
    vector<int64_t> weight = {V, H};
    vector<int64_t> targetMask = {BT};
    vector<int64_t> maskedTarget = {BT};
    vector<int64_t> logitsMaxOptional = {BT};
    vector<int64_t> sumExpLogitsOptional = {BT};
    vector<int64_t> softmaxOptional = {BT, V};
    vector<int64_t> inputGradOut = {BT, H};
    vector<int64_t> weightGradOut = {V, H};

    auto grad_desc = TensorDesc(grad, ACL_FLOAT, ACL_FORMAT_ND);
    auto input_desc = TensorDesc(input, ACL_FLOAT16, ACL_FORMAT_ND);
    auto weight_desc = TensorDesc(weight, ACL_FLOAT16, ACL_FORMAT_ND);
    auto targetMask_desc = TensorDesc(targetMask, ACL_BOOL, ACL_FORMAT_ND);
    auto maskedTarget_desc = TensorDesc(maskedTarget, ACL_INT32, ACL_FORMAT_ND);
    auto logitsMaxOptional_desc = TensorDesc(logitsMaxOptional, ACL_FLOAT, ACL_FORMAT_ND);
    auto sumExpLogitsOptional_desc = TensorDesc(sumExpLogitsOptional, ACL_FLOAT, ACL_FORMAT_ND);
    auto softmaxOptional_desc = TensorDesc(softmaxOptional, ACL_FLOAT, ACL_FORMAT_ND);
    auto inputGradOut_desc = TensorDesc(inputGradOut, ACL_FLOAT16, ACL_FORMAT_ND);
    auto weightGradOut_desc = TensorDesc(weightGradOut, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnFusedLinearCrossEntropyLossGrad,
        INPUT(grad_desc, input_desc, weight_desc, targetMask_desc, maskedTarget_desc, 0.0,
              logitsMaxOptional_desc, sumExpLogitsOptional_desc, softmaxOptional_desc),
        OUTPUT(inputGradOut_desc, weightGradOut_desc)
    );

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B or \
        GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93) {
        EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
    } else {
        EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_RUNTIME_ERROR);
    }
}

TEST_F(l2_fused_linear_cross_entropy_loss_grad_test, l2_fused_linear_cross_entropy_loss_grad_test_softmax_nullptr_failed) {
    int64_t BT = 8192;
    int64_t H = 4096;
    int64_t V = 19392;
    vector<int64_t> grad = {BT};
    vector<int64_t> input = {BT, H};
    vector<int64_t> weight = {V, H};
    vector<int64_t> targetMask = {BT};
    vector<int64_t> maskedTarget = {BT};
    vector<int64_t> logitsMaxOptional = {BT};
    vector<int64_t> sumExpLogitsOptional = {BT};
    vector<int64_t> softmaxOptional = {BT, V};
    vector<int64_t> inputGradOut = {BT, H};
    vector<int64_t> weightGradOut = {V, H};

    auto grad_desc = TensorDesc(grad, ACL_FLOAT, ACL_FORMAT_ND);
    auto input_desc = TensorDesc(input, ACL_FLOAT16, ACL_FORMAT_ND);
    auto weight_desc = TensorDesc(weight, ACL_FLOAT16, ACL_FORMAT_ND);
    auto targetMask_desc = TensorDesc(targetMask, ACL_BOOL, ACL_FORMAT_ND);
    auto maskedTarget_desc = TensorDesc(maskedTarget, ACL_INT32, ACL_FORMAT_ND);
    auto logitsMaxOptional_desc = TensorDesc(logitsMaxOptional, ACL_FLOAT, ACL_FORMAT_ND);
    auto sumExpLogitsOptional_desc = TensorDesc(sumExpLogitsOptional, ACL_FLOAT, ACL_FORMAT_ND);
    auto softmaxOptional_desc = nullptr;
    auto inputGradOut_desc = nullptr;
    auto weightGradOut_desc = nullptr;

    auto ut = OP_API_UT(
        aclnnFusedLinearCrossEntropyLossGrad,
        INPUT(grad_desc, input_desc, weight_desc, targetMask_desc, maskedTarget_desc, 0.0,
              logitsMaxOptional_desc, sumExpLogitsOptional_desc, softmaxOptional_desc),
        OUTPUT(inputGradOut_desc, weightGradOut_desc)
    );

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_fused_linear_cross_entropy_loss_grad_test, l2_fused_linear_cross_entropy_loss_grad_test_labelsmoothing_nonzero_failed) {
    int64_t BT = 8192;
    int64_t H = 4096;
    int64_t V = 19392;
    vector<int64_t> grad = {BT};
    vector<int64_t> input = {BT, H};
    vector<int64_t> weight = {V, H};
    vector<int64_t> targetMask = {BT};
    vector<int64_t> maskedTarget = {BT};
    vector<int64_t> logitsMaxOptional = {BT};
    vector<int64_t> sumExpLogitsOptional = {BT};
    vector<int64_t> softmaxOptional = {BT, V};
    vector<int64_t> inputGradOut = {BT, H};
    vector<int64_t> weightGradOut = {V, H};

    auto grad_desc = TensorDesc(grad, ACL_FLOAT, ACL_FORMAT_ND);
    auto input_desc = TensorDesc(input, ACL_FLOAT16, ACL_FORMAT_ND);
    auto weight_desc = TensorDesc(weight, ACL_FLOAT16, ACL_FORMAT_ND);
    auto targetMask_desc = TensorDesc(targetMask, ACL_BOOL, ACL_FORMAT_ND);
    auto maskedTarget_desc = TensorDesc(maskedTarget, ACL_INT32, ACL_FORMAT_ND);
    auto logitsMaxOptional_desc = TensorDesc(logitsMaxOptional, ACL_FLOAT, ACL_FORMAT_ND);
    auto sumExpLogitsOptional_desc = TensorDesc(sumExpLogitsOptional, ACL_FLOAT, ACL_FORMAT_ND);
    auto softmaxOptional_desc = TensorDesc(softmaxOptional, ACL_FLOAT, ACL_FORMAT_ND);
    auto inputGradOut_desc = TensorDesc(inputGradOut, ACL_FLOAT16, ACL_FORMAT_ND);
    auto weightGradOut_desc = TensorDesc(weightGradOut, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnFusedLinearCrossEntropyLossGrad,
        INPUT(grad_desc, input_desc, weight_desc, targetMask_desc, maskedTarget_desc, 1.0,
              logitsMaxOptional_desc, sumExpLogitsOptional_desc, softmaxOptional_desc),
        OUTPUT(inputGradOut_desc, weightGradOut_desc)
    );

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_fused_linear_cross_entropy_loss_grad_test, l2_fused_linear_cross_entropy_loss_grad_test_input_size0_not_eq_grad_size0_failed) {
    int64_t BT = 8192;
    int64_t H = 4096;
    int64_t V = 19392;
    vector<int64_t> grad = {BT};
    vector<int64_t> input = {BT + 1, H};
    vector<int64_t> weight = {V, H};
    vector<int64_t> targetMask = {BT};
    vector<int64_t> maskedTarget = {BT};
    vector<int64_t> logitsMaxOptional = {BT};
    vector<int64_t> sumExpLogitsOptional = {BT};
    vector<int64_t> softmaxOptional = {BT, V};
    vector<int64_t> inputGradOut = {BT, H};
    vector<int64_t> weightGradOut = {V, H};

    auto grad_desc = TensorDesc(grad, ACL_FLOAT, ACL_FORMAT_ND);
    auto input_desc = TensorDesc(input, ACL_FLOAT16, ACL_FORMAT_ND);
    auto weight_desc = TensorDesc(weight, ACL_FLOAT16, ACL_FORMAT_ND);
    auto targetMask_desc = TensorDesc(targetMask, ACL_BOOL, ACL_FORMAT_ND);
    auto maskedTarget_desc = TensorDesc(maskedTarget, ACL_INT32, ACL_FORMAT_ND);
    auto logitsMaxOptional_desc = TensorDesc(logitsMaxOptional, ACL_FLOAT, ACL_FORMAT_ND);
    auto sumExpLogitsOptional_desc = TensorDesc(sumExpLogitsOptional, ACL_FLOAT, ACL_FORMAT_ND);
    auto softmaxOptional_desc = TensorDesc(softmaxOptional, ACL_FLOAT, ACL_FORMAT_ND);
    auto inputGradOut_desc = TensorDesc(inputGradOut, ACL_FLOAT16, ACL_FORMAT_ND);
    auto weightGradOut_desc = TensorDesc(weightGradOut, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnFusedLinearCrossEntropyLossGrad,
        INPUT(grad_desc, input_desc, weight_desc, targetMask_desc, maskedTarget_desc, 0.0,
              logitsMaxOptional_desc, sumExpLogitsOptional_desc, softmaxOptional_desc),
        OUTPUT(inputGradOut_desc, weightGradOut_desc)
    );

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_fused_linear_cross_entropy_loss_grad_test, l2_fused_linear_cross_entropy_loss_grad_test_weight_size0_lt_512B_failed) {
    int64_t BT = 8192;
    int64_t H = 4096;
    int64_t V = 100;
    vector<int64_t> grad = {BT};
    vector<int64_t> input = {BT, H};
    vector<int64_t> weight = {V, H};
    vector<int64_t> targetMask = {BT};
    vector<int64_t> maskedTarget = {BT};
    vector<int64_t> logitsMaxOptional = {BT};
    vector<int64_t> sumExpLogitsOptional = {BT};
    vector<int64_t> softmaxOptional = {BT, V};
    vector<int64_t> inputGradOut = {BT, H};
    vector<int64_t> weightGradOut = {V, H};

    auto grad_desc = TensorDesc(grad, ACL_FLOAT, ACL_FORMAT_ND);
    auto input_desc = TensorDesc(input, ACL_FLOAT16, ACL_FORMAT_ND);
    auto weight_desc = TensorDesc(weight, ACL_FLOAT16, ACL_FORMAT_ND);
    auto targetMask_desc = TensorDesc(targetMask, ACL_BOOL, ACL_FORMAT_ND);
    auto maskedTarget_desc = TensorDesc(maskedTarget, ACL_INT32, ACL_FORMAT_ND);
    auto logitsMaxOptional_desc = TensorDesc(logitsMaxOptional, ACL_FLOAT, ACL_FORMAT_ND);
    auto sumExpLogitsOptional_desc = TensorDesc(sumExpLogitsOptional, ACL_FLOAT, ACL_FORMAT_ND);
    auto softmaxOptional_desc = TensorDesc(softmaxOptional, ACL_FLOAT, ACL_FORMAT_ND);
    auto inputGradOut_desc = TensorDesc(inputGradOut, ACL_FLOAT16, ACL_FORMAT_ND);
    auto weightGradOut_desc = TensorDesc(weightGradOut, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnFusedLinearCrossEntropyLossGrad,
        INPUT(grad_desc, input_desc, weight_desc, targetMask_desc, maskedTarget_desc, 0.0,
              logitsMaxOptional_desc, sumExpLogitsOptional_desc, softmaxOptional_desc),
        OUTPUT(inputGradOut_desc, weightGradOut_desc)
    );

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_fused_linear_cross_entropy_loss_grad_test, l2_fused_linear_cross_entropy_loss_grad_test_weight_size1_not_eq_input_size1_failed) {
    int64_t BT = 8192;
    int64_t H = 4096;
    int64_t V = 19392;
    vector<int64_t> grad = {BT};
    vector<int64_t> input = {BT, H};
    vector<int64_t> weight = {V, H + 1};
    vector<int64_t> targetMask = {BT};
    vector<int64_t> maskedTarget = {BT};
    vector<int64_t> logitsMaxOptional = {BT};
    vector<int64_t> sumExpLogitsOptional = {BT};
    vector<int64_t> softmaxOptional = {BT, V};
    vector<int64_t> inputGradOut = {BT, H};
    vector<int64_t> weightGradOut = {V, H};

    auto grad_desc = TensorDesc(grad, ACL_FLOAT, ACL_FORMAT_ND);
    auto input_desc = TensorDesc(input, ACL_FLOAT16, ACL_FORMAT_ND);
    auto weight_desc = TensorDesc(weight, ACL_FLOAT16, ACL_FORMAT_ND);
    auto targetMask_desc = TensorDesc(targetMask, ACL_BOOL, ACL_FORMAT_ND);
    auto maskedTarget_desc = TensorDesc(maskedTarget, ACL_INT32, ACL_FORMAT_ND);
    auto logitsMaxOptional_desc = TensorDesc(logitsMaxOptional, ACL_FLOAT, ACL_FORMAT_ND);
    auto sumExpLogitsOptional_desc = TensorDesc(sumExpLogitsOptional, ACL_FLOAT, ACL_FORMAT_ND);
    auto softmaxOptional_desc = TensorDesc(softmaxOptional, ACL_FLOAT, ACL_FORMAT_ND);
    auto inputGradOut_desc = TensorDesc(inputGradOut, ACL_FLOAT16, ACL_FORMAT_ND);
    auto weightGradOut_desc = TensorDesc(weightGradOut, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnFusedLinearCrossEntropyLossGrad,
        INPUT(grad_desc, input_desc, weight_desc, targetMask_desc, maskedTarget_desc, 0.0,
              logitsMaxOptional_desc, sumExpLogitsOptional_desc, softmaxOptional_desc),
        OUTPUT(inputGradOut_desc, weightGradOut_desc)
    );

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_fused_linear_cross_entropy_loss_grad_test, l2_fused_linear_cross_entropy_loss_grad_test_targetMask_size0_not_eq_grad_size0_failed) {
    int64_t BT = 8192;
    int64_t H = 4096;
    int64_t V = 19392;
    vector<int64_t> grad = {BT};
    vector<int64_t> input = {BT, H};
    vector<int64_t> weight = {V, H};
    vector<int64_t> targetMask = {BT + 1};
    vector<int64_t> maskedTarget = {BT};
    vector<int64_t> logitsMaxOptional = {BT};
    vector<int64_t> sumExpLogitsOptional = {BT};
    vector<int64_t> softmaxOptional = {BT, V};
    vector<int64_t> inputGradOut = {BT, H};
    vector<int64_t> weightGradOut = {V, H};

    auto grad_desc = TensorDesc(grad, ACL_FLOAT, ACL_FORMAT_ND);
    auto input_desc = TensorDesc(input, ACL_FLOAT16, ACL_FORMAT_ND);
    auto weight_desc = TensorDesc(weight, ACL_FLOAT16, ACL_FORMAT_ND);
    auto targetMask_desc = TensorDesc(targetMask, ACL_BOOL, ACL_FORMAT_ND);
    auto maskedTarget_desc = TensorDesc(maskedTarget, ACL_INT32, ACL_FORMAT_ND);
    auto logitsMaxOptional_desc = TensorDesc(logitsMaxOptional, ACL_FLOAT, ACL_FORMAT_ND);
    auto sumExpLogitsOptional_desc = TensorDesc(sumExpLogitsOptional, ACL_FLOAT, ACL_FORMAT_ND);
    auto softmaxOptional_desc = TensorDesc(softmaxOptional, ACL_FLOAT, ACL_FORMAT_ND);
    auto inputGradOut_desc = TensorDesc(inputGradOut, ACL_FLOAT16, ACL_FORMAT_ND);
    auto weightGradOut_desc = TensorDesc(weightGradOut, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnFusedLinearCrossEntropyLossGrad,
        INPUT(grad_desc, input_desc, weight_desc, targetMask_desc, maskedTarget_desc, 0.0,
              logitsMaxOptional_desc, sumExpLogitsOptional_desc, softmaxOptional_desc),
        OUTPUT(inputGradOut_desc, weightGradOut_desc)
    );

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_fused_linear_cross_entropy_loss_grad_test, l2_fused_linear_cross_entropy_loss_grad_test_maskedTarget_size0_not_eq_grad_size0_failed) {
    int64_t BT = 8192;
    int64_t H = 4096;
    int64_t V = 19392;
    vector<int64_t> grad = {BT};
    vector<int64_t> input = {BT, H};
    vector<int64_t> weight = {V, H};
    vector<int64_t> targetMask = {BT};
    vector<int64_t> maskedTarget = {BT + 1};
    vector<int64_t> logitsMaxOptional = {BT};
    vector<int64_t> sumExpLogitsOptional = {BT};
    vector<int64_t> softmaxOptional = {BT, V};
    vector<int64_t> inputGradOut = {BT, H};
    vector<int64_t> weightGradOut = {V, H};

    auto grad_desc = TensorDesc(grad, ACL_FLOAT, ACL_FORMAT_ND);
    auto input_desc = TensorDesc(input, ACL_FLOAT16, ACL_FORMAT_ND);
    auto weight_desc = TensorDesc(weight, ACL_FLOAT16, ACL_FORMAT_ND);
    auto targetMask_desc = TensorDesc(targetMask, ACL_BOOL, ACL_FORMAT_ND);
    auto maskedTarget_desc = TensorDesc(maskedTarget, ACL_INT32, ACL_FORMAT_ND);
    auto logitsMaxOptional_desc = TensorDesc(logitsMaxOptional, ACL_FLOAT, ACL_FORMAT_ND);
    auto sumExpLogitsOptional_desc = TensorDesc(sumExpLogitsOptional, ACL_FLOAT, ACL_FORMAT_ND);
    auto softmaxOptional_desc = TensorDesc(softmaxOptional, ACL_FLOAT, ACL_FORMAT_ND);
    auto inputGradOut_desc = TensorDesc(inputGradOut, ACL_FLOAT16, ACL_FORMAT_ND);
    auto weightGradOut_desc = TensorDesc(weightGradOut, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnFusedLinearCrossEntropyLossGrad,
        INPUT(grad_desc, input_desc, weight_desc, targetMask_desc, maskedTarget_desc, 0.0,
              logitsMaxOptional_desc, sumExpLogitsOptional_desc, softmaxOptional_desc),
        OUTPUT(inputGradOut_desc, weightGradOut_desc)
    );

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_fused_linear_cross_entropy_loss_grad_test, l2_fused_linear_cross_entropy_loss_grad_mem_friendly_test_small_case_success) {
    int64_t BT = 8192;
    int64_t H = 4096;
    int64_t V = 19392;
    vector<int64_t> grad = {BT};
    vector<int64_t> input = {BT, H};
    vector<int64_t> weight = {V, H};
    vector<int64_t> targetMask = {BT};
    vector<int64_t> maskedTarget = {BT};
    vector<int64_t> logitsMaxOptional = {BT};
    vector<int64_t> sumExpLogitsOptional = {BT};
    vector<int64_t> softmaxOptional = {BT, V};
    vector<int64_t> inputGradOut = {BT, H};
    vector<int64_t> weightGradOut = {V, H};

    auto grad_desc = TensorDesc(grad, ACL_FLOAT, ACL_FORMAT_ND);
    auto input_desc = TensorDesc(input, ACL_FLOAT16, ACL_FORMAT_ND);
    auto weight_desc = TensorDesc(weight, ACL_FLOAT16, ACL_FORMAT_ND);
    auto targetMask_desc = TensorDesc(targetMask, ACL_BOOL, ACL_FORMAT_ND);
    auto maskedTarget_desc = TensorDesc(maskedTarget, ACL_INT32, ACL_FORMAT_ND);
    auto softmaxOptional_desc = nullptr;
    auto logitsMaxOptional_desc = TensorDesc(logitsMaxOptional, ACL_FLOAT, ACL_FORMAT_ND);
    auto sumExpLogitsOptional_desc = TensorDesc(sumExpLogitsOptional, ACL_FLOAT, ACL_FORMAT_ND);
    auto inputGradOut_desc = TensorDesc(inputGradOut, ACL_FLOAT16, ACL_FORMAT_ND);
    auto weightGradOut_desc = TensorDesc(weightGradOut, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnFusedLinearCrossEntropyLossGrad,
        INPUT(grad_desc, input_desc, weight_desc, targetMask_desc, maskedTarget_desc, 0.0,
              logitsMaxOptional_desc, sumExpLogitsOptional_desc, softmaxOptional_desc),
        OUTPUT(inputGradOut_desc, weightGradOut_desc)
    );

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B or \
        GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93) {
        EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
    } else {
        EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_RUNTIME_ERROR);
    }
}
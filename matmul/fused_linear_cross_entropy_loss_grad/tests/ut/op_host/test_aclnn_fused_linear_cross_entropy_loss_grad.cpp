/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cstdint>
#include "gtest/gtest.h"
#include "../../../op_host/op_api/aclnn_fused_linear_cross_entropy_loss_grad.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;

class l2_fused_linear_cross_entropy_loss_grad_test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "l2_fused_linear_cross_entropy_loss_grad_test SetUp" << std::endl; }

    static void TearDownTestCase()
    {
        std::cout << "l2_fused_linear_cross_entropy_loss_grad_test TearDown" << std::endl;
    }

    uint64_t CheckWorkspace(int64_t bt, int64_t h, int64_t v, aclDataType dtype, aclDataType maskDtype,
                            int64_t maskLength, bool highPerformance, const TensorDesc& inputGradDesc,
                            const TensorDesc& weightGradDesc, aclnnStatus expected)
    {
        auto grad = TensorDesc({bt}, ACL_FLOAT, ACL_FORMAT_ND).ToAclType();
        auto input = TensorDesc({bt, h}, dtype, ACL_FORMAT_ND).ToAclType();
        auto weight = TensorDesc({v, h}, dtype, ACL_FORMAT_ND).ToAclType();
        auto mask = TensorDesc({maskLength}, maskDtype, ACL_FORMAT_ND).ToAclType();
        auto target = TensorDesc({bt}, ACL_INT64, ACL_FORMAT_ND).ToAclType();
        auto logitsMax = TensorDesc({bt}, ACL_FLOAT, ACL_FORMAT_ND).ToAclType();
        auto sumExp = TensorDesc({bt}, ACL_FLOAT, ACL_FORMAT_ND).ToAclType();
        auto softmax = TensorDesc({bt, v}, ACL_FLOAT, ACL_FORMAT_ND).ToAclType();
        auto inputGrad = inputGradDesc.ToAclType();
        auto weightGrad = weightGradDesc.ToAclType();
        uint64_t workspaceSize = UINT64_MAX;
        aclOpExecutor* executor = nullptr;
        auto ret = aclnnFusedLinearCrossEntropyLossGradGetWorkspaceSize(
            grad.get(), input.get(), weight.get(), mask.get(), target.get(), 0.0f,
            highPerformance ? nullptr : logitsMax.get(), highPerformance ? nullptr : sumExp.get(),
            highPerformance ? softmax.get() : nullptr, inputGrad.get(), weightGrad.get(), &workspaceSize, &executor);
        EXPECT_EQ(ret, expected);
        if (expected == ACLNN_SUCCESS && ret == ACLNN_SUCCESS) {
            EXPECT_NE(executor, nullptr);
            EXPECT_NE(workspaceSize, UINT64_MAX);
        }
        delete executor;
        return workspaceSize;
    }
};

TEST_F(l2_fused_linear_cross_entropy_loss_grad_test, empty_bt_schedules_weight_grad_zero)
{
    SocVersionManager versionManager(SocVersion::ASCEND910B);
    for (auto dtype : {ACL_FLOAT16, ACL_BF16}) {
        for (bool highPerformance : {false, true}) {
            SCOPED_TRACE(highPerformance ? "high_performance" : "memory_saving");
            auto inputGrad = TensorDesc({0, 16}, dtype, ACL_FORMAT_ND);
            auto weightGrad = TensorDesc({128, 16}, dtype, ACL_FORMAT_ND);
            CheckWorkspace(0, 16, 128, dtype, ACL_BOOL, 0, highPerformance, inputGrad, weightGrad, ACLNN_SUCCESS);

            auto stridedWeightGrad = TensorDesc({128, 16}, dtype, ACL_FORMAT_ND, {1, 128}, 0, {16, 128});
            CheckWorkspace(0, 16, 128, dtype, ACL_UINT8, 0, highPerformance, inputGrad, stridedWeightGrad,
                           ACLNN_SUCCESS);
        }
    }
}

TEST_F(l2_fused_linear_cross_entropy_loss_grad_test, empty_v_schedules_input_grad_zero)
{
    SocVersionManager versionManager(SocVersion::ASCEND910B);
    // V=0 is supported by the memory-saving API shape checks, not high-performance mode.
    for (auto dtype : {ACL_FLOAT16, ACL_BF16}) {
        auto weightGrad = TensorDesc({0, 16}, dtype, ACL_FORMAT_ND);
        auto inputGrad = TensorDesc({9, 16}, dtype, ACL_FORMAT_ND);
        CheckWorkspace(9, 16, 0, dtype, ACL_BOOL, 9, false, inputGrad, weightGrad, ACLNN_SUCCESS);
        auto stridedInputGrad = TensorDesc({9, 16}, dtype, ACL_FORMAT_ND, {1, 9}, 0, {16, 9});
        CheckWorkspace(9, 16, 0, dtype, ACL_UINT8, 2, false, stridedInputGrad, weightGrad, ACLNN_SUCCESS);
    }
}

TEST_F(l2_fused_linear_cross_entropy_loss_grad_test, both_outputs_empty_remain_noop)
{
    SocVersionManager versionManager(SocVersion::ASCEND910B);
    for (bool highPerformance : {false, true}) {
        auto inputGrad = TensorDesc({9, 0}, ACL_FLOAT16, ACL_FORMAT_ND);
        auto weightGrad = TensorDesc({128, 0}, ACL_FLOAT16, ACL_FORMAT_ND);
        EXPECT_EQ(
            CheckWorkspace(9, 0, 128, ACL_FLOAT16, ACL_UINT8, 2, highPerformance, inputGrad, weightGrad, ACLNN_SUCCESS),
            0U);
    }
    EXPECT_EQ(CheckWorkspace(0, 16, 0, ACL_FLOAT16, ACL_BOOL, 0, false, TensorDesc({0, 16}, ACL_FLOAT16, ACL_FORMAT_ND),
                             TensorDesc({0, 16}, ACL_FLOAT16, ACL_FORMAT_ND), ACLNN_SUCCESS),
              0U);
}

TEST_F(l2_fused_linear_cross_entropy_loss_grad_test, mask_length_checks_precede_empty_fast_path)
{
    SocVersionManager versionManager(SocVersion::ASCEND910B);
    auto inputGrad = TensorDesc({9, 0}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto weightGrad = TensorDesc({128, 0}, ACL_FLOAT16, ACL_FORMAT_ND);
    for (bool highPerformance : {false, true}) {
        for (int64_t length : {2, 3}) {
            EXPECT_EQ(CheckWorkspace(9, 0, 128, ACL_FLOAT16, ACL_UINT8, length, highPerformance, inputGrad, weightGrad,
                                     ACLNN_SUCCESS),
                      0U);
        }
        CheckWorkspace(9, 0, 128, ACL_FLOAT16, ACL_UINT8, 1, highPerformance, inputGrad, weightGrad,
                       ACLNN_ERR_PARAM_INVALID);
        CheckWorkspace(9, 0, 128, ACL_FLOAT16, ACL_BOOL, 8, highPerformance, inputGrad, weightGrad,
                       ACLNN_ERR_PARAM_INVALID);
        CheckWorkspace(9, 0, 128, ACL_FLOAT16, ACL_BOOL, 10, highPerformance, inputGrad, weightGrad,
                       ACLNN_ERR_PARAM_INVALID);
        EXPECT_EQ(
            CheckWorkspace(9, 0, 128, ACL_FLOAT16, ACL_BOOL, 9, highPerformance, inputGrad, weightGrad, ACLNN_SUCCESS),
            0U);
    }
}

TEST_F(l2_fused_linear_cross_entropy_loss_grad_test, empty_input_still_validates_output_shape_and_dtype)
{
    SocVersionManager versionManager(SocVersion::ASCEND910B);
    auto inputGrad = TensorDesc({0, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    // The old documentation incorrectly described weightGradOut as [V, BT].
    CheckWorkspace(0, 16, 128, ACL_FLOAT16, ACL_BOOL, 0, true, inputGrad,
                   TensorDesc({128, 0}, ACL_FLOAT16, ACL_FORMAT_ND), ACLNN_ERR_PARAM_INVALID);
    CheckWorkspace(0, 16, 128, ACL_FLOAT16, ACL_BOOL, 0, true, inputGrad,
                   TensorDesc({128, 16}, ACL_BF16, ACL_FORMAT_ND), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_fused_linear_cross_entropy_loss_grad_test, small_v_is_restricted_only_in_high_performance_mode)
{
    SocVersionManager versionManager(SocVersion::ASCEND910B);
    auto inputGrad = TensorDesc({0, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto weightGrad = TensorDesc({127, 16}, ACL_FLOAT16, ACL_FORMAT_ND);
    CheckWorkspace(0, 16, 127, ACL_FLOAT16, ACL_BOOL, 0, false, inputGrad, weightGrad, ACLNN_SUCCESS);
    CheckWorkspace(0, 16, 127, ACL_FLOAT16, ACL_BOOL, 0, true, inputGrad, weightGrad, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_fused_linear_cross_entropy_loss_grad_test, l2_fused_linear_cross_entropy_loss_grad_test_input_empty_success)
{
    SocVersionManager versionManager(SocVersion::ASCEND910B);
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

    auto ut = OP_API_UT(aclnnFusedLinearCrossEntropyLossGrad,
                        INPUT(grad_desc, input_desc, weight_desc, targetMask_desc, maskedTarget_desc, 0.0,
                              logitsMaxOptional_desc, sumExpLogitsOptional_desc, softmaxOptional_desc),
                        OUTPUT(inputGradOut_desc, weightGradOut_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_fused_linear_cross_entropy_loss_grad_test, l2_fused_linear_cross_entropy_loss_grad_test_ascend950_success)
{
    SocVersionManager versionManager(SocVersion::ASCEND950);
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

    auto ut = OP_API_UT(aclnnFusedLinearCrossEntropyLossGrad,
                        INPUT(grad_desc, input_desc, weight_desc, targetMask_desc, maskedTarget_desc, 0.0,
                              logitsMaxOptional_desc, sumExpLogitsOptional_desc, softmaxOptional_desc),
                        OUTPUT(inputGradOut_desc, weightGradOut_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
}

TEST_F(l2_fused_linear_cross_entropy_loss_grad_test, l2_fused_linear_cross_entropy_loss_grad_test_platform_failed)
{
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

    auto ut = OP_API_UT(aclnnFusedLinearCrossEntropyLossGrad,
                        INPUT(grad_desc, input_desc, weight_desc, targetMask_desc, maskedTarget_desc, 0.0,
                              logitsMaxOptional_desc, sumExpLogitsOptional_desc, softmaxOptional_desc),
                        OUTPUT(inputGradOut_desc, weightGradOut_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B or
        GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93 or
        GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND950) {
        EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
    } else {
        EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_RUNTIME_ERROR);
    }
}

TEST_F(l2_fused_linear_cross_entropy_loss_grad_test,
       l2_fused_linear_cross_entropy_loss_grad_test_softmax_nullptr_failed)
{
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

    auto ut = OP_API_UT(aclnnFusedLinearCrossEntropyLossGrad,
                        INPUT(grad_desc, input_desc, weight_desc, targetMask_desc, maskedTarget_desc, 0.0,
                              logitsMaxOptional_desc, sumExpLogitsOptional_desc, softmaxOptional_desc),
                        OUTPUT(inputGradOut_desc, weightGradOut_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_fused_linear_cross_entropy_loss_grad_test,
       l2_fused_linear_cross_entropy_loss_grad_test_labelsmoothing_nonzero_failed)
{
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

    auto ut = OP_API_UT(aclnnFusedLinearCrossEntropyLossGrad,
                        INPUT(grad_desc, input_desc, weight_desc, targetMask_desc, maskedTarget_desc, 1.0,
                              logitsMaxOptional_desc, sumExpLogitsOptional_desc, softmaxOptional_desc),
                        OUTPUT(inputGradOut_desc, weightGradOut_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_fused_linear_cross_entropy_loss_grad_test,
       l2_fused_linear_cross_entropy_loss_grad_test_input_size0_not_eq_grad_size0_failed)
{
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

    auto ut = OP_API_UT(aclnnFusedLinearCrossEntropyLossGrad,
                        INPUT(grad_desc, input_desc, weight_desc, targetMask_desc, maskedTarget_desc, 0.0,
                              logitsMaxOptional_desc, sumExpLogitsOptional_desc, softmaxOptional_desc),
                        OUTPUT(inputGradOut_desc, weightGradOut_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_fused_linear_cross_entropy_loss_grad_test,
       l2_fused_linear_cross_entropy_loss_grad_test_weight_size0_lt_512B_failed)
{
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

    auto ut = OP_API_UT(aclnnFusedLinearCrossEntropyLossGrad,
                        INPUT(grad_desc, input_desc, weight_desc, targetMask_desc, maskedTarget_desc, 0.0,
                              logitsMaxOptional_desc, sumExpLogitsOptional_desc, softmaxOptional_desc),
                        OUTPUT(inputGradOut_desc, weightGradOut_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_fused_linear_cross_entropy_loss_grad_test,
       l2_fused_linear_cross_entropy_loss_grad_test_weight_size1_not_eq_input_size1_failed)
{
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

    auto ut = OP_API_UT(aclnnFusedLinearCrossEntropyLossGrad,
                        INPUT(grad_desc, input_desc, weight_desc, targetMask_desc, maskedTarget_desc, 0.0,
                              logitsMaxOptional_desc, sumExpLogitsOptional_desc, softmaxOptional_desc),
                        OUTPUT(inputGradOut_desc, weightGradOut_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_fused_linear_cross_entropy_loss_grad_test,
       l2_fused_linear_cross_entropy_loss_grad_test_targetMask_size0_not_eq_grad_size0_failed)
{
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

    auto ut = OP_API_UT(aclnnFusedLinearCrossEntropyLossGrad,
                        INPUT(grad_desc, input_desc, weight_desc, targetMask_desc, maskedTarget_desc, 0.0,
                              logitsMaxOptional_desc, sumExpLogitsOptional_desc, softmaxOptional_desc),
                        OUTPUT(inputGradOut_desc, weightGradOut_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_fused_linear_cross_entropy_loss_grad_test,
       l2_fused_linear_cross_entropy_loss_grad_test_maskedTarget_size0_not_eq_grad_size0_failed)
{
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

    auto ut = OP_API_UT(aclnnFusedLinearCrossEntropyLossGrad,
                        INPUT(grad_desc, input_desc, weight_desc, targetMask_desc, maskedTarget_desc, 0.0,
                              logitsMaxOptional_desc, sumExpLogitsOptional_desc, softmaxOptional_desc),
                        OUTPUT(inputGradOut_desc, weightGradOut_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_fused_linear_cross_entropy_loss_grad_test,
       l2_fused_linear_cross_entropy_loss_grad_mem_friendly_test_small_case_success)
{
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

    auto ut = OP_API_UT(aclnnFusedLinearCrossEntropyLossGrad,
                        INPUT(grad_desc, input_desc, weight_desc, targetMask_desc, maskedTarget_desc, 0.0,
                              logitsMaxOptional_desc, sumExpLogitsOptional_desc, softmaxOptional_desc),
                        OUTPUT(inputGradOut_desc, weightGradOut_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B or
        GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93 or
        GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND950) {
        EXPECT_EQ(getWorkspaceResult, ACLNN_SUCCESS);
    } else {
        EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_RUNTIME_ERROR);
    }
}

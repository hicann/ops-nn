/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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

#include "../../../op_api/aclnn_batch_norm_backward_reduce.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class l2BnBackwardReduceTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "bn_backward_reduce_test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "bn_backward_reduce_test TearDown" << std::endl; }
};

// *** tensor dtype test ***
// test type: FLOAT/FLOAT32
TEST_F(l2BnBackwardReduceTest, case_bn_backward_recude_for_float_type)
{
    auto gradOutDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-8.0, 8.0);
    auto inputDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-8.0, 8.0);

    auto meanDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{-2, 2, 1});
    auto invstdDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0.125, 0.1, 0.3});
    auto weightDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{3, 3, 3});

    bool inputG = true;
    bool weightG = true;
    bool biasG = true;

    auto sumDyDesc = TensorDesc(meanDesc);
    auto sumDyXmuDesc = TensorDesc(meanDesc);
    auto gradWeightDesc = TensorDesc(invstdDesc);
    auto gradBiasDesc = TensorDesc(meanDesc);

    auto ut = OP_API_UT(aclnnBatchNormReduceBackward,
                        INPUT(gradOutDesc, inputDesc, meanDesc, invstdDesc, weightDesc, inputG, weightG, biasG),
                        OUTPUT(sumDyDesc, sumDyXmuDesc, gradWeightDesc, gradBiasDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// test type: FLOAT16
TEST_F(l2BnBackwardReduceTest, case_bn_backward_recude_for_float16_type)
{
    auto gradOutDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-8.0, 8.0);
    auto inputDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-8.0, 8.0);

    auto meanDesc = TensorDesc({3}, ACL_FLOAT16, ACL_FORMAT_ND).Value(vector<float>{-2, 2, 1});
    auto invstdDesc = TensorDesc({3}, ACL_FLOAT16, ACL_FORMAT_ND).Value(vector<float>{0.125, 0.1, 0.3});
    auto weightDesc = TensorDesc({3}, ACL_FLOAT16, ACL_FORMAT_ND).Value(vector<float>{3, 3, 3});

    bool inputG = true;
    bool weightG = true;
    bool biasG = true;

    auto sumDyDesc = TensorDesc(meanDesc);
    auto sumDyXmuDesc = TensorDesc(meanDesc);
    auto gradWeightDesc = TensorDesc(invstdDesc);
    auto gradBiasDesc = TensorDesc(meanDesc);

    auto ut = OP_API_UT(aclnnBatchNormReduceBackward,
                        INPUT(gradOutDesc, inputDesc, meanDesc, invstdDesc, weightDesc, inputG, weightG, biasG),
                        OUTPUT(sumDyDesc, sumDyXmuDesc, gradWeightDesc, gradBiasDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// test type: BFLOAT16
TEST_F(l2BnBackwardReduceTest, case_bn_backward_recude_for_bfloat16_type)
{
    auto gradOutDesc = TensorDesc({2, 3, 1, 4}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-8.0, 8.0);
    auto inputDesc = TensorDesc({2, 3, 1, 4}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-8.0, 8.0);

    auto meanDesc = TensorDesc({3}, ACL_BF16, ACL_FORMAT_ND).Value(vector<float>{-2, 2, 1});
    auto invstdDesc = TensorDesc({3}, ACL_BF16, ACL_FORMAT_ND).Value(vector<float>{0.125, 0.1, 0.3});
    auto weightDesc = TensorDesc({3}, ACL_BF16, ACL_FORMAT_ND).Value(vector<float>{3, 3, 3});

    bool inputG = true;
    bool weightG = true;
    bool biasG = true;

    auto sumDyDesc = TensorDesc(meanDesc);
    auto sumDyXmuDesc = TensorDesc(meanDesc);
    auto gradWeightDesc = TensorDesc(invstdDesc);
    auto gradBiasDesc = TensorDesc(meanDesc);

    auto ut = OP_API_UT(aclnnBatchNormReduceBackward,
                        INPUT(gradOutDesc, inputDesc, meanDesc, invstdDesc, weightDesc, inputG, weightG, biasG),
                        OUTPUT(sumDyDesc, sumDyXmuDesc, gradWeightDesc, gradBiasDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// test type: BFLOAT16
TEST_F(l2BnBackwardReduceTest, Ascend910B2_case_bn_backward_recude_for_bfloat16_type)
{
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B) {
        auto gradOutDesc = TensorDesc({2, 3, 1, 4}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-8.0, 8.0);
        auto inputDesc = TensorDesc({2, 3, 1, 4}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-8.0, 8.0);

        auto meanDesc = TensorDesc({3}, ACL_BF16, ACL_FORMAT_ND).Value(vector<float>{-2, 2, 1});
        auto invstdDesc = TensorDesc({3}, ACL_BF16, ACL_FORMAT_ND).Value(vector<float>{0.125, 0.1, 0.3});
        auto weightDesc = TensorDesc({3}, ACL_BF16, ACL_FORMAT_ND).Value(vector<float>{3, 3, 3});

        bool inputG = true;
        bool weightG = true;
        bool biasG = true;

        auto sumDyDesc = TensorDesc(meanDesc);
        auto sumDyXmuDesc = TensorDesc(meanDesc);
        auto gradWeightDesc = TensorDesc(invstdDesc);
        auto gradBiasDesc = TensorDesc(meanDesc);

        auto ut = OP_API_UT(aclnnBatchNormReduceBackward,
                            INPUT(gradOutDesc, inputDesc, meanDesc, invstdDesc, weightDesc, inputG, weightG, biasG),
                            OUTPUT(sumDyDesc, sumDyXmuDesc, gradWeightDesc, gradBiasDesc));

        uint64_t workspace_size = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        EXPECT_EQ(aclRet, ACL_SUCCESS);
    }
}

// test invalid dtype
TEST_F(l2BnBackwardReduceTest, case_bn_backward_recude_for_invalid_type)
{
    auto gradOutDesc = TensorDesc({2, 3, 1, 4}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-8.0, 8.0);
    auto inputDesc = TensorDesc({2, 3, 1, 4}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-8.0, 8.0);

    auto meanDesc = TensorDesc({3}, ACL_FLOAT16, ACL_FORMAT_ND).Value(vector<float>{-2, 2, 1});
    auto invstdDesc = TensorDesc({3}, ACL_FLOAT16, ACL_FORMAT_ND).Value(vector<float>{0.125, 0.1, 0.3});
    auto weightDesc = TensorDesc({3}, ACL_FLOAT16, ACL_FORMAT_ND).Value(vector<float>{3, 3, 3});

    bool inputG = true;
    bool weightG = true;
    bool biasG = true;

    auto sumDyDesc = TensorDesc(meanDesc);
    auto sumDyXmuDesc = TensorDesc(meanDesc);
    auto gradWeightDesc = TensorDesc(invstdDesc);
    auto gradBiasDesc = TensorDesc(meanDesc);

    auto ut = OP_API_UT(aclnnBatchNormReduceBackward,
                        INPUT(gradOutDesc, inputDesc, meanDesc, invstdDesc, weightDesc, inputG, weightG, biasG),
                        OUTPUT(sumDyDesc, sumDyXmuDesc, gradWeightDesc, gradBiasDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// test invalid output dtype, but mask is false
TEST_F(l2BnBackwardReduceTest, case_bn_backward_recude_for_invalid_type_with_false_mask)
{
    auto gradOutDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-8.0, 8.0);
    auto inputDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-8.0, 8.0);

    auto meanDesc = TensorDesc({3}, ACL_FLOAT16, ACL_FORMAT_ND).Value(vector<float>{-2, 2, 1});
    auto invstdDesc = TensorDesc({3}, ACL_FLOAT16, ACL_FORMAT_ND).Value(vector<float>{0.125, 0.1, 0.3});
    auto weightDesc = TensorDesc({3}, ACL_FLOAT16, ACL_FORMAT_ND).Value(vector<float>{3, 3, 3});

    bool inputG = false;
    bool weightG = true;
    bool biasG = true;

    auto sumDyDesc = TensorDesc({3}, ACL_INT32, ACL_FORMAT_ND);
    auto sumDyXmuDesc = TensorDesc(meanDesc);
    auto gradWeightDesc = TensorDesc(invstdDesc);
    auto gradBiasDesc = TensorDesc(meanDesc);

    auto ut = OP_API_UT(aclnnBatchNormReduceBackward,
                        INPUT(gradOutDesc, inputDesc, meanDesc, invstdDesc, weightDesc, inputG, weightG, biasG),
                        OUTPUT(sumDyDesc, sumDyXmuDesc, gradWeightDesc, gradBiasDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// *** tensor format test ***
// test format NCHW
TEST_F(l2BnBackwardReduceTest, case_bn_backward_recude_for_nchw_format)
{
    auto gradOutDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-8.0, 8.0);
    auto inputDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-8.0, 8.0);

    auto meanDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_NCHW).Value(vector<float>{-2, 2, 1});
    auto invstdDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_NCHW).Value(vector<float>{0.125, 0.1, 0.3});
    auto weightDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_NCHW).Value(vector<float>{3, 3, 3});

    bool inputG = true;
    bool weightG = true;
    bool biasG = true;

    auto sumDyDesc = TensorDesc(meanDesc);
    auto sumDyXmuDesc = TensorDesc(meanDesc);
    auto gradWeightDesc = TensorDesc(invstdDesc);
    auto gradBiasDesc = TensorDesc(meanDesc);

    auto ut = OP_API_UT(aclnnBatchNormReduceBackward,
                        INPUT(gradOutDesc, inputDesc, meanDesc, invstdDesc, weightDesc, inputG, weightG, biasG),
                        OUTPUT(sumDyDesc, sumDyXmuDesc, gradWeightDesc, gradBiasDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// test format NHWC
TEST_F(l2BnBackwardReduceTest, case_bn_backward_recude_for_nhwc_format)
{
    auto gradOutDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_NHWC).ValueRange(-8.0, 8.0);
    auto inputDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_NHWC).ValueRange(-8.0, 8.0);

    auto meanDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_NHWC).Value(vector<float>{-2, 2, 1});
    auto invstdDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_NHWC).Value(vector<float>{0.125, 0.1, 0.3});
    auto weightDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_NHWC).Value(vector<float>{3, 3, 3});

    bool inputG = true;
    bool weightG = true;
    bool biasG = true;

    auto sumDyDesc = TensorDesc(meanDesc);
    auto sumDyXmuDesc = TensorDesc(meanDesc);
    auto gradWeightDesc = TensorDesc(invstdDesc);
    auto gradBiasDesc = TensorDesc(meanDesc);

    auto ut = OP_API_UT(aclnnBatchNormReduceBackward,
                        INPUT(gradOutDesc, inputDesc, meanDesc, invstdDesc, weightDesc, inputG, weightG, biasG),
                        OUTPUT(sumDyDesc, sumDyXmuDesc, gradWeightDesc, gradBiasDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// test format HWCN
TEST_F(l2BnBackwardReduceTest, case_bn_backward_recude_for_hwcn_format)
{
    auto gradOutDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_HWCN).ValueRange(-8.0, 8.0);
    auto inputDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_HWCN).ValueRange(-8.0, 8.0);

    auto meanDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_HWCN).Value(vector<float>{-2, 2, 1});
    auto invstdDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_HWCN).Value(vector<float>{0.125, 0.1, 0.3});
    auto weightDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_HWCN).Value(vector<float>{3, 3, 3});

    bool inputG = true;
    bool weightG = true;
    bool biasG = true;

    auto sumDyDesc = TensorDesc(meanDesc);
    auto sumDyXmuDesc = TensorDesc(meanDesc);
    auto gradWeightDesc = TensorDesc(invstdDesc);
    auto gradBiasDesc = TensorDesc(meanDesc);

    auto ut = OP_API_UT(aclnnBatchNormReduceBackward,
                        INPUT(gradOutDesc, inputDesc, meanDesc, invstdDesc, weightDesc, inputG, weightG, biasG),
                        OUTPUT(sumDyDesc, sumDyXmuDesc, gradWeightDesc, gradBiasDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// test format NDHWC
TEST_F(l2BnBackwardReduceTest, case_bn_backward_recude_for_ndhwc_format)
{
    auto gradOutDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_NDHWC).ValueRange(-8.0, 8.0);
    auto inputDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_NDHWC).ValueRange(-8.0, 8.0);

    auto meanDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_NDHWC).Value(vector<float>{-2, 2, 1});
    auto invstdDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_NDHWC).Value(vector<float>{0.125, 0.1, 0.3});
    auto weightDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_NDHWC).Value(vector<float>{3, 3, 3});

    bool inputG = true;
    bool weightG = true;
    bool biasG = true;

    auto sumDyDesc = TensorDesc(meanDesc);
    auto sumDyXmuDesc = TensorDesc(meanDesc);
    auto gradWeightDesc = TensorDesc(invstdDesc);
    auto gradBiasDesc = TensorDesc(meanDesc);

    auto ut = OP_API_UT(aclnnBatchNormReduceBackward,
                        INPUT(gradOutDesc, inputDesc, meanDesc, invstdDesc, weightDesc, inputG, weightG, biasG),
                        OUTPUT(sumDyDesc, sumDyXmuDesc, gradWeightDesc, gradBiasDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// test format NCDHW
TEST_F(l2BnBackwardReduceTest, case_bn_backward_recude_for_ncdhw_format)
{
    auto gradOutDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-8.0, 8.0);
    auto inputDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_NCDHW).ValueRange(-8.0, 8.0);

    auto meanDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_NCDHW).Value(vector<float>{-2, 2, 1});
    auto invstdDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_NCDHW).Value(vector<float>{0.125, 0.1, 0.3});
    auto weightDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_NCDHW).Value(vector<float>{3, 3, 3});

    bool inputG = true;
    bool weightG = true;
    bool biasG = true;

    auto sumDyDesc = TensorDesc(meanDesc);
    auto sumDyXmuDesc = TensorDesc(meanDesc);
    auto gradWeightDesc = TensorDesc(invstdDesc);
    auto gradBiasDesc = TensorDesc(meanDesc);

    auto ut = OP_API_UT(aclnnBatchNormReduceBackward,
                        INPUT(gradOutDesc, inputDesc, meanDesc, invstdDesc, weightDesc, inputG, weightG, biasG),
                        OUTPUT(sumDyDesc, sumDyXmuDesc, gradWeightDesc, gradBiasDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// test invalid format
TEST_F(l2BnBackwardReduceTest, case_bn_backward_recude_for_invalid_format)
{
    auto gradOutDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_NC1HWC0).ValueRange(-8.0, 8.0);
    auto inputDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_NC1HWC0).ValueRange(-8.0, 8.0);

    auto meanDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_NC1HWC0).Value(vector<float>{-2, 2, 1});
    auto invstdDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_NC1HWC0).Value(vector<float>{0.125, 0.1, 0.3});
    auto weightDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_NC1HWC0).Value(vector<float>{3, 3, 3});

    bool inputG = true;
    bool weightG = true;
    bool biasG = true;

    auto sumDyDesc = TensorDesc(meanDesc);
    auto sumDyXmuDesc = TensorDesc(meanDesc);
    auto gradWeightDesc = TensorDesc(invstdDesc);
    auto gradBiasDesc = TensorDesc(meanDesc);

    auto ut = OP_API_UT(aclnnBatchNormReduceBackward,
                        INPUT(gradOutDesc, inputDesc, meanDesc, invstdDesc, weightDesc, inputG, weightG, biasG),
                        OUTPUT(sumDyDesc, sumDyXmuDesc, gradWeightDesc, gradBiasDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// test invalid output format, but mask is false
TEST_F(l2BnBackwardReduceTest, case_bn_backward_recude_for_invalid_format_with_false_mask)
{
    auto gradOutDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-8.0, 8.0);
    auto inputDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-8.0, 8.0);

    auto meanDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{-2, 2, 1});
    auto invstdDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0.125, 0.1, 0.3});
    auto weightDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{3, 3, 3});

    bool inputG = true;
    bool weightG = false;
    bool biasG = true;

    auto sumDyDesc = TensorDesc(meanDesc);
    auto sumDyXmuDesc = TensorDesc(meanDesc);
    auto gradWeightDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_NC1HWC0);
    auto gradBiasDesc = TensorDesc(meanDesc);

    auto ut = OP_API_UT(aclnnBatchNormReduceBackward,
                        INPUT(gradOutDesc, inputDesc, meanDesc, invstdDesc, weightDesc, inputG, weightG, biasG),
                        OUTPUT(sumDyDesc, sumDyXmuDesc, gradWeightDesc, gradBiasDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// *** tensor rank range ***
// empty tensor
TEST_F(l2BnBackwardReduceTest, case_bn_backward_recude_for_empty_input)
{
    auto gradOutDesc = TensorDesc({2, 3, 0, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto inputDesc = TensorDesc({2, 3, 0, 4}, ACL_FLOAT, ACL_FORMAT_ND);

    auto meanDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{-2, 2, 1});
    auto invstdDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0.125, 0.1, 0.3});
    auto weightDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{3, 3, 3});

    bool inputG = true;
    bool weightG = true;
    bool biasG = true;

    auto sumDyDesc = TensorDesc(meanDesc);
    auto sumDyXmuDesc = TensorDesc(meanDesc);
    auto gradWeightDesc = TensorDesc(invstdDesc);
    auto gradBiasDesc = TensorDesc(meanDesc);

    auto ut = OP_API_UT(aclnnBatchNormReduceBackward,
                        INPUT(gradOutDesc, inputDesc, meanDesc, invstdDesc, weightDesc, inputG, weightG, biasG),
                        OUTPUT(sumDyDesc, sumDyXmuDesc, gradWeightDesc, gradBiasDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// test abnormal rank with right boundary dim, nine
TEST_F(l2BnBackwardReduceTest, case_bn_backward_recude_for_gradout_over_rank_range)
{
    auto gradOutDesc = TensorDesc({2, 3, 1, 4, 9, 5, 2, 4, 7}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-8.0, 8.0);
    auto inputDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-8.0, 8.0);

    auto meanDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{-2, 2, 1});
    auto invstdDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0.125, 0.1, 0.3});
    auto weightDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{3, 3, 3});

    bool inputG = true;
    bool weightG = true;
    bool biasG = true;

    auto sumDyDesc = TensorDesc(meanDesc);
    auto sumDyXmuDesc = TensorDesc(meanDesc);
    auto gradWeightDesc = TensorDesc(invstdDesc);
    auto gradBiasDesc = TensorDesc(meanDesc);

    auto ut = OP_API_UT(aclnnBatchNormReduceBackward,
                        INPUT(gradOutDesc, inputDesc, meanDesc, invstdDesc, weightDesc, inputG, weightG, biasG),
                        OUTPUT(sumDyDesc, sumDyXmuDesc, gradWeightDesc, gradBiasDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2BnBackwardReduceTest, case_bn_backward_recude_for_input_over_rank_range)
{
    auto gradOutDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-8.0, 8.0);
    auto inputDesc = TensorDesc({2, 3, 1, 4, 9, 5, 2, 4, 7}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-8.0, 8.0);

    auto meanDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{-2, 2, 1});
    auto invstdDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0.125, 0.1, 0.3});
    auto weightDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{3, 3, 3});

    bool inputG = true;
    bool weightG = true;
    bool biasG = true;

    auto sumDyDesc = TensorDesc(meanDesc);
    auto sumDyXmuDesc = TensorDesc(meanDesc);
    auto gradWeightDesc = TensorDesc(invstdDesc);
    auto gradBiasDesc = TensorDesc(meanDesc);

    auto ut = OP_API_UT(aclnnBatchNormReduceBackward,
                        INPUT(gradOutDesc, inputDesc, meanDesc, invstdDesc, weightDesc, inputG, weightG, biasG),
                        OUTPUT(sumDyDesc, sumDyXmuDesc, gradWeightDesc, gradBiasDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// *** tensor relationship constraint test ***
// test invalid input with diff dtype
TEST_F(l2BnBackwardReduceTest, case_bn_backward_recude_with_diff_dtype)
{
    auto gradOutDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-8.0, 8.0);
    auto inputDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-8.0, 8.0);

    auto meanDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{-2, 2, 1});
    auto invstdDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0.125, 0.1, 0.3});
    auto weightDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{3, 3, 3});

    bool inputG = true;
    bool weightG = true;
    bool biasG = true;

    auto sumDyDesc = TensorDesc(meanDesc);
    auto sumDyXmuDesc = TensorDesc(meanDesc);
    auto gradWeightDesc = TensorDesc(invstdDesc);
    auto gradBiasDesc = TensorDesc(meanDesc);

    auto ut = OP_API_UT(aclnnBatchNormReduceBackward,
                        INPUT(gradOutDesc, inputDesc, meanDesc, invstdDesc, weightDesc, inputG, weightG, biasG),
                        OUTPUT(sumDyDesc, sumDyXmuDesc, gradWeightDesc, gradBiasDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// test invalid input with diff format
TEST_F(l2BnBackwardReduceTest, case_bn_backward_recude_with_diff_format)
{
    auto gradOutDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-8.0, 8.0);
    auto inputDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_NHWC).ValueRange(-8.0, 8.0);

    auto meanDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{-2, 2, 1});
    auto invstdDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0.125, 0.1, 0.3});
    auto weightDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{3, 3, 3});

    bool inputG = true;
    bool weightG = true;
    bool biasG = true;

    auto sumDyDesc = TensorDesc(meanDesc);
    auto sumDyXmuDesc = TensorDesc(meanDesc);
    auto gradWeightDesc = TensorDesc(invstdDesc);
    auto gradBiasDesc = TensorDesc(meanDesc);

    auto ut = OP_API_UT(aclnnBatchNormReduceBackward,
                        INPUT(gradOutDesc, inputDesc, meanDesc, invstdDesc, weightDesc, inputG, weightG, biasG),
                        OUTPUT(sumDyDesc, sumDyXmuDesc, gradWeightDesc, gradBiasDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// test invalid out shape
TEST_F(l2BnBackwardReduceTest, case_bn_backward_recude_out_shape_invalid)
{
    auto gradOutDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-8.0, 8.0);
    auto inputDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-8.0, 8.0);

    auto meanDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{-2, 2, 1});
    auto invstdDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0.125, 0.1, 0.3});
    auto weightDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{3, 3, 3});

    bool inputG = true;
    bool weightG = true;
    bool biasG = true;

    auto sumDyDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{-2, 2, 1});
    auto sumDyXmuDesc = TensorDesc(meanDesc);
    auto gradWeightDesc = TensorDesc(invstdDesc);
    auto gradBiasDesc = TensorDesc(meanDesc);

    auto ut = OP_API_UT(aclnnBatchNormReduceBackward,
                        INPUT(gradOutDesc, inputDesc, meanDesc, invstdDesc, weightDesc, inputG, weightG, biasG),
                        OUTPUT(sumDyDesc, sumDyXmuDesc, gradWeightDesc, gradBiasDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// *** nullptr test ***
// test nullptr input
TEST_F(l2BnBackwardReduceTest, case_bn_backward_recude_for_null_input)
{
    auto inputDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-8.0, 8.0);

    auto meanDesc = TensorDesc({3}, ACL_FLOAT16, ACL_FORMAT_ND).Value(vector<float>{-2, 2, 1});
    auto invstdDesc = TensorDesc({3}, ACL_FLOAT16, ACL_FORMAT_ND).Value(vector<float>{0.125, 0.1, 0.3});
    auto weightDesc = TensorDesc({3}, ACL_FLOAT16, ACL_FORMAT_ND).Value(vector<float>{3, 3, 3});

    bool inputG = true;
    bool weightG = true;
    bool biasG = true;

    auto sumDyDesc = TensorDesc(meanDesc);
    auto sumDyXmuDesc = TensorDesc(meanDesc);
    auto gradWeightDesc = TensorDesc(invstdDesc);
    auto gradBiasDesc = TensorDesc(meanDesc);

    auto ut = OP_API_UT(aclnnBatchNormReduceBackward,
                        INPUT((aclTensor*)nullptr, inputDesc, meanDesc, invstdDesc, weightDesc, inputG, weightG, biasG),
                        OUTPUT(sumDyDesc, sumDyXmuDesc, gradWeightDesc, gradBiasDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// test nullptr output
TEST_F(l2BnBackwardReduceTest, case_bn_backward_recude_for_null_output)
{
    auto gradOutDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_NC1HWC0).ValueRange(-8.0, 8.0);
    auto inputDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-8.0, 8.0);

    auto meanDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{-2, 2, 1});
    auto invstdDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0.125, 0.1, 0.3});
    auto weightDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{3, 3, 3});

    bool inputG = true;
    bool weightG = true;
    bool biasG = true;

    auto sumDyDesc = TensorDesc(meanDesc);
    auto gradWeightDesc = TensorDesc(invstdDesc);
    auto gradBiasDesc = TensorDesc(meanDesc);

    auto ut = OP_API_UT(aclnnBatchNormReduceBackward,
                        INPUT(gradOutDesc, inputDesc, meanDesc, invstdDesc, weightDesc, inputG, weightG, biasG),
                        OUTPUT(sumDyDesc, (aclTensor*)nullptr, gradWeightDesc, gradBiasDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

// *** precision test ***
// test float precision
TEST_F(l2BnBackwardReduceTest, case_bn_backward_recude_precision_for_float_type)
{
    auto gradOutDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 1);
    auto inputDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 1);

    auto meanDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{-2, 2, 1});
    auto invstdDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0.125, 0.1, 0.3});
    auto weightDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{3, 3, 3});

    bool inputG = true;
    bool weightG = true;
    bool biasG = true;

    auto sumDyDesc = TensorDesc(meanDesc);
    auto sumDyXmuDesc = TensorDesc(meanDesc);
    auto gradWeightDesc = TensorDesc(invstdDesc);
    auto gradBiasDesc = TensorDesc(meanDesc);

    auto ut = OP_API_UT(aclnnBatchNormReduceBackward,
                        INPUT(gradOutDesc, inputDesc, meanDesc, invstdDesc, weightDesc, inputG, weightG, biasG),
                        OUTPUT(sumDyDesc, sumDyXmuDesc, gradWeightDesc, gradBiasDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// *** contiguous ***
// test continuity
TEST_F(l2BnBackwardReduceTest, case_bn_backward_recude_for_contiguous)
{
    auto gradOutDesc = TensorDesc({3, 5, 3, 8}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 1);
    auto inputDesc = TensorDesc({3, 5, 3, 8}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1, 1);

    auto meanDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_NCHW).Value(vector<float>{-2, 2, 1, 0, -1});
    auto invstdDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_NCHW).Value(vector<float>{0.125, 0.1, 0.3, 0.05, 0.2});
    auto weightDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_NCHW).Value(vector<float>{3, 3, 3, 3, 3});

    bool inputG = true;
    bool weightG = true;
    bool biasG = true;

    auto sumDyDesc = TensorDesc(meanDesc);
    auto sumDyXmuDesc = TensorDesc(meanDesc);
    auto gradWeightDesc = TensorDesc(invstdDesc);
    auto gradBiasDesc = TensorDesc(meanDesc);

    auto ut = OP_API_UT(aclnnBatchNormReduceBackward,
                        INPUT(gradOutDesc, inputDesc, meanDesc, invstdDesc, weightDesc, inputG, weightG, biasG),
                        OUTPUT(sumDyDesc, sumDyXmuDesc, gradWeightDesc, gradBiasDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// *** issue 7105: mean/invstd shape校验（长度必须与input的channel轴一致） ***
// mean长度小于C: 越界读场景, 修复前静默返回成功并输出错值
TEST_F(l2BnBackwardReduceTest, case_bn_backward_recude_mean_shape_less_than_c)
{
    auto gradOutDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-8.0, 8.0);
    auto inputDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-8.0, 8.0);

    auto meanDesc = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{-2});
    auto invstdDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0.125, 0.1, 0.3});
    auto weightDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{3, 3, 3});

    bool inputG = true;
    bool weightG = true;
    bool biasG = true;

    auto sumDyDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sumDyXmuDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradWeightDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradBiasDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnBatchNormReduceBackward,
                        INPUT(gradOutDesc, inputDesc, meanDesc, invstdDesc, weightDesc, inputG, weightG, biasG),
                        OUTPUT(sumDyDesc, sumDyXmuDesc, gradWeightDesc, gradBiasDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// mean长度大于C: 修复前内核报561103内部错误码
TEST_F(l2BnBackwardReduceTest, case_bn_backward_recude_mean_shape_greater_than_c)
{
    auto gradOutDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-8.0, 8.0);
    auto inputDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-8.0, 8.0);

    auto meanDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{-2, 2, 1, 0});
    auto invstdDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0.125, 0.1, 0.3});
    auto weightDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{3, 3, 3});

    bool inputG = true;
    bool weightG = true;
    bool biasG = true;

    auto sumDyDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sumDyXmuDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradWeightDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradBiasDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnBatchNormReduceBackward,
                        INPUT(gradOutDesc, inputDesc, meanDesc, invstdDesc, weightDesc, inputG, weightG, biasG),
                        OUTPUT(sumDyDesc, sumDyXmuDesc, gradWeightDesc, gradBiasDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// invstd长度与C不一致: 修复前报561103内部错误码
TEST_F(l2BnBackwardReduceTest, case_bn_backward_recude_invstd_shape_not_equal_c)
{
    auto gradOutDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-8.0, 8.0);
    auto inputDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-8.0, 8.0);

    auto meanDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{-2, 2, 1});
    auto invstdDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0.125, 0.1});
    auto weightDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{3, 3, 3});

    bool inputG = true;
    bool weightG = true;
    bool biasG = true;

    auto sumDyDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sumDyXmuDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradWeightDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradBiasDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnBatchNormReduceBackward,
                        INPUT(gradOutDesc, inputDesc, meanDesc, invstdDesc, weightDesc, inputG, weightG, biasG),
                        OUTPUT(sumDyDesc, sumDyXmuDesc, gradWeightDesc, gradBiasDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// *** issue 7106: weight可选参数校验（非空时dtype/format/shape必须合法） ***
// weight dtype非法: 修复前静默返回成功, 后续kernel计算出错值
TEST_F(l2BnBackwardReduceTest, case_bn_backward_recude_weight_invalid_dtype)
{
    auto gradOutDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-8.0, 8.0);
    auto inputDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-8.0, 8.0);

    auto meanDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{-2, 2, 1});
    auto invstdDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0.125, 0.1, 0.3});
    auto weightDesc = TensorDesc({3}, ACL_INT8, ACL_FORMAT_ND).ValueRange(-8, 8);

    bool inputG = true;
    bool weightG = true;
    bool biasG = true;

    auto sumDyDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sumDyXmuDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradWeightDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradBiasDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnBatchNormReduceBackward,
                        INPUT(gradOutDesc, inputDesc, meanDesc, invstdDesc, weightDesc, inputG, weightG, biasG),
                        OUTPUT(sumDyDesc, sumDyXmuDesc, gradWeightDesc, gradBiasDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// weight shape非法(长度C-1): 修复前静默返回成功
TEST_F(l2BnBackwardReduceTest, case_bn_backward_recude_weight_shape_less_than_c)
{
    auto gradOutDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-8.0, 8.0);
    auto inputDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-8.0, 8.0);

    auto meanDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{-2, 2, 1});
    auto invstdDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0.125, 0.1, 0.3});
    auto weightDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{3, 3});

    bool inputG = true;
    bool weightG = true;
    bool biasG = true;

    auto sumDyDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sumDyXmuDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradWeightDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradBiasDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnBatchNormReduceBackward,
                        INPUT(gradOutDesc, inputDesc, meanDesc, invstdDesc, weightDesc, inputG, weightG, biasG),
                        OUTPUT(sumDyDesc, sumDyXmuDesc, gradWeightDesc, gradBiasDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// weight shape非法(长度C+1): 修复前静默返回成功
TEST_F(l2BnBackwardReduceTest, case_bn_backward_recude_weight_shape_greater_than_c)
{
    auto gradOutDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-8.0, 8.0);
    auto inputDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-8.0, 8.0);

    auto meanDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{-2, 2, 1});
    auto invstdDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0.125, 0.1, 0.3});
    auto weightDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{3, 3, 3, 3});

    bool inputG = true;
    bool weightG = true;
    bool biasG = true;

    auto sumDyDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sumDyXmuDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradWeightDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradBiasDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnBatchNormReduceBackward,
                        INPUT(gradOutDesc, inputDesc, meanDesc, invstdDesc, weightDesc, inputG, weightG, biasG),
                        OUTPUT(sumDyDesc, sumDyXmuDesc, gradWeightDesc, gradBiasDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// weight为空指针nullptr: 可选参数, 合法
TEST_F(l2BnBackwardReduceTest, case_bn_backward_recude_weight_nullptr_valid)
{
    auto gradOutDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-8.0, 8.0);
    auto inputDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-8.0, 8.0);

    auto meanDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{-2, 2, 1});
    auto invstdDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0.125, 0.1, 0.3});

    bool inputG = true;
    bool weightG = true;
    bool biasG = true;

    auto sumDyDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sumDyXmuDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradWeightDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradBiasDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnBatchNormReduceBackward,
        INPUT(gradOutDesc, inputDesc, meanDesc, invstdDesc, (aclTensor*)nullptr, inputG, weightG, biasG),
        OUTPUT(sumDyDesc, sumDyXmuDesc, gradWeightDesc, gradBiasDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// weight格式非法(私有格式NC1HWC0): 修复前静默返回成功
TEST_F(l2BnBackwardReduceTest, case_bn_backward_recude_weight_invalid_format)
{
    auto gradOutDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-8.0, 8.0);
    auto inputDesc = TensorDesc({2, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-8.0, 8.0);

    auto meanDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{-2, 2, 1});
    auto invstdDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0.125, 0.1, 0.3});
    auto weightDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_NC1HWC0).Value(vector<float>{3, 3, 3});

    bool inputG = true;
    bool weightG = true;
    bool biasG = true;

    auto sumDyDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sumDyXmuDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradWeightDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradBiasDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnBatchNormReduceBackward,
                        INPUT(gradOutDesc, inputDesc, meanDesc, invstdDesc, weightDesc, inputG, weightG, biasG),
                        OUTPUT(sumDyDesc, sumDyXmuDesc, gradWeightDesc, gradBiasDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// *** issue 7104: 空Tensor输出填零（所有架构一致，不再依赖IsRegbase） ***
// input第0维为0(空Tensor): 输出应为0, 修复前A2上保留调用方缓冲区脏数据
TEST_F(l2BnBackwardReduceTest, case_bn_backward_recude_empty_input_fill_zero)
{
    auto gradOutDesc = TensorDesc({0, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-8.0, 8.0);
    auto inputDesc = TensorDesc({0, 3, 1, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-8.0, 8.0);

    auto meanDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{-2, 2, 1});
    auto invstdDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0.125, 0.1, 0.3});
    auto weightDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{3, 3, 3});

    bool inputG = true;
    bool weightG = true;
    bool biasG = true;

    auto sumDyDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sumDyXmuDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradWeightDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradBiasDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnBatchNormReduceBackward,
                        INPUT(gradOutDesc, inputDesc, meanDesc, invstdDesc, weightDesc, inputG, weightG, biasG),
                        OUTPUT(sumDyDesc, sumDyXmuDesc, gradWeightDesc, gradBiasDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// input中间维为0(空Tensor): 输出应为0
TEST_F(l2BnBackwardReduceTest, case_bn_backward_recude_empty_mid_dim_fill_zero)
{
    auto gradOutDesc = TensorDesc({2, 3, 0, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-8.0, 8.0);
    auto inputDesc = TensorDesc({2, 3, 0, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-8.0, 8.0);

    auto meanDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{-2, 2, 1});
    auto invstdDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0.125, 0.1, 0.3});
    auto weightDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{3, 3, 3});

    bool inputG = true;
    bool weightG = true;
    bool biasG = true;

    auto sumDyDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sumDyXmuDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradWeightDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gradBiasDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnBatchNormReduceBackward,
                        INPUT(gradOutDesc, inputDesc, meanDesc, invstdDesc, weightDesc, inputG, weightG, biasG),
                        OUTPUT(sumDyDesc, sumDyXmuDesc, gradWeightDesc, gradBiasDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

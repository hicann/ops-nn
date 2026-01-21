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

#include "../../../op_api/aclnn_fused_linear_online_max_sum.h"
#include "op_api/op_api_def.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;
using namespace op;

class l2_fused_linear_online_max_sum_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "l2_fused_linear_online_max_sum_test SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "l2_fused_linear_online_max_sum_test TearDown" << endl;
    }
    static void FusedLinearOnlineMaxSumCommonTest(
        vector<vector<int64_t>> shapes, vector<aclDataType> dtypes, int64_t vocabStartIndex,
        int64_t vocabEndIndex, aclnnStatus expect_status)
    {
        auto input_desc = TensorDesc(shapes[0], dtypes[0], ACL_FORMAT_ND);
        auto weight_desc = TensorDesc(shapes[1], dtypes[1], ACL_FORMAT_ND);
        auto target_desc = TensorDesc(shapes[2], dtypes[2], ACL_FORMAT_ND);
        auto logits_max_local_out_desc = TensorDesc(shapes[3], dtypes[3], ACL_FORMAT_ND);
        auto sum_exp_logits_local_out_desc = TensorDesc(shapes[4], dtypes[4], ACL_FORMAT_ND);
        auto predicted_logits_local_out_desc = TensorDesc(shapes[5], dtypes[5], ACL_FORMAT_ND);
        auto target_mask_out_desc = TensorDesc(shapes[6], dtypes[6], ACL_FORMAT_ND);
        auto masked_target_out_desc = TensorDesc(shapes[7], dtypes[7], ACL_FORMAT_ND);
        auto vocab_parallel_logits_out_optional_desc = TensorDesc(shapes[8], dtypes[8], ACL_FORMAT_ND);

        auto ut = OP_API_UT(aclnnFusedLinearOnlineMaxSum,
            INPUT(input_desc, weight_desc, target_desc, vocabStartIndex, vocabEndIndex),
            OUTPUT(logits_max_local_out_desc, sum_exp_logits_local_out_desc, predicted_logits_local_out_desc,
                   target_mask_out_desc, masked_target_out_desc, vocab_parallel_logits_out_optional_desc));

        // SAMPLE: only test GetWorkspaceSize
        uint64_t workspace_size = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        EXPECT_EQ(aclRet, expect_status);
    }
};

TEST_F(l2_fused_linear_online_max_sum_test, l2_fused_linear_online_max_sum_test_input_empty_success)
{
    int64_t BT = 0;
    int64_t H = 4096;
    int64_t V = 19936;
    vector<vector<int64_t>> shapes = {{BT, H}, {V, H}, {BT}, {BT}, {BT}, {BT}, {(BT + 7) / 8}, {BT}, {BT, V}};
    vector<aclDataType> dtypes = {ACL_FLOAT16, ACL_FLOAT16, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
        ACL_UINT8, ACL_INT32, ACL_FLOAT16};
    FusedLinearOnlineMaxSumCommonTest(shapes, dtypes, 0, V, ACL_SUCCESS);
}

TEST_F(l2_fused_linear_online_max_sum_test, l2_fused_linear_online_max_sum_test_weight_empty_success)
{
    int64_t BT = 8192;
    int64_t H = 0;
    int64_t V = 19936;
    vector<vector<int64_t>> shapes = {{BT, H}, {V, H}, {BT}, {BT}, {BT}, {BT}, {(BT + 7) / 8}, {BT}, {BT, V}};
    vector<aclDataType> dtypes = {ACL_FLOAT16, ACL_FLOAT16, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
        ACL_UINT8, ACL_INT32, ACL_FLOAT16};
    FusedLinearOnlineMaxSumCommonTest(shapes, dtypes, 0, V, ACL_SUCCESS);
}

TEST_F(l2_fused_linear_online_max_sum_test, l2_fused_linear_online_max_sum_test_float16_success) {
    int64_t BT = 8192;
    int64_t H = 4096;
    int64_t V = 19936;
    vector<vector<int64_t>> shapes = {{BT, H}, {V, H}, {BT}, {BT}, {BT}, {BT}, {(BT + 7) / 8}, {BT}, {BT, V}};
    vector<aclDataType> dtypes = {ACL_FLOAT16, ACL_FLOAT16, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
        ACL_UINT8, ACL_INT32, ACL_FLOAT16};
    FusedLinearOnlineMaxSumCommonTest(shapes, dtypes, 0, V, ACL_SUCCESS);
}

TEST_F(l2_fused_linear_online_max_sum_test, l2_fused_linear_online_max_sum_test_bfloat16_success) {
    int64_t BT = 8192;
    int64_t H = 4096;
    int64_t V = 19936;
    vector<vector<int64_t>> shapes = {{BT, H}, {V, H}, {BT}, {BT}, {BT}, {BT}, {(BT + 7) / 8}, {BT}, {BT, V}};
    vector<aclDataType> dtypes = {ACL_BF16, ACL_BF16, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
        ACL_UINT8, ACL_INT32, ACL_BF16};
    FusedLinearOnlineMaxSumCommonTest(shapes, dtypes, 0, V, ACL_SUCCESS);
}

TEST_F(l2_fused_linear_online_max_sum_test, l2_fused_linear_online_max_sum_test_input_nullptr_invalid) {
    int64_t BT = 8192;
    int64_t H = 4096;
    int64_t V = 19936;

    auto input_desc = nullptr;
    auto weight_desc = TensorDesc({V, H}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto target_desc = TensorDesc({BT}, ACL_INT32, ACL_FORMAT_ND);
    auto logits_max_local_out_desc = TensorDesc({BT}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sum_exp_logits_local_out_desc = TensorDesc({BT}, ACL_FLOAT, ACL_FORMAT_ND);
    auto predicted_logits_local_out_desc = TensorDesc({BT}, ACL_FLOAT, ACL_FORMAT_ND);
    auto target_mask_out_desc = TensorDesc({(BT + 7) / 8}, ACL_UINT8, ACL_FORMAT_ND);
    auto masked_target_out_desc = TensorDesc({BT}, ACL_INT32, ACL_FORMAT_ND);
    auto vocab_parallel_logits_out_optional_desc = TensorDesc({BT, V}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnFusedLinearOnlineMaxSum,
        INPUT(input_desc, weight_desc, target_desc, 0, V),
        OUTPUT(logits_max_local_out_desc, sum_exp_logits_local_out_desc, predicted_logits_local_out_desc,
               target_mask_out_desc, masked_target_out_desc, vocab_parallel_logits_out_optional_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_fused_linear_online_max_sum_test, l2_fused_linear_online_max_sum_test_weight_nullptr_invalid) {
    int64_t BT = 8192;
    int64_t H = 4096;
    int64_t V = 19936;

    auto input_desc = TensorDesc({BT, H}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto weight_desc = nullptr;
    auto target_desc = TensorDesc({BT}, ACL_INT32, ACL_FORMAT_ND);
    auto logits_max_local_out_desc = TensorDesc({BT}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sum_exp_logits_local_out_desc = TensorDesc({BT}, ACL_FLOAT, ACL_FORMAT_ND);
    auto predicted_logits_local_out_desc = TensorDesc({BT}, ACL_FLOAT, ACL_FORMAT_ND);
    auto target_mask_out_desc = TensorDesc({(BT + 7) / 8}, ACL_UINT8, ACL_FORMAT_ND);
    auto masked_target_out_desc = TensorDesc({BT}, ACL_INT32, ACL_FORMAT_ND);
    auto vocab_parallel_logits_out_optional_desc = TensorDesc({BT, V}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnFusedLinearOnlineMaxSum,
        INPUT(input_desc, weight_desc, target_desc, 0, V),
        OUTPUT(logits_max_local_out_desc, sum_exp_logits_local_out_desc, predicted_logits_local_out_desc,
               target_mask_out_desc, masked_target_out_desc, vocab_parallel_logits_out_optional_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_fused_linear_online_max_sum_test, l2_fused_linear_online_max_sum_test_target_nullptr_invalid) {
    int64_t BT = 8192;
    int64_t H = 4096;
    int64_t V = 19936;

    auto input_desc = TensorDesc({BT, H}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto weight_desc = TensorDesc({V, H}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto target_desc = nullptr;
    auto logits_max_local_out_desc = TensorDesc({BT}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sum_exp_logits_local_out_desc = TensorDesc({BT}, ACL_FLOAT, ACL_FORMAT_ND);
    auto predicted_logits_local_out_desc = TensorDesc({BT}, ACL_FLOAT, ACL_FORMAT_ND);
    auto target_mask_out_desc = TensorDesc({(BT + 7) / 8}, ACL_UINT8, ACL_FORMAT_ND);
    auto masked_target_out_desc = TensorDesc({BT}, ACL_INT32, ACL_FORMAT_ND);
    auto vocab_parallel_logits_out_optional_desc = TensorDesc({BT, V}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnFusedLinearOnlineMaxSum,
        INPUT(input_desc, weight_desc, target_desc, 0, V),
        OUTPUT(logits_max_local_out_desc, sum_exp_logits_local_out_desc, predicted_logits_local_out_desc,
               target_mask_out_desc, masked_target_out_desc, vocab_parallel_logits_out_optional_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_fused_linear_online_max_sum_test, l2_fused_linear_online_max_sum_test_logits_max_local_out_nullptr_invalid) {
    int64_t BT = 8192;
    int64_t H = 4096;
    int64_t V = 19936;

    auto input_desc = TensorDesc({BT, H}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto weight_desc = TensorDesc({V, H}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto target_desc = TensorDesc({BT}, ACL_INT32, ACL_FORMAT_ND);
    auto logits_max_local_out_desc = nullptr;
    auto sum_exp_logits_local_out_desc = TensorDesc({BT}, ACL_FLOAT, ACL_FORMAT_ND);
    auto predicted_logits_local_out_desc = TensorDesc({BT}, ACL_FLOAT, ACL_FORMAT_ND);
    auto target_mask_out_desc = TensorDesc({(BT + 7) / 8}, ACL_UINT8, ACL_FORMAT_ND);
    auto masked_target_out_desc = TensorDesc({BT}, ACL_INT32, ACL_FORMAT_ND);
    auto vocab_parallel_logits_out_optional_desc = TensorDesc({BT, V}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnFusedLinearOnlineMaxSum,
        INPUT(input_desc, weight_desc, target_desc, 0, V),
        OUTPUT(logits_max_local_out_desc, sum_exp_logits_local_out_desc, predicted_logits_local_out_desc,
               target_mask_out_desc, masked_target_out_desc, vocab_parallel_logits_out_optional_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_fused_linear_online_max_sum_test, l2_fused_linear_online_max_sum_test_sum_exp_logits_local_out_nullptr_invalid) {
    int64_t BT = 8192;
    int64_t H = 4096;
    int64_t V = 19936;

    auto input_desc = TensorDesc({BT, H}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto weight_desc = TensorDesc({V, H}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto target_desc = TensorDesc({BT}, ACL_INT32, ACL_FORMAT_ND);
    auto logits_max_local_out_desc = TensorDesc({BT}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sum_exp_logits_local_out_desc = nullptr;
    auto predicted_logits_local_out_desc = TensorDesc({BT}, ACL_FLOAT, ACL_FORMAT_ND);
    auto target_mask_out_desc = TensorDesc({(BT + 7) / 8}, ACL_UINT8, ACL_FORMAT_ND);
    auto masked_target_out_desc = TensorDesc({BT}, ACL_INT32, ACL_FORMAT_ND);
    auto vocab_parallel_logits_out_optional_desc = TensorDesc({BT, V}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnFusedLinearOnlineMaxSum,
        INPUT(input_desc, weight_desc, target_desc, 0, V),
        OUTPUT(logits_max_local_out_desc, sum_exp_logits_local_out_desc, predicted_logits_local_out_desc,
               target_mask_out_desc, masked_target_out_desc, vocab_parallel_logits_out_optional_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_fused_linear_online_max_sum_test, l2_fused_linear_online_max_sum_test_predicted_logits_local_out_nullptr_invalid) {
    int64_t BT = 8192;
    int64_t H = 4096;
    int64_t V = 19936;

    auto input_desc = TensorDesc({BT, H}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto weight_desc = TensorDesc({V, H}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto target_desc = TensorDesc({BT}, ACL_INT32, ACL_FORMAT_ND);
    auto logits_max_local_out_desc = TensorDesc({BT}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sum_exp_logits_local_out_desc = TensorDesc({BT}, ACL_FLOAT, ACL_FORMAT_ND);
    auto predicted_logits_local_out_desc = nullptr;
    auto target_mask_out_desc = TensorDesc({(BT + 7) / 8}, ACL_UINT8, ACL_FORMAT_ND);
    auto masked_target_out_desc = TensorDesc({BT}, ACL_INT32, ACL_FORMAT_ND);
    auto vocab_parallel_logits_out_optional_desc = TensorDesc({BT, V}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnFusedLinearOnlineMaxSum,
        INPUT(input_desc, weight_desc, target_desc, 0, V),
        OUTPUT(logits_max_local_out_desc, sum_exp_logits_local_out_desc, predicted_logits_local_out_desc,
               target_mask_out_desc, masked_target_out_desc, vocab_parallel_logits_out_optional_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}


TEST_F(l2_fused_linear_online_max_sum_test, l2_fused_linear_online_max_sum_test_target_mask_out_nullptr_invalid) {
    int64_t BT = 8192;
    int64_t H = 4096;
    int64_t V = 19936;

    auto input_desc = TensorDesc({BT, H}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto weight_desc = TensorDesc({V, H}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto target_desc = TensorDesc({BT}, ACL_INT32, ACL_FORMAT_ND);
    auto logits_max_local_out_desc = TensorDesc({BT}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sum_exp_logits_local_out_desc = TensorDesc({BT}, ACL_FLOAT, ACL_FORMAT_ND);
    auto predicted_logits_local_out_desc = TensorDesc({BT}, ACL_FLOAT, ACL_FORMAT_ND);
    auto target_mask_out_desc = nullptr;
    auto masked_target_out_desc = TensorDesc({BT}, ACL_INT32, ACL_FORMAT_ND);
    auto vocab_parallel_logits_out_optional_desc = TensorDesc({BT, V}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnFusedLinearOnlineMaxSum,
        INPUT(input_desc, weight_desc, target_desc, 0, V),
        OUTPUT(logits_max_local_out_desc, sum_exp_logits_local_out_desc, predicted_logits_local_out_desc,
               target_mask_out_desc, masked_target_out_desc, vocab_parallel_logits_out_optional_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_fused_linear_online_max_sum_test, l2_fused_linear_online_max_sum_test_masked_target_out_nullptr_invalid) {
    int64_t BT = 8192;
    int64_t H = 4096;
    int64_t V = 19936;

    auto input_desc = TensorDesc({BT, H}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto weight_desc = TensorDesc({V, H}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto target_desc = TensorDesc({BT}, ACL_INT32, ACL_FORMAT_ND);
    auto logits_max_local_out_desc = TensorDesc({BT}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sum_exp_logits_local_out_desc = TensorDesc({BT}, ACL_FLOAT, ACL_FORMAT_ND);
    auto predicted_logits_local_out_desc = TensorDesc({BT}, ACL_FLOAT, ACL_FORMAT_ND);
    auto target_mask_out_desc = TensorDesc({(BT + 7) / 8}, ACL_INT32, ACL_FORMAT_ND);
    auto masked_target_out_desc = nullptr;
    auto vocab_parallel_logits_out_optional_desc = TensorDesc({BT, V}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnFusedLinearOnlineMaxSum,
        INPUT(input_desc, weight_desc, target_desc, 0, V),
        OUTPUT(logits_max_local_out_desc, sum_exp_logits_local_out_desc, predicted_logits_local_out_desc,
               target_mask_out_desc, masked_target_out_desc, vocab_parallel_logits_out_optional_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_fused_linear_online_max_sum_test, l2_fused_linear_online_max_sum_test_vocab_parallel_logits_out_optional_nullptr_success) {
    int64_t BT = 8192;
    int64_t H = 4096;
    int64_t V = 19936;

    auto input_desc = TensorDesc({BT, H}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto weight_desc = TensorDesc({V, H}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto target_desc = TensorDesc({BT}, ACL_INT32, ACL_FORMAT_ND);
    auto logits_max_local_out_desc = TensorDesc({BT}, ACL_FLOAT, ACL_FORMAT_ND);
    auto sum_exp_logits_local_out_desc = TensorDesc({BT}, ACL_FLOAT, ACL_FORMAT_ND);
    auto predicted_logits_local_out_desc = TensorDesc({BT}, ACL_FLOAT, ACL_FORMAT_ND);
    auto target_mask_out_desc = TensorDesc({(BT + 7) / 8}, ACL_UINT8, ACL_FORMAT_ND);
    auto masked_target_out_desc = TensorDesc({BT}, ACL_INT32, ACL_FORMAT_ND);
    auto vocab_parallel_logits_out_optional_desc = nullptr;

    auto ut = OP_API_UT(aclnnFusedLinearOnlineMaxSum,
        INPUT(input_desc, weight_desc, target_desc, 0, V),
        OUTPUT(logits_max_local_out_desc, sum_exp_logits_local_out_desc, predicted_logits_local_out_desc,
               target_mask_out_desc, masked_target_out_desc, vocab_parallel_logits_out_optional_desc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

TEST_F(l2_fused_linear_online_max_sum_test, l2_fused_linear_online_max_sum_test_input_dtype_invalid) {
    int64_t BT = 8192;
    int64_t H = 4096;
    int64_t V = 19936;
    vector<vector<int64_t>> shapes = {{BT, H}, {V, H}, {BT}, {BT}, {BT}, {BT}, {(BT + 7) / 8}, {BT}, {BT, V}};
    vector<aclDataType> dtypes = {ACL_FLOAT, ACL_FLOAT16, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
        ACL_UINT8, ACL_INT32, ACL_FLOAT16};
    FusedLinearOnlineMaxSumCommonTest(shapes, dtypes, 0, V, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_fused_linear_online_max_sum_test, l2_fused_linear_online_max_sum_test_weight_dtype_invalid) {
    int64_t BT = 8192;
    int64_t H = 4096;
    int64_t V = 19936;
    vector<vector<int64_t>> shapes = {{BT, H}, {V, H}, {BT}, {BT}, {BT}, {BT}, {(BT + 7) / 8}, {BT}, {BT, V}};
    vector<aclDataType> dtypes = {ACL_FLOAT16, ACL_FLOAT, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
        ACL_UINT8, ACL_INT32, ACL_FLOAT16};
    FusedLinearOnlineMaxSumCommonTest(shapes, dtypes, 0, V, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_fused_linear_online_max_sum_test, l2_fused_linear_online_max_sum_test_target_dtype_invalid) {
    int64_t BT = 8192;
    int64_t H = 4096;
    int64_t V = 19936;
    vector<vector<int64_t>> shapes = {{BT, H}, {V, H}, {BT}, {BT}, {BT}, {BT}, {(BT + 7) / 8}, {BT}, {BT, V}};
    vector<aclDataType> dtypes = {ACL_FLOAT16, ACL_FLOAT16, ACL_INT8, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
        ACL_UINT8, ACL_INT32, ACL_FLOAT16};
    FusedLinearOnlineMaxSumCommonTest(shapes, dtypes, 0, V, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_fused_linear_online_max_sum_test, l2_fused_linear_online_max_sum_test_logits_max_local_out_dtype_invalid) {
    int64_t BT = 8192;
    int64_t H = 4096;
    int64_t V = 19936;
    vector<vector<int64_t>> shapes = {{BT, H}, {V, H}, {BT}, {BT}, {BT}, {BT}, {(BT + 7) / 8}, {BT}, {BT, V}};
    vector<aclDataType> dtypes = {ACL_FLOAT16, ACL_FLOAT16, ACL_INT32, ACL_FLOAT16, ACL_FLOAT, ACL_FLOAT,
        ACL_UINT8, ACL_INT32, ACL_FLOAT16};
    FusedLinearOnlineMaxSumCommonTest(shapes, dtypes, 0, V, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_fused_linear_online_max_sum_test, l2_fused_linear_online_max_sum_test_sum_exp_logits_local_out_dtype_invalid) {
    int64_t BT = 8192;
    int64_t H = 4096;
    int64_t V = 19936;
    vector<vector<int64_t>> shapes = {{BT, H}, {V, H}, {BT}, {BT}, {BT}, {BT}, {(BT + 7) / 8}, {BT}, {BT, V}};
    vector<aclDataType> dtypes = {ACL_FLOAT16, ACL_FLOAT16, ACL_INT32, ACL_FLOAT, ACL_FLOAT16, ACL_FLOAT,
        ACL_UINT8, ACL_INT32, ACL_FLOAT16};
    FusedLinearOnlineMaxSumCommonTest(shapes, dtypes, 0, V, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_fused_linear_online_max_sum_test, l2_fused_linear_online_max_sum_test_predicted_logtis_local_out_dtype_invalid) {
    int64_t BT = 8192;
    int64_t H = 4096;
    int64_t V = 19936;
    vector<vector<int64_t>> shapes = {{BT, H}, {V, H}, {BT}, {BT}, {BT}, {BT}, {(BT + 7) / 8}, {BT}, {BT, V}};
    vector<aclDataType> dtypes = {ACL_FLOAT16, ACL_FLOAT16, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT16,
        ACL_UINT8, ACL_INT32, ACL_FLOAT16};
    FusedLinearOnlineMaxSumCommonTest(shapes, dtypes, 0, V, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_fused_linear_online_max_sum_test, l2_fused_linear_online_max_sum_test_target_mask_out_dtype_invalid) {
    int64_t BT = 8192;
    int64_t H = 4096;
    int64_t V = 19936;
    vector<vector<int64_t>> shapes = {{BT, H}, {V, H}, {BT}, {BT}, {BT}, {BT}, {(BT + 7) / 8}, {BT}, {BT, V}};
    vector<aclDataType> dtypes = {ACL_FLOAT16, ACL_FLOAT16, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
        ACL_INT8, ACL_INT32, ACL_FLOAT16};
    FusedLinearOnlineMaxSumCommonTest(shapes, dtypes, 0, V, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_fused_linear_online_max_sum_test, l2_fused_linear_online_max_sum_test_masked_target_out_dtype_invalid) {
    int64_t BT = 8192;
    int64_t H = 4096;
    int64_t V = 19936;
    vector<vector<int64_t>> shapes = {{BT, H}, {V, H}, {BT}, {BT}, {BT}, {BT}, {(BT + 7) / 8}, {BT}, {BT, V}};
    vector<aclDataType> dtypes = {ACL_FLOAT16, ACL_FLOAT16, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
        ACL_UINT8, ACL_FLOAT16, ACL_FLOAT16};
    FusedLinearOnlineMaxSumCommonTest(shapes, dtypes, 0, V, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_fused_linear_online_max_sum_test, l2_fused_linear_online_max_sum_test_vocab_parallel_logits_out_optional_dtype_invalid) {
    int64_t BT = 8192;
    int64_t H = 4096;
    int64_t V = 19936;
    vector<vector<int64_t>> shapes = {{BT, H}, {V, H}, {BT}, {BT}, {BT}, {BT}, {(BT + 7) / 8}, {BT}, {BT, V}};
    vector<aclDataType> dtypes = {ACL_FLOAT16, ACL_FLOAT16, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
        ACL_UINT8, ACL_INT32, ACL_FLOAT};
    FusedLinearOnlineMaxSumCommonTest(shapes, dtypes, 0, V, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_fused_linear_online_max_sum_test, l2_fused_linear_online_max_sum_test_input_weight_dtype_not_same) {
    int64_t BT = 8192;
    int64_t H = 4096;
    int64_t V = 19936;
    vector<vector<int64_t>> shapes = {{BT, H}, {V, H}, {BT}, {BT}, {BT}, {BT}, {(BT + 7) / 8}, {BT}, {BT, V}};
    vector<aclDataType> dtypes = {ACL_FLOAT16, ACL_BF16, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
        ACL_UINT8, ACL_INT32, ACL_FLOAT16};
    FusedLinearOnlineMaxSumCommonTest(shapes, dtypes, 0, V, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_fused_linear_online_max_sum_test, l2_fused_linear_online_max_sum_test_input_vocab_parallel_logits_out_optional_dtype_not_same) {
    int64_t BT = 8192;
    int64_t H = 4096;
    int64_t V = 19936;
    vector<vector<int64_t>> shapes = {{BT, H}, {V, H}, {BT}, {BT}, {BT}, {BT}, {(BT + 7) / 8}, {BT}, {BT, V}};
    vector<aclDataType> dtypes = {ACL_FLOAT16, ACL_FLOAT16, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
        ACL_UINT8, ACL_INT32, ACL_BF16};
    FusedLinearOnlineMaxSumCommonTest(shapes, dtypes, 0, V, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_fused_linear_online_max_sum_test, l2_fused_linear_online_max_sum_test_input_dim_invalid) {
    int64_t BT = 8192;
    int64_t H = 4096;
    int64_t V = 19936;
    vector<vector<int64_t>> shapes = {{BT, H, 1}, {V, H}, {BT}, {BT}, {BT}, {BT}, {(BT + 7) / 8}, {BT}, {BT, V}};
    vector<aclDataType> dtypes = {ACL_FLOAT16, ACL_FLOAT16, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
        ACL_UINT8, ACL_INT32, ACL_FLOAT16};
    FusedLinearOnlineMaxSumCommonTest(shapes, dtypes, 0, V, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_fused_linear_online_max_sum_test, l2_fused_linear_online_max_sum_test_weight_dim_invalid) {
    int64_t BT = 8192;
    int64_t H = 4096;
    int64_t V = 19936;
    vector<vector<int64_t>> shapes = {{BT, H}, {V, H, 1}, {BT}, {BT}, {BT}, {BT}, {(BT + 7) / 8}, {BT}, {BT, V}};
    vector<aclDataType> dtypes = {ACL_FLOAT16, ACL_FLOAT16, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
        ACL_UINT8, ACL_INT32, ACL_FLOAT16};
    FusedLinearOnlineMaxSumCommonTest(shapes, dtypes, 0, V, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_fused_linear_online_max_sum_test, l2_fused_linear_online_max_sum_test_target_dim_invalid) {
    int64_t BT = 8192;
    int64_t H = 4096;
    int64_t V = 19936;
    vector<vector<int64_t>> shapes = {{BT, H}, {V, H}, {BT, 1}, {BT}, {BT}, {BT}, {(BT + 7) / 8}, {BT}, {BT, V}};
    vector<aclDataType> dtypes = {ACL_FLOAT16, ACL_FLOAT16, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
        ACL_UINT8, ACL_INT32, ACL_FLOAT16};
    FusedLinearOnlineMaxSumCommonTest(shapes, dtypes, 0, V, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_fused_linear_online_max_sum_test, l2_fused_linear_online_max_sum_test_logits_max_local_out_dim_invalid) {
    int64_t BT = 8192;
    int64_t H = 4096;
    int64_t V = 19936;
    vector<vector<int64_t>> shapes = {{BT, H}, {V, H}, {BT}, {BT, 1}, {BT}, {BT}, {(BT + 7) / 8}, {BT}, {BT, V}};
    vector<aclDataType> dtypes = {ACL_FLOAT16, ACL_FLOAT16, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
        ACL_UINT8, ACL_INT32, ACL_FLOAT16};
    FusedLinearOnlineMaxSumCommonTest(shapes, dtypes, 0, V, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_fused_linear_online_max_sum_test, l2_fused_linear_online_max_sum_test_sum_exp_logits_local_out_dim_invalid) {
    int64_t BT = 8192;
    int64_t H = 4096;
    int64_t V = 19936;
    vector<vector<int64_t>> shapes = {{BT, H}, {V, H}, {BT}, {BT}, {BT, 1}, {BT}, {(BT + 7) / 8}, {BT}, {BT, V}};
    vector<aclDataType> dtypes = {ACL_FLOAT16, ACL_FLOAT16, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
        ACL_UINT8, ACL_INT32, ACL_FLOAT16};
    FusedLinearOnlineMaxSumCommonTest(shapes, dtypes, 0, V, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_fused_linear_online_max_sum_test, l2_fused_linear_online_max_sum_test_predicted_logits_local_out_dim_invalid) {
    int64_t BT = 8192;
    int64_t H = 4096;
    int64_t V = 19936;
    vector<vector<int64_t>> shapes = {{BT, H}, {V, H}, {BT}, {BT}, {BT}, {BT, 1}, {(BT + 7) / 8}, {BT}, {BT, V}};
    vector<aclDataType> dtypes = {ACL_FLOAT16, ACL_FLOAT16, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
        ACL_UINT8, ACL_INT32, ACL_FLOAT16};
    FusedLinearOnlineMaxSumCommonTest(shapes, dtypes, 0, V, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_fused_linear_online_max_sum_test, l2_fused_linear_online_max_sum_test_target_mask_out_dim_invalid) {
    int64_t BT = 8192;
    int64_t H = 4096;
    int64_t V = 19936;
    vector<vector<int64_t>> shapes = {{BT, H}, {V, H}, {BT}, {BT}, {BT}, {BT}, {(BT + 7) / 8, 1}, {BT}, {BT, V}};
    vector<aclDataType> dtypes = {ACL_FLOAT16, ACL_FLOAT16, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
        ACL_UINT8, ACL_INT32, ACL_FLOAT16};
    FusedLinearOnlineMaxSumCommonTest(shapes, dtypes, 0, V, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_fused_linear_online_max_sum_test, l2_fused_linear_online_max_sum_test_masked_target_out_dim_invalid) {
    int64_t BT = 8192;
    int64_t H = 4096;
    int64_t V = 19936;
    vector<vector<int64_t>> shapes = {{BT, H}, {V, H}, {BT}, {BT}, {BT}, {BT}, {(BT + 7) / 8}, {BT, 1}, {BT, V}};
    vector<aclDataType> dtypes = {ACL_FLOAT16, ACL_FLOAT16, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
        ACL_UINT8, ACL_INT32, ACL_FLOAT16};
    FusedLinearOnlineMaxSumCommonTest(shapes, dtypes, 0, V, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_fused_linear_online_max_sum_test, l2_fused_linear_online_max_sum_test_vocab_parallel_logits_out_optional_dim_invalid) {
    int64_t BT = 8192;
    int64_t H = 4096;
    int64_t V = 19936;
    vector<vector<int64_t>> shapes = {{BT, H}, {V, H}, {BT}, {BT}, {BT}, {BT}, {(BT + 7) / 8}, {BT}, {BT, V, 1}};
    vector<aclDataType> dtypes = {ACL_FLOAT16, ACL_FLOAT16, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
        ACL_UINT8, ACL_INT32, ACL_FLOAT16};
    FusedLinearOnlineMaxSumCommonTest(shapes, dtypes, 0, V, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_fused_linear_online_max_sum_test, l2_fused_linear_online_max_sum_test_input_weight_shape_invalid) {
    int64_t BT = 8192;
    int64_t H = 4096;
    int64_t V = 19936;
    vector<vector<int64_t>> shapes = {{BT, H}, {V, 2048}, {BT}, {BT}, {BT}, {BT}, {(BT + 7) / 8}, {BT}, {BT, V}};
    vector<aclDataType> dtypes = {ACL_FLOAT16, ACL_FLOAT16, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
        ACL_UINT8, ACL_INT32, ACL_FLOAT16};
    FusedLinearOnlineMaxSumCommonTest(shapes, dtypes, 0, V, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_fused_linear_online_max_sum_test, l2_fused_linear_online_max_sum_test_target_shape_invalid) {
    int64_t BT = 8192;
    int64_t H = 4096;
    int64_t V = 19936;
    vector<vector<int64_t>> shapes = {{BT, H}, {V, H}, {1}, {BT}, {BT}, {BT}, {(BT + 7) / 8}, {BT}, {BT, V}};
    vector<aclDataType> dtypes = {ACL_FLOAT16, ACL_FLOAT16, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
        ACL_UINT8, ACL_INT32, ACL_FLOAT16};
    FusedLinearOnlineMaxSumCommonTest(shapes, dtypes, 0, V, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_fused_linear_online_max_sum_test, l2_fused_linear_online_max_sum_test_logits_max_local_out_shape_invalid) {
    int64_t BT = 8192;
    int64_t H = 4096;
    int64_t V = 19936;
    vector<vector<int64_t>> shapes = {{BT, H}, {V, H}, {BT}, {1}, {BT}, {BT}, {(BT + 7) / 8}, {BT}, {BT, V}};
    vector<aclDataType> dtypes = {ACL_FLOAT16, ACL_FLOAT16, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
        ACL_UINT8, ACL_INT32, ACL_FLOAT16};
    FusedLinearOnlineMaxSumCommonTest(shapes, dtypes, 0, V, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_fused_linear_online_max_sum_test, l2_fused_linear_online_max_sum_test_sum_exp_logits_local_out_shape_invalid) {
    int64_t BT = 8192;
    int64_t H = 4096;
    int64_t V = 19936;
    vector<vector<int64_t>> shapes = {{BT, H}, {V, H}, {BT}, {BT}, {1}, {BT}, {(BT + 7) / 8}, {BT}, {BT, V}};
    vector<aclDataType> dtypes = {ACL_FLOAT16, ACL_FLOAT16, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
        ACL_UINT8, ACL_INT32, ACL_FLOAT16};
    FusedLinearOnlineMaxSumCommonTest(shapes, dtypes, 0, V, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_fused_linear_online_max_sum_test, l2_fused_linear_online_max_sum_test_predicted_logits_local_out_shape_invalid) {
    int64_t BT = 8192;
    int64_t H = 4096;
    int64_t V = 19936;
    vector<vector<int64_t>> shapes = {{BT, H}, {V, H}, {BT}, {BT}, {BT}, {1}, {(BT + 7) / 8}, {BT}, {BT, V}};
    vector<aclDataType> dtypes = {ACL_FLOAT16, ACL_FLOAT16, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
        ACL_UINT8, ACL_INT32, ACL_FLOAT16};
    FusedLinearOnlineMaxSumCommonTest(shapes, dtypes, 0, V, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_fused_linear_online_max_sum_test, l2_fused_linear_online_max_sum_test_target_mask_out_shape_invalid) {
    int64_t BT = 8192;
    int64_t H = 4096;
    int64_t V = 19936;
    vector<vector<int64_t>> shapes = {{BT, H}, {V, H}, {BT}, {BT}, {BT}, {BT}, {1}, {BT}, {BT, V}};
    vector<aclDataType> dtypes = {ACL_FLOAT16, ACL_FLOAT16, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
        ACL_UINT8, ACL_INT32, ACL_FLOAT16};
    FusedLinearOnlineMaxSumCommonTest(shapes, dtypes, 0, V, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_fused_linear_online_max_sum_test, l2_fused_linear_online_max_sum_test_masked_target_out_shape_invalid) {
    int64_t BT = 8192;
    int64_t H = 4096;
    int64_t V = 19936;
    vector<vector<int64_t>> shapes = {{BT, H}, {V, H}, {BT}, {BT}, {BT}, {BT}, {(BT + 7) / 8}, {1}, {BT, V}};
    vector<aclDataType> dtypes = {ACL_FLOAT16, ACL_FLOAT16, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
        ACL_UINT8, ACL_INT32, ACL_FLOAT16};
    FusedLinearOnlineMaxSumCommonTest(shapes, dtypes, 0, V, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_fused_linear_online_max_sum_test, l2_fused_linear_online_max_sum_test_vocab_parallel_logits_out_optional_shape_invalid) {
    int64_t BT = 8192;
    int64_t H = 4096;
    int64_t V = 19936;
    vector<vector<int64_t>> shapes = {{BT, H}, {V, H}, {BT}, {BT}, {BT}, {BT}, {(BT + 7) / 8}, {BT}, {BT, 1}};
    vector<aclDataType> dtypes = {ACL_FLOAT16, ACL_FLOAT16, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
        ACL_UINT8, ACL_INT32, ACL_FLOAT16};
    FusedLinearOnlineMaxSumCommonTest(shapes, dtypes, 0, V, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_fused_linear_online_max_sum_test, l2_fused_linear_online_max_sum_test_vocab_start_index_invalid) {
    int64_t BT = 8192;
    int64_t H = 4096;
    int64_t V = 19936;
    vector<vector<int64_t>> shapes = {{BT, H}, {V, H}, {BT}, {BT}, {BT}, {BT}, {(BT + 7) / 8}, {BT}, {BT, V}};
    vector<aclDataType> dtypes = {ACL_FLOAT16, ACL_FLOAT16, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
        ACL_UINT8, ACL_INT32, ACL_FLOAT16};
    FusedLinearOnlineMaxSumCommonTest(shapes, dtypes, -1, V, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_fused_linear_online_max_sum_test, l2_fused_linear_online_max_sum_test_vocab_end_index_invalid) {
    int64_t BT = 8192;
    int64_t H = 4096;
    int64_t V = 19936;
    vector<vector<int64_t>> shapes = {{BT, H}, {V, H}, {BT}, {BT}, {BT}, {BT}, {(BT + 7) / 8}, {BT}, {BT, V, 1}};
    vector<aclDataType> dtypes = {ACL_FLOAT16, ACL_FLOAT16, ACL_INT32, ACL_FLOAT, ACL_FLOAT, ACL_FLOAT,
        ACL_UINT8, ACL_INT32, ACL_FLOAT16};
    FusedLinearOnlineMaxSumCommonTest(shapes, dtypes, 10, 9, ACLNN_ERR_PARAM_INVALID);
}

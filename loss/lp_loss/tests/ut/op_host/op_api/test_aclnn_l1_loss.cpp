/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <array>
#include <vector>
#include "gtest/gtest.h"
#include "opdev/op_log.h"
#include "../../../../op_host/op_api/aclnn_l1_loss.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace op;
using namespace std;

class l2_l1_loss_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "L1 Loss Test Setup" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "L1 Loss Test TearDown" << std::endl;
    }
};

TEST_F(l2_l1_loss_test, aclnnL1Loss_01_float_nd_none)
{
    auto selfDesc = TensorDesc({1, 2, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto targetDesc = TensorDesc({2, 2, 1, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t reduction = 0;

    auto outDesc = TensorDesc({2, 2, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnL1Loss, INPUT(selfDesc, targetDesc, reduction), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_l1_loss_test, aclnnL1Loss_02_float16_nd_mean)
{
    auto selfDesc = TensorDesc({2, 3, 2}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto targetDesc = TensorDesc({2, 3, 2}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t reduction = 1;

    auto outDesc = TensorDesc({}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnL1Loss, INPUT(selfDesc, targetDesc, reduction), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_l1_loss_test, aclnnL1Loss_05_float_nd_empty_tensor_none)
{
    auto selfDesc = TensorDesc({1, 0, 1, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto targetDesc = TensorDesc({1, 0, 1, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t reduction = 0;

    auto outDesc = TensorDesc({1, 0, 1, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnL1Loss, INPUT(selfDesc, targetDesc, reduction), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_l1_loss_test, aclnnL1Loss_06_float_nd_empty_tensor_mean)
{
    auto selfDesc = TensorDesc({1, 0, 1, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto targetDesc = TensorDesc({1, 0, 1, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t reduction = 1;

    auto outDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnL1Loss, INPUT(selfDesc, targetDesc, reduction), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_l1_loss_test, aclnnL1Loss_07_float_nd_empty_tensor_sum)
{
    auto selfDesc = TensorDesc({1, 0, 1, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto targetDesc = TensorDesc({1, 0, 1, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    int64_t reduction = 2;

    auto outDesc = TensorDesc({}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnL1Loss, INPUT(selfDesc, targetDesc, reduction), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_l1_loss_test, aclnnL1Loss_11_float_nd_mean)
{
    auto selfDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto targetDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t reduction = 1;

    auto outDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnL1Loss, INPUT(selfDesc, targetDesc, reduction), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_l1_loss_test, aclnnL1Loss_12_float_nd_input_not_contiguous)
{
    auto selfDesc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(-2, 2);
    auto targetDesc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(-2, 2);
    int64_t reduction = 0;

    auto outDesc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnL1Loss, INPUT(selfDesc, targetDesc, reduction), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_l1_loss_test, aclnnL1Loss_13_float_nd_inoutput_not_contiguous)
{
    auto selfDesc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(-2, 2);
    auto targetDesc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).ValueRange(-2, 2);
    int64_t reduction = 0;

    auto outDesc = TensorDesc({5, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 5}, 0, {4, 5}).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnL1Loss, INPUT(selfDesc, targetDesc, reduction), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_l1_loss_test, aclnnL1Loss_14_reduction_error)
{
    auto selfDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto targetDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    int64_t reduction = 3;

    auto outDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnL1Loss, INPUT(selfDesc, targetDesc, reduction), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_l1_loss_test, aclnnL1Loss_15_input_out_nullptr)
{
    auto tensor_desc = TensorDesc({10, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    int64_t reduction = 0;

    auto ut_l = OP_API_UT(aclnnL1Loss, INPUT((aclTensor*)nullptr, tensor_desc, reduction), OUTPUT(tensor_desc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut_l.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut_r = OP_API_UT(aclnnL1Loss, INPUT(tensor_desc, (aclTensor*)nullptr, reduction), OUTPUT(tensor_desc));
    // SAMPLE: only test GetWorkspaceSize
    aclRet = ut_r.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut_o = OP_API_UT(aclnnL1Loss, INPUT(tensor_desc, tensor_desc, reduction), OUTPUT((aclTensor*)nullptr));
    // SAMPLE: only test GetWorkspaceSize
    aclRet = ut_o.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_l1_loss_test, aclnnL1Loss_16_aclnnL1Loss_input_error_shape)
{
    auto selfDesc = TensorDesc({123, 11, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto targetDesc = TensorDesc({123, 8, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    int64_t reduction = 0;

    auto outDesc = TensorDesc({123, 11, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnL1Loss, INPUT(selfDesc, targetDesc, reduction), OUTPUT(outDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_l1_loss_test, aclnnL1Loss_17_aclnnL1Loss_error_output_dtype)
{
    auto selfDesc = TensorDesc({6, 2, 1, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto targetDesc = TensorDesc({6, 2, 1, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    int64_t reduction = 1;

    auto outDesc = TensorDesc({6, 2, 1, 2}, ACL_BOOL, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnL1Loss, INPUT(selfDesc, targetDesc, reduction), OUTPUT(outDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_l1_loss_test, aclnnL1Loss_18_aclnnL1Loss_error_input_dtype)
{
    auto selfDesc = TensorDesc({6, 2, 1, 2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto targetDesc = TensorDesc({6, 2, 1, 2}, ACL_INT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    int64_t reduction = 1;

    auto outDesc = TensorDesc({6, 2, 1, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnL1Loss, INPUT(selfDesc, targetDesc, reduction), OUTPUT(outDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// TEST_F(l2_l1_loss_test, aclnnL1Loss_19_aclnnL1Loss_input_error_shape_len) {
//   auto tensorDesc9 = TensorDesc({2, 3, 4, 5, 6, 7, 8, 9, 10}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-2, 2);
//   auto tensorDesc1 = TensorDesc({7, 8, 9, 10}, ACL_INT16, ACL_FORMAT_ND).ValueRange(-2, 2);
//   auto tensorDesc2 = TensorDesc({7, 8, 9, 10}, ACL_INT16, ACL_FORMAT_ND).ValueRange(-2, 2);
//   int64_t reduction = 0;

//   auto ut_self = OP_API_UT(aclnnL1Loss, INPUT(tensorDesc9, tensorDesc1, reduction), OUTPUT(tensorDesc2));
//   // SAMPLE: only test GetWorkspaceSize
//   uint64_t workspace_size = 0;
//   aclnnStatus aclRet = ut_self.TestGetWorkspaceSize(&workspace_size);
//   EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

//   auto ut_tar = OP_API_UT(aclnnL1Loss, INPUT(tensorDesc1, tensorDesc9, reduction), OUTPUT(tensorDesc2));
//   // SAMPLE: only test GetWorkspaceSize
//   aclRet = ut_tar.TestGetWorkspaceSize(&workspace_size);
//   EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
// }

TEST_F(l2_l1_loss_test, aclnnL1Loss_20_error_input_dtype)
{
    auto selfDesc = TensorDesc({2, 2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto targetDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t reduction = 0;

    auto outDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnL1Loss, INPUT(selfDesc, targetDesc, reduction), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_l1_loss_test, aclnnL1Loss_21_input_dtype_none)
{
    auto selfDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto targetDesc = TensorDesc({2, 2}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t reduction = 0;

    auto outDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnL1Loss, INPUT(selfDesc, targetDesc, reduction), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_l1_loss_test, aclnnL1Loss_22_aclnnL1Loss_output_error_shape_none)
{
    auto selfDesc = TensorDesc({123, 11, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto targetDesc = TensorDesc({123, 11, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    int64_t reduction = 0;

    auto outDesc = TensorDesc({123, 8, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnL1Loss, INPUT(selfDesc, targetDesc, reduction), OUTPUT(outDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_l1_loss_test, aclnnL1Loss_23_aclnnL1Loss_output_error_shape_mean)
{
    auto selfDesc = TensorDesc({123, 11, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto targetDesc = TensorDesc({123, 11, 2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    int64_t reduction = 1;

    auto outDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnL1Loss, INPUT(selfDesc, targetDesc, reduction), OUTPUT(outDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_l1_loss_test, aclnnL1Loss_24_aclnnL1Loss_input_error_shape_len)
{
    auto tensorDesc9 = TensorDesc({2, 3, 4, 5, 6, 7, 8, 9, 10}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensorDesc1 = TensorDesc({7, 8, 9, 10}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto tensorDesc2 = TensorDesc({7, 8, 9, 10}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    int64_t reduction = 0;

    auto ut_grad = OP_API_UT(aclnnL1Loss, INPUT(tensorDesc9, tensorDesc1, reduction), OUTPUT(tensorDesc2));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut_grad.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

    auto ut_self = OP_API_UT(aclnnL1Loss, INPUT(tensorDesc1, tensorDesc9, reduction), OUTPUT(tensorDesc2));
    // SAMPLE: only test GetWorkspaceSize
    aclRet = ut_self.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_l1_loss_test, aclnnL1Loss_25_int64_sum)
{
    auto selfDesc = TensorDesc({1, 2, 3, 2}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto targetDesc = TensorDesc({2, 2, 1, 2}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-10, 10);
    int64_t reduction = 2;

    auto outDesc = TensorDesc({}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnL1Loss, INPUT(selfDesc, targetDesc, reduction), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_l1_loss_test, aclnnL1Loss_26_int64_sum)
{
    auto selfDesc = TensorDesc({1, 2, 3, 2}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto targetDesc = TensorDesc({2, 2, 1, 2}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-10, 10);
    int64_t reduction = 2;

    auto outDesc = TensorDesc({}, ACL_INT64, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnL1Loss, INPUT(selfDesc, targetDesc, reduction), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_l1_loss_test, aclnnL1Loss_27_int64_none)
{
    auto selfDesc = TensorDesc({1, 2, 3, 2}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto targetDesc = TensorDesc({2, 2, 1, 2}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-10, 10);
    int64_t reduction = 0;

    auto outDesc = TensorDesc({2, 2, 3, 2}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnL1Loss, INPUT(selfDesc, targetDesc, reduction), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_l1_loss_test, aclnnL1Loss_28_int64_none)
{
    auto selfDesc = TensorDesc({1, 2, 3, 2}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto targetDesc = TensorDesc({2, 2, 1, 2}, ACL_INT64, ACL_FORMAT_ND).ValueRange(-10, 10);
    int64_t reduction = 0;

    auto outDesc = TensorDesc({2, 2, 3, 2}, ACL_INT64, ACL_FORMAT_ND).Precision(0.001, 0.001);

    auto ut = OP_API_UT(aclnnL1Loss, INPUT(selfDesc, targetDesc, reduction), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_l1_loss_test, ascend910B2_lp_loss_bf16_none)
{
    auto selfDesc = TensorDesc({3, 5}, ACL_BF16, ACL_FORMAT_ND);
    auto targetDesc = TensorDesc({3, 5}, ACL_BF16, ACL_FORMAT_ND);
    int64_t reduction = 0;

    auto outDesc = TensorDesc({3, 5}, ACL_BF16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnL1Loss, INPUT(selfDesc, targetDesc, reduction), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    // ut.TestPrecision();
}
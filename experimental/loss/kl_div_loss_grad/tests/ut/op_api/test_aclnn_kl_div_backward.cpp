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
#include "../../../../op_host/op_api/aclnn_kl_div_backward.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

#include <unistd.h>

using namespace op;
using namespace std;

enum Reduction { None, Mean, Sum, BatchMean, END };

class kl_div_loss_backward_test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "kl_div_loss_backward_test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "kl_div_loss_backward_test TearDown" << std::endl; }
};

TEST_F(kl_div_loss_backward_test, aclnnKlDivBackward_01_float_none_broadcast)
{
    auto gradOutputDesc = TensorDesc({3, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto targetDesc = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 4);
    int64_t reduction = Reduction::None;
    bool logTarget = false;

    auto outDesc = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(
        aclnnKlDivBackward, INPUT(gradOutputDesc, selfDesc, targetDesc, reduction, logTarget), OUTPUT(outDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

TEST_F(kl_div_loss_backward_test, aclnnKlDivBackward_02_float_nchw_mean)
{
    auto gradOutputDesc = TensorDesc({3, 5, 2, 4}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto selfDesc = TensorDesc({3, 5, 2, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-1, 1);
    auto targetDesc = TensorDesc({3, 5, 2, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-1, 1);
    int64_t reduction = Reduction::Mean;
    bool logTarget = false;

    auto outDesc = TensorDesc({3, 5, 2, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(
        aclnnKlDivBackward, INPUT(gradOutputDesc, selfDesc, targetDesc, reduction, logTarget), OUTPUT(outDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

TEST_F(kl_div_loss_backward_test, aclnnKlDivBackward_03_float16_ncdhw_sum)
{
    auto gradOutputDesc = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto selfDesc = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT16, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
    auto targetDesc = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT16, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
    int64_t reduction = Reduction::Sum;
    bool logTarget = false;

    auto outDesc = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT16, ACL_FORMAT_NCDHW).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(
        aclnnKlDivBackward, INPUT(gradOutputDesc, selfDesc, targetDesc, reduction, logTarget), OUTPUT(outDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

TEST_F(kl_div_loss_backward_test, aclnnKlDivBackward_04_float_hwcn_batchmean)
{
    auto gradOutputDesc = TensorDesc({3, 5, 4, 6}, ACL_FLOAT, ACL_FORMAT_HWCN);
    auto selfDesc = TensorDesc({3, 5, 4, 6}, ACL_FLOAT, ACL_FORMAT_HWCN).ValueRange(-1, 1);
    auto targetDesc = TensorDesc({3, 5, 4, 6}, ACL_FLOAT, ACL_FORMAT_HWCN).ValueRange(-1, 1);
    int64_t reduction = Reduction::BatchMean;
    bool logTarget = false;

    auto outDesc = TensorDesc({3, 5, 4, 6}, ACL_FLOAT, ACL_FORMAT_HWCN).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(
        aclnnKlDivBackward, INPUT(gradOutputDesc, selfDesc, targetDesc, reduction, logTarget), OUTPUT(outDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

TEST_F(kl_div_loss_backward_test, aclnnKlDivBackward_05_float_none_logtarget_true)
{
    auto gradOutputDesc = TensorDesc({3, 5, 4, 6}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({3, 5, 4, 6}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto targetDesc = TensorDesc({3, 5, 4, 6}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t reduction = Reduction::None;
    bool logTarget = true;

    auto outDesc = TensorDesc({3, 5, 4, 6}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(
        aclnnKlDivBackward, INPUT(gradOutputDesc, selfDesc, targetDesc, reduction, logTarget), OUTPUT(outDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

TEST_F(kl_div_loss_backward_test, aclnnKlDivBackward_06_float16_to_float_logtarget_true)
{
    auto gradOutputDesc = TensorDesc({3, 5, 4, 6}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({3, 5, 4, 6}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto targetDesc = TensorDesc({3, 5, 4, 6}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t reduction = Reduction::None;
    bool logTarget = true;

    auto outDesc = TensorDesc({3, 5, 4, 6}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(
        aclnnKlDivBackward, INPUT(gradOutputDesc, selfDesc, targetDesc, reduction, logTarget), OUTPUT(outDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

TEST_F(kl_div_loss_backward_test, aclnnKlDivBackward_07_float_nhwc_logtarget_true)
{
    auto gradOutputDesc = TensorDesc({3, 1, 2, 5}, ACL_FLOAT, ACL_FORMAT_NHWC).ValueRange(-1, 1);
    auto selfDesc = TensorDesc({3, 1, 2, 5}, ACL_FLOAT, ACL_FORMAT_NHWC).ValueRange(-1, 1);
    auto targetDesc = TensorDesc({3, 1, 2, 5}, ACL_FLOAT, ACL_FORMAT_NHWC).ValueRange(-1, 1);
    int64_t reduction = Reduction::None;
    bool logTarget = true;

    auto outDesc = TensorDesc({3, 1, 2, 5}, ACL_FLOAT, ACL_FORMAT_NHWC).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(
        aclnnKlDivBackward, INPUT(gradOutputDesc, selfDesc, targetDesc, reduction, logTarget), OUTPUT(outDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

TEST_F(kl_div_loss_backward_test, aclnnKlDivBackward_08_float_ndhwc_logtarget_true)
{
    auto gradOutputDesc = TensorDesc({3, 1, 2, 5, 4}, ACL_FLOAT, ACL_FORMAT_NDHWC).ValueRange(-1, 1);
    auto selfDesc = TensorDesc({3, 1, 2, 5, 4}, ACL_FLOAT, ACL_FORMAT_NDHWC).ValueRange(-1, 1);
    auto targetDesc = TensorDesc({3, 1, 2, 5, 4}, ACL_FLOAT, ACL_FORMAT_NDHWC).ValueRange(-1, 1);
    int64_t reduction = Reduction::None;
    bool logTarget = true;

    auto outDesc = TensorDesc({3, 1, 2, 5, 4}, ACL_FLOAT, ACL_FORMAT_NDHWC).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(
        aclnnKlDivBackward, INPUT(gradOutputDesc, selfDesc, targetDesc, reduction, logTarget), OUTPUT(outDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

TEST_F(kl_div_loss_backward_test, aclnnKlDivBackward_09_empty_tensor)
{
    auto gradOutputDesc = TensorDesc({3, 1, 0, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto selfDesc = TensorDesc({3, 1, 0, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto targetDesc = TensorDesc({3, 1, 0, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t reduction = Reduction::None;
    bool logTarget = true;

    auto outDesc = TensorDesc({3, 1, 0, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(
        aclnnKlDivBackward, INPUT(gradOutputDesc, selfDesc, targetDesc, reduction, logTarget), OUTPUT(outDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

TEST_F(kl_div_loss_backward_test, aclnnKlDivBackward_10_onedim_tensor_logtarget_true)
{
    auto gradOutputDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto selfDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto targetDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t reduction = Reduction::None;
    bool logTarget = true;

    auto outDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(
        aclnnKlDivBackward, INPUT(gradOutputDesc, selfDesc, targetDesc, reduction, logTarget), OUTPUT(outDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

TEST_F(kl_div_loss_backward_test, aclnnKlDivBackward_11_threedim_tensor_broadcast)
{
    auto gradOutputDesc = TensorDesc({3, 4, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto selfDesc = TensorDesc({3, 4, 6}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto targetDesc = TensorDesc({3, 4, 6}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t reduction = Reduction::None;
    bool logTarget = true;

    auto outDesc = TensorDesc({3, 4, 6}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(
        aclnnKlDivBackward, INPUT(gradOutputDesc, selfDesc, targetDesc, reduction, logTarget), OUTPUT(outDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

TEST_F(kl_div_loss_backward_test, aclnnKlDivBackward_12_fivedim_tensor_logtarget_true)
{
    auto gradOutputDesc = TensorDesc({3, 4, 6, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto selfDesc = TensorDesc({3, 4, 6, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto targetDesc = TensorDesc({3, 4, 6, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t reduction = Reduction::None;
    bool logTarget = true;

    auto outDesc = TensorDesc({3, 4, 6, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(
        aclnnKlDivBackward, INPUT(gradOutputDesc, selfDesc, targetDesc, reduction, logTarget), OUTPUT(outDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

TEST_F(kl_div_loss_backward_test, aclnnKlDivBackward_13_dtype_promote_float16_to_float)
{
    auto gradOutputDesc = TensorDesc({3, 4, 1, 2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto selfDesc = TensorDesc({3, 4, 1, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto targetDesc = TensorDesc({3, 4, 1, 2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t reduction = Reduction::None;
    bool logTarget = true;

    auto outDesc = TensorDesc({3, 4, 1, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(
        aclnnKlDivBackward, INPUT(gradOutputDesc, selfDesc, targetDesc, reduction, logTarget), OUTPUT(outDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

TEST_F(kl_div_loss_backward_test, aclnnKlDivBackward_14_reduction_error)
{
    auto gradOutputDesc = TensorDesc({3, 5, 4, 6}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({3, 5, 4, 6}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto targetDesc = TensorDesc({3, 5, 4, 6}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t reduction = Reduction::END;
    bool logTarget = false;

    auto outDesc = TensorDesc({3, 5, 4, 6}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(
        aclnnKlDivBackward, INPUT(gradOutputDesc, selfDesc, targetDesc, reduction, logTarget), OUTPUT(outDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(kl_div_loss_backward_test, aclnnKlDivBackward_15_input_out_nullptr)
{
    auto tensorDesc = TensorDesc({10, 3, 5, 24}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t reduction = Reduction::None;
    bool logTarget = true;

    auto ut_grad = OP_API_UT(
        aclnnKlDivBackward, INPUT((aclTensor*)nullptr, tensorDesc, tensorDesc, reduction, logTarget),
        OUTPUT(tensorDesc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut_grad.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut_self = OP_API_UT(
        aclnnKlDivBackward, INPUT(tensorDesc, (aclTensor*)nullptr, tensorDesc, reduction, logTarget),
        OUTPUT(tensorDesc));
    aclRet = ut_self.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut_tar = OP_API_UT(
        aclnnKlDivBackward, INPUT(tensorDesc, tensorDesc, (aclTensor*)nullptr, reduction, logTarget),
        OUTPUT(tensorDesc));
    aclRet = ut_tar.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut_o = OP_API_UT(
        aclnnKlDivBackward, INPUT(tensorDesc, tensorDesc, tensorDesc, reduction, logTarget),
        OUTPUT((aclTensor*)nullptr));
    aclRet = ut_o.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(kl_div_loss_backward_test, aclnnKlDivBackward_16_input_error_shape_len)
{
    auto tensorDesc9 = TensorDesc({10, 24, 3, 5, 10, 22, 42, 30, 24}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorDesc4 = TensorDesc({3, 1, 0, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t reduction = Reduction::None;
    bool logTarget = true;

    auto ut_grad = OP_API_UT(
        aclnnKlDivBackward, INPUT(tensorDesc9, tensorDesc4, tensorDesc4, reduction, logTarget), OUTPUT(tensorDesc4));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut_grad.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

    auto ut_self = OP_API_UT(
        aclnnKlDivBackward, INPUT(tensorDesc4, tensorDesc9, tensorDesc4, reduction, logTarget), OUTPUT(tensorDesc4));
    aclRet = ut_self.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);

    auto ut_tar = OP_API_UT(
        aclnnKlDivBackward, INPUT(tensorDesc4, tensorDesc4, tensorDesc9, reduction, logTarget), OUTPUT(tensorDesc4));
    aclRet = ut_tar.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(kl_div_loss_backward_test, aclnnKlDivBackward_17_error_input_dtype)
{
    auto gradOutputDesc = TensorDesc({3, 4, 1, 1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto selfDesc = TensorDesc({3, 4, 6, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto targetDesc = TensorDesc({3, 4, 6, 1}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t reduction = Reduction::None;
    bool logTarget = true;

    auto outDesc = TensorDesc({3, 4, 6, 3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(
        aclnnKlDivBackward, INPUT(gradOutputDesc, selfDesc, targetDesc, reduction, logTarget), OUTPUT(outDesc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(kl_div_loss_backward_test, aclnnKlDivBackward_18_target_not_broadcast)
{
    auto gradOutputDesc = TensorDesc({3, 4, 1, 1}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto selfDesc = TensorDesc({3, 4, 6, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto targetDesc = TensorDesc({3, 4, 6, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t reduction = Reduction::None;
    bool logTarget = true;

    auto outDesc = TensorDesc({3, 4, 6, 3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(
        aclnnKlDivBackward, INPUT(gradOutputDesc, selfDesc, targetDesc, reduction, logTarget), OUTPUT(outDesc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(kl_div_loss_backward_test, aclnnKlDivBackward_19_out_shape_not_equal_self)
{
    auto gradOutputDesc = TensorDesc({3, 4, 1, 1}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto selfDesc = TensorDesc({3, 4, 6, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto targetDesc = TensorDesc({3, 4, 6, 1}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t reduction = Reduction::None;
    bool logTarget = true;

    auto outDesc = TensorDesc({3, 4, 6, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(
        aclnnKlDivBackward, INPUT(gradOutputDesc, selfDesc, targetDesc, reduction, logTarget), OUTPUT(outDesc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(kl_div_loss_backward_test, aclnnKlDivBackward_20_self_not_equal_broadcast_shape)
{
    auto gradOutputDesc = TensorDesc({3, 4, 1, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto selfDesc = TensorDesc({3, 4, 6, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto targetDesc = TensorDesc({3, 4, 6, 1}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t reduction = Reduction::None;
    bool logTarget = true;

    auto outDesc = TensorDesc({3, 4, 6, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(
        aclnnKlDivBackward, INPUT(gradOutputDesc, selfDesc, targetDesc, reduction, logTarget), OUTPUT(outDesc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(kl_div_loss_backward_test, aclnnKlDivBackward_21_broadcast_grad_1d)
{
    auto gradOutputDesc = TensorDesc({3, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({3, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto targetDesc = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 4);
    int64_t reduction = Reduction::None;
    bool logTarget = false;

    auto outDesc = TensorDesc({3, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(
        aclnnKlDivBackward, INPUT(gradOutputDesc, selfDesc, targetDesc, reduction, logTarget), OUTPUT(outDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

TEST_F(kl_div_loss_backward_test, aclnnKlDivBackward_22_broadcast_target_1d)
{
    auto gradOutputDesc = TensorDesc({3, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto targetDesc = TensorDesc({3, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 4);
    int64_t reduction = Reduction::None;
    bool logTarget = false;

    auto outDesc = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(
        aclnnKlDivBackward, INPUT(gradOutputDesc, selfDesc, targetDesc, reduction, logTarget), OUTPUT(outDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

TEST_F(kl_div_loss_backward_test, aclnnKlDivBackward_23_broadcast_mixed_rank)
{
    auto gradOutputDesc = TensorDesc({3, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({3, 5, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto targetDesc = TensorDesc({6, 3, 1, 7}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 4);
    int64_t reduction = Reduction::None;
    bool logTarget = false;

    auto outDesc = TensorDesc({3, 5, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(
        aclnnKlDivBackward, INPUT(gradOutputDesc, selfDesc, targetDesc, reduction, logTarget), OUTPUT(outDesc));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    ut.TestPrecision();
}

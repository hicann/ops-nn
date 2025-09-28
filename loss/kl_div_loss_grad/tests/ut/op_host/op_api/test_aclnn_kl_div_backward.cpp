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
#include "level2/aclnn_kl_div_backward.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;

class l2_kl_div_backward_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "kl_div_backward SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "kl_div_backward TearDown" << endl;
    }
};

TEST_F(l2_kl_div_backward_test, case_fp_reduction_none)
{
    auto gradDesc = TensorDesc({3, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto targetDesc = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 4);
    int64_t reduction = 0;
    bool logTarget = false;

    auto outDesc = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut =
        OP_API_UT(aclnnKlDivBackward, INPUT(gradDesc, selfDesc, targetDesc, reduction, logTarget), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_kl_div_backward_test, case_fp_reduction_mean)
{
    auto gradDesc = TensorDesc({3, 5, 2, 4}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto selfDesc = TensorDesc({3, 5, 2, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-1, 1);
    auto targetDesc = TensorDesc({3, 5, 2, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-1, 1);
    int64_t reduction = 1;
    bool logTarget = false;

    auto outDesc = TensorDesc({3, 5, 2, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).Precision(0.0001, 0.0001);

    auto ut =
        OP_API_UT(aclnnKlDivBackward, INPUT(gradDesc, selfDesc, targetDesc, reduction, logTarget), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    // ut.TestPrecision();
}

TEST_F(l2_kl_div_backward_test, case_fp_reduction_sum)
{
    auto gradDesc = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT16, ACL_FORMAT_NCDHW);
    auto selfDesc = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT16, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
    auto targetDesc = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT16, ACL_FORMAT_NCDHW).ValueRange(-1, 1);
    int64_t reduction = 2;
    bool logTarget = false;

    auto outDesc = TensorDesc({1, 2, 3, 4, 5}, ACL_FLOAT16, ACL_FORMAT_NCDHW).Precision(0.0001, 0.0001);

    auto ut =
        OP_API_UT(aclnnKlDivBackward, INPUT(gradDesc, selfDesc, targetDesc, reduction, logTarget), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    // ut.TestPrecision();
}

TEST_F(l2_kl_div_backward_test, case_fp_reduction_batchmean)
{
    auto gradDesc = TensorDesc({3, 5, 4, 6}, ACL_FLOAT, ACL_FORMAT_HWCN);
    auto selfDesc = TensorDesc({3, 5, 4, 6}, ACL_FLOAT, ACL_FORMAT_HWCN).ValueRange(-1, 1);
    auto targetDesc = TensorDesc({3, 5, 4, 6}, ACL_FLOAT, ACL_FORMAT_HWCN).ValueRange(-1, 1);
    int64_t reduction = 3;
    bool logTarget = false;

    auto outDesc = TensorDesc({3, 5, 4, 6}, ACL_FLOAT, ACL_FORMAT_HWCN).Precision(0.0001, 0.0001);

    auto ut =
        OP_API_UT(aclnnKlDivBackward, INPUT(gradDesc, selfDesc, targetDesc, reduction, logTarget), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    // ut.TestPrecision();
}

TEST_F(l2_kl_div_backward_test, case_fp_reduction_exceed)
{
    auto gradDesc = TensorDesc({3, 5, 4, 6}, ACL_FLOAT, ACL_FORMAT_HWCN);
    auto selfDesc = TensorDesc({3, 5, 4, 6}, ACL_FLOAT, ACL_FORMAT_HWCN).ValueRange(-1, 1);
    auto targetDesc = TensorDesc({3, 5, 4, 6}, ACL_FLOAT, ACL_FORMAT_HWCN).ValueRange(-1, 1);
    int64_t reduction = 6;
    bool logTarget = false;

    auto outDesc = TensorDesc({3, 5, 4, 6}, ACL_FLOAT, ACL_FORMAT_HWCN).Precision(0.0001, 0.0001);

    auto ut =
        OP_API_UT(aclnnKlDivBackward, INPUT(gradDesc, selfDesc, targetDesc, reduction, logTarget), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    // ut.TestPrecision();
}

TEST_F(l2_kl_div_backward_test, case_logtarget_true)
{
    auto gradDesc = TensorDesc({3, 5, 4, 6}, ACL_FLOAT, ACL_FORMAT_HWCN);
    auto selfDesc = TensorDesc({3, 5, 4, 6}, ACL_FLOAT, ACL_FORMAT_HWCN).ValueRange(-1, 1);
    auto targetDesc = TensorDesc({3, 5, 4, 6}, ACL_FLOAT, ACL_FORMAT_HWCN).ValueRange(-1, 1);
    int64_t reduction = 0;
    bool logTarget = true;

    auto outDesc = TensorDesc({3, 5, 4, 6}, ACL_FLOAT, ACL_FORMAT_HWCN).Precision(0.0001, 0.0001);

    auto ut =
        OP_API_UT(aclnnKlDivBackward, INPUT(gradDesc, selfDesc, targetDesc, reduction, logTarget), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_kl_div_backward_test, ascend910B2_case_fp_logtarget_true)
{
    auto gradDesc = TensorDesc({3, 5, 4, 6}, ACL_FLOAT16, ACL_FORMAT_HWCN);
    auto selfDesc = TensorDesc({3, 5, 4, 6}, ACL_FLOAT, ACL_FORMAT_HWCN).ValueRange(-1, 1);
    auto targetDesc = TensorDesc({3, 5, 4, 6}, ACL_FLOAT16, ACL_FORMAT_HWCN).ValueRange(-1, 1);
    int64_t reduction = 0;
    bool logTarget = true;

    auto outDesc = TensorDesc({3, 5, 4, 6}, ACL_FLOAT, ACL_FORMAT_HWCN).Precision(0.0001, 0.0001);

    auto ut =
        OP_API_UT(aclnnKlDivBackward, INPUT(gradDesc, selfDesc, targetDesc, reduction, logTarget), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_kl_div_backward_test, case_NHWC)
{
    auto gradDesc = TensorDesc({3, 1, 2, 5}, ACL_FLOAT, ACL_FORMAT_NHWC).ValueRange(-1, 1);
    auto selfDesc = TensorDesc({3, 1, 2, 5}, ACL_FLOAT, ACL_FORMAT_NHWC).ValueRange(-1, 1);
    auto targetDesc = TensorDesc({3, 1, 2, 5}, ACL_FLOAT, ACL_FORMAT_NHWC).ValueRange(-1, 1);
    int64_t reduction = 0;
    bool logTarget = true;

    auto outDesc = TensorDesc({3, 1, 2, 5}, ACL_FLOAT, ACL_FORMAT_NHWC).Precision(0.0001, 0.0001);

    auto ut =
        OP_API_UT(aclnnKlDivBackward, INPUT(gradDesc, selfDesc, targetDesc, reduction, logTarget), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_kl_div_backward_test, case_NDHWC)
{
    auto gradDesc = TensorDesc({3, 1, 2, 5, 4}, ACL_FLOAT, ACL_FORMAT_NDHWC).ValueRange(-1, 1);
    auto selfDesc = TensorDesc({3, 1, 2, 5, 4}, ACL_FLOAT, ACL_FORMAT_NDHWC).ValueRange(-1, 1);
    auto targetDesc = TensorDesc({3, 1, 2, 5, 4}, ACL_FLOAT, ACL_FORMAT_NDHWC).ValueRange(-1, 1);
    int64_t reduction = 0;
    bool logTarget = true;

    auto outDesc = TensorDesc({3, 1, 2, 5, 4}, ACL_FLOAT, ACL_FORMAT_NDHWC).Precision(0.0001, 0.0001);

    auto ut =
        OP_API_UT(aclnnKlDivBackward, INPUT(gradDesc, selfDesc, targetDesc, reduction, logTarget), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_kl_div_backward_test, case_empty_tensor)
{
    auto gradDesc = TensorDesc({3, 1, 0, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto selfDesc = TensorDesc({3, 1, 0, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto targetDesc = TensorDesc({3, 1, 0, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t reduction = 0;
    bool logTarget = true;

    auto outDesc = TensorDesc({3, 1, 0, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut =
        OP_API_UT(aclnnKlDivBackward, INPUT(gradDesc, selfDesc, targetDesc, reduction, logTarget), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_kl_div_backward_test, case_onedim_tensor)
{
    auto gradDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto selfDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto targetDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t reduction = 0;
    bool logTarget = true;

    auto outDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut =
        OP_API_UT(aclnnKlDivBackward, INPUT(gradDesc, selfDesc, targetDesc, reduction, logTarget), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_kl_div_backward_test, case_threedim_tensor)
{
    auto gradDesc = TensorDesc({3, 4, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto selfDesc = TensorDesc({3, 4, 6}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto targetDesc = TensorDesc({3, 4, 6}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t reduction = 0;
    bool logTarget = true;

    auto outDesc = TensorDesc({3, 4, 6}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut =
        OP_API_UT(aclnnKlDivBackward, INPUT(gradDesc, selfDesc, targetDesc, reduction, logTarget), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_kl_div_backward_test, case_fivedim_tensor)
{
    auto gradDesc = TensorDesc({3, 4, 6, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto selfDesc = TensorDesc({3, 4, 6, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto targetDesc = TensorDesc({3, 4, 6, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t reduction = 0;
    bool logTarget = true;

    auto outDesc = TensorDesc({3, 4, 6, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut =
        OP_API_UT(aclnnKlDivBackward, INPUT(gradDesc, selfDesc, targetDesc, reduction, logTarget), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_kl_div_backward_test, case_dtype_promte)
{
    auto gradDesc = TensorDesc({3, 4, 1, 2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto selfDesc = TensorDesc({3, 4, 1, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto targetDesc = TensorDesc({3, 4, 1, 2, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t reduction = 0;
    bool logTarget = true;

    auto outDesc = TensorDesc({3, 4, 1, 2, 3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut =
        OP_API_UT(aclnnKlDivBackward, INPUT(gradDesc, selfDesc, targetDesc, reduction, logTarget), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);

    // SAMPLE: precision simulate
    ut.TestPrecision();
}

TEST_F(l2_kl_div_backward_test, case_checkshape_self_noequal_broadshape)
{
    auto gradDesc = TensorDesc({3, 4, 1, 3}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto selfDesc = TensorDesc({3, 4, 6, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto targetDesc = TensorDesc({3, 4, 6, 1}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t reduction = 0;
    bool logTarget = true;

    auto outDesc = TensorDesc({3, 4, 6, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut =
        OP_API_UT(aclnnKlDivBackward, INPUT(gradDesc, selfDesc, targetDesc, reduction, logTarget), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_kl_div_backward_test, case_checkshape_target_not_broadcast)
{
    auto gradDesc = TensorDesc({3, 4, 1, 1}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto selfDesc = TensorDesc({3, 4, 6, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto targetDesc = TensorDesc({3, 4, 6, 4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t reduction = 0;
    bool logTarget = true;

    auto outDesc = TensorDesc({3, 4, 6, 3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut =
        OP_API_UT(aclnnKlDivBackward, INPUT(gradDesc, selfDesc, targetDesc, reduction, logTarget), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_kl_div_backward_test, case_checkshape_out_not_equal_self)
{
    auto gradDesc = TensorDesc({3, 4, 1, 1}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto selfDesc = TensorDesc({3, 4, 6, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto targetDesc = TensorDesc({3, 4, 6, 1}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t reduction = 0;
    bool logTarget = true;

    auto outDesc = TensorDesc({3, 4, 6, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut =
        OP_API_UT(aclnnKlDivBackward, INPUT(gradDesc, selfDesc, targetDesc, reduction, logTarget), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_kl_div_backward_test, case_checkdtype)
{
    auto gradDesc = TensorDesc({3, 4, 1, 1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto selfDesc = TensorDesc({3, 4, 6, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto targetDesc = TensorDesc({3, 4, 6, 1}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t reduction = 0;
    bool logTarget = true;

    auto outDesc = TensorDesc({3, 4, 6, 3}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut =
        OP_API_UT(aclnnKlDivBackward, INPUT(gradDesc, selfDesc, targetDesc, reduction, logTarget), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_kl_div_backward_test, case_nine_dims)
{
    auto gradDesc = TensorDesc({10, 24, 3, 5, 10, 22, 42, 30, 24}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto selfDesc = TensorDesc({3, 1, 0, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto targetDesc = TensorDesc({3, 1, 0, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    int64_t reduction = 0;
    bool logTarget = true;

    auto outDesc = TensorDesc({3, 1, 0, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut =
        OP_API_UT(aclnnKlDivBackward, INPUT(gradDesc, selfDesc, targetDesc, reduction, logTarget), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_kl_div_backward_test, case_checknull)
{
    auto gradDesc = TensorDesc({10, 3, 5, 24}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto selfDesc = TensorDesc({10, 3, 5, 24}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto targetDesc = TensorDesc({10, 3, 5, 24}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto outDesc = TensorDesc({10, 3, 5, 24}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    int64_t reduction = 0;
    bool logTarget = true;

    auto ut =
        OP_API_UT(aclnnKlDivBackward, INPUT(nullptr, selfDesc, targetDesc, reduction, logTarget), OUTPUT(outDesc));
    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut_2 =
        OP_API_UT(aclnnKlDivBackward, INPUT(gradDesc, nullptr, targetDesc, reduction, logTarget), OUTPUT(outDesc));
    aclRet = ut_2.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut_3 =
        OP_API_UT(aclnnKlDivBackward, INPUT(gradDesc, selfDesc, nullptr, reduction, logTarget), OUTPUT(outDesc));
    aclRet = ut_3.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);

    auto ut_4 =
        OP_API_UT(aclnnKlDivBackward, INPUT(gradDesc, selfDesc, targetDesc, reduction, logTarget), OUTPUT(nullptr));
    aclRet = ut_4.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_kl_div_backward_test, case_broadcast_1)
{
    auto gradDesc = TensorDesc({3, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({3, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto targetDesc = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 4);
    int64_t reduction = 0;
    bool logTarget = false;

    auto outDesc = TensorDesc({3, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut =
        OP_API_UT(aclnnKlDivBackward, INPUT(gradDesc, selfDesc, targetDesc, reduction, logTarget), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_kl_div_backward_test, case_broadcast_2)
{
    auto gradDesc = TensorDesc({3, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto targetDesc = TensorDesc({3, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 4);
    int64_t reduction = 0;
    bool logTarget = false;

    auto outDesc = TensorDesc({3, 5}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut =
        OP_API_UT(aclnnKlDivBackward, INPUT(gradDesc, selfDesc, targetDesc, reduction, logTarget), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_kl_div_backward_test, case_broadcast_3)
{
    auto gradDesc = TensorDesc({3, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto selfDesc = TensorDesc({3, 5, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto targetDesc = TensorDesc({6, 3, 1, 7}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 4);
    int64_t reduction = 0;
    bool logTarget = false;

    auto outDesc = TensorDesc({3, 5, 1}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut =
        OP_API_UT(aclnnKlDivBackward, INPUT(gradDesc, selfDesc, targetDesc, reduction, logTarget), OUTPUT(outDesc));

    // SAMPLE: only test GetWorkspaceSize
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

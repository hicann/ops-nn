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
#include "../../../../rms_norm_quant/op_host/op_api/aclnn_rms_norm_quant.h"
#include "opdev/platform.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;

class l2RmsNormQuantV2Test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "l2RmsNormQuantV2Test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "l2RmsNormQuantV2Test TearDown" << endl;
    }
};

TEST_F(l2RmsNormQuantV2Test, ascend950PR_9589_case_001)
{
    op::NpuArchManager archManager(NpuArch::DAV_3510); 
    auto xDesc = TensorDesc({8, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    auto gammaDesc = TensorDesc({64,}, ACL_FLOAT, ACL_FORMAT_ND);
    auto betaDesc = TensorDesc({64,}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scaleDesc = TensorDesc({64,}, ACL_FLOAT, ACL_FORMAT_ND);
    auto offserDesc = TensorDesc({64,}, ACL_FLOAT, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({8, 64}, ACL_INT8, ACL_FORMAT_ND);

    double eps = 1e-5;
    auto ut = OP_API_UT(
        aclnnRmsNormQuant,
        INPUT(xDesc, gammaDesc, betaDesc, scaleDesc, offserDesc, eps),
        OUTPUT(yDesc));

    uint64_t workSpace = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workSpace);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}


TEST_F(l2RmsNormQuantV2Test, ascend950PR_9589_case_002)
{
    op::NpuArchManager archManager(NpuArch::DAV_3510); 
    auto xDesc = TensorDesc({8, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto gammaDesc = TensorDesc({64,}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto betaDesc = TensorDesc({64,}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto scaleDesc = TensorDesc({64,}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto offserDesc = TensorDesc({64,}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({8, 8}, ACL_INT32, ACL_FORMAT_ND);

    double eps = 1e-5;
    auto ut = OP_API_UT(
        aclnnRmsNormQuant,
        INPUT(xDesc, gammaDesc, betaDesc, scaleDesc, offserDesc, eps),
        OUTPUT(yDesc));

    uint64_t workSpace = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workSpace);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2RmsNormQuantV2Test, ascend950PR_9589_case_003)
{
    op::NpuArchManager archManager(NpuArch::DAV_3510); 
    auto xDesc = TensorDesc({8, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto gammaDesc = TensorDesc({64,}, ACL_BF16, ACL_FORMAT_ND);
    auto betaDesc = TensorDesc({64,}, ACL_BF16, ACL_FORMAT_ND);
    auto scaleDesc = TensorDesc({64,}, ACL_BF16, ACL_FORMAT_ND);
    auto offserDesc = TensorDesc({64,}, ACL_BF16, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({1, 8}, ACL_INT8, ACL_FORMAT_ND);

    double eps = 1e-5;
    auto ut = OP_API_UT(
        aclnnRmsNormQuant,
        INPUT(xDesc, gammaDesc, betaDesc, scaleDesc, offserDesc, eps),
        OUTPUT(yDesc));

    uint64_t workSpace = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workSpace);
    EXPECT_NE(aclRet, ACL_SUCCESS);
}

TEST_F(l2RmsNormQuantV2Test, ascend950PR_9589_case_004)
{
    op::NpuArchManager archManager(NpuArch::DAV_3510); 
    auto xDesc = TensorDesc({8, 64}, ACL_BF16, ACL_FORMAT_ND);
    auto gammaDesc = TensorDesc({64,}, ACL_BF16, ACL_FORMAT_ND);
    auto betaDesc = TensorDesc({64,}, ACL_BF16, ACL_FORMAT_ND);
    auto scaleDesc = TensorDesc({64,}, ACL_BF16, ACL_FORMAT_ND);
    auto offserDesc = TensorDesc({64,}, ACL_BF16, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({8, 66}, ACL_INT4, ACL_FORMAT_ND);

    double eps = 1e-5;
    auto ut = OP_API_UT(
        aclnnRmsNormQuant,
        INPUT(xDesc, gammaDesc, betaDesc, scaleDesc, offserDesc, eps),
        OUTPUT(yDesc));

    uint64_t workSpace = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workSpace);
    EXPECT_NE(aclRet, ACL_SUCCESS);
}



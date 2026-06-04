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
#include "../../../../op_host/op_api/aclnn_elu_v2.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

#include <unistd.h>

using namespace op;
using namespace std;

class EluV2OpApiTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "EluV2OpApiTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "EluV2OpApiTest TearDown" << std::endl; }
};

TEST_F(EluV2OpApiTest, aclnnEluV2_float){
    auto xDesc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto yDesc = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.0001, 0.0001);
    float alpha = 0.5f;
    float scale = 1.5f;
    float input_scale = -2.0f;

    auto ut = OP_API_UT(aclnnEluV2, INPUT(x, alpha, scale, input_scale), OUTPUT(y));
    
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    
    ut.TestPrecision();
}
TEST_F(EluV2OpApiTest, aclnnEluV2_float16){
    auto xDesc = TensorDesc({256,256}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto yDesc = TensorDesc({256,256}, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    float alpha = 0.5f;
    float scale = 1.5f;
    float input_scale = -2.0f;
    
    auto ut = OP_API_UT(aclnnEluV2, INPUT(x, alpha, scale, input_scale), OUTPUT(y));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    
    ut.TestPrecision();
}
TEST_F(EluV2OpApiTest, aclnnEluV2_bfloat16){
    auto xDesc = TensorDesc({25,16,1023}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto yDesc = TensorDesc({25,16,1023}, ACL_BF16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    float alpha = 0.5f;
    float scale = 1.5f;
    float input_scale = -2.0f;
    
    auto ut = OP_API_UT(aclnnEluV2, INPUT(x, alpha, scale, input_scale), OUTPUT(y));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
    
    ut.TestPrecision();
}

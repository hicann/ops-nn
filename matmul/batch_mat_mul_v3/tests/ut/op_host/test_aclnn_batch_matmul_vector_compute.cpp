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

#include "../../../op_host/op_api/aclnn_batch_matmul.h"
#include "opdev/platform.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;
using namespace op;

class l2_batch_matmul_vector_compute_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "batch_matmul_vector_compute_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "batch_matmul_vector_compute_test TearDown" << endl;
    }
};

TEST_F(l2_batch_matmul_vector_compute_test, ascend910b_vector_case_1)
{
    auto self_desc = TensorDesc({12200, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto mat2_desc = TensorDesc({12200, 4, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_desc = TensorDesc({12200, 4, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);
    int8_t cube_math_type = 1;
    auto ut = OP_API_UT(aclnnBatchMatMul, INPUT(self_desc, mat2_desc), OUTPUT(out_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_batch_matmul_vector_compute_test, ascend910b_vector_case_2)
{
    auto self_desc = TensorDesc({220, 4, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto mat2_desc = TensorDesc({220, 4, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_desc = TensorDesc({220, 4, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);
    int8_t cube_math_type = 1;
    auto ut = OP_API_UT(aclnnBatchMatMul, INPUT(self_desc, mat2_desc), OUTPUT(out_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_batch_matmul_vector_compute_test, ascend910b_vector_case_3)
{
    auto self_desc = TensorDesc({12200, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto mat2_desc = TensorDesc({12200, 3, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_desc = TensorDesc({12200, 3, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);
    int8_t cube_math_type = 1;
    auto ut = OP_API_UT(aclnnBatchMatMul, INPUT(self_desc, mat2_desc), OUTPUT(out_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(l2_batch_matmul_vector_compute_test, ascend910b_vector_case_4)
{
    auto self_desc = TensorDesc({220, 3, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto mat2_desc = TensorDesc({220, 3, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto out_desc = TensorDesc({220, 3, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2).Precision(0.005, 0.005);
    int8_t cube_math_type = 1;
    auto ut = OP_API_UT(aclnnBatchMatMul, INPUT(self_desc, mat2_desc), OUTPUT(out_desc), cube_math_type);

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

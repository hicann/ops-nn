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
#include "../../../op_host/op_api/aclnn_chamfer_distance_backward.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class l2_chamfer_distance_backward_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "chamfer_distance_backward_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "chamfer_distance_backward_test TearDown" << endl;
    }
};

TEST_F(l2_chamfer_distance_backward_test, Ascend910B_case_float_normal)
{
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B) {
        const int64_t batch = 2;
        const int64_t num = 1000;
        auto xyz1 = TensorDesc({batch, num, 2}, ACL_FLOAT, ACL_FORMAT_ND);
        auto xyz2 = TensorDesc({batch, num, 2}, ACL_FLOAT, ACL_FORMAT_ND);
        auto idx1 = TensorDesc({batch, num}, ACL_INT32, ACL_FORMAT_ND);
        auto idx2 = TensorDesc({batch, num}, ACL_INT32, ACL_FORMAT_ND);
        auto grad_dist1 = TensorDesc({batch, num}, ACL_FLOAT, ACL_FORMAT_ND);
        auto grad_dist2 = TensorDesc({batch, num}, ACL_FLOAT, ACL_FORMAT_ND);
        auto grad_xyz1 = TensorDesc({batch, num, 2}, ACL_FLOAT, ACL_FORMAT_ND);
        auto grad_xyz2 = TensorDesc({batch, num, 2}, ACL_FLOAT, ACL_FORMAT_ND);

        auto ut = OP_API_UT(
            aclnnChamferDistanceBackward, INPUT(xyz1, xyz2, idx1, idx2, grad_dist1, grad_dist2),
            OUTPUT(grad_xyz1, grad_xyz2));

        uint64_t workspace_size = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        EXPECT_EQ(aclRet, ACL_SUCCESS);
    }
}

TEST_F(l2_chamfer_distance_backward_test, Ascend910B_case_half_normal)
{
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B) {
        const int64_t batch = 32;
        const int64_t num = 10000;
        auto xyz1 = TensorDesc({batch, num, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
        auto xyz2 = TensorDesc({batch, num, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
        auto idx1 = TensorDesc({batch, num}, ACL_INT32, ACL_FORMAT_ND);
        auto idx2 = TensorDesc({batch, num}, ACL_INT32, ACL_FORMAT_ND);
        auto grad_dist1 = TensorDesc({batch, num}, ACL_FLOAT16, ACL_FORMAT_ND);
        auto grad_dist2 = TensorDesc({batch, num}, ACL_FLOAT16, ACL_FORMAT_ND);
        auto grad_xyz1 = TensorDesc({batch, num, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
        auto grad_xyz2 = TensorDesc({batch, num, 2}, ACL_FLOAT16, ACL_FORMAT_ND);

        auto ut = OP_API_UT(
            aclnnChamferDistanceBackward, INPUT(xyz1, xyz2, idx1, idx2, grad_dist1, grad_dist2),
            OUTPUT(grad_xyz1, grad_xyz2));

        uint64_t workspace_size = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        EXPECT_EQ(aclRet, ACL_SUCCESS);
    }
}

TEST_F(l2_chamfer_distance_backward_test, Ascend910B_case_nullptr_normal)
{
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B) {
        const int64_t batch = 32;
        const int64_t num = 10000;
        auto xyz1 = TensorDesc({batch, num, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
        auto xyz2 = TensorDesc({batch, num, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
        auto idx1 = TensorDesc({batch, num}, ACL_INT32, ACL_FORMAT_ND);
        auto idx2 = TensorDesc({batch, num}, ACL_INT32, ACL_FORMAT_ND);
        auto grad_dist1 = TensorDesc({batch, num}, ACL_FLOAT16, ACL_FORMAT_ND);
        auto grad_dist2 = TensorDesc({batch, num}, ACL_FLOAT16, ACL_FORMAT_ND);
        auto grad_xyz1 = TensorDesc({batch, num, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
        auto grad_xyz2 = TensorDesc({batch, num, 2}, ACL_FLOAT16, ACL_FORMAT_ND);

        auto ut = OP_API_UT(
            aclnnChamferDistanceBackward, INPUT((aclTensor*)nullptr, xyz2, idx1, idx2, grad_dist1, grad_dist2),
            OUTPUT(grad_xyz1, grad_xyz2));

        uint64_t workspace_size = 0;
        aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
        EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
    }
}

TEST_F(l2_chamfer_distance_backward_test, Ascend910B_case_invalid_normal)
{
    const int64_t batch = 32;
    const int64_t num = 10000;
    auto xyz1 = TensorDesc({batch, num, 2}, ACL_INT32, ACL_FORMAT_ND);
    auto xyz2 = TensorDesc({batch, num, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto idx1 = TensorDesc({batch, num}, ACL_INT32, ACL_FORMAT_ND);
    auto idx2 = TensorDesc({batch, num}, ACL_INT32, ACL_FORMAT_ND);
    auto grad_dist1 = TensorDesc({batch, num}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto grad_dist2 = TensorDesc({batch, num}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto grad_xyz1 = TensorDesc({batch, num, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto grad_xyz2 = TensorDesc({batch, num, 2}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnChamferDistanceBackward, INPUT(xyz1, xyz2, idx1, idx2, grad_dist1, grad_dist2),
        OUTPUT(grad_xyz1, grad_xyz2));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
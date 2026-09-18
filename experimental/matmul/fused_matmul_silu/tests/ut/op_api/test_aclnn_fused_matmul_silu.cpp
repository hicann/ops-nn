/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>
#include <vector>

#include "gtest/gtest.h"
#include "../../../op_host/op_api/aclnn_fused_matmul_silu.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"

class FusedMatmulSiluAclnnTest : public testing::Test {};

TEST_F(FusedMatmulSiluAclnnTest, valid_bf16_nd)
{
    auto xDesc = TensorDesc({2, 256}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto weightDesc = TensorDesc({4096, 256}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto biasDesc = TensorDesc({4096}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto yDesc = TensorDesc({2, 4096}, ACL_BF16, ACL_FORMAT_ND).Precision(0.02, 0.02);

    auto ut = OP_API_UT(aclnnFusedMatmulSilu, INPUT(xDesc, weightDesc, biasDesc), OUTPUT(yDesc));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
}

TEST_F(FusedMatmulSiluAclnnTest, invalid_dtype)
{
    auto xDesc = TensorDesc({2, 256}, ACL_FLOAT, ACL_FORMAT_ND);
    auto weightDesc = TensorDesc({4096, 256}, ACL_BF16, ACL_FORMAT_ND);
    auto biasDesc = TensorDesc({4096}, ACL_BF16, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({2, 4096}, ACL_BF16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnFusedMatmulSilu, INPUT(xDesc, weightDesc, biasDesc), OUTPUT(yDesc));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(FusedMatmulSiluAclnnTest, invalid_shape)
{
    auto xDesc = TensorDesc({2, 256}, ACL_BF16, ACL_FORMAT_ND);
    auto weightDesc = TensorDesc({4096, 512}, ACL_BF16, ACL_FORMAT_ND);
    auto biasDesc = TensorDesc({4096}, ACL_BF16, ACL_FORMAT_ND);
    auto yDesc = TensorDesc({2, 4096}, ACL_BF16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnFusedMatmulSilu, INPUT(xDesc, weightDesc, biasDesc), OUTPUT(yDesc));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

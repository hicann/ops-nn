/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "../../../../op_api/aclnn_logit.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

namespace {
class L2LogitTest : public testing::TestWithParam<op::SocVersion> {};

// Phase one validates the non-contiguous execution plan; actual values and writeback need ST.
TEST_P(L2LogitTest, non_contiguous_input_and_output)
{
    op::SocVersionManager versionManager(GetParam());
    auto input = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2}).ValueRange(0.1F, 0.9F);
    auto out = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2});
    double eps = 0.1;
    auto ut = OP_API_UT(aclnnLogit, INPUT(input, eps), OUTPUT(out));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(Ascend910, L2LogitTest,
                         testing::Values(op::SocVersion::ASCEND910B, op::SocVersion::ASCEND910_93));
} // namespace

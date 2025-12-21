/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "../../../op_host/op_api/aclnn_batch_norm_reduce.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

class l2BatchNormReduceTest : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "l2l2BatchNormReduceTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "l2l2BatchNormReduceTest TearDown" << std::endl;
    }
};

TEST_F(l2BatchNormReduceTest, l2_batch_norm_reduce_bfloat16)
{
    auto x = TensorDesc({3, 5, 3, 8}, ACL_BF16, ACL_FORMAT_NCHW);
    auto sum = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto squareSum = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnBatchNormReduce,
        INPUT(x),
        OUTPUT(sum, squareSum));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    // EXPECT_EQ(aclRet, ACL_SUCCESS);
}

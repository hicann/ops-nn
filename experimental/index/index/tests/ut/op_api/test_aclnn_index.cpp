/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../../../op_api/aclnn_index_v2.h"
#include <gtest/gtest.h>
#include <vector>
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace op;
using namespace std;

class TestAclnnIndex : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "aclnn_index_test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "aclnn_index_test TearDown" << std::endl; }
};

TEST_F(TestAclnnIndex, l2_index_normal_fp32_int64_index)
{
    auto selfDesc = TensorDesc({30, 10}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 4);
    auto indexDesc = TensorDesc({5}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 29);
    auto indicesDesc = TensorListDesc({indexDesc});
    auto outDesc = TensorDesc({5, 10}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnIndex, INPUT(selfDesc, indicesDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
}

TEST_F(TestAclnnIndex, l2_index_normal_int32_broadcast_indices)
{
    auto selfDesc = TensorDesc({8, 9, 4}, ACL_INT32, ACL_FORMAT_ND).ValueRange(1, 4);
    auto index0Desc = TensorDesc({2, 1}, ACL_INT32, ACL_FORMAT_ND).ValueRange(0, 7);
    auto index1Desc = TensorDesc({1, 3}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 8);
    auto indicesDesc = TensorListDesc({index0Desc, index1Desc});
    auto outDesc = TensorDesc({2, 3, 4}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnIndex, INPUT(selfDesc, indicesDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
}

TEST_F(TestAclnnIndex, l2_index_bool_index_uses_aicpu_path)
{
    auto selfDesc = TensorDesc({4, 5}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(0, 1);
    auto indexDesc = TensorDesc({4, 5}, ACL_BOOL, ACL_FORMAT_ND).ValueRange(0, 1);
    auto indicesDesc = TensorListDesc({indexDesc});
    auto outDesc = TensorDesc({6}, ACL_BOOL, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnIndex, INPUT(selfDesc, indicesDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
}

TEST_F(TestAclnnIndex, l2_index_empty_self_success)
{
    auto selfDesc = TensorDesc({0, 10}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(1, 4);
    auto indexDesc = TensorDesc({5}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 1);
    auto indicesDesc = TensorListDesc({indexDesc});
    auto outDesc = TensorDesc({5, 10}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnIndex, INPUT(selfDesc, indicesDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
}

TEST_F(TestAclnnIndex, l2_index_indices_beyond_self_rank_success)
{
    auto selfDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 4);
    auto indexDesc = TensorDesc({2}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 2);
    auto indicesDesc = TensorListDesc({indexDesc, indexDesc});
    auto outDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnIndex, INPUT(selfDesc, indicesDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
}

TEST_F(TestAclnnIndex, l2_index_null_self)
{
    auto indexDesc = TensorDesc({5}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 4);
    auto indicesDesc = TensorListDesc({indexDesc});
    auto outDesc = TensorDesc({5, 10}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnIndex, INPUT(nullptr, indicesDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(TestAclnnIndex, l2_index_output_dtype_mismatch)
{
    auto selfDesc = TensorDesc({30, 10}, ACL_INT8, ACL_FORMAT_ND).ValueRange(1, 4);
    auto indexDesc = TensorDesc({5}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 29);
    auto indicesDesc = TensorListDesc({indexDesc});
    auto outDesc = TensorDesc({5, 10}, ACL_INT32, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnIndex, INPUT(selfDesc, indicesDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(TestAclnnIndex, l2_index_unsupported_index_dtype)
{
    auto selfDesc = TensorDesc({30, 10}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 4);
    auto indexDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 29);
    auto indicesDesc = TensorListDesc({indexDesc});
    auto outDesc = TensorDesc({5, 10}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnIndex, INPUT(selfDesc, indicesDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(TestAclnnIndex, l2_index_output_shape_mismatch)
{
    auto selfDesc = TensorDesc({30, 10}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 4);
    auto indexDesc = TensorDesc({5}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 29);
    auto indicesDesc = TensorListDesc({indexDesc});
    auto outDesc = TensorDesc({5, 9}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnIndex, INPUT(selfDesc, indicesDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(TestAclnnIndex, l2_index_rank_over_8)
{
    auto selfDesc = TensorDesc({1, 1, 1, 1, 1, 1, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 4);
    auto indexDesc = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND).ValueRange(0, 0);
    auto indicesDesc = TensorListDesc({indexDesc});
    auto outDesc = TensorDesc({1, 1, 1, 1, 1, 1, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnIndex, INPUT(selfDesc, indicesDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(TestAclnnIndex, l2_index_private_format_rejected)
{
    auto selfDesc = TensorDesc({30, 10}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 4);
    auto indexDesc = TensorDesc({5}, ACL_INT64, ACL_FORMAT_FRACTAL_NZ).ValueRange(0, 29);
    auto indicesDesc = TensorListDesc({indexDesc});
    auto outDesc = TensorDesc({5, 10}, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnIndex, INPUT(selfDesc, indicesDesc), OUTPUT(outDesc));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

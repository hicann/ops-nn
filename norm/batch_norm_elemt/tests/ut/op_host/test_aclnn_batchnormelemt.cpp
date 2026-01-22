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
#include <array>
#include <vector>

#include "../../../op_host/op_api/aclnn_batch_norm_elemt.h"

#include "op_api_ut_common/array_desc.h"
#include "op_api_ut_common/inner/types.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;

class l2BatchNormElemtTest : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "l2BatchNormElemtTest SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "l2BatchNormElemtTest TearDown" << endl;
    }
};

TEST_F(l2BatchNormElemtTest, l2_batch_norm_elemt_float16_4d)
{
    auto selfDesc =
        TensorDesc({1, 4, 2, 4}, ACL_FLOAT16, ACL_FORMAT_NCHW).Value(vector<float>{1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3,
                                                                                   4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2,
                                                                                   3, 4, 1, 2, 3, 4, 1, 2, 3, 4});
    auto weightDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4});
    auto biasDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{10, 11, 12, 13});
    auto MeanDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 3, 4, 5});
    auto InvstdDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 3, 4, 5});

    auto outDesc = TensorDesc({1, 4, 2, 4}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto ut = OP_API_UT(
        aclnnBatchNormElemt, INPUT(selfDesc, weightDesc, biasDesc, MeanDesc, InvstdDesc, 1e-5), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);

    // ut.TestPrecision();
}

/*
TEST_F(l2BatchNormElemtTest, l2_batch_norm_elemt_float16_3d)
{
    auto selfDesc = TensorDesc({2, 4, 2}, ACL_FLOAT16, ACL_FORMAT_ND)
                        .Value(vector<float>{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4});
    auto weightDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4});
    auto biasDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{10, 11, 12, 13});
    auto MeanDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 3, 4, 5});
    auto InvstdDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 3, 4, 5});

    auto outDesc = TensorDesc({2, 4, 2}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnBatchNormElemt, INPUT(selfDesc, weightDesc, biasDesc, MeanDesc, InvstdDesc, 1e-5), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);

    // ut.TestPrecision();
}

TEST_F(l2BatchNormElemtTest, l2_batch_norm_elemt_float16_2d)
{
    auto selfDesc = TensorDesc({2, 4}, ACL_FLOAT16, ACL_FORMAT_ND).Value(vector<float>{1, 1, 2, 2, 3, 3, 4, 4});
    auto weightDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4});
    auto biasDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{10, 11, 12, 13});
    auto MeanDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 3, 4, 5});
    auto InvstdDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 3, 4, 5});

    auto outDesc = TensorDesc({2, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnBatchNormElemt, INPUT(selfDesc, weightDesc, biasDesc, MeanDesc, InvstdDesc, 1e-5), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);

    // ut.TestPrecision();
}

TEST_F(l2BatchNormElemtTest, l2_batch_norm_elemt_float32_4d)
{
    auto selfDesc =
        TensorDesc({2, 2, 2, 4}, ACL_FLOAT, ACL_FORMAT_NCHW).Value(vector<float>{1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3,
                                                                                 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2,
                                                                                 3, 4, 1, 2, 3, 4, 1, 2, 3, 4});
    auto weightDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2});
    auto biasDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{10, 11});
    auto MeanDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 3});
    auto InvstdDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 3});

    auto outDesc = TensorDesc({2, 2, 2, 4}, ACL_FLOAT, ACL_FORMAT_NCHW);
    auto ut = OP_API_UT(
        aclnnBatchNormElemt, INPUT(selfDesc, weightDesc, biasDesc, MeanDesc, InvstdDesc, 1e-5), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);

    // ut.TestPrecision();
}

TEST_F(l2BatchNormElemtTest, l2_batch_norm_elemt_float32_3d)
{
    auto selfDesc =
        TensorDesc({3, 4, 2}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                                                                            4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6});
    auto weightDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4});
    auto biasDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{10, 11, 12, 13});
    auto MeanDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 3, 4, 5});
    auto InvstdDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 3, 4, 5});

    auto outDesc = TensorDesc({3, 4, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnBatchNormElemt, INPUT(selfDesc, weightDesc, biasDesc, MeanDesc, InvstdDesc, 1e-5), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);

    // ut.TestPrecision();
}

TEST_F(l2BatchNormElemtTest, l2_batch_norm_elemt_float32_2d)
{
    auto selfDesc =
        TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6});
    auto weightDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4});
    auto biasDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{10, 11, 12, 13});
    auto MeanDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 3, 4, 5});
    auto InvstdDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 3, 4, 5});

    auto outDesc = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnBatchNormElemt, INPUT(selfDesc, weightDesc, biasDesc, MeanDesc, InvstdDesc, 1e-5), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);

    // ut.TestPrecision();
}

TEST_F(l2BatchNormElemtTest, l2_batch_norm_elemt_float32_2d_discontinous)
{
    auto selfDesc = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 3}, 0, {4, 3})
                        .Value(vector<float>{1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6});
    auto weightDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4});
    auto biasDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{10, 11, 12, 13});
    auto MeanDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 3, 4, 5});
    auto InvstdDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 3, 4, 5});

    auto outDesc = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnBatchNormElemt, INPUT(selfDesc, weightDesc, biasDesc, MeanDesc, InvstdDesc, 1e-5), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);

    // ut.TestPrecision();
}

TEST_F(l2BatchNormElemtTest, l2_batch_norm_elemt_float32_2d_param_nullptr)
{
    auto selfDesc =
        TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6});
    auto outDesc = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto MeanDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 3, 4, 5});
    auto InvstdDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 3, 4, 5});
    auto ut =
        OP_API_UT(aclnnBatchNormElemt, INPUT(selfDesc, nullptr, nullptr, MeanDesc, InvstdDesc, 1e-5), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

TEST_F(l2BatchNormElemtTest, l2_batch_norm_elemt_float32_6d)
{
    auto selfDesc = TensorDesc({2, 3, 4, 5, 6, 7}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(40, 50);
    auto MeanDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 3, 4});
    auto InvstdDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 3, 4});
    auto outDesc = TensorDesc({2, 3, 4, 5, 6, 7}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnBatchNormElemt, INPUT(selfDesc, nullptr, nullptr, MeanDesc, InvstdDesc, 1e-5), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

TEST_F(l2BatchNormElemtTest, l2_batch_norm_elemt_float32_invstd_dtype_invalid)
{
    auto selfDesc =
        TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6});
    auto weightDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4});
    auto biasDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{10, 11, 12, 13});
    auto MeanDesc = TensorDesc({4}, ACL_INT32, ACL_FORMAT_ND).Value(vector<float>{2, 3, 4, 5});
    auto InvstdDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 3, 4, 5});

    auto outDesc = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnBatchNormElemt, INPUT(selfDesc, weightDesc, biasDesc, MeanDesc, InvstdDesc, 1e-5), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2BatchNormElemtTest, l2_batch_norm_elemt_float32_invstd_dim_invalid)
{
    auto selfDesc =
        TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6});
    auto weightDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4});
    auto biasDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{10, 11, 12, 13});
    auto MeanDesc = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 3, 4, 5});
    auto InvstdDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 3, 4, 5});

    auto outDesc = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnBatchNormElemt, INPUT(selfDesc, weightDesc, biasDesc, MeanDesc, InvstdDesc, 1e-5), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2BatchNormElemtTest, l2_batch_norm_elemt_float32_invstd_channel_size_zero)
{
    auto selfDesc =
        TensorDesc({3, 0}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6});
    auto weightDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4});
    auto biasDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{10, 11, 12, 13});
    auto MeanDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 3, 4, 5});
    auto InvstdDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 3, 4, 5});

    auto outDesc = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnBatchNormElemt, INPUT(selfDesc, weightDesc, biasDesc, MeanDesc, InvstdDesc, 1e-5), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2BatchNormElemtTest, l2_batch_norm_elemt_invstd_shape_invalid)
{
    auto selfDesc =
        TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6});
    auto weightDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4});
    auto biasDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{10, 11, 12, 13});
    auto MeanDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 3, 4, 5});
    auto InvstdDesc = TensorDesc({5}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 3, 4, 5});

    auto outDesc = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnBatchNormElemt, INPUT(selfDesc, weightDesc, biasDesc, MeanDesc, InvstdDesc, 1e-5), OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}
*/
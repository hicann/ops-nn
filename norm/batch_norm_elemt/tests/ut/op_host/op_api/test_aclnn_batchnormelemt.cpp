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
#include <limits>
#include <vector>

#include "../../../../op_host/op_api/aclnn_batch_norm_elemt.h"

#include "op_api_ut_common/array_desc.h"
#include "op_api_ut_common/inner/types.h"
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace std;

class l2BatchNormElemtTest : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "l2BatchNormElemtTest SetUp" << endl; }

    static void TearDownTestCase() { cout << "l2BatchNormElemtTest TearDown" << endl; }
};

TEST_F(l2BatchNormElemtTest, l2_batch_norm_elemt_float16_4d)
{
    auto selfDesc = TensorDesc({1, 4, 2, 4}, ACL_FLOAT16, ACL_FORMAT_NCHW).Value(vector<float>{1, 2, 3, 4, 1, 2, 3, 4,
                                                                                               1, 2, 3, 4, 1, 2, 3, 4,
                                                                                               1, 2, 3, 4, 1, 2, 3, 4,
                                                                                               1, 2, 3, 4, 1, 2, 3, 4});
    auto weightDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4});
    auto biasDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{10, 11, 12, 13});
    auto MeanDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 3, 4, 5});
    auto InvstdDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 3, 4, 5});

    auto outDesc = TensorDesc({1, 4, 2, 4}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    auto ut = OP_API_UT(aclnnBatchNormElemt, INPUT(selfDesc, weightDesc, biasDesc, MeanDesc, InvstdDesc, 1e-5),
                        OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);

    // ut.TestPrecision();
}

// Ascend 950 及后续 Regbase 芯片：invstd 为负时符号语义参与计算，eps 不参与计算
TEST_F(l2BatchNormElemtTest, l2_batch_norm_elemt_ascend950_negative_invstd)
{
    op::NpuArchManager archManager(NpuArch::DAV_3510);
    auto selfDesc = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND)
                        .Value(vector<float>{1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6});
    auto weightDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4});
    auto biasDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{10, 11, 12, 13});
    auto meanDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 3, 4, 5});
    auto invstdDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, -3, 4, -5});

    auto outDesc = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnBatchNormElemt, INPUT(selfDesc, weightDesc, biasDesc, meanDesc, invstdDesc, 1e-5),
                        OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

// invstd = 0 时输出为 bias
TEST_F(l2BatchNormElemtTest, l2_batch_norm_elemt_ascend950_invstd_zero)
{
    op::NpuArchManager archManager(NpuArch::DAV_3510);
    auto selfDesc = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND)
                        .Value(vector<float>{1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6});
    auto weightDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4});
    auto biasDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{10, 11, 12, 13});
    auto meanDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 3, 4, 5});
    auto invstdDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{0, 0, 0, 0});

    auto outDesc = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnBatchNormElemt, INPUT(selfDesc, weightDesc, biasDesc, meanDesc, invstdDesc, 1e-5),
                        OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

// eps 为兼容保留参数，NaN/±inf/大值均不参与计算
TEST_F(l2BatchNormElemtTest, l2_batch_norm_elemt_ascend950_eps_nan)
{
    op::NpuArchManager archManager(NpuArch::DAV_3510);
    auto selfDesc = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND)
                        .Value(vector<float>{1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6});
    auto weightDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4});
    auto biasDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{10, 11, 12, 13});
    auto meanDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 3, 4, 5});
    auto invstdDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 3, 4, 5});

    auto outDesc = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnBatchNormElemt,
        INPUT(selfDesc, weightDesc, biasDesc, meanDesc, invstdDesc, numeric_limits<double>::quiet_NaN()),
        OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

TEST_F(l2BatchNormElemtTest, l2_batch_norm_elemt_ascend950_eps_inf)
{
    op::NpuArchManager archManager(NpuArch::DAV_3510);
    auto selfDesc = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND)
                        .Value(vector<float>{1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6});
    auto weightDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4});
    auto biasDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{10, 11, 12, 13});
    auto meanDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 3, 4, 5});
    auto invstdDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 3, 4, 5});

    auto outDesc = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnBatchNormElemt,
                        INPUT(selfDesc, weightDesc, biasDesc, meanDesc, invstdDesc, numeric_limits<double>::infinity()),
                        OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

// fp16 统计量 + 大 invstd（1/invstd² 在 fp16 下溢为 0），统计量统一转 fp32 后规避
TEST_F(l2BatchNormElemtTest, l2_batch_norm_elemt_ascend950_fp16_stats_large_invstd)
{
    op::NpuArchManager archManager(NpuArch::DAV_3510);
    auto selfDesc = TensorDesc({3, 4}, ACL_FLOAT16, ACL_FORMAT_ND)
                        .Value(vector<float>{1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6});
    auto weightDesc = TensorDesc({4}, ACL_FLOAT16, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4});
    auto biasDesc = TensorDesc({4}, ACL_FLOAT16, ACL_FORMAT_ND).Value(vector<float>{10, 11, 12, 13});
    auto meanDesc = TensorDesc({4}, ACL_FLOAT16, ACL_FORMAT_ND).Value(vector<float>{2, 3, 4, 5});
    auto invstdDesc = TensorDesc({4}, ACL_FLOAT16, ACL_FORMAT_ND).Value(vector<float>{10000, 20000, 30000, 40000});

    auto outDesc = TensorDesc({3, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnBatchNormElemt, INPUT(selfDesc, weightDesc, biasDesc, meanDesc, invstdDesc, 1e-5),
                        OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

// NDHWC 存储格式（通道在末维）5 维
TEST_F(l2BatchNormElemtTest, l2_batch_norm_elemt_ascend950_ndhwc_5d)
{
    op::NpuArchManager archManager(NpuArch::DAV_3510);
    auto selfDesc = TensorDesc({1, 2, 3, 4, 8}, ACL_FLOAT, ACL_FORMAT_NDHWC).ValueRange(-2, 2);
    auto weightDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto biasDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto meanDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto invstdDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);

    auto outDesc = TensorDesc({1, 2, 3, 4, 8}, ACL_FLOAT, ACL_FORMAT_NDHWC);
    auto ut = OP_API_UT(aclnnBatchNormElemt, INPUT(selfDesc, weightDesc, biasDesc, meanDesc, invstdDesc, 1e-5),
                        OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

// NDHWC 存储格式下通道取末维，weight 长度按 dim1 校验应报错
TEST_F(l2BatchNormElemtTest, l2_batch_norm_elemt_ascend950_ndhwc_weight_shape_invalid)
{
    op::NpuArchManager archManager(NpuArch::DAV_3510);
    auto selfDesc = TensorDesc({1, 2, 3, 4, 8}, ACL_FLOAT, ACL_FORMAT_NDHWC).ValueRange(-2, 2);
    auto weightDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto biasDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto meanDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto invstdDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);

    auto outDesc = TensorDesc({1, 2, 3, 4, 8}, ACL_FLOAT, ACL_FORMAT_NDHWC);
    auto ut = OP_API_UT(aclnnBatchNormElemt, INPUT(selfDesc, weightDesc, biasDesc, meanDesc, invstdDesc, 1e-5),
                        OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 8 维输入（>5 维压 5D）
TEST_F(l2BatchNormElemtTest, l2_batch_norm_elemt_ascend950_8d)
{
    op::NpuArchManager archManager(NpuArch::DAV_3510);
    auto selfDesc = TensorDesc({2, 3, 4, 5, 6, 7, 8, 9}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(40, 50);
    auto weightDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3});
    auto biasDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{10, 11, 12});
    auto meanDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 3, 4});
    auto invstdDesc = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 3, 4});

    auto outDesc = TensorDesc({2, 3, 4, 5, 6, 7, 8, 9}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnBatchNormElemt, INPUT(selfDesc, weightDesc, biasDesc, meanDesc, invstdDesc, 1e-5),
                        OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

// weight/bias 为空时按 w=1、b=0 计算
TEST_F(l2BatchNormElemtTest, l2_batch_norm_elemt_ascend950_weight_bias_nullptr)
{
    op::NpuArchManager archManager(NpuArch::DAV_3510);
    auto selfDesc = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND)
                        .Value(vector<float>{1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6});
    auto meanDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 3, 4, 5});
    auto invstdDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 3, 4, 5});

    auto outDesc = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnBatchNormElemt, INPUT(selfDesc, nullptr, nullptr, meanDesc, invstdDesc, 1e-5),
                        OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

// 非 Regbase SoC 保留原实现，负 invstd 入参下仍可正常构建执行图
TEST_F(l2BatchNormElemtTest, l2_batch_norm_elemt_ascend910b_negative_invstd)
{
    auto selfDesc = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND)
                        .Value(vector<float>{1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6});
    auto weightDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{1, 2, 3, 4});
    auto biasDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{10, 11, 12, 13});
    auto meanDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, 3, 4, 5});
    auto invstdDesc = TensorDesc({4}, ACL_FLOAT, ACL_FORMAT_ND).Value(vector<float>{2, -3, 4, -5});

    auto outDesc = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnBatchNormElemt, INPUT(selfDesc, weightDesc, biasDesc, meanDesc, invstdDesc, 1e-5),
                        OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

// NHWC 存储格式（通道在末维）4 维
TEST_F(l2BatchNormElemtTest, l2_batch_norm_elemt_ascend950_nhwc_4d)
{
    op::NpuArchManager archManager(NpuArch::DAV_3510);
    auto selfDesc = TensorDesc({2, 2, 2, 8}, ACL_FLOAT, ACL_FORMAT_NHWC).ValueRange(-2, 2);
    auto weightDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto biasDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto meanDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto invstdDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);

    auto outDesc = TensorDesc({2, 2, 2, 8}, ACL_FLOAT, ACL_FORMAT_NHWC);
    auto ut = OP_API_UT(aclnnBatchNormElemt, INPUT(selfDesc, weightDesc, biasDesc, meanDesc, invstdDesc, 1e-5),
                        OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

// NDHWC/NHWC 与维度数不匹配时应拒绝（NDHWC 对应 5 维，NHWC 对应 4 维）
TEST_F(l2BatchNormElemtTest, l2_batch_norm_elemt_ascend950_ndhwc_6d_invalid)
{
    op::NpuArchManager archManager(NpuArch::DAV_3510);
    auto selfDesc = TensorDesc({2, 2, 2, 2, 2, 8}, ACL_FLOAT, ACL_FORMAT_NDHWC).ValueRange(-2, 2);
    auto weightDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto biasDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto meanDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto invstdDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);

    auto outDesc = TensorDesc({2, 2, 2, 2, 2, 8}, ACL_FLOAT, ACL_FORMAT_NDHWC);
    auto ut = OP_API_UT(aclnnBatchNormElemt, INPUT(selfDesc, weightDesc, biasDesc, meanDesc, invstdDesc, 1e-5),
                        OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2BatchNormElemtTest, l2_batch_norm_elemt_ascend950_nhwc_5d_invalid)
{
    op::NpuArchManager archManager(NpuArch::DAV_3510);
    auto selfDesc = TensorDesc({2, 2, 2, 2, 8}, ACL_FLOAT, ACL_FORMAT_NHWC).ValueRange(-2, 2);
    auto weightDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto biasDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto meanDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto invstdDesc = TensorDesc({8}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);

    auto outDesc = TensorDesc({2, 2, 2, 2, 8}, ACL_FLOAT, ACL_FORMAT_NHWC);
    auto ut = OP_API_UT(aclnnBatchNormElemt, INPUT(selfDesc, weightDesc, biasDesc, meanDesc, invstdDesc, 1e-5),
                        OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 非 Regbase SoC 不受维度-格式校验影响（6 维 NDHWC 走原实现的压 5D 路径）
TEST_F(l2BatchNormElemtTest, l2_batch_norm_elemt_ascend910b_ndhwc_6d)
{
    auto selfDesc = TensorDesc({2, 2, 2, 2, 2, 8}, ACL_FLOAT, ACL_FORMAT_NDHWC).ValueRange(-2, 2);
    auto weightDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto biasDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto meanDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto invstdDesc = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);

    auto outDesc = TensorDesc({2, 2, 2, 2, 2, 8}, ACL_FLOAT, ACL_FORMAT_NDHWC);
    auto ut = OP_API_UT(aclnnBatchNormElemt, INPUT(selfDesc, weightDesc, biasDesc, meanDesc, invstdDesc, 1e-5),
                        OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

// bf16 输入与统计量
TEST_F(l2BatchNormElemtTest, l2_batch_norm_elemt_ascend950_bfloat16)
{
    op::NpuArchManager archManager(NpuArch::DAV_3510);
    auto selfDesc = TensorDesc({3, 4}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto weightDesc = TensorDesc({4}, ACL_BF16, ACL_FORMAT_ND).ValueRange(1, 2);
    auto biasDesc = TensorDesc({4}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto meanDesc = TensorDesc({4}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto invstdDesc = TensorDesc({4}, ACL_BF16, ACL_FORMAT_ND).ValueRange(1, 2);

    auto outDesc = TensorDesc({3, 4}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnBatchNormElemt, INPUT(selfDesc, weightDesc, biasDesc, meanDesc, invstdDesc, 1e-5),
                        OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

// 混合 dtype：fp32 输入 + fp16 统计量（统计量统一转 fp32）
TEST_F(l2BatchNormElemtTest, l2_batch_norm_elemt_ascend950_mixed_dtype_fp32_in_fp16_stats)
{
    op::NpuArchManager archManager(NpuArch::DAV_3510);
    auto selfDesc = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto weightDesc = TensorDesc({4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(1, 2);
    auto biasDesc = TensorDesc({4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto meanDesc = TensorDesc({4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto invstdDesc = TensorDesc({4}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(1, 2);

    auto outDesc = TensorDesc({3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnBatchNormElemt, INPUT(selfDesc, weightDesc, biasDesc, meanDesc, invstdDesc, 1e-5),
                        OUTPUT(outDesc));

    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
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

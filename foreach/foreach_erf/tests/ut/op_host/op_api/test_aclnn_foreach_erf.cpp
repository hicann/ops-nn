/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include <array>
#include <float.h>
#include "gtest/gtest.h"
#include "../../../../op_host/op_api/aclnn_foreach_erf.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace std;

class l2_foreach_erf_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "foreach_erf_test SetUp" << endl; }

    static void TearDownTestCase() { cout << "foreach_erf_test TearDown" << endl; }
};

// support non-contiguous input and output
TEST_F(l2_foreach_erf_test, ascend910B2_foreach_erf_test_non_contiguous)
{
    vector<int64_t> selfDims = {2, 4};
    vector<int64_t> outDims = {2, 4};
    auto x = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2}).ValueRange(-1, 1);
    auto out = TensorDesc(outDims, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2}).Precision(0.001, 0.001);
    auto xList = TensorListDesc({x});
    auto outList = TensorListDesc({out});

    auto ut = OP_API_UT(aclnnForeachErf, INPUT(xList), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

// ===== 入口校验拒绝路径/边界正向 =====

// 连续输入输出正向: 非连续转换由既有用例覆盖, 此处覆盖直接使用分支
TEST_F(l2_foreach_erf_test, ascend910B2_foreach_erf_test_contiguous)
{
    vector<int64_t> selfDims = {2, 4};
    vector<int64_t> outDims = {2, 4};
    auto x = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out = TensorDesc(outDims, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto xList = TensorListDesc({x});
    auto outList = TensorListDesc({out});

    auto ut = OP_API_UT(aclnnForeachErf, INPUT(xList), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

// 输入空指针拒绝
TEST_F(l2_foreach_erf_test, ascend910B2_foreach_erf_test_null_input)
{
    vector<int64_t> outDims = {2, 4};
    auto out = TensorDesc(outDims, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto outList = TensorListDesc({out});

    auto ut = OP_API_UT(aclnnForeachErf, INPUT(nullptr), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_NULLPTR);
}

// 输入/输出列表成员数不一致拒绝
TEST_F(l2_foreach_erf_test, ascend910B2_foreach_erf_test_list_size_mismatch)
{
    vector<int64_t> selfDims = {2, 4};
    auto x0 = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto x1 = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto xList = TensorListDesc({x0, x1});
    auto outList = TensorListDesc({TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001)});

    auto ut = OP_API_UT(aclnnForeachErf, INPUT(xList), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 单个列表超过 256 个 Tensor 拒绝(与 tiling 容量上限一致)
TEST_F(l2_foreach_erf_test, ascend910B2_foreach_erf_test_tensor_num_exceed)
{
    vector<int64_t> selfDims = {2, 4};
    auto x = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto xList = TensorListDesc(257, x);
    auto outList = TensorListDesc(257, out);

    auto ut = OP_API_UT(aclnnForeachErf, INPUT(xList), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 空 Tensor 列表: 直接成功且 workspace 为 0(空列表 SUCCESS 契约)。
// op_api_ut 框架的 TensorListDesc 构造断言 size>0, 空列表须经 aclCreateTensorList 原生构造
TEST_F(l2_foreach_erf_test, ascend910B2_foreach_erf_test_empty_tensor_list)
{
    auto* emptyX = aclCreateTensorList(nullptr, 0);
    auto* emptyOut = aclCreateTensorList(nullptr, 0);
    ASSERT_NE(emptyX, nullptr);
    ASSERT_NE(emptyOut, nullptr);

    auto ut = OP_API_UT(aclnnForeachErf, INPUT(emptyX), OUTPUT(emptyOut));
    uint64_t workspaceSize = 1;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
    EXPECT_EQ(workspaceSize, 0U);
}

// 列表内 dtype 不一致拒绝(README 约束"所有 Tensor 数据类型保持一致")
TEST_F(l2_foreach_erf_test, ascend910B2_foreach_erf_test_dtype_mismatch_in_list)
{
    vector<int64_t> selfDims = {2, 4};
    auto x0 = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto x1 = TensorDesc(selfDims, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto xList = TensorListDesc({x0, x1});
    auto outList = TensorListDesc(2, TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001));

    auto ut = OP_API_UT(aclnnForeachErf, INPUT(xList), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 输出 dtype 与输入不一致拒绝
TEST_F(l2_foreach_erf_test, ascend910B2_foreach_erf_test_dtype_mismatch_out)
{
    vector<int64_t> selfDims = {2, 4};
    auto x = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto xList = TensorListDesc({x});
    auto outList = TensorListDesc({TensorDesc(selfDims, ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001)});

    auto ut = OP_API_UT(aclnnForeachErf, INPUT(xList), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 不支持的 dtype(INT32)拒绝
TEST_F(l2_foreach_erf_test, ascend910B2_foreach_erf_test_dtype_not_supported)
{
    vector<int64_t> selfDims = {2, 4};
    auto x = TensorDesc(selfDims, ACL_INT32, ACL_FORMAT_ND);
    auto xList = TensorListDesc({x});
    auto outList = TensorListDesc({TensorDesc(selfDims, ACL_INT32, ACL_FORMAT_ND)});

    auto ut = OP_API_UT(aclnnForeachErf, INPUT(xList), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 输出 shape 与输入不一致拒绝(实现为逐对全等校验)
TEST_F(l2_foreach_erf_test, ascend910B2_foreach_erf_test_shape_not_equal)
{
    vector<int64_t> selfDims = {2, 2};
    vector<int64_t> outDims = {2, 4};
    auto x = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto xList = TensorListDesc({x});
    auto outList = TensorListDesc({TensorDesc(outDims, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001)});

    auto ut = OP_API_UT(aclnnForeachErf, INPUT(xList), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 深 shape 输入放行(ND/elewise 逐元素族不设算子级 shape 限制, shape 自由度由框架层承载):
// 9 轴形状张量经首段校验与执行器构建应成功, 任意 shape 支持面的正向验证
TEST_F(l2_foreach_erf_test, ascend910B2_foreach_erf_test_deep_shape)
{
    vector<int64_t> selfDims = {2, 2, 2, 2, 2, 2, 2, 2, 2};
    auto x = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto xList = TensorListDesc({x});
    auto outList = TensorListDesc({TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001)});

    auto ut = OP_API_UT(aclnnForeachErf, INPUT(xList), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

// 不支持的产品形态拒绝(310 架构不在 dtype 支持门控内)
TEST_F(l2_foreach_erf_test, ascend310_foreach_erf_test_unsupported_soc)
{
    op::SocVersionManager versionManager(op::SocVersion::ASCEND310);
    vector<int64_t> selfDims = {2, 4};
    auto x = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto xList = TensorListDesc({x});
    auto outList = TensorListDesc({TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001)});

    auto ut = OP_API_UT(aclnnForeachErf, INPUT(xList), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

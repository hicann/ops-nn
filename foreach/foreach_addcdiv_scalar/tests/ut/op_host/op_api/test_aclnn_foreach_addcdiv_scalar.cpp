/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include "gtest/gtest.h"

#include "../../../../op_host/op_api/aclnn_foreach_addcdiv_scalar.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include <iostream>

using namespace std;

class l2_foreach_addcdiv_scalar_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "foreach_addcdiv_scalar_test SetUp" << endl; }

    static void TearDownTestCase() { cout << "foreach_addcdiv_scalar_test TearDown" << endl; }
};

// V1 ND 正例：文档口径仅 ND，ND 输入应通过
TEST_F(l2_foreach_addcdiv_scalar_test, ascend910B2_foreach_addcdiv_scalar_test_fp32_nd)
{
    vector<vector<int64_t>> selfDims = {{2, 2}};
    vector<int64_t> scalarDims = {1};
    auto x = TensorDesc(selfDims[0], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1.0, 1.0);
    auto x2 = TensorDesc(selfDims[0], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1.0, 1.0);
    auto x3 = TensorDesc(selfDims[0], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto scalar = TensorDesc(scalarDims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1.0, 2.0);
    auto out = TensorDesc(selfDims[0], ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto xList = TensorListDesc({x});
    auto x2List = TensorListDesc({x2});
    auto x3List = TensorListDesc({x3});
    auto outList = TensorListDesc({out});

    auto ut = OP_API_UT(aclnnForeachAddcdivScalar, INPUT(xList, x2List, x3List, scalar), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

// V1 公共非ND格式（NCHW）：文档口径仅 ND，应拒绝（守护 V1 契约，防止格式校验回归）
TEST_F(l2_foreach_addcdiv_scalar_test, ascend910B2_foreach_addcdiv_scalar_test_nchw_format)
{
    vector<vector<int64_t>> selfDims = {{2, 2}};
    vector<int64_t> scalarDims = {1};
    auto x = TensorDesc(selfDims[0], ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-1.0, 1.0);
    auto x2 = TensorDesc(selfDims[0], ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(-1.0, 1.0);
    auto x3 = TensorDesc(selfDims[0], ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(0.1, 1.0);
    auto scalar = TensorDesc(scalarDims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1.0, 2.0);
    auto out = TensorDesc(selfDims[0], ACL_FLOAT, ACL_FORMAT_NCHW).Precision(0.001, 0.001);
    auto xList = TensorListDesc({x});
    auto x2List = TensorListDesc({x2});
    auto x3List = TensorListDesc({x3});
    auto outList = TensorListDesc({out});

    auto ut = OP_API_UT(aclnnForeachAddcdivScalar, INPUT(xList, x2List, x3List, scalar), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// V1 scalar 非ND格式：应拒绝
TEST_F(l2_foreach_addcdiv_scalar_test, ascend910B2_foreach_addcdiv_scalar_test_scalar_nchw_format)
{
    vector<vector<int64_t>> selfDims = {{2, 2}};
    vector<int64_t> scalarDims = {1};
    auto x = TensorDesc(selfDims[0], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1.0, 1.0);
    auto x2 = TensorDesc(selfDims[0], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1.0, 1.0);
    auto x3 = TensorDesc(selfDims[0], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto scalar = TensorDesc(scalarDims, ACL_FLOAT, ACL_FORMAT_NCHW).ValueRange(1.0, 2.0);
    auto out = TensorDesc(selfDims[0], ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto xList = TensorListDesc({x});
    auto x2List = TensorListDesc({x2});
    auto x3List = TensorListDesc({x3});
    auto outList = TensorListDesc({out});

    auto ut = OP_API_UT(aclnnForeachAddcdivScalar, INPUT(xList, x2List, x3List, scalar), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 增量覆盖：x1 与 x2 的 tensor 个数不一致 -> 161002（命中 x2Mismatch 分支）
TEST_F(l2_foreach_addcdiv_scalar_test, ascend910B2_foreach_addcdiv_scalar_test_x2_list_shorter)
{
    vector<int64_t> dims = {2, 2};
    vector<int64_t> scalarDims = {1};
    auto x = TensorDesc(dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1.0, 1.0);
    auto x2 = TensorDesc(dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1.0, 1.0);
    auto x3 = TensorDesc(dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto scalar = TensorDesc(scalarDims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1.0, 2.0);
    auto out = TensorDesc(dims, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto xList = TensorListDesc({x, x});
    auto x2List = TensorListDesc({x2});
    auto x3List = TensorListDesc({x3, x3});
    auto outList = TensorListDesc({out, out});

    auto ut = OP_API_UT(aclnnForeachAddcdivScalar, INPUT(xList, x2List, x3List, scalar), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

// 增量覆盖：x1 与 x3 的 tensor 个数不一致 -> 161002（x2 长度一致，走 x3Mismatch 分支）
TEST_F(l2_foreach_addcdiv_scalar_test, ascend910B2_foreach_addcdiv_scalar_test_x3_list_shorter)
{
    vector<int64_t> dims = {2, 2};
    vector<int64_t> scalarDims = {1};
    auto x = TensorDesc(dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1.0, 1.0);
    auto x2 = TensorDesc(dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1.0, 1.0);
    auto x3 = TensorDesc(dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto scalar = TensorDesc(scalarDims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1.0, 2.0);
    auto out = TensorDesc(dims, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto xList = TensorListDesc({x, x});
    auto x2List = TensorListDesc({x2, x2});
    auto x3List = TensorListDesc({x3});
    auto outList = TensorListDesc({out, out});

    auto ut = OP_API_UT(aclnnForeachAddcdivScalar, INPUT(xList, x2List, x3List, scalar), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

// 增量覆盖：x 列表比 out 长 -> 161002（"少传拒绝"）
TEST_F(l2_foreach_addcdiv_scalar_test, ascend910B2_foreach_addcdiv_scalar_test_x_list_longer_than_out)
{
    vector<int64_t> dims = {2, 2};
    vector<int64_t> scalarDims = {1};
    auto x = TensorDesc(dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1.0, 1.0);
    auto x2 = TensorDesc(dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1.0, 1.0);
    auto x3 = TensorDesc(dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto scalar = TensorDesc(scalarDims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1.0, 2.0);
    auto out = TensorDesc(dims, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto xList = TensorListDesc({x, x});
    auto x2List = TensorListDesc({x2, x2});
    auto x3List = TensorListDesc({x3, x3});
    auto outList = TensorListDesc({out});

    auto ut = OP_API_UT(aclnnForeachAddcdivScalar, INPUT(xList, x2List, x3List, scalar), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

// 增量覆盖：x 列表比 out 短 -> 放行（"多传放行"），保证新增判据不是双向相等
TEST_F(l2_foreach_addcdiv_scalar_test, ascend910B2_foreach_addcdiv_scalar_test_x_list_shorter_than_out)
{
    vector<int64_t> dims = {2, 2};
    vector<int64_t> scalarDims = {1};
    auto x = TensorDesc(dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1.0, 1.0);
    auto x2 = TensorDesc(dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1.0, 1.0);
    auto x3 = TensorDesc(dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto scalar = TensorDesc(scalarDims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1.0, 2.0);
    auto out = TensorDesc(dims, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto xList = TensorListDesc({x});
    auto x2List = TensorListDesc({x2});
    auto x3List = TensorListDesc({x3});
    auto outList = TensorListDesc({out, out});

    auto ut = OP_API_UT(aclnnForeachAddcdivScalar, INPUT(xList, x2List, x3List, scalar), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACL_SUCCESS);
}

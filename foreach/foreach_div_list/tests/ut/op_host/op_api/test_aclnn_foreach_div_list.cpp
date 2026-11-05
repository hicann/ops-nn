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

#include "../../../../op_host/op_api/aclnn_foreach_div_list.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include <iostream>

using namespace std;

class l2_foreach_div_list_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "foreach_div_list_test SetUp" << endl; }

    static void TearDownTestCase() { cout << "foreach_div_list_test TearDown" << endl; }
};

// 基线正例：x1/x2/out 同 shape、ND、fp32
TEST_F(l2_foreach_div_list_test, ascend910B2_foreach_div_list_test_fp32_nd)
{
    vector<vector<int64_t>> selfDims = {{2, 3}};
    auto x1 = TensorDesc(selfDims[0], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1.0, 1.0);
    auto x2 = TensorDesc(selfDims[0], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto out = TensorDesc(selfDims[0], ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto x1List = TensorListDesc({x1});
    auto x2List = TensorListDesc({x2});
    auto outList = TensorListDesc({out});

    auto ut = OP_API_UT(aclnnForeachDivList, INPUT(x1List, x2List), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

// out 元素数大于 x1：文档要求 shape size >=，应放行（社区 issue #37）
TEST_F(l2_foreach_div_list_test, ascend910B2_foreach_div_list_test_out_shape_larger_numel)
{
    vector<vector<int64_t>> selfDims = {{2, 3}};
    vector<vector<int64_t>> outDims = {{2, 4}};
    auto x1 = TensorDesc(selfDims[0], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1.0, 1.0);
    auto x2 = TensorDesc(selfDims[0], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto out = TensorDesc(outDims[0], ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto x1List = TensorListDesc({x1});
    auto x2List = TensorListDesc({x2});
    auto outList = TensorListDesc({out});

    auto ut = OP_API_UT(aclnnForeachDivList, INPUT(x1List, x2List), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

// out 与 x1 元素数相同但 shape 不同：应放行（社区 issue #37）
TEST_F(l2_foreach_div_list_test, ascend910B2_foreach_div_list_test_out_shape_same_numel)
{
    vector<vector<int64_t>> selfDims = {{2, 3}};
    vector<vector<int64_t>> outDims = {{3, 2}};
    auto x1 = TensorDesc(selfDims[0], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1.0, 1.0);
    auto x2 = TensorDesc(selfDims[0], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto out = TensorDesc(outDims[0], ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto x1List = TensorListDesc({x1});
    auto x2List = TensorListDesc({x2});
    auto outList = TensorListDesc({out});

    auto ut = OP_API_UT(aclnnForeachDivList, INPUT(x1List, x2List), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

// out 元素数小于 x1：应拒绝（161002）
TEST_F(l2_foreach_div_list_test, ascend910B2_foreach_div_list_test_out_shape_smaller_numel)
{
    vector<vector<int64_t>> selfDims = {{2, 4}};
    vector<vector<int64_t>> outDims = {{2, 2}};
    auto x1 = TensorDesc(selfDims[0], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1.0, 1.0);
    auto x2 = TensorDesc(selfDims[0], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto out = TensorDesc(outDims[0], ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto x1List = TensorListDesc({x1});
    auto x2List = TensorListDesc({x2});
    auto outList = TensorListDesc({out});

    auto ut = OP_API_UT(aclnnForeachDivList, INPUT(x1List, x2List), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// out 列表比 x1 短：161002 且不越界（社区 issue #35 场景，错误码守护）
TEST_F(l2_foreach_div_list_test, ascend910B2_foreach_div_list_test_out_list_shorter)
{
    vector<int64_t> dims = {2, 3};
    auto x1 = TensorDesc(dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1.0, 1.0);
    auto x2 = TensorDesc(dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto out = TensorDesc(dims, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto x1List = TensorListDesc({x1, x1});
    auto x2List = TensorListDesc({x2, x2});
    auto outList = TensorListDesc({out});

    auto ut = OP_API_UT(aclnnForeachDivList, INPUT(x1List, x2List), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// out 列表比 x1 长：161002（社区 issue #35 场景，错误码守护）
TEST_F(l2_foreach_div_list_test, ascend910B2_foreach_div_list_test_out_list_longer)
{
    vector<int64_t> dims = {2, 3};
    auto x1 = TensorDesc(dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1.0, 1.0);
    auto x2 = TensorDesc(dims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto out = TensorDesc(dims, ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto x1List = TensorListDesc({x1});
    auto x2List = TensorListDesc({x2});
    auto outList = TensorListDesc({out, out});

    auto ut = OP_API_UT(aclnnForeachDivList, INPUT(x1List, x2List), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// x2 与 x1 shape 不一致：161002
TEST_F(l2_foreach_div_list_test, ascend910B2_foreach_div_list_test_input_shape_different)
{
    vector<vector<int64_t>> selfDims = {{2, 3}};
    vector<vector<int64_t>> x2Dims = {{2, 4}};
    auto x1 = TensorDesc(selfDims[0], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1.0, 1.0);
    auto x2 = TensorDesc(x2Dims[0], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto out = TensorDesc(selfDims[0], ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto x1List = TensorListDesc({x1});
    auto x2List = TensorListDesc({x2});
    auto outList = TensorListDesc({out});

    auto ut = OP_API_UT(aclnnForeachDivList, INPUT(x1List, x2List), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 不支持的 dtype：161002（社区 issue #36 场景——host 侧参数校验为 ACLNN_ERR_PARAM_INVALID 而非 561002）
TEST_F(l2_foreach_div_list_test, ascend910B2_foreach_div_list_test_not_supported_dtype)
{
    vector<vector<int64_t>> selfDims = {{2, 3}};
    auto x1 = TensorDesc(selfDims[0], ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-1.0, 1.0);
    auto x2 = TensorDesc(selfDims[0], ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(0.1, 1.0);
    auto out = TensorDesc(selfDims[0], ACL_DOUBLE, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto x1List = TensorListDesc({x1});
    auto x2List = TensorListDesc({x2});
    auto outList = TensorListDesc({out});

    auto ut = OP_API_UT(aclnnForeachDivList, INPUT(x1List, x2List), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

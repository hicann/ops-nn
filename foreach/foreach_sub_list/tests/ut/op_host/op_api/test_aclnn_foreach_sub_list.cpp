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
#include "../../../../op_host/op_api/aclnn_foreach_sub_list.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

class l2_foreach_sub_list_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "foreach_sub_list_test SetUp" << endl; }

    static void TearDownTestCase() { cout << "foreach_sub_list_test TearDown" << endl; }
};

// support non-contiguous input and output
TEST_F(l2_foreach_sub_list_test, ascend910B2_foreach_sub_list_test_non_contiguous)
{
    vector<int64_t> selfDims = {2, 4};
    auto alpha = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    vector<int64_t> outDims = {2, 4};
    auto x1 = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2}).ValueRange(-1, 1);
    auto x2 = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2}).ValueRange(-1, 1);
    auto out = TensorDesc(outDims, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2}).Precision(0.001, 0.001);
    auto x1List = TensorListDesc({x1});
    auto x2List = TensorListDesc({x2});
    auto outList = TensorListDesc({out});

    auto ut = OP_API_UT(aclnnForeachSubList, INPUT(x1List, x2List, alpha), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

TEST_F(l2_foreach_sub_list_test, ascend910B2_foreach_sub_list_test_out_list_too_short)
{
    auto x = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto alpha = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto x1List = TensorListDesc({x, x});
    auto x2List = TensorListDesc({x, x});
    auto outList = TensorListDesc({out});
    auto ut = OP_API_UT(aclnnForeachSubList, INPUT(x1List, x2List, alpha), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_foreach_sub_list_test, ascend910B2_foreach_sub_list_test_out_list_too_long)
{
    auto x = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto alpha = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto x1List = TensorListDesc({x});
    auto x2List = TensorListDesc({x});
    auto outList = TensorListDesc({out, out});
    auto ut = OP_API_UT(aclnnForeachSubList, INPUT(x1List, x2List, alpha), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_foreach_sub_list_test, ascend910B2_foreach_sub_list_test_invalid_alpha_element_count)
{
    auto x = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto alpha = TensorDesc({2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto x1List = TensorListDesc({x});
    auto x2List = TensorListDesc({x});
    auto outList = TensorListDesc({out});
    auto ut = OP_API_UT(aclnnForeachSubList, INPUT(x1List, x2List, alpha), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

// 看护（T13）：alpha 为共享 storage 的切片视图（view(1,) / storage(2,) / offset 1）。
// tiling 按 alpha 的 storage shape 校验单元素，透传切片会被拒绝（T11 修复前 ACLNN_EXECUTE_FAILED），
// aclnn 侧必须先规整为独立连续张量再下传。
TEST_F(l2_foreach_sub_list_test, ascend910B2_foreach_sub_list_test_alpha_storage_slice)
{
    auto selfDims = vector<int64_t>{2, 4};
    auto alpha = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND, {1}, 1, {2}).ValueRange(0, 2);
    auto x1 = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto x2 = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND);
    auto x1List = TensorListDesc({x1});
    auto x2List = TensorListDesc({x2});
    auto outList = TensorListDesc({out});

    auto ut = OP_API_UT(aclnnForeachSubList, INPUT(x1List, x2List, alpha), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACL_SUCCESS);
}

// 看护（T13）：alpha 切片视图与非连续输入/输出叠加，规整链与回写链同时生效且索引对齐。
TEST_F(l2_foreach_sub_list_test, ascend910B2_foreach_sub_list_test_alpha_storage_slice_with_non_contiguous_io)
{
    auto selfDims = vector<int64_t>{2, 4};
    auto alpha = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND, {1}, 1, {2}).ValueRange(0, 2);
    auto x1 = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2}).ValueRange(-1, 1);
    auto x2 = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2}).ValueRange(-1, 1);
    auto out = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2});
    auto x1List = TensorListDesc({x1});
    auto x2List = TensorListDesc({x2});
    auto outList = TensorListDesc({out});

    auto ut = OP_API_UT(aclnnForeachSubList, INPUT(x1List, x2List, alpha), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACL_SUCCESS);
}

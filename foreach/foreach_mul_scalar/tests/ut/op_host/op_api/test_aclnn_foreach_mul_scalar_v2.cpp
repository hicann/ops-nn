/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include <array>
#include <float.h>
#include "gtest/gtest.h"

#include "../../../../op_host/op_api/aclnn_foreach_mul_scalar_v2.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include <iostream>
#include "opdev/platform.h"

using namespace std;

class l2_foreach_mul_scalar_v2_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "foreach_mul_scalar_test SetUp" << endl; }

    static void TearDownTestCase() { cout << "foreach_mul_scalar_test TearDown" << endl; }
};

// dtype DOUBLE
TEST_F(l2_foreach_mul_scalar_v2_test, Ascend910B2_foreach_mul_scalar_v2_test_double)
{
    vector<vector<int64_t>> selfDims = {{2, 2}};
    auto scalar_desc = ScalarDesc(1.0);
    vector<vector<int64_t>> outDims = {{2, 2}};
    auto x = TensorDesc(selfDims[0], ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out = TensorDesc(outDims[0], ACL_DOUBLE, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto xList = TensorListDesc({x});
    auto outList = TensorListDesc({out});

    auto ut = OP_API_UT(aclnnForeachMulScalarV2, INPUT(xList, scalar_desc), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_foreach_mul_scalar_v2_test, ascend910B2_foreach_mul_scalar_v2_test_fp32)
{
    vector<vector<int64_t>> selfDims = {{2, 2}};
    auto scalar_desc = ScalarDesc(1.0);
    vector<vector<int64_t>> outDims = {{2, 2}};
    auto x = TensorDesc(selfDims[0], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out = TensorDesc(outDims[0], ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto xList = TensorListDesc({x});
    auto outList = TensorListDesc({out});

    auto ut = OP_API_UT(aclnnForeachMulScalarV2, INPUT(xList, scalar_desc), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

TEST_F(l2_foreach_mul_scalar_v2_test, ascend910B2_foreach_mul_scalar_v2_test_fp16)
{
    vector<vector<int64_t>> selfDims = {{2, 2}};
    auto scalar_desc = ScalarDesc(1.0);
    vector<vector<int64_t>> outDims = {{2, 2}};
    auto x = TensorDesc(selfDims[0], ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out = TensorDesc(outDims[0], ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto xList = TensorListDesc({x});
    auto outList = TensorListDesc({out});

    auto ut = OP_API_UT(aclnnForeachMulScalarV2, INPUT(xList, scalar_desc), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

// A contiguous input must also support a transposed output view.
TEST_F(l2_foreach_mul_scalar_v2_test, foreach_mul_scalar_v2_test_transposed_out)
{
    auto x = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2});
    auto xList = TensorListDesc({x});
    auto outList = TensorListDesc({out});
    auto scalar = ScalarDesc(2.0);

    auto ut = OP_API_UT(aclnnForeachMulScalarV2, INPUT(xList, scalar), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACL_SUCCESS);
}

TEST_F(l2_foreach_mul_scalar_v2_test, foreach_mul_scalar_v2_test_non_contiguous_input_and_out)
{
    auto x = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2}).ValueRange(-1, 1);
    auto out = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND, {9, 2}, 1, {2, 9});
    auto xList = TensorListDesc({x});
    auto outList = TensorListDesc({out});
    auto scalar = ScalarDesc(2.0);

    auto ut = OP_API_UT(aclnnForeachMulScalarV2, INPUT(xList, scalar), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACL_SUCCESS);
}

// Keep list indices aligned across direct outputs and outputs requiring ViewCopy.
TEST_F(l2_foreach_mul_scalar_v2_test, foreach_mul_scalar_v2_test_mixed_out_layouts)
{
    auto x0 = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto x1 = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto x2 = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out0 = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out1 = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2});
    auto out2 = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND, {7, 2}, 1, {2, 7});
    auto xList = TensorListDesc({x0, x1, x2});
    auto outList = TensorListDesc({out0, out1, out2});
    auto scalar = ScalarDesc(2.0);

    auto ut = OP_API_UT(aclnnForeachMulScalarV2, INPUT(xList, scalar), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACL_SUCCESS);
}

// out and self different dtype
TEST_F(l2_foreach_mul_scalar_v2_test, Ascend910B2_foreach_mul_scalar_v2_test_dtype_different)
{
    vector<vector<int64_t>> selfDims = {{2, 2}};
    auto scalar_desc = ScalarDesc(1.0);
    vector<vector<int64_t>> outDims = {{2, 2}};
    auto x = TensorDesc(selfDims[0], ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out = TensorDesc(outDims[0], ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto xList = TensorListDesc({x});
    auto outList = TensorListDesc({out});

    auto ut = OP_API_UT(aclnnForeachMulScalarV2, INPUT(xList, scalar_desc), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// out and self different shape
TEST_F(l2_foreach_mul_scalar_v2_test, Ascend910B2_foreach_mul_scalarforeach_mul_scalar_v2_test_shape_different)
{
    vector<vector<int64_t>> selfDims = {{2, 2}};
    auto scalar_desc = ScalarDesc(1.0);
    vector<vector<int64_t>> outDims = {{2, 6}};
    auto x = TensorDesc(selfDims[0], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out = TensorDesc(outDims[0], ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto xList = TensorListDesc({x});
    auto outList = TensorListDesc({out});

    auto ut = OP_API_UT(aclnnForeachMulScalarV2, INPUT(xList, scalar_desc), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// 910a
/*TEST_F(l2_foreach_mul_scalar_v2_test, Ascend910_foreach_mul_scalar_v2_test_fp32) {
    vector<vector<int64_t>> selfDims = {{2, 2}};
    auto scalar_desc = ScalarDesc(1.0);
    vector<vector<int64_t>> outDims = {{2, 2}};
    auto x = TensorDesc(selfDims[0], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out = TensorDesc(outDims[0], ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto xList = TensorListDesc({x});
    auto outList = TensorListDesc({out});

    auto ut = OP_API_UT(aclnnForeachMulScalarV2, INPUT(xList, scalar_desc), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}*/

// private format
TEST_F(l2_foreach_mul_scalar_v2_test, Ascend910B2_foreach_mul_scalar_v2_test_format)
{
    vector<vector<int64_t>> selfDims = {{2, 2}};
    auto scalar_desc = ScalarDesc(1.0);
    vector<vector<int64_t>> outDims = {{2, 2}};
    auto x = TensorDesc(selfDims[0], ACL_BF16, ACL_FORMAT_NC1HWC0).ValueRange(-1, 1);
    auto out = TensorDesc(outDims[0], ACL_BF16, ACL_FORMAT_NC1HWC0).Precision(0.001, 0.001);
    auto xList = TensorListDesc({x});
    auto outList = TensorListDesc({out});

    auto ut = OP_API_UT(aclnnForeachMulScalarV2, INPUT(xList, scalar_desc), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

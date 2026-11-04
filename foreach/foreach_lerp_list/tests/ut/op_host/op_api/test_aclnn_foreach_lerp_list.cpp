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

#include "../../../../op_api/aclnn_foreach_lerp_list.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include <iostream>
#include "opdev/platform.h"

using namespace std;

class l2_foreach_lerp_list_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "foreach_lerp_list_test SetUp" << endl; }

    static void TearDownTestCase() { cout << "foreach_lerp_list_test TearDown" << endl; }
};

// A contiguous input must also support a transposed output view.
TEST_F(l2_foreach_lerp_list_test, foreach_lerp_list_test_transposed_out)
{
    auto x1 = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto x2 = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto weight = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto out = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2});
    auto x1List = TensorListDesc({x1});
    auto x2List = TensorListDesc({x2});
    auto weightList = TensorListDesc({weight});
    auto outList = TensorListDesc({out});

    auto ut = OP_API_UT(aclnnForeachLerpList, INPUT(x1List, x2List, weightList), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACL_SUCCESS);
}

TEST_F(l2_foreach_lerp_list_test, foreach_lerp_list_test_non_contiguous_input_and_out)
{
    auto x1 = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2}).ValueRange(-1, 1);
    auto x2 = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2}).ValueRange(1, 2);
    auto weight = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2}).ValueRange(1, 2);
    auto out = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND, {9, 2}, 1, {2, 9});
    auto x1List = TensorListDesc({x1});
    auto x2List = TensorListDesc({x2});
    auto weightList = TensorListDesc({weight});
    auto outList = TensorListDesc({out});

    auto ut = OP_API_UT(aclnnForeachLerpList, INPUT(x1List, x2List, weightList), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACL_SUCCESS);
}

// Keep list indices aligned across direct outputs and outputs requiring ViewCopy.
TEST_F(l2_foreach_lerp_list_test, foreach_lerp_list_test_mixed_out_layouts)
{
    auto x10 = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto x11 = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto x12 = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto x20 = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto x21 = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto x22 = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto w0 = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto w1 = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto w2 = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto out0 = TensorDesc({3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out1 = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2});
    auto out2 = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND, {7, 2}, 1, {2, 7});
    auto x1List = TensorListDesc({x10, x11, x12});
    auto x2List = TensorListDesc({x20, x21, x22});
    auto weightList = TensorListDesc({w0, w1, w2});
    auto outList = TensorListDesc({out0, out1, out2});

    auto ut = OP_API_UT(aclnnForeachLerpList, INPUT(x1List, x2List, weightList), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACL_SUCCESS);
}

// out is rejected with PARAM_INVALID when its view shape differs from the corresponding x1.
// x1/x2/weight are {2,4} while out is {2,3}, so out's view shape does not match and it also holds
// fewer elements.
TEST_F(l2_foreach_lerp_list_test, foreach_lerp_list_test_smaller_out_shape_size)
{
    auto x1 = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto x2 = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto weight = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto out = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto x1List = TensorListDesc({x1});
    auto x2List = TensorListDesc({x2});
    auto weightList = TensorListDesc({weight});
    auto outList = TensorListDesc({out});

    auto ut = OP_API_UT(aclnnForeachLerpList, INPUT(x1List, x2List, weightList), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

// out must keep the same view shape as x1. x1/x2/weight are {2,4} while out is {2,5}; even though
// out holds more elements (10 > 8), its view shape ({2,5}) differs from x1's ({2,4}) and is rejected.
TEST_F(l2_foreach_lerp_list_test, foreach_lerp_list_test_larger_out_view_shape_mismatch)
{
    auto x1 = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto x2 = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto weight = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto out = TensorDesc({2, 5}, ACL_FLOAT, ACL_FORMAT_ND);
    auto x1List = TensorListDesc({x1});
    auto x2List = TensorListDesc({x2});
    auto weightList = TensorListDesc({weight});
    auto outList = TensorListDesc({out});

    auto ut = OP_API_UT(aclnnForeachLerpList, INPUT(x1List, x2List, weightList), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

// out must keep the same view shape as x1 even when the element count matches:
// x1/x2/weight are {2,4} and out is {1,8}; both hold 8 elements, but out's view shape is not {2,4}.
TEST_F(l2_foreach_lerp_list_test, foreach_lerp_list_test_same_numel_diff_view_shape)
{
    auto x1 = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto x2 = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto weight = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto out = TensorDesc({1, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto x1List = TensorListDesc({x1});
    auto x2List = TensorListDesc({x2});
    auto weightList = TensorListDesc({weight});
    auto outList = TensorListDesc({out});

    auto ut = OP_API_UT(aclnnForeachLerpList, INPUT(x1List, x2List, weightList), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

// Even with enough elements, out rank above 8 is rejected with PARAM_INVALID.
// x1/x2/weight are {2,4} (8 elements) while out is a 9-D tensor holding the same 8 elements.
TEST_F(l2_foreach_lerp_list_test, foreach_lerp_list_test_out_exceeds_max_dim)
{
    auto x1 = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto x2 = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto weight = TensorDesc({2, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(1, 2);
    auto out = TensorDesc({1, 1, 1, 1, 1, 1, 1, 1, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto x1List = TensorListDesc({x1});
    auto x2List = TensorListDesc({x2});
    auto weightList = TensorListDesc({weight});
    auto outList = TensorListDesc({out});

    auto ut = OP_API_UT(aclnnForeachLerpList, INPUT(x1List, x2List, weightList), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

// TensorList length 256 is the tiling upper bound and must be accepted.
TEST_F(l2_foreach_lerp_list_test, foreach_lerp_list_test_maximum_list_length)
{
    auto x1 = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto x2 = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto weight = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto x1List = TensorListDesc(256, x1);
    auto x2List = TensorListDesc(256, x2);
    auto weightList = TensorListDesc(256, weight);
    auto outList = TensorListDesc(256, out);

    auto ut = OP_API_UT(aclnnForeachLerpList, INPUT(x1List, x2List, weightList), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACL_SUCCESS);
}

// TensorList length 257 exceeds the tiling upper bound and must be rejected with TILING_ERROR.
TEST_F(l2_foreach_lerp_list_test, foreach_lerp_list_test_excessive_list_length)
{
    auto x1 = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto x2 = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto weight = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 3}, ACL_FLOAT, ACL_FORMAT_ND);
    auto x1List = TensorListDesc(257, x1);
    auto x2List = TensorListDesc(257, x2);
    auto weightList = TensorListDesc(257, weight);
    auto outList = TensorListDesc(257, out);

    auto ut = OP_API_UT(aclnnForeachLerpList, INPUT(x1List, x2List, weightList), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_INNER_TILING_ERROR);
}

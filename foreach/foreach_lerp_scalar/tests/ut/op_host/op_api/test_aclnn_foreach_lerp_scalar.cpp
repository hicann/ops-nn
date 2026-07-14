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

#include "../../../../op_host/op_api/aclnn_foreach_lerp_scalar.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include <iostream>
#include "opdev/platform.h"

using namespace std;

class l2_foreach_lerp_scalar_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "foreach_lerp_scalar_test SetUp" << endl; }

    static void TearDownTestCase() { cout << "foreach_lerp_scalar_test TearDown" << endl; }
};

// support non-contiguous input and output
TEST_F(l2_foreach_lerp_scalar_test, ascend910B2_foreach_lerp_scalar_test_non_contiguous)
{
    vector<int64_t> selfDims = {2, 4};
    auto weight_desc = ScalarDesc(0.5);
    vector<int64_t> outDims = {2, 4};
    auto x1 = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2}).ValueRange(-1, 1);
    auto x2 = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2}).ValueRange(-1, 1);
    auto out = TensorDesc(outDims, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2}).Precision(0.001, 0.001);
    auto x1List = TensorListDesc({x1});
    auto x2List = TensorListDesc({x2});
    auto outList = TensorListDesc({out});

    auto ut = OP_API_UT(aclnnForeachLerpScalar, INPUT(x1List, x2List, weight_desc), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

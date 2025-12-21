/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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

#include "../../../op_host/op_api/aclnn_foreach_addcdiv_scalar_v2.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include <iostream>
#include "opdev/platform.h"

using namespace std;

class l2_foreach_addcdiv_scalar_v2_test : public testing::Test {
protected:
  static void SetUpTestCase() { cout << "foreach_addcdiv_scalar_test SetUp" << endl; }

  static void TearDownTestCase() { cout << "foreach_addcdiv_scalar_test TearDown" << endl; }
};

/*TEST_F(l2_foreach_addcdiv_scalar_v2_test, ascend910_foreach_addcdiv_scalar_v2_test_fp32) {
    vector<vector<int64_t>> selfDims = {{2, 2}};
    auto scalar_desc = ScalarDesc(1.0);
    vector<vector<int64_t>> outDims = {{2, 2}};
    auto x = TensorDesc(selfDims[0], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto x2 = TensorDesc(selfDims[0], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto x3 = TensorDesc(selfDims[0], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out = TensorDesc(outDims[0], ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto xList = TensorListDesc({x});
    auto x2List = TensorListDesc({x2});
    auto x3List = TensorListDesc({x3});
    auto outList = TensorListDesc({out});

    auto ut = OP_API_UT(aclnnForeachAddcdivScalarV2, INPUT(xList, x2List, x3List, scalar_desc), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}*/

TEST_F(l2_foreach_addcdiv_scalar_v2_test, ascend910B2_foreach_addcdiv_scalar_v2_test_fp32) {
    vector<vector<int64_t>> selfDims = {{2, 2}};
    auto scalar_desc = ScalarDesc(1.0);
    vector<vector<int64_t>> outDims = {{2, 2}};
    auto x = TensorDesc(selfDims[0], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto x2 = TensorDesc(selfDims[0], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto x3 = TensorDesc(selfDims[0], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out = TensorDesc(outDims[0], ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto xList = TensorListDesc({x});
    auto x2List = TensorListDesc({x2});
    auto x3List = TensorListDesc({x3});
    auto outList = TensorListDesc({out});

    auto ut = OP_API_UT(aclnnForeachAddcdivScalarV2, INPUT(xList, x2List, x3List, scalar_desc), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

TEST_F(l2_foreach_addcdiv_scalar_v2_test, ascend910B2_foreach_addcdiv_scalar_v2_test_fp16) {
    vector<vector<int64_t>> selfDims = {{2, 2}};
    auto scalar_desc = ScalarDesc(1.0);
    vector<vector<int64_t>> outDims = {{2, 2}};
    auto x = TensorDesc(selfDims[0], ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto x2 = TensorDesc(selfDims[0], ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto x3 = TensorDesc(selfDims[0], ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out = TensorDesc(outDims[0], ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto xList = TensorListDesc({x});
    auto x2List = TensorListDesc({x2});
    auto x3List = TensorListDesc({x3});
    auto outList = TensorListDesc({out});

    auto ut = OP_API_UT(aclnnForeachAddcdivScalarV2, INPUT(xList, x2List, x3List, scalar_desc), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

// private format
TEST_F(l2_foreach_addcdiv_scalar_v2_test, ascend910B2_foreach_addcdiv_scalar_v2_test_private_format) {
    vector<vector<int64_t>> selfDims = {{2, 2}};
    auto scalar_desc = ScalarDesc(1.0);
    vector<vector<int64_t>> outDims = {{2, 2}};
    auto x = TensorDesc(selfDims[0], ACL_FLOAT, ACL_FORMAT_NC1HWC0).ValueRange(-1, 1);
    auto x2 = TensorDesc(selfDims[0], ACL_FLOAT, ACL_FORMAT_NC1HWC0).ValueRange(-1, 1);
    auto x3 = TensorDesc(selfDims[0], ACL_FLOAT, ACL_FORMAT_NC1HWC0).ValueRange(-1, 1);
    auto out = TensorDesc(outDims[0], ACL_FLOAT, ACL_FORMAT_NC1HWC0).Precision(0.001, 0.001);
    auto xList = TensorListDesc({x});
    auto x2List = TensorListDesc({x2});
    auto x3List = TensorListDesc({x3});
    auto outList = TensorListDesc({out});

    auto ut = OP_API_UT(aclnnForeachAddcdivScalarV2, INPUT(xList, x2List, x3List, scalar_desc), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// self and out diffrent shape
TEST_F(l2_foreach_addcdiv_scalar_v2_test, ascend910B2_foreach_addcdiv_scalar_v2_test_shape_different) {
    vector<vector<int64_t>> selfDims = {{2, 2}};
    auto scalar_desc = ScalarDesc(1.0);
    vector<vector<int64_t>> outDims = {{2, 4}};
    auto x = TensorDesc(selfDims[0], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto x2 = TensorDesc(selfDims[0], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto x3 = TensorDesc(selfDims[0], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out = TensorDesc(outDims[0], ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto xList = TensorListDesc({x});
    auto x2List = TensorListDesc({x2});
    auto x3List = TensorListDesc({x3});
    auto outList = TensorListDesc({out});

    auto ut = OP_API_UT(aclnnForeachAddcdivScalarV2, INPUT(xList, x2List, x3List, scalar_desc), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// self and x2 diffrent shape
TEST_F(l2_foreach_addcdiv_scalar_v2_test, ascend910B2_foreach_addcdiv_scalar_v2_test_input_shape_different) {
    vector<vector<int64_t>> selfDims = {{2, 2}};
    vector<vector<int64_t>> x2Dims = {{2, 4}};
    auto scalar_desc = ScalarDesc(1.0);
    vector<vector<int64_t>> outDims = {{2, 4}};
    auto x = TensorDesc(selfDims[0], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto x2 = TensorDesc(x2Dims[0], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto x3 = TensorDesc(selfDims[0], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out = TensorDesc(outDims[0], ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto xList = TensorListDesc({x});
    auto x2List = TensorListDesc({x2});
    auto x3List = TensorListDesc({x3});
    auto outList = TensorListDesc({out});

    auto ut = OP_API_UT(aclnnForeachAddcdivScalarV2, INPUT(xList, x2List, x3List, scalar_desc), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// not supported dtype
TEST_F(l2_foreach_addcdiv_scalar_v2_test, ascend910B2_foreach_addcdiv_scalar_v2_test_not_supported_shape) {
    vector<vector<int64_t>> selfDims = {{2, 2}};
    auto scalar_desc = ScalarDesc(1.0);
    vector<vector<int64_t>> outDims = {{2, 4}};
    auto x = TensorDesc(selfDims[0], ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto x2 = TensorDesc(selfDims[0], ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto x3 = TensorDesc(selfDims[0], ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out = TensorDesc(outDims[0], ACL_DOUBLE, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto xList = TensorListDesc({x});
    auto x2List = TensorListDesc({x2});
    auto x3List = TensorListDesc({x3});
    auto outList = TensorListDesc({out});

    auto ut = OP_API_UT(aclnnForeachAddcdivScalarV2, INPUT(xList, x2List, x3List, scalar_desc), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

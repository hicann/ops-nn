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
#include <map>
#include <string>
#include "experiment/metadef/common/util/error_manager/error_manager.h"
#include <array>
#include <float.h>
#include "gtest/gtest.h"

#include "../../../../op_host/op_api/aclnn_foreach_maximum_scalar_v2.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include <iostream>
#include "opdev/platform.h"

using namespace std;

class l2_foreach_maximum_scalar_v2_test : public testing::Test {
protected:
    void SetUp() override
    {
        auto& errorManager = ErrorManager::GetInstance();
        errorManager.GenWorkStreamIdDefault();
        (void)errorManager.GetRawErrorMessages();
    }

    static void ExpectError(const char* code, const std::map<std::string, std::string>& expectedArgs)
    {
        // The host executor can also emit EE runtime context diagnostics without an NPU.
        std::vector<ErrorManager::ErrorItem> errors;
        for (const auto& error : ErrorManager::GetInstance().GetRawErrorMessages()) {
            if (error.error_id.compare(0, 2, "EZ") == 0) {
                errors.push_back(error);
            }
        }
        ASSERT_EQ(errors.size(), 1U);
        EXPECT_EQ(errors[0].error_id, code);
        for (const auto& [key, value] : expectedArgs) {
            ASSERT_NE(errors[0].args_map.count(key), 0U) << key;
            EXPECT_EQ(errors[0].args_map.at(key), value) << key;
        }
    }

    static void SetUpTestCase() { cout << "foreach_maximum_scalar_test SetUp" << endl; }

    static void TearDownTestCase() { cout << "foreach_maximum_scalar_test TearDown" << endl; }
};

// dtype DOUBLE
TEST_F(l2_foreach_maximum_scalar_v2_test, Ascend910B2_foreach_maximum_v2_test_double)
{
    vector<vector<int64_t>> selfDims = {{2, 2}};
    auto scalar_desc = ScalarDesc(1.0);
    vector<vector<int64_t>> outDims = {{2, 2}};
    auto x = TensorDesc(selfDims[0], ACL_DOUBLE, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out = TensorDesc(outDims[0], ACL_DOUBLE, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto xList = TensorListDesc({x});
    auto outList = TensorListDesc({out});

    auto ut = OP_API_UT(aclnnForeachMaximumScalarV2, INPUT(xList, scalar_desc), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// out and self different dtype
TEST_F(l2_foreach_maximum_scalar_v2_test, Ascend910B2_foreach_maximum_v2_test_dtype_different)
{
    vector<vector<int64_t>> selfDims = {{2, 2}};
    auto scalar_desc = ScalarDesc(1.0);
    vector<vector<int64_t>> outDims = {{2, 2}};
    auto x = TensorDesc(selfDims[0], ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out = TensorDesc(outDims[0], ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto xList = TensorListDesc({x});
    auto outList = TensorListDesc({out});

    auto ut = OP_API_UT(aclnnForeachMaximumScalarV2, INPUT(xList, scalar_desc), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// out and self different shape
TEST_F(l2_foreach_maximum_scalar_v2_test, Ascend910B2_foreach_maximum_v2_test_shape_different)
{
    vector<vector<int64_t>> selfDims = {{2, 2}};
    auto scalar_desc = ScalarDesc(1.0);
    vector<vector<int64_t>> outDims = {{2, 6}};
    auto x = TensorDesc(selfDims[0], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out = TensorDesc(outDims[0], ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto xList = TensorListDesc({x});
    auto outList = TensorListDesc({out});

    auto ut = OP_API_UT(aclnnForeachMaximumScalarV2, INPUT(xList, scalar_desc), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_foreach_maximum_scalar_v2_test, ascend910B2_foreach_maximum_scalar_v2_test_dtype_fp32)
{
    vector<vector<int64_t>> selfDims = {{2, 2}};
    auto scalar_desc = ScalarDesc(1.0);
    vector<vector<int64_t>> outDims = {{2, 2}};
    auto x = TensorDesc(selfDims[0], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out = TensorDesc(outDims[0], ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto xList = TensorListDesc({x});
    auto outList = TensorListDesc({out});

    auto ut = OP_API_UT(aclnnForeachMaximumScalarV2, INPUT(xList, scalar_desc), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

TEST_F(l2_foreach_maximum_scalar_v2_test, ascend910B2_foreach_maximum_scalar_v2_test_dtype_fp16)
{
    vector<vector<int64_t>> selfDims = {{2, 2}};
    auto scalar_desc = ScalarDesc(1.0);
    vector<vector<int64_t>> outDims = {{2, 2}};
    auto x = TensorDesc(selfDims[0], ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out = TensorDesc(outDims[0], ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto xList = TensorListDesc({x});
    auto outList = TensorListDesc({out});

    auto ut = OP_API_UT(aclnnForeachMaximumScalarV2, INPUT(xList, scalar_desc), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

// 910a
/*TEST_F(l2_foreach_maximum_scalar_v2_test, Ascend910_foreach_maximum_v2_test_fp32) {
    vector<vector<int64_t>> selfDims = {{2, 2}};
    auto scalar_desc = ScalarDesc(1.0);
    vector<vector<int64_t>> outDims = {{2, 2}};
    auto x = TensorDesc(selfDims[0], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto out = TensorDesc(outDims[0], ACL_FLOAT, ACL_FORMAT_ND).Precision(0.001, 0.001);
    auto xList = TensorListDesc({x});
    auto outList = TensorListDesc({out});

    auto ut = OP_API_UT(aclnnForeachMaximumScalarV2, INPUT(xList, scalar_desc), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}*/

// private format
TEST_F(l2_foreach_maximum_scalar_v2_test, Ascend910B2_foreach_maximum_v2_test_format)
{
    vector<vector<int64_t>> selfDims = {{2, 2}};
    auto scalar_desc = ScalarDesc(1.0);
    vector<vector<int64_t>> outDims = {{2, 2}};
    auto x = TensorDesc(selfDims[0], ACL_BF16, ACL_FORMAT_NC1HWC0).ValueRange(-1, 1);
    auto out = TensorDesc(outDims[0], ACL_BF16, ACL_FORMAT_NC1HWC0).Precision(0.001, 0.001);
    auto xList = TensorListDesc({x});
    auto outList = TensorListDesc({out});

    auto ut = OP_API_UT(aclnnForeachMaximumScalarV2, INPUT(xList, scalar_desc), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

// support non-contiguous input and output
TEST_F(l2_foreach_maximum_scalar_v2_test, ascend910B2_foreach_maximum_scalar_v2_test_non_contiguous)
{
    vector<int64_t> selfDims = {2, 4};
    auto scalar_desc = ScalarDesc(1.0);
    vector<int64_t> outDims = {2, 4};
    auto x = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2}).ValueRange(-1, 1);
    auto out = TensorDesc(outDims, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2}).Precision(0.001, 0.001);
    auto xList = TensorListDesc({x});
    auto outList = TensorListDesc({out});

    auto ut = OP_API_UT(aclnnForeachMaximumScalarV2, INPUT(xList, scalar_desc), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

TEST_F(l2_foreach_maximum_scalar_v2_test, rejects_short_output_list)
{
    auto x = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scalar = ScalarDesc(1.0);
    auto ut = OP_API_UT(aclnnForeachMaximumScalarV2, INPUT(TensorListDesc({x, x}), scalar),
                        OUTPUT(TensorListDesc({x})));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
    ExpectError("EZ0023",
                {{"param_names", "x, out"}, {"incorrect_nums", "2, 1"}, {"op_name", "aclnnForeachMaximumScalarV2"}});
}

TEST_F(l2_foreach_maximum_scalar_v2_test, rejects_empty_output_list)
{
    auto x = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto xList = TensorListDesc({x}).ToAclType();
    auto scalar = ScalarDesc(1.0).ToAclType();
    auto* out = aclCreateTensorList(nullptr, 0);
    ASSERT_NE(out, nullptr);
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    EXPECT_EQ(aclnnForeachMaximumScalarV2GetWorkspaceSize(xList.get(), scalar.get(), out, &workspaceSize, &executor),
              ACLNN_ERR_PARAM_INVALID);
    ExpectError("EZ0023",
                {{"param_names", "x, out"}, {"incorrect_nums", "1, 0"}, {"op_name", "aclnnForeachMaximumScalarV2"}});
    aclDestroyTensorList(out);
}

TEST_F(l2_foreach_maximum_scalar_v2_test, rejects_long_output_list)
{
    auto x = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scalar = ScalarDesc(1.0);
    auto ut = OP_API_UT(aclnnForeachMaximumScalarV2, INPUT(TensorListDesc({x}), scalar),
                        OUTPUT(TensorListDesc({x, x})));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
    ExpectError("EZ0023",
                {{"param_names", "x, out"}, {"incorrect_nums", "1, 2"}, {"op_name", "aclnnForeachMaximumScalarV2"}});
}

TEST_F(l2_foreach_maximum_scalar_v2_test, accepts_contiguous_output_view_with_offset)
{
    auto x = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto out = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND, {2, 1}, 3, {10});
    auto scalar = ScalarDesc(1.0);
    auto ut = OP_API_UT(aclnnForeachMaximumScalarV2, INPUT(TensorListDesc({x}), scalar), OUTPUT(TensorListDesc({out})));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
}

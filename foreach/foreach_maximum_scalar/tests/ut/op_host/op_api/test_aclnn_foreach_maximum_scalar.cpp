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
#include <map>
#include <string>
#include "experiment/metadef/common/util/error_manager/error_manager.h"
#include <array>
#include <float.h>
#include "gtest/gtest.h"
#include "../../../../op_host/op_api/aclnn_foreach_maximum_scalar.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

class l2_foreach_maximum_scalar_test : public testing::Test {
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

// support non-contiguous input and output
TEST_F(l2_foreach_maximum_scalar_test, ascend910B2_foreach_maximum_scalar_test_non_contiguous)
{
    vector<int64_t> selfDims = {2, 4};
    auto scalar = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 2);
    vector<int64_t> outDims = {2, 4};
    auto x = TensorDesc(selfDims, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2}).ValueRange(-1, 1);
    auto out = TensorDesc(outDims, ACL_FLOAT, ACL_FORMAT_ND, {1, 2}, 0, {4, 2}).Precision(0.001, 0.001);
    auto xList = TensorListDesc({x});
    auto outList = TensorListDesc({out});

    auto ut = OP_API_UT(aclnnForeachMaximumScalar, INPUT(xList, scalar), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(getWorkspaceResult, ACL_SUCCESS);
}

TEST_F(l2_foreach_maximum_scalar_test, rejects_mismatched_output_list)
{
    auto x = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scalar = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnForeachMaximumScalar, INPUT(TensorListDesc({x, x}), scalar), OUTPUT(TensorListDesc({x})));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
    ExpectError("EZ0023",
                {{"param_names", "x, out"}, {"incorrect_nums", "2, 1"}, {"op_name", "aclnnForeachMaximumScalar"}});
}

TEST_F(l2_foreach_maximum_scalar_test, accepts_scalar_view_with_offset_and_larger_storage)
{
    auto x = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto scalar = TensorDesc({1}, ACL_FLOAT, ACL_FORMAT_ND, {1}, 2, {4});
    auto ut = OP_API_UT(aclnnForeachMaximumScalar, INPUT(TensorListDesc({x}), scalar), OUTPUT(TensorListDesc({x})));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
}

TEST_F(l2_foreach_maximum_scalar_test, rejects_non_scalar_tensor)
{
    auto x = TensorDesc({2, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    for (int64_t size : {0, 2}) {
        auto scalar = TensorDesc({size}, ACL_FLOAT, ACL_FORMAT_ND);
        auto ut = OP_API_UT(aclnnForeachMaximumScalar, INPUT(TensorListDesc({x}), scalar), OUTPUT(TensorListDesc({x})));
        uint64_t workspaceSize = 0;
        EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
        ExpectError("EZ0014", {{"param_name", "scalar"},
                               {"incorrect_size", std::to_string(size)},
                               {"correct_size", "1"},
                               {"op_name", "aclnnForeachMaximumScalar"}});
    }
}

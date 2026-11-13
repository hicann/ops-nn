/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <iostream>
#include <vector>
#include "gtest/gtest.h"

#include "../../../op_api/aclnn_median.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace op;
using namespace std;

class l2_median_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "median_test SetUp" << endl; }

    static void TearDownTestCase() { cout << "median_test TearDown" << endl; }
};

TEST_F(l2_median_test, case_008_complex64_normal)
{
    auto selfTensor = TensorDesc({3, 3}, ACL_COMPLEX64, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto outTensor = TensorDesc({1}, ACL_COMPLEX64, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnMedian, INPUT(selfTensor), OUTPUT(outTensor));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_median_test, case_009_complex128_normal)
{
    auto selfTensor = TensorDesc({3, 3}, ACL_COMPLEX128, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto outTensor = TensorDesc({1}, ACL_COMPLEX128, ACL_FORMAT_ND).Precision(0.0001, 0.0001);

    auto ut = OP_API_UT(aclnnMedian, INPUT(selfTensor), OUTPUT(outTensor));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(l2_median_test, empty_self_rejects_invalid_output_shape)
{
    for (const auto& shape : std::vector<std::vector<int64_t>>{{8}, {0}, {0, 2}}) {
        auto selfTensor = TensorDesc({0, 2}, ACL_FLOAT, ACL_FORMAT_ND);
        auto outTensor = TensorDesc(shape, ACL_FLOAT, ACL_FORMAT_ND);
        auto ut = OP_API_UT(aclnnMedian, INPUT(selfTensor), OUTPUT(outTensor));
        uint64_t workspaceSize = 0;
        EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
    }
}

TEST_F(l2_median_test, empty_self_accepts_single_element_output_shapes)
{
    for (const auto& shape : std::vector<std::vector<int64_t>>{{}, {1}, {1, 1}}) {
        auto selfTensor = TensorDesc({0, 2}, ACL_FLOAT, ACL_FORMAT_ND);
        auto outTensor = TensorDesc(shape, ACL_FLOAT, ACL_FORMAT_ND);
        auto ut = OP_API_UT(aclnnMedian, INPUT(selfTensor), OUTPUT(outTensor));
        uint64_t workspaceSize = 0;
        EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
    }
}

TEST_F(l2_median_test, nonempty_self_rejects_empty_output)
{
    auto selfTensor = TensorDesc({4, 2}, ACL_FLOAT, ACL_FORMAT_ND);
    auto outTensor = TensorDesc({0}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMedian, INPUT(selfTensor), OUTPUT(outTensor));
    uint64_t workspaceSize = 0;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_ERR_PARAM_INVALID);
}

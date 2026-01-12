/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_celu.cpp
 * \brief
 */
#include <array>
#include <vector>
#include "gtest/gtest.h"

#include "../../../op_api/aclnn_softmax_cross_entropy_with_logits.h"

#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/tensor_desc.h"

using namespace std;

namespace {
class l2_softmax_cross_entropy_with_logits_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "softmax_cross_entropy_with_logits_test SetUp" << endl; }

    static void TearDownTestCase() { cout << "softmax_cross_entropy_with_logits_test TearDown" << endl; }
};

// 测试aicore:FLOAT32类型支持
TEST_F(l2_softmax_cross_entropy_with_logits_test, ascend910B_case_dtype_aicore_support)
{
    auto features_tensor_desc = TensorDesc({16, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto labels_tensor_desc = TensorDesc({16, 4}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto loss_tensor_desc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto backprop_tensor_desc = TensorDesc({16, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSoftmaxCrossEntropyWithLogits, INPUT(features_tensor_desc, labels_tensor_desc),
                                          OUTPUT(loss_tensor_desc, backprop_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// 测试aicore:shape不支持
TEST_F(l2_softmax_cross_entropy_with_logits_test, ascend910B_case_dtype_aicore_support_1)
{
    auto features_tensor_desc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto labels_tensor_desc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto loss_tensor_desc = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-2, 2);
    auto backprop_tensor_desc = TensorDesc({16, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnSoftmaxCrossEntropyWithLogits, INPUT(features_tensor_desc, labels_tensor_desc),
                                          OUTPUT(loss_tensor_desc, backprop_tensor_desc));
    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
}
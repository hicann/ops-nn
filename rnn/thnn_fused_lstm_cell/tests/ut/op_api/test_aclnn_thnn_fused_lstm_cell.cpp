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
#include <array>
#include "gtest/gtest.h"

#include "opdev/op_log.h"
#include "../../../op_api/aclnn_thnn_fused_lstm_cell.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class l2_thnn_fused_lstm_cell_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "l2_thnn_fused_lstm_cell_test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "l2_thnn_fused_lstm_cell_test TearDown" << std::endl;
    }
};

// 正常input场景
TEST_F(l2_thnn_fused_lstm_cell_test, ascend910B2_normal_float)
{
    int64_t B = 3;
    int64_t H = 5;
    vector<int64_t> common_shape = {B, H};
    vector<int64_t> bias_shape = {4 * H};
    vector<int64_t> gates_shape = {B, 4 * H};

    auto inputGates = TensorDesc(gates_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hiddenGates = TensorDesc(gates_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto cx = TensorDesc(common_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto inputBias = TensorDesc(bias_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hiddenBias = TensorDesc(bias_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hy = TensorDesc(common_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto cy = TensorDesc(common_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto storage = TensorDesc(gates_shape, ACL_FLOAT, ACL_FORMAT_ND);

    auto ut = OP_API_UT(
        aclnnThnnFusedLstmCell,
        INPUT(
            inputGates, hiddenGates, cx, inputBias, hiddenBias),
        OUTPUT(hy, cy, storage));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}
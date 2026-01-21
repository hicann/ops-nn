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
#include "gtest/gtest.h"

#include "opdev/op_log.h"
#include "../../../op_api/aclnn_thnn_fused_lstm_cell_backward.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class l2_thnn_fused_lstm_cell_backward_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "l2_thnn_fused_lstm_cell_backward_test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "l2_thnn_fused_lstm_cell_backward_test TearDown" << std::endl;
    }
};

// 正常input场景
TEST_F(l2_thnn_fused_lstm_cell_backward_test, ascend910B2_normal_float)
{
    vector<int64_t> dhy_shape = {1, 8};
    vector<int64_t> b_shape = {32};
    vector<int64_t> gates_shape = {1, 32};

    auto dhy = TensorDesc(dhy_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto dc = TensorDesc(dhy_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto cx = TensorDesc(dhy_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto cy = TensorDesc(dhy_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto storage = TensorDesc(gates_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto dc_prev_out = TensorDesc(dhy_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto db_out = TensorDesc(b_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto dgates_out = TensorDesc(gates_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnThnnFusedLstmCellBackward,
        INPUT(
            dhy, dc, cx, cy, storage, true),
        OUTPUT(dgates_out, dc_prev_out, db_out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}
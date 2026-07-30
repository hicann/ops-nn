/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_aclnn_gru_backward.cpp
 * \brief aclnnGRUBackward L2 接口单元测试
 *
 * aclnnGRUBackwardGetWorkspaceSize 形参顺序为 (inputs..., scalar attrs..., outputs..., ws, exec),
 * 标量属性 (hasBias/numLayers/bidirectional/batchFirst) 位于输出之前, 因此按 lstm_backward 约定
 * 将标量放入 INPUT(...), OUTPUT(...) 仅含 3 个输出张量/张量列表。
 */

#include <vector>
#include "gtest/gtest.h"

#include "opdev/op_log.h"
#include "../../../op_api/aclnn_gru_backward.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class GruBackwardApiTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "GruBackwardApiTest SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "GruBackwardApiTest TearDown" << std::endl; }
};

// 正例: 单层单向 FP32 无偏置, GetWorkspaceSize 返回 ACLNN_SUCCESS
TEST_F(GruBackwardApiTest, case_single_layer_unidirectional_fp32_no_bias)
{
    int64_t T = 2;
    int64_t B = 3;
    int64_t I = 4;
    int64_t H = 5;
    int64_t gateNum = 3;

    auto input = TensorDesc({T, B, I}, ACL_FLOAT, ACL_FORMAT_ND);
    auto hx0 = TensorDesc({B, H}, ACL_FLOAT, ACL_FORMAT_ND);
    auto hx = TensorListDesc({hx0});

    auto wi = TensorDesc({gateNum * H, I}, ACL_FLOAT, ACL_FORMAT_ND);
    auto wh = TensorDesc({gateNum * H, H}, ACL_FLOAT, ACL_FORMAT_ND);
    auto params = TensorListDesc({wi, wh});

    auto dy = TensorDesc({T, B, H}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dh = TensorDesc({1, B, H}, ACL_FLOAT, ACL_FORMAT_ND);

    auto gate = TensorDesc({T, B, H}, ACL_FLOAT, ACL_FORMAT_ND);
    auto rList = TensorListDesc({gate});
    auto zList = TensorListDesc({gate});
    auto nList = TensorListDesc({gate});
    auto hnList = TensorListDesc({gate});
    auto hList = TensorListDesc({gate});

    auto dxOut = TensorDesc({T, B, I}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dhPrevOut = TensorDesc({1, B, H}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dwiOut = TensorDesc({gateNum * H, I}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dwhOut = TensorDesc({gateNum * H, H}, ACL_FLOAT, ACL_FORMAT_ND);
    auto dparamsOut = TensorListDesc({dwiOut, dwhOut});

    // INPUT: input, hx, params, dy, dh, r, z, n, h_n, h, batchSizes(nullptr),
    //        hasBias(false), numLayers(1), bidirectional(false), batchFirst(false)
    auto ut = OP_API_UT(
        aclnnGRUBackward,
        INPUT(input, hx, params, dy, dh, rList, zList, nList, hnList, hList, nullptr, false, 1, false, false),
        OUTPUT(dxOut, dhPrevOut, dparamsOut));

    uint64_t workspaceSize = 0;
    aclnnStatus ret = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(ret, ACLNN_SUCCESS);
}

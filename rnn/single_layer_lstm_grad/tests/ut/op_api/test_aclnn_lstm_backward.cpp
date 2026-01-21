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
#include "../../../op_host/op_api/aclnn_lstm_backward.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace op;
using namespace std;

class l2_lstm_backward_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "l2_lstm_backward_test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "l2_lstm_backward_test TearDown" << std::endl;
    }
};

// 正常input场景
TEST_F(l2_lstm_backward_test, ascend910B2_normal_float)
{
    vector<int64_t> x_shape = {2, 1, 8};
    vector<int64_t> y_shape = {2, 1, 8};
    vector<int64_t> init_h_shape = {1, 1, 8};
    vector<int64_t> w_ih_shape = {32, 8};
    vector<int64_t> w_hh_shape = {32, 8};
    vector<int64_t> b_shape = {32};
    vector<int64_t> gates_shape = {2, 1, 8};
    vector<bool> output_mask = {false, false, false, false};
    auto output_mask_desc = BoolArrayDesc(output_mask);

    auto input = TensorDesc(x_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto init_h = TensorDesc(init_h_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto init_c = TensorDesc(init_h_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hx_list = TensorListDesc({init_h, init_c});

    auto w_ih = TensorDesc(w_ih_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto w_hh = TensorDesc(w_hh_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto b_ih = TensorDesc(b_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto b_hh = TensorDesc(b_shape, ACL_FLOAT, ACL_FORMAT_ND);

    auto params_list = TensorListDesc({w_ih, w_hh, b_ih, b_hh});
    auto dy = TensorDesc(y_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto dh = TensorDesc(init_h_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto dc = TensorDesc(init_h_shape, ACL_FLOAT, ACL_FORMAT_ND);

    auto i = TensorDesc(gates_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto i_list = TensorListDesc({i});

    auto j = TensorDesc(gates_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto j_list = TensorListDesc({j});

    auto f = TensorDesc(gates_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto f_list = TensorListDesc({f});

    auto o = TensorDesc(gates_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto o_list = TensorListDesc({o});

    auto h = TensorDesc(gates_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto h_list = TensorListDesc({h});

    auto c = TensorDesc(gates_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto c_list = TensorListDesc({c});

    auto tanhc = TensorDesc(gates_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto tanhc_list = TensorListDesc({tanhc});

    auto dx_out = TensorDesc(x_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto dh_prev_out = TensorDesc(init_h_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto dc_prev_out = TensorDesc(init_h_shape, ACL_FLOAT, ACL_FORMAT_ND);

    auto dw_ih = TensorDesc(w_ih_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto dw_hh = TensorDesc(w_hh_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto db_ih = TensorDesc(b_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto db_hh = TensorDesc(b_shape, ACL_FLOAT, ACL_FORMAT_ND);

    auto dparams_out = TensorListDesc({dw_ih, dw_hh, db_ih, db_hh});
    auto ut = OP_API_UT(
        aclnnLstmBackward,
        INPUT(
            input, hx_list, params_list, dy, dh, dc, i_list, j_list, f_list, o_list, h_list, c_list, tanhc_list, nullptr, true, 1, 0.0d, true, false, false, 
            output_mask_desc),
        OUTPUT(dx_out, dh_prev_out, dc_prev_out, dparams_out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 正常input场景，双层
TEST_F(l2_lstm_backward_test, ascend910B2_normal_float_laryer_2_bid)
{
    vector<int64_t> x_shape = {2, 1, 8};
    vector<int64_t> y_shape = {2, 1, 16};
    vector<int64_t> init_h_shape = {4, 1, 8};
    vector<int64_t> w_ih_shape_0 = {32, 8};
    vector<int64_t> w_hh_shape_0 = {32, 8};
    vector<int64_t> w_ih_shape_1= {32, 16};
    vector<int64_t> w_hh_shape_1 = {32, 8};
    vector<int64_t> b_shape = {32};
    vector<int64_t> gates_shape = {2, 1, 8};
    vector<bool> output_mask = {false, false, false, false};
    auto output_mask_desc = BoolArrayDesc(output_mask);

    auto input = TensorDesc(x_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto init_h = TensorDesc(init_h_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto init_c = TensorDesc(init_h_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto hx_list = TensorListDesc({init_h, init_c});

    auto w_ih_0 = TensorDesc(w_ih_shape_0, ACL_FLOAT, ACL_FORMAT_ND);
    auto w_hh_0 = TensorDesc(w_hh_shape_0, ACL_FLOAT, ACL_FORMAT_ND);
    auto w_ih_1 = TensorDesc(w_ih_shape_1, ACL_FLOAT, ACL_FORMAT_ND);
    auto w_hh_1 = TensorDesc(w_hh_shape_1, ACL_FLOAT, ACL_FORMAT_ND);
    auto b = TensorDesc(b_shape, ACL_FLOAT, ACL_FORMAT_ND);

    auto params_list = TensorListDesc({w_ih_0, w_hh_0, b, b, w_ih_0, w_hh_0, b, b, w_ih_1, w_hh_1, b, b, w_ih_1, w_hh_1, b, b});
    auto dy = TensorDesc(y_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto dh = TensorDesc(init_h_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto dc = TensorDesc(init_h_shape, ACL_FLOAT, ACL_FORMAT_ND);

    auto i = TensorDesc(gates_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto i_list = TensorListDesc({i, i, i, i});

    auto j = TensorDesc(gates_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto j_list = TensorListDesc({j, j, j, j});

    auto f = TensorDesc(gates_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto f_list = TensorListDesc({f, f, f, f});

    auto o = TensorDesc(gates_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto o_list = TensorListDesc({o, o, o, o});

    auto h = TensorDesc(gates_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto h_list = TensorListDesc({h, h, h, h});

    auto c = TensorDesc(gates_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto c_list = TensorListDesc({c, c, c, c});

    auto tanhc = TensorDesc(gates_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto tanhc_list = TensorListDesc({tanhc, tanhc, tanhc, tanhc});

    auto dx_out = TensorDesc(x_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto dh_prev_out = TensorDesc(init_h_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto dc_prev_out = TensorDesc(init_h_shape, ACL_FLOAT, ACL_FORMAT_ND);

    auto dw_ih_0 = TensorDesc(w_ih_shape_0, ACL_FLOAT, ACL_FORMAT_ND);
    auto dw_hh_0 = TensorDesc(w_hh_shape_0, ACL_FLOAT, ACL_FORMAT_ND);
    auto dw_ih_1 = TensorDesc(w_ih_shape_1, ACL_FLOAT, ACL_FORMAT_ND);
    auto dw_hh_1 = TensorDesc(w_hh_shape_1, ACL_FLOAT, ACL_FORMAT_ND);
    auto db = TensorDesc(b_shape, ACL_FLOAT, ACL_FORMAT_ND);

    auto dparams_out = TensorListDesc({dw_ih_0, dw_hh_0, db, db, dw_ih_0, dw_hh_0, db, db, dw_ih_1, dw_hh_1, db, db,
                                       dw_ih_1, dw_hh_1, db, db});
    auto ut = OP_API_UT(
        aclnnLstmBackward,
        INPUT(
            input, hx_list, params_list, dy, dh, dc, i_list, j_list, f_list, o_list, h_list, c_list, tanhc_list,
            nullptr, true, 2, 0.0, true, true, false, output_mask_desc),
        OUTPUT(dx_out, dh_prev_out, dc_prev_out, dparams_out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

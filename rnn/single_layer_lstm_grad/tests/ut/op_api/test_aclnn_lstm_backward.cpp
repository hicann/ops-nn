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
#include <tuple>
#include "gtest/gtest.h"

#include "opdev/op_log.h"
#include "../../../op_api/aclnn_lstm_backward.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"
#include "lstm_backward_plan_spy.h"

using namespace op;
using namespace std;

class l2_lstm_backward_test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "l2_lstm_backward_test SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "l2_lstm_backward_test TearDown" << std::endl; }
};

// Public API planning only: device arithmetic is covered by the real-device ST.
// I=0 must still call the real Grad l0 planner, not return an empty executor.
class LstmBackward950Plan : public testing::TestWithParam<std::tuple<aclDataType, bool, int64_t>> {};

TEST_P(LstmBackward950Plan, same_dtype_caches_and_nonempty_recurrence)
{
    SocVersionManager soc(SocVersion::ASCEND950);
    const auto [dtype, batchFirst, inputSize] = GetParam();
    const vector<int64_t> xShape = batchFirst ? vector<int64_t>{2, 3, inputSize} : vector<int64_t>{3, 2, inputSize};
    const vector<int64_t> yShape = batchFirst ? vector<int64_t>{2, 3, 8} : vector<int64_t>{3, 2, 8};
    auto x = TensorDesc(xShape, dtype, ACL_FORMAT_NCL);
    auto state = TensorDesc({1, 2, 8}, dtype, ACL_FORMAT_NCL);
    auto hx = TensorListDesc({state, state});
    auto wIh = TensorDesc({32, inputSize}, dtype, ACL_FORMAT_ND);
    auto wHh = TensorDesc({32, 8}, dtype, ACL_FORMAT_ND);
    auto bias = TensorDesc({32}, dtype, ACL_FORMAT_ND);
    auto params = TensorListDesc({wIh, wHh, bias, bias});
    auto dy = TensorDesc(yShape, dtype, ACL_FORMAT_NCL);
    auto gate = TensorDesc({3, 2, 8}, dtype, ACL_FORMAT_NCL);
    auto gates = TensorListDesc({gate});
    auto mask = BoolArrayDesc({true, true, true, true});
    auto dparams = TensorListDesc({wIh, wHh, bias, bias});
    auto ut = OP_API_UT(LstmBackwardPlan,
                        INPUT(x, hx, params, dy, state, state, gates, gates, gates, gates, gates, gates, gates, nullptr,
                              true, 1, 0.0, true, false, batchFirst, mask),
                        OUTPUT(x, state, state, dparams));
    uint64_t workspaceSize = 0;
    lstm_test::GradPlanSpy spy;
    ASSERT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACL_SUCCESS);
    EXPECT_EQ(spy.calls, 1U);
    EXPECT_TRUE(spy.homogeneousInputs);
    EXPECT_EQ(spy.biasElements, dtype == ACL_FLOAT ? 32 : 64);
    EXPECT_EQ(spy.biasGradientElements, 32);
    if (dtype != ACL_FLOAT) {
        auto wideGate = TensorDesc({3, 2, 8}, ACL_FLOAT, ACL_FORMAT_NCL);
        auto wideGates = TensorListDesc({wideGate});
        auto rejected = OP_API_UT(
            LstmBackwardPlan,
            INPUT(x, hx, params, dy, state, state, wideGates, wideGates, wideGates, wideGates, wideGates, wideGates,
                  wideGates, nullptr, true, 1, 0.0, true, false, batchFirst, mask),
            OUTPUT(x, state, state, dparams));
        EXPECT_NE(rejected.TestGetWorkspaceSize(&workspaceSize), ACL_SUCCESS);
        EXPECT_EQ(spy.calls, 1U); // Rejected before reaching the Grad planner.
    }
}

INSTANTIATE_TEST_SUITE_P(PublicContract, LstmBackward950Plan,
                         testing::Combine(testing::Values(ACL_FLOAT, ACL_FLOAT16, ACL_BF16), testing::Bool(),
                                          testing::Values(int64_t{0}, int64_t{33})));

class LstmBackward950ZeroHidden : public testing::TestWithParam<std::tuple<aclDataType, bool>> {};

TEST_P(LstmBackward950ZeroHidden, returns_zero_dx_without_grad_kernel)
{
    SocVersionManager soc(SocVersion::ASCEND950);
    const auto [dtype, batchFirst] = GetParam();
    constexpr int64_t batch = 5;
    constexpr int64_t time = 3;
    constexpr int64_t inputSize = 33;
    constexpr int64_t hiddenSize = 0;
    const vector<int64_t> xShape = batchFirst ? vector<int64_t>{batch, time, inputSize} :
                                                vector<int64_t>{time, batch, inputSize};
    const vector<int64_t> yShape = batchFirst ? vector<int64_t>{batch, time, hiddenSize} :
                                                vector<int64_t>{time, batch, hiddenSize};
    auto x = TensorDesc(xShape, dtype, ACL_FORMAT_NCL);
    auto state = TensorDesc({1, batch, hiddenSize}, dtype, ACL_FORMAT_NCL);
    auto hx = TensorListDesc({state, state});
    auto wIh = TensorDesc({4 * hiddenSize, inputSize}, dtype, ACL_FORMAT_ND);
    auto wHh = TensorDesc({4 * hiddenSize, hiddenSize}, dtype, ACL_FORMAT_ND);
    auto bias = TensorDesc({4 * hiddenSize}, dtype, ACL_FORMAT_ND);
    auto params = TensorListDesc({wIh, wHh, bias, bias});
    auto dy = TensorDesc(yShape, dtype, ACL_FORMAT_NCL);
    auto gate = TensorDesc({time, batch, hiddenSize}, dtype, ACL_FORMAT_NCL);
    auto gates = TensorListDesc({gate});
    auto mask = BoolArrayDesc({true, true, true, true});
    auto dparams = TensorListDesc({wIh, wHh, bias, bias});
    auto ut = OP_API_UT(LstmBackwardPlan,
                        INPUT(x, hx, params, dy, state, state, gates, gates, gates, gates, gates, gates, gates, nullptr,
                              true, 1, 0.0, true, false, batchFirst, mask),
                        OUTPUT(x, state, state, dparams));
    uint64_t workspaceSize = 0;
    lstm_test::GradPlanSpy spy;
    ASSERT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACL_SUCCESS);
    EXPECT_EQ(spy.calls, 0U);
}

INSTANTIATE_TEST_SUITE_P(PublicContract, LstmBackward950ZeroHidden,
                         testing::Combine(testing::Values(ACL_FLOAT, ACL_FLOAT16, ACL_BF16), testing::Bool()));

namespace {
constexpr int64_t GATE_COUNT_UT = 4;

// One valid non-packed planning call. Each case below perturbs a single field.
struct Plan950 {
    int64_t timeStep = 3;
    int64_t batch = 2;
    int64_t inputSize = 8;
    int64_t hiddenSize = 8;
    int64_t numLayers = 1;
    bool bidirectional = false;
    bool batchFirst = false;
    bool train = true;
    double dropout = 0.0;
};

// TensorListDesc asserts on an empty list, so a nonpositive numLayers still describes one
// layer: the attribute under test is numLayers, not the list it would imply.
int64_t DescribedLayers(const Plan950& p) { return p.numLayers > 0 ? p.numLayers : 1; }

vector<TensorDesc> ParamDescs(const Plan950& p, int64_t dir)
{
    const int64_t gates = GATE_COUNT_UT * p.hiddenSize;
    auto bias = TensorDesc({gates}, ACL_FLOAT, ACL_FORMAT_ND);
    vector<TensorDesc> descs;
    for (int64_t layer = 0; layer < DescribedLayers(p); ++layer) {
        const int64_t fanIn = layer == 0 ? p.inputSize : p.hiddenSize * dir;
        for (int64_t slot = 0; slot < dir; ++slot) {
            descs.emplace_back(TensorDesc({gates, fanIn}, ACL_FLOAT, ACL_FORMAT_ND));
            descs.emplace_back(TensorDesc({gates, p.hiddenSize}, ACL_FLOAT, ACL_FORMAT_ND));
            descs.emplace_back(bias);
            descs.emplace_back(bias);
        }
    }
    return descs;
}

aclnnStatus RunPlan950(const Plan950& p, size_t& gradCalls)
{
    const int64_t dir = p.bidirectional ? 2 : 1;
    const int64_t outSize = p.hiddenSize * dir;
    const vector<int64_t> xShape = p.batchFirst ? vector<int64_t>{p.batch, p.timeStep, p.inputSize} :
                                                  vector<int64_t>{p.timeStep, p.batch, p.inputSize};
    const vector<int64_t> yShape = p.batchFirst ? vector<int64_t>{p.batch, p.timeStep, outSize} :
                                                  vector<int64_t>{p.timeStep, p.batch, outSize};
    auto x = TensorDesc(xShape, ACL_FLOAT, ACL_FORMAT_NCL);
    auto state = TensorDesc({DescribedLayers(p) * dir, p.batch, p.hiddenSize}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto hx = TensorListDesc({state, state});
    auto params = TensorListDesc(ParamDescs(p, dir));
    auto dy = TensorDesc(yShape, ACL_FLOAT, ACL_FORMAT_NCL);
    auto gate = TensorDesc({p.timeStep, p.batch, p.hiddenSize}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto gates = TensorListDesc(static_cast<int>(DescribedLayers(p) * dir), gate);
    auto mask = BoolArrayDesc({true, true, true, true});
    auto ut = OP_API_UT(LstmBackwardPlan,
                        INPUT(x, hx, params, dy, state, state, gates, gates, gates, gates, gates, gates, gates, nullptr,
                              true, p.numLayers, p.dropout, p.train, p.bidirectional, p.batchFirst, mask),
                        OUTPUT(x, state, state, params));
    uint64_t workspaceSize = 0;
    lstm_test::GradPlanSpy spy;
    const aclnnStatus ret = ut.TestGetWorkspaceSize(&workspaceSize);
    gradCalls = spy.calls;
    return ret;
}
} // namespace

// numLayers 校验与 SoC 无关: 它必须在按 numLayers 推导列表长度之前拒绝。
TEST(LstmBackwardValidation, rejects_nonpositive_num_layers)
{
    Plan950 plan;
    plan.numLayers = 0;
    size_t gradCalls = 1;
    EXPECT_EQ(RunPlan950(plan, gradCalls), ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(gradCalls, 0U);
}

TEST(LstmBackward950Gates, rejects_nonpositive_num_layers)
{
    SocVersionManager soc(SocVersion::ASCEND950);
    Plan950 plan;
    plan.numLayers = 0;
    size_t gradCalls = 1;
    EXPECT_EQ(RunPlan950(plan, gradCalls), ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(gradCalls, 0U);
}

// dropout 只在它真正生效时被拒: train && numLayers > 1。
TEST(LstmBackward950Gates, rejects_effective_dropout)
{
    SocVersionManager soc(SocVersion::ASCEND950);
    Plan950 plan;
    plan.numLayers = 2;
    plan.dropout = 0.5;
    size_t gradCalls = 1;
    EXPECT_EQ(RunPlan950(plan, gradCalls), ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(gradCalls, 0U);
}

TEST(LstmBackward950Gates, accepts_dropout_when_not_training)
{
    SocVersionManager soc(SocVersion::ASCEND950);
    Plan950 plan;
    plan.numLayers = 2;
    plan.dropout = 0.5;
    plan.train = false;
    size_t gradCalls = 0;
    EXPECT_EQ(RunPlan950(plan, gradCalls), ACL_SUCCESS);
    EXPECT_GT(gradCalls, 0U);
}

TEST(LstmBackward950Gates, accepts_dropout_for_single_layer)
{
    SocVersionManager soc(SocVersion::ASCEND950);
    Plan950 plan;
    plan.dropout = 0.5;
    size_t gradCalls = 0;
    EXPECT_EQ(RunPlan950(plan, gradCalls), ACL_SUCCESS);
    EXPECT_EQ(gradCalls, 1U);
}

TEST(LstmBackward950Gates, accepts_bidirectional)
{
    SocVersionManager soc(SocVersion::ASCEND950);
    Plan950 plan;
    plan.bidirectional = true;
    size_t gradCalls = 0;
    EXPECT_EQ(RunPlan950(plan, gradCalls), ACL_SUCCESS);
    EXPECT_GT(gradCalls, 0U);
}

TEST(LstmBackward950Gates, rejects_nonpositive_time_step)
{
    SocVersionManager soc(SocVersion::ASCEND950);
    Plan950 plan;
    plan.timeStep = 0;
    size_t gradCalls = 1;
    EXPECT_EQ(RunPlan950(plan, gradCalls), ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(gradCalls, 0U);
}

// batch == 0: 状态与输入梯度为空, 但参数梯度不是, 必须写零而不是提前返回。
TEST(LstmBackward950Gates, fills_zero_parameter_grads_for_empty_batch)
{
    SocVersionManager soc(SocVersion::ASCEND950);
    Plan950 plan;
    plan.batch = 0;
    size_t gradCalls = 1;
    EXPECT_EQ(RunPlan950(plan, gradCalls), ACL_SUCCESS);
    EXPECT_EQ(gradCalls, 0U);
}

// dy == nullptr 时按 input 的秩取特征维, 不写死三维。
TEST(LstmBackward950Gates, builds_default_dy_from_input_rank)
{
    SocVersionManager soc(SocVersion::ASCEND950);
    constexpr int64_t timeStep = 3;
    constexpr int64_t batch = 2;
    constexpr int64_t size = 8;
    auto x = TensorDesc({timeStep, batch, size}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto state = TensorDesc({1, batch, size}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto hx = TensorListDesc({state, state});
    auto params = TensorListDesc(ParamDescs(Plan950{}, 1));
    auto gate = TensorDesc({timeStep, batch, size}, ACL_FLOAT, ACL_FORMAT_NCL);
    auto gates = TensorListDesc({gate});
    auto mask = BoolArrayDesc({true, true, true, true});
    auto ut = OP_API_UT(LstmBackwardPlan,
                        INPUT(x, hx, params, nullptr, state, state, gates, gates, gates, gates, gates, gates, gates,
                              nullptr, true, 1, 0.0, true, false, false, mask),
                        OUTPUT(x, state, state, params));
    uint64_t workspaceSize = 0;
    lstm_test::GradPlanSpy spy;
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACL_SUCCESS);
    EXPECT_EQ(spy.calls, 1U);
}

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

    auto input = TensorDesc(x_shape, ACL_FLOAT, ACL_FORMAT_NCL);
    auto init_h = TensorDesc(init_h_shape, ACL_FLOAT, ACL_FORMAT_NCL);
    auto init_c = TensorDesc(init_h_shape, ACL_FLOAT, ACL_FORMAT_NCL);
    auto hx_list = TensorListDesc({init_h, init_c});

    auto w_ih = TensorDesc(w_ih_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto w_hh = TensorDesc(w_hh_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto b_ih = TensorDesc(b_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto b_hh = TensorDesc(b_shape, ACL_FLOAT, ACL_FORMAT_ND);

    auto params_list = TensorListDesc({w_ih, w_hh, b_ih, b_hh});
    auto dy = TensorDesc(y_shape, ACL_FLOAT, ACL_FORMAT_NCL);
    auto dh = TensorDesc(init_h_shape, ACL_FLOAT, ACL_FORMAT_NCL);
    auto dc = TensorDesc(init_h_shape, ACL_FLOAT, ACL_FORMAT_NCL);

    auto i = TensorDesc(gates_shape, ACL_FLOAT, ACL_FORMAT_NCL);
    auto i_list = TensorListDesc({i});

    auto j = TensorDesc(gates_shape, ACL_FLOAT, ACL_FORMAT_NCL);
    auto j_list = TensorListDesc({j});

    auto f = TensorDesc(gates_shape, ACL_FLOAT, ACL_FORMAT_NCL);
    auto f_list = TensorListDesc({f});

    auto o = TensorDesc(gates_shape, ACL_FLOAT, ACL_FORMAT_NCL);
    auto o_list = TensorListDesc({o});

    auto h = TensorDesc(gates_shape, ACL_FLOAT, ACL_FORMAT_NCL);
    auto h_list = TensorListDesc({h});

    auto c = TensorDesc(gates_shape, ACL_FLOAT, ACL_FORMAT_NCL);
    auto c_list = TensorListDesc({c});

    auto tanhc = TensorDesc(gates_shape, ACL_FLOAT, ACL_FORMAT_NCL);
    auto tanhc_list = TensorListDesc({tanhc});

    auto dx_out = TensorDesc(x_shape, ACL_FLOAT, ACL_FORMAT_NCL);
    auto dh_prev_out = TensorDesc(init_h_shape, ACL_FLOAT, ACL_FORMAT_NCL);
    auto dc_prev_out = TensorDesc(init_h_shape, ACL_FLOAT, ACL_FORMAT_NCL);

    auto dw_ih = TensorDesc(w_ih_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto dw_hh = TensorDesc(w_hh_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto db_ih = TensorDesc(b_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto db_hh = TensorDesc(b_shape, ACL_FLOAT, ACL_FORMAT_ND);

    auto dparams_out = TensorListDesc({dw_ih, dw_hh, db_ih, db_hh});
    auto ut = OP_API_UT(aclnnLstmBackward,
                        INPUT(input, hx_list, params_list, dy, dh, dc, i_list, j_list, f_list, o_list, h_list, c_list,
                              tanhc_list, nullptr, true, 1, 0.0d, true, false, false, output_mask_desc),
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
    vector<int64_t> w_ih_shape_1 = {32, 16};
    vector<int64_t> w_hh_shape_1 = {32, 8};
    vector<int64_t> b_shape = {32};
    vector<int64_t> gates_shape = {2, 1, 8};
    vector<bool> output_mask = {false, false, false, false};
    auto output_mask_desc = BoolArrayDesc(output_mask);

    auto input = TensorDesc(x_shape, ACL_FLOAT, ACL_FORMAT_NCL);
    auto init_h = TensorDesc(init_h_shape, ACL_FLOAT, ACL_FORMAT_NCL);
    auto init_c = TensorDesc(init_h_shape, ACL_FLOAT, ACL_FORMAT_NCL);
    auto hx_list = TensorListDesc({init_h, init_c});

    auto w_ih_0 = TensorDesc(w_ih_shape_0, ACL_FLOAT, ACL_FORMAT_ND);
    auto w_hh_0 = TensorDesc(w_hh_shape_0, ACL_FLOAT, ACL_FORMAT_ND);
    auto w_ih_1 = TensorDesc(w_ih_shape_1, ACL_FLOAT, ACL_FORMAT_ND);
    auto w_hh_1 = TensorDesc(w_hh_shape_1, ACL_FLOAT, ACL_FORMAT_ND);
    auto b = TensorDesc(b_shape, ACL_FLOAT, ACL_FORMAT_ND);

    auto params_list = TensorListDesc(
        {w_ih_0, w_hh_0, b, b, w_ih_0, w_hh_0, b, b, w_ih_1, w_hh_1, b, b, w_ih_1, w_hh_1, b, b});
    auto dy = TensorDesc(y_shape, ACL_FLOAT, ACL_FORMAT_NCL);
    auto dh = TensorDesc(init_h_shape, ACL_FLOAT, ACL_FORMAT_NCL);
    auto dc = TensorDesc(init_h_shape, ACL_FLOAT, ACL_FORMAT_NCL);

    auto i = TensorDesc(gates_shape, ACL_FLOAT, ACL_FORMAT_NCL);
    auto i_list = TensorListDesc({i, i, i, i});

    auto j = TensorDesc(gates_shape, ACL_FLOAT, ACL_FORMAT_NCL);
    auto j_list = TensorListDesc({j, j, j, j});

    auto f = TensorDesc(gates_shape, ACL_FLOAT, ACL_FORMAT_NCL);
    auto f_list = TensorListDesc({f, f, f, f});

    auto o = TensorDesc(gates_shape, ACL_FLOAT, ACL_FORMAT_NCL);
    auto o_list = TensorListDesc({o, o, o, o});

    auto h = TensorDesc(gates_shape, ACL_FLOAT, ACL_FORMAT_NCL);
    auto h_list = TensorListDesc({h, h, h, h});

    auto c = TensorDesc(gates_shape, ACL_FLOAT, ACL_FORMAT_NCL);
    auto c_list = TensorListDesc({c, c, c, c});

    auto tanhc = TensorDesc(gates_shape, ACL_FLOAT, ACL_FORMAT_NCL);
    auto tanhc_list = TensorListDesc({tanhc, tanhc, tanhc, tanhc});

    auto dx_out = TensorDesc(x_shape, ACL_FLOAT, ACL_FORMAT_NCL);
    auto dh_prev_out = TensorDesc(init_h_shape, ACL_FLOAT, ACL_FORMAT_NCL);
    auto dc_prev_out = TensorDesc(init_h_shape, ACL_FLOAT, ACL_FORMAT_NCL);

    auto dw_ih_0 = TensorDesc(w_ih_shape_0, ACL_FLOAT, ACL_FORMAT_ND);
    auto dw_hh_0 = TensorDesc(w_hh_shape_0, ACL_FLOAT, ACL_FORMAT_ND);
    auto dw_ih_1 = TensorDesc(w_ih_shape_1, ACL_FLOAT, ACL_FORMAT_ND);
    auto dw_hh_1 = TensorDesc(w_hh_shape_1, ACL_FLOAT, ACL_FORMAT_ND);
    auto db = TensorDesc(b_shape, ACL_FLOAT, ACL_FORMAT_ND);

    auto dparams_out = TensorListDesc(
        {dw_ih_0, dw_hh_0, db, db, dw_ih_0, dw_hh_0, db, db, dw_ih_1, dw_hh_1, db, db, dw_ih_1, dw_hh_1, db, db});
    auto ut = OP_API_UT(aclnnLstmBackward,
                        INPUT(input, hx_list, params_list, dy, dh, dc, i_list, j_list, f_list, o_list, h_list, c_list,
                              tanhc_list, nullptr, true, 2, 0.0, true, true, false, output_mask_desc),
                        OUTPUT(dx_out, dh_prev_out, dc_prev_out, dparams_out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// 防御性负向用例:hx[0]维度不足(2维)应被参数校验拦截
TEST_F(l2_lstm_backward_test, ascend910B2_hx_dim_invalid)
{
    vector<int64_t> x_shape = {2, 1, 8};
    vector<int64_t> y_shape = {2, 1, 8};
    vector<int64_t> init_h_shape = {1, 8}; // 期望3维,构造2维触发维度校验
    vector<int64_t> w_ih_shape = {32, 8};
    vector<int64_t> w_hh_shape = {32, 8};
    vector<int64_t> b_shape = {32};
    vector<int64_t> gates_shape = {2, 1, 8};
    vector<bool> output_mask = {false, false, false, false};
    auto output_mask_desc = BoolArrayDesc(output_mask);

    auto input = TensorDesc(x_shape, ACL_FLOAT, ACL_FORMAT_NCL);
    auto init_h = TensorDesc(init_h_shape, ACL_FLOAT, ACL_FORMAT_NCL);
    auto init_c = TensorDesc(init_h_shape, ACL_FLOAT, ACL_FORMAT_NCL);
    auto hx_list = TensorListDesc({init_h, init_c});

    auto w_ih = TensorDesc(w_ih_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto w_hh = TensorDesc(w_hh_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto b_ih = TensorDesc(b_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto b_hh = TensorDesc(b_shape, ACL_FLOAT, ACL_FORMAT_ND);

    auto params_list = TensorListDesc({w_ih, w_hh, b_ih, b_hh});
    auto dy = TensorDesc(y_shape, ACL_FLOAT, ACL_FORMAT_NCL);
    auto dh = TensorDesc(init_h_shape, ACL_FLOAT, ACL_FORMAT_NCL);
    auto dc = TensorDesc(init_h_shape, ACL_FLOAT, ACL_FORMAT_NCL);

    auto i = TensorDesc(gates_shape, ACL_FLOAT, ACL_FORMAT_NCL);
    auto i_list = TensorListDesc({i});

    auto j = TensorDesc(gates_shape, ACL_FLOAT, ACL_FORMAT_NCL);
    auto j_list = TensorListDesc({j});

    auto f = TensorDesc(gates_shape, ACL_FLOAT, ACL_FORMAT_NCL);
    auto f_list = TensorListDesc({f});

    auto o = TensorDesc(gates_shape, ACL_FLOAT, ACL_FORMAT_NCL);
    auto o_list = TensorListDesc({o});

    auto h = TensorDesc(gates_shape, ACL_FLOAT, ACL_FORMAT_NCL);
    auto h_list = TensorListDesc({h});

    auto c = TensorDesc(gates_shape, ACL_FLOAT, ACL_FORMAT_NCL);
    auto c_list = TensorListDesc({c});

    auto tanhc = TensorDesc(gates_shape, ACL_FLOAT, ACL_FORMAT_NCL);
    auto tanhc_list = TensorListDesc({tanhc});

    auto dx_out = TensorDesc(x_shape, ACL_FLOAT, ACL_FORMAT_NCL);
    auto dh_prev_out = TensorDesc(init_h_shape, ACL_FLOAT, ACL_FORMAT_NCL);
    auto dc_prev_out = TensorDesc(init_h_shape, ACL_FLOAT, ACL_FORMAT_NCL);

    auto dw_ih = TensorDesc(w_ih_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto dw_hh = TensorDesc(w_hh_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto db_ih = TensorDesc(b_shape, ACL_FLOAT, ACL_FORMAT_ND);
    auto db_hh = TensorDesc(b_shape, ACL_FLOAT, ACL_FORMAT_ND);

    auto dparams_out = TensorListDesc({dw_ih, dw_hh, db_ih, db_hh});
    auto ut = OP_API_UT(aclnnLstmBackward,
                        INPUT(input, hx_list, params_list, dy, dh, dc, i_list, j_list, f_list, o_list, h_list, c_list,
                              tanhc_list, nullptr, true, 1, 0.0d, true, false, false, output_mask_desc),
                        OUTPUT(dx_out, dh_prev_out, dc_prev_out, dparams_out));

    uint64_t workspace_size = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspace_size);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

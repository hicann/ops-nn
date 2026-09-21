/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "ut_op_common.h"
#include "ut_op_util.h"
#include "infershape_test_util.h"
#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"
#include "../../../op_graph/lamb_apply_optimizer_assign_proto.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "log/log.h"
#include "platform/platform_info.h"

using namespace ge;

class LambApplyOptimizerAssignProtoTest : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

TEST_F(LambApplyOptimizerAssignProtoTest, lamb_apply_optimizer_assign_case_2d)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("LambApplyOptimizerAssign")->infer_shape;
    gert::Shape inShape = {-1, -1};
    gert::Shape outShape = {};
    gert::Shape expShape = {-1, -1};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(12, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&inShape, &inShape, &inShape, &inShape, &inShape, &inShape, &inShape, &inShape,
                                    &inShape, &inShape, &inShape, &inShape})
                      .OutputShapes({&outShape, &outShape, &outShape})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();
    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto od0 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*od0), Ops::Base::ToString(expShape));
    auto od1 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(1);
    ASSERT_EQ(Ops::Base::ToString(*od1), Ops::Base::ToString(expShape));
    auto od2 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(2);
    ASSERT_EQ(Ops::Base::ToString(*od2), Ops::Base::ToString(expShape));
}

TEST_F(LambApplyOptimizerAssignProtoTest, lamb_apply_optimizer_assign_case_fp16_4d)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("LambApplyOptimizerAssign")->infer_shape;
    gert::Shape inShape = {-1, -1, -1, -1};
    gert::Shape outShape = {};
    gert::Shape expShape = {-1, -1, -1, -1};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(12, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&inShape, &inShape, &inShape, &inShape, &inShape, &inShape, &inShape, &inShape,
                                    &inShape, &inShape, &inShape, &inShape})
                      .OutputShapes({&outShape, &outShape, &outShape})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();
    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto od0 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*od0), Ops::Base::ToString(expShape));
    auto od1 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(1);
    ASSERT_EQ(Ops::Base::ToString(*od1), Ops::Base::ToString(expShape));
    auto od2 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(2);
    ASSERT_EQ(Ops::Base::ToString(*od2), Ops::Base::ToString(expShape));
}

namespace {
// 按 grad / inputv / inputm / input3 四个张量的形状跑一次 infershape，
// 标量输入统一用 {1}。用于核对 infershape 与 tiling 的支持范围是否一致。
ge::graphStatus RunInferShape(const gert::Shape& grad, const gert::Shape& inputv, const gert::Shape& inputm,
                              const gert::Shape& input3, gert::Shape* out0, gert::Shape* out1, gert::Shape* out2,
                              const gert::Shape& scalar = gert::Shape({1}))
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("LambApplyOptimizerAssign")->infer_shape;
    gert::Shape g = grad;
    gert::Shape v = inputv;
    gert::Shape m = inputm;
    gert::Shape p = input3;
    gert::Shape s = scalar;
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(12, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&g, &v, &m, &p, &s, &s, &s, &s, &s, &s, &s, &s})
                      .OutputShapes({out0, out1, out2})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();
    auto ctx = holder.GetContext<gert::InferShapeContext>();
    auto ret = inferShapeFunc(ctx);
    if (ret == ge::GRAPH_SUCCESS) {
        // faker 内部持有输出 shape 的副本，需从 context 取回
        *out0 = *ctx->GetOutputShape(0);
        *out1 = *ctx->GetOutputShape(1);
        *out2 = *ctx->GetOutputShape(2);
    }
    return ret;
}
} // namespace

// grad/inputv/inputm 三者同形，输出形状由它们决定。
TEST_F(LambApplyOptimizerAssignProtoTest, moment_shape_decides_output_shape)
{
    gert::Shape o0 = {}, o1 = {}, o2 = {};
    gert::Shape expect = {512, 1024};
    ASSERT_EQ(RunInferShape({512, 1024}, {512, 1024}, {512, 1024}, {512, 1024}, &o0, &o1, &o2), ge::GRAPH_SUCCESS);
    ASSERT_EQ(Ops::Base::ToString(o0), Ops::Base::ToString(expect));
    ASSERT_EQ(Ops::Base::ToString(o1), Ops::Base::ToString(expect));
    ASSERT_EQ(Ops::Base::ToString(o2), Ops::Base::ToString(expect));
}

// grad 与其余输入一样参与广播：小于动量形状时按右对齐广播进动量形状，须放行。
TEST_F(LambApplyOptimizerAssignProtoTest, grad_broadcast_into_moment_is_accepted)
{
    gert::Shape o0 = {}, o1 = {}, o2 = {};
    gert::Shape expect = {512, 1024};
    ASSERT_EQ(RunInferShape({1, 1024}, {512, 1024}, {512, 1024}, {512, 1024}, &o0, &o1, &o2), ge::GRAPH_SUCCESS);
    ASSERT_EQ(Ops::Base::ToString(o0), Ops::Base::ToString(expect));
    ASSERT_EQ(Ops::Base::ToString(o1), Ops::Base::ToString(expect));
    ASSERT_EQ(Ops::Base::ToString(o2), Ops::Base::ToString(expect));
}

// input3 是唯一参与广播的输入：小于动量形状时按右对齐广播，须放行。
TEST_F(LambApplyOptimizerAssignProtoTest, input3_broadcast_into_moment_is_accepted)
{
    gert::Shape o0 = {}, o1 = {}, o2 = {};
    gert::Shape expect = {512, 1024};
    ASSERT_EQ(RunInferShape({512, 1024}, {512, 1024}, {512, 1024}, {1, 1024}, &o0, &o1, &o2), ge::GRAPH_SUCCESS);
    ASSERT_EQ(Ops::Base::ToString(o0), Ops::Base::ToString(expect));
}

// grad 大于动量形状：结果无处容纳，infershape 须与 tiling 一样拒收。
TEST_F(LambApplyOptimizerAssignProtoTest, grad_larger_than_moment_is_rejected)
{
    gert::Shape o0 = {}, o1 = {}, o2 = {};
    ASSERT_EQ(RunInferShape({512, 1024}, {1, 1024}, {1, 1024}, {1, 1024}, &o0, &o1, &o2), ge::GRAPH_FAILED);
}

// input3 大于动量形状：同上。
TEST_F(LambApplyOptimizerAssignProtoTest, input3_larger_than_moment_is_rejected)
{
    gert::Shape o0 = {}, o1 = {}, o2 = {};
    ASSERT_EQ(RunInferShape({1, 1024}, {1, 1024}, {1, 1024}, {512, 1024}, &o0, &o1, &o2), ge::GRAPH_FAILED);
}

// 两个动量形状不一致：无法确定唯一的输出形状，须拒收。
TEST_F(LambApplyOptimizerAssignProtoTest, inputv_inputm_mismatch_is_rejected)
{
    gert::Shape o0 = {}, o1 = {}, o2 = {};
    ASSERT_EQ(RunInferShape({512, 1024}, {512, 1024}, {1, 1024}, {512, 1024}, &o0, &o1, &o2), ge::GRAPH_FAILED);
}

// 系数输入不再限定为标量: 与动量同形的系数张量须放行(对齐 A2 的可广播 ND Tensor 声明)。
TEST_F(LambApplyOptimizerAssignProtoTest, non_scalar_coefficient_is_accepted)
{
    gert::Shape o0 = {}, o1 = {}, o2 = {};
    gert::Shape expect = {512, 1024};
    ASSERT_EQ(RunInferShape({512, 1024}, {512, 1024}, {512, 1024}, {512, 1024}, &o0, &o1, &o2, {512, 1024}),
              ge::GRAPH_SUCCESS);
    ASSERT_EQ(Ops::Base::ToString(o0), Ops::Base::ToString(expect));
}

// 系数输入大于动量形状: 广播结果无处容纳(动量是原地输出), 须拒收。
TEST_F(LambApplyOptimizerAssignProtoTest, coefficient_larger_than_moment_is_rejected)
{
    gert::Shape o0 = {}, o1 = {}, o2 = {};
    ASSERT_EQ(RunInferShape({1, 1024}, {1, 1024}, {1, 1024}, {1, 1024}, &o0, &o1, &o2, {512, 1024}), ge::GRAPH_FAILED);
}

// 全部输入均为 0 维标量: 广播结果仍是 0 维, infershape 须归一为 (1,)。
// 跑批只能证明"声明 0 维输入能跑通、输出是 (1,)": TTK 在 kernel 通路会把 0 维输入
// 规整成 (1,), 区分不出这层归一是本算子做的还是上游先做掉的 —— 直接调 infershape
// 才能把这条分支钉死。
TEST_F(LambApplyOptimizerAssignProtoTest, all_rank0_inputs_normalize_to_one_dim)
{
    gert::Shape o0 = {}, o1 = {}, o2 = {};
    gert::Shape rank0 = {};
    gert::Shape expect = {1};
    ASSERT_EQ(RunInferShape(rank0, rank0, rank0, rank0, &o0, &o1, &o2, rank0), ge::GRAPH_SUCCESS);
    ASSERT_EQ(Ops::Base::ToString(o0), Ops::Base::ToString(expect));
    ASSERT_EQ(Ops::Base::ToString(o1), Ops::Base::ToString(expect));
    ASSERT_EQ(Ops::Base::ToString(o2), Ops::Base::ToString(expect));
}

// 0 维标量与张量混用: 0 维按长度 1 参与广播, 结果取张量形状。
TEST_F(LambApplyOptimizerAssignProtoTest, rank0_coefficient_broadcasts_into_tensor)
{
    gert::Shape o0 = {}, o1 = {}, o2 = {};
    gert::Shape rank0 = {};
    gert::Shape expect = {512, 1024};
    ASSERT_EQ(RunInferShape({512, 1024}, {512, 1024}, {512, 1024}, {512, 1024}, &o0, &o1, &o2, rank0),
              ge::GRAPH_SUCCESS);
    ASSERT_EQ(Ops::Base::ToString(o0), Ops::Base::ToString(expect));
}

// grad 本身为 0 维标量: 广播进动量形状, 须放行。
TEST_F(LambApplyOptimizerAssignProtoTest, rank0_grad_broadcasts_into_moment)
{
    gert::Shape o0 = {}, o1 = {}, o2 = {};
    gert::Shape expect = {512, 1024};
    ASSERT_EQ(RunInferShape({}, {512, 1024}, {512, 1024}, {512, 1024}, &o0, &o1, &o2), ge::GRAPH_SUCCESS);
    ASSERT_EQ(Ops::Base::ToString(o0), Ops::Base::ToString(expect));
}

TEST_F(LambApplyOptimizerAssignProtoTest, lambapplyoptimizerassign_infer_datatype)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("LambApplyOptimizerAssign"), nullptr);
    auto dataTypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("LambApplyOptimizerAssign")->infer_datatype;
    ASSERT_NE(dataTypeFunc, nullptr);
    ge::DataType in0 = ge::DT_FLOAT;
    ge::DataType in1 = ge::DT_FLOAT;
    ge::DataType in2 = ge::DT_FLOAT;
    ge::DataType in3 = ge::DT_FLOAT;
    ge::DataType in4 = ge::DT_FLOAT;
    ge::DataType in5 = ge::DT_FLOAT;
    ge::DataType in6 = ge::DT_FLOAT;
    ge::DataType in7 = ge::DT_FLOAT;
    ge::DataType in8 = ge::DT_FLOAT;
    ge::DataType in9 = ge::DT_FLOAT;
    ge::DataType in10 = ge::DT_FLOAT;
    ge::DataType in11 = ge::DT_FLOAT;
    ge::DataType expOut0 = ge::DT_FLOAT;
    ge::DataType expOut1 = ge::DT_FLOAT;
    ge::DataType expOut2 = ge::DT_FLOAT;
    auto contextHolder = gert::InferDataTypeContextFaker()
                             .NodeIoNum(12, 3)
                             .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(6, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(7, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(8, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(11, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                             .InputDataTypes({&in0, &in1, &in2, &in3, &in4, &in5, &in6, &in7, &in8, &in9, &in10, &in11})
                             .OutputDataTypes({&expOut0, &expOut1, &expOut2})
                             .Build();
    auto context = contextHolder.GetContext<gert::InferDataTypeContext>();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(dataTypeFunc(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT);
    EXPECT_EQ(context->GetOutputDataType(1), ge::DT_FLOAT);
    EXPECT_EQ(context->GetOutputDataType(2), ge::DT_FLOAT);
}

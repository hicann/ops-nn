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
#include "../../../op_graph/lamb_apply_weight_assign_proto.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "log/log.h"
#include "platform/platform_info.h"

using namespace ge;

class LambApplyWeightAssignProtoTest : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

TEST_F(LambApplyWeightAssignProtoTest, lamb_apply_weight_assign_case_2d)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("LambApplyWeightAssign")->infer_shape;
    gert::Shape inShape = {-1, -1};
    gert::Shape outShape = {};
    gert::Shape expShape = {-1, -1};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(5, 1)
                      .IrInstanceNum({1, 1, 1, 1, 1})
                      .InputShapes({&inShape, &inShape, &inShape, &inShape, &inShape})
                      .OutputShapes({&outShape})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();
    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto od0 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*od0), Ops::Base::ToString(expShape));
}

TEST_F(LambApplyWeightAssignProtoTest, lamb_apply_weight_assign_case_fp16_4d)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("LambApplyWeightAssign")->infer_shape;
    gert::Shape inShape = {-1, -1, -1, -1};
    gert::Shape outShape = {};
    gert::Shape expShape = {-1, -1, -1, -1};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(5, 1)
                      .IrInstanceNum({1, 1, 1, 1, 1})
                      .InputShapes({&inShape, &inShape, &inShape, &inShape, &inShape})
                      .OutputShapes({&outShape})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();
    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto od0 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*od0), Ops::Base::ToString(expShape));
}

namespace {
// input0~input3 与 input_param 一起参与广播; input_param 是原地(ref)输出, 广播结果须恰好等于它。
ge::graphStatus RunInferShape(const gert::Shape& i0, const gert::Shape& i1, const gert::Shape& i2,
                              const gert::Shape& i3, const gert::Shape& param, gert::Shape* out0)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("LambApplyWeightAssign")->infer_shape;
    gert::Shape s0 = i0;
    gert::Shape s1 = i1;
    gert::Shape s2 = i2;
    gert::Shape s3 = i3;
    gert::Shape sp = param;
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(5, 1)
                      .IrInstanceNum({1, 1, 1, 1, 1})
                      .InputShapes({&s0, &s1, &s2, &s3, &sp})
                      .OutputShapes({out0})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();
    auto ctx = holder.GetContext<gert::InferShapeContext>();
    auto ret = inferShapeFunc(ctx);
    if (ret == ge::GRAPH_SUCCESS) {
        // faker 内部持有输出 shape 的副本, 需从 context 取回
        *out0 = *ctx->GetOutputShape(0);
    }
    return ret;
}
} // namespace

// input_param 是原地输出, 输出形状由它决定。
TEST_F(LambApplyWeightAssignProtoTest, param_shape_decides_output_shape)
{
    gert::Shape o0 = {};
    gert::Shape expect = {512, 1024};
    ASSERT_EQ(RunInferShape({512, 1024}, {512, 1024}, {512, 1024}, {512, 1024}, {512, 1024}, &o0), ge::GRAPH_SUCCESS);
    ASSERT_EQ(Ops::Base::ToString(o0), Ops::Base::ToString(expect));
}

// update(input3) 小于 input_param: 按右对齐广播进 input_param 形状, 须放行。
TEST_F(LambApplyWeightAssignProtoTest, update_broadcast_into_param_is_accepted)
{
    gert::Shape o0 = {};
    gert::Shape expect = {512, 1024};
    ASSERT_EQ(RunInferShape({1}, {1}, {1}, {1, 1024}, {512, 1024}, &o0), ge::GRAPH_SUCCESS);
    ASSERT_EQ(Ops::Base::ToString(o0), Ops::Base::ToString(expect));
}

// update(input3) 大于 input_param: 可以广播, 但广播结果无处容纳(input_param 是原地输出), 须拒收。
TEST_F(LambApplyWeightAssignProtoTest, update_larger_than_param_is_rejected)
{
    gert::Shape o0 = {};
    ASSERT_EQ(RunInferShape({1}, {1}, {1}, {512, 1024}, {1, 1024}, &o0), ge::GRAPH_FAILED);
}

// 系数输入(input0)大于 input_param: 同上, 须拒收。
TEST_F(LambApplyWeightAssignProtoTest, coefficient_larger_than_param_is_rejected)
{
    gert::Shape o0 = {};
    ASSERT_EQ(RunInferShape({512, 1024}, {1}, {1}, {1, 1024}, {1, 1024}, &o0), ge::GRAPH_FAILED);
}

// 与上面两条区分开: 对应维既不相等也都不为 1, 压根无法广播, 在广播这一步就须拒收。
TEST_F(LambApplyWeightAssignProtoTest, non_broadcastable_input_is_rejected)
{
    gert::Shape o0 = {};
    ASSERT_EQ(RunInferShape({1}, {1}, {1}, {64}, {64, 128}, &o0), ge::GRAPH_FAILED);
}

// 全部输入均为 0 维标量: 广播结果仍是 0 维, infershape 须归一为 (1,)。
TEST_F(LambApplyWeightAssignProtoTest, all_rank0_inputs_normalize_to_one_dim)
{
    gert::Shape o0 = {};
    gert::Shape rank0 = {};
    gert::Shape expect = {1};
    ASSERT_EQ(RunInferShape(rank0, rank0, rank0, rank0, rank0, &o0), ge::GRAPH_SUCCESS);
    ASSERT_EQ(Ops::Base::ToString(o0), Ops::Base::ToString(expect));
}

// 0 维标量与张量混用: 0 维按长度 1 参与广播, 结果取 input_param 形状。
TEST_F(LambApplyWeightAssignProtoTest, rank0_input_broadcasts_into_param)
{
    gert::Shape o0 = {};
    gert::Shape rank0 = {};
    gert::Shape expect = {512, 1024};
    ASSERT_EQ(RunInferShape(rank0, rank0, rank0, rank0, {512, 1024}, &o0), ge::GRAPH_SUCCESS);
    ASSERT_EQ(Ops::Base::ToString(o0), Ops::Base::ToString(expect));
}

TEST_F(LambApplyWeightAssignProtoTest, lambapplyweightassign_infer_datatype)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("LambApplyWeightAssign"), nullptr);
    auto dataTypeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("LambApplyWeightAssign")->infer_datatype;
    ASSERT_NE(dataTypeFunc, nullptr);
    ge::DataType in0 = ge::DT_FLOAT;
    ge::DataType in1 = ge::DT_FLOAT;
    ge::DataType in2 = ge::DT_FLOAT;
    ge::DataType in3 = ge::DT_FLOAT;
    ge::DataType in4 = ge::DT_FLOAT;
    ge::DataType expOut0 = ge::DT_FLOAT;
    auto contextHolder = gert::InferDataTypeContextFaker()
                             .NodeIoNum(5, 1)
                             .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                             .InputDataTypes({&in0, &in1, &in2, &in3, &in4})
                             .OutputDataTypes({&expOut0})
                             .Build();
    auto context = contextHolder.GetContext<gert::InferDataTypeContext>();
    ASSERT_NE(context, nullptr);
    EXPECT_EQ(dataTypeFunc(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT);
}

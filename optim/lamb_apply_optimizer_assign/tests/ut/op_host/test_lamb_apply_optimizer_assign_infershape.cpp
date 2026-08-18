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
                              const gert::Shape& input3, gert::Shape* out0, gert::Shape* out1, gert::Shape* out2)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("LambApplyOptimizerAssign")->infer_shape;
    gert::Shape g = grad;
    gert::Shape v = inputv;
    gert::Shape m = inputm;
    gert::Shape p = input3;
    gert::Shape s = {1};
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

// inputv/inputm 承载动量更新结果，输出形状由它们决定；grad 更小时向上广播进来。
TEST_F(LambApplyOptimizerAssignProtoTest, moment_shape_decides_output_shape)
{
    gert::Shape o0 = {}, o1 = {}, o2 = {};
    gert::Shape expect = {512, 1024};
    ASSERT_EQ(RunInferShape({1, 1024}, {512, 1024}, {512, 1024}, {512, 1024}, &o0, &o1, &o2), ge::GRAPH_SUCCESS);
    ASSERT_EQ(Ops::Base::ToString(o0), Ops::Base::ToString(expect));
    ASSERT_EQ(Ops::Base::ToString(o1), Ops::Base::ToString(expect));
    ASSERT_EQ(Ops::Base::ToString(o2), Ops::Base::ToString(expect));
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

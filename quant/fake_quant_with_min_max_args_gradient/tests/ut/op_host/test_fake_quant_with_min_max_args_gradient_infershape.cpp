/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "gtest/gtest.h"
#include "ut_op_common.h"
#include "infershape_test_util.h"
#include "log/log.h"
#include "../../../op_graph/fake_quant_with_min_max_args_gradient_proto.h"

using namespace ge;
using namespace op;

class FakeQuantWithMinMaxArgsGradientProto : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "FakeQuantWithMinMaxArgsGradientProto SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "FakeQuantWithMinMaxArgsGradientProto TearDown" << std::endl; }
};

TEST_F(FakeQuantWithMinMaxArgsGradientProto, Gradient_proto_fp32_1d)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("FakeQuantWithMinMaxArgsGradient")->infer_shape;

    gert::StorageShape gradShape = {{1024}, {}};
    gert::StorageShape xShape = {{1024}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeAttrs({{"min", Ops::NN::AnyValue::CreateFrom<float>(-6.0f)},
                                  {"max", Ops::NN::AnyValue::CreateFrom<float>(6.0f)},
                                  {"num_bits", Ops::NN::AnyValue::CreateFrom<int64_t>(8)},
                                  {"narrow_range", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .InputShapes({&gradShape, &xShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), "[1024]");
}

TEST_F(FakeQuantWithMinMaxArgsGradientProto, Gradient_proto_fp32_2d)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("FakeQuantWithMinMaxArgsGradient")->infer_shape;

    gert::StorageShape gradShape = {{4, 1024}, {}};
    gert::StorageShape xShape = {{4, 1024}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeAttrs({{"min", Ops::NN::AnyValue::CreateFrom<float>(-1.0f)},
                                  {"max", Ops::NN::AnyValue::CreateFrom<float>(1.0f)},
                                  {"num_bits", Ops::NN::AnyValue::CreateFrom<int64_t>(4)},
                                  {"narrow_range", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
                      .InputShapes({&gradShape, &xShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), "[4, 1024]");
}

TEST_F(FakeQuantWithMinMaxArgsGradientProto, Gradient_proto_fp32_4d)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("FakeQuantWithMinMaxArgsGradient")->infer_shape;

    gert::StorageShape gradShape = {{2, 3, 4, 8}, {}};
    gert::StorageShape xShape = {{2, 3, 4, 8}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeAttrs({{"min", Ops::NN::AnyValue::CreateFrom<float>(0.0f)},
                                  {"max", Ops::NN::AnyValue::CreateFrom<float>(6.0f)},
                                  {"num_bits", Ops::NN::AnyValue::CreateFrom<int64_t>(8)},
                                  {"narrow_range", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .InputShapes({&gradShape, &xShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), "[2, 3, 4, 8]");
}

TEST_F(FakeQuantWithMinMaxArgsGradientProto, Gradient_proto_fp32_3d)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("FakeQuantWithMinMaxArgsGradient")->infer_shape;

    gert::StorageShape gradShape = {{2, 3, 1024}, {}};
    gert::StorageShape xShape = {{2, 3, 1024}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeAttrs({{"min", Ops::NN::AnyValue::CreateFrom<float>(-6.0f)},
                                  {"max", Ops::NN::AnyValue::CreateFrom<float>(6.0f)},
                                  {"num_bits", Ops::NN::AnyValue::CreateFrom<int64_t>(8)},
                                  {"narrow_range", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .InputShapes({&gradShape, &xShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), "[2, 3, 1024]");
}

TEST_F(FakeQuantWithMinMaxArgsGradientProto, Gradient_proto_fp32_5d)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("FakeQuantWithMinMaxArgsGradient")->infer_shape;

    gert::StorageShape gradShape = {{1, 2, 3, 4, 8}, {}};
    gert::StorageShape xShape = {{1, 2, 3, 4, 8}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeAttrs({{"min", Ops::NN::AnyValue::CreateFrom<float>(-6.0f)},
                                  {"max", Ops::NN::AnyValue::CreateFrom<float>(6.0f)},
                                  {"num_bits", Ops::NN::AnyValue::CreateFrom<int64_t>(8)},
                                  {"narrow_range", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .InputShapes({&gradShape, &xShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), "[1, 2, 3, 4, 8]");
}
// rank 9 超出声明的 [0,8] 支持范围，Host 门禁独立拒绝
TEST_F(FakeQuantWithMinMaxArgsGradientProto, Gradient_proto_rank9_rejected)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("FakeQuantWithMinMaxArgsGradient")->infer_shape;

    gert::StorageShape gradShape = {{1, 1, 1, 1, 1, 1, 1, 1, 1}, {}};
    gert::StorageShape xShape = {{1, 1, 1, 1, 1, 1, 1, 1, 1}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeAttrs({{"min", Ops::NN::AnyValue::CreateFrom<float>(-6.0f)},
                                  {"max", Ops::NN::AnyValue::CreateFrom<float>(6.0f)},
                                  {"num_bits", Ops::NN::AnyValue::CreateFrom<int64_t>(8)},
                                  {"narrow_range", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .InputShapes({&gradShape, &xShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

// min >= max 违反属性语义，Host 门禁独立拒绝
TEST_F(FakeQuantWithMinMaxArgsGradientProto, Gradient_proto_invalid_min_max_rejected)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("FakeQuantWithMinMaxArgsGradient")->infer_shape;

    gert::StorageShape gradShape = {{4, 8}, {}};
    gert::StorageShape xShape = {{4, 8}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeAttrs({{"min", Ops::NN::AnyValue::CreateFrom<float>(6.0f)},
                                  {"max", Ops::NN::AnyValue::CreateFrom<float>(-6.0f)},
                                  {"num_bits", Ops::NN::AnyValue::CreateFrom<int64_t>(8)},
                                  {"narrow_range", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .InputShapes({&gradShape, &xShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

// num_bits 超出 [2,16] 拒绝
TEST_F(FakeQuantWithMinMaxArgsGradientProto, Gradient_proto_invalid_num_bits_rejected)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("FakeQuantWithMinMaxArgsGradient")->infer_shape;

    gert::StorageShape gradShape = {{4, 8}, {}};
    gert::StorageShape xShape = {{4, 8}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeAttrs({{"min", Ops::NN::AnyValue::CreateFrom<float>(-6.0f)},
                                  {"max", Ops::NN::AnyValue::CreateFrom<float>(6.0f)},
                                  {"num_bits", Ops::NN::AnyValue::CreateFrom<int64_t>(17)},
                                  {"narrow_range", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .InputShapes({&gradShape, &xShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

// 部分未知shape（-1）场景：输出透传x
TEST_F(FakeQuantWithMinMaxArgsGradientProto, Gradient_proto_partial_unknown_dim)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("FakeQuantWithMinMaxArgsGradient")->infer_shape;

    gert::StorageShape gradShape = {{4, -1}, {4, -1}};
    gert::StorageShape xShape = {{4, -1}, {4, -1}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeAttrs({{"min", Ops::NN::AnyValue::CreateFrom<float>(-6.0f)},
                                  {"max", Ops::NN::AnyValue::CreateFrom<float>(6.0f)},
                                  {"num_bits", Ops::NN::AnyValue::CreateFrom<int64_t>(8)},
                                  {"narrow_range", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .InputShapes({&gradShape, &xShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), "[4, -1]");
}

// 未知rank（-2）场景：两侧对称-2通过，输出-2
TEST_F(FakeQuantWithMinMaxArgsGradientProto, Gradient_proto_unknown_rank)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("FakeQuantWithMinMaxArgsGradient")->infer_shape;

    gert::StorageShape gradShape = {{-2}, {}};
    gert::StorageShape xShape = {{-2}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeAttrs({{"min", Ops::NN::AnyValue::CreateFrom<float>(-6.0f)},
                                  {"max", Ops::NN::AnyValue::CreateFrom<float>(6.0f)},
                                  {"num_bits", Ops::NN::AnyValue::CreateFrom<int64_t>(8)},
                                  {"narrow_range", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .InputShapes({&gradShape, &xShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), "[-2]");
}

// 单侧未知rank场景：gradients为-2而x已知时，跳过校验，输出透传x
TEST_F(FakeQuantWithMinMaxArgsGradientProto, Gradient_proto_grad_unknown_rank_x_known)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("FakeQuantWithMinMaxArgsGradient")->infer_shape;

    gert::StorageShape gradShape = {{-2}, {}};
    gert::StorageShape xShape = {{4, 16, 16}, {4, 16, 16}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeAttrs({{"min", Ops::NN::AnyValue::CreateFrom<float>(-6.0f)},
                                  {"max", Ops::NN::AnyValue::CreateFrom<float>(6.0f)},
                                  {"num_bits", Ops::NN::AnyValue::CreateFrom<int64_t>(8)},
                                  {"narrow_range", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .InputShapes({&gradShape, &xShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), "[4, 16, 16]");
}

// 混合未知维场景：x首维为-1而gradients已知时，-1视为通配，输出透传x
TEST_F(FakeQuantWithMinMaxArgsGradientProto, Gradient_proto_x_partial_unknown_grad_known)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("FakeQuantWithMinMaxArgsGradient")->infer_shape;

    gert::StorageShape gradShape = {{4, 16, 16}, {4, 16, 16}};
    gert::StorageShape xShape = {{-1, 16, 16}, {-1, 16, 16}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeAttrs({{"min", Ops::NN::AnyValue::CreateFrom<float>(-6.0f)},
                                  {"max", Ops::NN::AnyValue::CreateFrom<float>(6.0f)},
                                  {"num_bits", Ops::NN::AnyValue::CreateFrom<int64_t>(8)},
                                  {"narrow_range", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .InputShapes({&gradShape, &xShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), "[-1, 16, 16]");
}

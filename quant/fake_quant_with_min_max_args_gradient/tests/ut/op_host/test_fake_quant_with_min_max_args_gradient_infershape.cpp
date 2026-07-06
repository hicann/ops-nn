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
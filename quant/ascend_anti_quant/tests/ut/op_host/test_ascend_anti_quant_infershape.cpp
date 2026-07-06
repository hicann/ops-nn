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
#include "../../../op_graph/ascend_anti_quant_proto.h"

using namespace ge;
using namespace op;

class AscendAntiQuantProto : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "AscendAntiQuantProto SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "AscendAntiQuantProto TearDown" << std::endl; }
};

TEST_F(AscendAntiQuantProto, AscendAntiQuant_proto_int8_fp16)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("AscendAntiQuant")->infer_shape;

    gert::StorageShape xShape = {{1, 64}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1})
                      .NodeInputTd(0, ge::DT_INT8, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeAttrs({{"scale", Ops::NN::AnyValue::CreateFrom<float>(0.5f)},
                                  {"offset", Ops::NN::AnyValue::CreateFrom<float>(1.0f)},
                                  {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                  {"sqrt_mode", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .InputShapes({&xShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), "[1, 64]");
}

TEST_F(AscendAntiQuantProto, AscendAntiQuant_proto_hifloat8_fp32)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("AscendAntiQuant")->infer_shape;

    gert::StorageShape xShape = {{4, 2, 1024}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1})
                      .NodeInputTd(0, ge::DT_HIFLOAT8, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeAttrs({{"scale", Ops::NN::AnyValue::CreateFrom<float>(2.0f)},
                                  {"offset", Ops::NN::AnyValue::CreateFrom<float>(0.0f)},
                                  {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                  {"sqrt_mode", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
                      .InputShapes({&xShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), "[4, 2, 1024]");
}

TEST_F(AscendAntiQuantProto, AscendAntiQuant_proto_fp8_e5m2_fp16)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("AscendAntiQuant")->infer_shape;

    gert::StorageShape xShape = {{8, 16, 32}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1})
                      .NodeInputTd(0, ge::DT_FLOAT8_E5M2, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeAttrs({{"scale", Ops::NN::AnyValue::CreateFrom<float>(1.5f)},
                                  {"offset", Ops::NN::AnyValue::CreateFrom<float>(-0.5f)},
                                  {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                  {"sqrt_mode", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .InputShapes({&xShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), "[8, 16, 32]");
}

TEST_F(AscendAntiQuantProto, AscendAntiQuant_proto_fp8_e4m3_fp32)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("AscendAntiQuant")->infer_shape;

    gert::StorageShape xShape = {{16}, {}};
    gert::StorageShape yShape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1})
                      .NodeInputTd(0, ge::DT_FLOAT8_E4M3FN, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::Format::FORMAT_ND, ge::Format::FORMAT_ND)
                      .NodeAttrs({{"scale", Ops::NN::AnyValue::CreateFrom<float>(1.0f)},
                                  {"offset", Ops::NN::AnyValue::CreateFrom<float>(0.0f)},
                                  {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                  {"sqrt_mode", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .InputShapes({&xShape})
                      .OutputShapes({&yShape})
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape* output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), "[16]");
}

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include <vector>
#include "array_ops.h"
#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "../../../../../tests/ut/common/any_value.h"

class EluGradV2Test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "elu_grad_v2 test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "elu_grad_v2 test TearDown" << std::endl;
    }
};

TEST_F(EluGradV2Test, elu_grad_v2_infershape_success_fp16_1)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("EluGradV2")->infer_shape;
    gert::StorageShape Shape = {{3, 4}, {3, 4}};
    gert::StorageShape yShape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&Shape, &Shape})
                      .OutputShapes({&yShape})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"alpha", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                           {"scale", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                           {"input_scale", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                           {"is_result", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
                      .Build();
    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output_desc = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape expected_output_shape = {3, 4};
    ASSERT_EQ(Ops::Base::ToString(*output_desc), Ops::Base::ToString(expected_output_shape));
}

TEST_F(EluGradV2Test, elu_grad_v2_infershape_success_fp16_2)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("EluGradV2")->infer_shape;
    gert::StorageShape Shape = {{128, 256}, {128, 256}};
    gert::StorageShape yShape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&Shape, &Shape})
                      .OutputShapes({&yShape})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"alpha", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                           {"scale", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                           {"input_scale", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                           {"is_result", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
                      .Build();
    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output_desc = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape expected_output_shape = {128, 256};
    ASSERT_EQ(Ops::Base::ToString(*output_desc), Ops::Base::ToString(expected_output_shape));
}

TEST_F(EluGradV2Test, elu_grad_v2_infershape_success_bf16_3)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("EluGradV2")->infer_shape;
    gert::StorageShape Shape = {{32, 64, 128}, {32, 64, 128}};
    gert::StorageShape yShape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&Shape, &Shape})
                      .OutputShapes({&yShape})
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"alpha", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                           {"scale", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                           {"input_scale", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                           {"is_result", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
                      .Build();
    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output_desc = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape expected_output_shape = {32, 64, 128};
    ASSERT_EQ(Ops::Base::ToString(*output_desc), Ops::Base::ToString(expected_output_shape));
}

TEST_F(EluGradV2Test, elu_grad_v2_infershape_success_empty_4)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("EluGradV2")->infer_shape;
    gert::StorageShape Shape = {{0, 0}, {0, 0}};
    gert::StorageShape yShape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&Shape, &Shape})
                      .OutputShapes({&yShape})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"alpha", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                           {"scale", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                           {"input_scale", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                           {"is_result", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
                      .Build();
    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output_desc = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape expected_output_shape = {0, 0};
    ASSERT_EQ(Ops::Base::ToString(*output_desc), Ops::Base::ToString(expected_output_shape));
}

TEST_F(EluGradV2Test, elu_grad_v2_infershape_success_scalar_5)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("EluGradV2")->infer_shape;
    gert::StorageShape Shape = {{}, {}};
    gert::StorageShape yShape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&Shape, &Shape})
                      .OutputShapes({&yShape})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"alpha", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                           {"scale", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                           {"input_scale", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                           {"is_result", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
                      .Build();
    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output_desc = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape expected_output_shape = {};
    ASSERT_EQ(Ops::Base::ToString(*output_desc), Ops::Base::ToString(expected_output_shape));
}

TEST_F(EluGradV2Test, elu_grad_v2_infershape_success_alpha_2_6)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("EluGradV2")->infer_shape;
    gert::StorageShape Shape = {{4, 8, 16}, {4, 8, 16}};
    gert::StorageShape yShape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&Shape, &Shape})
                      .OutputShapes({&yShape})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"alpha", Ops::NN::AnyValue::CreateFrom<float>(2.0)},
                           {"scale", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                           {"input_scale", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                           {"is_result", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
                      .Build();
    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output_desc = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape expected_output_shape = {4, 8, 16};
    ASSERT_EQ(Ops::Base::ToString(*output_desc), Ops::Base::ToString(expected_output_shape));
}

TEST_F(EluGradV2Test, elu_grad_v2_infershape_success_is_result_false_7)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("EluGradV2")->infer_shape;
    gert::StorageShape Shape = {{8, 16}, {8, 16}};
    gert::StorageShape yShape = {{}, {}};
    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&Shape, &Shape})
                      .OutputShapes({&yShape})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"alpha", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                           {"scale", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                           {"input_scale", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                           {"is_result", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .Build();
    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output_desc = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    gert::Shape expected_output_shape = {8, 16};
    ASSERT_EQ(Ops::Base::ToString(*output_desc), Ops::Base::ToString(expected_output_shape));
}

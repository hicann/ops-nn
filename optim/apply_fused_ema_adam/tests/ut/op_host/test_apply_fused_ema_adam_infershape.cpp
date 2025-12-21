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
#include "infershape_test_util.h" // NOLINT
#include "ut_op_common.h"
#include "../../../op_graph/apply_fused_ema_adam_proto.h"

std::vector<int64_t> ToVectorForFused(const gert::Shape& shape) {
  size_t shape_size = shape.GetDimNum();
  std::vector<int64_t> shape_vec(shape_size, 0);

  for (size_t i = 0; i < shape_size; i++) {
    shape_vec[i] = shape.GetDim(i);
  }
  return shape_vec;
}

class ApplyFusedEmaAdam : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "ApplyFusedEmaAdam Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "ApplyFusedEmaAdam Proto Test TearDown" << std::endl;
    }
};

TEST_F(ApplyFusedEmaAdam, apply_fused_ema_dam_infershape_case0)
{
    gert::StorageShape gradShape = {{42, 6480}, {42, 6480}};
    gert::StorageShape varShape = {{42, 6480}, {42, 6480}};
    gert::StorageShape mShape = {{42, 6480}, {42, 6480}};
    gert::StorageShape vShape = {{42, 6480}, {42, 6480}};
    gert::StorageShape sShape = {{42, 6480}, {42, 6480}};
    gert::StorageShape stepShape = {{2, 8}, {2, 8}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("ApplyFusedEmaAdam")
                      .NodeIoNum(6, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1})
                      .InputShapes({&gradShape, &varShape, &mShape, &vShape, &sShape, &stepShape})
                      .OutputShapes({&varShape, &mShape, &vShape, &sShape})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"lr", Ops::NN::AnyValue::CreateFrom(1e-3f)},
                           {"ema_decay", Ops::NN::AnyValue::CreateFrom(0.9999f)},
                           {"beta1", Ops::NN::AnyValue::CreateFrom(0.9f)},
                           {"beta2", Ops::NN::AnyValue::CreateFrom(0.999f)},
                           {"eps", Ops::NN::AnyValue::CreateFrom(1e-8f)},
                           {"mode", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                           {"bias_correction", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                           {"weight_decay", Ops::NN::AnyValue::CreateFrom(0.0001f)}})
                      .Build();

    gert::InferShapeContext* context = holder.GetContext<gert::InferShapeContext>();
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("ApplyFusedEmaAdam")->infer_shape;
    ge::graphStatus ret = infer_shape_func(context);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    std::vector<int64_t> expectedOutputShape = {42, 6480};
    auto varOutShape = context->GetOutputShape(0);
    auto mOutShape = context->GetOutputShape(1);
    auto vOutShape = context->GetOutputShape(2);
    auto sOutShape = context->GetOutputShape(3);
    EXPECT_EQ(ToVectorForFused(*varOutShape), expectedOutputShape);
    EXPECT_EQ(ToVectorForFused(*mOutShape), expectedOutputShape);
    EXPECT_EQ(ToVectorForFused(*vOutShape), expectedOutputShape);
    EXPECT_EQ(ToVectorForFused(*sOutShape), expectedOutputShape);
}

TEST_F(ApplyFusedEmaAdam, apply_fused_ema_dam_infershape_bf16_case0)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("ApplyFusedEmaAdam"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("ApplyFusedEmaAdam")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_ref = ge::DT_BF16;
        ge::DataType input_step_ref = ge::DT_INT64;
        ge::DataType output_ref = ge::DT_BF16;
        auto context_holder =
            gert::InferDataTypeContextFaker()
                .NodeIoNum(6, 4)
                .IrInstanceNum({1, 1, 1, 1, 1, 1})
                .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(3, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(4, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(5, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(3, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                .InputDataTypes({&input_ref, &input_ref, &input_ref, &input_ref, &input_ref, &input_step_ref})
                .OutputDataTypes({&output_ref, &output_ref, &output_ref, &output_ref})
                .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetInputDataType(0), input_ref);
        EXPECT_EQ(context->GetInputDataType(1), input_ref);
        EXPECT_EQ(context->GetInputDataType(2), input_ref);
        EXPECT_EQ(context->GetInputDataType(3), input_ref);
        EXPECT_EQ(context->GetInputDataType(4), input_ref);
        EXPECT_EQ(context->GetInputDataType(5), input_step_ref);
        EXPECT_EQ(context->GetOutputDataType(0), output_ref);
        EXPECT_EQ(context->GetOutputDataType(1), output_ref);
        EXPECT_EQ(context->GetOutputDataType(2), output_ref);
        EXPECT_EQ(context->GetOutputDataType(3), output_ref);
    }
}
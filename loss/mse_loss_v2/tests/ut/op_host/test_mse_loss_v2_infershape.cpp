/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h> // NOLINT
#include <iostream>
#include "infershape_test_util.h" // NOLINT
#include "ut_op_common.h"

std::vector<int64_t> ToVectorForMseLossV2(const gert::Shape& shape) {
  size_t shape_size = shape.GetDimNum();
  std::vector<int64_t> shape_vec(shape_size, 0);

  for (size_t i = 0; i < shape_size; i++) {
    shape_vec[i] = shape.GetDim(i);
  }
  return shape_vec;
}

class MSELossV2 : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "MSELossV2 Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MSELossV2 Proto Test TearDown" << std::endl;
    }
};

TEST_F(MSELossV2, MSELossV2_infershape_case_0)
{
    gert::StorageShape input_shape = {{30, 1024}, {30, 1024}};
    gert::StorageShape target_shape = {{30, 1024}, {30, 1024}};
    // output
    gert::StorageShape output_shape = {{30, 1024}, {30, 1024}};

    auto holder = gert::InferShapeContextFaker()
                      .SetOpType("MSELossV2")
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&input_shape, &target_shape})
                      .OutputShapes({&output_shape})
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"reduction", Ops::NN::AnyValue::CreateFrom<string>("none")}})
                      .Build();

    gert::InferShapeContext* context = holder.GetContext<gert::InferShapeContext>();
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MSELossV2")->infer_shape;
    ge::graphStatus ret = infer_shape_func(context);
    EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    std::vector<int64_t> expectedxGradShape = {30, 1024};
    auto xGradShape = context->GetOutputShape(0);
    EXPECT_EQ(ToVectorForMseLossV2(*xGradShape), expectedxGradShape);
}

TEST_F(MSELossV2, MSELossV2_infershape_bf16_case_0)
{
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MSELossV2"), nullptr);
    auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MSELossV2")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_ref = ge::DT_BF16;
        ge::DataType input_target_ref = ge::DT_BF16;
        ge::DataType output_ref = ge::DT_BF16;
        auto context_holder = gert::InferDataTypeContextFaker()
                                  .IrInputNum(2)
                                  .NodeIoNum(2, 1)
                                  .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                                  .InputDataTypes({&input_ref, &input_target_ref})
                                  .OutputDataTypes({&output_ref})
                                  .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetInputDataType(0), input_ref);
        EXPECT_EQ(context->GetInputDataType(1), input_target_ref);
        EXPECT_EQ(context->GetOutputDataType(0), output_ref);
    }
}
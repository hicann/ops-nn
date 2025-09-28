/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "ut_op_common.h"
#include "ut_op_util.h"

class CrossEntropyLossGrad : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "CrossEntropyLossGrad Proto Test SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "CrossEntropyLossGrad Proto Test TearDown" << std::endl;
    }
};

TEST_F(CrossEntropyLossGrad, CrossEntropyLossGrad_infershape_case_0) {
  gert::StorageShape y_grad_shape = {{}, {}};
  gert::StorageShape log_prob_shape = {{48, 112595}, {48, 112595}};
  gert::StorageShape target_shape = {{48,}, {48,}};
  gert::StorageShape weight_shape = {{112595, }, {112595, }};
  // output
  gert::StorageShape x_grad_shape = {{48, 112595}, {48, 112595}};

  auto holder = gert::InferShapeContextFaker()
                    .SetOpType("CrossEntropyLossGrad")
                    .NodeIoNum(4, 1)
                    .IrInstanceNum({1, 1, 1, 1})
                    .InputShapes({&y_grad_shape, &log_prob_shape, &target_shape, &weight_shape})
                    .OutputShapes({&x_grad_shape})
                    .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({{"reduction", ge::AnyValue::CreateFrom<string>("mean")},
                                {"ignore_index", ge::AnyValue::CreateFrom<int64_t>(-100)},
                                {"label_smoothing", ge::AnyValue::CreateFrom<float>(0.0)},
                                {"lse_square_scale_for_zloss", ge::AnyValue::CreateFrom<float>(0.0)}})
                    .Build();

  gert::InferShapeContext *context = holder.GetContext<gert::InferShapeContext>();
  auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("CrossEntropyLossGrad")->infer_shape;
  ge::graphStatus ret = infer_shape_func(context);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  std::vector<int64_t> expectedxGradShape = {48, 112595};
  auto xGradShape = context->GetOutputShape(0);
  EXPECT_EQ(ToVector(*xGradShape), expectedxGradShape);
}

TEST_F(CrossEntropyLossGrad, CrossEntropyLossGrad_infershape_bf16_case_0) {
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("CrossEntropyLossGrad"), nullptr);
  auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("CrossEntropyLossGrad")->infer_datatype;

  if (data_type_func != nullptr) {
    ge::DataType input_ref = ge::DT_BF16;
    ge::DataType input_target_ref = ge::DT_INT64;
    ge::DataType input_weight_ref = ge::DT_FLOAT;
    ge::DataType output_ref = ge::DT_BF16;
    auto context_holder = gert::InferDataTypeContextFaker()
        .IrInputNum(4)
        .NodeIoNum(4, 2)
        .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
        .InputDataTypes({&input_ref, &input_ref, &input_target_ref, &input_weight_ref})
        .OutputDataTypes({&output_ref})
        .Build();
    auto context = context_holder.GetContext<gert::InferDataTypeContext>();
    EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
    ASSERT_NE(context, nullptr);

    EXPECT_EQ(context->GetInputDataType(0), input_ref);
    EXPECT_EQ(context->GetInputDataType(1), input_ref);
    EXPECT_EQ(context->GetInputDataType(2), input_target_ref);
    EXPECT_EQ(context->GetInputDataType(3), input_weight_ref);
    EXPECT_EQ(context->GetOutputDataType(0), output_ref);
  }
}

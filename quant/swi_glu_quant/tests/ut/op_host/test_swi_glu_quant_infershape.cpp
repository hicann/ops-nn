/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file test_swi_glu_quant_infershape.cpp
 *
 * @brief
 *
 * @version 1.0
 *
 */
#include <gtest/gtest.h>
#include <iostream>
#include "infershape_test_util.h"
#include "ut_op_common.h"
#include "../../../op_graph/swi_glu_quant_proto.h"

class SwiGluQuantTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "SwiGluQuantTest Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "SwiGluQuantTest Proto Test TearDown" << std::endl;
  }
};

TEST_F(SwiGluQuantTest, SwiGluQuant_infershape_diff_test_legal_input) {
  ge::op::SwiGluQuant op;
  op.UpdateInputDesc("x", create_desc({4, 4}, ge::DT_BF16));
  op.SetAttr("activate_left", false);
  op.SetAttr("quant_mode", "dynamic");
  op.SetAttr("group_list_type", 0);
  op.SetAttr("dst_type", 2);
  Runtime2TestParam param{{"activate_left", "quant_mode", "group_list_type", "dst_type"}};
  EXPECT_EQ(InferShapeTest(op, param), ge::GRAPH_SUCCESS);

  auto output_y_desc = op.GetOutputDesc(0);
  std::vector<int64_t> expected_output_shape = {4, 2};
  EXPECT_EQ(output_y_desc.GetShape().GetDims(), expected_output_shape);
}

TEST_F(SwiGluQuantTest, SwiGluQuant_infer_dtype_test) {
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("SwiGluQuant"), nullptr);
  auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("SwiGluQuant")->infer_datatype;

  if (data_type_func != nullptr) {
    ge::DataType input_ref = ge::DT_BF16;
    ge::DataType input_ref0 = ge::DT_INT32;
    ge::DataType output_ref0 = ge::DT_INT8;
    ge::DataType output_ref = ge::DT_FLOAT;
    auto context_holder =
        gert::InferDataTypeContextFaker()
            .IrInputNum(4)
            .NodeIoNum(4, 2)
            .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .InputDataTypes({&input_ref, &output_ref, &output_ref, &input_ref0})
            .OutputDataTypes({&output_ref0, &output_ref})
            .NodeAttrs({{"activate_left", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                        {"quant_mode", Ops::NN::AnyValue::CreateFrom<std::string>("dynamic")},
                        {"group_list_type", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                        {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(2)}})
            .Build();
    auto context = context_holder.GetContext<gert::InferDataTypeContext>();
    EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
    ASSERT_NE(context, nullptr);

    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_INT8);
    EXPECT_EQ(context->GetOutputDataType(1), ge::DT_FLOAT);
  }
}
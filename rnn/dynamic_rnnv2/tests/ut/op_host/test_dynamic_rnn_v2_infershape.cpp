/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


#include <gtest/gtest.h>
#include <vector>
#include "../../../op_graph/dynamic_rnnv2_proto.h"
#include "ut_op_common.h"

class DynamicRnnV2Test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "dynamic_rnn_v2 test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "dynamic_rnn_v2 test TearDown" << std::endl;
  }
};

TEST_F(DynamicRnnV2Test, dynamic_rnn_test_case_1) {
  int t = 3;
  int batch = 16;
  int inputSize = 32;
  int outputSize = 48;
  ge::op::DynamicRNNV2 rnn_op;
  ge::TensorDesc XDesc;
  ge::Shape xShape({t, batch, inputSize});
  XDesc.SetDataType(ge::DT_FLOAT16);
  XDesc.SetShape(xShape);
  XDesc.SetOriginShape(xShape);

  ge::TensorDesc WiDesc;
  ge::TensorDesc WhDesc;
  ge::Shape wiShape({inputSize, 4 * outputSize});
  ge::Shape whShape({outputSize, 4 * outputSize});
  WiDesc.SetDataType(ge::DT_FLOAT16);
  WhDesc.SetDataType(ge::DT_FLOAT16);
  WiDesc.SetShape(wiShape);
  WhDesc.SetShape(whShape);
  WiDesc.SetOriginShape(wiShape);
  WhDesc.SetOriginShape(whShape);

  ge::TensorDesc BDesc;
  ge::Shape bShape({4 * outputSize});
  BDesc.SetDataType(ge::DT_FLOAT16);
  BDesc.SetShape(bShape);
  BDesc.SetOriginShape(bShape);

  rnn_op.UpdateInputDesc("x", XDesc);
  rnn_op.UpdateInputDesc("weight_input", WiDesc);
  rnn_op.UpdateInputDesc("weight_hidden", WhDesc);
  rnn_op.UpdateInputDesc("b", BDesc);

  // auto status = rnn_op.VerifyAllAttr(true);
  // EXPECT_EQ(status, ge::GRAPH_SUCCESS);
  // auto ret = rnn_op.InferShapeAndType();
  // EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

  // auto output_desc = rnn_op.GetOutputDesc("y");
  // EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
  std::vector<int64_t> expected_output_shape = {t, batch, outputSize};
  // EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);

  EXPECT_EQ(InferShapeTest(rnn_op), ge::GRAPH_SUCCESS);
  auto output0_desc = rnn_op.GetOutputDesc(0);
  EXPECT_EQ(output0_desc.GetShape().GetDims(), expected_output_shape);

  std::vector<int64_t> expected_output1_shape = {1, batch, outputSize};
  auto output1_desc = rnn_op.GetOutputDesc(1);
  EXPECT_EQ(output1_desc.GetShape().GetDims(), expected_output1_shape);
}

TEST_F(DynamicRnnV2Test, dynamic_rnn_test_case_2) {
  int t = 3;
  int batch = 16;
  int inputSize = 32;
  int outputSize = 48;
  ge::op::DynamicRNNV2 rnn_op;
  ge::TensorDesc XDesc;
  ge::Shape xShape({t, batch, inputSize, 1});
  XDesc.SetDataType(ge::DT_FLOAT16);
  XDesc.SetShape(xShape);
  XDesc.SetOriginShape(xShape);

  ge::TensorDesc WiDesc;
  ge::TensorDesc WhDesc;
  ge::Shape wiShape({inputSize, 4 * outputSize});
  ge::Shape whShape({outputSize, 4 * outputSize});
  WiDesc.SetDataType(ge::DT_FLOAT16);
  WhDesc.SetDataType(ge::DT_FLOAT16);
  WiDesc.SetShape(wiShape);
  WhDesc.SetShape(whShape);
  WiDesc.SetOriginShape(wiShape);
  WhDesc.SetOriginShape(whShape);

  ge::TensorDesc BDesc;
  ge::Shape bShape({4 * outputSize});
  BDesc.SetDataType(ge::DT_FLOAT16);
  BDesc.SetShape(bShape);
  BDesc.SetOriginShape(bShape);

  rnn_op.UpdateInputDesc("x", XDesc);
  rnn_op.UpdateInputDesc("weight_input", WiDesc);
  rnn_op.UpdateInputDesc("weight_hidden", WhDesc);
  rnn_op.UpdateInputDesc("b", BDesc);

  // auto ret = rnn_op.InferShapeAndType();
  // EXPECT_EQ(ret, ge::GRAPH_FAILED);

  EXPECT_EQ(InferShapeTest(rnn_op), ge::GRAPH_FAILED);
}

TEST_F(DynamicRnnV2Test, dynamic_rnn_v2_test_inferdtype_0) {
  ge::op::DynamicRNNV2 op;

  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("DynamicRNNV2"), nullptr);
  auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("DynamicRNNV2")->infer_datatype;
  ASSERT_NE(data_type_func, nullptr);

  ge::DataType x_datatype1 = ge::DT_FLOAT16;
  ge::DataType wi_datatype1 = ge::DT_FLOAT16;
  ge::DataType wh_datatype1 = ge::DT_FLOAT16;
  ge::DataType b_datatype1 = ge::DT_FLOAT16;
  ge::DataType out_datatype = ge::DT_FLOAT16;
  auto context_holder = gert::InferDataTypeContextFaker()
                            .IrInputNum(4)
                            .NodeIoNum(4, 8)
                            .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                            .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                            .InputDataTypes({&x_datatype1, &wi_datatype1, &wh_datatype1, &b_datatype1})
                            .OutputDataTypes({&out_datatype})
                            .Build();
  auto context = context_holder.GetContext<gert::InferDataTypeContext>();
  EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
  ASSERT_NE(context, nullptr);
}
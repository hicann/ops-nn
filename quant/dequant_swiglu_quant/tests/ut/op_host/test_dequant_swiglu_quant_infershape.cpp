/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_dequant_swiglu_quant_infershape.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infershape_test_util.h"
#include "ut_op_common.h"
#include "test_cube_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "ut_op_util.h"
#include "op_common/op_host/util/platform_util.h"

class DequantSwigluQuant : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "DequantSwigluQuant Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "DequantSwigluQuant Proto Test TearDown" << std::endl;
  }
};

TEST_F(DequantSwigluQuant, DequantSwigluQuant_infershape) {
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("DequantSwigluQuant"), nullptr);
  auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("DequantSwigluQuant")->infer_shape;
  ASSERT_NE(inferShapeFunc, nullptr);

  gert::StorageShape xShape = {{32, 128}, {32, 128}};
  gert::StorageShape out0Shape;
  gert::StorageShape out1Shape;

  auto holder = gert::InferShapeContextFaker()
      .NodeIoNum(7, 2)
      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
      .InputShapes({&xShape, &xShape, &xShape, &xShape, &xShape, &xShape, &xShape})
      .OutputShapes({&out0Shape, &out1Shape})
      .NodeAttrs({})
      .Build();

  ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
  auto output0 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
  auto output1 = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(1);

  ASSERT_EQ(Ops::Base::ToString(*output0), "[32, 64]");
  ASSERT_EQ(Ops::Base::ToString(*output1), "[32]");
}

TEST_F(DequantSwigluQuant, DequantSwigluQuant_inferdtype) {
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("DequantSwigluQuant"), nullptr);
  auto data_type_func = gert::OpImplRegistry::GetInstance().GetOpImpl("DequantSwigluQuant")->infer_datatype;
  ASSERT_NE(data_type_func, nullptr);
  ge::DataType input_0 = ge::DT_BF16;
  ge::DataType output_0 = ge::DT_BF16;
  ge::DataType output_1 = ge::DT_BF16;
  auto context_holder = gert::InferDataTypeContextFaker()
      .IrInputNum(7)
      .NodeIoNum(7, 2)
      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
      .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
      .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
      .NodeInputTd(4, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
      .NodeInputTd(5, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
      .NodeInputTd(6, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
      .NodeAttrs({})
      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
      .NodeOutputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
      .InputDataTypes({&input_0, &input_0, &input_0, &input_0, &input_0, &input_0, &input_0})
      .OutputDataTypes({&output_0, &output_1})
      .Build();
  auto context = context_holder.GetContext<gert::InferDataTypeContext>();
  EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
  ASSERT_NE(context, nullptr);

  EXPECT_EQ(context->GetOutputDataType(0), ge::DT_INT8);
  EXPECT_EQ(context->GetOutputDataType(1), ge::DT_FLOAT);
}


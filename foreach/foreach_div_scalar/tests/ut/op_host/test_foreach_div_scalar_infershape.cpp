/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * @file test_foreach_div_scalar_infershape.cpp
 *
 * @brief
 *
 * @Version 2.0
 *
 */

#include "gtest/gtest.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/kernel_context.h"
#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"
#include "ut_op_common.h"
#include "infershape_test_util.h"
#include "log/log.h"
#include "../../../op_graph/foreach_div_scalar_proto.h"

class ForeachDivScalarTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ForeachDivScalar SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ForeachDivScalar TearDown" << std::endl;
  }
};

TEST_F(ForeachDivScalarTest, infer_shape_known_success) {
  auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("ForeachDivScalar")->infer_shape;
  ASSERT_NE(infer_shape_func, nullptr);

  //x1
  gert::StorageShape x_shape_0 = {{1}, {}};
  gert::StorageShape x_shape_1 = {{2}, {}};
  gert::StorageShape x_shape_2 = {{3}, {}};
  //alpha
  gert::StorageShape x_shape_3 = {{}, {}};
  //y
  gert::StorageShape y_shape_0 = {{}, {}};
  gert::StorageShape y_shape_1 = {{}, {}};
  gert::StorageShape y_shape_2 = {{}, {}};

  std::vector<void*> input_shape_ref(4);
  input_shape_ref[0] = &x_shape_0;
  input_shape_ref[1] = &x_shape_1;
  input_shape_ref[2] = &x_shape_2;
  input_shape_ref[3] = &x_shape_3;

  std::vector<void*> output_shape_ref(3);
  output_shape_ref[0] = &y_shape_0;
  output_shape_ref[1] = &y_shape_1;
  output_shape_ref[2] = &y_shape_2;

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(4, 3)
                    .IrInstanceNum({3, 1})
                    .InputShapes(input_shape_ref)
                    .OutputShapes(output_shape_ref)
                    .Build();

  auto context = holder.GetContext<gert::InferShapeContext>();
  ASSERT_NE(context, nullptr);
  ASSERT_EQ(infer_shape_func(context), ge::GRAPH_SUCCESS);

  auto output_shape_0 = context->GetOutputShape(0);
  EXPECT_EQ(Ops::Base::ToString(*output_shape_0), "[1]");

  auto output_shape_1 = context->GetOutputShape(1);
  EXPECT_EQ(Ops::Base::ToString(*output_shape_1), "[2]");

  auto output_shape_2 = context->GetOutputShape(2);
  EXPECT_EQ(Ops::Base::ToString(*output_shape_2), "[3]");
}

TEST_F(ForeachDivScalarTest, infer_dtype_test_1) {
  auto infer_datatype_func = gert::OpImplRegistry::GetInstance().GetOpImpl("ForeachDivScalar")->infer_datatype;
  ASSERT_NE(infer_datatype_func, nullptr);

  //x1
  ge::DataType x_dtype_0 = ge::DT_FLOAT16;
  ge::DataType x_dtype_1 = ge::DT_FLOAT16;
  ge::DataType x_dtype_2 = ge::DT_FLOAT16;
  //alpha
  ge::DataType x_dtype_3 = ge::DT_FLOAT16;

  std::vector<void*> input_dtype_ref(6);
  input_dtype_ref[0] = &x_dtype_0;
  input_dtype_ref[1] = &x_dtype_1;
  input_dtype_ref[2] = &x_dtype_2;
  input_dtype_ref[3] = &x_dtype_3;

  std::vector<void*> output_dtype_ref(3);

  auto holder = gert::InferDataTypeContextFaker()
                    .NodeIoNum(4, 3)
                    .IrInstanceNum({3, 1})
                    .InputDataTypes(input_dtype_ref)
                    .OutputDataTypes(output_dtype_ref)
                    .Build();

  auto context = holder.GetContext<gert::InferDataTypeContext>();
  ASSERT_NE(context, nullptr);
  ASSERT_EQ(infer_datatype_func(context), ge::GRAPH_SUCCESS);

  ge::DataType expected_datatype = ge::DT_FLOAT16;
  // EXPECT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
  auto output_dtype_0 = context->GetOutputDataType(0);
  EXPECT_EQ(output_dtype_0, expected_datatype);

  auto output_dtype_1 = context->GetOutputDataType(1);
  EXPECT_EQ(output_dtype_1, expected_datatype);

  auto output_dtype_2 = context->GetOutputDataType(2);
  EXPECT_EQ(output_dtype_2, expected_datatype);
}


TEST_F(ForeachDivScalarTest, infer_dtype_test_2) {
  auto infer_datatype_func = gert::OpImplRegistry::GetInstance().GetOpImpl("ForeachDivScalar")->infer_datatype;
  ASSERT_NE(infer_datatype_func, nullptr);

  //x1
  ge::DataType x_dtype_0 = ge::DT_FLOAT;
  ge::DataType x_dtype_1 = ge::DT_FLOAT;
  ge::DataType x_dtype_2 = ge::DT_FLOAT;
  //alpha
  ge::DataType x_dtype_3 = ge::DT_FLOAT;

  std::vector<void*> input_dtype_ref(6);
  input_dtype_ref[0] = &x_dtype_0;
  input_dtype_ref[1] = &x_dtype_1;
  input_dtype_ref[2] = &x_dtype_2;
  input_dtype_ref[3] = &x_dtype_3;

  std::vector<void*> output_dtype_ref(3);

  auto holder = gert::InferDataTypeContextFaker()
                    .NodeIoNum(4, 3)
                    .IrInstanceNum({3, 1})
                    .InputDataTypes(input_dtype_ref)
                    .OutputDataTypes(output_dtype_ref)
                    .Build();

  auto context = holder.GetContext<gert::InferDataTypeContext>();
  ASSERT_NE(context, nullptr);
  ASSERT_EQ(infer_datatype_func(context), ge::GRAPH_SUCCESS);

  ge::DataType expected_datatype = ge::DT_FLOAT;
  auto output_dtype_0 = context->GetOutputDataType(0);
  EXPECT_EQ(output_dtype_0, expected_datatype);

  auto output_dtype_1 = context->GetOutputDataType(1);
  EXPECT_EQ(output_dtype_1, expected_datatype);

  auto output_dtype_2 = context->GetOutputDataType(2);
  EXPECT_EQ(output_dtype_2, expected_datatype);
}
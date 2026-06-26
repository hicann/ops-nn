/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

 #include <gtest/gtest.h>  // NOLINT
 #include <iostream>
 #include "infershape_test_util.h"
 #include "ut_op_common.h"
 #include "../../../op_graph/squared_relu_proto.h"
 #include "log/log.h"
 class SquaredRelu : public testing::Test {
  protected:
   static void SetUpTestCase() {
     std::cout << "SquaredRelu SetUp" << std::endl;
   }

   static void TearDownTestCase() {
     std::cout << "SquaredRelu TearDown" << std::endl;
   }
 };

 // 测试FLOAT16数据类型，验证输出shape与输入一致
 TEST_F(SquaredRelu, SquaredRelu_infershape_case_0) {
   ge::op::SquaredRelu op;
   op.UpdateInputDesc("x", create_desc({4, 1, 1280}, ge::DT_FLOAT16));

   EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);

   auto output_desc = op.GetOutputDesc("y");
   auto output_shape = output_desc.GetShape();
   EXPECT_EQ(output_shape.GetDimNum(), 3);
   std::vector<int64_t> expected_output_shape = {4, 1, 1280};
   EXPECT_EQ(output_shape.GetDims(), expected_output_shape);
 }

 // 测试FLOAT数据类型
 TEST_F(SquaredRelu, SquaredRelu_infershape_case_1_fp32) {
   ge::op::SquaredRelu op;
   op.UpdateInputDesc("x", create_desc({2, 3, 4}, ge::DT_FLOAT));

   EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);

   auto output_desc = op.GetOutputDesc("y");
   auto output_shape = output_desc.GetShape();
   EXPECT_EQ(output_shape.GetDimNum(), 3);
   std::vector<int64_t> expected_output_shape = {2, 3, 4};
   EXPECT_EQ(output_shape.GetDims(), expected_output_shape);
 }

 // 测试BF16数据类型
 TEST_F(SquaredRelu, SquaredRelu_infershape_case_2_bf16) {
   ge::op::SquaredRelu op;
   op.UpdateInputDesc("x", create_desc({8, 256}, ge::DT_BF16));

   EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);

   auto output_desc = op.GetOutputDesc("y");
   auto output_shape = output_desc.GetShape();
   EXPECT_EQ(output_shape.GetDimNum(), 2);
   std::vector<int64_t> expected_output_shape = {8, 256};
   EXPECT_EQ(output_shape.GetDims(), expected_output_shape);
 }

 // 测试1D shape
 TEST_F(SquaredRelu, SquaredRelu_infershape_case_3_1d) {
   ge::op::SquaredRelu op;
   op.UpdateInputDesc("x", create_desc({1024}, ge::DT_FLOAT16));

   EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);

   auto output_desc = op.GetOutputDesc("y");
   auto output_shape = output_desc.GetShape();
   EXPECT_EQ(output_shape.GetDimNum(), 1);
   std::vector<int64_t> expected_output_shape = {1024};
   EXPECT_EQ(output_shape.GetDims(), expected_output_shape);
 }

 // 测试4D shape
 TEST_F(SquaredRelu, SquaredRelu_infershape_case_4_4d) {
   ge::op::SquaredRelu op;
   op.UpdateInputDesc("x", create_desc({2, 3, 4, 5}, ge::DT_FLOAT));

   EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);

   auto output_desc = op.GetOutputDesc("y");
   auto output_shape = output_desc.GetShape();
   EXPECT_EQ(output_shape.GetDimNum(), 4);
   std::vector<int64_t> expected_output_shape = {2, 3, 4, 5};
   EXPECT_EQ(output_shape.GetDims(), expected_output_shape);
 }

 // 测试8D shape（最大支持维度）
 TEST_F(SquaredRelu, SquaredRelu_infershape_case_5_8d) {
   ge::op::SquaredRelu op;
   op.UpdateInputDesc("x", create_desc({1, 2, 1, 2, 1, 2, 1, 2}, ge::DT_FLOAT16));

   EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);

   auto output_desc = op.GetOutputDesc("y");
   auto output_shape = output_desc.GetShape();
   EXPECT_EQ(output_shape.GetDimNum(), 8);
   std::vector<int64_t> expected_output_shape = {1, 2, 1, 2, 1, 2, 1, 2};
   EXPECT_EQ(output_shape.GetDims(), expected_output_shape);
 }

 // 测试动态shape（含-1）
 TEST_F(SquaredRelu, SquaredRelu_infershape_case_6_dynamic) {
   ge::op::SquaredRelu op;

   std::vector<std::pair<int64_t, int64_t>> shape_range = {{1, 16}, {1, 16}};
   auto input_tensor =
       create_desc_shape_range({-1, -1}, ge::DT_FLOAT, ge::FORMAT_ND, {16, 16}, ge::FORMAT_ND, shape_range);
   op.UpdateInputDesc("x", input_tensor);

   EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);

   auto output_desc = op.GetOutputDesc("y");
   auto output_shape = output_desc.GetShape();
   EXPECT_EQ(output_shape.GetDimNum(), 2);
   std::vector<int64_t> expected_output_shape = {-1, -1};
   EXPECT_EQ(output_shape.GetDims(), expected_output_shape);
 }

 // 测试InferDataType输出数据类型与输入一致
 TEST_F(SquaredRelu, SquaredRelu_infershape_case_7_infer_dtype) {
   ge::op::SquaredRelu op;
   op.UpdateInputDesc("x", create_desc({4, 4}, ge::DT_FLOAT));

   EXPECT_EQ(InferShapeTest(op), ge::GRAPH_SUCCESS);

   auto output_desc = op.GetOutputDesc("y");
   EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT);
 }
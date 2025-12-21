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
#include "kernel_run_context_facker.h"
#include "common/utils/ut_op_common.h"

class AssertRt2UTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "AssertRt2UTest SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "AssertRt2UTest TearDown" << std::endl;
  }
};

TEST_F(AssertRt2UTest, InferShape4PrintV3_SUCCESS) {
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("PrintV3"), nullptr);
  auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("PrintV3")->infer_shape;
  ASSERT_NE(infer_shape_func, nullptr);

  gert::StorageShape x0_shape = {{3}, {3}};
  gert::StorageShape y0_shape = {{3}, {3}};
  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(1, 1)
                    .IrInputNum(1)
                    .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .InputShapes({&x0_shape})
                    .OutputShapes({&y0_shape})
                    .Build();
  EXPECT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}


TEST_F(AssertRt2UTest, InferDataType4PrintV3_SUCCESS) {
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("PrintV3"), nullptr);
    auto infer_datatype_func = gert::OpImplRegistry::GetInstance().GetOpImpl("PrintV3")->infer_datatype;
    ASSERT_NE(infer_datatype_func, nullptr);
    auto context_holder = gert::InferDataTypeContextFaker()
                            .NodeIoNum(1, 1)
                            .IrInputNum(1)
                            .NodeInputTd(0, ge::DT_STRING, ge::FORMAT_ND, ge::FORMAT_ND)
                            .Build();
    auto context = context_holder.GetContext<gert::InferDataTypeContext>();
    EXPECT_EQ(infer_datatype_func(context), ge::GRAPH_SUCCESS);
  }

  TEST_F(AssertRt2UTest, InferShape4PrintV2_SUCCESS) {
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("PrintV2"), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("PrintV2")->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::StorageShape x0_shape = {{}, {}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 0)
                      .IrInputNum(1)
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputShapes({&x0_shape})
                      .Build();
    EXPECT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
  }

  TEST_F(AssertRt2UTest, InferShape4PrintV2_Failed) {
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("PrintV2"), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("PrintV2")->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::StorageShape x0_shape = {{3}, {3}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 0)
                      .IrInputNum(1)
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputShapes({&x0_shape})
                      .Build();
    EXPECT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
  }

  TEST_F(AssertRt2UTest, InferShape4Assert_succ) {
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("Assert"), nullptr);
    auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("Assert")->infer_shape;
    ASSERT_NE(infer_shape_func, nullptr);

    gert::StorageShape x0_shape = {{3}, {3}};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(1, 0)
                      .IrInputNum(1)
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .InputShapes({&x0_shape})
                      .Build();
    EXPECT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
  }
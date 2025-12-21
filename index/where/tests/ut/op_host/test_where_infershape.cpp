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
#include "register/op_impl_registry.h"
#include "register/op_impl_registry_base.h"
#include "runtime/infer_shape_range_context.h"
#include "kernel_run_context_facker.h"

class WhereUT : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "Where SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "Where TearDown" << std::endl;
  }
};

TEST_F(WhereUT, infershape_range_test) {
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("Where"), nullptr);
  auto infer_shape_range_func = gert::OpImplRegistry::GetInstance().GetOpImpl("Where")->infer_shape_range;

  gert::Shape min1{2, 2, 3, 8};
  gert::Shape max1{2, -1, 3, 8};
  gert::Shape outmin{0, 4};
  gert::Shape outmax{-1, 4};
  gert::Shape null1{};
  gert::Shape null2{};

  gert::Range<gert::Shape> in_shape_range(&min1, &max1);
  gert::Range<gert::Shape> out_shape_range(&null1, &null2);
  gert::Range<gert::Shape> exp_out_shape_range(&outmin, &outmax);

  auto context_holder = gert::InferShapeRangeContextFaker()
      .IrInputNum(1)
      .NodeIoNum(1, 1)
      .NodeInputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
      .NodeOutputTd(0, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
      .InputShapeRanges({&in_shape_range})
      .OutputShapeRanges({&out_shape_range})
      .Build();

  auto context = context_holder.GetContext<gert::InferShapeRangeContext>();
  ASSERT_NE(context, nullptr);
  ASSERT_EQ(infer_shape_range_func(context), ge::GRAPH_SUCCESS);

  EXPECT_EQ(context->GetOutputShapeRange(0)->GetMin()->GetDim(0), 0);
  EXPECT_EQ(context->GetOutputShapeRange(0)->GetMin()->GetDim(1), 4);
  EXPECT_EQ(context->GetOutputShapeRange(0)->GetMax()->GetDim(0), -1);
  EXPECT_EQ(context->GetOutputShapeRange(0)->GetMax()->GetDim(1), 4);
}

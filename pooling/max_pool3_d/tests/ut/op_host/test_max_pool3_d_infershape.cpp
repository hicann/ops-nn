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
#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"
#include "../../../op_graph/max_pool3_d_proto.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "../../../../../tests/ut/common/any_value.h"

namespace {
class MaxPool3DUT : public testing::Test {protected:
  static void SetUpTestCase() {
    std::cout << "MaxPool3DUT Proto Test SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "MaxPool3DUT Proto Test TearDown" << std::endl;
  }};

TEST_F(MaxPool3DUT, InferShapeOk) {
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPool3D"), nullptr);
  auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MaxPool3D")->infer_shape;
  ASSERT_NE(infer_shape_func, nullptr);

  gert::StorageShape x_shape = {{2, 2, 5, 2, 7}, {2, 2, 5, 2, 7}};

  std::vector<gert::StorageShape> output_shapes(1);
  std::vector<void *> output_shapes_ref(1);
  for (size_t i = 0; i < output_shapes.size(); ++i) {
    output_shapes_ref[i] = &output_shapes[i];
  }

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(1, 1)
                    .IrInstanceNum({1})
                    .InputShapes({&x_shape})
                    .OutputShapes(output_shapes_ref)
                    .NodeAttrs({{"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 3, 1, 1})},
                                {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 4, 2, 4})},
                                {"padding", Ops::NN::AnyValue::CreateFrom<std::string>("SAME")},
                                {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1, 1, 1})},
                                {"dilation", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({5, 2, 4, 5, 2})},
                                {"ceil_mode", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")}})
                    .Build();

  EXPECT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
}
}

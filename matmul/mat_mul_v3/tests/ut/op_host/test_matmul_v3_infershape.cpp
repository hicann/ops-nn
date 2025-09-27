/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "register/op_impl_registry.h"
#include "kernel_run_context_facker.h"
#include "log/log.h"

namespace {
class TestMatMulV3InferShape : public testing::Test {
};

TEST_F(TestMatMulV3InferShape, Basic) {
  auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->infer_shape;

  gert::StorageShape x1_shape = {{32, 64}, {32, 64}};
  gert::StorageShape x2_shape = {{64, 128}, {8, 4, 16, 16}};
  gert::StorageShape output_shape = {{}, {}};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(2, 1)
                    .IrInstanceNum({1, 1})
                    .InputShapes({&x1_shape, &x2_shape})
                    .OutputShapes({&output_shape})
                    .NodeAttrs({{"transpose_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"transpose_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                    .Build();

  ASSERT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
  auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
  ASSERT_EQ(Ops::Base::ToString(*output), "[32, 128]");
}

TEST_F(TestMatMulV3InferShape, BasicWithBias) {
  auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->infer_shape;

  gert::StorageShape x1_shape = {{32, 64}, {4, 2, 16, 16}};
  gert::StorageShape x2_shape = {{64, 128}, {8, 4, 16, 16}};
  gert::StorageShape bias_shape = {{128}, {128}};
  gert::StorageShape output_shape = {{}, {}};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(3, 1)
                    .IrInstanceNum({1, 1, 1})
                    .InputShapes({&x1_shape, &x2_shape, &bias_shape})
                    .OutputShapes({&output_shape})
                    .NodeAttrs({{"transpose_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"transpose_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                    .Build();

  ASSERT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
  auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
  ASSERT_EQ(Ops::Base::ToString(*output), "[32, 128]");
}

TEST_F(TestMatMulV3InferShape, BasicX2Trans) {
  auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->infer_shape;

  gert::StorageShape x1_shape = {{32, 64}, {4, 2, 16, 16}};
  gert::StorageShape x2_shape = {{128, 64}, {4, 8, 16, 16}}; // (n, k), (k1, n1, n0, k0)
  gert::StorageShape bias_shape = {{128}, {128}};
  gert::StorageShape output_shape = {{}, {}};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(3, 1)
                    .IrInstanceNum({1, 1, 1})
                    .InputShapes({&x1_shape, &x2_shape, &bias_shape})
                    .OutputShapes({&output_shape})
                    .NodeAttrs({{"transpose_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"transpose_x2", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
                    .Build();

  ASSERT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
  auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
  ASSERT_EQ(Ops::Base::ToString(*output), "[32, 128]");
}

TEST_F(TestMatMulV3InferShape, X2OnlyOneDim) {
  auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->infer_shape;

  gert::Shape x1_shape = {2, 4};
  gert::Shape x2_shape = {4};
  gert::Shape expect_output_shape = {2};

  gert::Shape output_shape = {};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(2, 1)
                    .IrInstanceNum({1, 1})
                    .InputShapes({&x1_shape, &x2_shape})
                    .OutputShapes({&output_shape})
                    .NodeAttrs({{"transpose_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"transpose_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                    .Build();

  ASSERT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
  auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
  ASSERT_EQ(Ops::Base::ToString(*output), Ops::Base::ToString(expect_output_shape));
}

TEST_F(TestMatMulV3InferShape, DynamicShape) {
  auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->infer_shape;

  gert::StorageShape x1_shape = {{-1, 64}, {-1, 64}};
  gert::StorageShape x2_shape = {{64, 128}, {8, 4, 16, 16}};
  gert::StorageShape output_shape = {{}, {}};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(2, 1)
                    .IrInstanceNum({1, 1})
                    .InputShapes({&x1_shape, &x2_shape})
                    .OutputShapes({&output_shape})
                    .NodeAttrs({{"transpose_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"transpose_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                    .Build();

  ASSERT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
  auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
  ASSERT_EQ(Ops::Base::ToString(*output), "[-1, 128]");
}

TEST_F(TestMatMulV3InferShape, DynamicBias) {
  auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->infer_shape;

  gert::StorageShape x1_shape = {{32, 64}, {32, 64}};
  gert::StorageShape x2_shape = {{64, 128}, {8, 4, 16, 16}};
  gert::StorageShape bias_shape = {{1, -1}, {1, -1}};
  gert::StorageShape output_shape = {{}, {}};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(3, 1)
                    .IrInstanceNum({1, 1, 1})
                    .InputShapes({&x1_shape, &x2_shape, &bias_shape})
                    .OutputShapes({&output_shape})
                    .NodeAttrs({{"transpose_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"transpose_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                    .Build();

  ASSERT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
  auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
  ASSERT_EQ(Ops::Base::ToString(*output), "[32, 128]");
}

TEST_F(TestMatMulV3InferShape, UnknownDimNumWithBias01) {
  auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->infer_shape;

  gert::Shape x1_shape = {-2};
  gert::Shape x2_shape = {-2};
  gert::Shape bias_shape = {-2};
  gert::Shape expect_output_shape = {-2};

  gert::Shape output_shape = {};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(3, 1)
                    .IrInstanceNum({1, 1, 1})
                    .InputShapes({&x1_shape, &x2_shape, &bias_shape})
                    .OutputShapes({&output_shape})
                    .NodeAttrs({{"transpose_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"transpose_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                    .Build();

  ASSERT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
  auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
  ASSERT_EQ(Ops::Base::ToString(*output), Ops::Base::ToString(expect_output_shape));
}

TEST_F(TestMatMulV3InferShape, UnknownDimNumWithBias02) {
  auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->infer_shape;

  gert::Shape x1_shape = {-2};
  gert::Shape x2_shape = {-2};
  gert::Shape bias_shape = {1, 128};
  gert::Shape expect_output_shape = {-1, 128};

  gert::Shape output_shape = {};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(3, 1)
                    .IrInstanceNum({1, 1, 1})
                    .InputShapes({&x1_shape, &x2_shape, &bias_shape})
                    .OutputShapes({&output_shape})
                    .NodeAttrs({{"transpose_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"transpose_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                    .Build();

  ASSERT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
  auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
  ASSERT_EQ(Ops::Base::ToString(*output), Ops::Base::ToString(expect_output_shape));
}

TEST_F(TestMatMulV3InferShape, UnknownDimNumWithBias03) {
  auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->infer_shape;

  gert::Shape x1_shape = {-2};
  gert::Shape x2_shape = {64, 128};
  gert::Shape bias_shape = {-2};
  gert::Shape expect_output_shape = {-1, 128};

  gert::Shape output_shape = {};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(3, 1)
                    .IrInstanceNum({1, 1, 1})
                    .InputShapes({&x1_shape, &x2_shape, &bias_shape})
                    .OutputShapes({&output_shape})
                    .NodeAttrs({{"transpose_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"transpose_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                    .Build();

  ASSERT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
  auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
  ASSERT_EQ(Ops::Base::ToString(*output), Ops::Base::ToString(expect_output_shape));
}

TEST_F(TestMatMulV3InferShape, UnknownDimNumWithBias04) {
  auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->infer_shape;

  gert::Shape x1_shape = {-2};
  gert::Shape x2_shape = {64, 128};
  gert::Shape bias_shape = {1, 128};
  gert::Shape expect_output_shape = {-1, 128};

  gert::Shape output_shape = {};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(3, 1)
                    .IrInstanceNum({1, 1, 1})
                    .InputShapes({&x1_shape, &x2_shape, &bias_shape})
                    .OutputShapes({&output_shape})
                    .NodeAttrs({{"transpose_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"transpose_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                    .Build();

  ASSERT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
  auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
  ASSERT_EQ(Ops::Base::ToString(*output), Ops::Base::ToString(expect_output_shape));
}

TEST_F(TestMatMulV3InferShape, UnknownDimNumWithBias05) {
  auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->infer_shape;

  gert::Shape x1_shape = {2, 64};
  gert::Shape x2_shape = {-2};
  gert::Shape bias_shape = {-2};
  gert::Shape expect_output_shape = {2, -1};

  gert::Shape output_shape = {};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(3, 1)
                    .IrInstanceNum({1, 1, 1})
                    .InputShapes({&x1_shape, &x2_shape, &bias_shape})
                    .OutputShapes({&output_shape})
                    .NodeAttrs({{"transpose_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"transpose_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                    .Build();

  ASSERT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
  auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
  ASSERT_EQ(Ops::Base::ToString(*output), Ops::Base::ToString(expect_output_shape));
}

TEST_F(TestMatMulV3InferShape, UnknownDimNumWithBias06) {
  auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->infer_shape;

  gert::Shape x1_shape = {2, 64};
  gert::Shape x2_shape = {64, 128};
  gert::Shape bias_shape = {-2};
  gert::Shape expect_output_shape = {2, 128};

  gert::Shape output_shape = {};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(3, 1)
                    .IrInstanceNum({1, 1, 1})
                    .InputShapes({&x1_shape, &x2_shape, &bias_shape})
                    .OutputShapes({&output_shape})
                    .NodeAttrs({{"transpose_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"transpose_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                    .Build();

  ASSERT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
  auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
  ASSERT_EQ(Ops::Base::ToString(*output), Ops::Base::ToString(expect_output_shape));
}

TEST_F(TestMatMulV3InferShape, UnknownDimNumWithBias07) {
  auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->infer_shape;

  gert::Shape x1_shape = {2, 64};
  gert::Shape x2_shape = {-2};
  gert::Shape bias_shape = {1, 128};
  gert::Shape expect_output_shape = {2, 128};

  gert::Shape output_shape = {};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(3, 1)
                    .IrInstanceNum({1, 1, 1})
                    .InputShapes({&x1_shape, &x2_shape, &bias_shape})
                    .OutputShapes({&output_shape})
                    .NodeAttrs({{"transpose_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"transpose_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                    .Build();

  ASSERT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
  auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
  ASSERT_EQ(Ops::Base::ToString(*output), Ops::Base::ToString(expect_output_shape));
}

TEST_F(TestMatMulV3InferShape, UnknownDimNum01) {
  auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->infer_shape;

  gert::Shape x1_shape = {-2};
  gert::Shape x2_shape = {-2};
  gert::Shape expect_output_shape = {-2};

  gert::Shape output_shape = {};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(2, 1)
                    .IrInstanceNum({1, 1})
                    .InputShapes({&x1_shape, &x2_shape})
                    .OutputShapes({&output_shape})
                    .NodeAttrs({{"transpose_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"transpose_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                    .Build();

  ASSERT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
  auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
  ASSERT_EQ(Ops::Base::ToString(*output), Ops::Base::ToString(expect_output_shape));
}

TEST_F(TestMatMulV3InferShape, UnknownDimNum02) {
  auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->infer_shape;

  gert::Shape x1_shape = {-2};
  gert::Shape x2_shape = {64, 128};
  gert::Shape expect_output_shape = {-1, 128};

  gert::Shape output_shape = {};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(2, 1)
                    .IrInstanceNum({1, 1})
                    .InputShapes({&x1_shape, &x2_shape})
                    .OutputShapes({&output_shape})
                    .NodeAttrs({{"transpose_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"transpose_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                    .Build();

  ASSERT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
  auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
  ASSERT_EQ(Ops::Base::ToString(*output), Ops::Base::ToString(expect_output_shape));
}

TEST_F(TestMatMulV3InferShape, UnknownDimNum03) {
  auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("MatMulV3")->infer_shape;

  gert::Shape x1_shape = {2, 64};
  gert::Shape x2_shape = {-2};
  gert::Shape expect_output_shape = {2, -1};

  gert::Shape output_shape = {};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(2, 1)
                    .IrInstanceNum({1, 1})
                    .InputShapes({&x1_shape, &x2_shape})
                    .OutputShapes({&output_shape})
                    .NodeAttrs({{"transpose_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"transpose_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                    .Build();

  ASSERT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
  auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
  ASSERT_EQ(Ops::Base::ToString(*output), Ops::Base::ToString(expect_output_shape));
}
} // namespace

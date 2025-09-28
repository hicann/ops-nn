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
class TestBatchMatMulV3InferShape : public testing::Test {
};

TEST_F(TestBatchMatMulV3InferShape, Basic) {
  auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("BatchMatMulV3")->infer_shape;

  gert::StorageShape x1_shape = {{4, 8, 16, 32, 64}, {4, 8, 16, 4, 2, 16, 16}};
  gert::StorageShape x2_shape = {{16, 64, 64}, {16, 4, 4, 16, 16}};
  gert::StorageShape bias_shape = {{64}, {64}};
  gert::StorageShape output_shape = {{}, {}};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(3, 1)
                    .IrInstanceNum({1, 1, 1})
                    .InputShapes({&x1_shape, &x2_shape, &bias_shape})
                    .OutputShapes({&output_shape})
                    .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                    .Build();

  ASSERT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
  auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
  ASSERT_EQ(Ops::Base::ToString(*output), "[4, 8, 16, 32, 64]");
}

TEST_F(TestBatchMatMulV3InferShape, BiasZeroTensor) {
  auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("BatchMatMulV3")->infer_shape;

  gert::StorageShape x1_shape = {{4, 8, 16, 32, 64}, {4, 8, 16, 4, 2, 16, 16}};
  gert::StorageShape x2_shape = {{16, 64, 64}, {16, 4, 4, 16, 16}};
  gert::StorageShape bias_shape = {{0}, {0}};
  gert::StorageShape output_shape = {{}, {}};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(3, 1)
                    .IrInstanceNum({1, 1, 1})
                    .InputShapes({&x1_shape, &x2_shape, &bias_shape})
                    .OutputShapes({&output_shape})
                    .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                    .Build();

  ASSERT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
  auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
  ASSERT_EQ(Ops::Base::ToString(*output), "[4, 8, 16, 32, 64]");
}

TEST_F(TestBatchMatMulV3InferShape, UnknownDimCase01) {
  auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("BatchMatMulV3")->infer_shape;

  gert::Shape x1_shape = {4, 3, 5};
  gert::Shape x2_shape = {-2};
  gert::Shape expect_output_shape = {-2};

  gert::Shape output_shape = {};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(2, 1)
                    .IrInstanceNum({1, 1})
                    .InputShapes({&x1_shape, &x2_shape})
                    .OutputShapes({&output_shape})
                    .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                    .Build();

  ASSERT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
  auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
  ASSERT_EQ(Ops::Base::ToString(*output), Ops::Base::ToString(expect_output_shape));
}

TEST_F(TestBatchMatMulV3InferShape, UnknownDimCase02) {
  auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("BatchMatMulV3")->infer_shape;

  gert::Shape x1_shape = {-2};
  gert::Shape x2_shape = {-2};
  gert::Shape expect_output_shape = {-2};

  gert::Shape output_shape = {};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(2, 1)
                    .IrInstanceNum({1, 1})
                    .InputShapes({&x1_shape, &x2_shape})
                    .OutputShapes({&output_shape})
                    .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                    .Build();

  ASSERT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
  auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
  ASSERT_EQ(Ops::Base::ToString(*output), Ops::Base::ToString(expect_output_shape));
}

TEST_F(TestBatchMatMulV3InferShape, UnknownDimCase03) {
  auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("BatchMatMulV3")->infer_shape;

  gert::Shape x1_shape = {-2};
  gert::Shape x2_shape = {5, -1};
  gert::Shape expect_output_shape = {-2};

  gert::Shape output_shape = {};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(2, 1)
                    .IrInstanceNum({1, 1})
                    .InputShapes({&x1_shape, &x2_shape})
                    .OutputShapes({&output_shape})
                    .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                    .Build();

  ASSERT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
  auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
  ASSERT_EQ(Ops::Base::ToString(*output), Ops::Base::ToString(expect_output_shape));
}

TEST_F(TestBatchMatMulV3InferShape, UnknownDimCase04) {
  auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("BatchMatMulV3")->infer_shape;

  gert::Shape x1_shape = {-1, 13, 16};
  gert::Shape x2_shape = {-1, 13, 16};
  gert::Shape expect_output_shape = {-1, 16, 16};

  gert::Shape output_shape = {};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(2, 1)
                    .IrInstanceNum({1, 1})
                    .InputShapes({&x1_shape, &x2_shape})
                    .OutputShapes({&output_shape})
                    .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                    .Build();

  ASSERT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
  auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
  ASSERT_EQ(Ops::Base::ToString(*output), Ops::Base::ToString(expect_output_shape));
}

TEST_F(TestBatchMatMulV3InferShape, UnknownDimCase05) {
  auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("BatchMatMulV3")->infer_shape;

  gert::Shape x1_shape = {1024, 8, -1, -1};
  gert::Shape x2_shape = {1024, 8, -1, 64};
  gert::Shape expect_output_shape = {1024, 8, -1, 64};

  gert::Shape output_shape = {};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(2, 1)
                    .IrInstanceNum({1, 1})
                    .InputShapes({&x1_shape, &x2_shape})
                    .OutputShapes({&output_shape})
                    .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                    .Build();

  ASSERT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
  auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
  ASSERT_EQ(Ops::Base::ToString(*output), Ops::Base::ToString(expect_output_shape));
}

TEST_F(TestBatchMatMulV3InferShape, UnknownDimCase06) {
  auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("BatchMatMulV3")->infer_shape;

  gert::Shape x1_shape = {-1, 13, 16};
  gert::Shape x2_shape = {-1, 13, 16};

  gert::Shape output_shape = {};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(2, 1)
                    .IrInstanceNum({1, 1})
                    .InputShapes({&x1_shape, &x2_shape})
                    .OutputShapes({&output_shape})
                    .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                    .Build();

  ASSERT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(TestBatchMatMulV3InferShape, UnknownDimCase07) {
  auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("BatchMatMulV3")->infer_shape;

  gert::Shape x1_shape = {-1, -1, -1};
  gert::Shape x2_shape = {2048, 64, 300};
  gert::Shape expect_output_shape = {2048, -1, 64};

  gert::Shape output_shape = {};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(2, 1)
                    .IrInstanceNum({1, 1})
                    .InputShapes({&x1_shape, &x2_shape})
                    .OutputShapes({&output_shape})
                    .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
                    .Build();

  ASSERT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
  auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
  ASSERT_EQ(Ops::Base::ToString(*output), Ops::Base::ToString(expect_output_shape));
}

TEST_F(TestBatchMatMulV3InferShape, UnknownDimCase08) {
  auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("BatchMatMulV3")->infer_shape;

  gert::Shape x1_shape = {-1, -1, -1, 512};
  gert::Shape x2_shape = {512, -1};
  gert::Shape expect_output_shape = {-1, -1, -1, -1};

  gert::Shape output_shape = {};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(2, 1)
                    .IrInstanceNum({1, 1})
                    .InputShapes({&x1_shape, &x2_shape})
                    .OutputShapes({&output_shape})
                    .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                    .Build();

  ASSERT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
  auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
  ASSERT_EQ(Ops::Base::ToString(*output), Ops::Base::ToString(expect_output_shape));
}

TEST_F(TestBatchMatMulV3InferShape, UnknownDimCase09) {
  auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("BatchMatMulV3")->infer_shape;

  gert::Shape x1_shape = {-3, 10};
  gert::Shape x2_shape = {10, 20};

  gert::Shape output_shape = {};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(2, 1)
                    .IrInstanceNum({1, 1})
                    .InputShapes({&x1_shape, &x2_shape})
                    .OutputShapes({&output_shape})
                    .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                    .Build();

  ASSERT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(TestBatchMatMulV3InferShape, UnknownDimCase10) {
  auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("BatchMatMulV3")->infer_shape;

  gert::Shape x1_shape = {0, 10}; // 0代表这个维度是0，整个tensor本质是一个空tensor
  gert::Shape x2_shape = {10, 20};
  gert::Shape expect_output_shape = {0, 20};

  gert::Shape output_shape = {};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(2, 1)
                    .IrInstanceNum({1, 1})
                    .InputShapes({&x1_shape, &x2_shape})
                    .OutputShapes({&output_shape})
                    .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                    .Build();

  ASSERT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
  auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
  ASSERT_EQ(Ops::Base::ToString(*output), Ops::Base::ToString(expect_output_shape));
}

TEST_F(TestBatchMatMulV3InferShape, UnknownDimWithBiasCase01) {
  auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("BatchMatMulV3")->infer_shape;

  gert::Shape x1_shape = {-2};
  gert::Shape x2_shape = {4, 3, 5};
  gert::Shape bias_shape = {5};
  gert::Shape expect_output_shape = {-2};

  gert::StorageShape output_shape = {{}, {}};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(3, 1)
                    .IrInstanceNum({1, 1, 1})
                    .InputShapes({&x1_shape, &x2_shape, &bias_shape})
                    .OutputShapes({&output_shape})
                    .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                    .Build();

  ASSERT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
  auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
  ASSERT_EQ(Ops::Base::ToString(*output), Ops::Base::ToString(expect_output_shape));
}

TEST_F(TestBatchMatMulV3InferShape, UnknownDimWithBiasCase02) {
  auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("BatchMatMulV3")->infer_shape;

  gert::Shape x1_shape = {-2};
  gert::Shape x2_shape = {-2};
  gert::Shape bias_shape = {5};
  gert::Shape expect_output_shape = {-2};

  gert::StorageShape output_shape = {{}, {}};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(3, 1)
                    .IrInstanceNum({1, 1, 1})
                    .InputShapes({&x1_shape, &x2_shape, &bias_shape})
                    .OutputShapes({&output_shape})
                    .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                    .Build();

  ASSERT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
  auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
  ASSERT_EQ(Ops::Base::ToString(*output), Ops::Base::ToString(expect_output_shape));
}

TEST_F(TestBatchMatMulV3InferShape, UnknownDimWithBiasCase03) {
  auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("BatchMatMulV3")->infer_shape;

  gert::Shape x1_shape = {-2};
  gert::Shape x2_shape = {-2};
  gert::Shape bias_shape = {-2};
  gert::Shape expect_output_shape = {-2};

  gert::StorageShape output_shape = {{}, {}};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(3, 1)
                    .IrInstanceNum({1, 1, 1})
                    .InputShapes({&x1_shape, &x2_shape, &bias_shape})
                    .OutputShapes({&output_shape})
                    .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                    .Build();

  ASSERT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
  auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
  ASSERT_EQ(Ops::Base::ToString(*output), Ops::Base::ToString(expect_output_shape));
}

TEST_F(TestBatchMatMulV3InferShape, UnknownDimWithBiasCase04) {
  auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("BatchMatMulV3")->infer_shape;

  gert::Shape x1_shape = {-1, 3, -1, -1};
  gert::Shape x2_shape = {-1, -1, -1, -1};
  gert::Shape bias_shape = {5};
  gert::Shape expect_output_shape = {-1, 3, -1, 5};

  gert::StorageShape output_shape = {{}, {}};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(3, 1)
                    .IrInstanceNum({1, 1, 1})
                    .InputShapes({&x1_shape, &x2_shape, &bias_shape})
                    .OutputShapes({&output_shape})
                    .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                    .Build();

  ASSERT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
  auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
  ASSERT_EQ(Ops::Base::ToString(*output), Ops::Base::ToString(expect_output_shape));
}

TEST_F(TestBatchMatMulV3InferShape, UnknownDimWithBiasCase05) {
  auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("BatchMatMulV3")->infer_shape;

  gert::Shape x1_shape = {-1, 3, -1, -1};
  gert::Shape x2_shape = {-1, -1, -1, -1};
  gert::Shape bias_shape = {-1};
  gert::Shape expect_output_shape = {-1, 3, -1, -1};

  gert::StorageShape output_shape = {{}, {}};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(3, 1)
                    .IrInstanceNum({1, 1, 1})
                    .InputShapes({&x1_shape, &x2_shape, &bias_shape})
                    .OutputShapes({&output_shape})
                    .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                    .Build();

  ASSERT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
  auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
  ASSERT_EQ(Ops::Base::ToString(*output), Ops::Base::ToString(expect_output_shape));
}

TEST_F(TestBatchMatMulV3InferShape, UnknownDimWithBiasCase06) {
  auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("BatchMatMulV3")->infer_shape;

  gert::Shape x1_shape = {-1, -1};
  gert::Shape x2_shape = {1024, 2048};
  gert::Shape bias_shape = {-1};
  gert::Shape expect_output_shape = {-1, 2048};

  gert::StorageShape output_shape = {{}, {}};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(3, 1)
                    .IrInstanceNum({1, 1, 1})
                    .InputShapes({&x1_shape, &x2_shape, &bias_shape})
                    .OutputShapes({&output_shape})
                    .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                    .Build();

  ASSERT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
  auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
  ASSERT_EQ(Ops::Base::ToString(*output), Ops::Base::ToString(expect_output_shape));
}

TEST_F(TestBatchMatMulV3InferShape, BatchOneWithDynamic) {
  auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("BatchMatMulV3")->infer_shape;

  gert::Shape x1_shape = {1, 3, 4};
  gert::Shape x2_shape = {-1, 4, 5};
  gert::Shape expect_output_shape = {-1, 3, 5};

  gert::Shape output_shape = {};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(2, 1)
                    .IrInstanceNum({1, 1})
                    .InputShapes({&x1_shape, &x2_shape})
                    .OutputShapes({&output_shape})
                    .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                    .Build();

  ASSERT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
  auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
  ASSERT_EQ(Ops::Base::ToString(*output), Ops::Base::ToString(expect_output_shape));
}

TEST_F(TestBatchMatMulV3InferShape, BiasBatchOneWithDynamic) {
  auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("BatchMatMulV3")->infer_shape;

  gert::Shape x1_shape = {1, 3, 4};
  gert::Shape x2_shape = {4, 5};
  gert::Shape bias_shape = {-1, 1, 5};
  gert::Shape expect_output_shape = {-1, 3, 5};

  gert::Shape output_shape = {};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(3, 1)
                    .IrInstanceNum({1, 1, 1})
                    .InputShapes({&x1_shape, &x2_shape, &bias_shape})
                    .OutputShapes({&output_shape})
                    .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                    .Build();

  ASSERT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
  auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
  ASSERT_EQ(Ops::Base::ToString(*output), Ops::Base::ToString(expect_output_shape));
}

TEST_F(TestBatchMatMulV3InferShape, BiasBatchOneWithDynamic2) {
  auto infer_shape_func = gert::OpImplRegistry::GetInstance().GetOpImpl("BatchMatMulV3")->infer_shape;

  gert::Shape x1_shape = {1, 1, 3, 4};
  gert::Shape x2_shape = {1, 1, 4, 5};
  gert::Shape bias_shape = {-1, 1, 5};
  gert::Shape expect_output_shape = {1, -1, 3, 5};

  gert::Shape output_shape = {};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(3, 1)
                    .IrInstanceNum({1, 1, 1})
                    .InputShapes({&x1_shape, &x2_shape, &bias_shape})
                    .OutputShapes({&output_shape})
                    .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                    .Build();

  ASSERT_EQ(infer_shape_func(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
  auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
  ASSERT_EQ(Ops::Base::ToString(*output), Ops::Base::ToString(expect_output_shape));
}

class TestBatchMatMulV3InferShapeRange : public testing::Test {
};

TEST_F(TestBatchMatMulV3InferShapeRange, Case01) {
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("BatchMatMulV3"), nullptr);
    auto infer_shape_range_func =
        gert::OpImplRegistry::GetInstance().GetOpImpl("BatchMatMulV3")->infer_shape_range;
    gert::Shape x1_shape_min = {4, 1, 1};
    gert::Shape x1_shape_max = {9, -1, -1};
    gert::Shape x2_shape_min = {2, 1, 7};
    gert::Shape x2_shape_max = {5, -1, 7};
    gert::Shape out_shape_min;
    gert::Shape out_shape_max;
    gert::Range<gert::Shape> x1_shape_range(&x1_shape_min, &x1_shape_max);
    gert::Range<gert::Shape> x2_shape_range(&x2_shape_min, &x2_shape_max);
    gert::Range<gert::Shape> out_shape_range(&out_shape_min, &out_shape_max);

    auto holder = gert::InferShapeRangeContextFaker()
        .NodeIoNum(2, 1)
        .IrInputNum(2)
        .InputShapeRanges({&x1_shape_range, &x2_shape_range})
        .OutputShapeRanges({&out_shape_range})
        .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                    {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
        .Build();

    EXPECT_EQ(infer_shape_range_func(holder.GetContext<gert::InferShapeRangeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape target_shape_min = {4, 1, 7};
    gert::Shape target_shape_max = {5, -1, 7};
    gert::Range<gert::Shape> target_out_shape_range(&target_shape_min, &target_shape_max);
    EXPECT_EQ(out_shape_range, target_out_shape_range);
}

TEST_F(TestBatchMatMulV3InferShapeRange, Case02) {
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("BatchMatMulV3"), nullptr);
    auto infer_shape_range_func =
        gert::OpImplRegistry::GetInstance().GetOpImpl("BatchMatMulV3")->infer_shape_range;
    gert::Shape x1_shape_min = {1, 1, 1};
    gert::Shape x1_shape_max = {8, -1, -1};
    gert::Shape x2_shape_min = {1, 1, 7};
    gert::Shape x2_shape_max = {6, -1, 7};
    gert::Shape out_shape_min;
    gert::Shape out_shape_max;
    gert::Range<gert::Shape> x1_shape_range(&x1_shape_min, &x1_shape_max);
    gert::Range<gert::Shape> x2_shape_range(&x2_shape_min, &x2_shape_max);
    gert::Range<gert::Shape> out_shape_range(&out_shape_min, &out_shape_max);

    auto holder = gert::InferShapeRangeContextFaker()
        .NodeIoNum(2, 1)
        .IrInputNum(2)
        .InputShapeRanges({&x1_shape_range, &x2_shape_range})
        .OutputShapeRanges({&out_shape_range})
        .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                    {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
        .Build();

    EXPECT_EQ(infer_shape_range_func(holder.GetContext<gert::InferShapeRangeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape target_shape_min = {1, 1, 7};
    gert::Shape target_shape_max = {8, -1, 7};
    gert::Range<gert::Shape> target_out_shape_range(&target_shape_min, &target_shape_max);
    EXPECT_EQ(out_shape_range, target_out_shape_range);
}

TEST_F(TestBatchMatMulV3InferShapeRange, Case03) {
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("BatchMatMulV3"), nullptr);
    auto infer_shape_range_func =
        gert::OpImplRegistry::GetInstance().GetOpImpl("BatchMatMulV3")->infer_shape_range;
    gert::Shape x1_shape_min = {1, 1, 1};
    gert::Shape x1_shape_max = {1, -1, -1};
    gert::Shape x2_shape_min = {2, 1, 7};
    gert::Shape x2_shape_max = {5, -1, 7};
    gert::Shape out_shape_min;
    gert::Shape out_shape_max;
    gert::Range<gert::Shape> x1_shape_range(&x1_shape_min, &x1_shape_max);
    gert::Range<gert::Shape> x2_shape_range(&x2_shape_min, &x2_shape_max);
    gert::Range<gert::Shape> out_shape_range(&out_shape_min, &out_shape_max);

    auto holder = gert::InferShapeRangeContextFaker()
        .NodeIoNum(2, 1)
        .IrInputNum(2)
        .InputShapeRanges({&x1_shape_range, &x2_shape_range})
        .OutputShapeRanges({&out_shape_range})
        .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                    {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
        .Build();

    EXPECT_EQ(infer_shape_range_func(holder.GetContext<gert::InferShapeRangeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape target_shape_min = {2, 1, 7};
    gert::Shape target_shape_max = {5, -1, 7};
    gert::Range<gert::Shape> target_out_shape_range(&target_shape_min, &target_shape_max);
    EXPECT_EQ(out_shape_range, target_out_shape_range);
}

TEST_F(TestBatchMatMulV3InferShapeRange, Case04) {
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("BatchMatMulV3"), nullptr);
    auto infer_shape_range_func =
        gert::OpImplRegistry::GetInstance().GetOpImpl("BatchMatMulV3")->infer_shape_range;
    gert::Shape x1_shape_min = {2, 1, 1};
    gert::Shape x1_shape_max = {5, -1, -1};
    gert::Shape x2_shape_min = {1, 1, 7};
    gert::Shape x2_shape_max = {1, -1, 7};
    gert::Shape out_shape_min;
    gert::Shape out_shape_max;
    gert::Range<gert::Shape> x1_shape_range(&x1_shape_min, &x1_shape_max);
    gert::Range<gert::Shape> x2_shape_range(&x2_shape_min, &x2_shape_max);
    gert::Range<gert::Shape> out_shape_range(&out_shape_min, &out_shape_max);

    auto holder = gert::InferShapeRangeContextFaker()
        .NodeIoNum(2, 1)
        .IrInputNum(2)
        .InputShapeRanges({&x1_shape_range, &x2_shape_range})
        .OutputShapeRanges({&out_shape_range})
        .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                    {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
        .Build();

    EXPECT_EQ(infer_shape_range_func(holder.GetContext<gert::InferShapeRangeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape target_shape_min = {2, 1, 7};
    gert::Shape target_shape_max = {5, -1, 7};
    gert::Range<gert::Shape> target_out_shape_range(&target_shape_min, &target_shape_max);
    EXPECT_EQ(out_shape_range, target_out_shape_range);
}

TEST_F(TestBatchMatMulV3InferShapeRange, Case05) {
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("BatchMatMulV3"), nullptr);
    auto infer_shape_range_func =
        gert::OpImplRegistry::GetInstance().GetOpImpl("BatchMatMulV3")->infer_shape_range;
    gert::Shape x1_shape_min = {8, 13, 16};
    gert::Shape x1_shape_max = {-1, 13, 16};
    gert::Shape x2_shape_min = {8, 13, 16};
    gert::Shape x2_shape_max = {-1, 13, 16};
    gert::Shape out_shape_min;
    gert::Shape out_shape_max;
    gert::Range<gert::Shape> x1_shape_range(&x1_shape_min, &x1_shape_max);
    gert::Range<gert::Shape> x2_shape_range(&x2_shape_min, &x2_shape_max);
    gert::Range<gert::Shape> out_shape_range(&out_shape_min, &out_shape_max);

    auto holder = gert::InferShapeRangeContextFaker()
        .NodeIoNum(2, 1)
        .IrInputNum(2)
        .InputShapeRanges({&x1_shape_range, &x2_shape_range})
        .OutputShapeRanges({&out_shape_range})
        .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                    {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
        .Build();

    EXPECT_EQ(infer_shape_range_func(holder.GetContext<gert::InferShapeRangeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape target_shape_min = {8, 16, 16};
    gert::Shape target_shape_max = {-1, 16, 16};
    gert::Range<gert::Shape> target_out_shape_range(&target_shape_min, &target_shape_max);
    EXPECT_EQ(out_shape_range, target_out_shape_range);
}

TEST_F(TestBatchMatMulV3InferShapeRange, Case06) {
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("BatchMatMulV3"), nullptr);
    auto infer_shape_range_func =
        gert::OpImplRegistry::GetInstance().GetOpImpl("BatchMatMulV3")->infer_shape_range;
    gert::Shape x1_shape_min = {-1, 13, 16};
    gert::Shape x1_shape_max = {-1, 13, 16};
    gert::Shape x2_shape_min = {-1, 13, 16};
    gert::Shape x2_shape_max = {-1, 13, 16};
    gert::Shape out_shape_min;
    gert::Shape out_shape_max;
    gert::Range<gert::Shape> x1_shape_range(&x1_shape_min, &x1_shape_max);
    gert::Range<gert::Shape> x2_shape_range(&x2_shape_min, &x2_shape_max);
    gert::Range<gert::Shape> out_shape_range(&out_shape_min, &out_shape_max);

    auto holder = gert::InferShapeRangeContextFaker()
        .NodeIoNum(2, 1)
        .IrInputNum(2)
        .InputShapeRanges({&x1_shape_range, &x2_shape_range})
        .OutputShapeRanges({&out_shape_range})
        .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                    {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
        .Build();

    EXPECT_EQ(infer_shape_range_func(holder.GetContext<gert::InferShapeRangeContext>()), ge::GRAPH_FAILED);
}

TEST_F(TestBatchMatMulV3InferShapeRange, Case07) {
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("BatchMatMulV3"), nullptr);
    auto infer_shape_range_func =
        gert::OpImplRegistry::GetInstance().GetOpImpl("BatchMatMulV3")->infer_shape_range;
    gert::Shape x1_shape_min = {0, 0, 0};
    gert::Shape x1_shape_max = {-1, -1, -1};
    gert::Shape x2_shape_min = {2048, 64, 300};
    gert::Shape x2_shape_max = {2048, 64, 300};
    gert::Shape out_shape_min;
    gert::Shape out_shape_max;
    gert::Range<gert::Shape> x1_shape_range(&x1_shape_min, &x1_shape_max);
    gert::Range<gert::Shape> x2_shape_range(&x2_shape_min, &x2_shape_max);
    gert::Range<gert::Shape> out_shape_range(&out_shape_min, &out_shape_max);

    auto holder = gert::InferShapeRangeContextFaker()
        .NodeIoNum(2, 1)
        .IrInputNum(2)
        .InputShapeRanges({&x1_shape_range, &x2_shape_range})
        .OutputShapeRanges({&out_shape_range})
        .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                    {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
        .Build();

    EXPECT_EQ(infer_shape_range_func(holder.GetContext<gert::InferShapeRangeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape target_shape_min = {2048, 0, 64};
    gert::Shape target_shape_max = {2048, -1, 64};
    gert::Range<gert::Shape> target_out_shape_range(&target_shape_min, &target_shape_max);
    EXPECT_EQ(out_shape_range, target_out_shape_range);
}

TEST_F(TestBatchMatMulV3InferShapeRange, Case08) {
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("BatchMatMulV3"), nullptr);
    auto infer_shape_range_func =
        gert::OpImplRegistry::GetInstance().GetOpImpl("BatchMatMulV3")->infer_shape_range;
    gert::Shape x1_shape_min = {1, 2, 1, 512};
    gert::Shape x1_shape_max = {-1, 3, 2, 512};
    gert::Shape x2_shape_min = {512, 3};
    gert::Shape x2_shape_max = {512, 4};
    gert::Shape out_shape_min;
    gert::Shape out_shape_max;
    gert::Range<gert::Shape> x1_shape_range(&x1_shape_min, &x1_shape_max);
    gert::Range<gert::Shape> x2_shape_range(&x2_shape_min, &x2_shape_max);
    gert::Range<gert::Shape> out_shape_range(&out_shape_min, &out_shape_max);

    auto holder = gert::InferShapeRangeContextFaker()
        .NodeIoNum(2, 1)
        .IrInputNum(2)
        .InputShapeRanges({&x1_shape_range, &x2_shape_range})
        .OutputShapeRanges({&out_shape_range})
        .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                    {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
        .Build();

    EXPECT_EQ(infer_shape_range_func(holder.GetContext<gert::InferShapeRangeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape target_shape_min = {1, 2, 1, 3};
    gert::Shape target_shape_max = {-1, 3, 2, 4};
    gert::Range<gert::Shape> target_out_shape_range(&target_shape_min, &target_shape_max);
    EXPECT_EQ(out_shape_range, target_out_shape_range);
}

TEST_F(TestBatchMatMulV3InferShapeRange, Case09) {
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("BatchMatMulV3"), nullptr);
    auto infer_shape_range_func =
        gert::OpImplRegistry::GetInstance().GetOpImpl("BatchMatMulV3")->infer_shape_range;
    gert::Shape x1_shape_min = {1024, 8, 1, 1};
    gert::Shape x1_shape_max = {1024, 8, 536870912, -1};
    gert::Shape x2_shape_min = {1024, 8, 2, 64};
    gert::Shape x2_shape_max = {1024, 8, -1, 64};
    gert::Shape out_shape_min;
    gert::Shape out_shape_max;
    gert::Range<gert::Shape> x1_shape_range(&x1_shape_min, &x1_shape_max);
    gert::Range<gert::Shape> x2_shape_range(&x2_shape_min, &x2_shape_max);
    gert::Range<gert::Shape> out_shape_range(&out_shape_min, &out_shape_max);

    auto holder = gert::InferShapeRangeContextFaker()
        .NodeIoNum(2, 1)
        .IrInputNum(2)
        .InputShapeRanges({&x1_shape_range, &x2_shape_range})
        .OutputShapeRanges({&out_shape_range})
        .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                    {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
        .Build();

    EXPECT_EQ(infer_shape_range_func(holder.GetContext<gert::InferShapeRangeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape target_shape_min = {1024, 8, 1, 64};
    gert::Shape target_shape_max = {1024, 8, 536870912, 64};
    gert::Range<gert::Shape> target_out_shape_range(&target_shape_min, &target_shape_max);
    EXPECT_EQ(out_shape_range, target_out_shape_range);
}

TEST_F(TestBatchMatMulV3InferShapeRange, ChatGLM2) {
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("BatchMatMulV3"), nullptr);
    auto infer_shape_range_func =
        gert::OpImplRegistry::GetInstance().GetOpImpl("BatchMatMulV3")->infer_shape_range;
    gert::Shape x1_shape_min = {0, 0};
    gert::Shape x1_shape_max = {-1, -1};
    gert::Shape x2_shape_min = {4096, 4096};
    gert::Shape x2_shape_max = {4096, 4096};
    gert::Shape out_shape_min;
    gert::Shape out_shape_max;
    gert::Range<gert::Shape> x1_shape_range(&x1_shape_min, &x1_shape_max);
    gert::Range<gert::Shape> x2_shape_range(&x2_shape_min, &x2_shape_max);
    gert::Range<gert::Shape> out_shape_range(&out_shape_min, &out_shape_max);

    auto holder = gert::InferShapeRangeContextFaker()
        .NodeIoNum(2, 1)
        .IrInputNum(2)
        .InputShapeRanges({&x1_shape_range, &x2_shape_range})
        .OutputShapeRanges({&out_shape_range})
        .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                    {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
        .Build();

    EXPECT_EQ(infer_shape_range_func(holder.GetContext<gert::InferShapeRangeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape target_shape_min = {0, 4096};
    gert::Shape target_shape_max = {-1, 4096};
    gert::Range<gert::Shape> target_out_shape_range(&target_shape_min, &target_shape_max);
    EXPECT_EQ(out_shape_range, target_out_shape_range);
}

TEST_F(TestBatchMatMulV3InferShapeRange, BiasCase01) {
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("BatchMatMulV3"), nullptr);
    auto infer_shape_range_func =
        gert::OpImplRegistry::GetInstance().GetOpImpl("BatchMatMulV3")->infer_shape_range;
    gert::Shape x1_shape_min = {4, 7, 1};
    gert::Shape x1_shape_max = {9, 7, -1};
    gert::Shape x2_shape_min = {1, 7};
    gert::Shape x2_shape_max = {-1, 9};
    gert::Shape bias_shape_min = {8};
    gert::Shape bias_shape_max = {8};
    gert::Shape out_shape_min;
    gert::Shape out_shape_max;
    gert::Range<gert::Shape> x1_shape_range(&x1_shape_min, &x1_shape_max);
    gert::Range<gert::Shape> x2_shape_range(&x2_shape_min, &x2_shape_max);
    gert::Range<gert::Shape> bias_shape_range(&bias_shape_min, &bias_shape_max);
    gert::Range<gert::Shape> out_shape_range(&out_shape_min, &out_shape_max);

    auto holder = gert::InferShapeRangeContextFaker()
        .NodeIoNum(3, 1)
        .IrInputNum(3)
        .InputShapeRanges({&x1_shape_range, &x2_shape_range, &bias_shape_range})
        .OutputShapeRanges({&out_shape_range})
        .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                    {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
        .Build();

    EXPECT_EQ(infer_shape_range_func(holder.GetContext<gert::InferShapeRangeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape target_shape_min = {4, 7, 8};
    gert::Shape target_shape_max = {9, 7, 8};
    gert::Range<gert::Shape> target_out_shape_range(&target_shape_min, &target_shape_max);
    EXPECT_EQ(out_shape_range, target_out_shape_range);
}

TEST_F(TestBatchMatMulV3InferShapeRange, BiasCase02) {
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("BatchMatMulV3"), nullptr);
    auto infer_shape_range_func =
        gert::OpImplRegistry::GetInstance().GetOpImpl("BatchMatMulV3")->infer_shape_range;
    gert::Shape x1_shape_min = {1, 3, 1, 1};
    gert::Shape x1_shape_max = {5, 3, 5, 5};
    gert::Shape x2_shape_min = {2, 2, 2, 2};
    gert::Shape x2_shape_max = {80, 80, 80, 80};
    gert::Shape bias_shape_min = {5};
    gert::Shape bias_shape_max = {5};
    gert::Shape out_shape_min;
    gert::Shape out_shape_max;
    gert::Range<gert::Shape> x1_shape_range(&x1_shape_min, &x1_shape_max);
    gert::Range<gert::Shape> x2_shape_range(&x2_shape_min, &x2_shape_max);
    gert::Range<gert::Shape> bias_shape_range(&bias_shape_min, &bias_shape_max);
    gert::Range<gert::Shape> out_shape_range(&out_shape_min, &out_shape_max);

    auto holder = gert::InferShapeRangeContextFaker()
        .NodeIoNum(3, 1)
        .IrInputNum(3)
        .InputShapeRanges({&x1_shape_range, &x2_shape_range, &bias_shape_range})
        .OutputShapeRanges({&out_shape_range})
        .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                    {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
        .Build();

    EXPECT_EQ(infer_shape_range_func(holder.GetContext<gert::InferShapeRangeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape target_shape_min = {2, 3, 1, 5};
    gert::Shape target_shape_max = {80, 3, 5, 5};
    gert::Range<gert::Shape> target_out_shape_range(&target_shape_min, &target_shape_max);
    EXPECT_EQ(out_shape_range, target_out_shape_range);
}

TEST_F(TestBatchMatMulV3InferShapeRange, BiasWithBatch) {
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("BatchMatMulV3"), nullptr);
    auto infer_shape_range_func =
        gert::OpImplRegistry::GetInstance().GetOpImpl("BatchMatMulV3")->infer_shape_range;
    gert::Shape x1_shape_min = {3, 4};
    gert::Shape x1_shape_max = {3, 4};
    gert::Shape x2_shape_min = {4, 5};
    gert::Shape x2_shape_max = {4, 5};
    gert::Shape bias_shape_min = {2, 3, 1, 5};
    gert::Shape bias_shape_max = {2, 3, 1, 5};
    gert::Shape out_shape_min;
    gert::Shape out_shape_max;
    gert::Range<gert::Shape> x1_shape_range(&x1_shape_min, &x1_shape_max);
    gert::Range<gert::Shape> x2_shape_range(&x2_shape_min, &x2_shape_max);
    gert::Range<gert::Shape> bias_shape_range(&bias_shape_min, &bias_shape_max);
    gert::Range<gert::Shape> out_shape_range(&out_shape_min, &out_shape_max);

    auto holder = gert::InferShapeRangeContextFaker()
        .NodeIoNum(3, 1)
        .IrInputNum(3)
        .InputShapeRanges({&x1_shape_range, &x2_shape_range, &bias_shape_range})
        .OutputShapeRanges({&out_shape_range})
        .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                    {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
        .Build();

    EXPECT_EQ(infer_shape_range_func(holder.GetContext<gert::InferShapeRangeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape target_shape_min = {2, 3, 3, 5};
    gert::Shape target_shape_max = {2, 3, 3, 5};
    gert::Range<gert::Shape> target_out_shape_range(&target_shape_min, &target_shape_max);
    EXPECT_EQ(out_shape_range, target_out_shape_range);
}

TEST_F(TestBatchMatMulV3InferShapeRange, X1OneDim) {
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("BatchMatMulV3"), nullptr);
    auto infer_shape_range_func =
        gert::OpImplRegistry::GetInstance().GetOpImpl("BatchMatMulV3")->infer_shape_range;
    gert::Shape x1_shape_min = {4};
    gert::Shape x1_shape_max = {4};
    gert::Shape x2_shape_min = {4, 5};
    gert::Shape x2_shape_max = {4, 5};
    gert::Shape bias_shape_min = {5};
    gert::Shape bias_shape_max = {5};
    gert::Shape out_shape_min;
    gert::Shape out_shape_max;
    gert::Range<gert::Shape> x1_shape_range(&x1_shape_min, &x1_shape_max);
    gert::Range<gert::Shape> x2_shape_range(&x2_shape_min, &x2_shape_max);
    gert::Range<gert::Shape> bias_shape_range(&bias_shape_min, &bias_shape_max);
    gert::Range<gert::Shape> out_shape_range(&out_shape_min, &out_shape_max);

    auto holder = gert::InferShapeRangeContextFaker()
        .NodeIoNum(3, 1)
        .IrInputNum(3)
        .InputShapeRanges({&x1_shape_range, &x2_shape_range, &bias_shape_range})
        .OutputShapeRanges({&out_shape_range})
        .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                    {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
        .Build();

    EXPECT_EQ(infer_shape_range_func(holder.GetContext<gert::InferShapeRangeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape target_shape_min = {5};
    gert::Shape target_shape_max = {5};
    gert::Range<gert::Shape> target_out_shape_range(&target_shape_min, &target_shape_max);
    EXPECT_EQ(out_shape_range, target_out_shape_range);
}

TEST_F(TestBatchMatMulV3InferShapeRange, X2OneDim) {
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("BatchMatMulV3"), nullptr);
    auto infer_shape_range_func =
        gert::OpImplRegistry::GetInstance().GetOpImpl("BatchMatMulV3")->infer_shape_range;
    gert::Shape x1_shape_min = {3, 4};
    gert::Shape x1_shape_max = {3, 4};
    gert::Shape x2_shape_min = {4};
    gert::Shape x2_shape_max = {4};
    gert::Shape out_shape_min;
    gert::Shape out_shape_max;
    gert::Range<gert::Shape> x1_shape_range(&x1_shape_min, &x1_shape_max);
    gert::Range<gert::Shape> x2_shape_range(&x2_shape_min, &x2_shape_max);
    gert::Range<gert::Shape> out_shape_range(&out_shape_min, &out_shape_max);

    auto holder = gert::InferShapeRangeContextFaker()
        .NodeIoNum(2, 1)
        .IrInputNum(2)
        .InputShapeRanges({&x1_shape_range, &x2_shape_range})
        .OutputShapeRanges({&out_shape_range})
        .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                    {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
        .Build();

    EXPECT_EQ(infer_shape_range_func(holder.GetContext<gert::InferShapeRangeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape target_shape_min = {3};
    gert::Shape target_shape_max = {3};
    gert::Range<gert::Shape> target_out_shape_range(&target_shape_min, &target_shape_max);
    EXPECT_EQ(out_shape_range, target_out_shape_range);
}

TEST_F(TestBatchMatMulV3InferShapeRange, BatchOneWithDynamic) {
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("BatchMatMulV3"), nullptr);
    auto infer_shape_range_func =
        gert::OpImplRegistry::GetInstance().GetOpImpl("BatchMatMulV3")->infer_shape_range;
    gert::Shape x1_shape_min = {1, 3, 4};
    gert::Shape x1_shape_max = {1, 3, 4};
    gert::Shape x2_shape_min = {0, 4, 5};
    gert::Shape x2_shape_max = {-1, 4, 5};
    gert::Shape out_shape_min;
    gert::Shape out_shape_max;
    gert::Range<gert::Shape> x1_shape_range(&x1_shape_min, &x1_shape_max);
    gert::Range<gert::Shape> x2_shape_range(&x2_shape_min, &x2_shape_max);
    gert::Range<gert::Shape> out_shape_range(&out_shape_min, &out_shape_max);

    auto holder = gert::InferShapeRangeContextFaker()
        .NodeIoNum(2, 1)
        .IrInputNum(2)
        .InputShapeRanges({&x1_shape_range, &x2_shape_range})
        .OutputShapeRanges({&out_shape_range})
        .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                    {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
        .Build();

    EXPECT_EQ(infer_shape_range_func(holder.GetContext<gert::InferShapeRangeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape target_shape_min = {0, 3, 5};
    gert::Shape target_shape_max = {-1, 3, 5};
    gert::Range<gert::Shape> target_out_shape_range(&target_shape_min, &target_shape_max);
    EXPECT_EQ(out_shape_range, target_out_shape_range);
}

TEST_F(TestBatchMatMulV3InferShapeRange, BiasBatchOneWithDynamic) {
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("BatchMatMulV3"), nullptr);
    auto infer_shape_range_func =
        gert::OpImplRegistry::GetInstance().GetOpImpl("BatchMatMulV3")->infer_shape_range;
    gert::Shape x1_shape_min = {1, 3, 4};
    gert::Shape x1_shape_max = {1, 3, 4};
    gert::Shape x2_shape_min = {4, 5};
    gert::Shape x2_shape_max = {4, 5};
    gert::Shape bias_shape_min = {0, 1, 5};
    gert::Shape bias_shape_max = {-1, 1, 5};
    gert::Shape out_shape_min;
    gert::Shape out_shape_max;
    gert::Range<gert::Shape> x1_shape_range(&x1_shape_min, &x1_shape_max);
    gert::Range<gert::Shape> x2_shape_range(&x2_shape_min, &x2_shape_max);
    gert::Range<gert::Shape> bias_shape_range(&bias_shape_min, &bias_shape_max);
    gert::Range<gert::Shape> out_shape_range(&out_shape_min, &out_shape_max);

    auto holder = gert::InferShapeRangeContextFaker()
        .NodeIoNum(3, 1)
        .IrInputNum(3)
        .InputShapeRanges({&x1_shape_range, &x2_shape_range, &bias_shape_range})
        .OutputShapeRanges({&out_shape_range})
        .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                    {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
        .Build();

    EXPECT_EQ(infer_shape_range_func(holder.GetContext<gert::InferShapeRangeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape target_shape_min = {0, 3, 5};
    gert::Shape target_shape_max = {-1, 3, 5};
    gert::Range<gert::Shape> target_out_shape_range(&target_shape_min, &target_shape_max);
    EXPECT_EQ(out_shape_range, target_out_shape_range);
}

TEST_F(TestBatchMatMulV3InferShapeRange, BiasBatchOneWithDynamic2) {
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("BatchMatMulV3"), nullptr);
    auto infer_shape_range_func =
        gert::OpImplRegistry::GetInstance().GetOpImpl("BatchMatMulV3")->infer_shape_range;
    gert::Shape x1_shape_min = {1, 1, 3, 4};
    gert::Shape x1_shape_max = {1, 1, 3, 4};
    gert::Shape x2_shape_min = {1, 1, 4, 5};
    gert::Shape x2_shape_max = {1, 1, 4, 5};
    gert::Shape bias_shape_min = {0, 1, 5};
    gert::Shape bias_shape_max = {-1, 1, 5};
    gert::Shape out_shape_min;
    gert::Shape out_shape_max;
    gert::Range<gert::Shape> x1_shape_range(&x1_shape_min, &x1_shape_max);
    gert::Range<gert::Shape> x2_shape_range(&x2_shape_min, &x2_shape_max);
    gert::Range<gert::Shape> bias_shape_range(&bias_shape_min, &bias_shape_max);
    gert::Range<gert::Shape> out_shape_range(&out_shape_min, &out_shape_max);

    auto holder = gert::InferShapeRangeContextFaker()
        .NodeIoNum(3, 1)
        .IrInputNum(3)
        .InputShapeRanges({&x1_shape_range, &x2_shape_range, &bias_shape_range})
        .OutputShapeRanges({&out_shape_range})
        .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                    {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
        .Build();

    EXPECT_EQ(infer_shape_range_func(holder.GetContext<gert::InferShapeRangeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape target_shape_min = {1, 0, 3, 5};
    gert::Shape target_shape_max = {1, -1, 3, 5};
    gert::Range<gert::Shape> target_out_shape_range(&target_shape_min, &target_shape_max);
    EXPECT_EQ(out_shape_range, target_out_shape_range);
}

TEST_F(TestBatchMatMulV3InferShapeRange, Gpt2200BBiasDyn) {
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("BatchMatMulV3"), nullptr);
    auto infer_shape_range_func =
        gert::OpImplRegistry::GetInstance().GetOpImpl("BatchMatMulV3")->infer_shape_range;
    gert::Shape x1_shape_min = {0, 0};
    gert::Shape x1_shape_max = {-1, -1};
    gert::Shape x2_shape_min = {1664, 1664};
    gert::Shape x2_shape_max = {1664, 1664};
    gert::Shape bias_shape_min = {0};
    gert::Shape bias_shape_max = {-1};
    gert::Shape out_shape_min;
    gert::Shape out_shape_max;
    gert::Range<gert::Shape> x1_shape_range(&x1_shape_min, &x1_shape_max);
    gert::Range<gert::Shape> x2_shape_range(&x2_shape_min, &x2_shape_max);
    gert::Range<gert::Shape> bias_shape_range(&bias_shape_min, &bias_shape_max);
    gert::Range<gert::Shape> out_shape_range(&out_shape_min, &out_shape_max);

    auto holder = gert::InferShapeRangeContextFaker()
        .NodeIoNum(3, 1)
        .IrInputNum(3)
        .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
        .InputShapeRanges({&x1_shape_range, &x2_shape_range, &bias_shape_range})
        .OutputShapeRanges({&out_shape_range})
        .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                    {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
        .Build();

    EXPECT_EQ(infer_shape_range_func(holder.GetContext<gert::InferShapeRangeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape target_shape_min = {0, 1664};
    gert::Shape target_shape_max = {-1, 1664};
    gert::Range<gert::Shape> target_out_shape_range(&target_shape_min, &target_shape_max);
    EXPECT_EQ(out_shape_range, target_out_shape_range);
}

TEST_F(TestBatchMatMulV3InferShapeRange, Gpt2200BBiasLen0) {
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl("BatchMatMulV3"), nullptr);
    auto infer_shape_range_func =
        gert::OpImplRegistry::GetInstance().GetOpImpl("BatchMatMulV3")->infer_shape_range;
    gert::Shape x1_shape_min = {0, 0};
    gert::Shape x1_shape_max = {-1, -1};
    gert::Shape x2_shape_min = {1664, 1664};
    gert::Shape x2_shape_max = {1664, 1664};
    gert::Shape bias_shape_min = {};
    gert::Shape bias_shape_max = {};
    gert::Shape out_shape_min;
    gert::Shape out_shape_max;
    gert::Range<gert::Shape> x1_shape_range(&x1_shape_min, &x1_shape_max);
    gert::Range<gert::Shape> x2_shape_range(&x2_shape_min, &x2_shape_max);
    gert::Range<gert::Shape> bias_shape_range(&bias_shape_min, &bias_shape_max);
    gert::Range<gert::Shape> out_shape_range(&out_shape_min, &out_shape_max);

    auto holder = gert::InferShapeRangeContextFaker()
        .NodeIoNum(3, 1)
        .IrInputNum(3)
        .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
        .InputShapeRanges({&x1_shape_range, &x2_shape_range, &bias_shape_range})
        .OutputShapeRanges({&out_shape_range})
        .NodeAttrs({{"adj_x1", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                    {"adj_x2", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
        .Build();

    EXPECT_EQ(infer_shape_range_func(holder.GetContext<gert::InferShapeRangeContext>()), ge::GRAPH_SUCCESS);
    gert::Shape target_shape_min = {0, 1664};
    gert::Shape target_shape_max = {-1, 1664};
    gert::Range<gert::Shape> target_out_shape_range(&target_shape_min, &target_shape_max);
    EXPECT_EQ(out_shape_range, target_out_shape_range);
}

} // namespace

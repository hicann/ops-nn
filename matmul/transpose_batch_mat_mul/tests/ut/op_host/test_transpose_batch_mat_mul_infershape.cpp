/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include <gtest/gtest.h>
#include "kernel_run_context_facker.h"
#include "../../../op_graph/transpose_batch_mat_mul_proto.h"
#include "register/op_impl_registry.h"
#include "log/log.h"

namespace {
class TransposeBatchMatMulInferShape : public testing::Test {
};

//   pass cases
TEST_F(TransposeBatchMatMulInferShape, Basic) {
  auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("TransposeBatchMatMul")->infer_shape;

  gert::Shape x1_shape = {32, 16, 512};
  gert::Shape x2_shape = {16, 512, 128};
  gert::Shape scale_shape = {2048};
  gert::Shape expect_output_shape = {32, 16, 128};

  gert::Shape output_shape = {};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(4, 1)
                    .IrInstanceNum({1, 1, 1, 1})
                    .InputShapes({&x1_shape, &x2_shape, nullptr, nullptr})
                    .OutputShapes({&output_shape})
                    .NodeAttrs({{"perm_x1", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({1, 0, 2})},
                                {"perm_x2", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({0, 1, 2})},
                                {"perm_y", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({1, 0, 2})},
                                {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                {"batch_split_factor", Ops::NN::AnyValue::CreateFrom<int64_t>(1)}})
                    .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .Build();

  ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
  auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
  ASSERT_EQ(Ops::Base::ToString(*output), Ops::Base::ToString(expect_output_shape));
}

TEST_F(TransposeBatchMatMulInferShape, Basic_Scale) {
  auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("TransposeBatchMatMul")->infer_shape;

  gert::Shape x1_shape = {32, 16, 512};
  gert::Shape x2_shape = {16, 512, 128};
  gert::Shape expect_output_shape = {2, 32, 1024};

  gert::Shape output_shape = {};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(4, 1)
                    .IrInstanceNum({1, 1, 1, 1})
                    .InputShapes({&x1_shape, &x2_shape, nullptr, nullptr})
                    .OutputShapes({&output_shape})
                    .NodeAttrs({{"perm_x1", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({1, 0, 2})},
                                {"perm_x2", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({0, 1, 2})},
                                {"perm_y", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({1, 0, 2})},
                                {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                {"batch_split_factor", Ops::NN::AnyValue::CreateFrom<int64_t>(2)}})
                    .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                    .Build();

  ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
  auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
  ASSERT_EQ(Ops::Base::ToString(*output), Ops::Base::ToString(expect_output_shape));
}

//  fail cases
TEST_F(TransposeBatchMatMulInferShape, InvalidX2Case01) {
  auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("TransposeBatchMatMul")->infer_shape;

  gert::StorageShape x1_shape = {{3, 6, 4}, {6, 12, 8}};
  gert::StorageShape x2_shape = {{6, 4, 5}, {12, 8, 10}};

  gert::StorageShape output_shape = {{}, {}};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(4, 1)
                    .IrInstanceNum({1, 1, 1, 1})
                    .InputShapes({&x1_shape, &x2_shape, nullptr, nullptr})
                    .OutputShapes({&output_shape})
                    .NodeAttrs({{"perm_x1", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({1, 0, 2})},
                                {"perm_x2", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({0, 1, 2})},
                                {"perm_y", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({1, 0, 2})},
                                {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                {"batch_split_factor", Ops::NN::AnyValue::CreateFrom<int64_t>(1)}})
                    .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .Build();

  ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(TransposeBatchMatMulInferShape, InvalidX2_Scale) {
  auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("TransposeBatchMatMul")->infer_shape;

  gert::StorageShape x1_shape = {{3, 6, 4}, {6, 12, 8}};
  gert::StorageShape x2_shape = {{6, 4, 5}, {12, 8, 10}};
  gert::StorageShape scale_shape = {{30}, {120}};

  gert::StorageShape output_shape = {{}, {}};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(4, 1)
                    .IrInstanceNum({1, 1, 1, 1})
                    .InputShapes({&x1_shape, &x2_shape, nullptr, &scale_shape})
                    .OutputShapes({&output_shape})
                    .NodeAttrs({{"perm_x1", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({1, 0, 2})},
                                {"perm_x2", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({0, 1, 2})},
                                {"perm_y", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({1, 0, 2})},
                                {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                {"batch_split_factor", Ops::NN::AnyValue::CreateFrom<int64_t>(1)}})
                    .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .Build();

  ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(TransposeBatchMatMulInferShape, InvalidX1X2Case01) {
  auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("TransposeBatchMatMul")->infer_shape;

  gert::Shape x1_shape = {3, 4, 5};
  gert::Shape x2_shape = {3, 5, 4};

  gert::Shape output_shape = {};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(4, 1)
                    .IrInstanceNum({1, 1, 1, 1})
                    .InputShapes({&x1_shape, &x2_shape, nullptr, nullptr})
                    .OutputShapes({&output_shape})
                    .NodeAttrs({{"perm_x1", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({1, 0, 2})},
                                {"perm_x2", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({0, 1, 2})},
                                {"perm_y", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({1, 0, 2})},
                                {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                {"batch_split_factor", Ops::NN::AnyValue::CreateFrom<int64_t>(1)}})
                    .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .Build();

  ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(TransposeBatchMatMulInferShape, InvalidPermX1) {
  auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("TransposeBatchMatMul")->infer_shape;

  gert::Shape x1_shape = {3, 5, 4};
  gert::Shape x2_shape = {3, 5, 4};

  gert::Shape output_shape = {};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(4, 1)
                    .IrInstanceNum({1, 1, 1, 1})
                    .InputShapes({&x1_shape, &x2_shape, nullptr, nullptr})
                    .OutputShapes({&output_shape})
                    .NodeAttrs({{"perm_x1", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({0, 2, 1})},
                                {"perm_x2", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({0, 1, 2})},
                                {"perm_y", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({1, 0, 2})},
                                {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                {"batch_split_factor", Ops::NN::AnyValue::CreateFrom<int64_t>(1)}})
                    .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .Build();

  ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(TransposeBatchMatMulInferShape, InvalidPermX2Case01) {
  auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("TransposeBatchMatMul")->infer_shape;

  gert::Shape x1_shape = {4, 3, 5};
  gert::Shape x2_shape = {4, 3, 5};

  gert::Shape output_shape = {};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(4, 1)
                    .IrInstanceNum({1, 1, 1, 1})
                    .InputShapes({&x1_shape, &x2_shape, nullptr, nullptr})
                    .OutputShapes({&output_shape})
                    .NodeAttrs({{"perm_x1", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({1, 0, 2})},
                                {"perm_x2", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({1, 2, 0})},
                                {"perm_y", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({1, 0, 2})},
                                {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                {"batch_split_factor", Ops::NN::AnyValue::CreateFrom<int64_t>(1)}})
                    .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .Build();

  ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(TransposeBatchMatMulInferShape, InvalidPermX2_Bias) {
  auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("TransposeBatchMatMul")->infer_shape;

  gert::Shape x1_shape = {4, 3, 5};
  gert::Shape x2_shape = {4, 3, 5};
  gert::Shape bias_shape = {4};

  gert::Shape output_shape = {};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(4, 1)
                    .IrInstanceNum({1, 1, 1, 1})
                    .InputShapes({&x1_shape, &x2_shape, &bias_shape, nullptr})
                    .OutputShapes({&output_shape})
                    .NodeAttrs({{"perm_x1", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({1, 0, 2})},
                                {"perm_x2", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({1, 2, 0})},
                                {"perm_y", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({1, 0, 2})},
                                {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                {"batch_split_factor", Ops::NN::AnyValue::CreateFrom<int64_t>(1)}})
                    .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .Build();

  ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(TransposeBatchMatMulInferShape, InvalidPermX2Y_Bias) {
  auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("TransposeBatchMatMul")->infer_shape;

  gert::Shape x1_shape = {4, 3, 5};
  gert::Shape x2_shape = {4, 3, 5};
  gert::Shape bias_shape = {4};

  gert::Shape output_shape = {};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(4, 1)
                    .IrInstanceNum({1, 1, 1, 1})
                    .InputShapes({&x1_shape, &x2_shape, &bias_shape, nullptr})
                    .OutputShapes({&output_shape})
                    .NodeAttrs({{"perm_x1", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({1, 0, 2})},
                                {"perm_x2", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({1, 2, 0})},
                                {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                {"batch_split_factor", Ops::NN::AnyValue::CreateFrom<int64_t>(1)}})
                    .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .Build();

  ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

//  fail cases: broadcast
TEST_F(TransposeBatchMatMulInferShape, InvalidX2Case02) {
  auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("TransposeBatchMatMul")->infer_shape;

  gert::Shape x1_shape = {3, 4, 5};
  gert::Shape x2_shape = {5, 4};

  gert::Shape output_shape = {};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(4, 1)
                    .IrInstanceNum({1, 1, 1, 1})
                    .InputShapes({&x1_shape, &x2_shape, nullptr, nullptr})
                    .OutputShapes({&output_shape})
                    .NodeAttrs({{"perm_x1", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({1, 0, 2})},
                                {"perm_x2", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({0, 1, 2})},
                                {"perm_y", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({1, 0, 2})},
                                {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                {"batch_split_factor", Ops::NN::AnyValue::CreateFrom<int64_t>(1)}})
                    .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .Build();

  ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(TransposeBatchMatMulInferShape, InvalidX1X2Case02) {
  auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("TransposeBatchMatMul")->infer_shape;

  gert::Shape x1_shape = {3, 4, 5};
  gert::Shape x2_shape = {1, 5, 4};

  gert::Shape output_shape = {};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(4, 1)
                    .IrInstanceNum({1, 1, 1, 1})
                    .InputShapes({&x1_shape, &x2_shape, nullptr, nullptr})
                    .OutputShapes({&output_shape})
                    .NodeAttrs({{"perm_x1", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({1, 0, 2})},
                                {"perm_x2", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({0, 1, 2})},
                                {"perm_y", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({1, 0, 2})},
                                {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                {"batch_split_factor", Ops::NN::AnyValue::CreateFrom<int64_t>(1)}})
                    .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .Build();

  ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

//  fail cases: dim != 3
TEST_F(TransposeBatchMatMulInferShape, InvalidX1X2Case03) {
  auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("TransposeBatchMatMul")->infer_shape;

  gert::Shape x1_shape = {2, 3, 4, 5};
  gert::Shape x2_shape = {2, 3, 5, 4};

  gert::Shape output_shape = {};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(4, 1)
                    .IrInstanceNum({1, 1, 1, 1})
                    .InputShapes({&x1_shape, &x2_shape, nullptr, nullptr})
                    .OutputShapes({&output_shape})
                    .NodeAttrs({{"perm_x1", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({1, 0, 2})},
                                {"perm_x2", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({0, 1, 2})},
                                {"perm_y", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({1, 0, 2})},
                                {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                {"batch_split_factor", Ops::NN::AnyValue::CreateFrom<int64_t>(1)}})
                    .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .Build();

  ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

// dynamic shape
//   -2
TEST_F(TransposeBatchMatMulInferShape, InvalidX1X2Case04) {
  auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("TransposeBatchMatMul")->infer_shape;

  gert::Shape x1_shape = {-2};
  gert::Shape x2_shape = {-2};
  gert::Shape output_shape = {};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(4, 1)
                    .IrInstanceNum({1, 1, 1, 1})
                    .InputShapes({&x1_shape, &x2_shape, nullptr, nullptr})
                    .OutputShapes({&output_shape})
                    .NodeAttrs({{"perm_x1", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({1, 0, 2})},
                                {"perm_x2", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({0, 1, 2})},
                                {"perm_y", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({1, 0, 2})},
                                {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                {"batch_split_factor", Ops::NN::AnyValue::CreateFrom<int64_t>(1)}})
                    .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .Build();

  ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(TransposeBatchMatMulInferShape, InvalidBatchSplitFactor) {
  auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("TransposeBatchMatMul")->infer_shape;

  gert::Shape x1_shape = {32, 16, 512};
  gert::Shape x2_shape = {16, 512, 128};

  gert::Shape output_shape = {};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(4, 1)
                    .IrInstanceNum({1, 1, 1, 1})
                    .InputShapes({&x1_shape, &x2_shape, nullptr, nullptr})
                    .OutputShapes({&output_shape})
                    .NodeAttrs({{"perm_x1", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({1, 0, 2})},
                                {"perm_x2", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({0, 1, 2})},
                                {"perm_y", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({1, 0, 2})},
                                {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                {"batch_split_factor", Ops::NN::AnyValue::CreateFrom<int64_t>(10)}})
                    .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                    .Build();

  ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

TEST_F(TransposeBatchMatMulInferShape, InvalidPermX2Case02) {
  auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("TransposeBatchMatMul")->infer_shape;

  gert::Shape x1_shape = {-1, -1, -1};
  gert::Shape x2_shape = {-1, -1, -1};

  gert::Shape output_shape = {};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(4, 1)
                    .IrInstanceNum({1, 1, 1, 1})
                    .InputShapes({&x1_shape, &x2_shape, nullptr, nullptr})
                    .OutputShapes({&output_shape})
                    .NodeAttrs({{"perm_x1", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({1, 0, 2})},
                                {"perm_x2", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({1, 2, 0})},
                                {"perm_y", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({1, 0, 2})},
                                {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                {"batch_split_factor", Ops::NN::AnyValue::CreateFrom<int64_t>(1)}})
                    .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .Build();

  ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}
} // namespace
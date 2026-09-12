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
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include <gtest/gtest.h>
#include "kernel_run_context_facker.h"
#include "../../../op_graph/transpose_batch_mat_mul_proto.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "platform/platform_info.h"

namespace {
class TransposeBatchMatMulInferShape : public testing::Test {};

//   pass cases
TEST_F(TransposeBatchMatMulInferShape, Basic)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950PR_9589";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950PR_9589"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

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

TEST_F(TransposeBatchMatMulInferShape, Basic_Scale)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend910B";
    optiCompilationInfo.soc_version = "Ascend910B1";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910B1"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

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
TEST_F(TransposeBatchMatMulInferShape, InvalidX1X2Case01)
{
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

TEST_F(TransposeBatchMatMulInferShape, InvalidPermX1)
{
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

TEST_F(TransposeBatchMatMulInferShape, InvalidPermX2Case01)
{
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

TEST_F(TransposeBatchMatMulInferShape, InvalidPermX2_Bias)
{
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

TEST_F(TransposeBatchMatMulInferShape, InvalidPermX2Y_Bias)
{
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
TEST_F(TransposeBatchMatMulInferShape, InvalidX2Case02)
{
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

TEST_F(TransposeBatchMatMulInferShape, InvalidX1X2Case02)
{
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
TEST_F(TransposeBatchMatMulInferShape, InvalidX1X2Case03)
{
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
TEST_F(TransposeBatchMatMulInferShape, InvalidX1X2Case04)
{
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

TEST_F(TransposeBatchMatMulInferShape, InvalidBatchSplitFactor)
{
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

TEST_F(TransposeBatchMatMulInferShape, InvalidPermX2Case02)
{
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
// ================== Scale mode tests (covers infershape.cpp scale path, lines 88, 176, 190-191) ==================
TEST_F(TransposeBatchMatMulInferShape, ScaleMode)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend910B";
    optiCompilationInfo.soc_version = "Ascend910B1";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910B1"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("TransposeBatchMatMul")->infer_shape;

    // x1={32,16,512}, x2={16,512,128}, perm_x1={1,0,2}, perm_x2={0,1,2}
    // After transpose: x1={16,32,512}, x2={16,512,128}
    // batch=16, m=32, k=512, n=128
    // scale dim = batch*n = 16*128 = 2048
    gert::Shape x1_shape = {32, 16, 512};
    gert::Shape x2_shape = {16, 512, 128};
    gert::Shape scale_shape = {2048};
    gert::Shape expect_output_shape = {32, 1, 2048}; // scale mode: [m, 1, batch*n]

    gert::Shape output_shape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(4, 1)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&x1_shape, &x2_shape, nullptr, &scale_shape})
                      .OutputShapes({&output_shape})
                      .NodeAttrs({{"perm_x1", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({1, 0, 2})},
                                  {"perm_x2", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({0, 1, 2})},
                                  {"perm_y", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({1, 0, 2})},
                                  {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"batch_split_factor", Ops::NN::AnyValue::CreateFrom<int64_t>(1)}})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), Ops::Base::ToString(expect_output_shape));
}

// 覆盖 lines 53, 180: CheckIsUnknownDimNum with scale={-2}
TEST_F(TransposeBatchMatMulInferShape, ScaleModeUnknownDim)
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend910B";
    optiCompilationInfo.soc_version = "Ascend910B1";
    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend910B1"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);

    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("TransposeBatchMatMul")->infer_shape;

    gert::Shape x1_shape = {32, 16, 512};
    gert::Shape x2_shape = {16, 512, 128};
    gert::Shape scale_shape = {-2}; // unknown dim, triggers CheckIsUnknownDimNum = true

    gert::Shape output_shape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(4, 1)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&x1_shape, &x2_shape, nullptr, &scale_shape})
                      .OutputShapes({&output_shape})
                      .NodeAttrs({{"perm_x1", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({1, 0, 2})},
                                  {"perm_x2", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({0, 1, 2})},
                                  {"perm_y", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({1, 0, 2})},
                                  {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"batch_split_factor", Ops::NN::AnyValue::CreateFrom<int64_t>(1)}})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

// 覆盖 line 177: batch_split_factor != 1 when scale present
TEST_F(TransposeBatchMatMulInferShape, ScaleModeInvalidBatchSplit)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("TransposeBatchMatMul")->infer_shape;

    gert::Shape x1_shape = {32, 16, 512};
    gert::Shape x2_shape = {16, 512, 128};
    gert::Shape scale_shape = {2048};

    gert::Shape output_shape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(4, 1)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&x1_shape, &x2_shape, nullptr, &scale_shape})
                      .OutputShapes({&output_shape})
                      .NodeAttrs({{"perm_x1", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({1, 0, 2})},
                                  {"perm_x2", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({0, 1, 2})},
                                  {"perm_y", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({1, 0, 2})},
                                  {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"batch_split_factor", Ops::NN::AnyValue::CreateFrom<int64_t>(2)}})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

// 覆盖 line 180: scale shape does not match batch*n
TEST_F(TransposeBatchMatMulInferShape, ScaleModeShapeMismatch)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("TransposeBatchMatMul")->infer_shape;

    gert::Shape x1_shape = {32, 16, 512};
    gert::Shape x2_shape = {16, 512, 128};
    gert::Shape scale_shape = {1024}; // batch*n = 2048, mismatch

    gert::Shape output_shape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(4, 1)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&x1_shape, &x2_shape, nullptr, &scale_shape})
                      .OutputShapes({&output_shape})
                      .NodeAttrs({{"perm_x1", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({1, 0, 2})},
                                  {"perm_x2", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({0, 1, 2})},
                                  {"perm_y", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({1, 0, 2})},
                                  {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"batch_split_factor", Ops::NN::AnyValue::CreateFrom<int64_t>(1)}})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

// 覆盖 lines 192, 195: dtype_x1 != DT_FLOAT16 when scale present
TEST_F(TransposeBatchMatMulInferShape, ScaleModeInvalidDtype)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("TransposeBatchMatMul")->infer_shape;

    gert::Shape x1_shape = {32, 16, 512};
    gert::Shape x2_shape = {16, 512, 128};
    gert::Shape scale_shape = {2048};

    gert::Shape output_shape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(4, 1)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&x1_shape, &x2_shape, nullptr, &scale_shape})
                      .OutputShapes({&output_shape})
                      .NodeAttrs({{"perm_x1", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({1, 0, 2})},
                                  {"perm_x2", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({0, 1, 2})},
                                  {"perm_y", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({1, 0, 2})},
                                  {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"batch_split_factor", Ops::NN::AnyValue::CreateFrom<int64_t>(1)}})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND) // not FP16
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND) // not FP16
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

// 覆盖 line 195: dtype_x2 != DT_FLOAT16 when scale present (x1 is FP16, x2 is not)
TEST_F(TransposeBatchMatMulInferShape, ScaleModeInvalidDtypeX2)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("TransposeBatchMatMul")->infer_shape;

    gert::Shape x1_shape = {32, 16, 512};
    gert::Shape x2_shape = {16, 512, 128};
    gert::Shape scale_shape = {2048};

    gert::Shape output_shape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(4, 1)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&x1_shape, &x2_shape, nullptr, &scale_shape})
                      .OutputShapes({&output_shape})
                      .NodeAttrs({{"perm_x1", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({1, 0, 2})},
                                  {"perm_x2", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({0, 1, 2})},
                                  {"perm_y", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({1, 0, 2})},
                                  {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"batch_split_factor", Ops::NN::AnyValue::CreateFrom<int64_t>(1)}})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND) // x1 is FP16, passes line 192
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)   // x2 not FP16, hits line 195
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

// 覆盖 line 122: perm_x2 = {0,2,1} (second condition of OR in CheckPerm)
TEST_F(TransposeBatchMatMulInferShape, PermX2_021)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("TransposeBatchMatMul")->infer_shape;

    // x1={32,16,512}, x2={16,128,512}, perm_x2={0,2,1}
    // After transpose: x1={16,32,512}, x2={16,512,128}
    // k=512, batch=16, m=32, n=128
    // perm_y={1,0,2}: [batch,m,n] -> [m,batch,n]
    gert::Shape x1_shape = {32, 16, 512};
    gert::Shape x2_shape = {16, 128, 512};
    gert::Shape expect_output_shape = {32, 16, 128};

    gert::Shape output_shape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(4, 1)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&x1_shape, &x2_shape, nullptr, nullptr})
                      .OutputShapes({&output_shape})
                      .NodeAttrs({{"perm_x1", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({1, 0, 2})},
                                  {"perm_x2", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({0, 2, 1})},
                                  {"perm_y", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({1, 0, 2})},
                                  {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"batch_split_factor", Ops::NN::AnyValue::CreateFrom<int64_t>(1)}})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
    auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
    ASSERT_EQ(Ops::Base::ToString(*output), Ops::Base::ToString(expect_output_shape));
}

// 覆盖 line 128: invalid perm_y error return (perm_y should be {1,0,2})
TEST_F(TransposeBatchMatMulInferShape, InvalidPermY)
{
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
                                  {"perm_y", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({0, 1, 2})},
                                  {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"batch_split_factor", Ops::NN::AnyValue::CreateFrom<int64_t>(1)}})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

// 覆盖 line 162: k-axis mismatch after transpose
TEST_F(TransposeBatchMatMulInferShape, KMismatch)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("TransposeBatchMatMul")->infer_shape;

    // x1={3,4,5}, x2={3,6,4}, perm_x1={1,0,2}, perm_x2={0,1,2}
    // After transpose: x1={4,3,5} (k=dim2=5), x2={3,6,4} (k=dim1=6) -> mismatch
    gert::Shape x1_shape = {3, 4, 5};
    gert::Shape x2_shape = {3, 6, 4};

    gert::Shape output_shape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(4, 1)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&x1_shape, &x2_shape, nullptr, nullptr})
                      .OutputShapes({&output_shape})
                      .NodeAttrs({{"perm_x1", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({1, 0, 2})},
                                  {"perm_x2", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({0, 1, 2})},
                                  {"perm_y", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({1, 0, 2})},
                                  {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"batch_split_factor", Ops::NN::AnyValue::CreateFrom<int64_t>(1)}})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

// 覆盖 line 122: perm_x2 = {0,2,1} with dynamic shape
TEST_F(TransposeBatchMatMulInferShape, PermX2_021_Dynamic)
{
    auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("TransposeBatchMatMul")->infer_shape;

    // Dynamic shapes with perm_x2={0,2,1}: x2_transposed = {batch, n, k}
    // x1={-1,-1,-1}, x2={-1,-1,-1}: all dynamic, fails because k dims cannot be matched
    gert::Shape x1_shape = {-1, -1, -1};
    gert::Shape x2_shape = {-1, -1, -1};

    gert::Shape output_shape = {};

    auto holder = gert::InferShapeContextFaker()
                      .NodeIoNum(4, 1)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&x1_shape, &x2_shape, nullptr, nullptr})
                      .OutputShapes({&output_shape})
                      .NodeAttrs({{"perm_x1", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({1, 0, 2})},
                                  {"perm_x2", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({0, 2, 1})},
                                  {"perm_y", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({1, 0, 2})},
                                  {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                                  {"batch_split_factor", Ops::NN::AnyValue::CreateFrom<int64_t>(1)}})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .Build();

    ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

} // namespace

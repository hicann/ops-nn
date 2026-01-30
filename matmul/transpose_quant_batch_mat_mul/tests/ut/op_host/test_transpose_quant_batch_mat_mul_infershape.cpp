/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
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
#include "../../../op_graph/transpose_quant_batch_mat_mul_proto.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "platform/platform_info.h"

namespace {
class TransposeQuantBatchMatMulInferShape : public testing::Test {
};

// pass cases
TEST_F(TransposeQuantBatchMatMulInferShape, Basic01) {
  fe::PlatformInfo platformInfo;
  fe::OptionalInfo optiCompilationInfo;
  platformInfo.soc_info.ai_core_cnt = 64;
  platformInfo.str_info.short_soc_version = "Ascend950";
  optiCompilationInfo.soc_version = "Ascend950PR_9589";
  fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950PR_9589"] = platformInfo;
  fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
  
  auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("TransposeQuantBatchMatMul")->infer_shape;
  // M = 36
  gert::Shape x1_shape = {32, 16, 512};
  gert::Shape x2_shape = {16, 512, 128};
  gert::Shape x1_scale_shape = {32};
  gert::Shape x2_scale_shape = {128};
  gert::Shape expect_output_shape = {32, 16, 128};

  gert::Shape output_shape = {};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(5, 1)
                    .IrInstanceNum({1, 1, 1, 1, 1})
                    .InputShapes({&x1_shape, &x2_shape, nullptr, &x1_scale_shape, &x2_scale_shape})
                    .OutputShapes({&output_shape})
                    .NodeAttrs({{"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"group_size", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"perm_x1", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({1, 0, 2})},
                                {"perm_x2", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({0, 1, 2})},
                                {"perm_y", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({1, 0, 2})},
                                {"batch_split_factor", Ops::NN::AnyValue::CreateFrom<int64_t>(1)}})
                    .NodeInputTd(0, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .Build();

  ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
  auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
  ASSERT_EQ(Ops::Base::ToString(*output), Ops::Base::ToString(expect_output_shape));
}

// pass cases
TEST_F(TransposeQuantBatchMatMulInferShape, Basic02) {
  fe::PlatformInfo platformInfo;
  fe::OptionalInfo optiCompilationInfo;
  platformInfo.soc_info.ai_core_cnt = 64;
  platformInfo.str_info.short_soc_version = "Ascend950";
  optiCompilationInfo.soc_version = "Ascend950PR_9589";
  fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950PR_9589"] = platformInfo;
  fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
  
  auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("TransposeQuantBatchMatMul")->infer_shape;
  // M = 36
  gert::Shape x1_shape = {32, 16, 512};
  gert::Shape x2_shape = {16, 512, 128};
  gert::Shape x1_scale_shape = {32};
  gert::Shape x2_scale_shape = {128};
  gert::Shape expect_output_shape = {32, 16, 128};

  gert::Shape output_shape = {};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(5, 1)
                    .IrInstanceNum({1, 1, 1, 1, 1})
                    .InputShapes({&x1_shape, &x2_shape, nullptr, &x1_scale_shape, &x2_scale_shape})
                    .OutputShapes({&output_shape})
                    .NodeAttrs({{"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(27)},
                                {"group_size", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"perm_x1", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({1, 0, 2})},
                                {"perm_x2", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({0, 1, 2})},
                                {"perm_y", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({1, 0, 2})},
                                {"batch_split_factor", Ops::NN::AnyValue::CreateFrom<int64_t>(1)}})
                    .NodeInputTd(0, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .Build();

  ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
  auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
  ASSERT_EQ(Ops::Base::ToString(*output), Ops::Base::ToString(expect_output_shape));
}

// pass cases
TEST_F(TransposeQuantBatchMatMulInferShape, Basic03) {
  fe::PlatformInfo platformInfo;
  fe::OptionalInfo optiCompilationInfo;
  platformInfo.soc_info.ai_core_cnt = 64;
  platformInfo.str_info.short_soc_version = "Ascend950";
  optiCompilationInfo.soc_version = "Ascend950PR_9589";
  fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950PR_9589"] = platformInfo;
  fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
  
  auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("TransposeQuantBatchMatMul")->infer_shape;
  // M = 36
  gert::Shape x1_shape = {32, 16, 512};
  gert::Shape x2_shape = {16, 512, 128};
  gert::Shape x1_scale_shape = {32};
  gert::Shape x2_scale_shape = {128};
  gert::Shape expect_output_shape = {32, 16, 128};

  gert::Shape output_shape = {};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(5, 1)
                    .IrInstanceNum({1, 1, 1, 1, 1})
                    .InputShapes({&x1_shape, &x2_shape, nullptr, &x1_scale_shape, &x2_scale_shape})
                    .OutputShapes({&output_shape})
                    .NodeAttrs({{"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"group_size", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"perm_x1", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({1, 0, 2})},
                                {"perm_x2", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({0, 1, 2})},
                                {"perm_y", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({1, 0, 2})},
                                {"batch_split_factor", Ops::NN::AnyValue::CreateFrom<int64_t>(1)}})
                    .NodeInputTd(0, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .Build();

  ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
  auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
  ASSERT_EQ(Ops::Base::ToString(*output), Ops::Base::ToString(expect_output_shape));
}

// pass cases
TEST_F(TransposeQuantBatchMatMulInferShape, Basic04) {
  fe::PlatformInfo platformInfo;
  fe::OptionalInfo optiCompilationInfo;
  platformInfo.soc_info.ai_core_cnt = 64;
  platformInfo.str_info.short_soc_version = "Ascend950";
  optiCompilationInfo.soc_version = "Ascend950PR_9589";
  fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950PR_9589"] = platformInfo;
  fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
  
  auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("TransposeQuantBatchMatMul")->infer_shape;
  // M = 36
  gert::Shape x1_shape = {32, 16, 512};
  gert::Shape x2_shape = {16, 512, 128};
  gert::Shape x1_scale_shape = {32};
  gert::Shape x2_scale_shape = {128};
  gert::Shape expect_output_shape = {32, 16, 128};

  gert::Shape output_shape = {};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(5, 1)
                    .IrInstanceNum({1, 1, 1, 1, 1})
                    .InputShapes({&x1_shape, &x2_shape, nullptr, &x1_scale_shape, &x2_scale_shape})
                    .OutputShapes({&output_shape})
                    .NodeAttrs({{"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(27)},
                                {"group_size", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"perm_x1", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({1, 0, 2})},
                                {"perm_x2", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({0, 1, 2})},
                                {"perm_y", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({1, 0, 2})},
                                {"batch_split_factor", Ops::NN::AnyValue::CreateFrom<int64_t>(1)}})
                    .NodeInputTd(0, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .Build();

  ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_SUCCESS);
  auto output = holder.GetContext<gert::InferShapeContext>()->GetOutputShape(0);
  ASSERT_EQ(Ops::Base::ToString(*output), Ops::Base::ToString(expect_output_shape));
}

//  fail cases
TEST_F(TransposeQuantBatchMatMulInferShape, InvalidBatchSplitFactor) {
  fe::PlatformInfo platformInfo;
  fe::OptionalInfo optiCompilationInfo;
  platformInfo.soc_info.ai_core_cnt = 64;
  platformInfo.str_info.short_soc_version = "Ascend950";
  optiCompilationInfo.soc_version = "Ascend950PR_9589";
  fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950PR_9589"] = platformInfo;
  fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
  
  auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("TransposeQuantBatchMatMul")->infer_shape;
  // M = 36
  gert::Shape x1_shape = {32, 16, 512};
  gert::Shape x2_shape = {16, 512, 128};
  gert::Shape x1_scale_shape = {32};
  gert::Shape x2_scale_shape = {128};
  gert::Shape expect_output_shape = {32, 16, 128};

  gert::Shape output_shape = {};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(5, 1)
                    .IrInstanceNum({1, 1, 1, 1, 1})
                    .InputShapes({&x1_shape, &x2_shape, nullptr, &x1_scale_shape, &x1_scale_shape})
                    .OutputShapes({&output_shape})
                    .NodeAttrs({{"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(27)},
                                {"group_size", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"perm_x1", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({1, 0, 2})},
                                {"perm_x2", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({0, 1, 2})},
                                {"perm_y", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({1, 0, 2})},
                                {"batch_split_factor", Ops::NN::AnyValue::CreateFrom<int64_t>(2)}})
                    .NodeInputTd(0, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .Build();

  ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

//  fail cases
TEST_F(TransposeQuantBatchMatMulInferShape, InvalidK) {
  fe::PlatformInfo platformInfo;
  fe::OptionalInfo optiCompilationInfo;
  platformInfo.soc_info.ai_core_cnt = 64;
  platformInfo.str_info.short_soc_version = "Ascend950";
  optiCompilationInfo.soc_version = "Ascend950PR_9589";
  fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950PR_9589"] = platformInfo;
  fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
  
  auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("TransposeQuantBatchMatMul")->infer_shape;
  // M = 36
  gert::Shape x1_shape = {32, 16, 256};
  gert::Shape x2_shape = {16, 256, 128};
  gert::Shape x1_scale_shape = {32};
  gert::Shape x2_scale_shape = {128};
  gert::Shape expect_output_shape = {32, 16, 128};

  gert::Shape output_shape = {};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(5, 1)
                    .IrInstanceNum({1, 1, 1, 1, 1})
                    .InputShapes({&x1_shape, &x2_shape, nullptr, &x1_scale_shape, &x1_scale_shape})
                    .OutputShapes({&output_shape})
                    .NodeAttrs({{"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(27)},
                                {"group_size", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"perm_x1", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({1, 0, 2})},
                                {"perm_x2", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({0, 1, 2})},
                                {"perm_y", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({1, 0, 2})},
                                {"batch_split_factor", Ops::NN::AnyValue::CreateFrom<int64_t>(2)}})
                    .NodeInputTd(0, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .Build();

  ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}

//  fail cases
TEST_F(TransposeQuantBatchMatMulInferShape, InvalidN) {
  fe::PlatformInfo platformInfo;
  fe::OptionalInfo optiCompilationInfo;
  platformInfo.soc_info.ai_core_cnt = 64;
  platformInfo.str_info.short_soc_version = "Ascend950";
  optiCompilationInfo.soc_version = "Ascend950PR_9589";
  fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950PR_9589"] = platformInfo;
  fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
  
  auto inferShapeFunc = gert::OpImplRegistry::GetInstance().GetOpImpl("TransposeQuantBatchMatMul")->infer_shape;
  // M = 36
  gert::Shape x1_shape = {32, 16, 512};
  gert::Shape x2_shape = {16, 512, 256};
  gert::Shape x1_scale_shape = {32};
  gert::Shape x2_scale_shape = {256};
  gert::Shape expect_output_shape = {32, 16, 256};

  gert::Shape output_shape = {};

  auto holder = gert::InferShapeContextFaker()
                    .NodeIoNum(5, 1)
                    .IrInstanceNum({1, 1, 1, 1, 1})
                    .InputShapes({&x1_shape, &x2_shape, nullptr, &x1_scale_shape, &x1_scale_shape})
                    .OutputShapes({&output_shape})
                    .NodeAttrs({{"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(27)},
                                {"group_size", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                                {"perm_x1", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({1, 0, 2})},
                                {"perm_x2", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({0, 1, 2})},
                                {"perm_y", Ops::NN::AnyValue::CreateFrom<vector<int64_t>>({1, 0, 2})},
                                {"batch_split_factor", Ops::NN::AnyValue::CreateFrom<int64_t>(2)}})
                    .NodeInputTd(0, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .Build();

  ASSERT_EQ(inferShapeFunc(holder.GetContext<gert::InferShapeContext>()), ge::GRAPH_FAILED);
}
} // namespace
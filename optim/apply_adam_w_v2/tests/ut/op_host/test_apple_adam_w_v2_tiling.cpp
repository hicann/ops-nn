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
#include <fstream>
#include <vector>
#include <gtest/gtest.h>
#include "../../../op_host/apply_adam_w_v2_tiling.h"
#include "log/log.h"
#include "ut_op_common.h"
#include "register/op_impl_registry.h"
#include "platform/platform_infos_def.h"
#include "ut_op_util.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"

using namespace ut_util;
using namespace std;
using namespace ge;

class ApplyAdamWV2Tiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ApplyAdamWV2Tiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ApplyAdamWV2Tiling TearDown" << std::endl;
  }
};

static string TilingData2Str(const gert::TilingData *tiling_data) {
  auto data = tiling_data->GetData();
  string result;
  for (size_t i = 0; i < tiling_data->GetDataSize();) {
    if (i == 7 * sizeof(int64_t) || i == 7 * sizeof(int64_t) + sizeof(float) ||
      i == 7 * sizeof(int64_t) + 2 * sizeof(float) || i == 7 * sizeof(int64_t) + 3 * sizeof(float) ||
      i == 7 * sizeof(int64_t) + 4 * sizeof(float)) {
      result += std::to_string((reinterpret_cast<const float*>(tiling_data->GetData())[i / sizeof(float)]));
      i += sizeof(float);
    } else {
      if (i == 7 * sizeof(int64_t) + 5 * sizeof(float)) {
        i += sizeof(float);
      }
      result += std::to_string((reinterpret_cast<const int64_t*>(tiling_data->GetData())[i / sizeof(int64_t)]));
      i += sizeof(int64_t);
    }
    result += " ";
  }

  return result;
}

TEST_F(ApplyAdamWV2Tiling, ApplyAdamWV2_tiling_float) {
  // input shape
  gert::StorageShape varShape = {{300, 4, 2}, {300, 4, 2}};
  gert::StorageShape mShape = {{300, 4, 2}, {300, 4, 2}};
  gert::StorageShape vShape = {{300, 4, 2}, {300, 4, 2}};
  gert::StorageShape gradShape = {{300, 4, 2}, {300, 4, 2}};
  gert::StorageShape stepShape = {{1}, {1}};
  gert::StorageShape maxGradNorShape = {{300, 4, 2}, {300, 4, 2}};

  string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 40}
                          })";
  map<string, string> soc_infos;
  map<string, string> aicore_spec;
  map<string, string> intrinsics;

  GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

  // platform info
  fe::PlatFormInfos platform_info;
  platform_info.Init();

  // compile info
  optiling::ApplyAdamWV2CompileInfo compile_info;

  std::string op_type("ApplyAdamWV2");
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
  auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
  auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

  // tilingParseFunc simulate
  auto kernel_holder = gert::KernelRunContextFaker()
                    .KernelIONum(2, 1)
                    .Inputs({const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void *>(&platform_info)})
                    .Outputs({&compile_info})
                    .Build();

  ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

  ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
  // tilingFunc simulate
  auto param = gert::TilingData::CreateCap(4096);
  auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
  auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
  ASSERT_NE(param, nullptr);
  auto holder = gert::TilingContextFaker()
                    .SetOpType("ApplyAdamWV2")
                    .NodeIoNum(6, 4)
                    .IrInstanceNum({1, 1, 1, 1, 1, 1})
                    .InputShapes({&varShape,
                                  &mShape,
                                  &vShape,
                                  &gradShape,
                                  &stepShape,
                                  &maxGradNorShape})
                    .InputShapes({&varShape,
                                  &mShape,
                                  &vShape,
                                  &maxGradNorShape})
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                    .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({{"lr", Ops::NN::AnyValue::CreateFrom(0.001f)},
                                {"beta1", Ops::NN::AnyValue::CreateFrom(0.01f)},
                                {"beta2", Ops::NN::AnyValue::CreateFrom(0.09f)},
                                {"weightDecay", Ops::NN::AnyValue::CreateFrom(0.0001f)},
                                {"eps", Ops::NN::AnyValue::CreateFrom(0.000001f)},
                                {"amsgrad", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                {"maximize", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
                    .TilingData(param.get())
                    .Workspace(ws_size)
                    .Build();

  gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
  ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

  // workspaces nullptr return failed
  EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);

  // todo check tiling result
  auto tiling_key = tiling_context->GetTilingKey();

  auto block_dim = tiling_context->GetBlockDim();

  auto tiling_data_result = TilingData2Str(tiling_context->GetRawTilingData());
  std::cout << tiling_data_result << std::endl;
}

TEST_F(ApplyAdamWV2Tiling, ApplyAdamWV2_tiling_diff_dtype) {
  // input shape
  gert::StorageShape varShape = {{300, 4, 2}, {300, 4, 2}};
  gert::StorageShape mShape = {{300, 4, 2}, {300, 4, 2}};
  gert::StorageShape vShape = {{300, 4, 2}, {300, 4, 2}};
  gert::StorageShape gradShape = {{300, 4, 2}, {300, 4, 2}};
  gert::StorageShape stepShape = {{1}, {1}};
  gert::StorageShape maxGradNorShape = {{300, 4, 2}, {300, 4, 2}};

  string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 40}
                          })";
  map<string, string> soc_infos;
  map<string, string> aicore_spec;
  map<string, string> intrinsics;

  GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

  // platform info
  fe::PlatFormInfos platform_info;
  platform_info.Init();

  // compile info
  optiling::ApplyAdamWV2CompileInfo compile_info;

  std::string op_type("ApplyAdamWV2");
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
  auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
  auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

  // tilingParseFunc simulate
  auto kernel_holder = gert::KernelRunContextFaker()
                    .KernelIONum(2, 1)
                    .Inputs({const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void *>(&platform_info)})
                    .Outputs({&compile_info})
                    .Build();

  ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

  ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
  // tilingFunc simulate
  auto param = gert::TilingData::CreateCap(4096);
  auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
  auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
  ASSERT_NE(param, nullptr);
  auto holder = gert::TilingContextFaker()
                    .SetOpType("ApplyAdamWV2")
                    .NodeIoNum(6, 4)
                    .IrInstanceNum({1, 1, 1, 1, 1, 1})
                    .InputShapes({&varShape,
                                  &mShape,
                                  &vShape,
                                  &gradShape,
                                  &stepShape,
                                  &maxGradNorShape})
                    .InputShapes({&varShape,
                                  &mShape,
                                  &vShape,
                                  &maxGradNorShape})
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                    .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({{"lr", Ops::NN::AnyValue::CreateFrom(0.001f)},
                                {"beta1", Ops::NN::AnyValue::CreateFrom(0.01f)},
                                {"beta2", Ops::NN::AnyValue::CreateFrom(0.09f)},
                                {"weightDecay", Ops::NN::AnyValue::CreateFrom(0.0001f)},
                                {"eps", Ops::NN::AnyValue::CreateFrom(0.000001f)},
                                {"amsgrad", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                {"maximize", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
                    .TilingData(param.get())
                    .Workspace(ws_size)
                    .Build();

  gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
  ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

  // workspaces nullptr return failed
  EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);

  // todo check tiling result
  auto tiling_key = tiling_context->GetTilingKey();

  auto block_dim = tiling_context->GetBlockDim();

  auto tiling_data_result = TilingData2Str(tiling_context->GetRawTilingData());
  std::cout << tiling_data_result << std::endl;
}

TEST_F(ApplyAdamWV2Tiling, ApplyAdamWV2_tiling_float_can_avg_div) {
  // input shape
  gert::StorageShape varShape = {{2, 2, 2, 2, 2, 2, 241662, 2}, {2, 2, 2, 2, 2, 2, 241662, 2}};
  gert::StorageShape mShape = {{2, 2, 2, 2, 2, 2, 241662, 2}, {2, 2, 2, 2, 2, 2, 241662, 2}};
  gert::StorageShape vShape = {{2, 2, 2, 2, 2, 2, 241662, 2}, {2, 2, 2, 2, 2, 2, 241662, 2}};
  gert::StorageShape gradShape = {{2, 2, 2, 2, 2, 2, 241662, 2}, {2, 2, 2, 2, 2, 2, 241662, 2}};
  gert::StorageShape stepShape = {{1}, {1}};
  gert::StorageShape maxGradNorShape = {{2, 2, 2, 2, 2, 2, 241662, 2}, {2, 2, 2, 2, 2, 2, 241662, 2}};

  string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 40}
                          })";
  map<string, string> soc_infos;
  map<string, string> aicore_spec;
  map<string, string> intrinsics;

  GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

  // platform info
  fe::PlatFormInfos platform_info;
  platform_info.Init();

  // compile info
  optiling::ApplyAdamWV2CompileInfo compile_info;

  std::string op_type("ApplyAdamWV2");
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
  auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
  auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

  // tilingParseFunc simulate
  auto kernel_holder = gert::KernelRunContextFaker()
                    .KernelIONum(2, 1)
                    .Inputs({const_cast<char *>(compile_info_string.c_str()), reinterpret_cast<void *>(&platform_info)})
                    .Outputs({&compile_info})
                    .Build();

  ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

  ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
  // tilingFunc simulate
  auto param = gert::TilingData::CreateCap(4096);
  auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
  auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
  ASSERT_NE(param, nullptr);
  auto holder = gert::TilingContextFaker()
                    .SetOpType("ApplyAdamWV2")
                    .NodeIoNum(6, 4)
                    .IrInstanceNum({1, 1, 1, 1, 1, 1})
                    .InputShapes({&varShape,
                                  &mShape,
                                  &vShape,
                                  &gradShape,
                                  &stepShape,
                                  &maxGradNorShape})
                    .InputShapes({&varShape,
                                  &mShape,
                                  &vShape,
                                  &maxGradNorShape})
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                    .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({{"lr", Ops::NN::AnyValue::CreateFrom(0.001f)},
                                {"beta1", Ops::NN::AnyValue::CreateFrom(0.01f)},
                                {"beta2", Ops::NN::AnyValue::CreateFrom(0.09f)},
                                {"weightDecay", Ops::NN::AnyValue::CreateFrom(0.0001f)},
                                {"eps", Ops::NN::AnyValue::CreateFrom(0.000001f)},
                                {"amsgrad", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                                {"maximize", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
                    .TilingData(param.get())
                    .Workspace(ws_size)
                    .Build();

  gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
  ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

  // workspaces nullptr return failed
  EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);

  // todo check tiling result
  auto tiling_key = tiling_context->GetTilingKey();

  auto block_dim = tiling_context->GetBlockDim();

  auto tiling_data_result = TilingData2Str(tiling_context->GetRawTilingData());
  std::cout << tiling_data_result << std::endl;
}
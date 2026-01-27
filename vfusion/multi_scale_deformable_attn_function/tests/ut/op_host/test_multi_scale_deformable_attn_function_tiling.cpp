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
#include <vector>
#include "log/log.h"
#include <gtest/gtest.h>
#include "register/op_impl_registry.h"
#include "array_ops.h"
#include "vfusion/multi_scale_deformable_attn_function/op_host/multi_scale_deformable_attn_function_tiling.h"
#include "test_cube_util.h"
#include "kernel_run_context_facker.h"
#include "ut_op_util.h"
#include "platform/platform_infos_def.h"

using namespace std;
using namespace ge;
using namespace ut_util;

class MultiScaleDeformableAttnFunctionTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "MultiScaleDeformableAttnFunctionTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "MultiScaleDeformableAttnFunctionTiling TearDown" << std::endl;
  }
};

static string TilingData2Str(const gert::TilingData *tiling_data) {
  auto data = tiling_data->GetData();
  string result;
  for (size_t i = 0; i < tiling_data->GetDataSize(); i += sizeof(int64_t)) {
    result += std::to_string((reinterpret_cast<const int64_t *>(tiling_data->GetData())[i / sizeof(int64_t)]));
    result += " ";
  }

  return result;
}

TEST_F(MultiScaleDeformableAttnFunctionTiling, MultiScaleDeformableAttnFunction_tiling_generic) {
  std::string op_type("MultiScaleDeformableAttnFunction");
  string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 48}
                          })";

  map<string, string> soc_infos;
  map<string, string> aicore_spec;
  map<string, string> intrinsics;
  GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

  // platform info
  fe::PlatFormInfos platform_info;
  platform_info.Init();
  struct MultiScaleDeformableAttnFunctionCompileInfo {
    int32_t total_core_num = 0;
    uint64_t ub_size_platform = 0;
    bool is_inf_base = false;
  } compile_info;


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

  uint64_t batch = 1;
  uint64_t spatial_size = 24;
  uint64_t num_heads = 8;
  uint64_t channels = 32;
  uint64_t num_levels = 1;
  uint64_t num_query = 100;
  uint64_t num_point = 2;
  gert::StorageShape value_shape = {{batch, spatial_size, num_heads, channels}, {batch, spatial_size, num_heads, channels}};
  gert::StorageShape value_spatial_shapes_shape = {{num_levels, 2}, {num_levels, 2}};
  gert::StorageShape value_level_start_index_shape = {{num_levels}, {num_levels}};
  gert::StorageShape sampling_locations_shape = {{batch, num_query, num_heads, num_levels, num_point, 2},
    {batch, num_query, num_heads, num_levels, num_point, 2}};
  gert::StorageShape attention_weights_shape = {{batch, num_query, num_heads, num_levels, num_point}, {batch, num_query, num_heads, num_levels, num_point}};
  gert::StorageShape output_shape = {{batch, num_query, num_heads * channels}, {batch, num_query, num_heads * channels}};

  auto holder = gert::TilingContextFaker()
                    .NodeIoNum(5, 1)
                    .IrInstanceNum({1, 1, 1, 1, 1})
                    .InputShapes({&value_shape, &value_spatial_shapes_shape, &value_level_start_index_shape, &sampling_locations_shape, &attention_weights_shape})
                    .OutputShapes({&output_shape})
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                    .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
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
  auto tiling_data_result = TilingData2Str(tiling_context->GetRawTilingData());
  std::cout << tiling_data_result << std::endl;
}

TEST_F(MultiScaleDeformableAttnFunctionTiling, MultiScaleDeformableAttnFunction_tiling_high_perf) {
  std::string op_type("MultiScaleDeformableAttnFunction");
  string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 48}
                          })";

  map<string, string> soc_infos;
  map<string, string> aicore_spec;
  map<string, string> intrinsics;
  GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

  // platform info
  fe::PlatFormInfos platform_info;
  platform_info.Init();
  struct MultiScaleDeformableAttnFunctionCompileInfo {
    int32_t total_core_num = 0;
    uint64_t ub_size_platform = 0;
    bool is_inf_base = false;
  } compile_info;


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

  uint64_t batch = 1;
  uint64_t spatial_size = 24;
  uint64_t num_heads = 8;
  uint64_t channels = 32;
  uint64_t num_levels = 1;
  uint64_t num_query = 100;
  uint64_t num_point = 2;
  gert::StorageShape value_shape = {{batch, spatial_size, num_heads, channels}, {batch, spatial_size, num_heads, channels}};
  gert::StorageShape value_spatial_shapes_shape = {{num_levels, 2}, {num_levels, 2}};
  gert::StorageShape value_level_start_index_shape = {{num_levels}, {num_levels}};
  gert::StorageShape sampling_locations_shape = {{batch, num_query, num_heads, num_levels, num_point, 2},
    {batch, num_query, num_heads, num_levels, num_point, 2}};
  gert::StorageShape attention_weights_shape = {{batch, num_query, num_heads, num_levels, num_point}, {batch, num_query, num_heads, num_levels, num_point}};
  gert::StorageShape output_shape = {{batch, num_query, num_heads * channels}, {batch, num_query, num_heads * channels}};

  auto holder = gert::TilingContextFaker()
                    .NodeIoNum(5, 1)
                    .IrInstanceNum({1, 1, 1, 1, 1})
                    .InputShapes({&value_shape, &value_spatial_shapes_shape, &value_level_start_index_shape, &sampling_locations_shape, &attention_weights_shape})
                    .OutputShapes({&output_shape})
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                    .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
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
  auto tiling_data_result = TilingData2Str(tiling_context->GetRawTilingData());
  std::cout << tiling_data_result << std::endl;
}

TEST_F(MultiScaleDeformableAttnFunctionTiling, MultiScaleDeformableAttnFunction_tiling_generic_numHeads) {
  std::string op_type("MultiScaleDeformableAttnFunction");
  string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 48}
                          })";

  map<string, string> soc_infos;
  map<string, string> aicore_spec;
  map<string, string> intrinsics;
  GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

  // platform info
  fe::PlatFormInfos platform_info;
  platform_info.Init();
  struct MultiScaleDeformableAttnFunctionCompileInfo {
    int32_t total_core_num = 0;
    uint64_t ub_size_platform = 0;
    bool is_inf_base = false;
  } compile_info;


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

  uint64_t batch = 1;
  uint64_t spatial_size = 24;
  uint64_t num_heads = 32;
  uint64_t channels = 32;
  uint64_t num_levels = 1;
  uint64_t num_query = 100;
  uint64_t num_point = 2;
  gert::StorageShape value_shape = {{batch, spatial_size, num_heads, channels}, {batch, spatial_size, num_heads, channels}};
  gert::StorageShape value_spatial_shapes_shape = {{num_levels, 2}, {num_levels, 2}};
  gert::StorageShape value_level_start_index_shape = {{num_levels}, {num_levels}};
  gert::StorageShape sampling_locations_shape = {{batch, num_query, num_heads, num_levels, num_point, 2},
    {batch, num_query, num_heads, num_levels, num_point, 2}};
  gert::StorageShape attention_weights_shape = {{batch, num_query, num_heads, num_levels, num_point}, {batch, num_query, num_heads, num_levels, num_point}};
  gert::StorageShape output_shape = {{batch, num_query, num_heads * channels}, {batch, num_query, num_heads * channels}};

  auto holder = gert::TilingContextFaker()
                    .NodeIoNum(5, 1)
                    .IrInstanceNum({1, 1, 1, 1, 1})
                    .InputShapes({&value_shape, &value_spatial_shapes_shape, &value_level_start_index_shape, &sampling_locations_shape, &attention_weights_shape})
                    .OutputShapes({&output_shape})
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                    .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
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
  EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
  // todo check tiling result
  auto tiling_key = tiling_context->GetTilingKey();
  auto tiling_data_result = TilingData2Str(tiling_context->GetRawTilingData());
  std::cout << tiling_data_result << std::endl;
}

TEST_F(MultiScaleDeformableAttnFunctionTiling, MultiScaleDeformableAttnFunction_tiling_generic_numLevels) {
  std::string op_type("MultiScaleDeformableAttnFunction");
  string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 48}
                          })";

  map<string, string> soc_infos;
  map<string, string> aicore_spec;
  map<string, string> intrinsics;
  GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

  // platform info
  fe::PlatFormInfos platform_info;
  platform_info.Init();
  struct MultiScaleDeformableAttnFunctionCompileInfo {
    int32_t total_core_num = 0;
    uint64_t ub_size_platform = 0;
    bool is_inf_base = false;
  } compile_info;


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

  uint64_t batch = 1;
  uint64_t spatial_size = 24;
  uint64_t num_heads = 8;
  uint64_t channels = 32;
  uint64_t num_levels = 32;
  uint64_t num_query = 100;
  uint64_t num_point = 2;
  gert::StorageShape value_shape = {{batch, spatial_size, num_heads, channels}, {batch, spatial_size, num_heads, channels}};
  gert::StorageShape value_spatial_shapes_shape = {{num_levels, 2}, {num_levels, 2}};
  gert::StorageShape value_level_start_index_shape = {{num_levels}, {num_levels}};
  gert::StorageShape sampling_locations_shape = {{batch, num_query, num_heads, num_levels, num_point, 2},
    {batch, num_query, num_heads, num_levels, num_point, 2}};
  gert::StorageShape attention_weights_shape = {{batch, num_query, num_heads, num_levels, num_point}, {batch, num_query, num_heads, num_levels, num_point}};
  gert::StorageShape output_shape = {{batch, num_query, num_heads * channels}, {batch, num_query, num_heads * channels}};

  auto holder = gert::TilingContextFaker()
                    .NodeIoNum(5, 1)
                    .IrInstanceNum({1, 1, 1, 1, 1})
                    .InputShapes({&value_shape, &value_spatial_shapes_shape, &value_level_start_index_shape, &sampling_locations_shape, &attention_weights_shape})
                    .OutputShapes({&output_shape})
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                    .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
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
  EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
  // todo check tiling result
  auto tiling_key = tiling_context->GetTilingKey();
  auto tiling_data_result = TilingData2Str(tiling_context->GetRawTilingData());
  std::cout << tiling_data_result << std::endl;
}

TEST_F(MultiScaleDeformableAttnFunctionTiling, MultiScaleDeformableAttnFunction_tiling_generic_numPoint) {
  std::string op_type("MultiScaleDeformableAttnFunction");
  string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 48}
                          })";

  map<string, string> soc_infos;
  map<string, string> aicore_spec;
  map<string, string> intrinsics;
  GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

  // platform info
  fe::PlatFormInfos platform_info;
  platform_info.Init();
  struct MultiScaleDeformableAttnFunctionCompileInfo {
    int32_t total_core_num = 0;
    uint64_t ub_size_platform = 0;
    bool is_inf_base = false;
  } compile_info;


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

  uint64_t batch = 1;
  uint64_t spatial_size = 24;
  uint64_t num_heads = 8;
  uint64_t channels = 32;
  uint64_t num_levels = 1;
  uint64_t num_query = 100;
  uint64_t num_point = 32;
  gert::StorageShape value_shape = {{batch, spatial_size, num_heads, channels}, {batch, spatial_size, num_heads, channels}};
  gert::StorageShape value_spatial_shapes_shape = {{num_levels, 2}, {num_levels, 2}};
  gert::StorageShape value_level_start_index_shape = {{num_levels}, {num_levels}};
  gert::StorageShape sampling_locations_shape = {{batch, num_query, num_heads, num_levels, num_point, 2},
    {batch, num_query, num_heads, num_levels, num_point, 2}};
  gert::StorageShape attention_weights_shape = {{batch, num_query, num_heads, num_levels, num_point}, {batch, num_query, num_heads, num_levels, num_point}};
  gert::StorageShape output_shape = {{batch, num_query, num_heads * channels}, {batch, num_query, num_heads * channels}};

  auto holder = gert::TilingContextFaker()
                    .NodeIoNum(5, 1)
                    .IrInstanceNum({1, 1, 1, 1, 1})
                    .InputShapes({&value_shape, &value_spatial_shapes_shape, &value_level_start_index_shape, &sampling_locations_shape, &attention_weights_shape})
                    .OutputShapes({&output_shape})
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                    .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
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
  EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
  // todo check tiling result
  auto tiling_key = tiling_context->GetTilingKey();
  auto tiling_data_result = TilingData2Str(tiling_context->GetRawTilingData());
  std::cout << tiling_data_result << std::endl;
}

TEST_F(MultiScaleDeformableAttnFunctionTiling, MultiScaleDeformableAttnFunction_tiling_trans) {
  std::string op_type("MultiScaleDeformableAttnFunction");
  string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 48}
                          })";

  map<string, string> soc_infos;
  map<string, string> aicore_spec;
  map<string, string> intrinsics;
  GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

  // platform info
  fe::PlatFormInfos platform_info;
  platform_info.Init();
  struct MultiScaleDeformableAttnFunctionCompileInfo {
    int32_t total_core_num = 0;
    uint64_t ub_size_platform = 0;
    bool is_inf_base = false;
  } compile_info;


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

  uint64_t batch = 1;
  uint64_t spatial_size = 24;
  uint64_t num_heads = 8;
  uint64_t channels = 32;
  uint64_t num_levels = 1;
  uint64_t num_query = 100;
  uint64_t num_point = 2;
  gert::StorageShape value_shape = {{batch, num_heads, spatial_size, channels}, {batch, num_heads, spatial_size, channels}};
  gert::StorageShape value_spatial_shapes_shape = {{num_levels, 2}, {num_levels, 2}};
  gert::StorageShape value_level_start_index_shape = {{num_levels}, {num_levels}};
  gert::StorageShape sampling_locations_shape = {{batch, num_heads, num_levels, num_point, 2, num_query},
    {batch, num_heads, num_levels, num_point, 2, num_query}};
  gert::StorageShape attention_weights_shape = {{batch, num_heads, num_levels, num_point, num_query}, {batch, num_heads, num_levels, num_point, num_query}};
  gert::StorageShape output_shape = {{batch, num_heads * channels, num_query}, {batch, num_heads * channels, num_query}};

  auto holder = gert::TilingContextFaker()
                    .NodeIoNum(5, 1)
                    .IrInstanceNum({1, 1, 1, 1, 1})
                    .InputShapes({&value_shape, &value_spatial_shapes_shape, &value_level_start_index_shape, &sampling_locations_shape, &attention_weights_shape})
                    .OutputShapes({&output_shape})
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                    .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
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
  auto tiling_data_result = TilingData2Str(tiling_context->GetRawTilingData());
  std::cout << tiling_data_result << std::endl;
}

TEST_F(MultiScaleDeformableAttnFunctionTiling, MultiScaleDeformableAttnFunction_tiling_generic_numQueries) {
  std::string op_type("MultiScaleDeformableAttnFunction");
  string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 48}
                          })";

  map<string, string> soc_infos;
  map<string, string> aicore_spec;
  map<string, string> intrinsics;
  GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

  // platform info
  fe::PlatFormInfos platform_info;
  platform_info.Init();
  struct MultiScaleDeformableAttnFunctionCompileInfo {
    int32_t total_core_num = 0;
    uint64_t ub_size_platform = 0;
    bool is_inf_base = false;
  } compile_info;


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

  uint64_t batch = 1;
  uint64_t spatial_size = 24;
  uint64_t num_heads = 8;
  uint64_t channels = 32;
  uint64_t num_levels = 1;
  uint64_t num_query = 16;
  uint64_t num_point = 4;
  gert::StorageShape value_shape = {{batch, spatial_size, num_heads, channels}, {batch, spatial_size, num_heads, channels}};
  gert::StorageShape value_spatial_shapes_shape = {{num_levels, 2}, {num_levels, 2}};
  gert::StorageShape value_level_start_index_shape = {{num_levels}, {num_levels}};
  gert::StorageShape sampling_locations_shape = {{batch, num_query, num_heads, num_levels, num_point, 2},
    {batch, num_query, num_heads, num_levels, num_point, 2}};
  gert::StorageShape attention_weights_shape = {{batch, num_query, num_heads, num_levels, num_point}, {batch, num_query, num_heads, num_levels, num_point}};
  gert::StorageShape output_shape = {{batch, num_query, num_heads * channels}, {batch, num_query, num_heads * channels}};

  auto holder = gert::TilingContextFaker()
                    .NodeIoNum(5, 1)
                    .IrInstanceNum({1, 1, 1, 1, 1})
                    .InputShapes({&value_shape, &value_spatial_shapes_shape, &value_level_start_index_shape, &sampling_locations_shape, &attention_weights_shape})
                    .OutputShapes({&output_shape})
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                    .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
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
  EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
  // todo check tiling result
  auto tiling_key = tiling_context->GetTilingKey();
  auto tiling_data_result = TilingData2Str(tiling_context->GetRawTilingData());
  std::cout << tiling_data_result << std::endl;
}

TEST_F(MultiScaleDeformableAttnFunctionTiling, MultiScaleDeformableAttnFunction_tiling_generic_embedDims) {
  std::string op_type("MultiScaleDeformableAttnFunction");
  string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 48}
                          })";

  map<string, string> soc_infos;
  map<string, string> aicore_spec;
  map<string, string> intrinsics;
  GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

  // platform info
  fe::PlatFormInfos platform_info;
  platform_info.Init();
  struct MultiScaleDeformableAttnFunctionCompileInfo {
    int32_t total_core_num = 0;
    uint64_t ub_size_platform = 0;
    bool is_inf_base = false;
  } compile_info;


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

  uint64_t batch = 1;
  uint64_t spatial_size = 24;
  uint64_t num_heads = 8;
  uint64_t channels = 12;
  uint64_t num_levels = 1;
  uint64_t num_query = 32;
  uint64_t num_point = 4;
  gert::StorageShape value_shape = {{batch, spatial_size, num_heads, channels}, {batch, spatial_size, num_heads, channels}};
  gert::StorageShape value_spatial_shapes_shape = {{num_levels, 2}, {num_levels, 2}};
  gert::StorageShape value_level_start_index_shape = {{num_levels}, {num_levels}};
  gert::StorageShape sampling_locations_shape = {{batch, num_query, num_heads, num_levels, num_point, 2},
    {batch, num_query, num_heads, num_levels, num_point, 2}};
  gert::StorageShape attention_weights_shape = {{batch, num_query, num_heads, num_levels, num_point}, {batch, num_query, num_heads, num_levels, num_point}};
  gert::StorageShape output_shape = {{batch, num_query, num_heads * channels}, {batch, num_query, num_heads * channels}};

  auto holder = gert::TilingContextFaker()
                    .NodeIoNum(5, 1)
                    .IrInstanceNum({1, 1, 1, 1, 1})
                    .InputShapes({&value_shape, &value_spatial_shapes_shape, &value_level_start_index_shape, &sampling_locations_shape, &attention_weights_shape})
                    .OutputShapes({&output_shape})
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                    .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
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
  EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
  // todo check tiling result
  auto tiling_key = tiling_context->GetTilingKey();
  auto tiling_data_result = TilingData2Str(tiling_context->GetRawTilingData());
  std::cout << tiling_data_result << std::endl;
}

TEST_F(MultiScaleDeformableAttnFunctionTiling, MultiScaleDeformableAttnFunction_tiling_key_0) {
  std::string op_type("MultiScaleDeformableAttnFunction");
  string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 48}
                          })";

  map<string, string> soc_infos;
  map<string, string> aicore_spec;
  map<string, string> intrinsics;
  GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

  // platform info
  fe::PlatFormInfos platform_info;
  platform_info.Init();
  struct MultiScaleDeformableAttnFunctionCompileInfo {
    int32_t total_core_num = 0;
    uint64_t ub_size_platform = 0;
    bool is_inf_base = false;
  } compile_info;


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

  uint64_t batch = 1;
  uint64_t spatial_size = 24;
  uint64_t num_heads = 8;
  uint64_t channels = 32;
  uint64_t num_levels = 1;
  uint64_t num_query = 100;
  uint64_t num_point = 3;
  gert::StorageShape value_shape = {{batch, spatial_size, num_heads, channels}, {batch, spatial_size, num_heads, channels}};
  gert::StorageShape value_spatial_shapes_shape = {{num_levels, 2}, {num_levels, 2}};
  gert::StorageShape value_level_start_index_shape = {{num_levels}, {num_levels}};
  gert::StorageShape sampling_locations_shape = {{batch, num_query, num_heads, num_levels, num_point, 2},
    {batch, num_query, num_heads, num_levels, num_point, 2}};
  gert::StorageShape attention_weights_shape = {{batch, num_query, num_heads, num_levels, num_point}, {batch, num_query, num_heads, num_levels, num_point}};
  gert::StorageShape output_shape = {{batch, num_query, num_heads * channels}, {batch, num_query, num_heads * channels}};

  auto holder = gert::TilingContextFaker()
                    .NodeIoNum(5, 1)
                    .IrInstanceNum({1, 1, 1, 1, 1})
                    .InputShapes({&value_shape, &value_spatial_shapes_shape, &value_level_start_index_shape, &sampling_locations_shape, &attention_weights_shape})
                    .OutputShapes({&output_shape})
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                    .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
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
  auto tiling_data_result = TilingData2Str(tiling_context->GetRawTilingData());
  std::cout << tiling_data_result << std::endl;
}
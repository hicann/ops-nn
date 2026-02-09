/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_scatter_tiling.cpp
 * \brief
 */
#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#include "log/log.h"

#include "common/kernel_run_context_facker.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "array_ops.h"
#include "common/ut_op_util.h"
#include "common/test_cube_util.h"
#include "../../../../op_host/arch35/scatter_tiling.h"


using namespace std;

class ScatterTiling : public testing::Test
{
protected:
  static void SetUpTestCase()
  {
    std::cout << "ScatterTiling SetUp" << std::endl;
  }

  static void TearDownTestCase()
  {
    std::cout << "ScatterTiling TearDown" << std::endl;
  }
};

using namespace ge;
using namespace ut_util;

TEST_F(ScatterTiling, scatter_axis_2) {
  gert::StorageShape data_shape = {{8, 4, 4096, 256}, {8, 4, 4096, 256}};//{8, 4, 4096, 256};
  gert::StorageShape indices_shape = {{8}, {8}};
  gert::StorageShape update_shape = {{8, 4, 1, 256}, {8, 4, 1, 256}};
  int64_t axis = -2;
  string reduction = "none";
  std::vector<gert::StorageShape> output_shapes(1, {{8, 4, 4096, 256}, {8, 4, 4096, 256}});

  std::vector<void *> output_shapes_ref(1);
  for (size_t i = 0; i < output_shapes.size(); ++i) {
    output_shapes_ref[i] = &output_shapes[i];
  }

  string platform_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "cube_core_cnt": 24, "vector_core_cnt": 48, "core_type_list": "CubeCore,VectorCore"}
                          })";
  map<string, string> soc_infos;
  map<string, string> aicore_spec;
  map<string, string> intrinsics;
  GetPlatFormInfos(platform_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

  // platform info
  fe::PlatFormInfos platform_info;
  platform_info.Init();
  // compile info
  struct ScatterKvCompileInfo {
    int64_t core_num;
    int64_t ub_size;
    bool is910_95;
    bool support_data_move_pad;
  };
  ScatterKvCompileInfo compile_info = {48, 196608, false, true};

  std::string op_type("Scatter");
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
  auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
  auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

  // generate TilingParseContext to do TilingPrepareForScatter
  auto kernel_holder = gert::KernelRunContextFaker()
                    .KernelIONum(2, 1)
                    .Inputs({const_cast<char *>(platform_info_string.c_str()), reinterpret_cast<void *>(&platform_info)})
                    .Outputs({&compile_info})
                    .Build();

  ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("VectorCore");
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

  // generate TilingContext to do TilingScatter
  auto param = gert::TilingData::CreateCap(4096);
  auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
  auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
  ASSERT_NE(param, nullptr);
  auto holder = gert::TilingContextFaker()
                    .SetOpType(op_type)
                    .NodeIoNum(3, 1)
                    .IrInstanceNum({1, 1, 1})
                    .InputShapes({&data_shape, &indices_shape, &update_shape})
                    .OutputShapes(output_shapes_ref)
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                    .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({
                      {"reduction", ge::AnyValue::CreateFrom<string>(reduction)},
                      {"axis", ge::AnyValue::CreateFrom<int64_t>(axis)}
                      })
                    .TilingData(param.get())
                    .Workspace(ws_size)
                    .Build();

  gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
  ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("VectorCore");
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

  EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
}

TEST_F(ScatterTiling, scatter_axis_2_large_batch) {
  gert::StorageShape data_shape = {{1024, 256, 4096, 256}, {1024, 256, 4096, 256}};//{8, 4, 4096, 256};
  gert::StorageShape indices_shape = {{1024}, {1024}};
  gert::StorageShape update_shape = {{1024, 256, 1, 256}, {1024, 256, 1, 256}};
  int64_t axis = -2;
  string reduction = "none";
  std::vector<gert::StorageShape> output_shapes(1, {{1024, 256, 4096, 256}, {1024, 256, 4096, 256}});

  std::vector<void *> output_shapes_ref(1);
  for (size_t i = 0; i < output_shapes.size(); ++i) {
    output_shapes_ref[i] = &output_shapes[i];
  }

  string platform_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "cube_core_cnt": 24, "vector_core_cnt": 48, "core_type_list": "CubeCore,VectorCore"}
                          })";
  map<string, string> soc_infos;
  map<string, string> aicore_spec;
  map<string, string> intrinsics;
  GetPlatFormInfos(platform_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

  // platform info
  fe::PlatFormInfos platform_info;
  platform_info.Init();
  // compile info
  struct ScatterKvCompileInfo {
    int64_t core_num;
    int64_t ub_size;
    bool is910_95;
    bool support_data_move_pad;
  };
  ScatterKvCompileInfo compile_info = {48, 196608, false, true};

  std::string op_type("Scatter");
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
  auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
  auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

  // generate TilingParseContext to do TilingPrepareForScatter
  auto kernel_holder = gert::KernelRunContextFaker()
                    .KernelIONum(2, 1)
                    .Inputs({const_cast<char *>(platform_info_string.c_str()), reinterpret_cast<void *>(&platform_info)})
                    .Outputs({&compile_info})
                    .Build();

  ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("VectorCore");
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

  // generate TilingContext to do TilingScatter
  auto param = gert::TilingData::CreateCap(4096);
  auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
  auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
  ASSERT_NE(param, nullptr);
  auto holder = gert::TilingContextFaker()
                    .SetOpType(op_type)
                    .NodeIoNum(3, 1)
                    .IrInstanceNum({1, 1, 1})
                    .InputShapes({&data_shape, &indices_shape, &update_shape})
                    .OutputShapes(output_shapes_ref)
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                    .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({
                      {"reduction", ge::AnyValue::CreateFrom<string>(reduction)},
                      {"axis", ge::AnyValue::CreateFrom<int64_t>(axis)}
                      })
                    .TilingData(param.get())
                    .Workspace(ws_size)
                    .Build();

  gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
  ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("VectorCore");
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

  EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
}

TEST_F(ScatterTiling, scatter_axis_2_large_update) {
  gert::StorageShape data_shape = {{1024, 256, 4096, 2560000}, {1024, 256, 4096, 2560000}};
  gert::StorageShape indices_shape = {{1024}, {1024}};
  gert::StorageShape update_shape = {{1024, 256, 1, 2560000}, {1024, 256, 1, 2560000}};
  int64_t axis = -2;
  string reduction = "none";
  std::vector<gert::StorageShape> output_shapes(1, {{1024, 256, 4096, 2560000}, {1024, 256, 4096, 2560000}});

  std::vector<void *> output_shapes_ref(1);
  for (size_t i = 0; i < output_shapes.size(); ++i) {
    output_shapes_ref[i] = &output_shapes[i];
  }

  string platform_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "cube_core_cnt": 24, "vector_core_cnt": 48, "core_type_list": "CubeCore,VectorCore"}
                          })";
  map<string, string> soc_infos;
  map<string, string> aicore_spec;
  map<string, string> intrinsics;
  GetPlatFormInfos(platform_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

  // platform info
  fe::PlatFormInfos platform_info;
  platform_info.Init();
  // compile info
  struct ScatterKvCompileInfo {
    int64_t core_num;
    int64_t ub_size;
    bool is910_95;
    bool support_data_move_pad;
  };
  ScatterKvCompileInfo compile_info = {48, 196608, false, true};

  std::string op_type("Scatter");
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
  auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
  auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

  // generate TilingParseContext to do TilingPrepareForScatter
  auto kernel_holder = gert::KernelRunContextFaker()
                    .KernelIONum(2, 1)
                    .Inputs({const_cast<char *>(platform_info_string.c_str()), reinterpret_cast<void *>(&platform_info)})
                    .Outputs({&compile_info})
                    .Build();

  ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("VectorCore");
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

  // generate TilingContext to do TilingScatter
  auto param = gert::TilingData::CreateCap(4096);
  auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
  auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
  ASSERT_NE(param, nullptr);
  auto holder = gert::TilingContextFaker()
                    .SetOpType(op_type)
                    .NodeIoNum(3, 1)
                    .IrInstanceNum({1, 1, 1})
                    .InputShapes({&data_shape, &indices_shape, &update_shape})
                    .OutputShapes(output_shapes_ref)
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                    .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({
                      {"reduction", ge::AnyValue::CreateFrom<string>(reduction)},
                      {"axis", ge::AnyValue::CreateFrom<int64_t>(axis)}
                      })
                    .TilingData(param.get())
                    .Workspace(ws_size)
                    .Build();

  gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
  ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("VectorCore");
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

  EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
}

TEST_F(ScatterTiling, scatter_axis_3) {
  gert::StorageShape data_shape = {{8, 4, 256, 4096}, {8, 4, 256, 4096}};//{8, 4, 4096, 256};
  gert::StorageShape indices_shape = {{8}, {8}};
  gert::StorageShape update_shape = {{8, 4, 256, 1}, {8, 4, 256, 1}};
  int64_t axis = -1;
  string reduction = "update";
  std::vector<gert::StorageShape> output_shapes(1, {{8, 4, 256, 4096}, {8, 4, 256, 4096}});

  std::vector<void *> output_shapes_ref(1);
  for (size_t i = 0; i < output_shapes.size(); ++i) {
    output_shapes_ref[i] = &output_shapes[i];
  }

  string platform_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "cube_core_cnt": 24, "vector_core_cnt": 48, "core_type_list": "CubeCore,VectorCore"}
                          })";
  map<string, string> soc_infos;
  map<string, string> aicore_spec;
  map<string, string> intrinsics;
  GetPlatFormInfos(platform_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

  // platform info
  fe::PlatFormInfos platform_info;
  platform_info.Init();
  // compile info
  struct ScatterKvCompileInfo {
    int64_t core_num;
    int64_t ub_size;
    bool is910_95;
    bool support_data_move_pad;
  };
  ScatterKvCompileInfo compile_info = {48, 196608, false, true};

  std::string op_type("Scatter");
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
  auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
  auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

  // generate TilingParseContext to do TilingPrepareForScatter
  auto kernel_holder = gert::KernelRunContextFaker()
                    .KernelIONum(2, 1)
                    .Inputs({const_cast<char *>(platform_info_string.c_str()), reinterpret_cast<void *>(&platform_info)})
                    .Outputs({&compile_info})
                    .Build();

  ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("VectorCore");
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

  // generate TilingContext to do TilingScatter
  auto param = gert::TilingData::CreateCap(4096);
  auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
  auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
  ASSERT_NE(param, nullptr);
  auto holder = gert::TilingContextFaker()
                    .SetOpType(op_type)
                    .NodeIoNum(3, 1)
                    .IrInstanceNum({1, 1, 1})
                    .InputShapes({&data_shape, &indices_shape, &update_shape})
                    .OutputShapes(output_shapes_ref)
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                    .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({
                      {"reduction", ge::AnyValue::CreateFrom<string>(reduction)},
                      {"axis", ge::AnyValue::CreateFrom<int64_t>(axis)}
                      })
                    .TilingData(param.get())
                    .Workspace(ws_size)
                    .Build();

  gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
  ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("VectorCore");
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

  EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
}

TEST_F(ScatterTiling, scatter_axis_3_large_batch) {
  gert::StorageShape data_shape = {{1024, 256, 256, 4096}, {1024, 256, 256, 4096}};//{8, 4, 4096, 256};
  gert::StorageShape indices_shape = {{1024}, {1024}};
  gert::StorageShape update_shape = {{1024, 256, 256, 1}, {1024, 256, 256, 1}};
  int64_t axis = -1;
  string reduction = "update";
  std::vector<gert::StorageShape> output_shapes(1, {{1024, 256, 256, 4096}, {1024, 256, 256, 4096}});

  std::vector<void *> output_shapes_ref(1);
  for (size_t i = 0; i < output_shapes.size(); ++i) {
    output_shapes_ref[i] = &output_shapes[i];
  }

  string platform_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "cube_core_cnt": 24, "vector_core_cnt": 48, "core_type_list": "CubeCore,VectorCore"}
                          })";
  map<string, string> soc_infos;
  map<string, string> aicore_spec;
  map<string, string> intrinsics;
  GetPlatFormInfos(platform_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

  // platform info
  fe::PlatFormInfos platform_info;
  platform_info.Init();
  // compile info
  struct ScatterKvCompileInfo {
    int64_t core_num;
    int64_t ub_size;
    bool is910_95;
    bool support_data_move_pad;
  };
  ScatterKvCompileInfo compile_info = {48, 196608, false, true};

  std::string op_type("Scatter");
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
  auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
  auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

  // generate TilingParseContext to do TilingPrepareForScatter
  auto kernel_holder = gert::KernelRunContextFaker()
                    .KernelIONum(2, 1)
                    .Inputs({const_cast<char *>(platform_info_string.c_str()), reinterpret_cast<void *>(&platform_info)})
                    .Outputs({&compile_info})
                    .Build();

  ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("VectorCore");
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

  // generate TilingContext to do TilingScatter
  auto param = gert::TilingData::CreateCap(4096);
  auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
  auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
  ASSERT_NE(param, nullptr);
  auto holder = gert::TilingContextFaker()
                    .SetOpType(op_type)
                    .NodeIoNum(3, 1)
                    .IrInstanceNum({1, 1, 1})
                    .InputShapes({&data_shape, &indices_shape, &update_shape})
                    .OutputShapes(output_shapes_ref)
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                    .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({
                      {"reduction", ge::AnyValue::CreateFrom<string>(reduction)},
                      {"axis", ge::AnyValue::CreateFrom<int64_t>(axis)}
                      })
                    .TilingData(param.get())
                    .Workspace(ws_size)
                    .Build();

  gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
  ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("VectorCore");
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

  EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
}

TEST_F(ScatterTiling, scatter_axis_3_large_update) {
  gert::StorageShape data_shape = {{1024, 256, 2560000, 4096}, {1024, 256, 2560000, 4096}};//{8, 4, 4096, 256};
  gert::StorageShape indices_shape = {{1024}, {1024}};
  gert::StorageShape update_shape = {{1024, 256, 2560000, 1}, {1024, 256, 2560000, 1}};
  int64_t axis = -1;
  string reduction = "update";
  std::vector<gert::StorageShape> output_shapes(1, {{1024, 256, 2560000, 4096}, {1024, 256, 2560000, 4096}});

  std::vector<void *> output_shapes_ref(1);
  for (size_t i = 0; i < output_shapes.size(); ++i) {
    output_shapes_ref[i] = &output_shapes[i];
  }

  string platform_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "cube_core_cnt": 24, "vector_core_cnt": 48, "core_type_list": "CubeCore,VectorCore"}
                          })";
  map<string, string> soc_infos;
  map<string, string> aicore_spec;
  map<string, string> intrinsics;
  GetPlatFormInfos(platform_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

  // platform info
  fe::PlatFormInfos platform_info;
  platform_info.Init();
  // compile info
  struct ScatterKvCompileInfo {
    int64_t core_num;
    int64_t ub_size;
    bool is910_95;
    bool support_data_move_pad;
  };
  ScatterKvCompileInfo compile_info = {48, 196608, false, true};

  std::string op_type("Scatter");
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
  auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
  auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

  // generate TilingParseContext to do TilingPrepareForScatter
  auto kernel_holder = gert::KernelRunContextFaker()
                    .KernelIONum(2, 1)
                    .Inputs({const_cast<char *>(platform_info_string.c_str()), reinterpret_cast<void *>(&platform_info)})
                    .Outputs({&compile_info})
                    .Build();

  ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("VectorCore");
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

  // generate TilingContext to do TilingScatter
  auto param = gert::TilingData::CreateCap(4096);
  auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
  auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
  ASSERT_NE(param, nullptr);
  auto holder = gert::TilingContextFaker()
                    .SetOpType(op_type)
                    .NodeIoNum(3, 1)
                    .IrInstanceNum({1, 1, 1})
                    .InputShapes({&data_shape, &indices_shape, &update_shape})
                    .OutputShapes(output_shapes_ref)
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                    .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({
                      {"reduction", ge::AnyValue::CreateFrom<string>(reduction)},
                      {"axis", ge::AnyValue::CreateFrom<int64_t>(axis)}
                      })
                    .TilingData(param.get())
                    .Workspace(ws_size)
                    .Build();

  gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
  ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("VectorCore");
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

  EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
}

TEST_F(ScatterTiling, scatter_axis_3_large_update_910) {
  gert::StorageShape data_shape = {{1024, 256, 2560000, 4096}, {1024, 256, 2560000, 4096}};//{8, 4, 4096, 256};
  gert::StorageShape indices_shape = {{1024}, {1024}};
  gert::StorageShape update_shape = {{1024, 256, 2560000, 1}, {1024, 256, 2560000, 1}};
  int64_t axis = -1;
  string reduction = "update";
  std::vector<gert::StorageShape> output_shapes(1, {{1024, 256, 2560000, 4096}, {1024, 256, 2560000, 4096}});

  std::vector<void *> output_shapes_ref(1);
  for (size_t i = 0; i < output_shapes.size(); ++i) {
    output_shapes_ref[i] = &output_shapes[i];
  }

  string platform_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 262144, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "cube_core_cnt": 32, "vector_core_cnt": 32, "core_type_list": "CubeCore,VectorCore"}
                          })";
  map<string, string> soc_infos;
  map<string, string> aicore_spec;
  map<string, string> intrinsics;
  GetPlatFormInfos(platform_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

  // platform info
  fe::PlatFormInfos platform_info;
  platform_info.Init();
  // compile info
  struct ScatterKvCompileInfo {
    int64_t core_num;
    int64_t ub_size;
    bool is910_95;
    bool support_data_move_pad;
  };
  ScatterKvCompileInfo compile_info = {48, 196608, false, true};

  std::string op_type("Scatter");
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
  auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
  auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

  // generate TilingParseContext to do TilingPrepareForScatter
  auto kernel_holder = gert::KernelRunContextFaker()
                    .KernelIONum(2, 1)
                    .Inputs({const_cast<char *>(platform_info_string.c_str()), reinterpret_cast<void *>(&platform_info)})
                    .Outputs({&compile_info})
                    .Build();

  ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("VectorCore");
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

  // generate TilingContext to do TilingScatter
  auto param = gert::TilingData::CreateCap(4096);
  auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
  auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
  ASSERT_NE(param, nullptr);
  auto holder = gert::TilingContextFaker()
                    .SetOpType(op_type)
                    .NodeIoNum(3, 1)
                    .IrInstanceNum({1, 1, 1})
                    .InputShapes({&data_shape, &indices_shape, &update_shape})
                    .OutputShapes(output_shapes_ref)
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                    .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({
                      {"reduction", ge::AnyValue::CreateFrom<string>(reduction)},
                      {"axis", ge::AnyValue::CreateFrom<int64_t>(axis)}
                      })
                    .TilingData(param.get())
                    .Workspace(ws_size)
                    .Build();

  gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
  ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("VectorCore");
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

  EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
}

TEST_F(ScatterTiling, Scattertiling_error) {
  gert::StorageShape data_shape = {{8, 4}, {8, 4}};//{8, 4, 4096, 256};
  gert::StorageShape indices_shape = {{8}, {8}};
  gert::StorageShape update_shape = {{8, 4, 1, 256}, {8, 4, 1, 256}};
  int64_t axis = -2;
  string reduction = "none";
  std::vector<gert::StorageShape> output_shapes(1, {{8, 4, 4096, 256}, {8, 4, 4096, 256}});

  std::vector<void *> output_shapes_ref(1);
  for (size_t i = 0; i < output_shapes.size(); ++i) {
    output_shapes_ref[i] = &output_shapes[i];
  }

  string platform_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "cube_core_cnt": 24, "vector_core_cnt": 48, "core_type_list": "CubeCore,VectorCore"}
                          })";
  map<string, string> soc_infos;
  map<string, string> aicore_spec;
  map<string, string> intrinsics;
  GetPlatFormInfos(platform_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

  // platform info
  fe::PlatFormInfos platform_info;
  platform_info.Init();
  // compile info
  struct ScatterKvCompileInfo {
    int64_t core_num;
    int64_t ub_size;
    bool is910_95;
    bool support_data_move_pad;
  };
  ScatterKvCompileInfo compile_info = {48, 196608, false, true};

  std::string op_type("Scatter");
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
  auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
  auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

  // generate TilingParseContext to do TilingPrepareForScatter
  auto kernel_holder = gert::KernelRunContextFaker()
                    .KernelIONum(2, 1)
                    .Inputs({const_cast<char *>(platform_info_string.c_str()), reinterpret_cast<void *>(&platform_info)})
                    .Outputs({&compile_info})
                    .Build();

  ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("VectorCore");
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

  // generate TilingContext to do TilingScatter
  auto param = gert::TilingData::CreateCap(4096);
  auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
  auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
  ASSERT_NE(param, nullptr);
  auto holder = gert::TilingContextFaker()
                    .SetOpType(op_type)
                    .NodeIoNum(3, 1)
                    .IrInstanceNum({1, 1, 1})
                    .InputShapes({&data_shape, &indices_shape, &update_shape})
                    .OutputShapes(output_shapes_ref)
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                    .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({
                      {"reduction", ge::AnyValue::CreateFrom<string>(reduction)},
                      {"axis", ge::AnyValue::CreateFrom<int64_t>(axis)}
                      })
                    .TilingData(param.get())
                    .Workspace(ws_size)
                    .Build();

  gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
  ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("VectorCore");
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

  EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}


TEST_F(ScatterTiling, Scattertiling_error_2) {
  gert::StorageShape data_shape = {{8, 4, 256, 4096}, {8, 4, 256, 4096}};//{8, 4, 4096, 256};
  gert::StorageShape indices_shape = {{8}, {8}};
  gert::StorageShape update_shape = {{8, 4, 256, 1}, {8, 4, 256, 1}};
  int64_t axis = -4;
  string reduction = "update";
  std::vector<gert::StorageShape> output_shapes(1, {{8, 4, 256, 4096}, {8, 4, 256, 4096}});

  std::vector<void *> output_shapes_ref(1);
  for (size_t i = 0; i < output_shapes.size(); ++i) {
    output_shapes_ref[i] = &output_shapes[i];
  }

  string platform_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "cube_core_cnt": 24, "vector_core_cnt": 48, "core_type_list": "CubeCore,VectorCore"}
                          })";
  map<string, string> soc_infos;
  map<string, string> aicore_spec;
  map<string, string> intrinsics;
  GetPlatFormInfos(platform_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

  // platform info
  fe::PlatFormInfos platform_info;
  platform_info.Init();
  // compile info
  struct ScatterKvCompileInfo {
    int64_t core_num;
    int64_t ub_size;
    bool is910_95;
    bool support_data_move_pad;
  };
  ScatterKvCompileInfo compile_info = {48, 196608, false, true};

  std::string op_type("Scatter");
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
  auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
  auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

  // generate TilingParseContext to do TilingPrepareForScatter
  auto kernel_holder = gert::KernelRunContextFaker()
                    .KernelIONum(2, 1)
                    .Inputs({const_cast<char *>(platform_info_string.c_str()), reinterpret_cast<void *>(&platform_info)})
                    .Outputs({&compile_info})
                    .Build();

  ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("VectorCore");
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

  // generate TilingContext to do TilingScatter
  auto param = gert::TilingData::CreateCap(4096);
  auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
  auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
  ASSERT_NE(param, nullptr);
  auto holder = gert::TilingContextFaker()
                    .SetOpType(op_type)
                    .NodeIoNum(3, 1)
                    .IrInstanceNum({1, 1, 1})
                    .InputShapes({&data_shape, &indices_shape, &update_shape})
                    .OutputShapes(output_shapes_ref)
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                    .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({
                      {"reduction", ge::AnyValue::CreateFrom<string>(reduction)},
                      {"axis", ge::AnyValue::CreateFrom<int64_t>(axis)}
                      })
                    .TilingData(param.get())
                    .Workspace(ws_size)
                    .Build();

  gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
  ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("VectorCore");
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

  EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(ScatterTiling, Scattertiling_prepare) {
  std::string compileInfo = R"({"vars": {"ub_size": 253952, "core_num": 32, "support_data_move_pad": false}})";
  optiling::ScatterKvCompileInfo info;
  EXPECT_EQ(TilingParseTest("Scatter", compileInfo, &info), ge::GRAPH_SUCCESS);
}

TEST_F(ScatterTiling, scatter_axis_2_small_update) {
  gert::StorageShape data_shape = {{8, 1, 256, 4096}, {8, 1, 256, 4096}};//{8, 1, 4096, 256};
  gert::StorageShape indices_shape = {{8}, {8}};
  gert::StorageShape update_shape = {{8, 1, 200, 4096}, {8, 1, 200, 4096}};
  int64_t axis = -2;
  string reduction = "update";
  std::vector<gert::StorageShape> output_shapes(1, {{8, 1, 256, 4096}, {8, 1, 256, 4096}});

  std::vector<void *> output_shapes_ref(1);
  for (size_t i = 0; i < output_shapes.size(); ++i) {
    output_shapes_ref[i] = &output_shapes[i];
  }

  string platform_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "cube_core_cnt": 24, "vector_core_cnt": 48, "core_type_list": "CubeCore,VectorCore"}
                          })";
  map<string, string> soc_infos;
  map<string, string> aicore_spec;
  map<string, string> intrinsics;
  GetPlatFormInfos(platform_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

  // platform info
  fe::PlatFormInfos platform_info;
  platform_info.Init();
  // compile info
  struct ScatterKvCompileInfo {
    int64_t core_num;
    int64_t ub_size;
    bool is910_95;
    bool support_data_move_pad;
  };
  ScatterKvCompileInfo compile_info = {48, 196608, false, true};

  std::string op_type("Scatter");
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
  auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
  auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

  // generate TilingParseContext to do TilingPrepareForScatter
  auto kernel_holder = gert::KernelRunContextFaker()
                    .KernelIONum(2, 1)
                    .Inputs({const_cast<char *>(platform_info_string.c_str()), reinterpret_cast<void *>(&platform_info)})
                    .Outputs({&compile_info})
                    .Build();

  ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("VectorCore");
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

  // generate TilingContext to do TilingScatter
  auto param = gert::TilingData::CreateCap(4096);
  auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
  auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
  ASSERT_NE(param, nullptr);
  auto holder = gert::TilingContextFaker()
                    .SetOpType(op_type)
                    .NodeIoNum(3, 1)
                    .IrInstanceNum({1, 1, 1})
                    .InputShapes({&data_shape, &indices_shape, &update_shape})
                    .OutputShapes(output_shapes_ref)
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                    .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({
                      {"reduction", ge::AnyValue::CreateFrom<string>(reduction)},
                      {"axis", ge::AnyValue::CreateFrom<int64_t>(axis)}
                      })
                    .TilingData(param.get())
                    .Workspace(ws_size)
                    .Build();

  gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
  ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("VectorCore");
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

  EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
}

TEST_F(ScatterTiling, scatter_axis_2_one_update) {
  gert::StorageShape data_shape = {{1, 1, 256, 4096}, {1, 1, 256, 4096}};//{1, 1, 4096, 256};
  gert::StorageShape indices_shape = {{1}, {1}};
  gert::StorageShape update_shape = {{1, 1, 200, 4096}, {1, 1, 200, 4096}};
  int64_t axis = -2;
  string reduction = "update";
  std::vector<gert::StorageShape> output_shapes(1, {{1, 1, 256, 4096}, {1, 1, 256, 4096}});

  std::vector<void *> output_shapes_ref(1);
  for (size_t i = 0; i < output_shapes.size(); ++i) {
    output_shapes_ref[i] = &output_shapes[i];
  }

  string platform_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "cube_core_cnt": 24, "vector_core_cnt": 48, "core_type_list": "CubeCore,VectorCore"}
                          })";
  map<string, string> soc_infos;
  map<string, string> aicore_spec;
  map<string, string> intrinsics;
  GetPlatFormInfos(platform_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

  // platform info
  fe::PlatFormInfos platform_info;
  platform_info.Init();
  // compile info
  struct ScatterKvCompileInfo {
    int64_t core_num;
    int64_t ub_size;
    bool is910_95;
    bool support_data_move_pad;
  };
  ScatterKvCompileInfo compile_info = {48, 196608, false, true};

  std::string op_type("Scatter");
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
  auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
  auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

  // generate TilingParseContext to do TilingPrepareForScatter
  auto kernel_holder = gert::KernelRunContextFaker()
                    .KernelIONum(2, 1)
                    .Inputs({const_cast<char *>(platform_info_string.c_str()), reinterpret_cast<void *>(&platform_info)})
                    .Outputs({&compile_info})
                    .Build();

  ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("VectorCore");
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

  // generate TilingContext to do TilingScatter
  auto param = gert::TilingData::CreateCap(4096);
  auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
  auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
  ASSERT_NE(param, nullptr);
  auto holder = gert::TilingContextFaker()
                    .SetOpType(op_type)
                    .NodeIoNum(3, 1)
                    .IrInstanceNum({1, 1, 1})
                    .InputShapes({&data_shape, &indices_shape, &update_shape})
                    .OutputShapes(output_shapes_ref)
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                    .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({
                      {"reduction", ge::AnyValue::CreateFrom<string>(reduction)},
                      {"axis", ge::AnyValue::CreateFrom<int64_t>(axis)}
                      })
                    .TilingData(param.get())
                    .Workspace(ws_size)
                    .Build();

  gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
  ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("VectorCore");
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

  EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
}

TEST_F(ScatterTiling, scatter_tilinfg_110) {
  gert::StorageShape data_shape = {{4, 2048, 16, 128}, {4, 2048, 16, 128}};//{1, 1, 4096, 256};
  gert::StorageShape indices_shape = {{4}, {4}};
  gert::StorageShape update_shape = {{4, 1024, 16, 128}, {4, 1024, 16, 128}};
  int64_t axis = 1;
  string reduction = "update";
  std::vector<gert::StorageShape> output_shapes(1, {{4, 2048, 16, 128}, {4, 2048, 16, 128}});

  std::vector<void *> output_shapes_ref(1);
  for (size_t i = 0; i < output_shapes.size(); ++i) {
    output_shapes_ref[i] = &output_shapes[i];
  }

  string platform_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 262144, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "cube_core_cnt": 32, "vector_core_cnt": 32, "core_type_list": "CubeCore,VectorCore"}
                          })";
  map<string, string> soc_infos;
  map<string, string> aicore_spec;
  map<string, string> intrinsics;
  GetPlatFormInfos(platform_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

  // platform info
  fe::PlatFormInfos platform_info;
  platform_info.Init();
  // compile info
  struct ScatterKvCompileInfo {
    int64_t core_num;
    int64_t ub_size;
    bool is910_95;
    bool support_data_move_pad;
  };
  ScatterKvCompileInfo compile_info = {8, 262144, false, false};  // compile info

  std::string op_type("Scatter");
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
  auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
  auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

  // generate TilingParseContext to do TilingPrepareForScatter
  auto kernel_holder = gert::KernelRunContextFaker()
                    .KernelIONum(2, 1)
                    .Inputs({const_cast<char *>(platform_info_string.c_str()), reinterpret_cast<void *>(&platform_info)})
                    .Outputs({&compile_info})
                    .Build();

  ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("VectorCore");
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

  // generate TilingContext to do TilingScatter
  auto param = gert::TilingData::CreateCap(4096);
  auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
  auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
  ASSERT_NE(param, nullptr);
  auto holder = gert::TilingContextFaker()
                    .SetOpType(op_type)
                    .NodeIoNum(3, 1)
                    .IrInstanceNum({1, 1, 1})
                    .InputShapes({&data_shape, &indices_shape, &update_shape})
                    .OutputShapes(output_shapes_ref)
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                    .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({
                      {"reduction", ge::AnyValue::CreateFrom<string>(reduction)},
                      {"axis", ge::AnyValue::CreateFrom<int64_t>(axis)}
                      })
                    .TilingData(param.get())
                    .Workspace(ws_size)
                    .Build();

  gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
  ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("VectorCore");
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

  EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
}

TEST_F(ScatterTiling, scatter_BS1_1) {
  gert::StorageShape data_shape = {{24, 4096, 1}, {24, 4096, 1}};
  gert::StorageShape indices_shape = {{1, 2}, {1, 2}};
  gert::StorageShape update_shape = {{1, 1024, 1}, {1, 1024, 1}};
  int64_t axis = 1;
  string reduction = "update";
  std::vector<gert::StorageShape> output_shapes(1, {{24, 4096, 1}, {24, 4096, 1}});

  std::vector<void *> output_shapes_ref(1);
  for (size_t i = 0; i < output_shapes.size(); ++i) {
    output_shapes_ref[i] = &output_shapes[i];
  }

  string platform_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "cube_core_cnt": 24, "vector_core_cnt": 48, "core_type_list": "CubeCore,VectorCore"}
                          })";
  map<string, string> soc_infos;
  map<string, string> aicore_spec;
  map<string, string> intrinsics;
  GetPlatFormInfos(platform_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

  // platform info
  fe::PlatFormInfos platform_info;
  platform_info.Init();
  // compile info
  struct ScatterKvCompileInfo {
    int64_t core_num;
    int64_t ub_size;
    bool is910_95;
    bool support_data_move_pad;
  };
  ScatterKvCompileInfo compile_info = {48, 196608, false, true};

  std::string op_type("Scatter");
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
  auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
  auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

  // generate TilingParseContext to do TilingPrepareForScatter
  auto kernel_holder = gert::KernelRunContextFaker()
                    .KernelIONum(2, 1)
                    .Inputs({const_cast<char *>(platform_info_string.c_str()), reinterpret_cast<void *>(&platform_info)})
                    .Outputs({&compile_info})
                    .Build();

  ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("VectorCore");
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

  // generate TilingContext to do TilingScatter
  auto param = gert::TilingData::CreateCap(4096);
  auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
  auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
  ASSERT_NE(param, nullptr);
  auto holder = gert::TilingContextFaker()
                    .SetOpType(op_type)
                    .NodeIoNum(3, 1)
                    .IrInstanceNum({1, 1, 1})
                    .InputShapes({&data_shape, &indices_shape, &update_shape})
                    .OutputShapes(output_shapes_ref)
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                    .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({
                      {"reduction", ge::AnyValue::CreateFrom<string>(reduction)},
                      {"axis", ge::AnyValue::CreateFrom<int64_t>(axis)}
                      })
                    .TilingData(param.get())
                    .Workspace(ws_size)
                    .Build();

  gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
  ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("VectorCore");
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

  EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
}

TEST_F(ScatterTiling, scatter_BS1_2) {
  gert::StorageShape data_shape = {{24, 4096, 1}, {24, 4096, 1}};
  gert::StorageShape indices_shape = {{24}, {24}};
  gert::StorageShape update_shape = {{24, 1, 1}, {24, 1, 1}};
  int64_t axis = -2;
  string reduction = "update";
  std::vector<gert::StorageShape> output_shapes(1, {{24, 4096, 1}, {24, 4096, 1}});

  std::vector<void *> output_shapes_ref(1);
  for (size_t i = 0; i < output_shapes.size(); ++i) {
    output_shapes_ref[i] = &output_shapes[i];
  }

  string platform_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "cube_core_cnt": 24, "vector_core_cnt": 48, "core_type_list": "CubeCore,VectorCore"}
                          })";
  map<string, string> soc_infos;
  map<string, string> aicore_spec;
  map<string, string> intrinsics;
  GetPlatFormInfos(platform_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

  // platform info
  fe::PlatFormInfos platform_info;
  platform_info.Init();
  // compile info
  struct ScatterKvCompileInfo {
    int64_t core_num;
    int64_t ub_size;
    bool is910_95;
    bool support_data_move_pad;
  };
  ScatterKvCompileInfo compile_info = {48, 196608, false, true};

  std::string op_type("Scatter");
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
  auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
  auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

  // generate TilingParseContext to do TilingPrepareForScatter
  auto kernel_holder = gert::KernelRunContextFaker()
                    .KernelIONum(2, 1)
                    .Inputs({const_cast<char *>(platform_info_string.c_str()), reinterpret_cast<void *>(&platform_info)})
                    .Outputs({&compile_info})
                    .Build();

  ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("VectorCore");
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

  // generate TilingContext to do TilingScatter
  auto param = gert::TilingData::CreateCap(4096);
  auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
  auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
  ASSERT_NE(param, nullptr);
  auto holder = gert::TilingContextFaker()
                    .SetOpType(op_type)
                    .NodeIoNum(3, 1)
                    .IrInstanceNum({1, 1, 1})
                    .InputShapes({&data_shape, &indices_shape, &update_shape})
                    .OutputShapes(output_shapes_ref)
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                    .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({
                      {"reduction", ge::AnyValue::CreateFrom<string>(reduction)},
                      {"axis", ge::AnyValue::CreateFrom<int64_t>(axis)}
                      })
                    .TilingData(param.get())
                    .Workspace(ws_size)
                    .Build();

  gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
  ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("VectorCore");
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

  EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
}

TEST_F(ScatterTiling, scatter_david_axis_2) {
  gert::StorageShape data_shape = {{8, 4, 4096, 256}, {8, 4, 4096, 256}};//{8, 4, 4096, 256};
  gert::StorageShape indices_shape = {{8}, {8}};
  gert::StorageShape update_shape = {{8, 4, 1, 256}, {8, 4, 1, 256}};
  int64_t axis = -2;
  string reduction = "none";
  std::vector<gert::StorageShape> output_shapes(1, {{8, 4, 4096, 256}, {8, 4, 4096, 256}});

  std::vector<void *> output_shapes_ref(1);
  for (size_t i = 0; i < output_shapes.size(); ++i) {
    output_shapes_ref[i] = &output_shapes[i];
  }

  string platform_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "cube_core_cnt": 24, "vector_core_cnt": 48, "core_type_list": "CubeCore,VectorCore"}
                          })";
  map<string, string> soc_infos;
  map<string, string> aicore_spec;
  map<string, string> intrinsics;
  GetPlatFormInfos(platform_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

  // platform info
  fe::PlatFormInfos platform_info;
  platform_info.Init();
  // compile info
  struct ScatterKvCompileInfo {
    int64_t core_num;
    int64_t ub_size;
    bool is910_95;
    bool support_data_move_pad;
  };
  ScatterKvCompileInfo compile_info = {48, 196608, true, true};

  std::string op_type("Scatter");
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
  auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
  auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

  // generate TilingParseContext to do TilingPrepareForScatter
  auto kernel_holder = gert::KernelRunContextFaker()
                    .KernelIONum(2, 1)
                    .Inputs({const_cast<char *>(platform_info_string.c_str()), reinterpret_cast<void *>(&platform_info)})
                    .Outputs({&compile_info})
                    .Build();

  ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("VectorCore");
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

  // generate TilingContext to do TilingScatter
  auto param = gert::TilingData::CreateCap(4096);
  auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
  auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
  ASSERT_NE(param, nullptr);
  auto holder = gert::TilingContextFaker()
                    .SetOpType(op_type)
                    .NodeIoNum(3, 1)
                    .IrInstanceNum({1, 1, 1})
                    .InputShapes({&data_shape, &indices_shape, &update_shape})
                    .OutputShapes(output_shapes_ref)
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                    .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({
                      {"reduction", ge::AnyValue::CreateFrom<string>(reduction)},
                      {"axis", ge::AnyValue::CreateFrom<int64_t>(axis)}
                      })
                    .TilingData(param.get())
                    .Workspace(ws_size)
                    .Build();

  gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
  ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("VectorCore");
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

  EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
}

TEST_F(ScatterTiling, scatter_david_axis_2_large_batch) {
  gert::StorageShape data_shape = {{1024, 256, 4096, 256}, {1024, 256, 4096, 256}};//{8, 4, 4096, 256};
  gert::StorageShape indices_shape = {{1024}, {1024}};
  gert::StorageShape update_shape = {{1024, 256, 1, 256}, {1024, 256, 1, 256}};
  int64_t axis = -2;
  string reduction = "none";
  std::vector<gert::StorageShape> output_shapes(1, {{1024, 256, 4096, 256}, {1024, 256, 4096, 256}});

  std::vector<void *> output_shapes_ref(1);
  for (size_t i = 0; i < output_shapes.size(); ++i) {
    output_shapes_ref[i] = &output_shapes[i];
  }

  string platform_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "cube_core_cnt": 24, "vector_core_cnt": 48, "core_type_list": "CubeCore,VectorCore"}
                          })";
  map<string, string> soc_infos;
  map<string, string> aicore_spec;
  map<string, string> intrinsics;
  GetPlatFormInfos(platform_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

  // platform info
  fe::PlatFormInfos platform_info;
  platform_info.Init();
  // compile info
  struct ScatterKvCompileInfo {
    int64_t core_num;
    int64_t ub_size;
    bool is910_95;
    bool support_data_move_pad;
  };
  ScatterKvCompileInfo compile_info = {48, 196608, true, true};

  std::string op_type("Scatter");
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
  auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
  auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

  // generate TilingParseContext to do TilingPrepareForScatter
  auto kernel_holder = gert::KernelRunContextFaker()
                    .KernelIONum(2, 1)
                    .Inputs({const_cast<char *>(platform_info_string.c_str()), reinterpret_cast<void *>(&platform_info)})
                    .Outputs({&compile_info})
                    .Build();

  ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("VectorCore");
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

  // generate TilingContext to do TilingScatter
  auto param = gert::TilingData::CreateCap(4096);
  auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
  auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
  ASSERT_NE(param, nullptr);
  auto holder = gert::TilingContextFaker()
                    .SetOpType(op_type)
                    .NodeIoNum(3, 1)
                    .IrInstanceNum({1, 1, 1})
                    .InputShapes({&data_shape, &indices_shape, &update_shape})
                    .OutputShapes(output_shapes_ref)
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                    .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({
                      {"reduction", ge::AnyValue::CreateFrom<string>(reduction)},
                      {"axis", ge::AnyValue::CreateFrom<int64_t>(axis)}
                      })
                    .TilingData(param.get())
                    .Workspace(ws_size)
                    .Build();

  gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
  ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("VectorCore");
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

  EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
}

TEST_F(ScatterTiling, scatter_david_axis_2_large_update) {
  gert::StorageShape data_shape = {{1024, 256, 4096, 2560000}, {1024, 256, 4096, 2560000}};
  gert::StorageShape indices_shape = {{1024}, {1024}};
  gert::StorageShape update_shape = {{1024, 256, 1, 2560000}, {1024, 256, 1, 2560000}};
  int64_t axis = -2;
  string reduction = "none";
  std::vector<gert::StorageShape> output_shapes(1, {{1024, 256, 4096, 2560000}, {1024, 256, 4096, 2560000}});

  std::vector<void *> output_shapes_ref(1);
  for (size_t i = 0; i < output_shapes.size(); ++i) {
    output_shapes_ref[i] = &output_shapes[i];
  }

  string platform_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "cube_core_cnt": 24, "vector_core_cnt": 48, "core_type_list": "CubeCore,VectorCore"}
                          })";
  map<string, string> soc_infos;
  map<string, string> aicore_spec;
  map<string, string> intrinsics;
  GetPlatFormInfos(platform_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

  // platform info
  fe::PlatFormInfos platform_info;
  platform_info.Init();
  // compile info
  struct ScatterKvCompileInfo {
    int64_t core_num;
    int64_t ub_size;
    bool is910_95;
    bool support_data_move_pad;
  };
  ScatterKvCompileInfo compile_info = {48, 196608, true, true};

  std::string op_type("Scatter");
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
  auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
  auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

  // generate TilingParseContext to do TilingPrepareForScatter
  auto kernel_holder = gert::KernelRunContextFaker()
                    .KernelIONum(2, 1)
                    .Inputs({const_cast<char *>(platform_info_string.c_str()), reinterpret_cast<void *>(&platform_info)})
                    .Outputs({&compile_info})
                    .Build();

  ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("VectorCore");
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

  // generate TilingContext to do TilingScatter
  auto param = gert::TilingData::CreateCap(4096);
  auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
  auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
  ASSERT_NE(param, nullptr);
  auto holder = gert::TilingContextFaker()
                    .SetOpType(op_type)
                    .NodeIoNum(3, 1)
                    .IrInstanceNum({1, 1, 1})
                    .InputShapes({&data_shape, &indices_shape, &update_shape})
                    .OutputShapes(output_shapes_ref)
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                    .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({
                      {"reduction", ge::AnyValue::CreateFrom<string>(reduction)},
                      {"axis", ge::AnyValue::CreateFrom<int64_t>(axis)}
                      })
                    .TilingData(param.get())
                    .Workspace(ws_size)
                    .Build();

  gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
  ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("VectorCore");
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

  EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
}

TEST_F(ScatterTiling, scatter_david_axis_3) {
  gert::StorageShape data_shape = {{8, 4, 256, 4096}, {8, 4, 256, 4096}};//{8, 4, 4096, 256};
  gert::StorageShape indices_shape = {{8}, {8}};
  gert::StorageShape update_shape = {{8, 4, 256, 1}, {8, 4, 256, 1}};
  int64_t axis = -1;
  string reduction = "update";
  std::vector<gert::StorageShape> output_shapes(1, {{8, 4, 256, 4096}, {8, 4, 256, 4096}});

  std::vector<void *> output_shapes_ref(1);
  for (size_t i = 0; i < output_shapes.size(); ++i) {
    output_shapes_ref[i] = &output_shapes[i];
  }

  string platform_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "cube_core_cnt": 24, "vector_core_cnt": 48, "core_type_list": "CubeCore,VectorCore"}
                          })";
  map<string, string> soc_infos;
  map<string, string> aicore_spec;
  map<string, string> intrinsics;
  GetPlatFormInfos(platform_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

  // platform info
  fe::PlatFormInfos platform_info;
  platform_info.Init();
  // compile info
  struct ScatterKvCompileInfo {
    int64_t core_num;
    int64_t ub_size;
    bool is910_95;
    bool support_data_move_pad;
  };
  ScatterKvCompileInfo compile_info = {48, 196608, true, true};

  std::string op_type("Scatter");
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
  auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
  auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

  // generate TilingParseContext to do TilingPrepareForScatter
  auto kernel_holder = gert::KernelRunContextFaker()
                    .KernelIONum(2, 1)
                    .Inputs({const_cast<char *>(platform_info_string.c_str()), reinterpret_cast<void *>(&platform_info)})
                    .Outputs({&compile_info})
                    .Build();

  ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("VectorCore");
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

  // generate TilingContext to do TilingScatter
  auto param = gert::TilingData::CreateCap(4096);
  auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
  auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
  ASSERT_NE(param, nullptr);
  auto holder = gert::TilingContextFaker()
                    .SetOpType(op_type)
                    .NodeIoNum(3, 1)
                    .IrInstanceNum({1, 1, 1})
                    .InputShapes({&data_shape, &indices_shape, &update_shape})
                    .OutputShapes(output_shapes_ref)
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                    .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({
                      {"reduction", ge::AnyValue::CreateFrom<string>(reduction)},
                      {"axis", ge::AnyValue::CreateFrom<int64_t>(axis)}
                      })
                    .TilingData(param.get())
                    .Workspace(ws_size)
                    .Build();

  gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
  ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("VectorCore");
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

  EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
}

TEST_F(ScatterTiling, scatter_david_axis_3_large_batch) {
  gert::StorageShape data_shape = {{1024, 256, 256, 4096}, {1024, 256, 256, 4096}};//{8, 4, 4096, 256};
  gert::StorageShape indices_shape = {{1024}, {1024}};
  gert::StorageShape update_shape = {{1024, 256, 256, 1}, {1024, 256, 256, 1}};
  int64_t axis = -1;
  string reduction = "update";
  std::vector<gert::StorageShape> output_shapes(1, {{1024, 256, 256, 4096}, {1024, 256, 256, 4096}});

  std::vector<void *> output_shapes_ref(1);
  for (size_t i = 0; i < output_shapes.size(); ++i) {
    output_shapes_ref[i] = &output_shapes[i];
  }

  string platform_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "cube_core_cnt": 24, "vector_core_cnt": 48, "core_type_list": "CubeCore,VectorCore"}
                          })";
  map<string, string> soc_infos;
  map<string, string> aicore_spec;
  map<string, string> intrinsics;
  GetPlatFormInfos(platform_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

  // platform info
  fe::PlatFormInfos platform_info;
  platform_info.Init();
  // compile info
  struct ScatterKvCompileInfo {
    int64_t core_num;
    int64_t ub_size;
    bool is910_95;
    bool support_data_move_pad;
  };
  ScatterKvCompileInfo compile_info = {48, 196608, true, true};

  std::string op_type("Scatter");
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
  auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
  auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

  // generate TilingParseContext to do TilingPrepareForScatter
  auto kernel_holder = gert::KernelRunContextFaker()
                    .KernelIONum(2, 1)
                    .Inputs({const_cast<char *>(platform_info_string.c_str()), reinterpret_cast<void *>(&platform_info)})
                    .Outputs({&compile_info})
                    .Build();

  ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("VectorCore");
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

  // generate TilingContext to do TilingScatter
  auto param = gert::TilingData::CreateCap(4096);
  auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
  auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
  ASSERT_NE(param, nullptr);
  auto holder = gert::TilingContextFaker()
                    .SetOpType(op_type)
                    .NodeIoNum(3, 1)
                    .IrInstanceNum({1, 1, 1})
                    .InputShapes({&data_shape, &indices_shape, &update_shape})
                    .OutputShapes(output_shapes_ref)
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                    .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({
                      {"reduction", ge::AnyValue::CreateFrom<string>(reduction)},
                      {"axis", ge::AnyValue::CreateFrom<int64_t>(axis)}
                      })
                    .TilingData(param.get())
                    .Workspace(ws_size)
                    .Build();

  gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
  ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("VectorCore");
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

  EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
}

TEST_F(ScatterTiling, scatter_david_axis_3_large_update) {
  gert::StorageShape data_shape = {{1024, 256, 2560000, 4096}, {1024, 256, 2560000, 4096}};//{8, 4, 4096, 256};
  gert::StorageShape indices_shape = {{1024}, {1024}};
  gert::StorageShape update_shape = {{1024, 256, 2560000, 1}, {1024, 256, 2560000, 1}};
  int64_t axis = -1;
  string reduction = "update";
  std::vector<gert::StorageShape> output_shapes(1, {{1024, 256, 2560000, 4096}, {1024, 256, 2560000, 4096}});

  std::vector<void *> output_shapes_ref(1);
  for (size_t i = 0; i < output_shapes.size(); ++i) {
    output_shapes_ref[i] = &output_shapes[i];
  }

  string platform_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "cube_core_cnt": 24, "vector_core_cnt": 48, "core_type_list": "CubeCore,VectorCore"}
                          })";
  map<string, string> soc_infos;
  map<string, string> aicore_spec;
  map<string, string> intrinsics;
  GetPlatFormInfos(platform_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

  // platform info
  fe::PlatFormInfos platform_info;
  platform_info.Init();
  // compile info
  struct ScatterKvCompileInfo {
    int64_t core_num;
    int64_t ub_size;
    bool is910_95;
    bool support_data_move_pad;
  };
  ScatterKvCompileInfo compile_info = {48, 196608, true, true};

  std::string op_type("Scatter");
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
  auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
  auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

  // generate TilingParseContext to do TilingPrepareForScatter
  auto kernel_holder = gert::KernelRunContextFaker()
                    .KernelIONum(2, 1)
                    .Inputs({const_cast<char *>(platform_info_string.c_str()), reinterpret_cast<void *>(&platform_info)})
                    .Outputs({&compile_info})
                    .Build();

  ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("VectorCore");
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

  // generate TilingContext to do TilingScatter
  auto param = gert::TilingData::CreateCap(4096);
  auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
  auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
  ASSERT_NE(param, nullptr);
  auto holder = gert::TilingContextFaker()
                    .SetOpType(op_type)
                    .NodeIoNum(3, 1)
                    .IrInstanceNum({1, 1, 1})
                    .InputShapes({&data_shape, &indices_shape, &update_shape})
                    .OutputShapes(output_shapes_ref)
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                    .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({
                      {"reduction", ge::AnyValue::CreateFrom<string>(reduction)},
                      {"axis", ge::AnyValue::CreateFrom<int64_t>(axis)}
                      })
                    .TilingData(param.get())
                    .Workspace(ws_size)
                    .Build();

  gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
  ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("VectorCore");
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

  EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
}

TEST_F(ScatterTiling, scatter_david_axis_3_large_update_910) {
  gert::StorageShape data_shape = {{1024, 256, 2560000, 4096}, {1024, 256, 2560000, 4096}};//{8, 4, 4096, 256};
  gert::StorageShape indices_shape = {{1024}, {1024}};
  gert::StorageShape update_shape = {{1024, 256, 2560000, 1}, {1024, 256, 2560000, 1}};
  int64_t axis = -1;
  string reduction = "update";
  std::vector<gert::StorageShape> output_shapes(1, {{1024, 256, 2560000, 4096}, {1024, 256, 2560000, 4096}});

  std::vector<void *> output_shapes_ref(1);
  for (size_t i = 0; i < output_shapes.size(); ++i) {
    output_shapes_ref[i] = &output_shapes[i];
  }

  string platform_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 262144, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "cube_core_cnt": 32, "vector_core_cnt": 32, "core_type_list": "CubeCore,VectorCore"}
                          })";
  map<string, string> soc_infos;
  map<string, string> aicore_spec;
  map<string, string> intrinsics;
  GetPlatFormInfos(platform_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

  // platform info
  fe::PlatFormInfos platform_info;
  platform_info.Init();
  // compile info
  struct ScatterKvCompileInfo {
    int64_t core_num;
    int64_t ub_size;
    bool is910_95;
    bool support_data_move_pad;
  };
  ScatterKvCompileInfo compile_info = {48, 196608, true, true};

  std::string op_type("Scatter");
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
  auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
  auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

  // generate TilingParseContext to do TilingPrepareForScatter
  auto kernel_holder = gert::KernelRunContextFaker()
                    .KernelIONum(2, 1)
                    .Inputs({const_cast<char *>(platform_info_string.c_str()), reinterpret_cast<void *>(&platform_info)})
                    .Outputs({&compile_info})
                    .Build();

  ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("VectorCore");
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

  // generate TilingContext to do TilingScatter
  auto param = gert::TilingData::CreateCap(4096);
  auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
  auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
  ASSERT_NE(param, nullptr);
  auto holder = gert::TilingContextFaker()
                    .SetOpType(op_type)
                    .NodeIoNum(3, 1)
                    .IrInstanceNum({1, 1, 1})
                    .InputShapes({&data_shape, &indices_shape, &update_shape})
                    .OutputShapes(output_shapes_ref)
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                    .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({
                      {"reduction", ge::AnyValue::CreateFrom<string>(reduction)},
                      {"axis", ge::AnyValue::CreateFrom<int64_t>(axis)}
                      })
                    .TilingData(param.get())
                    .Workspace(ws_size)
                    .Build();

  gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
  ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("VectorCore");
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

  EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
}

TEST_F(ScatterTiling, Scatter_david_tiling_error) {
  gert::StorageShape data_shape = {{8, 4}, {8, 4}};//{8, 4, 4096, 256};
  gert::StorageShape indices_shape = {{8}, {8}};
  gert::StorageShape update_shape = {{8, 4, 1, 256}, {8, 4, 1, 256}};
  int64_t axis = -2;
  string reduction = "none";
  std::vector<gert::StorageShape> output_shapes(1, {{8, 4, 4096, 256}, {8, 4, 4096, 256}});

  std::vector<void *> output_shapes_ref(1);
  for (size_t i = 0; i < output_shapes.size(); ++i) {
    output_shapes_ref[i] = &output_shapes[i];
  }

  string platform_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "cube_core_cnt": 24, "vector_core_cnt": 48, "core_type_list": "CubeCore,VectorCore"}
                          })";
  map<string, string> soc_infos;
  map<string, string> aicore_spec;
  map<string, string> intrinsics;
  GetPlatFormInfos(platform_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

  // platform info
  fe::PlatFormInfos platform_info;
  platform_info.Init();
  // compile info
  struct ScatterKvCompileInfo {
    int64_t core_num;
    int64_t ub_size;
    bool is910_95;
    bool support_data_move_pad;
  };
  ScatterKvCompileInfo compile_info = {48, 196608, true, true};

  std::string op_type("Scatter");
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
  auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
  auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

  // generate TilingParseContext to do TilingPrepareForScatter
  auto kernel_holder = gert::KernelRunContextFaker()
                    .KernelIONum(2, 1)
                    .Inputs({const_cast<char *>(platform_info_string.c_str()), reinterpret_cast<void *>(&platform_info)})
                    .Outputs({&compile_info})
                    .Build();

  ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("VectorCore");
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

  // generate TilingContext to do TilingScatter
  auto param = gert::TilingData::CreateCap(4096);
  auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
  auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
  ASSERT_NE(param, nullptr);
  auto holder = gert::TilingContextFaker()
                    .SetOpType(op_type)
                    .NodeIoNum(3, 1)
                    .IrInstanceNum({1, 1, 1})
                    .InputShapes({&data_shape, &indices_shape, &update_shape})
                    .OutputShapes(output_shapes_ref)
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                    .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({
                      {"reduction", ge::AnyValue::CreateFrom<string>(reduction)},
                      {"axis", ge::AnyValue::CreateFrom<int64_t>(axis)}
                      })
                    .TilingData(param.get())
                    .Workspace(ws_size)
                    .Build();

  gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
  ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("VectorCore");
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

  EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}


TEST_F(ScatterTiling, Scatter_david_tiling_error_2) {
  gert::StorageShape data_shape = {{8, 4, 256, 4096}, {8, 4, 256, 4096}};//{8, 4, 4096, 256};
  gert::StorageShape indices_shape = {{8}, {8}};
  gert::StorageShape update_shape = {{8, 4, 256, 1}, {8, 4, 256, 1}};
  int64_t axis = -4;
  string reduction = "update";
  std::vector<gert::StorageShape> output_shapes(1, {{8, 4, 256, 4096}, {8, 4, 256, 4096}});

  std::vector<void *> output_shapes_ref(1);
  for (size_t i = 0; i < output_shapes.size(); ++i) {
    output_shapes_ref[i] = &output_shapes[i];
  }

  string platform_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "cube_core_cnt": 24, "vector_core_cnt": 48, "core_type_list": "CubeCore,VectorCore"}
                          })";
  map<string, string> soc_infos;
  map<string, string> aicore_spec;
  map<string, string> intrinsics;
  GetPlatFormInfos(platform_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

  // platform info
  fe::PlatFormInfos platform_info;
  platform_info.Init();
  // compile info
  struct ScatterKvCompileInfo {
    int64_t core_num;
    int64_t ub_size;
    bool is910_95;
    bool support_data_move_pad;
  };
  ScatterKvCompileInfo compile_info = {48, 196608, true, true};

  std::string op_type("Scatter");
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
  auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
  auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

  // generate TilingParseContext to do TilingPrepareForScatter
  auto kernel_holder = gert::KernelRunContextFaker()
                    .KernelIONum(2, 1)
                    .Inputs({const_cast<char *>(platform_info_string.c_str()), reinterpret_cast<void *>(&platform_info)})
                    .Outputs({&compile_info})
                    .Build();

  ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("VectorCore");
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

  // generate TilingContext to do TilingScatter
  auto param = gert::TilingData::CreateCap(4096);
  auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
  auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
  ASSERT_NE(param, nullptr);
  auto holder = gert::TilingContextFaker()
                    .SetOpType(op_type)
                    .NodeIoNum(3, 1)
                    .IrInstanceNum({1, 1, 1})
                    .InputShapes({&data_shape, &indices_shape, &update_shape})
                    .OutputShapes(output_shapes_ref)
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                    .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({
                      {"reduction", ge::AnyValue::CreateFrom<string>(reduction)},
                      {"axis", ge::AnyValue::CreateFrom<int64_t>(axis)}
                      })
                    .TilingData(param.get())
                    .Workspace(ws_size)
                    .Build();

  gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
  ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("VectorCore");
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

  EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(ScatterTiling, scatter_david_axis_2_small_update) {
  gert::StorageShape data_shape = {{8, 1, 256, 4096}, {8, 1, 256, 4096}};//{8, 1, 4096, 256};
  gert::StorageShape indices_shape = {{8}, {8}};
  gert::StorageShape update_shape = {{8, 1, 200, 4096}, {8, 1, 200, 4096}};
  int64_t axis = -2;
  string reduction = "update";
  std::vector<gert::StorageShape> output_shapes(1, {{8, 1, 256, 4096}, {8, 1, 256, 4096}});

  std::vector<void *> output_shapes_ref(1);
  for (size_t i = 0; i < output_shapes.size(); ++i) {
    output_shapes_ref[i] = &output_shapes[i];
  }

  string platform_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "cube_core_cnt": 24, "vector_core_cnt": 48, "core_type_list": "CubeCore,VectorCore"}
                          })";
  map<string, string> soc_infos;
  map<string, string> aicore_spec;
  map<string, string> intrinsics;
  GetPlatFormInfos(platform_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

  // platform info
  fe::PlatFormInfos platform_info;
  platform_info.Init();
  // compile info
  struct ScatterKvCompileInfo {
    int64_t core_num;
    int64_t ub_size;
    bool is910_95;
    bool support_data_move_pad;
  };
  ScatterKvCompileInfo compile_info = {48, 196608, true, true};

  std::string op_type("Scatter");
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
  auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
  auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

  // generate TilingParseContext to do TilingPrepareForScatter
  auto kernel_holder = gert::KernelRunContextFaker()
                    .KernelIONum(2, 1)
                    .Inputs({const_cast<char *>(platform_info_string.c_str()), reinterpret_cast<void *>(&platform_info)})
                    .Outputs({&compile_info})
                    .Build();

  ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("VectorCore");
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

  // generate TilingContext to do TilingScatter
  auto param = gert::TilingData::CreateCap(4096);
  auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
  auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
  ASSERT_NE(param, nullptr);
  auto holder = gert::TilingContextFaker()
                    .SetOpType(op_type)
                    .NodeIoNum(3, 1)
                    .IrInstanceNum({1, 1, 1})
                    .InputShapes({&data_shape, &indices_shape, &update_shape})
                    .OutputShapes(output_shapes_ref)
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                    .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({
                      {"reduction", ge::AnyValue::CreateFrom<string>(reduction)},
                      {"axis", ge::AnyValue::CreateFrom<int64_t>(axis)}
                      })
                    .TilingData(param.get())
                    .Workspace(ws_size)
                    .Build();

  gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
  ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("VectorCore");
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

  EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
}

TEST_F(ScatterTiling, scatter_david_axis_2_one_update) {
  gert::StorageShape data_shape = {{1, 1, 256, 4096}, {1, 1, 256, 4096}};//{1, 1, 4096, 256};
  gert::StorageShape indices_shape = {{1}, {1}};
  gert::StorageShape update_shape = {{1, 1, 200, 4096}, {1, 1, 200, 4096}};
  int64_t axis = -2;
  string reduction = "update";
  std::vector<gert::StorageShape> output_shapes(1, {{1, 1, 256, 4096}, {1, 1, 256, 4096}});

  std::vector<void *> output_shapes_ref(1);
  for (size_t i = 0; i < output_shapes.size(); ++i) {
    output_shapes_ref[i] = &output_shapes[i];
  }

  string platform_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "cube_core_cnt": 24, "vector_core_cnt": 48, "core_type_list": "CubeCore,VectorCore"}
                          })";
  map<string, string> soc_infos;
  map<string, string> aicore_spec;
  map<string, string> intrinsics;
  GetPlatFormInfos(platform_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

  // platform info
  fe::PlatFormInfos platform_info;
  platform_info.Init();
  // compile info
  struct ScatterKvCompileInfo {
    int64_t core_num;
    int64_t ub_size;
    bool is910_95;
    bool support_data_move_pad;
  };
  ScatterKvCompileInfo compile_info = {48, 196608, true, true};

  std::string op_type("Scatter");
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
  auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
  auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

  // generate TilingParseContext to do TilingPrepareForScatter
  auto kernel_holder = gert::KernelRunContextFaker()
                    .KernelIONum(2, 1)
                    .Inputs({const_cast<char *>(platform_info_string.c_str()), reinterpret_cast<void *>(&platform_info)})
                    .Outputs({&compile_info})
                    .Build();

  ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("VectorCore");
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

  // generate TilingContext to do TilingScatter
  auto param = gert::TilingData::CreateCap(4096);
  auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
  auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
  ASSERT_NE(param, nullptr);
  auto holder = gert::TilingContextFaker()
                    .SetOpType(op_type)
                    .NodeIoNum(3, 1)
                    .IrInstanceNum({1, 1, 1})
                    .InputShapes({&data_shape, &indices_shape, &update_shape})
                    .OutputShapes(output_shapes_ref)
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                    .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({
                      {"reduction", ge::AnyValue::CreateFrom<string>(reduction)},
                      {"axis", ge::AnyValue::CreateFrom<int64_t>(axis)}
                      })
                    .TilingData(param.get())
                    .Workspace(ws_size)
                    .Build();

  gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
  ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("VectorCore");
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

  EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
}

TEST_F(ScatterTiling, scatter_david_tilinfg_110) {
  gert::StorageShape data_shape = {{4, 2048, 16, 128}, {4, 2048, 16, 128}};//{1, 1, 4096, 256};
  gert::StorageShape indices_shape = {{4}, {4}};
  gert::StorageShape update_shape = {{4, 1024, 16, 128}, {4, 1024, 16, 128}};
  int64_t axis = 1;
  string reduction = "update";
  std::vector<gert::StorageShape> output_shapes(1, {{4, 2048, 16, 128}, {4, 2048, 16, 128}});

  std::vector<void *> output_shapes_ref(1);
  for (size_t i = 0; i < output_shapes.size(); ++i) {
    output_shapes_ref[i] = &output_shapes[i];
  }

  string platform_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 262144, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "cube_core_cnt": 32, "vector_core_cnt": 32, "core_type_list": "CubeCore,VectorCore"}
                          })";
  map<string, string> soc_infos;
  map<string, string> aicore_spec;
  map<string, string> intrinsics;
  GetPlatFormInfos(platform_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

  // platform info
  fe::PlatFormInfos platform_info;
  platform_info.Init();
  // compile info
  struct ScatterKvCompileInfo {
    int64_t core_num;
    int64_t ub_size;
    bool is910_95;
    bool support_data_move_pad;
  };
  ScatterKvCompileInfo compile_info = {8, 262144, true, false};  // compile info

  std::string op_type("Scatter");
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
  auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
  auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

  // generate TilingParseContext to do TilingPrepareForScatter
  auto kernel_holder = gert::KernelRunContextFaker()
                    .KernelIONum(2, 1)
                    .Inputs({const_cast<char *>(platform_info_string.c_str()), reinterpret_cast<void *>(&platform_info)})
                    .Outputs({&compile_info})
                    .Build();

  ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("VectorCore");
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

  // generate TilingContext to do TilingScatter
  auto param = gert::TilingData::CreateCap(4096);
  auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
  auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
  ASSERT_NE(param, nullptr);
  auto holder = gert::TilingContextFaker()
                    .SetOpType(op_type)
                    .NodeIoNum(3, 1)
                    .IrInstanceNum({1, 1, 1})
                    .InputShapes({&data_shape, &indices_shape, &update_shape})
                    .OutputShapes(output_shapes_ref)
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                    .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({
                      {"reduction", ge::AnyValue::CreateFrom<string>(reduction)},
                      {"axis", ge::AnyValue::CreateFrom<int64_t>(axis)}
                      })
                    .TilingData(param.get())
                    .Workspace(ws_size)
                    .Build();

  gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
  ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("VectorCore");
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

  EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
}

TEST_F(ScatterTiling, scatter_david_indices_0_update_2) {
  gert::StorageShape data_shape = {{6, 1, 256, 4096}, {6, 1, 256, 4096}};//{1, 1, 4096, 256};
  gert::StorageShape indices_shape = {};
  gert::StorageShape update_shape = {{2, 1, 200, 4096}, {2, 1, 200, 4096}};
  int64_t axis = -2;
  string reduction = "update";
  std::vector<gert::StorageShape> output_shapes(1, {{6, 1, 256, 4096}, {6, 1, 256, 4096}});

  std::vector<void *> output_shapes_ref(1);
  for (size_t i = 0; i < output_shapes.size(); ++i) {
    output_shapes_ref[i] = &output_shapes[i];
  }

  string platform_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "cube_core_cnt": 24, "vector_core_cnt": 48, "core_type_list": "CubeCore,VectorCore"}
                          })";
  map<string, string> soc_infos;
  map<string, string> aicore_spec;
  map<string, string> intrinsics;
  std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}};
  GetPlatFormInfos(platform_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

  // platform info
  fe::PlatFormInfos platform_info;
  platform_info.Init();
  // compile info
  struct ScatterKvCompileInfo {
    int64_t core_num;
    int64_t ub_size;
    bool is910_95;
    bool support_data_move_pad;
  };
  ScatterKvCompileInfo compile_info = {48, 196608, true, true};

  std::string op_type("Scatter");
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
  auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
  auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

  // generate TilingParseContext to do TilingPrepareForScatter
  auto kernel_holder = gert::KernelRunContextFaker()
                    .KernelIONum(2, 1)
                    .Inputs({const_cast<char *>(platform_info_string.c_str()), reinterpret_cast<void *>(&platform_info)})
                    .Outputs({&compile_info})
                    .Build();

  ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("VectorCore");
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version_infos);

  // generate TilingContext to do TilingScatter
  auto param = gert::TilingData::CreateCap(4096);
  auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
  auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
  ASSERT_NE(param, nullptr);
  auto holder = gert::TilingContextFaker()
                    .SetOpType(op_type)
                    .NodeIoNum(3, 1)
                    .IrInstanceNum({1, 1, 1})
                    .InputShapes({&data_shape, &indices_shape, &update_shape})
                    .OutputShapes(output_shapes_ref)
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                    .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({
                      {"reduction", ge::AnyValue::CreateFrom<string>(reduction)},
                      {"axis", ge::AnyValue::CreateFrom<int64_t>(axis)}
                      })
                    .TilingData(param.get())
                    .Workspace(ws_size)
                    .Build();

  gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
  ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("VectorCore");
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

  EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(ScatterTiling, scatter_david_indices_4_test) {
  gert::StorageShape data_shape = {{1, 1, 256, 4096}, {1, 1, 256, 4096}};//{1, 1, 4096, 256};
  gert::StorageShape indices_shape = {{4, 4}, {4, 4}};
  gert::StorageShape update_shape = {{1, 1, 200, 4096}, {1, 1, 200, 4096}};
  int64_t axis = -2;
  string reduction = "update";
  std::vector<gert::StorageShape> output_shapes(1, {{6, 1, 256, 4096}, {6, 1, 256, 4096}});

  std::vector<void *> output_shapes_ref(1);
  for (size_t i = 0; i < output_shapes.size(); ++i) {
    output_shapes_ref[i] = &output_shapes[i];
  }

  string platform_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "cube_core_cnt": 24, "vector_core_cnt": 48, "core_type_list": "CubeCore,VectorCore"}
                          })";
  map<string, string> soc_infos;
  map<string, string> aicore_spec;
  map<string, string> intrinsics;
  std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}};
  GetPlatFormInfos(platform_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

  // platform info
  fe::PlatFormInfos platform_info;
  platform_info.Init();
  // compile info
  struct ScatterKvCompileInfo {
    int64_t core_num;
    int64_t ub_size;
    bool is910_95;
    bool support_data_move_pad;
  };
  ScatterKvCompileInfo compile_info = {48, 196608, true, true};

  std::string op_type("Scatter");
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
  auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
  auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

  // generate TilingParseContext to do TilingPrepareForScatter
  auto kernel_holder = gert::KernelRunContextFaker()
                    .KernelIONum(2, 1)
                    .Inputs({const_cast<char *>(platform_info_string.c_str()), reinterpret_cast<void *>(&platform_info)})
                    .Outputs({&compile_info})
                    .Build();

  ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("VectorCore");
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version_infos);

  // generate TilingContext to do TilingScatter
  auto param = gert::TilingData::CreateCap(4096);
  auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
  auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
  ASSERT_NE(param, nullptr);
  auto holder = gert::TilingContextFaker()
                    .SetOpType(op_type)
                    .NodeIoNum(3, 1)
                    .IrInstanceNum({1, 1, 1})
                    .InputShapes({&data_shape, &indices_shape, &update_shape})
                    .OutputShapes(output_shapes_ref)
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                    .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({
                      {"reduction", ge::AnyValue::CreateFrom<string>(reduction)},
                      {"axis", ge::AnyValue::CreateFrom<int64_t>(axis)}
                      })
                    .TilingData(param.get())
                    .Workspace(ws_size)
                    .Build();

  gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
  ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("VectorCore");
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

  EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(ScatterTiling, scatter_david_onecore_test) {
  gert::StorageShape data_shape = {{4, 8192, 2, 64}, {4, 8192, 2, 64}};//{1, 1, 4096, 256};
  gert::StorageShape indices_shape = {{4}, {4}};
  gert::StorageShape update_shape = {{4, 1, 2, 64}, {4, 1, 2, 64}};
  int64_t axis = 1;
  string reduction = "update";
  std::vector<gert::StorageShape> output_shapes(1, {{4, 8192, 2, 64}, {4, 8192, 2, 64}});

  std::vector<void *> output_shapes_ref(1);
  for (size_t i = 0; i < output_shapes.size(); ++i) {
    output_shapes_ref[i] = &output_shapes[i];
  }

  string platform_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "cube_core_cnt": 24, "vector_core_cnt": 48, "core_type_list": "CubeCore,VectorCore"}
                          })";
  map<string, string> soc_infos;
  map<string, string> aicore_spec;
  map<string, string> intrinsics;
  std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}};
  GetPlatFormInfos(platform_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

  // platform info
  fe::PlatFormInfos platform_info;
  platform_info.Init();
  // compile info
  struct ScatterKvCompileInfo {
    int64_t core_num;
    int64_t ub_size;
    bool is910_95;
    bool support_data_move_pad;
  };
  ScatterKvCompileInfo compile_info = {48, 196608, true, true};

  std::string op_type("Scatter");
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
  auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
  auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

  // generate TilingParseContext to do TilingPrepareForScatter
  auto kernel_holder = gert::KernelRunContextFaker()
                    .KernelIONum(2, 1)
                    .Inputs({const_cast<char *>(platform_info_string.c_str()), reinterpret_cast<void *>(&platform_info)})
                    .Outputs({&compile_info})
                    .Build();

  ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("VectorCore");
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version_infos);

  // generate TilingContext to do TilingScatter
  auto param = gert::TilingData::CreateCap(4096);
  auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
  auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
  ASSERT_NE(param, nullptr);
  auto holder = gert::TilingContextFaker()
                    .SetOpType(op_type)
                    .NodeIoNum(3, 1)
                    .IrInstanceNum({1, 1, 1})
                    .InputShapes({&data_shape, &indices_shape, &update_shape})
                    .OutputShapes(output_shapes_ref)
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                    .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({
                      {"reduction", ge::AnyValue::CreateFrom<string>(reduction)},
                      {"axis", ge::AnyValue::CreateFrom<int64_t>(axis)}
                      })
                    .TilingData(param.get())
                    .Workspace(ws_size)
                    .Build();

  gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
  ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("VectorCore");
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

  EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
}

TEST_F(ScatterTiling, scatter_david_onecore_int64_test) {
  gert::StorageShape data_shape = {{4, 8192, 2, 64}, {4, 8192, 2, 64}};//{1, 1, 4096, 256};
  gert::StorageShape indices_shape = {{4}, {4}};
  gert::StorageShape update_shape = {{4, 1, 2, 64}, {4, 1, 2, 64}};
  int64_t axis = 1;
  string reduction = "update";
  std::vector<gert::StorageShape> output_shapes(1, {{4, 8192, 2, 64}, {4, 8192, 2, 64}});

  std::vector<void *> output_shapes_ref(1);
  for (size_t i = 0; i < output_shapes.size(); ++i) {
    output_shapes_ref[i] = &output_shapes[i];
  }

  string platform_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "cube_core_cnt": 24, "vector_core_cnt": 48, "core_type_list": "CubeCore,VectorCore"}
                          })";
  map<string, string> soc_infos;
  map<string, string> aicore_spec;
  map<string, string> intrinsics;
  std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}};
  GetPlatFormInfos(platform_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

  // platform info
  fe::PlatFormInfos platform_info;
  platform_info.Init();
  // compile info
  struct ScatterKvCompileInfo {
    int64_t core_num;
    int64_t ub_size;
    bool is910_95;
    bool support_data_move_pad;
  };
  ScatterKvCompileInfo compile_info = {48, 196608, true, true};

  std::string op_type("Scatter");
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
  auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
  auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

  // generate TilingParseContext to do TilingPrepareForScatter
  auto kernel_holder = gert::KernelRunContextFaker()
                    .KernelIONum(2, 1)
                    .Inputs({const_cast<char *>(platform_info_string.c_str()), reinterpret_cast<void *>(&platform_info)})
                    .Outputs({&compile_info})
                    .Build();

  ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("VectorCore");
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version_infos);

  // generate TilingContext to do TilingScatter
  auto param = gert::TilingData::CreateCap(4096);
  auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
  auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
  ASSERT_NE(param, nullptr);
  auto holder = gert::TilingContextFaker()
                    .SetOpType(op_type)
                    .NodeIoNum(3, 1)
                    .IrInstanceNum({1, 1, 1})
                    .InputShapes({&data_shape, &indices_shape, &update_shape})
                    .OutputShapes(output_shapes_ref)
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                    .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({
                      {"reduction", ge::AnyValue::CreateFrom<string>(reduction)},
                      {"axis", ge::AnyValue::CreateFrom<int64_t>(axis)}
                      })
                    .TilingData(param.get())
                    .Workspace(ws_size)
                    .Build();

  gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
  ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("VectorCore");
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

  EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
}

TEST_F(ScatterTiling, scatter_david_onecore_int32_test) {
  gert::StorageShape data_shape = {{4, 8192, 2, 64}, {4, 8192, 2, 64}};//{1, 1, 4096, 256};
  gert::StorageShape indices_shape = {{4}, {4}};
  gert::StorageShape update_shape = {{4, 1, 2, 64}, {4, 1, 2, 64}};
  int64_t axis = 1;
  string reduction = "update";
  std::vector<gert::StorageShape> output_shapes(1, {{4, 8192, 2, 64}, {4, 8192, 2, 64}});

  std::vector<void *> output_shapes_ref(1);
  for (size_t i = 0; i < output_shapes.size(); ++i) {
    output_shapes_ref[i] = &output_shapes[i];
  }

  string platform_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "cube_core_cnt": 24, "vector_core_cnt": 48, "core_type_list": "CubeCore,VectorCore"}
                          })";
  map<string, string> soc_infos;
  map<string, string> aicore_spec;
  map<string, string> intrinsics;
  std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}};
  GetPlatFormInfos(platform_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

  // platform info
  fe::PlatFormInfos platform_info;
  platform_info.Init();
  // compile info
  struct ScatterKvCompileInfo {
    int64_t core_num;
    int64_t ub_size;
    bool is910_95;
    bool support_data_move_pad;
  };
  ScatterKvCompileInfo compile_info = {48, 196608, true, true};

  std::string op_type("Scatter");
  ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
  auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
  auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

  // generate TilingParseContext to do TilingPrepareForScatter
  auto kernel_holder = gert::KernelRunContextFaker()
                    .KernelIONum(2, 1)
                    .Inputs({const_cast<char *>(platform_info_string.c_str()), reinterpret_cast<void *>(&platform_info)})
                    .Outputs({&compile_info})
                    .Build();

  ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("VectorCore");
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
  kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version_infos);

  // generate TilingContext to do TilingScatter
  auto param = gert::TilingData::CreateCap(4096);
  auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
  auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
  ASSERT_NE(param, nullptr);
  auto holder = gert::TilingContextFaker()
                    .SetOpType(op_type)
                    .NodeIoNum(3, 1)
                    .IrInstanceNum({1, 1, 1})
                    .InputShapes({&data_shape, &indices_shape, &update_shape})
                    .OutputShapes(output_shapes_ref)
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                    .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({
                      {"reduction", ge::AnyValue::CreateFrom<string>(reduction)},
                      {"axis", ge::AnyValue::CreateFrom<int64_t>(axis)}
                      })
                    .TilingData(param.get())
                    .Workspace(ws_size)
                    .Build();

  gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
  ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("VectorCore");
  holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

  EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
}
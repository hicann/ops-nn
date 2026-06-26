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
 #include <fstream>
 #include <vector>
 #include <gtest/gtest.h>
 #include "log/log.h"
 #include "../../../op_host/squared_relu_tiling.h"
 #include "kernel_run_context_facker.h"
 #include "test_cube_util.h"
 #include "platform/platform_infos_def.h"
 #include "register/op_impl_registry.h"
 #include "exe_graph/runtime/storage_format.h"
 #include "exe_graph/runtime/storage_shape.h"
 #include "ut_op_util.h"

 using namespace ut_util;
 using namespace std;
 using namespace ge;

 class SquaredReluTilingData : public testing::Test {
  protected:
   static void SetUpTestCase() {
     std::cout << "SquaredReluTilingData SetUp" << std::endl;
   }

   static void TearDownTestCase() {
     std::cout << "SquaredReluTilingData TearDown" << std::endl;
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


 TEST_F(SquaredReluTilingData, SquaredReluTilingData_01) {
   size_t M = 2;
   size_t N = 4;
   size_t D = N;

   gert::StorageShape x_shape = {{M, N},{M, N}};
   gert::StorageShape y_shape = {{M, D},{M, D}};

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
   optiling::SquaredReluCompileInfo compile_info;

   std::string op_type("SquaredRelu");
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
   auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
   auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holder.get());
   ASSERT_NE(param, nullptr);

   auto holder = gert::TilingContextFaker()
                     .NodeIoNum(1, 1)
                     .IrInstanceNum({1})
                     .InputShapes({&x_shape})
                     .OutputShapes({&y_shape})
                     .CompileInfo(&compile_info)
                     .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                     .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
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

   auto tiling_key = tiling_context->GetTilingKey();
   ASSERT_EQ(tiling_key, 2);
 }

 // 测试FLOAT16数据类型tiling，tiling_key应为1
 TEST_F(SquaredReluTilingData, SquaredReluTilingData_02_fp16) {
   size_t M = 2;
   size_t N = 4;
   size_t D = N;

   gert::StorageShape x_shape = {{M, N},{M, N}};
   gert::StorageShape y_shape = {{M, D},{M, D}};

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

   fe::PlatFormInfos platform_info;
   platform_info.Init();
   optiling::SquaredReluCompileInfo compile_info;

   std::string op_type("SquaredRelu");
   ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
   auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
   auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

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

   auto param = gert::TilingData::CreateCap(4096);
   auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
   auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holder.get());
   ASSERT_NE(param, nullptr);

   auto holder = gert::TilingContextFaker()
                     .NodeIoNum(1, 1)
                     .IrInstanceNum({1})
                     .InputShapes({&x_shape})
                     .OutputShapes({&y_shape})
                     .CompileInfo(&compile_info)
                     .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                     .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                     .TilingData(param.get())
                     .Workspace(ws_size)
                     .Build();

   gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
   ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
   holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
   holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
   holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
   holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

   EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);

   auto tiling_key = tiling_context->GetTilingKey();
   ASSERT_EQ(tiling_key, 1);
 }

 // 测试BF16数据类型tiling，tiling_key应为3
 TEST_F(SquaredReluTilingData, SquaredReluTilingData_03_bf16) {
   size_t M = 2;
   size_t N = 4;
   size_t D = N;

   gert::StorageShape x_shape = {{M, N},{M, N}};
   gert::StorageShape y_shape = {{M, D},{M, D}};

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

   fe::PlatFormInfos platform_info;
   platform_info.Init();
   optiling::SquaredReluCompileInfo compile_info;

   std::string op_type("SquaredRelu");
   ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
   auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
   auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

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

   auto param = gert::TilingData::CreateCap(4096);
   auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
   auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holder.get());
   ASSERT_NE(param, nullptr);

   auto holder = gert::TilingContextFaker()
                     .NodeIoNum(1, 1)
                     .IrInstanceNum({1})
                     .InputShapes({&x_shape})
                     .OutputShapes({&y_shape})
                     .CompileInfo(&compile_info)
                     .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                     .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                     .TilingData(param.get())
                     .Workspace(ws_size)
                     .Build();

   gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
   ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
   holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
   holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
   holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
   holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

   EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);

   auto tiling_key = tiling_context->GetTilingKey();
   ASSERT_EQ(tiling_key, 3);
 }

 // 测试不支持的数据类型INT32，tiling应失败
 TEST_F(SquaredReluTilingData, SquaredReluTilingData_04_unsupported_dtype) {
   size_t M = 2;
   size_t N = 4;
   size_t D = N;

   gert::StorageShape x_shape = {{M, N},{M, N}};
   gert::StorageShape y_shape = {{M, D},{M, D}};

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

   fe::PlatFormInfos platform_info;
   platform_info.Init();
   optiling::SquaredReluCompileInfo compile_info;

   std::string op_type("SquaredRelu");
   ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
   auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
   auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

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

   auto param = gert::TilingData::CreateCap(4096);
   auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
   auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holder.get());
   ASSERT_NE(param, nullptr);

   auto holder = gert::TilingContextFaker()
                     .NodeIoNum(1, 1)
                     .IrInstanceNum({1})
                     .InputShapes({&x_shape})
                     .OutputShapes({&y_shape})
                     .CompileInfo(&compile_info)
                     .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                     .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                     .TilingData(param.get())
                     .Workspace(ws_size)
                     .Build();

   gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
   ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
   holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
   holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
   holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
   holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

   // INT32不在支持列表内，tiling应失败
   EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
 }

 // 测试大shape场景，验证多核tiling逻辑
 TEST_F(SquaredReluTilingData, SquaredReluTilingData_05_large_shape_multicore) {
   // 元素数32768 > MAX_ELEMENT_NUM_EACH_CORE(24576)，需要多核
   size_t M = 1;
   size_t N = 32768;
   size_t D = N;

   gert::StorageShape x_shape = {{M, N},{M, N}};
   gert::StorageShape y_shape = {{M, D},{M, D}};

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

   fe::PlatFormInfos platform_info;
   platform_info.Init();
   optiling::SquaredReluCompileInfo compile_info;

   std::string op_type("SquaredRelu");
   ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
   auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
   auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

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

   auto param = gert::TilingData::CreateCap(4096);
   auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
   auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holder.get());
   ASSERT_NE(param, nullptr);

   auto holder = gert::TilingContextFaker()
                     .NodeIoNum(1, 1)
                     .IrInstanceNum({1})
                     .InputShapes({&x_shape})
                     .OutputShapes({&y_shape})
                     .CompileInfo(&compile_info)
                     .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                     .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
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

   EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);

   auto tiling_key = tiling_context->GetTilingKey();
   ASSERT_EQ(tiling_key, 2);
   // 32768个元素，每个核最多24576个元素，CeilA2B(32768, 24576) = 2
   auto block_dim = tiling_context->GetBlockDim();
   ASSERT_EQ(block_dim, 2);
 }

 // 测试超大shape场景，需要的核数超过平台核数，应返回平台核数
 TEST_F(SquaredReluTilingData, SquaredReluTilingData_06_huge_shape_maxcore) {
   // 元素数远超40*24576=983040，需要的核数应被限制为平台核数40
   size_t M = 1;
   size_t N = 2000000;
   size_t D = N;

   gert::StorageShape x_shape = {{M, N},{M, N}};
   gert::StorageShape y_shape = {{M, D},{M, D}};

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

   fe::PlatFormInfos platform_info;
   platform_info.Init();
   optiling::SquaredReluCompileInfo compile_info;

   std::string op_type("SquaredRelu");
   ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
   auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
   auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

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

   auto param = gert::TilingData::CreateCap(4096);
   auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
   auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holder.get());
   ASSERT_NE(param, nullptr);

   auto holder = gert::TilingContextFaker()
                     .NodeIoNum(1, 1)
                     .IrInstanceNum({1})
                     .InputShapes({&x_shape})
                     .OutputShapes({&y_shape})
                     .CompileInfo(&compile_info)
                     .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                     .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                     .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                     .TilingData(param.get())
                     .Workspace(ws_size)
                     .Build();

   gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
   ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
   holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
   holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
   holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
   holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

   EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);

   auto tiling_key = tiling_context->GetTilingKey();
   ASSERT_EQ(tiling_key, 1);
   // 需要的核数超过平台核数40，应被限制为40
   auto block_dim = tiling_context->GetBlockDim();
   ASSERT_EQ(block_dim, 40);
 }
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_sync_batch_norm_backward_elemt_tiling.cpp
 * \brief
 */

#include "../../../../op_host/arch35/sync_batch_norm_gather_stats_tiling_arch35.h"
#include <gtest/gtest.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdio>
#include <string>
#include "log/log.h"
#include "log/log.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "platform/platform_infos_def.h"
#include "ut_op_util.h"

using namespace ge;
using namespace ut_util;
using namespace std;

class SyncBatchNormGatherStatsTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "SyncBatchNormGatherStatsTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "SyncBatchNormGatherStatsTiling TearDown" << std::endl;
  }
};

TEST_F(SyncBatchNormGatherStatsTiling, SyncBatchNormGatherStatsTiling5) {
    dlog_setlevel(0, 0, 0);
    gert::StorageShape total_sum_shape = {{16, 100}, {16, 100}};
    gert::StorageShape total_square_sum_shape = {{16, 100}, {16, 100}};
    gert::StorageShape sample_count_shape = {{16}, {16}};
    gert::StorageShape mean_shape = {{100}, {100}};
    gert::StorageShape var_shape = {{100}, {100}};
    gert::StorageShape batch_mean_shape = {{100}, {100}};
    gert::StorageShape batch_invstd_shape = {{100}, {100}};
    gert::StorageShape running_mean_update = {{100}, {100}};
    gert::StorageShape running_var_update = {{100}, {100}};
    ge::DataType dataType = ge::DT_FLOAT;
    ge::DataType countType = ge::DT_INT32;
    ge::Format dataFormat = ge::FORMAT_ND;
    string compile_info_string = R"({
          "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                            "Intrinsic_fix_pipe_l0c2out": false,
                            "Intrinsic_data_move_l12ub": true,
                            "Intrinsic_data_move_l0c2ub": true,
                            "Intrinsic_data_move_out2l1_nd2nz": false,
                            "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                            "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                            "CORE_NUM": 64, "socVersion": "Ascend950"}})";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_version;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec,
                      intrinsics, soc_version);
  
    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
  
    // compile info
    optiling::SyncBatchNormGatherStatsCompileInfo compile_info;
  
    string op_type("SyncBatchNormGatherStats");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()),
              nullptr);
    auto tiling_func =
        gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance()
                                  .GetOpImpl(op_type.c_str())
                                  ->tiling_parse;
  
    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char *>(compile_info_string.c_str()),
                      reinterpret_cast<void *>(&platform_info)})
            .Outputs({&compile_info})
            .Build();
    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()
                    ->GetPlatformInfo()
                    ->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()
        ->GetPlatformInfo()
        ->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()
        ->GetPlatformInfo()
        ->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()
        ->GetPlatformInfo()
        ->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()
        ->GetPlatformInfo()
        ->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()
        ->GetPlatformInfo()
        ->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
  
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()),
              ge::GRAPH_SUCCESS);
    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size =
        reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(5, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1})
                      .InputShapes({&total_sum_shape, &total_square_sum_shape, &sample_count_shape, &mean_shape, &var_shape})
                      .OutputShapes({&batch_mean_shape, &batch_invstd_shape, &running_mean_update, &running_var_update})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                      .NodeInputTd(0, dataType, dataFormat, dataFormat)
                      .NodeInputTd(1, dataType, dataFormat, dataFormat)
                      .NodeInputTd(2, countType, dataFormat, dataFormat)
                      .NodeInputTd(3, dataType, dataFormat, dataFormat)
                      .NodeInputTd(4, dataType, dataFormat, dataFormat)
                      .NodeOutputTd(0, dataType, dataFormat, dataFormat)
                      .NodeOutputTd(1, dataType, dataFormat, dataFormat)
                      .NodeOutputTd(2, dataType, dataFormat, dataFormat)
                      .NodeOutputTd(3, dataType, dataFormat, dataFormat)
                      .NodeAttrs({{"momentum", Ops::NN::AnyValue::CreateFrom<float>(1e-01)},
                                  {"eps", Ops::NN::AnyValue::CreateFrom<float>(1e-05)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();
    gert::TilingContext *tiling_context =
        holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes(
        "SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()
        ->GetPlatformInfo()
        ->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);
  
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    dlog_setlevel(0, 3, 0);
}

TEST_F(SyncBatchNormGatherStatsTiling, SyncBatchNormGatherStatsTiling6) {
    dlog_setlevel(0, 0, 0);
    gert::StorageShape total_sum_shape = {{16, 10000}, {16, 10000}};
    gert::StorageShape total_square_sum_shape = {{16, 10000}, {16, 10000}};
    gert::StorageShape sample_count_shape = {{16}, {16}};
    gert::StorageShape mean_shape = {{10000}, {10000}};
    gert::StorageShape var_shape = {{10000}, {10000}};
    gert::StorageShape batch_mean_shape = {{10000}, {10000}};
    gert::StorageShape batch_invstd_shape = {{10000}, {10000}};
    gert::StorageShape running_mean_update = {{10000}, {10000}};
    gert::StorageShape running_var_update = {{10000}, {10000}};
    ge::DataType dataType = ge::DT_FLOAT;
    ge::DataType countType = ge::DT_INT32;
    ge::Format dataFormat = ge::FORMAT_ND;
    string compile_info_string = R"({
          "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                            "Intrinsic_fix_pipe_l0c2out": false,
                            "Intrinsic_data_move_l12ub": true,
                            "Intrinsic_data_move_l0c2ub": true,
                            "Intrinsic_data_move_out2l1_nd2nz": false,
                            "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                            "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                            "CORE_NUM": 64, "socVersion": "Ascend950"}})";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_version;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec,
                      intrinsics, soc_version);
  
    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
  
    // compile info
    optiling::SyncBatchNormGatherStatsCompileInfo compile_info;
  
    string op_type("SyncBatchNormGatherStats");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()),
              nullptr);
    auto tiling_func =
        gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance()
                                  .GetOpImpl(op_type.c_str())
                                  ->tiling_parse;
  
    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char *>(compile_info_string.c_str()),
                      reinterpret_cast<void *>(&platform_info)})
            .Outputs({&compile_info})
            .Build();
    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()
                    ->GetPlatformInfo()
                    ->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()
        ->GetPlatformInfo()
        ->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()
        ->GetPlatformInfo()
        ->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()
        ->GetPlatformInfo()
        ->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()
        ->GetPlatformInfo()
        ->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()
        ->GetPlatformInfo()
        ->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
  
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()),
              ge::GRAPH_SUCCESS);
    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size =
        reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(5, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1})
                      .InputShapes({&total_sum_shape, &total_square_sum_shape, &sample_count_shape, &mean_shape, &var_shape})
                      .OutputShapes({&batch_mean_shape, &batch_invstd_shape, &running_mean_update, &running_var_update})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                      .NodeInputTd(0, dataType, dataFormat, dataFormat)
                      .NodeInputTd(1, dataType, dataFormat, dataFormat)
                      .NodeInputTd(2, countType, dataFormat, dataFormat)
                      .NodeInputTd(3, dataType, dataFormat, dataFormat)
                      .NodeInputTd(4, dataType, dataFormat, dataFormat)
                      .NodeOutputTd(0, dataType, dataFormat, dataFormat)
                      .NodeOutputTd(1, dataType, dataFormat, dataFormat)
                      .NodeOutputTd(2, dataType, dataFormat, dataFormat)
                      .NodeOutputTd(3, dataType, dataFormat, dataFormat)
                      .NodeAttrs({{"momentum", Ops::NN::AnyValue::CreateFrom<float>(1e-01)},
                                  {"eps", Ops::NN::AnyValue::CreateFrom<float>(1e-05)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();
    gert::TilingContext *tiling_context =
        holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes(
        "SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()
        ->GetPlatformInfo()
        ->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);
  
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    dlog_setlevel(0, 3, 0);
}

TEST_F(SyncBatchNormGatherStatsTiling, SyncBatchNormGatherStatsTiling7) {
    dlog_setlevel(0, 0, 0);
    gert::StorageShape total_sum_shape = {{16, 100}, {16, 100}};
    gert::StorageShape total_square_sum_shape = {{16, 100}, {16, 100}};
    gert::StorageShape sample_count_shape = {{16}, {16}};
    gert::StorageShape mean_shape = {{100}, {100}};
    gert::StorageShape var_shape = {{100}, {100}};
    gert::StorageShape batch_mean_shape = {{100}, {100}};
    gert::StorageShape batch_invstd_shape = {{100}, {100}};
    gert::StorageShape running_mean_update = {{100}, {100}};
    gert::StorageShape running_var_update = {{100}, {100}};
    ge::DataType dataType = ge::DT_FLOAT16;
    ge::DataType countType = ge::DT_INT32;
    ge::Format dataFormat = ge::FORMAT_ND;
    string compile_info_string = R"({
          "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                            "Intrinsic_fix_pipe_l0c2out": false,
                            "Intrinsic_data_move_l12ub": true,
                            "Intrinsic_data_move_l0c2ub": true,
                            "Intrinsic_data_move_out2l1_nd2nz": false,
                            "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                            "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                            "CORE_NUM": 64, "socVersion": "Ascend950"}})";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_version;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec,
                      intrinsics, soc_version);
  
    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
  
    // compile info
    optiling::SyncBatchNormGatherStatsCompileInfo compile_info;
  
    string op_type("SyncBatchNormGatherStats");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()),
              nullptr);
    auto tiling_func =
        gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance()
                                  .GetOpImpl(op_type.c_str())
                                  ->tiling_parse;
  
    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char *>(compile_info_string.c_str()),
                      reinterpret_cast<void *>(&platform_info)})
            .Outputs({&compile_info})
            .Build();
    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()
                    ->GetPlatformInfo()
                    ->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()
        ->GetPlatformInfo()
        ->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()
        ->GetPlatformInfo()
        ->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()
        ->GetPlatformInfo()
        ->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()
        ->GetPlatformInfo()
        ->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()
        ->GetPlatformInfo()
        ->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
  
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()),
              ge::GRAPH_SUCCESS);
    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size =
        reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(5, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1})
                      .InputShapes({&total_sum_shape, &total_square_sum_shape, &sample_count_shape, &mean_shape, &var_shape})
                      .OutputShapes({&batch_mean_shape, &batch_invstd_shape, &running_mean_update, &running_var_update})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                      .NodeInputTd(0, dataType, dataFormat, dataFormat)
                      .NodeInputTd(1, dataType, dataFormat, dataFormat)
                      .NodeInputTd(2, countType, dataFormat, dataFormat)
                      .NodeInputTd(3, dataType, dataFormat, dataFormat)
                      .NodeInputTd(4, dataType, dataFormat, dataFormat)
                      .NodeOutputTd(0, dataType, dataFormat, dataFormat)
                      .NodeOutputTd(1, dataType, dataFormat, dataFormat)
                      .NodeOutputTd(2, dataType, dataFormat, dataFormat)
                      .NodeOutputTd(3, dataType, dataFormat, dataFormat)
                      .NodeAttrs({{"momentum", Ops::NN::AnyValue::CreateFrom<float>(1e-01)},
                                  {"eps", Ops::NN::AnyValue::CreateFrom<float>(1e-05)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();
    gert::TilingContext *tiling_context =
        holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes(
        "SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()
        ->GetPlatformInfo()
        ->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);
  
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    dlog_setlevel(0, 3, 0);
}

TEST_F(SyncBatchNormGatherStatsTiling, SyncBatchNormGatherStatsTiling8) {
    dlog_setlevel(0, 0, 0);
    gert::StorageShape total_sum_shape = {{16, 10000}, {16, 10000}};
    gert::StorageShape total_square_sum_shape = {{16, 10000}, {16, 10000}};
    gert::StorageShape sample_count_shape = {{16}, {16}};
    gert::StorageShape mean_shape = {{10000}, {10000}};
    gert::StorageShape var_shape = {{10000}, {10000}};
    gert::StorageShape batch_mean_shape = {{10000}, {10000}};
    gert::StorageShape batch_invstd_shape = {{10000}, {10000}};
    gert::StorageShape running_mean_update = {{10000}, {10000}};
    gert::StorageShape running_var_update = {{10000}, {10000}};
    ge::DataType dataType = ge::DT_FLOAT16;
    ge::DataType countType = ge::DT_INT32;
    ge::Format dataFormat = ge::FORMAT_ND;
    string compile_info_string = R"({
          "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                            "Intrinsic_fix_pipe_l0c2out": false,
                            "Intrinsic_data_move_l12ub": true,
                            "Intrinsic_data_move_l0c2ub": true,
                            "Intrinsic_data_move_out2l1_nd2nz": false,
                            "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                            "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                            "CORE_NUM": 64, "socVersion": "Ascend950"}})";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_version;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec,
                      intrinsics, soc_version);
  
    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
  
    // compile info
    optiling::SyncBatchNormGatherStatsCompileInfo compile_info;
  
    string op_type("SyncBatchNormGatherStats");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()),
              nullptr);
    auto tiling_func =
        gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance()
                                  .GetOpImpl(op_type.c_str())
                                  ->tiling_parse;
  
    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char *>(compile_info_string.c_str()),
                      reinterpret_cast<void *>(&platform_info)})
            .Outputs({&compile_info})
            .Build();
    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()
                    ->GetPlatformInfo()
                    ->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()
        ->GetPlatformInfo()
        ->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()
        ->GetPlatformInfo()
        ->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()
        ->GetPlatformInfo()
        ->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()
        ->GetPlatformInfo()
        ->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()
        ->GetPlatformInfo()
        ->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
  
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()),
              ge::GRAPH_SUCCESS);
    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size =
        reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(5, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1})
                      .InputShapes({&total_sum_shape, &total_square_sum_shape, &sample_count_shape, &mean_shape, &var_shape})
                      .OutputShapes({&batch_mean_shape, &batch_invstd_shape, &running_mean_update, &running_var_update})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                      .NodeInputTd(0, dataType, dataFormat, dataFormat)
                      .NodeInputTd(1, dataType, dataFormat, dataFormat)
                      .NodeInputTd(2, countType, dataFormat, dataFormat)
                      .NodeInputTd(3, dataType, dataFormat, dataFormat)
                      .NodeInputTd(4, dataType, dataFormat, dataFormat)
                      .NodeOutputTd(0, dataType, dataFormat, dataFormat)
                      .NodeOutputTd(1, dataType, dataFormat, dataFormat)
                      .NodeOutputTd(2, dataType, dataFormat, dataFormat)
                      .NodeOutputTd(3, dataType, dataFormat, dataFormat)
                      .NodeAttrs({{"momentum", Ops::NN::AnyValue::CreateFrom<float>(1e-01)},
                                  {"eps", Ops::NN::AnyValue::CreateFrom<float>(1e-05)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();
    gert::TilingContext *tiling_context =
        holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes(
        "SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()
        ->GetPlatformInfo()
        ->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);
  
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    dlog_setlevel(0, 3, 0);
}

TEST_F(SyncBatchNormGatherStatsTiling, SyncBatchNormGatherStatsTiling9) {
    dlog_setlevel(0, 0, 0);
    gert::StorageShape total_sum_shape = {{5089, 1678}, {5089, 1678}};
    gert::StorageShape total_square_sum_shape = {{5089, 1678}, {5089, 1678}};
    gert::StorageShape sample_count_shape = {{5089}, {5089}};
    gert::StorageShape mean_shape = {{1678}, {1678}};
    gert::StorageShape var_shape = {{1678}, {1678}};
    gert::StorageShape batch_mean_shape = {{1678}, {1678}};
    gert::StorageShape batch_invstd_shape = {{1678}, {1678}};
    gert::StorageShape running_mean_update = {{1678}, {1678}};
    gert::StorageShape running_var_update = {{1678}, {1678}};
    ge::DataType dataType = ge::DT_FLOAT16;
    ge::DataType countType = ge::DT_INT32;
    ge::Format dataFormat = ge::FORMAT_ND;
    string compile_info_string = R"({
          "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                            "Intrinsic_fix_pipe_l0c2out": false,
                            "Intrinsic_data_move_l12ub": true,
                            "Intrinsic_data_move_l0c2ub": true,
                            "Intrinsic_data_move_out2l1_nd2nz": false,
                            "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                            "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                            "CORE_NUM": 64, "socVersion": "Ascend950"}})";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_version;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec,
                      intrinsics, soc_version);
  
    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
  
    // compile info
    optiling::SyncBatchNormGatherStatsCompileInfo compile_info;
  
    string op_type("SyncBatchNormGatherStats");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()),
              nullptr);
    auto tiling_func =
        gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance()
                                  .GetOpImpl(op_type.c_str())
                                  ->tiling_parse;
  
    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char *>(compile_info_string.c_str()),
                      reinterpret_cast<void *>(&platform_info)})
            .Outputs({&compile_info})
            .Build();
    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()
                    ->GetPlatformInfo()
                    ->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()
        ->GetPlatformInfo()
        ->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()
        ->GetPlatformInfo()
        ->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()
        ->GetPlatformInfo()
        ->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()
        ->GetPlatformInfo()
        ->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()
        ->GetPlatformInfo()
        ->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
  
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()),
              ge::GRAPH_SUCCESS);
    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size =
        reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(5, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1})
                      .InputShapes({&total_sum_shape, &total_square_sum_shape, &sample_count_shape, &mean_shape, &var_shape})
                      .OutputShapes({&batch_mean_shape, &batch_invstd_shape, &running_mean_update, &running_var_update})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                      .NodeInputTd(0, dataType, dataFormat, dataFormat)
                      .NodeInputTd(1, dataType, dataFormat, dataFormat)
                      .NodeInputTd(2, countType, dataFormat, dataFormat)
                      .NodeInputTd(3, dataType, dataFormat, dataFormat)
                      .NodeInputTd(4, dataType, dataFormat, dataFormat)
                      .NodeOutputTd(0, dataType, dataFormat, dataFormat)
                      .NodeOutputTd(1, dataType, dataFormat, dataFormat)
                      .NodeOutputTd(2, dataType, dataFormat, dataFormat)
                      .NodeOutputTd(3, dataType, dataFormat, dataFormat)
                      .NodeAttrs({{"momentum", Ops::NN::AnyValue::CreateFrom<float>(1e-01)},
                                  {"eps", Ops::NN::AnyValue::CreateFrom<float>(1e-05)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();
    gert::TilingContext *tiling_context =
        holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes(
        "SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()
        ->GetPlatformInfo()
        ->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);
  
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    dlog_setlevel(0, 3, 0);
}

TEST_F(SyncBatchNormGatherStatsTiling, SyncBatchNormGatherStatsTiling10) {
    dlog_setlevel(0, 0, 0);
    gert::StorageShape total_sum_shape = {{5089, 1678}, {5089, 1678}};
    gert::StorageShape total_square_sum_shape = {{5089, 1678}, {5089, 1678}};
    gert::StorageShape sample_count_shape = {{5089}, {5089}};
    gert::StorageShape mean_shape = {{1678}, {1678}};
    gert::StorageShape var_shape = {{1678}, {1678}};
    gert::StorageShape batch_mean_shape = {{1678}, {1678}};
    gert::StorageShape batch_invstd_shape = {{1678}, {1678}};
    gert::StorageShape running_mean_update = {{1678}, {1678}};
    gert::StorageShape running_var_update = {{1678}, {1678}};
    ge::DataType dataType = ge::DT_FLOAT;
    ge::DataType countType = ge::DT_INT32;
    ge::Format dataFormat = ge::FORMAT_ND;
    string compile_info_string = R"({
          "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                            "Intrinsic_fix_pipe_l0c2out": false,
                            "Intrinsic_data_move_l12ub": true,
                            "Intrinsic_data_move_l0c2ub": true,
                            "Intrinsic_data_move_out2l1_nd2nz": false,
                            "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                            "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                            "CORE_NUM": 64, "socVersion": "Ascend950"}})";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_version;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec,
                      intrinsics, soc_version);
  
    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
  
    // compile info
    optiling::SyncBatchNormGatherStatsCompileInfo compile_info;
  
    string op_type("SyncBatchNormGatherStats");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()),
              nullptr);
    auto tiling_func =
        gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance()
                                  .GetOpImpl(op_type.c_str())
                                  ->tiling_parse;
  
    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char *>(compile_info_string.c_str()),
                      reinterpret_cast<void *>(&platform_info)})
            .Outputs({&compile_info})
            .Build();
    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()
                    ->GetPlatformInfo()
                    ->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()
        ->GetPlatformInfo()
        ->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()
        ->GetPlatformInfo()
        ->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()
        ->GetPlatformInfo()
        ->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()
        ->GetPlatformInfo()
        ->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()
        ->GetPlatformInfo()
        ->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
  
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()),
              ge::GRAPH_SUCCESS);
    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size =
        reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(5, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1})
                      .InputShapes({&total_sum_shape, &total_square_sum_shape, &sample_count_shape, &mean_shape, &var_shape})
                      .OutputShapes({&batch_mean_shape, &batch_invstd_shape, &running_mean_update, &running_var_update})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                      .NodeInputTd(0, dataType, dataFormat, dataFormat)
                      .NodeInputTd(1, dataType, dataFormat, dataFormat)
                      .NodeInputTd(2, countType, dataFormat, dataFormat)
                      .NodeInputTd(3, dataType, dataFormat, dataFormat)
                      .NodeInputTd(4, dataType, dataFormat, dataFormat)
                      .NodeOutputTd(0, dataType, dataFormat, dataFormat)
                      .NodeOutputTd(1, dataType, dataFormat, dataFormat)
                      .NodeOutputTd(2, dataType, dataFormat, dataFormat)
                      .NodeOutputTd(3, dataType, dataFormat, dataFormat)
                      .NodeAttrs({{"momentum", Ops::NN::AnyValue::CreateFrom<float>(1e-01)},
                                  {"eps", Ops::NN::AnyValue::CreateFrom<float>(1e-05)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();
    gert::TilingContext *tiling_context =
        holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes(
        "SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()
        ->GetPlatformInfo()
        ->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);
  
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    dlog_setlevel(0, 3, 0);
}
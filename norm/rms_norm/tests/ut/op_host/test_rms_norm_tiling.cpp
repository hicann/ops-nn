/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <iostream>
#include <fstream>
#include <vector>
#include <gtest/gtest.h>
#include "log/log.h"
#include "ut_op_util.h"
#include "platform/platform_infos_def.h"
#include "../../../op_host/rms_norm_tiling.h"
#include "kernel_run_context_facker.h"
#include "tiling_context_faker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"

using namespace ut_util;
using namespace std;
using namespace ge;

class RmsNormTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "RmsNormTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "RmsNormTiling TearDown" << std::endl; }
};

TEST_F(RmsNormTiling, rms_norm_mixtype_tiling_001)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape = {{24, 1, 25600}, {24, 1, 25600}};
    gert::StorageShape gamma_shape = {{
                                          25600,
                                      },
                                      {
                                          25600,
                                      }};
    gert::StorageShape out_shape = {{24, 1, 25600}, {24, 1, 25600}};
    gert::StorageShape rstd_shape = {{24, 1, 1}, {24, 1, 1}};
    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 64}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    static optiling::Tiling4RmsNormCompileInfo compile_info;

    std::string op_type("RmsNorm");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(compile_info_string.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(2, 2)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&input_shape, &gamma_shape})
                      .OutputShapes({&out_shape, &rstd_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)}})
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
    ASSERT_EQ(tiling_key, 1);
    // dlog_setlevel(0, 3, 0);
}

TEST_F(RmsNormTiling, rms_norm_tiling_001)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape = {{24, 1, 2560}, {24, 1, 2560}};
    gert::StorageShape gamma_shape = {{
                                          2560,
                                      },
                                      {
                                          2560,
                                      }};
    gert::StorageShape out_shape = {{24, 1, 2560}, {24, 1, 2560}};
    gert::StorageShape rstd_shape = {{24, 1, 1}, {24, 1, 1}};

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

    static optiling::Tiling4RmsNormCompileInfo compile_info;

    std::string op_type("RmsNorm");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(compile_info_string.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
                             .Outputs({nullptr})
                             .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);

    tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>());

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(2, 2)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&input_shape, &gamma_shape})
                      .OutputShapes({&out_shape, &rstd_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    tiling_func(tiling_context);
    // workspaces nullptr return failed
    // EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // // todo check tiling result
    // auto tiling_key = tiling_context->GetTilingKey();
    // ASSERT_EQ(tiling_key, 0);
    // dlog_setlevel(0, 3, 0);
}

TEST_F(RmsNormTiling, rms_norm_tiling_002)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape = {{24, 1, 25600}, {24, 1, 25600}};
    gert::StorageShape gamma_shape = {{
                                          25600,
                                      },
                                      {
                                          25600,
                                      }};
    gert::StorageShape out_shape = {{24, 1, 25600}, {24, 1, 25600}};
    gert::StorageShape rstd_shape = {{24, 1, 1}, {24, 1, 1}};

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
    static optiling::Tiling4RmsNormCompileInfo compile_info = {40, 196608, platform_ascendc::SocVersion::ASCEND910B};

    std::string op_type("RmsNorm");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(compile_info_string.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(2, 2)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&input_shape, &gamma_shape})
                      .OutputShapes({&out_shape, &rstd_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)}})
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
    ASSERT_EQ(tiling_key, 1);
    // dlog_setlevel(0, 3, 0);
}

TEST_F(RmsNormTiling, rms_norm_tiling_003)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape = {{24, 1, 2560}, {24, 1, 2560}};
    gert::StorageShape gamma_shape = {{
                                          2560,
                                      },
                                      {
                                          2560,
                                      }};
    gert::StorageShape out_shape = {{24, 1, 2560}, {24, 1, 2560}};
    gert::StorageShape rstd_shape = {{24, 1, 1}, {24, 1, 1}};

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
    struct Tiling4RmsNormCompileInfo {
        uint32_t totalCoreNum = 40;
        uint64_t totalUbSize = 196608;
        platform_ascendc::SocVersion curSocVersion = platform_ascendc::SocVersion::ASCEND910B;
    };
    Tiling4RmsNormCompileInfo compile_info;
    std::string op_type("RmsNorm");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(compile_info_string.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    compile_info.curSocVersion = platform_ascendc::SocVersion::ASCEND910;
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(2, 2)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&input_shape, &gamma_shape})
                      .OutputShapes({&out_shape, &rstd_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)}})
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
    ASSERT_EQ(tiling_key, 1110);
    // dlog_setlevel(0, 3, 0);
}

TEST_F(RmsNormTiling, rms_norm_tiling_004)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape = {{24, 1, 25600}, {24, 1, 25600}};
    gert::StorageShape gamma_shape = {{
                                          25600,
                                      },
                                      {
                                          25600,
                                      }};
    gert::StorageShape out_shape = {{24, 1, 25600}, {24, 1, 25600}};
    gert::StorageShape rstd_shape = {{24, 1, 1}, {24, 1, 1}};

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
    struct Tiling4RmsNormCompileInfo {
        uint32_t totalCoreNum = 40;
        uint64_t totalUbSize = 196608;
        platform_ascendc::SocVersion curSocVersion = platform_ascendc::SocVersion::ASCEND910B;
    };
    Tiling4RmsNormCompileInfo compile_info;

    std::string op_type("RmsNorm");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(compile_info_string.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    compile_info.curSocVersion = platform_ascendc::SocVersion::ASCEND910;
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(2, 2)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&input_shape, &gamma_shape})
                      .OutputShapes({&out_shape, &rstd_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)}})
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
    ASSERT_EQ(tiling_key, 1);
    // dlog_setlevel(0, 3, 0);
}

TEST_F(RmsNormTiling, rms_norm_tiling_005)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape = {{2, 1, 186175500}, {2, 1, 186175500}};
    gert::StorageShape gamma_shape = {{
                                          186175500,
                                      },
                                      {
                                          186175500,
                                      }};
    gert::StorageShape out_shape = {{2, 1, 186175500}, {2, 1, 186175500}};
    gert::StorageShape rstd_shape = {{2, 1, 1}, {2, 1, 1}};

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
    static optiling::Tiling4RmsNormCompileInfo compile_info = {40, 196608, platform_ascendc::SocVersion::ASCEND910B};

    std::string op_type("RmsNorm");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(compile_info_string.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(2, 2)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&input_shape, &gamma_shape})
                      .OutputShapes({&out_shape, &rstd_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)}})
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
    ASSERT_EQ(tiling_key, 1);
    // dlog_setlevel(0, 3, 0);
}

TEST_F(RmsNormTiling, rms_norm_tiling_006)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape = {{24, 1, 256}, {24, 1, 256}};
    gert::StorageShape gamma_shape = {{
                                          256,
                                      },
                                      {
                                          256,
                                      }};
    gert::StorageShape out_shape = {{24, 1, 256}, {24, 1, 256}};
    gert::StorageShape rstd_shape = {{24, 1, 1}, {24, 1, 1}};

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
    struct Tiling4RmsNormCompileInfo {
        uint32_t totalCoreNum = 40;
        uint64_t totalUbSize = 196608;
        platform_ascendc::SocVersion curSocVersion = platform_ascendc::SocVersion::ASCEND910B;
    };
    Tiling4RmsNormCompileInfo compile_info;
    std::string op_type("RmsNorm");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(compile_info_string.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    compile_info.curSocVersion = platform_ascendc::SocVersion::ASCEND910B;
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(2, 2)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&input_shape, &gamma_shape})
                      .OutputShapes({&out_shape, &rstd_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)}})
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
    ASSERT_EQ(tiling_key, 2);
    // dlog_setlevel(0, 3, 0);
}

TEST_F(RmsNormTiling, rms_norm_perf_tiling_001)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape = {{512, 24, 128}, {512, 24, 128}};
    gert::StorageShape gamma_shape = {{
                                          128,
                                      },
                                      {
                                          128,
                                      }};
    gert::StorageShape out_shape = {{512, 24, 128}, {512, 24, 128}};
    gert::StorageShape rstd_shape = {{512, 24, 1}, {512, 24, 1}};
    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 64}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> npuarchs = {{"NpuArch", "3510"}};
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    static optiling::Tiling4RmsNormCompileInfo compile_info;
    compile_info.totalCoreNum = 64;
    compile_info.totalUbSize = 245760;

    std::string op_type("RmsNorm");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(compile_info_string.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
    compile_info.curSocVersion = platform_ascendc::SocVersion::ASCEND950;
    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(2, 2)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&input_shape, &gamma_shape})
                      .OutputShapes({&out_shape, &rstd_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 5000);
    // dlog_setlevel(0, 3, 0);
}

TEST_F(RmsNormTiling, rms_norm_tiling_split_D_001)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape = {{1, 24, 20000}, {1, 24, 20000}};
    gert::StorageShape gamma_shape = {{
                                          20000,
                                      },
                                      {
                                          20000,
                                      }};
    gert::StorageShape out_shape = {{1, 24, 20000}, {1, 24, 20000}};
    gert::StorageShape rstd_shape = {{1, 24, 1}, {1, 24, 1}};
    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 64}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> npuarchs = {{"NpuArch", "3510"}};
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    static optiling::Tiling4RmsNormCompileInfo compile_info;
    compile_info.totalCoreNum = 64;
    compile_info.totalUbSize = 245760;

    std::string op_type("RmsNorm");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(compile_info_string.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
    compile_info.curSocVersion = platform_ascendc::SocVersion::ASCEND950;
    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(2, 2)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&input_shape, &gamma_shape})
                      .OutputShapes({&out_shape, &rstd_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 2001);
    // dlog_setlevel(0, 3, 0);
}

TEST_F(RmsNormTiling, rms_norm_tiling_mix_dtype_1)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape = {{24, 1, 25600}, {24, 1, 25600}};
    gert::StorageShape gamma_shape = {{
                                          25600,
                                      },
                                      {
                                          25600,
                                      }};
    gert::StorageShape out_shape = {{24, 1, 25600}, {24, 1, 25600}};
    gert::StorageShape rstd_shape = {{24, 1, 1}, {24, 1, 1}};

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
    struct Tiling4RmsNormCompileInfo {
        uint32_t totalCoreNum = 40;
        uint64_t totalUbSize = 196608;
        platform_ascendc::SocVersion curSocVersion = platform_ascendc::SocVersion::ASCEND910B;
    };
    Tiling4RmsNormCompileInfo compile_info;

    std::string op_type("RmsNorm");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(compile_info_string.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    compile_info.curSocVersion = platform_ascendc::SocVersion::ASCEND910B;
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(2, 2)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&input_shape, &gamma_shape})
                      .OutputShapes({&out_shape, &rstd_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)}})
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
    ASSERT_EQ(tiling_key, 1);
    // dlog_setlevel(0, 3, 0);
}

TEST_F(RmsNormTiling, rms_norm_tiling_mix_dtype_3)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape = {{24, 1, 12288}, {24, 1, 12288}};
    gert::StorageShape gamma_shape = {{
                                          12288,
                                      },
                                      {
                                          12288,
                                      }};
    gert::StorageShape out_shape = {{24, 1, 12288}, {24, 1, 12288}};
    gert::StorageShape rstd_shape = {{24, 1, 1}, {24, 1, 1}};

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
    struct Tiling4RmsNormCompileInfo {
        uint32_t totalCoreNum = 40;
        uint64_t totalUbSize = 196608;
        platform_ascendc::SocVersion curSocVersion = platform_ascendc::SocVersion::ASCEND910B;
    };
    Tiling4RmsNormCompileInfo compile_info;

    std::string op_type("RmsNorm");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(compile_info_string.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    compile_info.curSocVersion = platform_ascendc::SocVersion::ASCEND910B;
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(2, 2)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&input_shape, &gamma_shape})
                      .OutputShapes({&out_shape, &rstd_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)}})
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
    ASSERT_EQ(tiling_key, 3);
    // dlog_setlevel(0, 3, 0);
}

TEST_F(RmsNormTiling, rms_norm_tiling_err_shape_001)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape = {{24, 1, 2560}, {24, 1, 2560}};
    gert::StorageShape gamma_shape = {{
                                          2000,
                                      },
                                      {
                                          2000,
                                      }};
    gert::StorageShape out_shape = {{24, 1, 2560}, {24, 1, 2560}};
    gert::StorageShape rstd_shape = {{24, 1, 1}, {24, 1, 1}};

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

    static optiling::Tiling4RmsNormCompileInfo compile_info = {40, 196608, platform_ascendc::SocVersion::ASCEND910B};

    std::string op_type("RmsNorm");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(compile_info_string.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(2, 2)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&input_shape, &gamma_shape})
                      .OutputShapes({&out_shape, &rstd_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)}})
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
    // dlog_setlevel(0, 3, 0);
}

TEST_F(RmsNormTiling, rms_norm_tiling_err_shape_002)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape = {{24, 0, 256}, {24, 0, 256}};
    gert::StorageShape gamma_shape = {{
                                          256,
                                      },
                                      {
                                          256,
                                      }};
    gert::StorageShape out_shape = {{24, 1, 256}, {24, 1, 256}};
    gert::StorageShape rstd_shape = {{24, 1, 1}, {24, 1, 1}};

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
    struct Tiling4RmsNormCompileInfo {
        uint32_t totalCoreNum = 40;
        uint64_t totalUbSize = 196608;
        platform_ascendc::SocVersion curSocVersion = platform_ascendc::SocVersion::ASCEND910B;
    };
    Tiling4RmsNormCompileInfo compile_info;
    std::string op_type("RmsNorm");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(compile_info_string.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    compile_info.curSocVersion = platform_ascendc::SocVersion::ASCEND910B;
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(2, 2)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&input_shape, &gamma_shape})
                      .OutputShapes({&out_shape, &rstd_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
    // dlog_setlevel(0, 3, 0);
}

TEST_F(RmsNormTiling, rms_norm_tiling_err_shape_003)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape = {{24, 1, 256}, {24, 1, 256}};
    gert::StorageShape gamma_shape = {{
                                          0,
                                      },
                                      {
                                          0,
                                      }};
    gert::StorageShape out_shape = {{24, 1, 256}, {24, 1, 256}};
    gert::StorageShape rstd_shape = {{24, 1, 1}, {24, 1, 1}};

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
    struct Tiling4RmsNormCompileInfo {
        uint32_t totalCoreNum = 40;
        uint64_t totalUbSize = 196608;
        platform_ascendc::SocVersion curSocVersion = platform_ascendc::SocVersion::ASCEND910B;
    };
    Tiling4RmsNormCompileInfo compile_info;
    std::string op_type("RmsNorm");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(compile_info_string.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    compile_info.curSocVersion = platform_ascendc::SocVersion::ASCEND910B;
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(2, 2)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&input_shape, &gamma_shape})
                      .OutputShapes({&out_shape, &rstd_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
    // dlog_setlevel(0, 3, 0);
}

TEST_F(RmsNormTiling, rms_norm_tiling_err_shape_004)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape = {{24, 1, 256}, {24, 1, 256}};
    gert::StorageShape gamma_shape = {{
                                          256,
                                      },
                                      {
                                          256,
                                      }};
    gert::StorageShape out_shape = {{24, 1, 33}, {24, 1, 33}};
    gert::StorageShape rstd_shape = {{24, 1, 1}, {24, 1, 1}};

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
    struct Tiling4RmsNormCompileInfo {
        uint32_t totalCoreNum = 40;
        uint64_t totalUbSize = 196608;
        platform_ascendc::SocVersion curSocVersion = platform_ascendc::SocVersion::ASCEND910B;
    };
    Tiling4RmsNormCompileInfo compile_info;
    std::string op_type("RmsNorm");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(compile_info_string.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    compile_info.curSocVersion = platform_ascendc::SocVersion::ASCEND910B;
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(2, 2)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&input_shape, &gamma_shape})
                      .OutputShapes({&out_shape, &rstd_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
    // dlog_setlevel(0, 3, 0);
}

TEST_F(RmsNormTiling, rms_norm_tiling_err_shape_005)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape = {{24, 1, 256}, {24, 1, 256}};
    gert::StorageShape gamma_shape = {{
                                          256,
                                      },
                                      {
                                          256,
                                      }};
    gert::StorageShape out_shape = {{24, 1, 256}, {24, 1, 256}};
    gert::StorageShape rstd_shape = {{0, 1, 1}, {0, 1, 1}};

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
    struct Tiling4RmsNormCompileInfo {
        uint32_t totalCoreNum = 40;
        uint64_t totalUbSize = 196608;
        platform_ascendc::SocVersion curSocVersion = platform_ascendc::SocVersion::ASCEND910B;
    };
    Tiling4RmsNormCompileInfo compile_info;
    std::string op_type("RmsNorm");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(compile_info_string.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    compile_info.curSocVersion = platform_ascendc::SocVersion::ASCEND910B;
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(2, 2)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&input_shape, &gamma_shape})
                      .OutputShapes({&out_shape, &rstd_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
    // dlog_setlevel(0, 3, 0);
}

namespace {
struct TilingResult {
    ge::graphStatus status = ge::GRAPH_FAILED;
    uint64_t key = 0;
    uint32_t blockDim = 0;
    size_t workspaceCount = 0;
    size_t workspace = 0;
    size_t systemWorkspace = 0;
    std::array<uint8_t, 4096> data{};
};

// Wire layouts are checked independently of the host TilingDef setter cache.
struct SplitARPacket {
    uint64_t numRow, numCol, rBlockFactor, rBlockNum, tailR, ubFactor, numRowAlign;
    float epsilon, avgFactor;
};
struct TransPacket {
    uint64_t numRow, numCol, rAligned, rTileBase, tileALen;
    float epsilon, avgFactor;
};
struct EmptyPacket {
    uint64_t numRow, usedCoreNum, rowsPerCore, rowsPerTailCore, tailCoreStartIndex, rowsPerLoop;
};
template <typename T>
T Decode(const TilingResult& result)
{
    T packet{};
    std::memcpy(&packet, result.data.data(), sizeof(packet));
    return packet;
}

TilingResult RunShapeTiling(gert::StorageShape x, gert::StorageShape gamma, gert::StorageShape rstd, ge::DataType dx,
                            ge::DataType dg, int32_t deterministic = 0, const char* opType = "RmsNorm",
                            uint32_t cores = 64, uint64_t ubSize = 253952, const char* soc = "Ascend950",
                            const char* arch = "3510")
{
    gert::StorageShape y = x;
    fe::PlatFormInfos platform;
    EXPECT_TRUE(platform.Init());
    std::map<std::string, std::string> socInfos, aicoreSpec, intrinsics;
    const char* hardware = R"({"hardware_info": {"UB_SIZE":253952, "CORE_NUM":64,
        "L2_SIZE":33554432, "L1_SIZE":524288, "L0A_SIZE":65536, "L0B_SIZE":65536, "L0C_SIZE":131072}})";
    GetPlatFormInfos(hardware, socInfos, aicoreSpec, intrinsics);
    platform.SetPlatformRes("SoCInfo", socInfos);
    platform.SetPlatformRes("AICoreSpec", aicoreSpec);
    platform.SetCoreNumByCoreType("AICore");
    std::map<std::string, std::string> version = {{"Short_SoC_version", soc}, {"NpuArch", arch}};
    platform.SetPlatformRes("version", version);
    optiling::Tiling4RmsNormCompileInfo compileInfo;
    compileInfo.totalCoreNum = cores;
    compileInfo.totalUbSize = ubSize;
    compileInfo.curSocVersion = platform_ascendc::PlatformAscendC(&platform).GetSocVersion();
    auto raw = gert::TilingData::CreateCap(4096);
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(1);
    auto* workspace = reinterpret_cast<gert::ContinuousVector*>(workspaceHolder.get());
    auto holder = gert::TilingContextFaker()
                      .SetOpType(opType)
                      .NodeIoNum(2, 2)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&x, &gamma})
                      .OutputShapes({&y, &rstd})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platform))
                      .NodeInputTd(0, dx, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, dg, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, dx, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(1e-6f)}})
                      .DeterministicLevelInfo(deterministic)
                      .TilingData(raw.get())
                      .Workspace(workspace)
                      .Build();
    auto* context = holder.GetContext<gert::TilingContext>();
    TilingResult result;
    const auto* impl = gert::OpImplRegistry::GetInstance().GetOpImpl(opType);
    EXPECT_NE(impl, nullptr);
    if (impl == nullptr) {
        return result;
    }
    result.status = impl->tiling(context);
    result.key = context->GetTilingKey();
    result.blockDim = context->GetBlockDim();
    result.systemWorkspace = platform_ascendc::PlatformAscendC(&platform).GetLibApiWorkSpaceSize();
    if (result.status == ge::GRAPH_SUCCESS) {
        result.workspaceCount = context->GetWorkspaceNum();
        if (result.workspaceCount > 0) {
            result.workspace = context->GetWorkspaceSizes(result.workspaceCount)[0];
        }
        auto* rawData = context->GetRawTilingData();
        std::memcpy(result.data.data(), rawData->GetData(), rawData->GetDataSize());
    }
    return result;
}

TilingResult RunTiling(int64_t a, int64_t r, ge::DataType dx, ge::DataType dg, int32_t deterministic = 0,
                       const char* opType = "RmsNorm", uint32_t cores = 64, uint64_t ubSize = 253952)
{
    return RunShapeTiling({{a, r}, {a, r}}, {{r}, {r}}, {{a, 1}, {a, 1}}, dx, dg, deterministic, opType, cores, ubSize);
}

const std::array<std::pair<ge::DataType, ge::DataType>, 5> DTYPES = {{{ge::DT_FLOAT16, ge::DT_FLOAT16},
                                                                      {ge::DT_BF16, ge::DT_BF16},
                                                                      {ge::DT_FLOAT, ge::DT_FLOAT},
                                                                      {ge::DT_FLOAT16, ge::DT_FLOAT},
                                                                      {ge::DT_BF16, ge::DT_FLOAT}}};

TEST(RmsNormTemplatesTiling, TransposeAdmissionAndBufferBudget)
{
    for (const auto& [dx, dg] : DTYPES) {
        const int64_t xBytes = dx == ge::DT_FLOAT ? 4 : 2;
        const int64_t gBytes = dg == ge::DT_FLOAT ? 4 : 2;
        const int64_t maxR = 32 / xBytes;
        EXPECT_EQ(RunTiling(32768, maxR, dx, dg).key, optiling::RMSNORM_REGBASE_NORMAL);
        for (int64_t r : {1L, maxR - 1, maxR}) {
            auto result = RunTiling(32769, r, dx, dg);
            ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
            ASSERT_EQ(result.key, optiling::RMSNORM_REGBASE_TRANSPOSE);
            auto data = Decode<TransPacket>(result);
            EXPECT_LE(r * xBytes, 32);
            EXPECT_LT(r * xBytes, 128);
            const uint64_t ubBytes = data.rAligned * gBytes +
                                     data.tileALen * (data.rAligned * (xBytes * 4 + sizeof(float)) + sizeof(float) * 2);
            EXPECT_LE(ubBytes, 253952 - 1024);
            EXPECT_EQ(data.tileALen % 16, 0);
            EXPECT_GT(result.blockDim, 0);
            EXPECT_LE(result.blockDim, 64);
        }
        EXPECT_EQ(RunTiling(32769, maxR + 1, dx, dg).key, optiling::RMSNORM_REGBASE_NORMAL);
        EXPECT_EQ(RunTiling(32769, 128 / xBytes, dx, dg).key, optiling::RMSNORM_REGBASE_NORMAL);
        EXPECT_EQ(RunTiling(32769, maxR, dx, dg, 3).key, optiling::RMSNORM_REGBASE_NORMAL);
    }
}

TEST(RmsNormTemplatesTiling, TransposeFullVectorReadHeadroom)
{
    constexpr uint64_t ubSize = 253952;
    constexpr uint64_t reservedBytes = 1024;
    for (const auto& [dx, dg] : DTYPES) {
        const uint64_t xBytes = dx == ge::DT_FLOAT ? 4 : 2;
        const uint64_t gBytes = dg == ge::DT_FLOAT ? 4 : 2;
        for (int64_t a : {32769L, 65537L, 242688L, 288768L, 1000001L}) {
            for (uint32_t cores : {56U, 64U}) {
                auto result = RunTiling(a, 32 / xBytes, dx, dg, 0, "RmsNorm", cores);
                ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
                ASSERT_EQ(result.key, optiling::RMSNORM_REGBASE_TRANSPOSE);
                const auto data = Decode<TransPacket>(result);
                const uint64_t allocatedBytes = data.rAligned * gBytes +
                                                data.tileALen *
                                                    (data.rAligned * (4 * xBytes + sizeof(float)) + 2 * sizeof(float));
                const uint64_t tailElements = (64 - data.tileALen % 64) % 64;
                EXPECT_EQ(data.tileALen % 16, 0);
                EXPECT_LE(tailElements, 48);
                EXPECT_LE(allocatedBytes, ubSize - reservedBytes);
                // This checks the UB-wide read window, not individual buffer padding.
                EXPECT_LE(allocatedBytes + tailElements * sizeof(float), ubSize);
            }
        }
    }
}

TEST(RmsNormTemplatesTiling, NewPerformanceTemplatesOnlyOnAscend950)
{
    for (const auto& [dx, dg] : DTYPES) {
        for (const auto& [a, r, oldKey] : {std::array<int64_t, 3>{32769, 8, optiling::RMSNORM_REGBASE_NORMAL},
                                           std::array<int64_t, 3>{3, 1048579, optiling::RMSNORM_REGBASE_SPLIT}}) {
            auto result = RunShapeTiling({{a, r}, {a, r}}, {{r}, {r}}, {{a, 1}, {a, 1}}, dx, dg, 0, "RmsNorm", 64,
                                         253952, "Ascend350", "3510");
            ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
            EXPECT_EQ(result.key, oldKey);
        }
    }
}

TEST(RmsNormTemplatesTiling, DeterministicLevelsKeepFixedReductionTemplates)
{
    for (const auto& [dx, dg] : DTYPES) {
        for (int32_t level : {0, 1, 2}) {
            EXPECT_EQ(RunTiling(32769, 8, dx, dg, level).key, optiling::RMSNORM_REGBASE_TRANSPOSE);
            EXPECT_EQ(RunTiling(3, 1048579, dx, dg, level).key, optiling::RMSNORM_REGBASE_SPLIT_AR);
        }
        EXPECT_EQ(RunTiling(32769, 8, dx, dg, 3).key, optiling::RMSNORM_REGBASE_NORMAL);
        EXPECT_EQ(RunTiling(3, 1048579, dx, dg, 3).key, optiling::RMSNORM_REGBASE_SPLIT);
    }
}

TEST(RmsNormTemplatesTiling, SplitReduceAdmissionWorkspaceAndTails)
{
    for (const auto& [dx, dg] : DTYPES) {
        for (int64_t a : {1L, 3L, 7L, 8L}) {
            auto result = RunTiling(a, 1048579, dx, dg);
            ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
            ASSERT_EQ(result.key, optiling::RMSNORM_REGBASE_SPLIT_AR);
            auto data = Decode<SplitARPacket>(result);
            EXPECT_EQ(data.rBlockNum, result.blockDim);
            EXPECT_GE(result.blockDim, 2);
            EXPECT_LE(result.blockDim, 64);
            EXPECT_LT(a * 4, 64 * 32);
            const uint64_t xBytes = dx == ge::DT_FLOAT ? 4 : 2;
            const uint64_t gBytes = dg == ge::DT_FLOAT ? 4 : 2;
            EXPECT_EQ(data.rBlockFactor * xBytes % 32, 0);
            EXPECT_EQ(data.rBlockFactor * gBytes % 32, 0);
            EXPECT_EQ((data.rBlockNum - 1) * data.rBlockFactor + data.tailR, 1048579);
            EXPECT_GT(data.tailR, 0);
            EXPECT_LE(data.tailR, data.rBlockFactor);
            EXPECT_EQ(data.ubFactor & (data.ubFactor - 1), 0);
            EXPECT_GE(data.rBlockFactor, data.ubFactor);
            EXPECT_EQ(result.workspace - result.systemWorkspace, data.rBlockNum * data.numRowAlign * 4);
        }
        auto huge = RunTiling(1, 1048579, dx, dg);
        auto data = Decode<SplitARPacket>(huge);
        const int64_t threshold = data.ubFactor * 8 * 2;
        EXPECT_EQ(RunTiling(1, threshold - 1, dx, dg).key, optiling::RMSNORM_REGBASE_SPLIT);
        EXPECT_EQ(RunTiling(1, threshold, dx, dg).key, optiling::RMSNORM_REGBASE_SPLIT_AR);
        // The fixed A cap and inherited R threshold are independent boundaries.
        // Include low-core platforms so the two-way R parallelism cap is retained.
        for (uint32_t cores : {2U, 8U, 16U, 56U, 64U}) {
            const int64_t maxA = std::min<int64_t>(8, cores / 2);
            for (int64_t a : {1L, maxA}) {
                auto reference = RunTiling(a, 1048579, dx, dg, 0, "RmsNorm", cores);
                ASSERT_EQ(reference.key, optiling::RMSNORM_REGBASE_SPLIT_AR);
                auto packet = Decode<SplitARPacket>(reference);
                const int64_t rThreshold = 16 * packet.ubFactor;
                EXPECT_EQ(RunTiling(a, rThreshold - 1, dx, dg, 0, "RmsNorm", cores).key,
                          optiling::RMSNORM_REGBASE_SPLIT);
                for (int64_t r : {rThreshold, rThreshold + 1, 2 * rThreshold - 1, 2 * rThreshold + 1}) {
                    auto boundary = RunTiling(a, r, dx, dg, 0, "RmsNorm", cores);
                    ASSERT_EQ(boundary.key, optiling::RMSNORM_REGBASE_SPLIT_AR);
                    auto boundaryData = Decode<SplitARPacket>(boundary);
                    EXPECT_GE(boundaryData.rBlockFactor, boundaryData.ubFactor);
                }
            }
            EXPECT_EQ(RunTiling(maxA + 1, 1048579, dx, dg, 0, "RmsNorm", cores).key, optiling::RMSNORM_REGBASE_SPLIT);
        }
        EXPECT_EQ(RunTiling(1, 1048579, dx, dg, 0, "RmsNorm", 1).key, optiling::RMSNORM_REGBASE_SPLIT);
        EXPECT_EQ(RunTiling(9, 1048579, dx, dg).key, optiling::RMSNORM_REGBASE_SPLIT);
        EXPECT_EQ(RunTiling(28, 1048579, dx, dg, 0, "RmsNorm", 56).key, optiling::RMSNORM_REGBASE_SPLIT);
        EXPECT_EQ(RunTiling(28, threshold, dx, dg, 0, "RmsNorm", 56).key, optiling::RMSNORM_REGBASE_SPLIT);
        EXPECT_EQ(RunTiling(33, 1048579, dx, dg).key, optiling::RMSNORM_REGBASE_SPLIT);
        EXPECT_EQ(RunTiling(1, 1048579, dx, dg, 3).key, optiling::RMSNORM_REGBASE_SPLIT);
    }
    EXPECT_EQ(RunTiling(1, (1LL << 32) + 1, ge::DT_FLOAT, ge::DT_FLOAT).key, optiling::RMSNORM_REGBASE_SPLIT_AR);
}

TEST(RmsNormTemplatesTiling, SplitReduceAvoidsRepeatedTinyTiles)
{
    for (const auto& [dx, dg] : DTYPES) {
        if (dx == ge::DT_FLOAT) {
            continue;
        }
        auto result = RunTiling(8, 262147, dx, dg, 0, "RmsNorm", 56);
        ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
        ASSERT_EQ(result.key, optiling::RMSNORM_REGBASE_SPLIT_AR);
        auto data = Decode<SplitARPacket>(result);
        // 32 个完整 tile 和一个三元素尾块，避免 31 个核各有一个 16 元素尾 tile。
        EXPECT_EQ(data.ubFactor, 8192);
        EXPECT_EQ(data.rBlockFactor, 8192);
        EXPECT_EQ(data.rBlockNum, 33);
        EXPECT_EQ(data.tailR, 3);
        EXPECT_EQ(result.blockDim, 33);
        EXPECT_EQ(result.workspace - result.systemWorkspace, 33 * data.numRowAlign * sizeof(float));
    }
}

TEST(RmsNormTemplatesTiling, SplitReduceTileAlignmentRespectsCoreLimit)
{
    for (const auto& [dx, dg] : DTYPES) {
        auto result = RunTiling(8, 1048576, dx, dg, 0, "RmsNorm", 56);
        ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
        ASSERT_EQ(result.key, optiling::RMSNORM_REGBASE_SPLIT_AR);
        auto data = Decode<SplitARPacket>(result);
        // 向下对齐需要 64 核，超过实际 56 核时必须保留原分块。
        EXPECT_EQ(data.rBlockFactor, dx == ge::DT_FLOAT ? 18728 : 18736);
        EXPECT_EQ(data.rBlockNum, 56);
        EXPECT_EQ((data.rBlockNum - 1) * data.rBlockFactor + data.tailR, 1048576);
        auto aligned = RunTiling(8, 1048576, dx, dg, 0, "RmsNorm", 64);
        ASSERT_EQ(aligned.key, optiling::RMSNORM_REGBASE_SPLIT_AR);
        auto alignedData = Decode<SplitARPacket>(aligned);
        EXPECT_EQ(alignedData.rBlockFactor, 16384);
        EXPECT_EQ(alignedData.rBlockNum, 64);
        EXPECT_EQ(alignedData.tailR, 16384);
    }
}

TEST(RmsNormTemplatesTiling, SplitReduceSingleRowKeepsBalancedPartitions)
{
    for (const auto& [dx, dg] : DTYPES) {
        const bool isFp32 = dx == ge::DT_FLOAT;
        const int64_t r = isFp32 ? 65537 : 131073;
        auto result = RunTiling(1, r, dx, dg, 0, "RmsNorm", 56);
        ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
        ASSERT_EQ(result.key, optiling::RMSNORM_REGBASE_SPLIT_AR);
        auto data = Decode<SplitARPacket>(result);
        EXPECT_EQ(data.rBlockNum, 2);
        EXPECT_EQ(data.rBlockFactor, isFp32 ? 32776 : 65552);
        EXPECT_EQ(data.tailR, isFp32 ? 32761 : 65521);
    }
}

TEST(RmsNormTemplatesTiling, EmptyReduceAndEmptyOuterDimension)
{
    for (const auto& [dx, dg] : DTYPES) {
        for (int64_t a : {0L, 1L, 7L, 8193L, 1000001L}) {
            auto result = RunTiling(a, 0, dx, dg);
            ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
            ASSERT_EQ(result.key, optiling::RMSNORM_REGBASE_REDUCE_EMPTY);
            auto data = Decode<EmptyPacket>(result);
            EXPECT_EQ(data.numRow, a);
            EXPECT_EQ(data.usedCoreNum, result.blockDim);
            EXPECT_GT(data.rowsPerLoop, 0);
            EXPECT_EQ((data.usedCoreNum - 1) * data.rowsPerCore + data.rowsPerTailCore, a);
            if (a > 0) {
                EXPECT_EQ(data.rowsPerCore % 8, 0);
            }
        }
        EXPECT_EQ(RunTiling(0, 64, dx, dg).key, optiling::RMSNORM_REGBASE_REDUCE_EMPTY);
        EXPECT_EQ(RunTiling(7, 0, dx, dg, 3).key, optiling::RMSNORM_REGBASE_REDUCE_EMPTY);
    }
}

TEST(RmsNormTemplatesTiling, EmptyReductionMultiLoopBoundaries)
{
    struct EmptyReductionCase {
        uint64_t numRow;
        uint64_t expectedRowsPerCore;
        uint64_t expectedRowsPerTailCore;
        uint64_t expectedNormalLoops;
        uint64_t expectedTailLoops;
        uint64_t expectedNormalLast;
        uint64_t expectedTailLast;
    };
    // Raw UB is 253952 B; host reserves 1024 B before the empty-reduction tiler.
    constexpr uint64_t rowsPerLoop = 63232;
    constexpr uint64_t rowsPerBlock = 8;
    for (uint32_t cores : {56U, 64U}) {
        const uint64_t boundary = cores * rowsPerLoop;
        const uint64_t firstMultiLoopTail = rowsPerLoop - rowsPerBlock * (cores - 1) + 1;
        const std::array<EmptyReductionCase, 6> cases = {{
            {boundary - 1, rowsPerLoop, rowsPerLoop - 1, 1, 1, rowsPerLoop, rowsPerLoop - 1},
            {boundary, rowsPerLoop, rowsPerLoop, 1, 1, rowsPerLoop, rowsPerLoop},
            {boundary + 1, rowsPerLoop + rowsPerBlock, firstMultiLoopTail, 2, 1, rowsPerBlock, firstMultiLoopTail},
            {cores * (rowsPerLoop + rowsPerBlock) - 1, rowsPerLoop + rowsPerBlock, rowsPerLoop + rowsPerBlock - 1, 2, 2,
             rowsPerBlock, rowsPerBlock - 1},
            {2 * boundary, 2 * rowsPerLoop, 2 * rowsPerLoop, 2, 2, rowsPerLoop, rowsPerLoop},
            {cores * (2 * rowsPerLoop + rowsPerBlock) - 1, 2 * rowsPerLoop + rowsPerBlock,
             2 * rowsPerLoop + rowsPerBlock - 1, 3, 3, rowsPerBlock, rowsPerBlock - 1},
        }};
        for (const auto& [dx, dg] : DTYPES) {
            for (const auto& testCase : cases) {
                SCOPED_TRACE(::testing::Message()
                             << "cores=" << cores << " A=" << testCase.numRow << " dx=" << dx << " dg=" << dg);
                auto result = RunTiling(testCase.numRow, 0, dx, dg, 0, "RmsNorm", cores);
                ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
                ASSERT_EQ(result.key, optiling::RMSNORM_REGBASE_REDUCE_EMPTY);
                const auto data = Decode<EmptyPacket>(result);
                EXPECT_EQ(result.blockDim, cores);
                EXPECT_EQ(data.usedCoreNum, cores);
                EXPECT_EQ(data.numRow, testCase.numRow);
                ASSERT_EQ(data.rowsPerLoop, rowsPerLoop);
                ASSERT_EQ(data.rowsPerCore, testCase.expectedRowsPerCore);
                ASSERT_EQ(data.rowsPerTailCore, testCase.expectedRowsPerTailCore);
                EXPECT_EQ(data.tailCoreStartIndex, cores - 1);
                EXPECT_EQ((cores - 1) * data.rowsPerCore + data.rowsPerTailCore, testCase.numRow);
                // Check both full-round reuse and the final DataCopyPad length.
                EXPECT_EQ((data.rowsPerCore - 1) / data.rowsPerLoop + 1, testCase.expectedNormalLoops);
                EXPECT_EQ((data.rowsPerTailCore - 1) / data.rowsPerLoop + 1, testCase.expectedTailLoops);
                EXPECT_EQ((data.rowsPerCore - 1) % data.rowsPerLoop + 1, testCase.expectedNormalLast);
                EXPECT_EQ((data.rowsPerTailCore - 1) % data.rowsPerLoop + 1, testCase.expectedTailLast);
            }
        }
    }
}

TEST(RmsNormTemplatesTiling, EmptyOutputMustMatchReducedShape)
{
    const gert::StorageShape x = {{2, 3, 0, 4}, {2, 3, 0, 4}};
    const gert::StorageShape gamma = {{0, 4}, {0, 4}};
    for (const auto& [dx, dg] : DTYPES) {
        for (const gert::StorageShape& valid :
             {gert::StorageShape{{2, 3, 1, 1}, {2, 3, 1, 1}}, gert::StorageShape{{2, 3}, {2, 3}}}) {
            auto result = RunShapeTiling(x, gamma, valid, dx, dg);
            ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
            ASSERT_EQ(result.key, optiling::RMSNORM_REGBASE_REDUCE_EMPTY);
            EXPECT_EQ(Decode<EmptyPacket>(result).numRow, 6);
        }
        for (const gert::StorageShape& invalid :
             {gert::StorageShape{{2, 3, 0, 1}, {2, 3, 0, 1}}, gert::StorageShape{{1, 1, 1, 1}, {1, 1, 1, 1}},
              gert::StorageShape{{2, 1}, {2, 1}}, gert::StorageShape{{2, 3, 1}, {2, 3, 1}},
              gert::StorageShape{{2, 3, 1, 1, 1}, {2, 3, 1, 1, 1}}}) {
            EXPECT_NE(RunShapeTiling(x, gamma, invalid, dx, dg).status, ge::GRAPH_SUCCESS);
        }
        // A==0 的合法输出为空，不能接受非空输出，也不能访问额外的 rank。
        EXPECT_EQ(RunShapeTiling({{0, 3, 4}, {0, 3, 4}}, {{4}, {4}}, {{0, 3, 1}, {0, 3, 1}}, dx, dg).status,
                  ge::GRAPH_SUCCESS);
        EXPECT_NE(RunShapeTiling({{0, 3, 4}, {0, 3, 4}}, {{4}, {4}}, {{1, 3, 1}, {1, 3, 1}}, dx, dg).status,
                  ge::GRAPH_SUCCESS);
    }
}

TEST(RmsNormTemplatesTiling, FullAxisEmptyReductionAcceptsScalarOutput)
{
    const gert::StorageShape scalar = {{}, {}};
    ASSERT_TRUE(scalar.GetStorageShape().IsScalar());
    ASSERT_EQ(scalar.GetStorageShape().GetShapeSize(), 1);
    for (const auto& [dx, dg] : DTYPES) {
        for (const gert::StorageShape& x :
             {gert::StorageShape{{0}, {0}}, gert::StorageShape{{0, 4}, {0, 4}},
              gert::StorageShape{{2, 0, 4}, {2, 0, 4}}, gert::StorageShape{{0, 0}, {0, 0}}}) {
            auto result = RunShapeTiling(x, x, scalar, dx, dg);
            ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
            ASSERT_EQ(result.key, optiling::RMSNORM_REGBASE_REDUCE_EMPTY);
            auto data = Decode<EmptyPacket>(result);
            EXPECT_EQ(data.numRow, scalar.GetStorageShape().GetShapeSize());
            EXPECT_EQ(data.usedCoreNum, 1);
            EXPECT_EQ(data.rowsPerTailCore, 1);
        }
    }
}

TEST(RmsNormTemplatesTiling, FullAxisEmptyReductionRejectsZeroElementOutput)
{
    const gert::StorageShape x = {{0}, {0}};
    ASSERT_EQ(x.GetStorageShape().GetShapeSize(), 0);
    for (const auto& [dx, dg] : DTYPES) {
        EXPECT_NE(RunShapeTiling(x, x, x, dx, dg).status, ge::GRAPH_SUCCESS);
    }
}

TEST(RmsNormTemplatesTiling, EmptyInputProductScope)
{
    for (const auto& [dx, dg] : DTYPES) {
        for (const char* soc : {"Ascend950", "Ascend350"}) {
            auto empty = RunShapeTiling({{2, 0}, {2, 0}}, {{0}, {0}}, {{2, 1}, {2, 1}}, dx, dg, 0, "RmsNorm", 56,
                                        253952, soc);
            EXPECT_EQ(empty.status == ge::GRAPH_SUCCESS, std::string(soc) == "Ascend950");
            auto nonEmpty = RunShapeTiling({{2, 16}, {2, 16}}, {{16}, {16}}, {{2, 1}, {2, 1}}, dx, dg, 0, "RmsNorm", 56,
                                           253952, soc);
            EXPECT_EQ(nonEmpty.status, ge::GRAPH_SUCCESS);
        }
        EXPECT_NE(RunShapeTiling({{2, 0}, {2, 0}}, {{0}, {0}}, {{2, 1}, {2, 1}}, dx, dg, 0, "RmsNorm", 56, 253952,
                                 "Unknown", "5102")
                      .status,
                  ge::GRAPH_SUCCESS);
        EXPECT_NE(RunTiling(2, 0, dx, dg, 0, "GemmaRmsNorm").status, ge::GRAPH_SUCCESS);
    }
}

TEST(RmsNormTemplatesTiling, WorkspaceOnlyForSplitAR)
{
    const std::array<std::pair<int64_t, int64_t>, 5> shapes = {{{0, 64}, {7, 0}, {32769, 3}, {64, 256}, {16, 1048579}}};
    const std::array<uint64_t, 5> keys = {{optiling::RMSNORM_REGBASE_REDUCE_EMPTY,
                                           optiling::RMSNORM_REGBASE_REDUCE_EMPTY, optiling::RMSNORM_REGBASE_TRANSPOSE,
                                           optiling::RMSNORM_REGBASE_NORMAL, optiling::RMSNORM_REGBASE_SPLIT}};
    for (const auto& [dx, dg] : DTYPES) {
        for (size_t i = 0; i < shapes.size(); ++i) {
            auto result = RunTiling(shapes[i].first, shapes[i].second, dx, dg);
            ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
            EXPECT_EQ(result.key, keys[i]);
            EXPECT_EQ(result.workspaceCount, 1);
            EXPECT_EQ(result.workspace, 0);
        }
        auto split = RunTiling(3, 1048579, dx, dg);
        ASSERT_EQ(split.status, ge::GRAPH_SUCCESS);
        ASSERT_EQ(split.key, optiling::RMSNORM_REGBASE_SPLIT_AR);
        EXPECT_EQ(split.workspaceCount, 1);
        auto data = Decode<SplitARPacket>(split);
        EXPECT_EQ(split.workspace, split.systemWorkspace + data.rBlockNum * data.numRowAlign * sizeof(float));
        auto deterministic = RunTiling(3, 1048579, dx, dg, 3);
        ASSERT_EQ(deterministic.status, ge::GRAPH_SUCCESS);
        EXPECT_EQ(deterministic.key, optiling::RMSNORM_REGBASE_SPLIT);
        EXPECT_EQ(deterministic.workspaceCount, 1);
        EXPECT_EQ(deterministic.workspace, 0);
    }
}

TEST(RmsNormTemplatesTiling, GemmaWorkspaceRemainsUnchanged)
{
    for (const auto& [a, r] : {std::pair<int64_t, int64_t>{32769, 8}, {1, 1048579}}) {
        auto result = RunTiling(a, r, ge::DT_FLOAT16, ge::DT_FLOAT16, 0, "GemmaRmsNorm");
        ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
        // Gemma 的 arch35 入口原本不申请 workspace。
        EXPECT_EQ(result.workspaceCount, 0);
        EXPECT_EQ(result.workspace, 0);
    }
}

TEST(RmsNormTemplatesTiling, ExistingFallbackAndInvalidPlatform)
{
    EXPECT_EQ(RunTiling(64, 256, ge::DT_FLOAT16, ge::DT_FLOAT16).key, optiling::RMSNORM_REGBASE_NORMAL);
    EXPECT_EQ(RunTiling(1, 20000, ge::DT_FLOAT16, ge::DT_FLOAT16).key, optiling::RMSNORM_REGBASE_SPLIT);
    EXPECT_EQ(RunTiling(1, 1048579, ge::DT_FLOAT16, ge::DT_FLOAT16, 0, "RmsNorm", 0).status, ge::GRAPH_FAILED);
    EXPECT_EQ(RunTiling(1, 1, ge::DT_FLOAT16, ge::DT_FLOAT16, 0, "RmsNorm", 64, 2048).status, ge::GRAPH_FAILED);
    for (uint64_t ubSize : {0UL, 512UL, 1024UL}) {
        EXPECT_EQ(RunTiling(1, 1, ge::DT_FLOAT16, ge::DT_FLOAT16, 0, "RmsNorm", 64, ubSize).status, ge::GRAPH_FAILED);
    }
    EXPECT_EQ(RunTiling(33, std::numeric_limits<uint32_t>::max(), ge::DT_FLOAT16, ge::DT_FLOAT16).status,
              ge::GRAPH_FAILED);
    EXPECT_EQ(RunTiling(32769, 8, ge::DT_FLOAT16, ge::DT_FLOAT16, 0, "GemmaRmsNorm").key,
              optiling::RMSNORM_REGBASE_NORMAL);
    EXPECT_EQ(RunTiling(1, 1048579, ge::DT_FLOAT16, ge::DT_FLOAT16, 0, "GemmaRmsNorm").key,
              optiling::RMSNORM_REGBASE_SPLIT);
    EXPECT_EQ(RunTiling(1, 0, ge::DT_FLOAT16, ge::DT_FLOAT16, 0, "GemmaRmsNorm").status, ge::GRAPH_FAILED);
}

TEST(RmsNormTemplatesTiling, NonRegbaseRstdNumelMustEqualRows)
{
    // 非 regbase 平台（910B，NpuArch=2201）：rstdOut 元素数必须等于归一化行数 x.numel / gamma.numel，
    // 否则内核按行数写 rstd 会产生越界写（issue #7333）。声明 1 个元素而 rows=8 时 tiling 应失败。
    auto bad = RunShapeTiling({{8, 128}, {8, 128}}, {{128}, {128}}, {{1, 1}, {1, 1}}, ge::DT_FLOAT16, ge::DT_FLOAT, 0,
                              "RmsNorm", 64, 253952, "Ascend910B", "2201");
    EXPECT_EQ(bad.status, ge::GRAPH_FAILED);
    auto badGemma = RunShapeTiling({{8, 128}, {8, 128}}, {{128}, {128}}, {{1, 1}, {1, 1}}, ge::DT_FLOAT16, ge::DT_FLOAT,
                                   0, "GemmaRmsNorm", 64, 253952, "Ascend910B", "2201");
    EXPECT_EQ(badGemma.status, ge::GRAPH_FAILED);

    // 合法形态（rstdOut 元素数等于行数）不被新增校验误伤。
    auto good = RunShapeTiling({{8, 128}, {8, 128}}, {{128}, {128}}, {{8, 1}, {8, 1}}, ge::DT_FLOAT16, ge::DT_FLOAT, 0,
                               "RmsNorm", 64, 253952, "Ascend910B", "2201");
    EXPECT_EQ(good.status, ge::GRAPH_SUCCESS);
    auto goodGemma = RunShapeTiling({{8, 128}, {8, 128}}, {{128}, {128}}, {{8, 1}, {8, 1}}, ge::DT_FLOAT16,
                                    ge::DT_FLOAT, 0, "GemmaRmsNorm", 64, 253952, "Ascend910B", "2201");
    EXPECT_EQ(goodGemma.status, ge::GRAPH_SUCCESS);
}
} // namespace

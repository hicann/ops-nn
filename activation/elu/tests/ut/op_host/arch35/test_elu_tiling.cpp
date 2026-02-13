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
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "platform/platform_infos_def.h"
#include "ut_op_util.h"
#include "../../../../op_host/arch35/elu_tiling_arch35.h"

using namespace ut_util;
using namespace std;
using namespace ge;

class EluTilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "EluTilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "EluTilingTest TearDown" << std::endl;
    }
};

TEST_F(EluTilingTest, test_tiling_fp16_001)
{
    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    string compile_info_string = R"({
                                    "hardware_info": {
                                        "BT_SIZE": 0,
                                        "load3d_constraints": "1",
                                        "Intrinsic_fix_pipe_l0c2out": false,
                                        "Intrinsic_data_move_l12ub": true,
                                        "Intrinsic_data_move_l0c2ub": true,
                                        "Intrinsic_data_move_out2l1_nd2nz": false,
                                        "UB_SIZE": 262144,
                                        "L2_SIZE": 33554432,
                                        "L1_SIZE": 524288,
                                        "L0A_SIZE": 65536,
                                        "L0B_SIZE": 65536,
                                        "L0C_SIZE": 131072,
                                        "CORE_NUM": 64
                                    }
                                })";
    // 获取平台信息
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // 初始化平台信息
    fe::PlatFormInfos platform_info;
    platform_info.Init();

    // 定义编译信息结构体
    optiling::EluCompileInfo compile_info;
    compile_info.coreNum = 64;
    compile_info.ubSize = 262144;

    // 获取操作符实现
    std::string op_type("Elu");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // 模拟 tilingParseFunc
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(4, 2)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();
    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "version", soc_version_infos);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    std::cout << "test>> kernel_holder.GetContext" << std::endl;

    // 定义操作符和输入性状
    gert::StorageShape Shape = {{1, 64, 2, 64}, {1, 64, 2, 64}};

    // 模拟 tilingFunc
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1})
                      .InputShapes({&Shape})
                      .OutputShapes({&Shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"alpha", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                           {"scale", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                           {"input_scale", Ops::NN::AnyValue::CreateFrom<float>(1.0)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();
    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    std::cout << tiling_key << std::endl;
    ASSERT_EQ(tiling_key, 101);
}

TEST_F(EluTilingTest, test_tiling_fp32_002)
{
    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    string compile_info_string = R"({
                                    "hardware_info": {
                                        "BT_SIZE": 0,
                                        "load3d_constraints": "1",
                                        "Intrinsic_fix_pipe_l0c2out": false,
                                        "Intrinsic_data_move_l12ub": true,
                                        "Intrinsic_data_move_l0c2ub": true,
                                        "Intrinsic_data_move_out2l1_nd2nz": false,
                                        "UB_SIZE": 262144,
                                        "L2_SIZE": 33554432,
                                        "L1_SIZE": 524288,
                                        "L0A_SIZE": 65536,
                                        "L0B_SIZE": 65536,
                                        "L0C_SIZE": 131072,
                                        "CORE_NUM": 64
                                    }
                                })";
    // 获取平台信息
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // 初始化平台信息
    fe::PlatFormInfos platform_info;
    platform_info.Init();

    optiling::EluCompileInfo compile_info;
    compile_info.coreNum = 64;
    compile_info.ubSize = 262144;

    // 获取操作符实现
    std::string op_type("Elu");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // 模拟 tilingParseFunc
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(4, 2)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();
    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "version", soc_version_infos);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    std::cout << "test>> kernel_holder.GetContext" << std::endl;

    // 定义操作符和输入性状
    gert::StorageShape Shape = {{1, 64, 2, 64}, {1, 64, 2, 64}};

    // 模拟 tilingFunc
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1})
                      .InputShapes({&Shape})
                      .OutputShapes({&Shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"alpha", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                           {"scale", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                           {"input_scale", Ops::NN::AnyValue::CreateFrom<float>(1.0)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();
    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);

    auto tiling_key = tiling_context->GetTilingKey();
    std::cout << tiling_key << std::endl;
    ASSERT_EQ(tiling_key, 101);
}

TEST_F(EluTilingTest, test_tiling_bf16_007)
{
    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    string compile_info_string = R"({
                                    "hardware_info": {
                                        "BT_SIZE": 0,
                                        "load3d_constraints": "1",
                                        "Intrinsic_fix_pipe_l0c2out": false,
                                        "Intrinsic_data_move_l12ub": true,
                                        "Intrinsic_data_move_l0c2ub": true,
                                        "Intrinsic_data_move_out2l1_nd2nz": false,
                                        "UB_SIZE": 262144,
                                        "L2_SIZE": 33554432,
                                        "L1_SIZE": 524288,
                                        "L0A_SIZE": 65536,
                                        "L0B_SIZE": 65536,
                                        "L0C_SIZE": 131072,
                                        "CORE_NUM": 64
                                    }
                                })";
    // 获取平台信息
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // 初始化平台信息
    fe::PlatFormInfos platform_info;
    platform_info.Init();

    optiling::EluCompileInfo compile_info;
    compile_info.coreNum = 64;
    compile_info.ubSize = 262144;

    // 获取操作符实现
    std::string op_type("Elu");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // 模拟 tilingParseFunc
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(4, 2)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();
    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "version", soc_version_infos);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    std::cout << "test>> kernel_holder.GetContext" << std::endl;

    // 定义操作符和输入性状
    gert::StorageShape Shape = {{1, 64, 2, 64}, {1, 64, 2, 64}};

    // 模拟 tilingFunc
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1})
                      .InputShapes({&Shape})
                      .OutputShapes({&Shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"alpha", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                           {"scale", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                           {"input_scale", Ops::NN::AnyValue::CreateFrom<float>(1.0)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();
    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);

    auto tiling_key = tiling_context->GetTilingKey();
    std::cout << tiling_key << std::endl;
    ASSERT_EQ(tiling_key, 101);
}

TEST_F(EluTilingTest, test_tiling_failed_bf16_fp32_008)
{
    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    string compile_info_string = R"({
                                    "hardware_info": {
                                        "BT_SIZE": 0,
                                        "load3d_constraints": "1",
                                        "Intrinsic_fix_pipe_l0c2out": false,
                                        "Intrinsic_data_move_l12ub": true,
                                        "Intrinsic_data_move_l0c2ub": true,
                                        "Intrinsic_data_move_out2l1_nd2nz": false,
                                        "UB_SIZE": 262144,
                                        "L2_SIZE": 33554432,
                                        "L1_SIZE": 524288,
                                        "L0A_SIZE": 65536,
                                        "L0B_SIZE": 65536,
                                        "L0C_SIZE": 131072,
                                        "CORE_NUM": 64
                                    }
                                })";
    // 获取平台信息
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // 初始化平台信息
    fe::PlatFormInfos platform_info;
    platform_info.Init();

    optiling::EluCompileInfo compile_info;
    compile_info.coreNum = 64;
    compile_info.ubSize = 262144;

    // 获取操作符实现
    std::string op_type("Elu");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // 模拟 tilingParseFunc
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(4, 2)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();
    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "version", soc_version_infos);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    std::cout << "test>> kernel_holder.GetContext" << std::endl;

    // 定义操作符和输入性状
    gert::StorageShape Shape = {{1, 64, 2, 64}, {1, 64, 2, 64}};

    // 模拟 tilingFunc
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1})
                      .InputShapes({&Shape})
                      .OutputShapes({&Shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"alpha", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                           {"scale", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                           {"input_scale", Ops::NN::AnyValue::CreateFrom<float>(1.0)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();
    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(EluTilingTest, test_tiling_failed_empty_tensor_009)
{
    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    string compile_info_string = R"({
                                    "hardware_info": {
                                        "BT_SIZE": 0,
                                        "load3d_constraints": "1",
                                        "Intrinsic_fix_pipe_l0c2out": false,
                                        "Intrinsic_data_move_l12ub": true,
                                        "Intrinsic_data_move_l0c2ub": true,
                                        "Intrinsic_data_move_out2l1_nd2nz": false,
                                        "UB_SIZE": 262144,
                                        "L2_SIZE": 33554432,
                                        "L1_SIZE": 524288,
                                        "L0A_SIZE": 65536,
                                        "L0B_SIZE": 65536,
                                        "L0C_SIZE": 131072,
                                        "CORE_NUM": 64
                                    }
                                })";
    // 获取平台信息
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // 初始化平台信息
    fe::PlatFormInfos platform_info;
    platform_info.Init();

    optiling::EluCompileInfo compile_info;
    compile_info.coreNum = 64;
    compile_info.ubSize = 262144;

    // 获取操作符实现
    std::string op_type("Elu");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // 模拟 tilingParseFunc
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(4, 2)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();
    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "version", soc_version_infos);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    std::cout << "test>> kernel_holder.GetContext" << std::endl;

    // 定义操作符和输入性状
    gert::StorageShape Shape = {{1, 0, 2, 64}, {1, 0, 2, 64}};

    // 模拟 tilingFunc
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1})
                      .InputShapes({&Shape})
                      .OutputShapes({&Shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"alpha", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                           {"scale", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                           {"input_scale", Ops::NN::AnyValue::CreateFrom<float>(1.0)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();
    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(EluTilingTest, test_tiling_failed_unsupport_type_010)
{
    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    string compile_info_string = R"({
                                    "hardware_info": {
                                        "BT_SIZE": 0,
                                        "load3d_constraints": "1",
                                        "Intrinsic_fix_pipe_l0c2out": false,
                                        "Intrinsic_data_move_l12ub": true,
                                        "Intrinsic_data_move_l0c2ub": true,
                                        "Intrinsic_data_move_out2l1_nd2nz": false,
                                        "UB_SIZE": 262144,
                                        "L2_SIZE": 33554432,
                                        "L1_SIZE": 524288,
                                        "L0A_SIZE": 65536,
                                        "L0B_SIZE": 65536,
                                        "L0C_SIZE": 131072,
                                        "CORE_NUM": 64
                                    }
                                })";
    // 获取平台信息
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // 初始化平台信息
    fe::PlatFormInfos platform_info;
    platform_info.Init();

    optiling::EluCompileInfo compile_info;
    compile_info.coreNum = 64;
    compile_info.ubSize = 262144;

    // 获取操作符实现
    std::string op_type("Elu");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // 模拟 tilingParseFunc
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(4, 2)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();
    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "version", soc_version_infos);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    std::cout << "test>> kernel_holder.GetContext" << std::endl;

    // 定义操作符和输入性状
    gert::StorageShape Shape = {{1, 1, 2, 64}, {1, 1, 2, 64}};

    // 模拟 tilingFunc
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1})
                      .InputShapes({&Shape})
                      .OutputShapes({&Shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_DOUBLE, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_DOUBLE, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"alpha", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                           {"scale", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                           {"input_scale", Ops::NN::AnyValue::CreateFrom<float>(1.0)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();
    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(EluTilingTest, test_tiling_failed_diff_shape_010)
{
    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    string compile_info_string = R"({
                                    "hardware_info": {
                                        "BT_SIZE": 0,
                                        "load3d_constraints": "1",
                                        "Intrinsic_fix_pipe_l0c2out": false,
                                        "Intrinsic_data_move_l12ub": true,
                                        "Intrinsic_data_move_l0c2ub": true,
                                        "Intrinsic_data_move_out2l1_nd2nz": false,
                                        "UB_SIZE": 262144,
                                        "L2_SIZE": 33554432,
                                        "L1_SIZE": 524288,
                                        "L0A_SIZE": 65536,
                                        "L0B_SIZE": 65536,
                                        "L0C_SIZE": 131072,
                                        "CORE_NUM": 64
                                    }
                                })";
    // 获取平台信息
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // 初始化平台信息
    fe::PlatFormInfos platform_info;
    platform_info.Init();

    optiling::EluCompileInfo compile_info;
    compile_info.coreNum = 64;
    compile_info.ubSize = 262144;

    // 获取操作符实现
    std::string op_type("Elu");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // 模拟 tilingParseFunc
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(4, 2)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();
    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "version", soc_version_infos);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    std::cout << "test>> kernel_holder.GetContext" << std::endl;

    // 定义操作符和输入性状
    gert::StorageShape Shape1 = {{1, 1, 2, 64}, {1, 1, 2, 64}};
    gert::StorageShape Shape2 = {{1, 1, 1, 64}, {1, 1, 1, 64}};

    // 模拟 tilingFunc
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1})
                      .InputShapes({&Shape1})
                      .OutputShapes({&Shape2})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"alpha", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                           {"scale", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                           {"input_scale", Ops::NN::AnyValue::CreateFrom<float>(1.0)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();
    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}
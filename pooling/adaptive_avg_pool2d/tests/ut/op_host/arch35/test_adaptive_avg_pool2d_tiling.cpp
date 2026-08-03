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
 * \file test_adaptive_avg_pool2d_tiling.cpp
 * \brief Tiling测试 - AdaptiveAvgPool2dTiling950Test
 */

#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include <string>

#include "../../../../op_host/arch35/adaptive_avg_pool2d_tiling.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "platform/platform_infos_def.h"
#include "ut_op_util.h"

using namespace ut_util;
using namespace std;
using namespace ge;

static void SetAscend950GlobalPlatformInfo()
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;

    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";

    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
}

class AdaptiveAvgPool2dTiling950Test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AdaptiveAvgPool2dTiling950Test SetUp" << std::endl;
        SetAscend950GlobalPlatformInfo();
    }

    static void TearDownTestCase() { std::cout << "AdaptiveAvgPool2dTiling950Test TearDown" << std::endl; }
};

static void ExecuteAdaptiveAvgPool2d950TestCase(gert::StorageShape xShape, gert::StorageShape yShape,
                                                std::vector<int64_t> outputSize, ge::DataType dtype,
                                                uint64_t expect_tiling_key)
{
    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64}
                          })";

    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};

    fe::PlatFormInfos platform_info;
    platform_info.Init();

    optiling::AdaptiveAvgPool2dCompileInfo compile_info;

    std::string op_type("AdaptiveAvgPool2d");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(1, 1)
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version",
                                                                                            soc_version_infos);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto tiling_data = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(tiling_data, nullptr);

    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1})
                      .InputShapes({&xShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, dtype, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeOutputTd(0, dtype, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeAttrs({{"output_size", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(outputSize)}})
                      .TilingData(tiling_data.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version_infos);

    auto ret = tiling_func(tiling_context);

    ASSERT_EQ(ret, ge::GRAPH_SUCCESS);

    auto real_tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(real_tiling_key, expect_tiling_key);

    auto raw_tiling = tiling_context->GetRawTilingData();
    ASSERT_NE(raw_tiling, nullptr);
}

static void ExecuteAdaptiveAvgPool2d950FailTestCase(gert::StorageShape xShape, gert::StorageShape yShape,
                                                    std::vector<int64_t> outputSize, ge::DataType dtype,
                                                    ge::graphStatus expect_result)
{
    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64}
                          })";

    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};

    fe::PlatFormInfos platform_info;
    platform_info.Init();

    optiling::AdaptiveAvgPool2dCompileInfo compile_info;

    std::string op_type("AdaptiveAvgPool2d");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(1, 1)
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version",
                                                                                            soc_version_infos);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto tiling_data = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(tiling_data, nullptr);

    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1})
                      .InputShapes({&xShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, dtype, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeOutputTd(0, dtype, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeAttrs({{"output_size", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(outputSize)}})
                      .TilingData(tiling_data.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version_infos);

    auto ret = tiling_func(tiling_context);

    ASSERT_EQ(ret, expect_result);
}

TEST_F(AdaptiveAvgPool2dTiling950Test, test_tiling_invalid_dtype_int32)
{
    gert::StorageShape x_shape = {{2, 3, 4, 4}, {2, 3, 4, 4}};
    gert::StorageShape y_shape = {{2, 3, 1, 1}, {2, 3, 1, 1}};
    ExecuteAdaptiveAvgPool2d950FailTestCase(x_shape, y_shape, {1, 1}, ge::DT_INT32, ge::GRAPH_FAILED);
}

TEST_F(AdaptiveAvgPool2dTiling950Test, test_tiling_invalid_input_dims_2d)
{
    gert::StorageShape x_shape = {{4, 4}, {4, 4}};
    gert::StorageShape y_shape = {{1, 1}, {1, 1}};
    ExecuteAdaptiveAvgPool2d950FailTestCase(x_shape, y_shape, {1, 1}, ge::DT_FLOAT, ge::GRAPH_FAILED);
}

TEST_F(AdaptiveAvgPool2dTiling950Test, test_tiling_invalid_input_dims_5d)
{
    gert::StorageShape x_shape = {{1, 2, 3, 4, 4}, {1, 2, 3, 4, 4}};
    gert::StorageShape y_shape = {{1, 2, 3, 1, 1}, {1, 2, 3, 1, 1}};
    ExecuteAdaptiveAvgPool2d950FailTestCase(x_shape, y_shape, {1, 1}, ge::DT_FLOAT, ge::GRAPH_FAILED);
}

TEST_F(AdaptiveAvgPool2dTiling950Test, test_tiling_invalid_input_shape_zero_dim)
{
    gert::StorageShape x_shape = {{0, 3, 4, 4}, {0, 3, 4, 4}};
    gert::StorageShape y_shape = {{0, 3, 1, 1}, {0, 3, 1, 1}};
    ExecuteAdaptiveAvgPool2d950FailTestCase(x_shape, y_shape, {1, 1}, ge::DT_FLOAT, ge::GRAPH_FAILED);
}

TEST_F(AdaptiveAvgPool2dTiling950Test, test_tiling_invalid_output_size_len_3)
{
    gert::StorageShape x_shape = {{2, 3, 4, 4}, {2, 3, 4, 4}};
    gert::StorageShape y_shape = {{2, 3, 1, 1}, {2, 3, 1, 1}};
    ExecuteAdaptiveAvgPool2d950FailTestCase(x_shape, y_shape, {1, 1, 1}, ge::DT_FLOAT, ge::GRAPH_FAILED);
}

TEST_F(AdaptiveAvgPool2dTiling950Test, test_tiling_invalid_output_size_negative)
{
    gert::StorageShape x_shape = {{2, 3, 4, 4}, {2, 3, 4, 4}};
    gert::StorageShape y_shape = {{2, 3, -1, -1}, {2, 3, -1, -1}};
    ExecuteAdaptiveAvgPool2d950FailTestCase(x_shape, y_shape, {-1, -1}, ge::DT_FLOAT, ge::GRAPH_FAILED);
}

TEST_F(AdaptiveAvgPool2dTiling950Test, test_tiling_invalid_out_dims_2d)
{
    gert::StorageShape x_shape = {{2, 3, 4, 4}, {2, 3, 4, 4}};
    gert::StorageShape y_shape = {{1, 1}, {1, 1}};
    ExecuteAdaptiveAvgPool2d950FailTestCase(x_shape, y_shape, {1, 1}, ge::DT_FLOAT, ge::GRAPH_FAILED);
}

TEST_F(AdaptiveAvgPool2dTiling950Test, test_tiling_invalid_out_shape_mismatch)
{
    gert::StorageShape x_shape = {{2, 3, 4, 4}, {2, 3, 4, 4}};
    gert::StorageShape y_shape = {{2, 5, 1, 1}, {2, 5, 1, 1}};
    ExecuteAdaptiveAvgPool2d950FailTestCase(x_shape, y_shape, {1, 1}, ge::DT_FLOAT, ge::GRAPH_FAILED);
}

TEST_F(AdaptiveAvgPool2dTiling950Test, test_tiling_invalid_output_dtype_int32)
{
    gert::StorageShape x_shape = {{2, 32, 16, 16}, {2, 32, 16, 16}};
    gert::StorageShape y_shape = {{2, 32, 8, 8}, {2, 32, 8, 8}};
    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);
    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    optiling::AdaptiveAvgPool2dCompileInfo compile_info;
    std::string op_type("AdaptiveAvgPool2d");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(1, 1)
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version",
                                                                                            soc_version_infos);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
    auto tiling_data = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(tiling_data, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1})
                      .InputShapes({&x_shape})
                      .OutputShapes({&y_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeAttrs({{"output_size", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({8, 8})}})
                      .TilingData(tiling_data.get())
                      .Workspace(ws_size)
                      .Build();
    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version_infos);
    auto ret = tiling_func(tiling_context);
    ASSERT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(AdaptiveAvgPool2dTiling950Test, test_tiling_parse_invalid_core_num_zero)
{
    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 0}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);
    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    optiling::AdaptiveAvgPool2dCompileInfo compile_info;
    std::string op_type("AdaptiveAvgPool2d");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(1, 1)
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version",
                                                                                            soc_version_infos);
    auto ret = tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>());
    ASSERT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(AdaptiveAvgPool2dTiling950Test, test_tiling_parse_invalid_ub_size_zero)
{
    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 0, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);
    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    optiling::AdaptiveAvgPool2dCompileInfo compile_info;
    std::string op_type("AdaptiveAvgPool2d");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(1, 1)
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version",
                                                                                            soc_version_infos);
    auto ret = tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>());
    ASSERT_EQ(ret, ge::GRAPH_FAILED);
}

// 测试用例1: simt - 4D
TEST_F(AdaptiveAvgPool2dTiling950Test, test_simt_4d_enough)
{
    gert::StorageShape x_shape = {{3, 1, 79, 109}, {3, 1, 79, 109}};
    gert::StorageShape y_shape = {{3, 1, 68, 46}, {3, 1, 68, 46}};
    ExecuteAdaptiveAvgPool2d950TestCase(x_shape, y_shape, {68, 46}, ge::DT_FLOAT16, 2);
}

// 测试用例2: simt - 3D
TEST_F(AdaptiveAvgPool2dTiling950Test, test_simt_3d_enough)
{
    gert::StorageShape x_shape = {{2, 9226, 3}, {2, 9226, 3}};
    gert::StorageShape y_shape = {{2, 88, 53}, {2, 88, 53}};
    ExecuteAdaptiveAvgPool2d950TestCase(x_shape, y_shape, {88, 53}, ge::DT_FLOAT16, 2);
}

// 测试用例1: FP32 小kernel 2x2，n*c=64 >= 32
TEST_F(AdaptiveAvgPool2dTiling950Test, test_small_kernel_fp32_2x2)
{
    gert::StorageShape x_shape = {{2, 32, 2, 2}, {2, 32, 2, 2}}; // n*c = 64
    gert::StorageShape y_shape = {{2, 32, 1, 1}, {2, 32, 1, 1}};
    ExecuteAdaptiveAvgPool2d950TestCase(x_shape, y_shape, {1, 1}, ge::DT_FLOAT, 0);
}

// 测试用例2: FP16 3x3 -> 2x2，n*c=64
// W上采样? 否 (wOut=2<=wIn=3); H下采样 hOut=2<hIn=3 → SplitH(优先级1) 先于 SmallKernel 接管
// fp16 ncFactor=128 != VRegSize/sizeof(float)=64 → TPL_NC_FACTOR_128 → key=5|(1<<6)=69
TEST_F(AdaptiveAvgPool2dTiling950Test, test_small_kernel_fp16_3x3)
{
    gert::StorageShape x_shape = {{1, 64, 3, 3}, {1, 64, 3, 3}}; // n*c = 64
    gert::StorageShape y_shape = {{1, 64, 2, 2}, {1, 64, 2, 2}};
    ExecuteAdaptiveAvgPool2d950TestCase(x_shape, y_shape, {2, 2}, ge::DT_FLOAT16, 69);
}

// 测试用例3: BF16 小kernel 4x4，n*c=128 >= 32
TEST_F(AdaptiveAvgPool2dTiling950Test, test_small_kernel_bf16_4x4)
{
    gert::StorageShape x_shape = {{4, 32, 4, 4}, {4, 32, 4, 4}}; // n*c = 128
    gert::StorageShape y_shape = {{4, 32, 1, 1}, {4, 32, 1, 1}};
    ExecuteAdaptiveAvgPool2d950TestCase(x_shape, y_shape, {1, 1}, ge::DT_BF16, 64);
}

// 测试用例4: kernel 大小在限制内 (8x8 = 64 < 128)，但 H 下采样 → SplitH 优先接管
// kernelHMax = ceil(16/2) = 8, kernelWMax = ceil(16/2) = 8, 8*8=64 < 128
// hOut=2 < hIn=16 → SplitH(优先级1); fp32 ncFactor=64 → TPL_NC_FACTOR_64 → key=5
TEST_F(AdaptiveAvgPool2dTiling950Test, test_small_kernel_boundary_8x8)
{
    gert::StorageShape x_shape = {{1, 32, 16, 16}, {1, 32, 16, 16}}; // n*c = 32
    gert::StorageShape y_shape = {{1, 32, 2, 2}, {1, 32, 2, 2}};
    ExecuteAdaptiveAvgPool2d950TestCase(x_shape, y_shape, {2, 2}, ge::DT_FLOAT, 5);
}

// 测试用例5: 非方形输出，n*c=64；H 下采样 (3<6) → SplitH 优先接管，fp32 → key=5
TEST_F(AdaptiveAvgPool2dTiling950Test, test_small_kernel_non_square_output)
{
    gert::StorageShape x_shape = {{1, 64, 6, 8}, {1, 64, 6, 8}}; // n*c = 64
    gert::StorageShape y_shape = {{1, 64, 3, 4}, {1, 64, 3, 4}};
    ExecuteAdaptiveAvgPool2d950TestCase(x_shape, y_shape, {3, 4}, ge::DT_FLOAT, 5);
}

// UpsampleH guard: wOut=1 触发 isWOutValid=false → UpsampleH 拒绝
// hOut=4 > hIn=2 → UpsampleH tries; wOut=1 → reject; H上采样 SplitH 也不适用
// → 落 SplitC(优先级5)，fp32 ncFactor=64 → key=3
TEST_F(AdaptiveAvgPool2dTiling950Test, test_upsample_h_reject_wout_eq1)
{
    gert::StorageShape x_shape = {{1, 64, 2, 4}, {1, 64, 2, 4}};
    gert::StorageShape y_shape = {{1, 64, 4, 1}, {1, 64, 4, 1}};
    ExecuteAdaptiveAvgPool2d950TestCase(x_shape, y_shape, {4, 1}, ge::DT_FLOAT, 3);
}

// SplitW guard: wOut=1 触发 isWOutValid=false → SplitW 拒绝
// kWMax=ceil(32/1)=32 >= 32 → SplitW tries; wOut=1 → reject
// H下采样 (2<4) → SplitH(优先级1) 先接管，fp32 → key=5
TEST_F(AdaptiveAvgPool2dTiling950Test, test_split_w_reject_wout_eq1)
{
    gert::StorageShape x_shape = {{1, 64, 4, 32}, {1, 64, 4, 32}};
    gert::StorageShape y_shape = {{1, 64, 2, 1}, {1, 64, 2, 1}};
    ExecuteAdaptiveAvgPool2d950TestCase(x_shape, y_shape, {2, 1}, ge::DT_FLOAT, 5);
}

// SplitH guard: hIn > 90*hOut*wOut triggers isDmaPerOutputTooHigh, fallback to BigKernel/SIMT
// wOut=5 > wIn=3 → SplitH tries; 1000 > 90*2*5=900 → reject; kH*kW=500 → not SmallKernel
TEST_F(AdaptiveAvgPool2dTiling950Test, test_split_h_reject_dma_per_output_too_high)
{
    gert::StorageShape x_shape = {{1, 64, 1000, 3}, {1, 64, 1000, 3}};
    gert::StorageShape y_shape = {{1, 64, 2, 5}, {1, 64, 2, 5}};
    ExecuteAdaptiveAvgPool2d950TestCase(x_shape, y_shape, {2, 5}, ge::DT_FLOAT, 1);
}

// SplitH guard: hOut<=1 && kH*kW<128 triggers isSingleRowSmallKernel, falls to SmallKernel
// wOut=5 > wIn=3 → SplitH tries; hOut=1, kH*kW=10*1=10 < 128 → reject → SmallKernel
TEST_F(AdaptiveAvgPool2dTiling950Test, test_split_h_reject_single_row_small_kernel)
{
    gert::StorageShape x_shape = {{1, 64, 10, 3}, {1, 64, 10, 3}};
    gert::StorageShape y_shape = {{1, 64, 1, 5}, {1, 64, 1, 5}};
    ExecuteAdaptiveAvgPool2d950TestCase(x_shape, y_shape, {1, 5}, ge::DT_FLOAT, 0);
}

// SplitC guard: kernelWMax>=1000 triggers isWKernelBounded=false, fallback to BigKernel/SIMT
// SplitW fails IsMeetUbSize (wIn=3000 too large); SmallKernel rejects (kH*kW=3000);
// SplitC: kWMax=1500 >= 1000 → reject
TEST_F(AdaptiveAvgPool2dTiling950Test, test_split_c_reject_w_kernel_unbounded)
{
    gert::StorageShape x_shape = {{1, 64, 4, 3000}, {1, 64, 4, 3000}};
    gert::StorageShape y_shape = {{1, 64, 2, 2}, {1, 64, 2, 2}};
    ExecuteAdaptiveAvgPool2d950TestCase(x_shape, y_shape, {2, 2}, ge::DT_FLOAT, 129);
}

// ===================== 正向选中测试 =====================

// UpsampleH 正向选中: hOut>hIn(H上采样) + wOut<=wIn(W下采样) + 小wIn
TEST_F(AdaptiveAvgPool2dTiling950Test, test_upsample_h_positive_select)
{
    gert::StorageShape x_shape = {{1, 64, 4, 8}, {1, 64, 4, 8}};
    gert::StorageShape y_shape = {{1, 64, 16, 4}, {1, 64, 16, 4}};
    ExecuteAdaptiveAvgPool2d950TestCase(x_shape, y_shape, {16, 4}, ge::DT_FLOAT, 6);
}

// SplitH 正向选中: wOut>wIn(W上采样) + hOut<hIn(H下采样)
TEST_F(AdaptiveAvgPool2dTiling950Test, test_split_h_positive_select)
{
    gert::StorageShape x_shape = {{1, 64, 16, 4}, {1, 64, 16, 4}};
    gert::StorageShape y_shape = {{1, 64, 8, 8}, {1, 64, 8, 8}};
    ExecuteAdaptiveAvgPool2d950TestCase(x_shape, y_shape, {8, 8}, ge::DT_FLOAT, 5);
}

// kWMax>=32(大W kernel) + W下采样，但 H 也下采样 (2<4)
// → SplitH(优先级1) 先于 SplitW(优先级3) 接管，fp32 → key=5
TEST_F(AdaptiveAvgPool2dTiling950Test, test_split_w_positive_select)
{
    gert::StorageShape x_shape = {{1, 64, 4, 128}, {1, 64, 4, 128}};
    gert::StorageShape y_shape = {{1, 64, 2, 4}, {1, 64, 2, 4}};
    ExecuteAdaptiveAvgPool2d950TestCase(x_shape, y_shape, {2, 4}, ge::DT_FLOAT, 5);
}

// SplitC 正向选中: 大wIn导致SplitW UB溢出 + kH*kW>=128导致SmallKernel拒绝
TEST_F(AdaptiveAvgPool2dTiling950Test, test_split_c_positive_select)
{
    gert::StorageShape x_shape = {{1, 64, 8, 600}, {1, 64, 8, 600}};
    gert::StorageShape y_shape = {{1, 64, 4, 4}, {1, 64, 4, 4}};
    ExecuteAdaptiveAvgPool2d950TestCase(x_shape, y_shape, {4, 4}, ge::DT_FLOAT, 3);
}

// BigKernel 正向选中: NC<32拒绝所有Split + kH*kW>=128拒绝SmallKernel
TEST_F(AdaptiveAvgPool2dTiling950Test, test_big_kernel_positive_select)
{
    gert::StorageShape x_shape = {{2, 8, 32, 32}, {2, 8, 32, 32}};
    gert::StorageShape y_shape = {{2, 8, 2, 2}, {2, 8, 2, 2}};
    ExecuteAdaptiveAvgPool2d950TestCase(x_shape, y_shape, {2, 2}, ge::DT_FLOAT, 129);
}

// ===================== 额外拒绝路径测试 =====================

// SplitH guard: isSmallKernelBetter - H低倍率上采样+小NC+小kernel → SplitH 拒绝
// hOut=12 > hIn=8 且 < 2*8=16, NC=32 < vfLen=64, kH*kW=1 < 128
// H上采样且 wOut=8 > wIn=4 → UpsampleH(优先级0) 接管，fp32 → key=6
TEST_F(AdaptiveAvgPool2dTiling950Test, test_split_h_reject_small_kernel_better)
{
    gert::StorageShape x_shape = {{1, 32, 8, 4}, {1, 32, 8, 4}};
    gert::StorageShape y_shape = {{1, 32, 12, 8}, {1, 32, 12, 8}};
    ExecuteAdaptiveAvgPool2d950TestCase(x_shape, y_shape, {12, 8}, ge::DT_FLOAT, 6);
}

// SplitC guard: isHOutValid - hOut<=1 单行输出 → SplitC拒绝，回退BigKernel
// kH*kW=8*150=1200 → SmallKernel拒绝; SplitW UB溢出; SplitC: hOut=1 → reject
TEST_F(AdaptiveAvgPool2dTiling950Test, test_split_c_reject_hout_invalid)
{
    gert::StorageShape x_shape = {{1, 64, 8, 600}, {1, 64, 8, 600}};
    gert::StorageShape y_shape = {{1, 64, 1, 4}, {1, 64, 1, 4}};
    ExecuteAdaptiveAvgPool2d950TestCase(x_shape, y_shape, {1, 4}, ge::DT_FLOAT, 129);
}

// SplitC: hIn=4 且 NC=64；kH*kW=2*150=300 → SmallKernel拒绝; SplitW UB溢出
// hOut=2 < hIn=4 且 hIn 不满足 SplitH 的 DMA 代价约束 → 由 SplitC(优先级5) 接管
// fp32 ncFactor=64 → key=3
TEST_F(AdaptiveAvgPool2dTiling950Test, test_split_c_reject_shape_inefficient)
{
    gert::StorageShape x_shape = {{1, 64, 4, 600}, {1, 64, 4, 600}};
    gert::StorageShape y_shape = {{1, 64, 2, 4}, {1, 64, 2, 4}};
    ExecuteAdaptiveAvgPool2d950TestCase(x_shape, y_shape, {2, 4}, ge::DT_FLOAT, 3);
}

// SmallKernel guard: isSimtFallbackHUpWDown 正向选中 (bf16_NCHW_random_000178)
// H上采样(25>3) + W下采样(25<=2408) + NC=24 < vfLen/2=64 → 所有向量化模板拒绝，原本落Simt;
// kH*kW=2*98=196 >= 128 且 wIn=2408 <= 4096 → 放宽两个门限，由 SmallKernel 接管
// bf16 后处理 (kH*kW=196>32 且 hi*wi=97<256) 将 ncFactor 降为 64
// = GetVRegSize/sizeof(float) → TPL_NC_FACTOR_64=0，各字段全 0 → tiling_key=0
TEST_F(AdaptiveAvgPool2dTiling950Test, test_small_kernel_simt_fallback_h_up_w_down)
{
    gert::StorageShape x_shape = {{6, 4, 3, 2408}, {6, 4, 3, 2408}};
    gert::StorageShape y_shape = {{6, 4, 25, 25}, {6, 4, 25, 25}};
    ExecuteAdaptiveAvgPool2d950TestCase(x_shape, y_shape, {25, 25}, ge::DT_BF16, 0);
}

// SmallKernel guard: isSimtFallbackHUpWDown 负向 - kH*kW < 128 不被捕获 (bf16_NCHW_random_000353)
// H上采样(73>25) + W下采样(69<=2095) + NC=56 < 64，但 kH*kW=2*32=64 < 128
// → guard 不成立，isNcLenEnough 仍为 false，保持原有 Simt 路由 (tiling_key=2)
TEST_F(AdaptiveAvgPool2dTiling950Test, test_small_kernel_simt_fallback_reject_small_kernel_size)
{
    gert::StorageShape x_shape = {{1, 56, 25, 2095}, {1, 56, 25, 2095}};
    gert::StorageShape y_shape = {{1, 56, 73, 69}, {1, 56, 73, 69}};
    ExecuteAdaptiveAvgPool2d950TestCase(x_shape, y_shape, {73, 69}, ge::DT_BF16, 2);
}

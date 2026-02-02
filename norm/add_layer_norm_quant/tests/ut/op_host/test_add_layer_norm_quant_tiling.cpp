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
#include "log/log.h"
#include "ut_op_util.h"
#include "platform/platform_infos_def.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"

using namespace ut_util;
using namespace std;
using namespace ge;

class AddLayerNormQuantTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AddLayerNormQuantTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AddLayerNormQuantTiling TearDown" << std::endl;
    }
};

TEST_F(AddLayerNormQuantTiling, add_layer_norm_quant_tiling_001)
{
    //dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape = {{24, 1, 11264}, {24, 1, 11264}};
    gert::StorageShape gamma_shape = {
        {
            11264,
        },
        {
            11264,
        }};
    gert::StorageShape out_shape = {{24, 1, 11264}, {24, 1, 11264}};
    gert::StorageShape reduce_shape = {{24, 1, 1}, {24, 1, 1}};

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
    struct AddLayerNormQuantCompileInfo {
        uint32_t aivCoreNum_ = 0;
        uint32_t sysWorkspaceSize_ = 0;
        uint64_t ubSize_ = 0;
        uint32_t vecRegSize_ = 0;
        uint32_t blockSize_ = 0;
        bool isRegbase = false;
    };
    AddLayerNormQuantCompileInfo compile_info;

    std::string op_type("AddLayerNormQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder =
        gert::TilingContextFaker()
            .SetOpType(op_type)
            .NodeIoNum(7, 5)
            .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
            .InputShapes(
                {&input_shape, &input_shape, &gamma_shape, &gamma_shape, &gamma_shape, &gamma_shape, &gamma_shape})
            .OutputShapes({&out_shape, &out_shape, &out_shape, &reduce_shape, &reduce_shape})
            .CompileInfo(&compile_info)
            .PlatformInfo(reinterpret_cast<char*>(&platform_info))
            .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(1, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeAttrs(
                {{"quant_mode", Ops::NN::AnyValue::CreateFrom<string>("dynamic")},
                 {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)},
                 {"additional_output", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
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
    ASSERT_EQ(tiling_key, 1122);
    //dlog_setlevel(0, 3, 0);
}

TEST_F(AddLayerNormQuantTiling, add_layer_norm_quant_tiling_002)
{
    //dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape = {{24, 1, 30000}, {24, 1, 30000}};
    gert::StorageShape gamma_shape = {
        {
            30000,
        },
        {
            30000,
        }};
    gert::StorageShape out_shape = {{24, 1, 30000}, {24, 1, 30000}};
    gert::StorageShape reduce_shape = {{24, 1, 1}, {24, 1, 1}};

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
    struct AddLayerNormQuantCompileInfo {
        uint32_t aivCoreNum_ = 0;
        uint32_t sysWorkspaceSize_ = 0;
        uint64_t ubSize_ = 0;
        uint32_t vecRegSize_ = 0;
        uint32_t blockSize_ = 0;
        bool isRegbase = false;
    };
    AddLayerNormQuantCompileInfo compile_info;

    std::string op_type("AddLayerNormQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder =
        gert::TilingContextFaker()
            .SetOpType(op_type)
            .NodeIoNum(7, 5)
            .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
            .InputShapes(
                {&input_shape, &input_shape, &gamma_shape, &gamma_shape, &gamma_shape, &gamma_shape, &gamma_shape})
            .OutputShapes({&out_shape, &out_shape, &out_shape, &reduce_shape, &reduce_shape})
            .CompileInfo(&compile_info)
            .PlatformInfo(reinterpret_cast<char*>(&platform_info))
            .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(5, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(6, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(1, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeAttrs(
                {{"quant_mode", Ops::NN::AnyValue::CreateFrom<string>("dynamic")},
                 {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)},
                 {"additional_output", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
            .TilingData(param.get())
            .Workspace(ws_size)
            .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // CutD return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
    //dlog_setlevel(0, 3, 0);
}

TEST_F(AddLayerNormQuantTiling, add_layer_norm_quant_tiling_regbase_fullload_001)
{
    //dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape = {{1, 1024, 768}, {1, 1024, 768}};
    gert::StorageShape gamma_shape = {{768}, {768}};
    gert::StorageShape out_shape = {{1, 1024, 768}, {1, 1024, 768}};
    gert::StorageShape reduce_shape = {{1, 1024}, {1, 1024}};

    std::map<std::string, std::string> soc_infos;
    std::map<std::string, std::string> aicore_spec;
    std::map<std::string, std::string> intrinsics;
    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}};
    map<string, string> npuarchs = {{"NpuArch", "3510"}};
    std::string compile_info_string = R"({
      "hardware_info": {
        "BT_SIZE": 0, "load3d_constraints": "1",
        "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
        "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
        "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
        "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 64
      }
    })";
    std::string op_type("AddLayerNormQuant");

    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();

    struct AddLayerNormQuantCompileInfo {
        uint32_t aivCoreNum_ = 0;
        uint32_t sysWorkspaceSize_ = 0;
        uint64_t ubSize_ = 0;
        uint32_t vecRegSize_ = 0;
        uint32_t blockSize_ = 0;
        bool isRegbase = false;
    };
    AddLayerNormQuantCompileInfo compile_info;

    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl("AddLayerNormQuant")->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl("AddLayerNormQuant")->tiling_parse;
    auto gen_simplifiedkey_func = gert::OpImplRegistry::GetInstance().GetOpImpl("AddLayerNormQuant")->gen_simplifiedkey;

    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
    ASSERT_NE(param, nullptr);

    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(9, 5)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&input_shape, &input_shape, &gamma_shape, &gamma_shape, &gamma_shape, &gamma_shape,
                           &gamma_shape, &gamma_shape, &gamma_shape})
                      .OutputShapes({&out_shape, &out_shape, &out_shape, &reduce_shape, &reduce_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"quant_mode", Ops::NN::AnyValue::CreateFrom<string>("static")},
                           {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01f)},
                           {"additional_output", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                           {"div_mode", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    ge::char_t simplifiedKey[100];
    ASSERT_EQ(gen_simplifiedkey_func(tiling_context, simplifiedKey), ge::GRAPH_SUCCESS);

    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    tiling_context->GetPlatformInfo()->SetPlatformRes("version", npuarchs);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 8012);
    auto block_dim = tiling_context->GetBlockDim();
    ASSERT_EQ(block_dim, 64);

    //dlog_setlevel(0, 3, 0);
}

TEST_F(AddLayerNormQuantTiling, add_layer_norm_qaunt_tiling_regbase_welford_001)
{
    //dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape = {{1, 1024, 8192}, {1, 1024, 8192}};
    gert::StorageShape gamma_shape = {{8192}, {8192}};
    gert::StorageShape out_shape = {{1, 1024, 8192}, {1, 1024, 8192}};
    gert::StorageShape reduce_shape = {{1, 1024}, {1, 1024}};

    std::map<std::string, std::string> soc_infos;
    std::map<std::string, std::string> aicore_spec;
    std::map<std::string, std::string> intrinsics;
    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}};
    map<string, string> npuarchs = {{"NpuArch", "3510"}};
    std::string compile_info_string = R"({
      "hardware_info": {
        "BT_SIZE": 0, "load3d_constraints": "1",
        "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
        "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
        "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
        "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 64
      }
    })";
    std::string op_type("AddLayerNormQuant");

    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();

    struct AddLayerNormQuantCompileInfo {
        uint32_t aivCoreNum_ = 0;
        uint32_t sysWorkspaceSize_ = 0;
        uint64_t ubSize_ = 0;
        uint32_t vecRegSize_ = 0;
        uint32_t blockSize_ = 0;
        bool isRegbase = false;
    };
    AddLayerNormQuantCompileInfo compile_info;

    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl("AddLayerNormQuant")->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl("AddLayerNormQuant")->tiling_parse;

    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
    ASSERT_NE(param, nullptr);

    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(9, 5)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&input_shape, &input_shape, &gamma_shape, &gamma_shape, &gamma_shape, &gamma_shape,
                           &gamma_shape, &gamma_shape, &gamma_shape})
                      .OutputShapes({&out_shape, &out_shape, &out_shape, &reduce_shape, &reduce_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"quant_mode", Ops::NN::AnyValue::CreateFrom<string>("static")},
                           {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01f)},
                           {"additional_output", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                           {"div_mode", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    tiling_context->GetPlatformInfo()->SetPlatformRes("version", npuarchs);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 8112);
    auto block_dim = tiling_context->GetBlockDim();
    ASSERT_EQ(block_dim, 64);
    //dlog_setlevel(0, 3, 0);
}

TEST_F(AddLayerNormQuantTiling, add_layer_norm_quant_tiling_regbase_dyn_fullload_001)
{
    //dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape = {{1, 1024, 768}, {1, 1024, 768}};
    gert::StorageShape gamma_shape = {{768}, {768}};
    gert::StorageShape out_shape = {{1, 1024, 768}, {1, 1024, 768}};
    gert::StorageShape reduce_shape = {{1, 1024}, {1, 1024}};

    std::map<std::string, std::string> soc_infos;
    std::map<std::string, std::string> aicore_spec;
    std::map<std::string, std::string> intrinsics;
    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}};
    map<string, string> npuarchs = {{"NpuArch", "3510"}};
    std::string compile_info_string = R"({
      "hardware_info": {
        "BT_SIZE": 0, "load3d_constraints": "1",
        "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
        "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
        "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
        "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 64
      }
    })";
    std::string op_type("AddLayerNormQuant");

    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();

    struct AddLayerNormQuantCompileInfo {
        uint32_t aivCoreNum_ = 0;
        uint32_t sysWorkspaceSize_ = 0;
        uint64_t ubSize_ = 0;
        uint32_t vecRegSize_ = 0;
        uint32_t blockSize_ = 0;
        bool isRegbase = false;
    };
    AddLayerNormQuantCompileInfo compile_info;

    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl("AddLayerNormQuant")->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl("AddLayerNormQuant")->tiling_parse;
    auto gen_simplifiedkey_func = gert::OpImplRegistry::GetInstance().GetOpImpl("AddLayerNormQuant")->gen_simplifiedkey;

    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
    ASSERT_NE(param, nullptr);

    auto holder =
        gert::TilingContextFaker()
            .SetOpType(op_type)
            .NodeIoNum(7, 5)
            .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
            .InputShapes(
                {&input_shape, &input_shape, &gamma_shape, &gamma_shape, &gamma_shape, &gamma_shape, &gamma_shape})
            .OutputShapes({&out_shape, &out_shape, &out_shape, &reduce_shape, &reduce_shape})
            .CompileInfo(&compile_info)
            .PlatformInfo(reinterpret_cast<char*>(&platform_info))
            .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(3, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(4, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(5, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(6, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(1, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeAttrs(
                {{"quant_mode", Ops::NN::AnyValue::CreateFrom<string>("dynamic")},
                 {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01f)},
                 {"additional_output", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                 {"div_mode", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
            .TilingData(param.get())
            .Workspace(ws_size)
            .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    ge::char_t simplifiedKey[100];
    ASSERT_EQ(gen_simplifiedkey_func(tiling_context, simplifiedKey), ge::GRAPH_SUCCESS);

    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    tiling_context->GetPlatformInfo()->SetPlatformRes("version", npuarchs);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 8022);
    auto block_dim = tiling_context->GetBlockDim();
    ASSERT_EQ(block_dim, 64);

    //dlog_setlevel(0, 3, 0);
}

TEST_F(AddLayerNormQuantTiling, add_layer_norm_qaunt_tiling_regbase_dyn_welford_001)
{
    //dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape = {{1, 1024, 16000}, {1, 1024, 16000}};
    gert::StorageShape gamma_shape = {{16000}, {16000}};
    gert::StorageShape out_shape = {{1, 1024, 16000}, {1, 1024, 16000}};
    gert::StorageShape reduce_shape = {{1, 1024}, {1, 1024}};

    std::map<std::string, std::string> soc_infos;
    std::map<std::string, std::string> aicore_spec;
    std::map<std::string, std::string> intrinsics;
    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}};
    map<string, string> npuarchs = {{"NpuArch", "3510"}};
    std::string compile_info_string = R"({
      "hardware_info": {
        "BT_SIZE": 0, "load3d_constraints": "1",
        "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
        "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
        "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
        "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 64
      }
    })";
    std::string op_type("AddLayerNormQuant");

    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();

    struct AddLayerNormQuantCompileInfo {
        uint32_t aivCoreNum_ = 0;
        uint32_t sysWorkspaceSize_ = 0;
        uint64_t ubSize_ = 0;
        uint32_t vecRegSize_ = 0;
        uint32_t blockSize_ = 0;
        bool isRegbase = false;
    };
    AddLayerNormQuantCompileInfo compile_info;

    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl("AddLayerNormQuant")->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl("AddLayerNormQuant")->tiling_parse;

    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
    ASSERT_NE(param, nullptr);

    auto holder =
        gert::TilingContextFaker()
            .SetOpType(op_type)
            .NodeIoNum(7, 5)
            .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
            .InputShapes(
                {&input_shape, &input_shape, &gamma_shape, &gamma_shape, &gamma_shape, &gamma_shape, &gamma_shape})
            .OutputShapes({&out_shape, &out_shape, &out_shape, &reduce_shape, &reduce_shape})
            .CompileInfo(&compile_info)
            .PlatformInfo(reinterpret_cast<char*>(&platform_info))
            .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(3, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(4, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(5, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(6, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(1, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeAttrs(
                {{"quant_mode", Ops::NN::AnyValue::CreateFrom<string>("dynamic")},
                 {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01f)},
                 {"additional_output", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                 {"div_mode", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
            .TilingData(param.get())
            .Workspace(ws_size)
            .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    tiling_context->GetPlatformInfo()->SetPlatformRes("version", npuarchs);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 8122);
    auto block_dim = tiling_context->GetBlockDim();
    ASSERT_EQ(block_dim, 64);
    //dlog_setlevel(0, 3, 0);
}

TEST_F(AddLayerNormQuantTiling, add_layer_norm_quant_empty_tiling_9)
{
    //dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape = {{1, 24, 0}, {1, 24, 0}};
    gert::StorageShape gamma_shape = {{0}, {0}};
    gert::StorageShape out_shape = {{1, 24, 0}, {1, 24, 0}};
    gert::StorageShape reduce_shape = {{1, 24}, {1, 24}};

    std::map<std::string, std::string> soc_infos;
    std::map<std::string, std::string> aicore_spec;
    std::map<std::string, std::string> intrinsics;
    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}};
    map<string, string> npuarchs = {{"NpuArch", "3510"}};
    std::string compile_info_string = R"({
      "hardware_info": {
        "BT_SIZE": 0, "load3d_constraints": "1",
        "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
        "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
        "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
        "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 64
      }
    })";
    std::string op_type("AddLayerNormQuant");

    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();

    struct AddLayerNormQuantCompileInfo {
        uint32_t aivCoreNum_ = 0;
        uint32_t sysWorkspaceSize_ = 0;
        uint64_t ubSize_ = 0;
        uint32_t vecRegSize_ = 0;
        uint32_t blockSize_ = 0;
        bool isRegbase = false;
    };
    AddLayerNormQuantCompileInfo compile_info;

    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl("AddLayerNormQuant")->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl("AddLayerNormQuant")->tiling_parse;
    auto gen_simplifiedkey_func = gert::OpImplRegistry::GetInstance().GetOpImpl("AddLayerNormQuant")->gen_simplifiedkey;

    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
    ASSERT_NE(param, nullptr);

    auto holder =
        gert::TilingContextFaker()
            .SetOpType(op_type)
            .NodeIoNum(7, 5)
            .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
            .InputShapes(
                {&input_shape, &input_shape, &gamma_shape, &gamma_shape, &gamma_shape, &gamma_shape, &gamma_shape})
            .OutputShapes({&out_shape, &out_shape, &out_shape, &reduce_shape, &reduce_shape})
            .CompileInfo(&compile_info)
            .PlatformInfo(reinterpret_cast<char*>(&platform_info))
            .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(3, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(4, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(5, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(6, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(1, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeAttrs(
                {{"quant_mode", Ops::NN::AnyValue::CreateFrom<std::string>("dynamic")},
                 {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01f)},
                 {"additional_output", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                 {"div_mode", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
            .TilingData(param.get())
            .Workspace(ws_size)
            .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    ge::char_t simplifiedKey[100];
    ASSERT_EQ(gen_simplifiedkey_func(tiling_context, simplifiedKey), ge::GRAPH_SUCCESS);

    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    tiling_context->GetPlatformInfo()->SetPlatformRes("version", npuarchs);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    auto tiling_key = tiling_context->GetTilingKey();
    std::cout << "tiling_key: " << tiling_key << endl;
    ASSERT_EQ(tiling_key, 9);
}

TEST_F(AddLayerNormQuantTiling, add_layer_norm_quant_empty_tiling_9_large_size_240000)
{
    //dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape = {{1, 240000, 0}, {1, 240000, 0}};
    gert::StorageShape gamma_shape = {{0}, {0}};
    gert::StorageShape out_shape = {{1, 240000, 0}, {1, 240000, 0}};
    gert::StorageShape reduce_shape = {{1, 240000}, {1, 240000}};

    std::map<std::string, std::string> soc_infos;
    std::map<std::string, std::string> aicore_spec;
    std::map<std::string, std::string> intrinsics;
    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}};
    map<string, string> npuarchs = {{"NpuArch", "3510"}};
    std::string compile_info_string = R"({
      "hardware_info": {
        "BT_SIZE": 0, "load3d_constraints": "1",
        "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
        "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
        "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
        "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 64
      }
    })";
    std::string op_type("AddLayerNormQuant");

    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();

    struct AddLayerNormQuantCompileInfo {
        uint32_t aivCoreNum_ = 0;
        uint32_t sysWorkspaceSize_ = 0;
        uint64_t ubSize_ = 0;
        uint32_t vecRegSize_ = 0;
        uint32_t blockSize_ = 0;
        bool isRegbase = false;
    };
    AddLayerNormQuantCompileInfo compile_info;

    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl("AddLayerNormQuant")->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl("AddLayerNormQuant")->tiling_parse;
    auto gen_simplifiedkey_func = gert::OpImplRegistry::GetInstance().GetOpImpl("AddLayerNormQuant")->gen_simplifiedkey;

    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
    ASSERT_NE(param, nullptr);

    auto holder =
        gert::TilingContextFaker()
            .SetOpType(op_type)
            .NodeIoNum(7, 5)
            .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
            .InputShapes(
                {&input_shape, &input_shape, &gamma_shape, &gamma_shape, &gamma_shape, &gamma_shape, &gamma_shape})
            .OutputShapes({&out_shape, &out_shape, &out_shape, &reduce_shape, &reduce_shape})
            .CompileInfo(&compile_info)
            .PlatformInfo(reinterpret_cast<char*>(&platform_info))
            .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(3, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(4, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(5, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(6, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(1, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeAttrs(
                {{"quant_mode", Ops::NN::AnyValue::CreateFrom<std::string>("dynamic")},
                 {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01f)},
                 {"additional_output", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                 {"div_mode", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
            .TilingData(param.get())
            .Workspace(ws_size)
            .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    ge::char_t simplifiedKey[100];
    ASSERT_EQ(gen_simplifiedkey_func(tiling_context, simplifiedKey), ge::GRAPH_SUCCESS);

    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    tiling_context->GetPlatformInfo()->SetPlatformRes("version", npuarchs);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    auto tiling_key = tiling_context->GetTilingKey();
    std::cout << "tiling_key: " << tiling_key << endl;
    ASSERT_EQ(tiling_key, 9);
}

TEST_F(AddLayerNormQuantTiling, add_layer_norm_qaunt_tiling_reShape_00)
{
    //dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape = {{1, 1024, 8192}, {1, 1024, 8192}};
    gert::StorageShape gamma_shape = {{8192}, {8192}};
    gert::StorageShape optional_shape = {{1, 1, 8192}, {1, 1, 8192}};
    gert::StorageShape out_shape = {{1, 1024, 8192}, {1, 1024, 8192}};
    gert::StorageShape reduce_shape = {{1, 1024}, {1, 1024}};

    std::map<std::string, std::string> soc_infos;
    std::map<std::string, std::string> aicore_spec;
    std::map<std::string, std::string> intrinsics;
    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}};
    map<string, string> npuarchs = {{"NpuArch", "3510"}};
    std::string compile_info_string = R"({
      "hardware_info": {
        "BT_SIZE": 0, "load3d_constraints": "1",
        "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
        "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
        "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
        "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 64
      }
    })";
    std::string op_type("AddLayerNormQuant");

    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();

    struct AddLayerNormQuantCompileInfo {
        uint32_t aivCoreNum_ = 0;
        uint32_t sysWorkspaceSize_ = 0;
        uint64_t ubSize_ = 0;
        uint32_t vecRegSize_ = 0;
        uint32_t blockSize_ = 0;
        bool isRegbase = false;
    };
    AddLayerNormQuantCompileInfo compile_info;

    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl("AddLayerNormQuant")->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl("AddLayerNormQuant")->tiling_parse;

    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
    ASSERT_NE(param, nullptr);

    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(9, 5)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&input_shape, &input_shape, &gamma_shape, &gamma_shape, &gamma_shape, &optional_shape,
                           &optional_shape, &optional_shape, &optional_shape})
                      .OutputShapes({&out_shape, &out_shape, &out_shape, &reduce_shape, &reduce_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"quant_mode", Ops::NN::AnyValue::CreateFrom<string>("static")},
                           {"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01f)},
                           {"additional_output", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                           {"div_mode", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    tiling_context->GetPlatformInfo()->SetPlatformRes("version", npuarchs);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 8112);
    auto block_dim = tiling_context->GetBlockDim();
    ASSERT_EQ(block_dim, 64);
}

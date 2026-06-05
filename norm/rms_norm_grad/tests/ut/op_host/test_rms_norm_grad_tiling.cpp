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
#include "platform/platform_infos_def.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "ut_op_util.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"

using namespace ut_util;
using namespace std;
using namespace ge;

class RmsNormGradTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "RmsNormGradTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "RmsNormGradTiling TearDown" << std::endl;
    }
};
struct RmsNormCompileInfo {};

TEST_F(RmsNormGradTiling, rms_norm_grad_tiling_001)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape = {{16, 16}, {16, 16}};
    gert::StorageShape input1_shape = {{16, 16}, {16, 16}};
    gert::StorageShape input2_shape = {{16}, {16}};
    gert::StorageShape input3_shape = {{16}, {16}};
    gert::StorageShape out_shape = {{16, 16}, {16, 16}};
    gert::StorageShape out1_shape = {{16}, {16}};

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

    RmsNormCompileInfo compile_info;

    std::string op_type("RmsNormGrad");
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
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(4, 2)
                      .IrInstanceNum({4})
                      .InputShapes({&input_shape, &input1_shape, &input2_shape, &input3_shape})
                      .OutputShapes({&out_shape, &out1_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
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
    // dlog_setlevel(0, 3, 0);
}

TEST_F(RmsNormGradTiling, rms_norm_grad_tiling_002)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape = {{16, 12288}, {16, 12288}};
    gert::StorageShape input1_shape = {{16, 12288}, {16, 12288}};
    gert::StorageShape input2_shape = {{16}, {16}};
    gert::StorageShape input3_shape = {{12288}, {12288}};
    gert::StorageShape out_shape = {{16, 12288}, {16, 12288}};
    gert::StorageShape out1_shape = {{12288}, {12288}};

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

    RmsNormCompileInfo compile_info;

    std::string op_type("RmsNormGrad");
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
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(4, 2)
                      .IrInstanceNum({4})
                      .InputShapes({&input_shape, &input1_shape, &input2_shape, &input3_shape})
                      .OutputShapes({&out_shape, &out1_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
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
    // dlog_setlevel(0, 3, 0);
}

TEST_F(RmsNormGradTiling, rms_norm_grad_tiling_error_core_num)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape = {{16, 12288}, {16, 12288}};
    gert::StorageShape input1_shape = {{16, 12288}, {16, 12288}};
    gert::StorageShape input2_shape = {{16}, {16}};
    gert::StorageShape input3_shape = {{12288}, {12288}};
    gert::StorageShape out_shape = {{16, 12288}, {16, 12288}};
    gert::StorageShape out1_shape = {{12288}, {12288}};

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 0}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info

    RmsNormCompileInfo compile_info;

    std::string op_type("RmsNormGrad");
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
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(4, 2)
                      .IrInstanceNum({4})
                      .InputShapes({&input_shape, &input1_shape, &input2_shape, &input3_shape})
                      .OutputShapes({&out_shape, &out1_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
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
    // auto tiling_key = tiling_context->GetTilingKey();
    // dlog_setlevel(0, 3, 0);
}

TEST_F(RmsNormGradTiling, rms_norm_grad_tiling_003_bf16)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape = {{130, 16}, {130, 16}};
    gert::StorageShape input1_shape = {{130, 16}, {130, 16}};
    gert::StorageShape input2_shape = {{130}, {130}};
    gert::StorageShape input3_shape = {{16}, {16}};
    gert::StorageShape out_shape = {{130, 16}, {130, 16}};
    gert::StorageShape out1_shape = {{16}, {16}};

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

    RmsNormCompileInfo compile_info;

    std::string op_type("RmsNormGrad");
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
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(4, 2)
                      .IrInstanceNum({4})
                      .InputShapes({&input_shape, &input1_shape, &input2_shape, &input3_shape})
                      .OutputShapes({&out_shape, &out1_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
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
    // dlog_setlevel(0, 3, 0);
}

TEST_F(RmsNormGradTiling, rms_norm_grad_tiling_004_bf16)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape = {{16, 12290}, {16, 12290}};
    gert::StorageShape input1_shape = {{16, 12290}, {16, 12290}};
    gert::StorageShape input2_shape = {{16}, {16}};
    gert::StorageShape input3_shape = {{12290}, {12290}};
    gert::StorageShape out_shape = {{16, 12290}, {16, 12290}};
    gert::StorageShape out1_shape = {{12290}, {12290}};

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

    RmsNormCompileInfo compile_info;

    std::string op_type("RmsNormGrad");
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
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(4, 2)
                      .IrInstanceNum({4})
                      .InputShapes({&input_shape, &input1_shape, &input2_shape, &input3_shape})
                      .OutputShapes({&out_shape, &out1_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
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
    // dlog_setlevel(0, 3, 0);
}

TEST_F(RmsNormGradTiling, rms_norm_grad_tiling_error_core_num_1)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape = {{16, 3000}, {16, 3000}};
    gert::StorageShape input1_shape = {{16, 3000}, {16, 3000}};
    gert::StorageShape input2_shape = {{16}, {16}};
    gert::StorageShape input3_shape = {{3000}, {3000}};
    gert::StorageShape out_shape = {{16, 3000}, {16, 3000}};
    gert::StorageShape out1_shape = {{3000}, {3000}};

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 0}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info

    RmsNormCompileInfo compile_info;

    std::string op_type("RmsNormGrad");
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
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(4, 2)
                      .IrInstanceNum({4})
                      .InputShapes({&input_shape, &input1_shape, &input2_shape, &input3_shape})
                      .OutputShapes({&out_shape, &out1_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
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
    // auto tiling_key = tiling_context->GetTilingKey();
    // dlog_setlevel(0, 3, 0);
}

TEST_F(RmsNormGradTiling, rms_norm_grad_tiling_err_shape_001)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape = {{16, 16}, {16, 16}};
    gert::StorageShape input1_shape = {{14, 14}, {14, 14}};
    gert::StorageShape input2_shape = {{16}, {16}};
    gert::StorageShape input3_shape = {{16}, {16}};
    gert::StorageShape out_shape = {{16, 16}, {16, 16}};
    gert::StorageShape out1_shape = {{16}, {16}};

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

    RmsNormCompileInfo compile_info;

    std::string op_type("RmsNormGrad");
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
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(4, 2)
                      .IrInstanceNum({4})
                      .InputShapes({&input_shape, &input1_shape, &input2_shape, &input3_shape})
                      .OutputShapes({&out_shape, &out1_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
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

TEST_F(RmsNormGradTiling, rms_norm_grad_tiling_err_shape_002)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape = {{16, 16}, {16, 16}};
    gert::StorageShape input1_shape = {{16, 16}, {16, 16}};
    gert::StorageShape input2_shape = {{10}, {10}};
    gert::StorageShape input3_shape = {{16}, {16}};
    gert::StorageShape out_shape = {{16, 16}, {16, 16}};
    gert::StorageShape out1_shape = {{16}, {16}};

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

    RmsNormCompileInfo compile_info;

    std::string op_type("RmsNormGrad");
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
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(4, 2)
                      .IrInstanceNum({4})
                      .InputShapes({&input_shape, &input1_shape, &input2_shape, &input3_shape})
                      .OutputShapes({&out_shape, &out1_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
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

TEST_F(RmsNormGradTiling, rms_norm_grad_tiling_err_shape_003)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape = {{16, 16}, {16, 16}};
    gert::StorageShape input1_shape = {{16, 16}, {16, 16}};
    gert::StorageShape input2_shape = {{16}, {16}};
    gert::StorageShape input3_shape = {{10}, {10}};
    gert::StorageShape out_shape = {{16, 16}, {16, 16}};
    gert::StorageShape out1_shape = {{16}, {16}};

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

    RmsNormCompileInfo compile_info;

    std::string op_type("RmsNormGrad");
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
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(4, 2)
                      .IrInstanceNum({4})
                      .InputShapes({&input_shape, &input1_shape, &input2_shape, &input3_shape})
                      .OutputShapes({&out_shape, &out1_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
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

TEST_F(RmsNormGradTiling, rms_norm_grad_tiling_small_rstd)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape = {{16, 1, 12288}, {16, 1, 12288}};
    gert::StorageShape input1_shape = {{16, 1, 12288}, {16, 1, 12288}};
    gert::StorageShape input2_shape = {{16}, {16}};
    gert::StorageShape input3_shape = {{12288}, {12288}};
    gert::StorageShape out_shape = {{16, 1, 12288}, {16, 1, 12288}};
    gert::StorageShape out1_shape = {{12288}, {12288}};

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

    RmsNormCompileInfo compile_info;

    std::string op_type("RmsNormGrad");
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
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(4, 2)
                      .IrInstanceNum({4})
                      .InputShapes({&input_shape, &input1_shape, &input2_shape, &input3_shape})
                      .OutputShapes({&out_shape, &out1_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
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
    // dlog_setlevel(0, 3, 0);
}

TEST_F(RmsNormGradTiling, rms_norm_grad_tiling_regbase_001)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape = {{16, 3000}, {16, 3000}};
    gert::StorageShape input1_shape = {{16, 3000}, {16, 3000}};
    gert::StorageShape input2_shape = {{16}, {16}};
    gert::StorageShape input3_shape = {{3000}, {3000}};
    gert::StorageShape out_shape = {{16, 3000}, {16, 3000}};
    gert::StorageShape out1_shape = {{3000}, {3000}};

    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
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

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info

    RmsNormCompileInfo compile_info;

    std::string op_type("RmsNormGrad");
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
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(4, 2)
                      .IrInstanceNum({4})
                      .InputShapes({&input_shape, &input1_shape, &input2_shape, &input3_shape})
                      .OutputShapes({&out_shape, &out1_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
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
    // dlog_setlevel(0, 3, 0);
}

TEST_F(RmsNormGradTiling, rms_norm_grad_tiling_regbase_002)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape = {{512, 8192}, {512, 8192}};
    gert::StorageShape input1_shape = {{512, 8192}, {512, 8192}};
    gert::StorageShape input2_shape = {{512}, {512}};
    gert::StorageShape input3_shape = {{8192}, {8192}};
    gert::StorageShape out_shape = {{512, 8192}, {512, 8192}};
    gert::StorageShape out1_shape = {{8192}, {8192}};

    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
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

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info

    RmsNormCompileInfo compile_info;

    std::string op_type("RmsNormGrad");
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
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(4, 2)
                      .IrInstanceNum({4})
                      .InputShapes({&input_shape, &input1_shape, &input2_shape, &input3_shape})
                      .OutputShapes({&out_shape, &out1_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
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
    // dlog_setlevel(0, 3, 0);
}

TEST_F(RmsNormGradTiling, rms_norm_grad_empty_tiling_001)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape = {{630, 0, 710}, {630, 0, 710}};
    gert::StorageShape input1_shape = {{630, 0, 710}, {630, 0, 710}};
    gert::StorageShape input2_shape = {{630, 0}, {630, 0}};
    gert::StorageShape input3_shape = {{710}, {710}};
    gert::StorageShape out_shape = {{630, 0, 710}, {630, 0, 710}};
    gert::StorageShape out1_shape = {{710}, {710}};
    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    map<string, string> npuarchs = {{"NpuArch", "3510"}};
    string compile_info_string = R"({
      "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                        "Intrinsic_fix_pipe_l0c2out": false,
                        "Intrinsic_data_move_l12ub": true,
                        "Intrinsic_data_move_l0c2ub": true,
                        "Intrinsic_data_move_out2l1_nd2nz": false,
                        "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                        "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                        "CORE_NUM": 64,"socVersion":"Ascend950"}
                        })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_Version;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_Version);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info

    RmsNormCompileInfo compile_info;

    std::string op_type("RmsNormGrad");
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_Version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(4, 2)
                      .IrInstanceNum({4})
                      .InputShapes({&input_shape, &input1_shape, &input2_shape, &input3_shape})
                      .OutputShapes({&out_shape, &out1_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
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

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
}

TEST_F(RmsNormGradTiling, rms_norm_grad_empty_tiling_002)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape = {{630, 0, 9000}, {630, 0, 9000}};
    gert::StorageShape input1_shape = {{630, 0, 9000}, {630, 0, 9000}};
    gert::StorageShape input2_shape = {{630, 0}, {630, 0}};
    gert::StorageShape input3_shape = {{9000}, {9000}};
    gert::StorageShape out_shape = {{630, 0, 9000}, {630, 0, 9000}};
    gert::StorageShape out1_shape = {{9000}, {9000}};
    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    map<string, string> npuarchs = {{"NpuArch", "3510"}};
    string compile_info_string = R"({
      "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                        "Intrinsic_fix_pipe_l0c2out": false,
                        "Intrinsic_data_move_l12ub": true,
                        "Intrinsic_data_move_l0c2ub": true,
                        "Intrinsic_data_move_out2l1_nd2nz": false,
                        "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                        "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                        "CORE_NUM": 64,"socVersion":"Ascend950"}
                        })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_Version;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_Version);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info

    RmsNormCompileInfo compile_info;

    std::string op_type("RmsNormGrad");
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_Version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(4, 2)
                      .IrInstanceNum({4})
                      .InputShapes({&input_shape, &input1_shape, &input2_shape, &input3_shape})
                      .OutputShapes({&out_shape, &out1_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
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

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
}

TEST_F(RmsNormGradTiling, rms_norm_grad_empty_tiling_003)
{
    // dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape = {{630, 0, 20000000}, {630, 0, 20000000}};
    gert::StorageShape input1_shape = {{630, 0, 20000000}, {630, 0, 20000000}};
    gert::StorageShape input2_shape = {{630, 0}, {630, 0}};
    gert::StorageShape input3_shape = {{20000000}, {20000000}};
    gert::StorageShape out_shape = {{630, 0, 20000000}, {630, 0, 20000000}};
    gert::StorageShape out1_shape = {{20000000}, {20000000}};
    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    map<string, string> npuarchs = {{"NpuArch", "3510"}};
    string compile_info_string = R"({
      "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                        "Intrinsic_fix_pipe_l0c2out": false,
                        "Intrinsic_data_move_l12ub": true,
                        "Intrinsic_data_move_l0c2ub": true,
                        "Intrinsic_data_move_out2l1_nd2nz": false,
                        "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                        "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                        "CORE_NUM": 64,"socVersion":"Ascend950"}
                        })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_Version;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_Version);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info

    RmsNormCompileInfo compile_info;

    std::string op_type("RmsNormGrad");
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_Version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(4, 2)
                      .IrInstanceNum({4})
                      .InputShapes({&input_shape, &input1_shape, &input2_shape, &input3_shape})
                      .OutputShapes({&out_shape, &out1_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
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

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
}



TEST_F(RmsNormGradTiling, rms_norm_grad_tiling_regbase_bigm_001)
{
    dlog_setlevel(0, 0, 1);
    gert::StorageShape input_shape = {{4096, 30}, {4096, 30}};
    gert::StorageShape input1_shape = {{4096, 30}, {4096, 30}};
    gert::StorageShape input2_shape = {{4096}, {4096}};
    gert::StorageShape input3_shape = {{30}, {30}};
    gert::StorageShape out_shape = {{4096, 30}, {4096, 30}};
    gert::StorageShape out1_shape = {{30}, {30}};

    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    string compile_info_string = R"({
      "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                        "Intrinsic_fix_pipe_l0c2out": false,
                        "Intrinsic_data_move_l12ub": true,
                        "Intrinsic_data_move_l0c2ub": true,
                        "Intrinsic_data_move_out2l1_nd2nz": false,
                        "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                        "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                        "CORE_NUM": 64, "socVersion": "Ascend950"}
                        })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_version;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info

    RmsNormCompileInfo compile_info;

    std::string op_type("RmsNormGrad");
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
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
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(4, 2)
                      .IrInstanceNum({4})
                      .InputShapes({&input_shape, &input1_shape, &input2_shape, &input3_shape})
                      .OutputShapes({&out_shape, &out1_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
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
    dlog_setlevel(0, 3, 0);
}

static std::string regbase_compile_info_950 = R"({
  "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                    "Intrinsic_fix_pipe_l0c2out": false,
                    "Intrinsic_data_move_l12ub": true,
                    "Intrinsic_data_move_l0c2ub": true,
                    "Intrinsic_data_move_out2l1_nd2nz": false,
                    "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                    "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                    "CORE_NUM": 64, "socVersion": "Ascend950"}
                    })";

static void SetupRegbasePlatformInfo(fe::PlatFormInfos& platform_info,
                                      map<string, string>& soc_infos,
                                      map<string, string>& aicore_spec,
                                      map<string, string>& intrinsics,
                                      map<string, string>& soc_version)
{
    GetPlatFormInfos(regbase_compile_info_950.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);
    platform_info.Init();
}

static void SetupRegbaseTilingParseContext(
    gert::KernelRunContextHolder& kernel_holder,
    fe::PlatFormInfos& platform_info,
    map<string, string>& soc_infos,
    map<string, string>& aicore_spec,
    map<string, string>& intrinsics,
    map<string, string>& soc_version,
    map<string, string>& npuarchs)
{
    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);
}

static void SetupRegbaseTilingContext(
    gert::KernelRunContextHolder& holder,
    map<string, string>& soc_infos,
    map<string, string>& aicore_spec,
    map<string, string>& intrinsics,
    map<string, string>& npuarchs)
{
    auto* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    tiling_context->GetPlatformInfo()->SetPlatformRes("version", npuarchs);
}

TEST_F(RmsNormGradTiling, rms_norm_grad_regbase_fullload_nosplit_fp16)
{
    gert::StorageShape dy_shape = {{8, 128}, {8, 128}};
    gert::StorageShape x_shape = {{8, 128}, {8, 128}};
    gert::StorageShape rstd_shape = {{8}, {8}};
    gert::StorageShape gamma_shape = {{128}, {128}};
    gert::StorageShape dx_shape = {{8, 128}, {8, 128}};
    gert::StorageShape dgamma_shape = {{128}, {128}};

    map<string, string> soc_infos, aicore_spec, intrinsics, soc_version;
    map<string, string> npuarchs = {{{"NpuArch", "3510"}}};
    fe::PlatFormInfos platform_info;
    SetupRegbasePlatformInfo(platform_info, soc_infos, aicore_spec, intrinsics, soc_version);

    RmsNormCompileInfo compile_info;
    std::string op_type("RmsNormGrad");
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(regbase_compile_info_950.c_str()), reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();
    SetupRegbaseTilingParseContext(kernel_holder, platform_info, soc_infos, aicore_spec, intrinsics, soc_version, npuarchs);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto param = gert::TilingData::CreateCap(8192);
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(8);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
    ASSERT_NE(param, nullptr);

    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(4, 2)
                      .IrInstanceNum({4})
                      .InputShapes({&dy_shape, &x_shape, &rstd_shape, &gamma_shape})
                      .OutputShapes({&dx_shape, &dgamma_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();
    SetupRegbaseTilingContext(holder, soc_infos, aicore_spec, intrinsics, npuarchs);

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
}

TEST_F(RmsNormGradTiling, rms_norm_grad_regbase_ub_bisection_nosplit_fp16)
{
    gert::StorageShape dy_shape = {{256, 128}, {256, 128}};
    gert::StorageShape x_shape = {{256, 128}, {256, 128}};
    gert::StorageShape rstd_shape = {{256}, {256}};
    gert::StorageShape gamma_shape = {{128}, {128}};
    gert::StorageShape dx_shape = {{256, 128}, {256, 128}};
    gert::StorageShape dgamma_shape = {{128}, {128}};

    map<string, string> soc_infos, aicore_spec, intrinsics, soc_version;
    map<string, string> npuarchs = {{{"NpuArch", "3510"}}};
    fe::PlatFormInfos platform_info;
    SetupRegbasePlatformInfo(platform_info, soc_infos, aicore_spec, intrinsics, soc_version);

    RmsNormCompileInfo compile_info;
    std::string op_type("RmsNormGrad");
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(regbase_compile_info_950.c_str()), reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();
    SetupRegbaseTilingParseContext(kernel_holder, platform_info, soc_infos, aicore_spec, intrinsics, soc_version, npuarchs);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto param = gert::TilingData::CreateCap(8192);
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(8);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
    ASSERT_NE(param, nullptr);

    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(4, 2)
                      .IrInstanceNum({4})
                      .InputShapes({&dy_shape, &x_shape, &rstd_shape, &gamma_shape})
                      .OutputShapes({&dx_shape, &dgamma_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();
    SetupRegbaseTilingContext(holder, soc_infos, aicore_spec, intrinsics, npuarchs);

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
}

TEST_F(RmsNormGradTiling, rms_norm_grad_regbase_fullload_splitd_fp16)
{
    gert::StorageShape dy_shape = {{8, 8192}, {8, 8192}};
    gert::StorageShape x_shape = {{8, 8192}, {8, 8192}};
    gert::StorageShape rstd_shape = {{8}, {8}};
    gert::StorageShape gamma_shape = {{8192}, {8192}};
    gert::StorageShape dx_shape = {{8, 8192}, {8, 8192}};
    gert::StorageShape dgamma_shape = {{8192}, {8192}};

    map<string, string> soc_infos, aicore_spec, intrinsics, soc_version;
    map<string, string> npuarchs = {{{"NpuArch", "3510"}}};
    fe::PlatFormInfos platform_info;
    SetupRegbasePlatformInfo(platform_info, soc_infos, aicore_spec, intrinsics, soc_version);

    RmsNormCompileInfo compile_info;
    std::string op_type("RmsNormGrad");
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(regbase_compile_info_950.c_str()), reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();
    SetupRegbaseTilingParseContext(kernel_holder, platform_info, soc_infos, aicore_spec, intrinsics, soc_version, npuarchs);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto param = gert::TilingData::CreateCap(8192);
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(8);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
    ASSERT_NE(param, nullptr);

    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(4, 2)
                      .IrInstanceNum({4})
                      .InputShapes({&dy_shape, &x_shape, &rstd_shape, &gamma_shape})
                      .OutputShapes({&dx_shape, &dgamma_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();
    SetupRegbaseTilingContext(holder, soc_infos, aicore_spec, intrinsics, npuarchs);

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
}

TEST_F(RmsNormGradTiling, rms_norm_grad_regbase_ub_bisection_splitd_fp16)
{
    gert::StorageShape dy_shape = {{256, 8192}, {256, 8192}};
    gert::StorageShape x_shape = {{256, 8192}, {256, 8192}};
    gert::StorageShape rstd_shape = {{256}, {256}};
    gert::StorageShape gamma_shape = {{8192}, {8192}};
    gert::StorageShape dx_shape = {{256, 8192}, {256, 8192}};
    gert::StorageShape dgamma_shape = {{8192}, {8192}};

    map<string, string> soc_infos, aicore_spec, intrinsics, soc_version;
    map<string, string> npuarchs = {{{"NpuArch", "3510"}}};
    fe::PlatFormInfos platform_info;
    SetupRegbasePlatformInfo(platform_info, soc_infos, aicore_spec, intrinsics, soc_version);

    RmsNormCompileInfo compile_info;
    std::string op_type("RmsNormGrad");
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(regbase_compile_info_950.c_str()), reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();
    SetupRegbaseTilingParseContext(kernel_holder, platform_info, soc_infos, aicore_spec, intrinsics, soc_version, npuarchs);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto param = gert::TilingData::CreateCap(8192);
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(8);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
    ASSERT_NE(param, nullptr);

    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(4, 2)
                      .IrInstanceNum({4})
                      .InputShapes({&dy_shape, &x_shape, &rstd_shape, &gamma_shape})
                      .OutputShapes({&dx_shape, &dgamma_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();
    SetupRegbaseTilingContext(holder, soc_infos, aicore_spec, intrinsics, npuarchs);

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
}

TEST_F(RmsNormGradTiling, rms_norm_grad_regbase_fullload_nosplit_bf16)
{
    gert::StorageShape dy_shape = {{8, 128}, {8, 128}};
    gert::StorageShape x_shape = {{8, 128}, {8, 128}};
    gert::StorageShape rstd_shape = {{8}, {8}};
    gert::StorageShape gamma_shape = {{128}, {128}};
    gert::StorageShape dx_shape = {{8, 128}, {8, 128}};
    gert::StorageShape dgamma_shape = {{128}, {128}};

    map<string, string> soc_infos, aicore_spec, intrinsics, soc_version;
    map<string, string> npuarchs = {{{"NpuArch", "3510"}}};
    fe::PlatFormInfos platform_info;
    SetupRegbasePlatformInfo(platform_info, soc_infos, aicore_spec, intrinsics, soc_version);

    RmsNormCompileInfo compile_info;
    std::string op_type("RmsNormGrad");
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(regbase_compile_info_950.c_str()), reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();
    SetupRegbaseTilingParseContext(kernel_holder, platform_info, soc_infos, aicore_spec, intrinsics, soc_version, npuarchs);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto param = gert::TilingData::CreateCap(8192);
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(8);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
    ASSERT_NE(param, nullptr);

    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(4, 2)
                      .IrInstanceNum({4})
                      .InputShapes({&dy_shape, &x_shape, &rstd_shape, &gamma_shape})
                      .OutputShapes({&dx_shape, &dgamma_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();
    SetupRegbaseTilingContext(holder, soc_infos, aicore_spec, intrinsics, npuarchs);

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
}

TEST_F(RmsNormGradTiling, rms_norm_grad_regbase_fullload_nosplit_fp32)
{
    gert::StorageShape dy_shape = {{8, 128}, {8, 128}};
    gert::StorageShape x_shape = {{8, 128}, {8, 128}};
    gert::StorageShape rstd_shape = {{8}, {8}};
    gert::StorageShape gamma_shape = {{128}, {128}};
    gert::StorageShape dx_shape = {{8, 128}, {8, 128}};
    gert::StorageShape dgamma_shape = {{128}, {128}};

    map<string, string> soc_infos, aicore_spec, intrinsics, soc_version;
    map<string, string> npuarchs = {{{"NpuArch", "3510"}}};
    fe::PlatFormInfos platform_info;
    SetupRegbasePlatformInfo(platform_info, soc_infos, aicore_spec, intrinsics, soc_version);

    RmsNormCompileInfo compile_info;
    std::string op_type("RmsNormGrad");
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(regbase_compile_info_950.c_str()), reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();
    SetupRegbaseTilingParseContext(kernel_holder, platform_info, soc_infos, aicore_spec, intrinsics, soc_version, npuarchs);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto param = gert::TilingData::CreateCap(8192);
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(8);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
    ASSERT_NE(param, nullptr);

    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(4, 2)
                      .IrInstanceNum({4})
                      .InputShapes({&dy_shape, &x_shape, &rstd_shape, &gamma_shape})
                      .OutputShapes({&dx_shape, &dgamma_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();
    SetupRegbaseTilingContext(holder, soc_infos, aicore_spec, intrinsics, npuarchs);

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
}

TEST_F(RmsNormGradTiling, rms_norm_grad_regbase_multi_colset_fp16)
{
    gert::StorageShape dy_shape = {{8, 4096}, {8, 4096}};
    gert::StorageShape x_shape = {{8, 4096}, {8, 4096}};
    gert::StorageShape rstd_shape = {{8}, {8}};
    gert::StorageShape gamma_shape = {{4096}, {4096}};
    gert::StorageShape dx_shape = {{8, 4096}, {8, 4096}};
    gert::StorageShape dgamma_shape = {{4096}, {4096}};

    map<string, string> soc_infos, aicore_spec, intrinsics, soc_version;
    map<string, string> npuarchs = {{{"NpuArch", "3510"}}};
    fe::PlatFormInfos platform_info;
    SetupRegbasePlatformInfo(platform_info, soc_infos, aicore_spec, intrinsics, soc_version);

    RmsNormCompileInfo compile_info;
    std::string op_type("RmsNormGrad");
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(regbase_compile_info_950.c_str()), reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();
    SetupRegbaseTilingParseContext(kernel_holder, platform_info, soc_infos, aicore_spec, intrinsics, soc_version, npuarchs);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto param = gert::TilingData::CreateCap(8192);
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(8);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
    ASSERT_NE(param, nullptr);

    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(4, 2)
                      .IrInstanceNum({4})
                      .InputShapes({&dy_shape, &x_shape, &rstd_shape, &gamma_shape})
                      .OutputShapes({&dx_shape, &dgamma_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();
    SetupRegbaseTilingContext(holder, soc_infos, aicore_spec, intrinsics, npuarchs);

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
}

TEST_F(RmsNormGradTiling, rms_norm_grad_regbase_err_dy_x_mismatch)
{
    gert::StorageShape dy_shape = {{16, 128}, {16, 128}};
    gert::StorageShape x_shape = {{16, 64}, {16, 64}};
    gert::StorageShape rstd_shape = {{16}, {16}};
    gert::StorageShape gamma_shape = {{128}, {128}};
    gert::StorageShape dx_shape = {{16, 128}, {16, 128}};
    gert::StorageShape dgamma_shape = {{128}, {128}};

    map<string, string> soc_infos, aicore_spec, intrinsics, soc_version;
    map<string, string> npuarchs = {{{"NpuArch", "3510"}}};
    fe::PlatFormInfos platform_info;
    SetupRegbasePlatformInfo(platform_info, soc_infos, aicore_spec, intrinsics, soc_version);

    RmsNormCompileInfo compile_info;
    std::string op_type("RmsNormGrad");
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(regbase_compile_info_950.c_str()), reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();
    SetupRegbaseTilingParseContext(kernel_holder, platform_info, soc_infos, aicore_spec, intrinsics, soc_version, npuarchs);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto param = gert::TilingData::CreateCap(8192);
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(8);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
    ASSERT_NE(param, nullptr);

    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(4, 2)
                      .IrInstanceNum({4})
                      .InputShapes({&dy_shape, &x_shape, &rstd_shape, &gamma_shape})
                      .OutputShapes({&dx_shape, &dgamma_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();
    SetupRegbaseTilingContext(holder, soc_infos, aicore_spec, intrinsics, npuarchs);

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(RmsNormGradTiling, rms_norm_grad_regbase_err_dtype_mismatch)
{
    gert::StorageShape dy_shape = {{8, 128}, {8, 128}};
    gert::StorageShape x_shape = {{8, 128}, {8, 128}};
    gert::StorageShape rstd_shape = {{8}, {8}};
    gert::StorageShape gamma_shape = {{128}, {128}};
    gert::StorageShape dx_shape = {{8, 128}, {8, 128}};
    gert::StorageShape dgamma_shape = {{128}, {128}};

    map<string, string> soc_infos, aicore_spec, intrinsics, soc_version;
    map<string, string> npuarchs = {{{"NpuArch", "3510"}}};
    fe::PlatFormInfos platform_info;
    SetupRegbasePlatformInfo(platform_info, soc_infos, aicore_spec, intrinsics, soc_version);

    RmsNormCompileInfo compile_info;
    std::string op_type("RmsNormGrad");
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(regbase_compile_info_950.c_str()), reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();
    SetupRegbaseTilingParseContext(kernel_holder, platform_info, soc_infos, aicore_spec, intrinsics, soc_version, npuarchs);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto param = gert::TilingData::CreateCap(8192);
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(8);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
    ASSERT_NE(param, nullptr);

    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(4, 2)
                      .IrInstanceNum({4})
                      .InputShapes({&dy_shape, &x_shape, &rstd_shape, &gamma_shape})
                      .OutputShapes({&dx_shape, &dgamma_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();
    SetupRegbaseTilingContext(holder, soc_infos, aicore_spec, intrinsics, npuarchs);

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(RmsNormGradTiling, rms_norm_grad_regbase_err_invalid_dtype_int32)
{
    gert::StorageShape dy_shape = {{8, 128}, {8, 128}};
    gert::StorageShape x_shape = {{8, 128}, {8, 128}};
    gert::StorageShape rstd_shape = {{8}, {8}};
    gert::StorageShape gamma_shape = {{128}, {128}};
    gert::StorageShape dx_shape = {{8, 128}, {8, 128}};
    gert::StorageShape dgamma_shape = {{128}, {128}};

    map<string, string> soc_infos, aicore_spec, intrinsics, soc_version;
    map<string, string> npuarchs = {{{"NpuArch", "3510"}}};
    fe::PlatFormInfos platform_info;
    SetupRegbasePlatformInfo(platform_info, soc_infos, aicore_spec, intrinsics, soc_version);

    RmsNormCompileInfo compile_info;
    std::string op_type("RmsNormGrad");
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(regbase_compile_info_950.c_str()), reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();
    SetupRegbaseTilingParseContext(kernel_holder, platform_info, soc_infos, aicore_spec, intrinsics, soc_version, npuarchs);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto param = gert::TilingData::CreateCap(8192);
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(8);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
    ASSERT_NE(param, nullptr);

    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(4, 2)
                      .IrInstanceNum({4})
                      .InputShapes({&dy_shape, &x_shape, &rstd_shape, &gamma_shape})
                      .OutputShapes({&dx_shape, &dgamma_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();
    SetupRegbaseTilingContext(holder, soc_infos, aicore_spec, intrinsics, npuarchs);

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(RmsNormGradTiling, rms_norm_grad_regbase_err_invalid_format)
{
    gert::StorageShape dy_shape = {{8, 128}, {8, 128}};
    gert::StorageShape x_shape = {{8, 128}, {8, 128}};
    gert::StorageShape rstd_shape = {{8}, {8}};
    gert::StorageShape gamma_shape = {{128}, {128}};
    gert::StorageShape dx_shape = {{8, 128}, {8, 128}};
    gert::StorageShape dgamma_shape = {{128}, {128}};

    map<string, string> soc_infos, aicore_spec, intrinsics, soc_version;
    map<string, string> npuarchs = {{{"NpuArch", "3510"}}};
    fe::PlatFormInfos platform_info;
    SetupRegbasePlatformInfo(platform_info, soc_infos, aicore_spec, intrinsics, soc_version);

    RmsNormCompileInfo compile_info;
    std::string op_type("RmsNormGrad");
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(regbase_compile_info_950.c_str()), reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();
    SetupRegbaseTilingParseContext(kernel_holder, platform_info, soc_infos, aicore_spec, intrinsics, soc_version, npuarchs);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto param = gert::TilingData::CreateCap(8192);
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(8);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
    ASSERT_NE(param, nullptr);

    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(4, 2)
                      .IrInstanceNum({4})
                      .InputShapes({&dy_shape, &x_shape, &rstd_shape, &gamma_shape})
                      .OutputShapes({&dx_shape, &dgamma_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();
    SetupRegbaseTilingContext(holder, soc_infos, aicore_spec, intrinsics, npuarchs);

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(RmsNormGradTiling, rms_norm_grad_regbase_err_rstd_dtype_not_float)
{
    gert::StorageShape dy_shape = {{8, 128}, {8, 128}};
    gert::StorageShape x_shape = {{8, 128}, {8, 128}};
    gert::StorageShape rstd_shape = {{8}, {8}};
    gert::StorageShape gamma_shape = {{128}, {128}};
    gert::StorageShape dx_shape = {{8, 128}, {8, 128}};
    gert::StorageShape dgamma_shape = {{128}, {128}};

    map<string, string> soc_infos, aicore_spec, intrinsics, soc_version;
    map<string, string> npuarchs = {{{"NpuArch", "3510"}}};
    fe::PlatFormInfos platform_info;
    SetupRegbasePlatformInfo(platform_info, soc_infos, aicore_spec, intrinsics, soc_version);

    RmsNormCompileInfo compile_info;
    std::string op_type("RmsNormGrad");
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(regbase_compile_info_950.c_str()), reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();
    SetupRegbaseTilingParseContext(kernel_holder, platform_info, soc_infos, aicore_spec, intrinsics, soc_version, npuarchs);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto param = gert::TilingData::CreateCap(8192);
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(8);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
    ASSERT_NE(param, nullptr);

    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(4, 2)
                      .IrInstanceNum({4})
                      .InputShapes({&dy_shape, &x_shape, &rstd_shape, &gamma_shape})
                      .OutputShapes({&dx_shape, &dgamma_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();
    SetupRegbaseTilingContext(holder, soc_infos, aicore_spec, intrinsics, npuarchs);

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(RmsNormGradTiling, rms_norm_grad_regbase_err_dx_dy_dtype_mismatch)
{
    gert::StorageShape dy_shape = {{8, 128}, {8, 128}};
    gert::StorageShape x_shape = {{8, 128}, {8, 128}};
    gert::StorageShape rstd_shape = {{8}, {8}};
    gert::StorageShape gamma_shape = {{128}, {128}};
    gert::StorageShape dx_shape = {{8, 128}, {8, 128}};
    gert::StorageShape dgamma_shape = {{128}, {128}};

    map<string, string> soc_infos, aicore_spec, intrinsics, soc_version;
    map<string, string> npuarchs = {{{"NpuArch", "3510"}}};
    fe::PlatFormInfos platform_info;
    SetupRegbasePlatformInfo(platform_info, soc_infos, aicore_spec, intrinsics, soc_version);

    RmsNormCompileInfo compile_info;
    std::string op_type("RmsNormGrad");
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(regbase_compile_info_950.c_str()), reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();
    SetupRegbaseTilingParseContext(kernel_holder, platform_info, soc_infos, aicore_spec, intrinsics, soc_version, npuarchs);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto param = gert::TilingData::CreateCap(8192);
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(8);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
    ASSERT_NE(param, nullptr);

    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(4, 2)
                      .IrInstanceNum({4})
                      .InputShapes({&dy_shape, &x_shape, &rstd_shape, &gamma_shape})
                      .OutputShapes({&dx_shape, &dgamma_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();
    SetupRegbaseTilingContext(holder, soc_infos, aicore_spec, intrinsics, npuarchs);

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(RmsNormGradTiling, rms_norm_grad_regbase_err_gamma_dtype_invalid)
{
    gert::StorageShape dy_shape = {{8, 128}, {8, 128}};
    gert::StorageShape x_shape = {{8, 128}, {8, 128}};
    gert::StorageShape rstd_shape = {{8}, {8}};
    gert::StorageShape gamma_shape = {{128}, {128}};
    gert::StorageShape dx_shape = {{8, 128}, {8, 128}};
    gert::StorageShape dgamma_shape = {{128}, {128}};

    map<string, string> soc_infos, aicore_spec, intrinsics, soc_version;
    map<string, string> npuarchs = {{{"NpuArch", "3510"}}};
    fe::PlatFormInfos platform_info;
    SetupRegbasePlatformInfo(platform_info, soc_infos, aicore_spec, intrinsics, soc_version);

    RmsNormCompileInfo compile_info;
    std::string op_type("RmsNormGrad");
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(regbase_compile_info_950.c_str()), reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();
    SetupRegbaseTilingParseContext(kernel_holder, platform_info, soc_infos, aicore_spec, intrinsics, soc_version, npuarchs);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto param = gert::TilingData::CreateCap(8192);
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(8);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
    ASSERT_NE(param, nullptr);

    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(4, 2)
                      .IrInstanceNum({4})
                      .InputShapes({&dy_shape, &x_shape, &rstd_shape, &gamma_shape})
                      .OutputShapes({&dx_shape, &dgamma_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();
    SetupRegbaseTilingContext(holder, soc_infos, aicore_spec, intrinsics, npuarchs);

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(RmsNormGradTiling, rms_norm_grad_regbase_err_dgamma_dtype_not_float)
{
    gert::StorageShape dy_shape = {{8, 128}, {8, 128}};
    gert::StorageShape x_shape = {{8, 128}, {8, 128}};
    gert::StorageShape rstd_shape = {{8}, {8}};
    gert::StorageShape gamma_shape = {{128}, {128}};
    gert::StorageShape dx_shape = {{8, 128}, {8, 128}};
    gert::StorageShape dgamma_shape = {{128}, {128}};

    map<string, string> soc_infos, aicore_spec, intrinsics, soc_version;
    map<string, string> npuarchs = {{{"NpuArch", "3510"}}};
    fe::PlatFormInfos platform_info;
    SetupRegbasePlatformInfo(platform_info, soc_infos, aicore_spec, intrinsics, soc_version);

    RmsNormCompileInfo compile_info;
    std::string op_type("RmsNormGrad");
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(regbase_compile_info_950.c_str()), reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();
    SetupRegbaseTilingParseContext(kernel_holder, platform_info, soc_infos, aicore_spec, intrinsics, soc_version, npuarchs);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto param = gert::TilingData::CreateCap(8192);
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(8);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
    ASSERT_NE(param, nullptr);

    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(4, 2)
                      .IrInstanceNum({4})
                      .InputShapes({&dy_shape, &x_shape, &rstd_shape, &gamma_shape})
                      .OutputShapes({&dx_shape, &dgamma_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();
    SetupRegbaseTilingContext(holder, soc_infos, aicore_spec, intrinsics, npuarchs);

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(RmsNormGradTiling, rms_norm_grad_tiling_deterministic)
{
    gert::StorageShape dy_shape = {{16, 16}, {16, 16}};
    gert::StorageShape x_shape = {{16, 16}, {16, 16}};
    gert::StorageShape rstd_shape = {{16}, {16}};
    gert::StorageShape gamma_shape = {{16}, {16}};
    gert::StorageShape dx_shape = {{16, 16}, {16, 16}};
    gert::StorageShape dgamma_shape = {{16}, {16}};

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
    RmsNormCompileInfo compile_info;

    std::string op_type("RmsNormGrad");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(4, 2)
                      .IrInstanceNum({4})
                      .InputShapes({&dy_shape, &x_shape, &rstd_shape, &gamma_shape})
                      .OutputShapes({&dx_shape, &dgamma_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .DeterministicInfo(1)
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
}

TEST_F(RmsNormGradTiling, rms_norm_grad_tiling_large_d_tail_core)
{
    gert::StorageShape dy_shape = {{41, 8192}, {41, 8192}};
    gert::StorageShape x_shape = {{41, 8192}, {41, 8192}};
    gert::StorageShape rstd_shape = {{41}, {41}};
    gert::StorageShape gamma_shape = {{8192}, {8192}};
    gert::StorageShape dx_shape = {{41, 8192}, {41, 8192}};
    gert::StorageShape dgamma_shape = {{8192}, {8192}};

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
    RmsNormCompileInfo compile_info;

    std::string op_type("RmsNormGrad");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto param = gert::TilingData::CreateCap(8192);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(8192);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(4, 2)
                      .IrInstanceNum({4})
                      .InputShapes({&dy_shape, &x_shape, &rstd_shape, &gamma_shape})
                      .OutputShapes({&dx_shape, &dgamma_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
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
}

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
#include "log/log.h"
#include <gtest/gtest.h>
#include "register/op_impl_registry.h"
#include "platform/platform_infos_def.h"
#include "ut_op_common.h"
#include "tests/ut/common/ut_op_util.h"
#include "norm/renorm/op_host/arch35/renorm_tiling_arch35.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "array_ops.h"
#include "tiling/platform/platform_ascendc.h"

using namespace std;
using namespace ge;
using namespace ut_util;

class RenormDavidTiling : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "RenormDavidTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "RenormDavidTiling TearDown" << std::endl;
    }
};

static string TilingData2Str(const gert::TilingData* tiling_data)
{
    auto data = tiling_data->GetData();
    string result;
    for (size_t i = 0; i < tiling_data->GetDataSize(); i += sizeof(int64_t)) {
        result += std::to_string((reinterpret_cast<const int64_t*>(tiling_data->GetData())[i / sizeof(int64_t)]));
        result += " ";
    }

    return result;
}

TEST_F(RenormDavidTiling, renorm_david_tiling1_p1)
{
    dlog_setlevel(0, 0, 0);
    gert::StorageShape x_shape = {{2, 2, 2}, {2, 2, 2}};
    gert::StorageShape y_shape = {{2, 2, 2}, {2, 2, 2}};

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
    map<string, string> npuarchs = {{"NpuArch", "3510"}};
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    Ops::Base::ReduceOpCompileInfo compile_info;

    std::string op_type("Renorm");
    auto tiling_instance = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str());
    if (tiling_instance == nullptr) {
        std::cout << "test>> tilingFunc is invalid" << std::endl;
    } else {
        std::cout << "test>> tilingFunc is valid" << std::endl;
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
        kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);

        ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

        // tilingFunc simulate
        auto param = gert::TilingData::CreateCap(4096);
        auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(16 * 4096);
        auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
        ASSERT_NE(param, nullptr);
        compile_info.vectorCoreNum = 64;
        auto holder = gert::TilingContextFaker()
                        .SetOpType(op_type)
                        .NodeIoNum(1, 1)
                        .IrInstanceNum({1}, {1})
                        .InputShapes(
                            {&x_shape})
                        .OutputShapes({&y_shape})
                        .CompileInfo(&compile_info)
                        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                        .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeAttrs(
                            {{"p", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                            {"dim", Ops::NN::AnyValue::CreateFrom<int64_t>(2)},
                            {"maxnorm", Ops::NN::AnyValue::CreateFrom<float>(0)}})
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
    }
    dlog_setlevel(0, 3, 0);
}

TEST_F(RenormDavidTiling, renorm_david_tiling2_p2)
{
    dlog_setlevel(0, 0, 0);
    gert::StorageShape x_shape = {{2, 2, 2, 2}, {2, 2, 2, 2}};
    gert::StorageShape y_shape = {{2, 2, 2, 2}, {2, 2, 2, 2}};

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
    map<string, string> npuarchs = {{"NpuArch", "3510"}};
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    Ops::Base::ReduceOpCompileInfo compile_info;

    std::string op_type("Renorm");
    auto tiling_instance = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str());
    if (tiling_instance == nullptr) {
        std::cout << "test>> tilingFunc is invalid" << std::endl;
    } else {
        std::cout << "test>> tilingFunc is valid" << std::endl;
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
        kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);

        ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

        // tilingFunc simulate
        auto param = gert::TilingData::CreateCap(4096);
        auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(16 * 4096);
        auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
        ASSERT_NE(param, nullptr);
        auto holder = gert::TilingContextFaker()
                        .SetOpType(op_type)
                        .NodeIoNum(1, 1)
                        .IrInstanceNum({1}, {1})
                        .InputShapes(
                            {&x_shape})
                        .OutputShapes({&y_shape})
                        .CompileInfo(&compile_info)
                        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                        .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                        .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                        .NodeAttrs(
                            {{"p", Ops::NN::AnyValue::CreateFrom<float>(2.0)},
                            {"dim", Ops::NN::AnyValue::CreateFrom<int64_t>(2)},
                            {"maxnorm", Ops::NN::AnyValue::CreateFrom<float>(0)}})
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
    }
    dlog_setlevel(0, 3, 0);
}

TEST_F(RenormDavidTiling, renorm_david_tiling3_p3)
{
    dlog_setlevel(0, 0, 0);
    gert::StorageShape x_shape = {{2, 2, 2, 2}, {2, 2, 2, 2}};
    gert::StorageShape y_shape = {{2, 2, 2, 2}, {2, 2, 2, 2}};

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
    map<string, string> npuarchs = {{"NpuArch", "3510"}};
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    Ops::Base::ReduceOpCompileInfo compile_info;

    std::string op_type("Renorm");
    auto tiling_instance = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str());
    if (tiling_instance == nullptr) {
        std::cout << "test>> tilingFunc is invalid" << std::endl;
    } else {
        std::cout << "test>> tilingFunc is valid" << std::endl;
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
        kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);

        ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

        // tilingFunc simulate
        auto param = gert::TilingData::CreateCap(4096);
        auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(16 * 4096);
        auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
        ASSERT_NE(param, nullptr);
        auto holder = gert::TilingContextFaker()
                        .SetOpType(op_type)
                        .NodeIoNum(1, 1)
                        .IrInstanceNum({1}, {1})
                        .InputShapes(
                            {&x_shape})
                        .OutputShapes({&y_shape})
                        .CompileInfo(&compile_info)
                        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                        .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                        .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                        .NodeAttrs(
                            {{"p", Ops::NN::AnyValue::CreateFrom<float>(3.0)},
                            {"dim", Ops::NN::AnyValue::CreateFrom<int64_t>(2)},
                            {"maxnorm", Ops::NN::AnyValue::CreateFrom<float>(0)}})
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
    }
    dlog_setlevel(0, 3, 0);
}

TEST_F(RenormDavidTiling, renorm_david_tiling6_p4)
{
    dlog_setlevel(0, 0, 0);
    gert::StorageShape x_shape = {{2, 2, 2, 2}, {2, 2, 2, 2}};
    gert::StorageShape y_shape = {{2, 2, 2, 2}, {2, 2, 2, 2}};

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
    map<string, string> npuarchs = {{"NpuArch", "3510"}};
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics, soc_version);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    Ops::Base::ReduceOpCompileInfo compile_info;

    std::string op_type("Renorm");
    auto tiling_instance = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str());
    if (tiling_instance == nullptr) {
        std::cout << "test>> tilingFunc is invalid" << std::endl;
    } else {
        std::cout << "test>> tilingFunc is valid" << std::endl;
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
        kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);

        ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

        // tilingFunc simulate
        auto param = gert::TilingData::CreateCap(4096);
        auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(16 * 4096);
        auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
        ASSERT_NE(param, nullptr);
        auto holder = gert::TilingContextFaker()
                        .SetOpType(op_type)
                        .NodeIoNum(1, 1)
                        .IrInstanceNum({1}, {1})
                        .InputShapes(
                            {&x_shape})
                        .OutputShapes({&y_shape})
                        .CompileInfo(&compile_info)
                        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                        .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                        .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                        .NodeAttrs(
                            {{"p", Ops::NN::AnyValue::CreateFrom<float>(4.0)},
                            {"dim", Ops::NN::AnyValue::CreateFrom<int64_t>(2)},
                            {"maxnorm", Ops::NN::AnyValue::CreateFrom<float>(0)}})
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
    }
    dlog_setlevel(0, 3, 0);
}

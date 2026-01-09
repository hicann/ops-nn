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
#include "ut_op_util.h"
#include "../../../../op_host/arch35/apply_adam_w_tiling.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"

using namespace ge;
using namespace ut_util;

class ApplyAdamWTilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "ApplyAdamWTilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "ApplyAdamWTilingTest TearDown" << std::endl;
    }
};

TEST_F(ApplyAdamWTilingTest, apply_adam_w_tiling_1000)
{
    std::string op_type("ApplyAdamW");
    
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    string compile_info_string = R"({
                                            "hardware_info": {
                                                "BT_SIZE": 0,
                                                "load3d_constraints": "1",
                                                "Intrinsic_fix_pipe_l0c2out": false,
                                                "Intrinsic_data_move_l12ub": true,
                                                "Intrinsic_data_move_l0c2ub": true,
                                                "Intrinsic_data_move_out2l1_nd2nz": false,
                                                "UB_SIZE": 196608,
                                                "L2_SIZE": 33554432,
                                                "L1_SIZE": 524288,
                                                "L0A_SIZE": 65536,
                                                "L0B_SIZE": 65536,
                                                "L0C_SIZE": 131072,
                                                "CORE_NUM": 64
                                            }
                                        })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::ApplyAdamWCompileInfo compile_info;
    compile_info.coreNum = 64;
    compile_info.ubSize = 245760;

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(8192);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(32);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape inputShape1 = {{64, 10, 10, 32}, {64, 10, 10, 32}};
    gert::StorageShape inputShape2 = {{1}, {1}};
    gert::StorageShape outputShape = {{64, 10, 10, 32}, {64, 10, 10, 32}};
    ge::DataType varDtype = ge::DT_FLOAT;
    ge::Format format = ge::FORMAT_ND;
    bool amsgrad = false;
    bool maximize = false;

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(12, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&inputShape1, &inputShape1, &inputShape1, &inputShape2, &inputShape2, &inputShape2,
                           &inputShape2, &inputShape2, &inputShape2, &inputShape2, &inputShape1, &inputShape1})
                      .OutputShapes({&outputShape, &outputShape, &outputShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, varDtype, format, format)
                      .NodeInputTd(1, varDtype, format, format)
                      .NodeInputTd(2, varDtype, format, format)
                      .NodeInputTd(3, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, varDtype, format, format)
                      .NodeInputTd(11, varDtype, format, format)
                      .NodeOutputTd(0, varDtype, format, format)
                      .NodeOutputTd(1, varDtype, format, format)
                      .NodeOutputTd(2, varDtype, format, format)
                      .NodeAttrs(
                          {{"amsgrad", Ops::NN::AnyValue::CreateFrom<bool>(amsgrad)},
                           {"maximize", Ops::NN::AnyValue::CreateFrom<bool>(maximize)}})
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
}

TEST_F(ApplyAdamWTilingTest, apply_adam_w_tiling_1001)
{
    std::string op_type("ApplyAdamW");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    string compile_info_string = R"({
                                            "hardware_info": {
                                                "BT_SIZE": 0,
                                                "load3d_constraints": "1",
                                                "Intrinsic_fix_pipe_l0c2out": false,
                                                "Intrinsic_data_move_l12ub": true,
                                                "Intrinsic_data_move_l0c2ub": true,
                                                "Intrinsic_data_move_out2l1_nd2nz": false,
                                                "UB_SIZE": 196608,
                                                "L2_SIZE": 33554432,
                                                "L1_SIZE": 524288,
                                                "L0A_SIZE": 65536,
                                                "L0B_SIZE": 65536,
                                                "L0C_SIZE": 131072,
                                                "CORE_NUM": 64
                                            }
                                        })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::ApplyAdamWCompileInfo compile_info;
    compile_info.coreNum = 64;
    compile_info.ubSize = 245760;

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(8192);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(32);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape inputShape1 = {{64, 10, 10, 32}, {64, 10, 10, 32}};
    gert::StorageShape inputShape2 = {{1}, {1}};
    gert::StorageShape outputShape = {{64, 10, 10, 32}, {64, 10, 10, 32}};
    ge::DataType varDtype = ge::DT_FLOAT16;
    ge::Format format = ge::FORMAT_ND;
    bool amsgrad = false;
    bool maximize = false;

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(12, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&inputShape1, &inputShape1, &inputShape1, &inputShape2, &inputShape2, &inputShape2,
                           &inputShape2, &inputShape2, &inputShape2, &inputShape2, &inputShape1, &inputShape1})
                      .OutputShapes({&outputShape, &outputShape, &outputShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, varDtype, format, format)
                      .NodeInputTd(1, varDtype, format, format)
                      .NodeInputTd(2, varDtype, format, format)
                      .NodeInputTd(3, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, varDtype, format, format)
                      .NodeInputTd(11, varDtype, format, format)
                      .NodeOutputTd(0, varDtype, format, format)
                      .NodeOutputTd(1, varDtype, format, format)
                      .NodeOutputTd(2, varDtype, format, format)
                      .NodeAttrs(
                          {{"amsgrad", Ops::NN::AnyValue::CreateFrom<bool>(amsgrad)},
                           {"maximize", Ops::NN::AnyValue::CreateFrom<bool>(maximize)}})
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
}

TEST_F(ApplyAdamWTilingTest, apply_adam_w_tiling_1002)
{
    std::string op_type("ApplyAdamW");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    string compile_info_string = R"({
                                            "hardware_info": {
                                                "BT_SIZE": 0,
                                                "load3d_constraints": "1",
                                                "Intrinsic_fix_pipe_l0c2out": false,
                                                "Intrinsic_data_move_l12ub": true,
                                                "Intrinsic_data_move_l0c2ub": true,
                                                "Intrinsic_data_move_out2l1_nd2nz": false,
                                                "UB_SIZE": 196608,
                                                "L2_SIZE": 33554432,
                                                "L1_SIZE": 524288,
                                                "L0A_SIZE": 65536,
                                                "L0B_SIZE": 65536,
                                                "L0C_SIZE": 131072,
                                                "CORE_NUM": 64
                                            }
                                        })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::ApplyAdamWCompileInfo compile_info;
    compile_info.coreNum = 64;
    compile_info.ubSize = 245760;

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(8192);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(32);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape inputShape1 = {{64, 10, 10, 32}, {64, 10, 10, 32}};
    gert::StorageShape inputShape2 = {{1}, {1}};
    gert::StorageShape outputShape = {{64, 10, 10, 32}, {64, 10, 10, 32}};
    ge::DataType varDtype = ge::DT_BF16;
    ge::Format format = ge::FORMAT_ND;
    bool amsgrad = false;
    bool maximize = false;

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(12, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&inputShape1, &inputShape1, &inputShape1, &inputShape2, &inputShape2, &inputShape2,
                           &inputShape2, &inputShape2, &inputShape2, &inputShape2, &inputShape1, &inputShape1})
                      .OutputShapes({&outputShape, &outputShape, &outputShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, varDtype, format, format)
                      .NodeInputTd(1, varDtype, format, format)
                      .NodeInputTd(2, varDtype, format, format)
                      .NodeInputTd(3, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, varDtype, format, format)
                      .NodeInputTd(11, varDtype, format, format)
                      .NodeOutputTd(0, varDtype, format, format)
                      .NodeOutputTd(1, varDtype, format, format)
                      .NodeOutputTd(2, varDtype, format, format)
                      .NodeAttrs(
                          {{"amsgrad", Ops::NN::AnyValue::CreateFrom<bool>(amsgrad)},
                           {"maximize", Ops::NN::AnyValue::CreateFrom<bool>(maximize)}})
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
}

TEST_F(ApplyAdamWTilingTest, apply_adam_w_tiling_1003)
{
    std::string op_type("ApplyAdamW");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    string compile_info_string = R"({
                                            "hardware_info": {
                                                "BT_SIZE": 0,
                                                "load3d_constraints": "1",
                                                "Intrinsic_fix_pipe_l0c2out": false,
                                                "Intrinsic_data_move_l12ub": true,
                                                "Intrinsic_data_move_l0c2ub": true,
                                                "Intrinsic_data_move_out2l1_nd2nz": false,
                                                "UB_SIZE": 196608,
                                                "L2_SIZE": 33554432,
                                                "L1_SIZE": 524288,
                                                "L0A_SIZE": 65536,
                                                "L0B_SIZE": 65536,
                                                "L0C_SIZE": 131072,
                                                "CORE_NUM": 64
                                            }
                                        })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::ApplyAdamWCompileInfo compile_info;
    compile_info.coreNum = 64;
    compile_info.ubSize = 245760;

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(8192);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(32);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape inputShape1 = {{64, 10, 10, 32}, {64, 10, 10, 32}};
    gert::StorageShape inputShape2 = {{1}, {1}};
    gert::StorageShape outputShape = {{64, 10, 10, 32}, {64, 10, 10, 32}};
    ge::DataType varDtype = ge::DT_FLOAT16;
    ge::Format format = ge::FORMAT_ND;
    bool amsgrad = false;
    bool maximize = true;

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(12, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&inputShape1, &inputShape1, &inputShape1, &inputShape2, &inputShape2, &inputShape2,
                           &inputShape2, &inputShape2, &inputShape2, &inputShape2, &inputShape1, &inputShape1})
                      .OutputShapes({&outputShape, &outputShape, &outputShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, varDtype, format, format)
                      .NodeInputTd(1, varDtype, format, format)
                      .NodeInputTd(2, varDtype, format, format)
                      .NodeInputTd(3, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, varDtype, format, format)
                      .NodeInputTd(11, varDtype, format, format)
                      .NodeOutputTd(0, varDtype, format, format)
                      .NodeOutputTd(1, varDtype, format, format)
                      .NodeOutputTd(2, varDtype, format, format)
                      .NodeAttrs(
                          {{"amsgrad", Ops::NN::AnyValue::CreateFrom<bool>(amsgrad)},
                           {"maximize", Ops::NN::AnyValue::CreateFrom<bool>(maximize)}})
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
}

TEST_F(ApplyAdamWTilingTest, apply_adam_w_tiling_1004)
{
    std::string op_type("ApplyAdamW");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    string compile_info_string = R"({
                                            "hardware_info": {
                                                "BT_SIZE": 0,
                                                "load3d_constraints": "1",
                                                "Intrinsic_fix_pipe_l0c2out": false,
                                                "Intrinsic_data_move_l12ub": true,
                                                "Intrinsic_data_move_l0c2ub": true,
                                                "Intrinsic_data_move_out2l1_nd2nz": false,
                                                "UB_SIZE": 196608,
                                                "L2_SIZE": 33554432,
                                                "L1_SIZE": 524288,
                                                "L0A_SIZE": 65536,
                                                "L0B_SIZE": 65536,
                                                "L0C_SIZE": 131072,
                                                "CORE_NUM": 64
                                            }
                                        })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::ApplyAdamWCompileInfo compile_info;
    compile_info.coreNum = 64;
    compile_info.ubSize = 245760;

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(8192);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(32);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape inputShape1 = {{64, 10, 10, 32}, {64, 10, 10, 32}};
    gert::StorageShape inputShape2 = {{1}, {1}};
    gert::StorageShape outputShape = {{64, 10, 10, 32}, {64, 10, 10, 32}};
    ge::DataType varDtype = ge::DT_BF16;
    ge::Format format = ge::FORMAT_ND;
    bool amsgrad = false;
    bool maximize = true;

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(12, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&inputShape1, &inputShape1, &inputShape1, &inputShape2, &inputShape2, &inputShape2,
                           &inputShape2, &inputShape2, &inputShape2, &inputShape2, &inputShape1, &inputShape1})
                      .OutputShapes({&outputShape, &outputShape, &outputShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, varDtype, format, format)
                      .NodeInputTd(1, varDtype, format, format)
                      .NodeInputTd(2, varDtype, format, format)
                      .NodeInputTd(3, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, varDtype, format, format)
                      .NodeInputTd(11, varDtype, format, format)
                      .NodeOutputTd(0, varDtype, format, format)
                      .NodeOutputTd(1, varDtype, format, format)
                      .NodeOutputTd(2, varDtype, format, format)
                      .NodeAttrs(
                          {{"amsgrad", Ops::NN::AnyValue::CreateFrom<bool>(amsgrad)},
                           {"maximize", Ops::NN::AnyValue::CreateFrom<bool>(maximize)}})
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
}

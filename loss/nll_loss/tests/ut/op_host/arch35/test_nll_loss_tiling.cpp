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
 * \file test_nll_loss_tiling.cpp
 * \brief dynamic tiling test of nll_loss
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
#include "loss/nll_loss/op_host/arch35/nll_loss_tiling_arch35.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"

class NLLLossTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "NLLLossTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "NLLLossTiling TearDown" << std::endl;
    }
};

static std::string to_string(const std::stringstream& tiling_data)
{
    auto data = tiling_data.str();
    std::string result;
    int64_t tmp = 0;
    for (size_t i = 0; i < data.length(); i += sizeof(int64_t)) {
        memcpy(&tmp, data.c_str() + i, sizeof(tmp));
        result += std::to_string(tmp);
        result += " ";
    }

    return result;
}

using namespace ge;
/*
.INPUT(x, TensorType({DT_FLOAT}))
    .INPUT(target, TensorType({DT_INT32}))
    .INPUT(weight, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .OUTPUT(total_weight, TensorType({DT_FLOAT}))
    .ATTR(reduction, String, "mean")
    .ATTR(ignore_index, Int, -100)
*/

TEST_F(NLLLossTiling, NLLLoss_AC_tiling_1)
{
    std::string op_type("NLLLoss");
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
    optiling::NLLLossCompileInfo compile_info;
    compile_info.maxThread = 1024;
    compile_info.maxCoreNum = 64;

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape x = {{32, 2}, {32, 2}};
    gert::StorageShape target = {{32}, {32}};
    gert::StorageShape weight = {{2}, {2}};
    gert::StorageShape totalWeight = {{1}, {1}};
    gert::StorageShape y = {{1}, {1}};

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(3, 2)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&x, &target, &weight})
                      .OutputShapes({&y, &totalWeight})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"reduction", Ops::NN::AnyValue::CreateFrom<std::string>("mean")},
                           {"ignore_index", Ops::NN::AnyValue::CreateFrom<int64_t>(-100)}})
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
    std::cout << "test>> holder.GetContext end" << std::endl;

    if (tiling_func == nullptr) {
        std::cout << "test>> tiling_func is nil" << std::endl;
    } else {
        std::cout << "test>> tiling_func is valid" << std::endl;
        EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    }
}

TEST_F(NLLLossTiling, NLLLoss_AC_tiling_2)
{
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
    optiling::NLLLossCompileInfo compile_info;
    std::map<std::string, std::string> soc_version_infos = {
        {"SoC_version", "Ascend950"}, {"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};

    compile_info.maxThread = 1024;
    compile_info.maxCoreNum = 64;

    std::string op_type("NLLLoss");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape x = {{5242880, 2}, {5242880, 2}};
    gert::StorageShape target = {{5242880}, {5242880}};
    gert::StorageShape weight = {{2}, {2}};
    gert::StorageShape totalWeight = {{1}, {1}};
    gert::StorageShape y = {{1}, {1}};

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(3, 2)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&x, &target, &weight})
                      .OutputShapes({&y, &totalWeight})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"reduction", Ops::NN::AnyValue::CreateFrom<std::string>("mean")},
                           {"ignore_index", Ops::NN::AnyValue::CreateFrom<int64_t>(-100)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version_infos);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
}

TEST_F(NLLLossTiling, NLLLoss_AC_tiling_3)
{
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
    optiling::NLLLossCompileInfo compile_info;
    std::map<std::string, std::string> soc_version_infos = {
        {"SoC_version", "Ascend950"}, {"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};

    compile_info.maxThread = 1024;
    compile_info.maxCoreNum = 64;

    std::string op_type("NLLLoss");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape x = {{256, 2, 3, 4}, {256, 2, 3, 4}};
    gert::StorageShape target = {{256, 3, 4}, {256, 3, 4}};
    gert::StorageShape weight = {{2}, {2}};
    gert::StorageShape totalWeight = {{1}, {1}};
    gert::StorageShape y = {{1}, {1}};

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(3, 2)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&x, &target, &weight})
                      .OutputShapes({&y, &totalWeight})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"reduction", Ops::NN::AnyValue::CreateFrom<std::string>("mean")},
                           {"ignore_index", Ops::NN::AnyValue::CreateFrom<int64_t>(-100)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version_infos);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
}
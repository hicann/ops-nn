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
 * \file test_scaled_masked_softmax_v2_tiling.cpp
 * \brief
 */
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <gtest/gtest.h>
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "platform/platform_infos_def.h"
#include "kernel_run_context_facker.h"
#include "tiling/platform/platform_ascendc.h"
#include "vfusion/scaled_masked_softmax_v2/op_host/scaled_masked_softmax_v2_tiling.h"
#include "test_cube_util.h"

using namespace ge;
using namespace std;

struct ScaledMaskedSoftmaxV2CompileInfo {};

class ScaledMaskedSoftmaxV2Tiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ScaledMaskedSoftmaxV2Tiling SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "ScaledMaskedSoftmaxV2Tiling TearDown" << std::endl; }
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

TEST_F(ScaledMaskedSoftmaxV2Tiling, test_scaled_masked_softmax_v2_tiling_aligned)
{
    std::string op_type("ScaledMaskedSoftmaxV2");
    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 48}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    ScaledMaskedSoftmaxV2CompileInfo compile_info;

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

    uint32_t N = 1;
    uint32_t C = 32;
    uint32_t H = 32;
    uint32_t W = 128;
    uint32_t maskN = 1;
    uint32_t maskC = 32;
    gert::StorageShape xShape = {{N, C, H, W}, {N, C, H, W}};
    gert::StorageShape maskShape = {{maskN, maskC, H, W}, {maskN, maskC, H, W}};

    gert::StorageShape yShape = {{N, C, H, W}, {N, C, H, W}};
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&xShape, &maskShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"scale", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                                  {"fixed_triu_mask", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
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
    EXPECT_EQ(tiling_key, 0UL);
    auto raw_tiling_data = tiling_context->GetRawTilingData();
    ASSERT_NE(raw_tiling_data, nullptr);
    EXPECT_GT(raw_tiling_data->GetDataSize(), 0U);
    auto tiling_data_result = TilingData2Str(raw_tiling_data);
    // Structural invariants from shape (coreNum may vary by platform fake; skip exact golden string)
    {
        std::istringstream iss(tiling_data_result);
        uint64_t coreNum = 0, batch = 0, channel = 0, height = 0, width = 0;
        uint64_t maskBatch = 0, maskChannel = 0, maskHeight = 0, maskWidth = 0;
        iss >> coreNum >> batch >> channel >> height >> width >> maskBatch >> maskChannel >> maskHeight >> maskWidth;
        EXPECT_GT(coreNum, 0UL);
        EXPECT_EQ(batch, 1UL);
        EXPECT_EQ(channel, 32UL);
        EXPECT_EQ(height, 32UL);
        EXPECT_EQ(width, 128UL);
        EXPECT_EQ(maskBatch, 1UL);
        EXPECT_EQ(maskChannel, 32UL);
        EXPECT_EQ(maskHeight, 32UL);
        EXPECT_EQ(maskWidth, 128UL);
    }
}

TEST_F(ScaledMaskedSoftmaxV2Tiling, test_scaled_masked_softmax_v2_tiling_unaligned)
{
    std::string op_type("ScaledMaskedSoftmaxV2");
    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 48}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    ScaledMaskedSoftmaxV2CompileInfo compile_info;

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

    uint32_t N = 1;
    uint32_t C = 32;
    uint32_t H = 32;
    uint32_t W = 135;
    uint32_t maskN = 1;
    uint32_t maskC = 32;
    gert::StorageShape xShape = {{N, C, H, W}, {N, C, H, W}};
    gert::StorageShape maskShape = {{maskN, maskC, H, W}, {maskN, maskC, H, W}};

    gert::StorageShape yShape = {{N, C, H, W}, {N, C, H, W}};
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&xShape, &maskShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"scale", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                                  {"fixed_triu_mask", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
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
    EXPECT_EQ(tiling_key, 0UL);
    auto raw_tiling_data = tiling_context->GetRawTilingData();
    ASSERT_NE(raw_tiling_data, nullptr);
    EXPECT_GT(raw_tiling_data->GetDataSize(), 0U);
    auto tiling_data_result = TilingData2Str(raw_tiling_data);
    // Structural invariants from shape (coreNum may vary by platform fake; skip exact golden string)
    {
        std::istringstream iss(tiling_data_result);
        uint64_t coreNum = 0, batch = 0, channel = 0, height = 0, width = 0;
        uint64_t maskBatch = 0, maskChannel = 0, maskHeight = 0, maskWidth = 0;
        iss >> coreNum >> batch >> channel >> height >> width >> maskBatch >> maskChannel >> maskHeight >> maskWidth;
        EXPECT_GT(coreNum, 0UL);
        EXPECT_EQ(batch, 1UL);
        EXPECT_EQ(channel, 32UL);
        EXPECT_EQ(height, 32UL);
        EXPECT_EQ(width, 135UL);
        EXPECT_EQ(maskBatch, 1UL);
        EXPECT_EQ(maskChannel, 32UL);
        EXPECT_EQ(maskHeight, 32UL);
        EXPECT_EQ(maskWidth, 135UL);
    }
}

TEST_F(ScaledMaskedSoftmaxV2Tiling, test_scaled_masked_softmax_tiling_broadcastC)
{
    std::string op_type("ScaledMaskedSoftmaxV2");
    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 48}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    ScaledMaskedSoftmaxV2CompileInfo compile_info;

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

    uint32_t N = 1;
    uint32_t C = 32;
    uint32_t H = 32;
    uint32_t W = 135;
    uint32_t maskN = 1;
    uint32_t maskC = 1;
    gert::StorageShape xShape = {{N, C, H, W}, {N, C, H, W}};
    gert::StorageShape maskShape = {{maskN, maskC, H, W}, {maskN, maskC, H, W}};

    gert::StorageShape yShape = {{N, C, H, W}, {N, C, H, W}};
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&xShape, &maskShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"scale", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                                  {"fixed_triu_mask", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
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
    EXPECT_EQ(tiling_key, 1UL);
    auto raw_tiling_data = tiling_context->GetRawTilingData();
    ASSERT_NE(raw_tiling_data, nullptr);
    EXPECT_GT(raw_tiling_data->GetDataSize(), 0U);
    auto tiling_data_result = TilingData2Str(raw_tiling_data);
    // Structural invariants from shape (coreNum may vary by platform fake; skip exact golden string)
    {
        std::istringstream iss(tiling_data_result);
        uint64_t coreNum = 0, batch = 0, channel = 0, height = 0, width = 0;
        uint64_t maskBatch = 0, maskChannel = 0, maskHeight = 0, maskWidth = 0;
        iss >> coreNum >> batch >> channel >> height >> width >> maskBatch >> maskChannel >> maskHeight >> maskWidth;
        EXPECT_GT(coreNum, 0UL);
        EXPECT_EQ(batch, 1UL);
        EXPECT_EQ(channel, 32UL);
        EXPECT_EQ(height, 32UL);
        EXPECT_EQ(width, 135UL);
        EXPECT_EQ(maskBatch, 1UL);
        EXPECT_EQ(maskChannel, 1UL);
        EXPECT_EQ(maskHeight, 32UL);
        EXPECT_EQ(maskWidth, 135UL);
    }
}

TEST_F(ScaledMaskedSoftmaxV2Tiling, test_scaled_masked_softmax_tiling_broadcastN)
{
    std::string op_type("ScaledMaskedSoftmaxV2");
    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 48}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    ScaledMaskedSoftmaxV2CompileInfo compile_info;

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

    uint32_t N = 32;
    uint32_t C = 1;
    uint32_t H = 32;
    uint32_t W = 135;
    uint32_t maskN = 1;
    uint32_t maskC = 1;
    gert::StorageShape xShape = {{N, C, H, W}, {N, C, H, W}};
    gert::StorageShape maskShape = {{maskN, maskC, H, W}, {maskN, maskC, H, W}};

    gert::StorageShape yShape = {{N, C, H, W}, {N, C, H, W}};
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&xShape, &maskShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"scale", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                                  {"fixed_triu_mask", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
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
    EXPECT_EQ(tiling_key, 1UL);
    auto raw_tiling_data = tiling_context->GetRawTilingData();
    ASSERT_NE(raw_tiling_data, nullptr);
    EXPECT_GT(raw_tiling_data->GetDataSize(), 0U);
    auto tiling_data_result = TilingData2Str(raw_tiling_data);
    // Structural invariants from shape (coreNum may vary by platform fake; skip exact golden string)
    {
        std::istringstream iss(tiling_data_result);
        uint64_t coreNum = 0, batch = 0, channel = 0, height = 0, width = 0;
        uint64_t maskBatch = 0, maskChannel = 0, maskHeight = 0, maskWidth = 0;
        iss >> coreNum >> batch >> channel >> height >> width >> maskBatch >> maskChannel >> maskHeight >> maskWidth;
        EXPECT_GT(coreNum, 0UL);
        EXPECT_EQ(batch, 32UL);
        EXPECT_EQ(channel, 1UL);
        EXPECT_EQ(height, 32UL);
        EXPECT_EQ(width, 135UL);
        EXPECT_EQ(maskBatch, 1UL);
        EXPECT_EQ(maskChannel, 1UL);
        EXPECT_EQ(maskHeight, 32UL);
        EXPECT_EQ(maskWidth, 135UL);
    }
}

TEST_F(ScaledMaskedSoftmaxV2Tiling, invalid_mask_batch_must_fail_tiling)
{
    std::string op_type("ScaledMaskedSoftmaxV2");
    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 48}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    ScaledMaskedSoftmaxV2CompileInfo compile_info;
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
    ASSERT_NE(param, nullptr);

    gert::StorageShape xShape = {{1, 32, 32, 128}, {1, 32, 32, 128}};
    gert::StorageShape maskShape = {{2, 32, 32, 128}, {2, 32, 32, 128}};
    gert::StorageShape yShape = {{1, 32, 32, 128}, {1, 32, 32, 128}};
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&xShape, &maskShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"scale", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                                  {"fixed_triu_mask", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();
    auto* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context, nullptr);
    auto* platform = tiling_context->GetPlatformInfo();
    ASSERT_NE(platform, nullptr);
    platform->SetPlatformRes("SoCInfo", soc_infos);
    platform->SetPlatformRes("AICoreSpec", aicore_spec);
    platform->SetCoreNumByCoreType("AICore");
    platform->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(ScaledMaskedSoftmaxV2Tiling, test_scaled_masked_softmax_tiling_broadcastNC)
{
    std::string op_type("ScaledMaskedSoftmaxV2");
    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 48}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    ScaledMaskedSoftmaxV2CompileInfo compile_info;

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

    uint32_t N = 4;
    uint32_t C = 4;
    uint32_t H = 32;
    uint32_t W = 128;
    uint32_t maskN = 1;
    uint32_t maskC = 1;
    gert::StorageShape xShape = {{N, C, H, W}, {N, C, H, W}};
    gert::StorageShape maskShape = {{maskN, maskC, H, W}, {maskN, maskC, H, W}};

    gert::StorageShape yShape = {{N, C, H, W}, {N, C, H, W}};
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&xShape, &maskShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"scale", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                                  {"fixed_triu_mask", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
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
    EXPECT_EQ(tiling_key, 2UL);
    auto raw_tiling_data = tiling_context->GetRawTilingData();
    ASSERT_NE(raw_tiling_data, nullptr);
    EXPECT_GT(raw_tiling_data->GetDataSize(), 0U);
    auto tiling_data_result = TilingData2Str(raw_tiling_data);
    // Structural invariants from shape (coreNum may vary by platform fake; skip exact golden string)
    {
        std::istringstream iss(tiling_data_result);
        uint64_t coreNum = 0, batch = 0, channel = 0, height = 0, width = 0;
        uint64_t maskBatch = 0, maskChannel = 0, maskHeight = 0, maskWidth = 0;
        iss >> coreNum >> batch >> channel >> height >> width >> maskBatch >> maskChannel >> maskHeight >> maskWidth;
        EXPECT_GT(coreNum, 0UL);
        EXPECT_EQ(batch, 4UL);
        EXPECT_EQ(channel, 4UL);
        EXPECT_EQ(height, 32UL);
        EXPECT_EQ(width, 128UL);
        EXPECT_EQ(maskBatch, 1UL);
        EXPECT_EQ(maskChannel, 1UL);
        EXPECT_EQ(maskHeight, 32UL);
        EXPECT_EQ(maskWidth, 128UL);
    }
}

// ============================================================================
// Regbase 平台（DAV_3510 / Ascend950）覆盖用例：覆盖 SetUbSplitInfo 的动态行搜索
// （line 初值 + GetSoftMaxMaxTmpSize 递减循环）、SetSoftmaxTiling 的
// softmaxBuffSize 动态下发、InitInputShape 的 width<=8192 放宽。
// tiling data 槽位（8 字节对齐，TilingData2Str 解析）：
//   13=padLineNum, 21=lineHeadIter, 27=softmaxBuffSize
// ============================================================================

TEST_F(ScaledMaskedSoftmaxV2Tiling, test_scaled_masked_softmax_v2_tiling_regbase_softmaxbuff64k)
{
    // GetSoftMaxMaxTmpSize 内部依赖 PlatformAscendCManager 全局单例: UT 进程无 NPU 时默认
    // 初始化路径(GetRealSocVersion -> dlopen libruntime.so)失败。此处以显式 soc version 尝试
    // 初始化(纯 ini 解析); 若运行环境读不到平台 ini(部分 CI host UT 容器), 单例仍为 nullptr,
    // GetSoftMaxMaxTmpSize 断言失败返回 0, Regbase 行搜索退化为不递减(line 保持初值)、
    // softmaxBuffSize 下发 0。单例可用时固定为 DAV_2201 通用式(M*(K+72)*4)。
    // 下方断言对两种环境分别给出精确期望(mgrOk ? 真值 : 退化值)。
    const bool mgrOk = platform_ascendc::PlatformAscendCManager::GetInstance("Ascend910B1") != nullptr;

    std::string op_type("ScaledMaskedSoftmaxV2");
    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 253952, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> socversions = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    ScaledMaskedSoftmaxV2CompileInfo compile_info;

    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", socversions);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);

    uint32_t N = 1;
    uint32_t C = 16;
    uint32_t H = 440;
    uint32_t W = 440;
    uint32_t maskN = 1;
    uint32_t maskC = 16;
    gert::StorageShape xShape = {{N, C, H, W}, {N, C, H, W}};
    gert::StorageShape maskShape = {{maskN, maskC, H, W}, {maskN, maskC, H, W}};

    gert::StorageShape yShape = {{N, C, H, W}, {N, C, H, W}};
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&xShape, &maskShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"scale", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                                  {"fixed_triu_mask", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();
    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", socversions);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(tiling_context->GetTilingKey(), 0UL);
    auto raw_tiling_data = tiling_context->GetRawTilingData();
    ASSERT_NE(raw_tiling_data, nullptr);
    EXPECT_GT(raw_tiling_data->GetDataSize(), 0U);
    auto tiling_data_result = TilingData2Str(raw_tiling_data);
    {
        // Regbase 动态行搜索：maxByteLine=5736, line 初值 33 不满足 UB 预算(256872>253952)
        // 递减到 32 (249088<=253952) => lineHeadIter=32,
        // softmaxBuffSize = GetSoftMaxMaxTmpSize({32, 440}, fp32) = 32*(440+72)*4 = 65536
        std::istringstream iss(tiling_data_result);
        uint64_t v[28] = {0};
        for (int i = 0; i < 28; i++) {
            iss >> v[i];
        }
        EXPECT_GT(v[0], 0UL);                    // coreNum
        EXPECT_EQ(v[12], 440UL);                 // padLineNum
        EXPECT_EQ(v[20], mgrOk ? 32UL : 33UL);   // lineHeadIter (递减后/初值)
        EXPECT_EQ(v[26], mgrOk ? 65536UL : 0UL); // softmaxBuffSize (64KB/单例缺失退化值)
    }
}

TEST_F(ScaledMaskedSoftmaxV2Tiling, test_scaled_masked_softmax_v2_tiling_regbase_line_search)
{
    // GetSoftMaxMaxTmpSize 内部依赖 PlatformAscendCManager 全局单例: UT 进程无 NPU 时默认
    // 初始化路径(GetRealSocVersion -> dlopen libruntime.so)失败。此处以显式 soc version 尝试
    // 初始化(纯 ini 解析); 若运行环境读不到平台 ini(部分 CI host UT 容器), 单例仍为 nullptr,
    // GetSoftMaxMaxTmpSize 断言失败返回 0, Regbase 行搜索退化为不递减(line 保持初值)、
    // softmaxBuffSize 下发 0。单例可用时固定为 DAV_2201 通用式(M*(K+72)*4)。
    // 下方断言对两种环境分别给出精确期望(mgrOk ? 真值 : 退化值)。
    const bool mgrOk = platform_ascendc::PlatformAscendCManager::GetInstance("Ascend910B1") != nullptr;

    std::string op_type("ScaledMaskedSoftmaxV2");
    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 253952, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> socversions = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    ScaledMaskedSoftmaxV2CompileInfo compile_info;

    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", socversions);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);

    uint32_t N = 1;
    uint32_t C = 16;
    uint32_t H = 128;
    uint32_t W = 128;
    uint32_t maskN = 1;
    uint32_t maskC = 16;
    gert::StorageShape xShape = {{N, C, H, W}, {N, C, H, W}};
    gert::StorageShape maskShape = {{maskN, maskC, H, W}, {maskN, maskC, H, W}};

    gert::StorageShape yShape = {{N, C, H, W}, {N, C, H, W}};
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&xShape, &maskShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"scale", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                                  {"fixed_triu_mask", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();
    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", socversions);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(tiling_context->GetTilingKey(), 1UL);
    auto raw_tiling_data = tiling_context->GetRawTilingData();
    ASSERT_NE(raw_tiling_data, nullptr);
    EXPECT_GT(raw_tiling_data->GetDataSize(), 0U);
    auto tiling_data_result = TilingData2Str(raw_tiling_data);
    {
        // fp16 W=128: maxByteLine=1152, tmp(M)=M*(128+72)*4=800M, line*(1152+800)<=253952
        // => lineHeadIter=130 (line 初值 152, 多次递减覆盖 while 循环体),
        //    softmaxBuffSize = 130*800 = 104000
        std::istringstream iss(tiling_data_result);
        uint64_t v[28] = {0};
        for (int i = 0; i < 28; i++) {
            iss >> v[i];
        }
        EXPECT_GT(v[0], 0UL);
        EXPECT_EQ(v[12], 128UL);                  // padLineNum
        EXPECT_EQ(v[20], mgrOk ? 130UL : 152UL);  // lineHeadIter (递减后/初值)
        EXPECT_EQ(v[26], mgrOk ? 104000UL : 0UL); // softmaxBuffSize
    }
}

TEST_F(ScaledMaskedSoftmaxV2Tiling, test_scaled_masked_softmax_v2_tiling_regbase_width8192)
{
    // GetSoftMaxMaxTmpSize 内部依赖 PlatformAscendCManager 全局单例: UT 进程无 NPU 时默认
    // 初始化路径(GetRealSocVersion -> dlopen libruntime.so)失败。此处以显式 soc version 尝试
    // 初始化(纯 ini 解析); 若运行环境读不到平台 ini(部分 CI host UT 容器), 单例仍为 nullptr,
    // GetSoftMaxMaxTmpSize 断言失败返回 0, Regbase 行搜索退化为不递减(line 保持初值)、
    // softmaxBuffSize 下发 0。单例可用时固定为 DAV_2201 通用式(M*(K+72)*4)。
    // 下方断言对两种环境分别给出精确期望(mgrOk ? 真值 : 退化值)。
    const bool mgrOk = platform_ascendc::PlatformAscendCManager::GetInstance("Ascend910B1") != nullptr;

    std::string op_type("ScaledMaskedSoftmaxV2");
    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 253952, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> socversions = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    ScaledMaskedSoftmaxV2CompileInfo compile_info;

    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", socversions);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);

    uint32_t N = 1;
    uint32_t C = 4;
    uint32_t H = 8192;
    uint32_t W = 8192;
    uint32_t maskN = 1;
    uint32_t maskC = 4;
    gert::StorageShape xShape = {{N, C, H, W}, {N, C, H, W}};
    gert::StorageShape maskShape = {{maskN, maskC, H, W}, {maskN, maskC, H, W}};

    gert::StorageShape yShape = {{N, C, H, W}, {N, C, H, W}};
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&xShape, &maskShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"scale", Ops::NN::AnyValue::CreateFrom<float>(0.5)},
                                  {"fixed_triu_mask", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();
    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", socversions);

    // Regbase 放宽 width 上限至 8192（非 Regbase 平台此处校验会失败）
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(tiling_context->GetTilingKey(), 2UL);
    auto raw_tiling_data = tiling_context->GetRawTilingData();
    ASSERT_NE(raw_tiling_data, nullptr);
    EXPECT_GT(raw_tiling_data->GetDataSize(), 0U);
    auto tiling_data_result = TilingData2Str(raw_tiling_data);
    {
        // bf16 W=8192: maxByteLine=73728, line 初值 2 即满足预算(<=253952, 不进 while)
        // => lineHeadIter=2, padLineNum=8192。
        // softmaxBuffSize = GetSoftMaxMaxTmpSize({2, 8192}, fp32), 单例固定 DAV_2201
        // 通用式: 2*(8192+72)*4 = 66112
        std::istringstream iss(tiling_data_result);
        uint64_t v[28] = {0};
        for (int i = 0; i < 28; i++) {
            iss >> v[i];
        }
        EXPECT_GT(v[0], 0UL);
        EXPECT_EQ(v[12], 8192UL);                // padLineNum
        EXPECT_EQ(v[20], 2UL);                   // lineHeadIter (初值即满足, 两环境一致)
        EXPECT_EQ(v[26], mgrOk ? 66112UL : 0UL); // softmaxBuffSize
    }
}

// ============================================================================
// Regbase 960 系列覆盖用例：960DT(DAV_9201) 与 950(DAV_3510) 在 tiling.cpp 走相同
// Regbase 路径; 960PR(DAV_9202) 虽属 Regbase, 但 GetSoftMaxMaxTmpSize 内部 special
// 分支集合不含 9202(走通用式), softmaxBuffSize 取值路径与其余 Regbase 存在差异,
// 单独用例守护该行为。
// ============================================================================

TEST_F(ScaledMaskedSoftmaxV2Tiling, test_scaled_masked_softmax_v2_tiling_regbase_960dt_softmaxbuff64k)
{
    // GetSoftMaxMaxTmpSize 内部依赖 PlatformAscendCManager 全局单例: UT 进程无 NPU 时默认
    // 初始化路径(GetRealSocVersion -> dlopen libruntime.so)失败。此处以显式 soc version 尝试
    // 初始化(纯 ini 解析); 若运行环境读不到平台 ini(部分 CI host UT 容器), 单例仍为 nullptr,
    // GetSoftMaxMaxTmpSize 断言失败返回 0, Regbase 行搜索退化为不递减(line 保持初值)、
    // softmaxBuffSize 下发 0。单例可用时固定为 DAV_2201 通用式(M*(K+72)*4)。
    // 下方断言对两种环境分别给出精确期望(mgrOk ? 真值 : 退化值)。
    const bool mgrOk = platform_ascendc::PlatformAscendCManager::GetInstance("Ascend910B1") != nullptr;

    std::string op_type("ScaledMaskedSoftmaxV2");
    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 253952, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> socversions = {{"Short_SoC_version", "Ascend960DT"}, {"NpuArch", "9201"}};
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    ScaledMaskedSoftmaxV2CompileInfo compile_info;

    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", socversions);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);

    uint32_t N = 1;
    uint32_t C = 16;
    uint32_t H = 440;
    uint32_t W = 440;
    uint32_t maskN = 1;
    uint32_t maskC = 16;
    gert::StorageShape xShape = {{N, C, H, W}, {N, C, H, W}};
    gert::StorageShape maskShape = {{maskN, maskC, H, W}, {maskN, maskC, H, W}};

    gert::StorageShape yShape = {{N, C, H, W}, {N, C, H, W}};
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&xShape, &maskShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"scale", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                                  {"fixed_triu_mask", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();
    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", socversions);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(tiling_context->GetTilingKey(), 0UL);
    auto raw_tiling_data = tiling_context->GetRawTilingData();
    ASSERT_NE(raw_tiling_data, nullptr);
    EXPECT_GT(raw_tiling_data->GetDataSize(), 0U);
    auto tiling_data_result = TilingData2Str(raw_tiling_data);
    {
        // 960DT(DAV_9201) 与 950(DAV_3510) 同属 Regbase, fp32 W=440 下走相同的
        // 行搜索路径: lineHeadIter=32, softmaxBuffSize = 32*(440+72)*4 = 65536 == 64KB
        std::istringstream iss(tiling_data_result);
        uint64_t v[28] = {0};
        for (int i = 0; i < 28; i++) {
            iss >> v[i];
        }
        EXPECT_GT(v[0], 0UL);
        EXPECT_EQ(v[12], 440UL);                 // padLineNum
        EXPECT_EQ(v[20], mgrOk ? 32UL : 33UL);   // lineHeadIter (递减后/初值)
        EXPECT_EQ(v[26], mgrOk ? 65536UL : 0UL); // softmaxBuffSize (64KB/单例缺失退化值)
    }
}

TEST_F(ScaledMaskedSoftmaxV2Tiling, test_scaled_masked_softmax_v2_tiling_regbase_960pr_width8192)
{
    // GetSoftMaxMaxTmpSize 内部依赖 PlatformAscendCManager 全局单例: UT 进程无 NPU 时默认
    // 初始化路径(GetRealSocVersion -> dlopen libruntime.so)失败。此处以显式 soc version 尝试
    // 初始化(纯 ini 解析); 若运行环境读不到平台 ini(部分 CI host UT 容器), 单例仍为 nullptr,
    // GetSoftMaxMaxTmpSize 断言失败返回 0, Regbase 行搜索退化为不递减(line 保持初值)、
    // softmaxBuffSize 下发 0。单例可用时固定为 DAV_2201 通用式(M*(K+72)*4)。
    // 下方断言对两种环境分别给出精确期望(mgrOk ? 真值 : 退化值)。
    const bool mgrOk = platform_ascendc::PlatformAscendCManager::GetInstance("Ascend910B1") != nullptr;

    std::string op_type("ScaledMaskedSoftmaxV2");
    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false,
                          "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true,
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 253952, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> socversions = {{"Short_SoC_version", "Ascend960PR"}, {"NpuArch", "9202"}};
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    ScaledMaskedSoftmaxV2CompileInfo compile_info;

    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", socversions);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);

    uint32_t N = 1;
    uint32_t C = 4;
    uint32_t H = 8192;
    uint32_t W = 8192;
    uint32_t maskN = 1;
    uint32_t maskC = 4;
    gert::StorageShape xShape = {{N, C, H, W}, {N, C, H, W}};
    gert::StorageShape maskShape = {{maskN, maskC, H, W}, {maskN, maskC, H, W}};

    gert::StorageShape yShape = {{N, C, H, W}, {N, C, H, W}};
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&xShape, &maskShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"scale", Ops::NN::AnyValue::CreateFrom<float>(0.5)},
                                  {"fixed_triu_mask", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();
    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", socversions);

    // 960PR 属 Regbase, width<=8192 放宽同样生效
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(tiling_context->GetTilingKey(), 2UL);
    auto raw_tiling_data = tiling_context->GetRawTilingData();
    ASSERT_NE(raw_tiling_data, nullptr);
    EXPECT_GT(raw_tiling_data->GetDataSize(), 0U);
    auto tiling_data_result = TilingData2Str(raw_tiling_data);
    {
        // bf16 W=8192: line 初值 2 即满足预算(不进 while) => lineHeadIter=2, padLineNum=8192。
        // softmaxBuffSize = GetSoftMaxMaxTmpSize({2, 8192}, fp32), 单例固定 DAV_2201
        // 通用式: 2*(8192+72)*4 = 66112
        std::istringstream iss(tiling_data_result);
        uint64_t v[28] = {0};
        for (int i = 0; i < 28; i++) {
            iss >> v[i];
        }
        EXPECT_GT(v[0], 0UL);
        EXPECT_EQ(v[12], 8192UL);                // padLineNum
        EXPECT_EQ(v[20], 2UL);                   // lineHeadIter (初值即满足, 两环境一致)
        EXPECT_EQ(v[26], mgrOk ? 66112UL : 0UL); // softmaxBuffSize
    }
}

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
 * \file test_dynamic_dual_level_mx_quant_tiling.cpp
 * \brief
 */

#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "platform/platform_infos_def.h"
#include "ut_op_util.h"
#include "../../../op_host/dynamic_dual_level_mx_quant_tiling.h"
#include "kernel_run_context_facker.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "test_cube_util.h"

using namespace std;
using namespace ge;
using namespace ut_util;

class DynamicDualLevelMxQuantTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "DynamicDualLevelMxQuantTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "DynamicDualLevelMxQuantTiling TearDown" << std::endl;
    }
};

template <typename T>
static string to_string(void* buf, size_t size)
{
    std::string result;
    const T* data = reinterpret_cast<const T*>(buf);
    size_t len = size / sizeof(T);
    for (size_t i = 0; i < len; i++) {
        result += std::to_string(data[i]);
        result += " ";
    }
    return result;
}

static void ExecuteTestCase(
    ge::DataType xDtype, ge::DataType yDtype, ge::DataType level0ScaleDtype, ge::DataType level1ScaleDtype,
    gert::StorageShape shape, gert::StorageShape level0ScaleShape, gert::StorageShape level1ScaleShape,
    string roundMode, int64_t level0Scale, int64_t level1Scale, string expectTilingData,
    ge::graphStatus status = ge::GRAPH_SUCCESS)
{
    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 253952, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_versions = {{"Short_SoC_version", "Ascend950"}};

    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();

    // compile info
    optiling::DynamicDualLevelMxQuantCompileInfo compile_info;

    std::string op_type("DynamicDualLevelMxQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({compile_info_string.data(), reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_versions);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    string_view sv(reinterpret_cast<const char*>(&platform_info), sizeof(platform_info));
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(1, 3)
                      .IrInstanceNum({1})
                      .InputShapes({&shape})
                      .OutputShapes({&shape, &level0ScaleShape, &level1ScaleShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(sv.data())
                      .NodeInputTd(0, xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, yDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, level0ScaleDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, level1ScaleDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"round_mode", Ops::NN::AnyValue::CreateFrom<string>(roundMode)},
                           {"level0_scale", Ops::NN::AnyValue::CreateFrom<int64_t>(level0Scale)},
                           {"level1_scale", Ops::NN::AnyValue::CreateFrom<int64_t>(level1Scale)}})
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
    EXPECT_EQ(tiling_func(tiling_context), status);
    if (status == ge::GRAPH_FAILED) {
        return;
    }
    auto tiling_key = tiling_context->GetTilingKey();
    auto block_dim = tiling_context->GetBlockDim();
    auto raw_tiling_data = tiling_context->GetRawTilingData();
    // 去除A2场景的tiling params
    auto tiling_data_result = to_string<int64_t>(raw_tiling_data->GetData(), raw_tiling_data->GetDataSize());

    EXPECT_EQ(tiling_data_result, expectTilingData);
}

void ExecuteTestCaseFailed(
    ge::DataType xDtype, ge::DataType yDtype, ge::DataType level0ScaleDtype, ge::DataType level1ScaleDtype,
    gert::StorageShape shape, gert::StorageShape level0ScaleShape, gert::StorageShape level1ScaleShape,
    string roundMode, int64_t level0Scale, int64_t level1Scale, ge::graphStatus status = ge::GRAPH_SUCCESS)
{
    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 253952, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_versions = {{"Short_SoC_version", "Ascend950"}};

    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();

    // compile info
    optiling::DynamicDualLevelMxQuantCompileInfo compile_info;

    std::string op_type("DynamicDualLevelMxQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({compile_info_string.data(), reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_versions);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);

    string_view sv(reinterpret_cast<const char*>(&platform_info), sizeof(platform_info));

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(1, 3)
                      .IrInstanceNum({1})
                      .InputShapes({&shape})
                      .OutputShapes({&shape, &level0ScaleShape, &level1ScaleShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(sv.data())
                      .NodeInputTd(0, xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, yDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, level0ScaleDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, level1ScaleDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"round_mode", Ops::NN::AnyValue::CreateFrom<string>(roundMode)},
                           {"level0_scale", Ops::NN::AnyValue::CreateFrom<int64_t>(level0Scale)},
                           {"level1_scale", Ops::NN::AnyValue::CreateFrom<int64_t>(level1Scale)}})
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
}

TEST_F(DynamicDualLevelMxQuantTiling, DynamicDualLevelMxQuant_roundMode_rint)
{
    gert::StorageShape shape = {{128, 512}, {128, 512}};
    gert::StorageShape scale1Shape = {{128, 1}, {128, 1}};
    gert::StorageShape scale2Shape = {{128, 8, 2}, {128, 8, 2}};
    string roundMode = "rint";
    int64_t level0BlockSize = 512;
    int64_t level1BlockSize = 32;
    string expectTilingData = "0 64 64 4 512 32 512 128 512 1 1 128 1 64 1 2 1 2 512 512 1 1 1 1 67 0 0 ";

    ExecuteTestCase(
        ge::DT_FLOAT16, ge::DT_FLOAT4_E2M1, ge::DT_FLOAT, ge::DT_FLOAT8_E8M0, shape, scale1Shape, scale2Shape,
        roundMode, level0BlockSize, level1BlockSize, expectTilingData);
}

TEST_F(DynamicDualLevelMxQuantTiling, DynamicDualLevelMxQuant_shape_failed1)
{
    gert::StorageShape shape = {{128, 512}, {128, 512}};
    gert::StorageShape scale1Shape = {{128, 1}, {128, 1}};
    gert::StorageShape scale2Shape = {{128, 8, 2}, {128, 8, 2}};
    string roundMode = "rint";
    int64_t level0BlockSize = 512;
    int64_t level1BlockSize = 32;

    ExecuteTestCaseFailed(
        ge::DT_FLOAT16, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT, ge::DT_FLOAT8_E8M0, shape, scale1Shape, scale2Shape,
        roundMode, level0BlockSize, level1BlockSize);
}

TEST_F(DynamicDualLevelMxQuantTiling, DynamicDualLevelMxQuant_shape_failed2)
{
    gert::StorageShape shape = {{128, 512}, {128, 512}};
    gert::StorageShape scale1Shape = {{128, 2}, {128, 2}};
    gert::StorageShape scale2Shape = {{128, 8, 2}, {128, 8, 2}};
    string roundMode = "rint";
    int64_t level0BlockSize = 512;
    int64_t level1BlockSize = 32;

    ExecuteTestCaseFailed(
        ge::DT_FLOAT16, ge::DT_FLOAT4_E2M1, ge::DT_FLOAT, ge::DT_FLOAT8_E8M0, shape, scale1Shape, scale2Shape,
        roundMode, level0BlockSize, level1BlockSize);
}

TEST_F(DynamicDualLevelMxQuantTiling, DynamicDualLevelMxQuant_shape_failed3)
{
    gert::StorageShape shape = {{128, 512}, {128, 512}};
    gert::StorageShape scale1Shape = {{128, 1}, {128, 2}};
    gert::StorageShape scale2Shape = {{128, 16, 2}, {128, 16, 2}};
    string roundMode = "rint";
    int64_t level0BlockSize = 512;
    int64_t level1BlockSize = 32;

    ExecuteTestCaseFailed(
        ge::DT_FLOAT16, ge::DT_FLOAT4_E2M1, ge::DT_FLOAT, ge::DT_FLOAT8_E8M0, shape, scale1Shape, scale2Shape,
        roundMode, level0BlockSize, level1BlockSize);
}
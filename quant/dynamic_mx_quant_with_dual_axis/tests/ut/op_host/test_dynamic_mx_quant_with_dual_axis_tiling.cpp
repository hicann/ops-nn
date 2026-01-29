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
 * \file test_dynamic_mx_quant_with_dual_axis_tiling.cpp
 * \brief
 */

#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "platform/platform_infos_def.h"
#include "ut_op_util.h"
#include "quant/dynamic_mx_quant_with_dual_axis/op_host/dynamic_mx_quant_with_dual_axis_tiling.h"
#include "kernel_run_context_facker.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "test_cube_util.h"

using namespace std;
using namespace ge;
using namespace ut_util;

class DynamicMxQuantWithDualAxisTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "DynamicMxQuantWithDualAxisTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "DynamicMxQuantWithDualAxisTiling TearDown" << std::endl;
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
    ge::DataType inDtype, ge::DataType yDtype, ge::DataType scaleDtype, gert::StorageShape shape,
    gert::StorageShape scale1Shape, gert::StorageShape scale2Shape, string roundMode, int64_t dstDtype,
    int64_t scaleAlg, string expectTilingData, ge::graphStatus status = ge::GRAPH_SUCCESS)
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
    optiling::DynamicMxQuantWithDualAxisCompileInfo compile_info;

    std::string op_type("DynamicMxQuantWithDualAxis");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>("{}"), reinterpret_cast<void*>(&platform_info)})
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
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(1, 4)
                      .IrInstanceNum({1})
                      .InputShapes({&shape})
                      .OutputShapes({&shape, &scale1Shape, &shape, &scale2Shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, inDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, yDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, scaleDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, yDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, scaleDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"round_mode", Ops::NN::AnyValue::CreateFrom<string>(roundMode)},
                           {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(dstDtype)},
                           {"scale_alg", Ops::NN::AnyValue::CreateFrom<int64_t>(scaleAlg)}})
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
    ge::DataType inDtype, ge::DataType yDtype, ge::DataType scaleDtype, gert::StorageShape shape,
    gert::StorageShape scale1Shape, gert::StorageShape scale2Shape, string roundMode, int64_t dstDtype,
    int64_t scaleAlg)
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
    optiling::DynamicMxQuantWithDualAxisCompileInfo compile_info;

    std::string op_type("DynamicMxQuantWithDualAxis");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>("{}"), reinterpret_cast<void*>(&platform_info)})
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
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(1, 4)
                      .IrInstanceNum({1})
                      .InputShapes({&shape})
                      .OutputShapes({&shape, &scale1Shape, &shape, &scale2Shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, inDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, yDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, scaleDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, yDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, scaleDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"round_mode", Ops::NN::AnyValue::CreateFrom<string>(roundMode)},
                           {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(dstDtype)},
                           {"scale_alg", Ops::NN::AnyValue::CreateFrom<int64_t>(scaleAlg)}})
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

TEST_F(DynamicMxQuantWithDualAxisTiling, DynamicMxQuantWithDualAxis_roundMode_rint)
{
    gert::StorageShape shape = {{128, 256}, {128, 256}};
    gert::StorageShape scale1Shape = {{128, 4, 2}, {128, 4, 2}};
    gert::StorageShape scale2Shape = {{2, 256, 2}, {2, 256, 2}};
    string roundMode = "rint";
    int64_t dstDtype = 35;
    int64_t scaleAlg = 0;
    string expectTilingData = "64 2 4 35 0 32 1 128 256 256 64 0 64 256 2 1 1 0 2 0 0 0 2 8 4 ";

    ExecuteTestCase(
        ge::DT_FLOAT16, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E8M0, shape, scale1Shape, scale2Shape, roundMode, dstDtype,
        scaleAlg, expectTilingData);
}

TEST_F(DynamicMxQuantWithDualAxisTiling, DynamicMxQuantWithDualAxis_roundMode_round)
{
    gert::StorageShape shape = {{128, 256}, {128, 256}};
    gert::StorageShape scale1Shape = {{128, 4, 2}, {128, 4, 2}};
    gert::StorageShape scale2Shape = {{2, 256, 2}, {2, 256, 2}};
    string roundMode = "round";
    int64_t dstDtype = 41;
    int64_t scaleAlg = 0;
    string expectTilingData = "64 2 0 41 0 32 1 128 256 256 64 0 64 256 2 1 1 0 2 0 0 0 2 8 4 ";

    ExecuteTestCase(
        ge::DT_FLOAT16, ge::DT_FLOAT4_E1M2, ge::DT_FLOAT8_E8M0, shape, scale1Shape, scale2Shape, roundMode, dstDtype,
        scaleAlg, expectTilingData);
}

TEST_F(DynamicMxQuantWithDualAxisTiling, DynamicMxQuantWithDualAxis_roundMode_floor)
{
    gert::StorageShape shape = {{128, 256}, {128, 256}};
    gert::StorageShape scale1Shape = {{128, 4, 2}, {128, 4, 2}};
    gert::StorageShape scale2Shape = {{2, 256, 2}, {2, 256, 2}};
    string roundMode = "floor";
    int64_t dstDtype = 41;
    int64_t scaleAlg = 0;
    string expectTilingData = "64 2 1 41 0 32 1 128 256 256 64 0 64 256 2 1 1 0 2 0 0 0 2 8 4 ";

    ExecuteTestCase(
        ge::DT_FLOAT16, ge::DT_FLOAT4_E1M2, ge::DT_FLOAT8_E8M0, shape, scale1Shape, scale2Shape, roundMode, dstDtype,
        scaleAlg, expectTilingData);
}

TEST_F(DynamicMxQuantWithDualAxisTiling, DynamicMxQuantWithDualAxis_To_Fp4)
{
    gert::StorageShape shape = {{128, 256}, {128, 256}};
    gert::StorageShape scale1Shape = {{128, 4, 2}, {128, 4, 2}};
    gert::StorageShape scale2Shape = {{2, 256, 2}, {2, 256, 2}};
    string roundMode = "floor";
    int64_t dstDtype = 41;
    int64_t scaleAlg = 0;
    string expectTilingData = "64 2 1 41 0 32 1 128 256 256 64 0 64 256 2 1 1 0 2 0 0 0 2 8 4 ";

    ExecuteTestCase(
        ge::DT_FLOAT16, ge::DT_FLOAT4_E1M2, ge::DT_FLOAT8_E8M0, shape, scale1Shape, scale2Shape, roundMode, dstDtype,
        scaleAlg, expectTilingData);
}

TEST_F(DynamicMxQuantWithDualAxisTiling, DynamicMxQuantWithDualAxis_To_Fp8)
{
    gert::StorageShape shape = {{128, 256}, {128, 256}};
    gert::StorageShape scale1Shape = {{128, 4, 2}, {128, 4, 2}};
    gert::StorageShape scale2Shape = {{2, 256, 2}, {2, 256, 2}};
    string roundMode = "rint";
    int64_t dstDtype = 35;
    int64_t scaleAlg = 0;
    string expectTilingData = "64 2 4 35 0 32 1 128 256 256 64 0 64 256 2 1 1 0 2 0 0 0 2 8 4 ";

    ExecuteTestCase(
        ge::DT_FLOAT16, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E8M0, shape, scale1Shape, scale2Shape, roundMode, dstDtype,
        scaleAlg, expectTilingData);
}

TEST_F(DynamicMxQuantWithDualAxisTiling, DynamicMxQuantWithDualAxis_From_fp16)
{
    gert::StorageShape shape = {{128, 256}, {128, 256}};
    gert::StorageShape scale1Shape = {{128, 4, 2}, {128, 4, 2}};
    gert::StorageShape scale2Shape = {{2, 256, 2}, {2, 256, 2}};
    string roundMode = "floor";
    int64_t dstDtype = 41;
    int64_t scaleAlg = 0;
    string expectTilingData = "64 2 1 41 0 32 1 128 256 256 64 0 64 256 2 1 1 0 2 0 0 0 2 8 4 ";

    ExecuteTestCase(
        ge::DT_FLOAT16, ge::DT_FLOAT4_E1M2, ge::DT_FLOAT8_E8M0, shape, scale1Shape, scale2Shape, roundMode, dstDtype,
        scaleAlg, expectTilingData);
}

TEST_F(DynamicMxQuantWithDualAxisTiling, DynamicMxQuantWithDualAxis_From_Bf16)
{
    gert::StorageShape shape = {{128, 256}, {128, 256}};
    gert::StorageShape scale1Shape = {{128, 4, 2}, {128, 4, 2}};
    gert::StorageShape scale2Shape = {{2, 256, 2}, {2, 256, 2}};
    string roundMode = "floor";
    int64_t dstDtype = 41;
    int64_t scaleAlg = 0;
    string expectTilingData = "64 2 1 41 0 32 1 128 256 256 64 0 64 256 2 1 1 0 2 0 0 0 2 8 4 ";

    ExecuteTestCase(
        ge::DT_BF16, ge::DT_FLOAT4_E1M2, ge::DT_FLOAT8_E8M0, shape, scale1Shape, scale2Shape, roundMode, dstDtype,
        scaleAlg, expectTilingData);
}

TEST_F(DynamicMxQuantWithDualAxisTiling, DynamicMxQuantWithDualAxis_dstDtype_yDtype_not_corresponded)
{
    gert::StorageShape shape = {{128, 256}, {128, 256}};
    gert::StorageShape scale1Shape = {{128, 4, 2}, {128, 4, 2}};
    gert::StorageShape scale2Shape = {{2, 256, 2}, {2, 256, 2}};
    string roundMode = "rint";
    int64_t dstDtype = 36;
    int64_t scaleAlg = 0;

    ExecuteTestCaseFailed(
        ge::DT_FLOAT16, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E8M0, shape, scale1Shape, scale2Shape, roundMode, dstDtype,
        scaleAlg);
}

TEST_F(DynamicMxQuantWithDualAxisTiling, DynamicMxQuantWithDualAxis_yDtype_failed)
{
    gert::StorageShape shape = {{128, 256}, {128, 256}};
    gert::StorageShape scale1Shape = {{128, 4, 2}, {128, 4, 2}};
    gert::StorageShape scale2Shape = {{2, 256, 2}, {2, 256, 2}};
    string roundMode = "rint";
    int64_t dstDtype = 35;
    int64_t scaleAlg = 0;

    ExecuteTestCaseFailed(
        ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT8_E8M0, shape, scale1Shape, scale2Shape, roundMode, dstDtype,
        scaleAlg);
}

TEST_F(DynamicMxQuantWithDualAxisTiling, DynamicMxQuantWithDualAxis_xDtype_failed)
{
    gert::StorageShape shape = {{128, 256}, {128, 256}};
    gert::StorageShape scale1Shape = {{128, 4, 2}, {128, 4, 2}};
    gert::StorageShape scale2Shape = {{2, 256, 2}, {2, 256, 2}};
    string roundMode = "rint";
    int64_t dstDtype = 35;
    int64_t scaleAlg = 0;

    ExecuteTestCaseFailed(
        ge::DT_FLOAT, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E8M0, shape, scale1Shape, scale2Shape, roundMode, dstDtype,
        scaleAlg);
}

TEST_F(DynamicMxQuantWithDualAxisTiling, DynamicMxQuantWithDualAxis_scaleDtype_failed)
{
    gert::StorageShape shape = {{128, 256}, {128, 256}};
    gert::StorageShape scale1Shape = {{128, 4, 2}, {128, 4, 2}};
    gert::StorageShape scale2Shape = {{2, 256, 2}, {2, 256, 2}};
    string roundMode = "rint";
    int64_t dstDtype = 35;
    int64_t scaleAlg = 0;

    ExecuteTestCaseFailed(
        ge::DT_FLOAT16, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E5M2, shape, scale1Shape, scale2Shape, roundMode, dstDtype,
        scaleAlg);
}

TEST_F(DynamicMxQuantWithDualAxisTiling, DynamicMxQuantWithDualAxis_roundMode_failed1)
{
    gert::StorageShape shape = {{128, 256}, {128, 256}};
    gert::StorageShape scale1Shape = {{128, 4, 2}, {128, 4, 2}};
    gert::StorageShape scale2Shape = {{2, 256, 2}, {2, 256, 2}};
    string roundMode = "trunc";
    int64_t dstDtype = 35;
    int64_t scaleAlg = 0;

    ExecuteTestCaseFailed(
        ge::DT_FLOAT16, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E8M0, shape, scale1Shape, scale2Shape, roundMode, dstDtype,
        scaleAlg);
}

TEST_F(DynamicMxQuantWithDualAxisTiling, DynamicMxQuantWithDualAxis_scaleAlg_failed)
{
    gert::StorageShape shape = {{128, 256}, {128, 256}};
    gert::StorageShape scale1Shape = {{128, 4, 2}, {128, 4, 2}};
    gert::StorageShape scale2Shape = {{2, 256, 2}, {2, 256, 2}};
    string roundMode = "rint";
    int64_t dstDtype = 35;
    int64_t scaleAlg = 1;

    ExecuteTestCaseFailed(
        ge::DT_FLOAT16, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E8M0, shape, scale1Shape, scale2Shape, roundMode, dstDtype,
        scaleAlg);
}

TEST_F(DynamicMxQuantWithDualAxisTiling, DynamicMxQuantWithDualAxis_roundMode_failed2)
{
    gert::StorageShape shape = {{128, 256}, {128, 256}};
    gert::StorageShape scale1Shape = {{128, 4, 2}, {128, 4, 2}};
    gert::StorageShape scale2Shape = {{2, 256, 2}, {2, 256, 2}};
    string roundMode = "round";
    int64_t dstDtype = 35;
    int64_t scaleAlg = 0;

    ExecuteTestCaseFailed(
        ge::DT_FLOAT16, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E8M0, shape, scale1Shape, scale2Shape, roundMode, dstDtype,
        scaleAlg);
}

TEST_F(DynamicMxQuantWithDualAxisTiling, DynamicMxQuantWithDualAxis_shape_failed1)
{
    gert::StorageShape shape = {{128, 256}, {128, 256}};
    gert::StorageShape scale1Shape = {{128, 5, 2}, {128, 5, 2}};
    gert::StorageShape scale2Shape = {{2, 256, 2}, {2, 256, 2}};
    string roundMode = "rint";
    int64_t dstDtype = 35;
    int64_t scaleAlg = 0;

    ExecuteTestCaseFailed(
        ge::DT_FLOAT16, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E8M0, shape, scale1Shape, scale2Shape, roundMode, dstDtype,
        scaleAlg);
}

TEST_F(DynamicMxQuantWithDualAxisTiling, DynamicMxQuantWithDualAxis_shape_failed2)
{
    gert::StorageShape shape = {{128, 256}, {128, 256}};
    gert::StorageShape scale1Shape = {{128, 4, 2}, {128, 4, 2}};
    gert::StorageShape scale2Shape = {{3, 256, 2}, {3, 256, 2}};
    string roundMode = "rint";
    int64_t dstDtype = 35;
    int64_t scaleAlg = 0;

    ExecuteTestCaseFailed(
        ge::DT_FLOAT16, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E8M0, shape, scale1Shape, scale2Shape, roundMode, dstDtype,
        scaleAlg);
}

TEST_F(DynamicMxQuantWithDualAxisTiling, DynamicMxQuantWithDualAxis_shape_failed3)
{
    gert::StorageShape shape = {{128, 255}, {128, 255}};
    gert::StorageShape scale1Shape = {{128, 4, 2}, {128, 4, 2}};
    gert::StorageShape scale2Shape = {{2, 255, 2}, {2, 255, 2}};
    string roundMode = "rint";
    int64_t dstDtype = 41;
    int64_t scaleAlg = 0;

    ExecuteTestCaseFailed(
        ge::DT_FLOAT16, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT4_E1M2, shape, scale1Shape, scale2Shape, roundMode, dstDtype,
        scaleAlg);
}

TEST_F(DynamicMxQuantWithDualAxisTiling, DynamicMxQuantWithDualAxis_shape_failed4)
{
    gert::StorageShape shape = {{256}, {256}};
    gert::StorageShape scale1Shape = {{128, 4, 2}, {128, 4, 2}};
    gert::StorageShape scale2Shape = {{2, 256, 2}, {2, 256, 2}};
    string roundMode = "rint";
    int64_t dstDtype = 35;
    int64_t scaleAlg = 0;

    ExecuteTestCaseFailed(
        ge::DT_FLOAT16, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E8M0, shape, scale1Shape, scale2Shape, roundMode, dstDtype,
        scaleAlg);
}
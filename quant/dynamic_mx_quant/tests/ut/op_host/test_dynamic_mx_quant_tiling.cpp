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
#include "../../../op_host/arch35/dynamic_mx_quant_tiling_arch35.h"

using namespace ut_util;
using namespace std;
using namespace ge;

class DynamicMxQuantTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "DynamicMxQuantTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "DynamicMxQuantTiling TearDown" << std::endl;
    }
};

static string to_string(const std::stringstream& tiling_data)
{
    auto data = tiling_data.str();
    string result;
    int64_t tmp = 0;
    for (size_t i = 0; i < data.length(); i += sizeof(int64_t)) {
        memcpy(&tmp, data.c_str() + i, sizeof(tmp));
        result += std::to_string(tmp);
        result += " ";
    }

    return result;
}

static void ExecuteTestCase(
    ge::DataType inDtype, ge::DataType outDtype, gert::StorageShape shape, gert::StorageShape scaleShape, int64_t axis,
    int64_t blockSize, string expectTilingData, ge::graphStatus status = ge::GRAPH_SUCCESS)
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

    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();

    // compile info
    optiling::DynamicMxQuantCompileInfo compile_info;

    std::string op_type("DynamicMxQuant");
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

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(1, 2)
                      .IrInstanceNum({1})
                      .InputShapes({&shape})
                      .OutputShapes({&shape, &scaleShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, inDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, outDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"axis", Ops::NN::AnyValue::CreateFrom(axis)},
                           {"round_mode", Ops::NN::AnyValue::CreateFrom<string>("rint")},
                           {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(outDtype)},
                           {"blocksize", Ops::NN::AnyValue::CreateFrom(blockSize)},
                           {"scale_alg", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
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
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();

    auto block_dim = tiling_context->GetBlockDim();
    //  ASSERT_EQ(block_dim, except_tilingkey);
    //  auto raw_tiling_data = tiling_context->GetRawTilingData();
    //  auto tiling_data_result = to_string<int64_t>(raw_tiling_data->GetData(), raw_tiling_data->GetDataSize());
    //  EXPECT_EQ(tiling_data_result, expectTilingData);
}

TEST_F(DynamicMxQuantTiling, DynamicMxQuant_tiling_ascendc_bfloat16_fp4e2m1_tail_axis)
{
    gert::StorageShape shape = {{60, 14, 16, 128}, {60, 14, 16, 128}};
    gert::StorageShape scaleShape = {{60, 14, 16, 2, 2}, {60, 14, 16, 2, 2}};
    int64_t axis = -1;
    int64_t blockSize = 32;
    string expectTilingData =
        "64 45 0 0 0 0 0 0 4 137438953512 32 4 0 0 1 13440 1 53760 2040 152 4 2 1 1 4 44 0 128 64 0 0 0 0 ";

    ExecuteTestCase(ge::DT_BF16, ge::DT_FLOAT4_E2M1, shape, scaleShape, axis, blockSize, expectTilingData);
}

TEST_F(DynamicMxQuantTiling, DynamicMxQuant_tiling_ascendc_bfloat16_fp4e2m1_not_tail_axis)
{
    gert::StorageShape shape = {{60, 14, 32, 128}, {60, 14, 32, 128}};
    gert::StorageShape scaleShape = {{60, 14, 1, 128, 2}, {60, 14, 1, 128, 2}};
    int64_t axis = -2;
    int64_t blockSize = 32;
    string expectTilingData =
        "64 56 3 3 0 168 5 5 4 137438953512 32 1 32 0 0 840 128 215040 2011 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";

    ExecuteTestCase(ge::DT_BF16, ge::DT_FLOAT4_E2M1, shape, scaleShape, axis, blockSize, expectTilingData);
}

TEST_F(DynamicMxQuantTiling, DynamicMxQuant_tiling_ascendc_bfloat16_fp4e1m2_not_tail_axis)
{
    gert::StorageShape shape = {{1024, 512, 64, 32}, {1024, 512, 64, 32}};
    gert::StorageShape scaleShape = {{1024, 512, 1, 32, 2}, {1024, 512, 1, 32, 2}};
    int64_t axis = 2;
    int64_t blockSize = 32;
    string expectTilingData =
        "64 64 820 769 0 52429 20 16 4 137438953513 32 2 32 0 0 524288 32 33554432 2121 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";

    ExecuteTestCase(ge::DT_BF16, ge::DT_FLOAT4_E1M2, shape, scaleShape, axis, blockSize, expectTilingData);
}

TEST_F(DynamicMxQuantTiling, DynamicMxQuant_tiling_ascendc_bfloat16_fp4e1m2_tail_axis)
{
    gert::StorageShape shape = {{1024, 512, 64, 32}, {1024, 512, 64, 32}};
    gert::StorageShape scaleShape = {{1024, 512, 64, 1, 2}, {1024, 512, 64, 1, 2}};
    int64_t axis = 3;
    int64_t blockSize = 32;
    string expectTilingData =
        "64 64 0 0 0 0 0 0 4 137438953513 32 1 0 0 1 33554432 1 67108864 2141 152 1 3450 1 3403 1 63 0 32 128 0 0 0 0 ";

    ExecuteTestCase(ge::DT_BF16, ge::DT_FLOAT4_E1M2, shape, scaleShape, axis, blockSize, expectTilingData);
}

TEST_F(DynamicMxQuantTiling, DynamicMxQuant_tiling_ascendc_float16_fp4e2m1_tail_axis)
{
    gert::StorageShape shape = {{60, 14, 16, 128}, {60, 14, 16, 128}};
    gert::StorageShape scaleShape = {{60, 14, 16, 1, 2}, {60, 14, 16, 1, 2}};
    int64_t axis = -1;
    int64_t blockSize = 64;
    string expectTilingData =
        "64 41 2 2 0 82 330 150 4 274877906984 64 2 64 0 1 13440 1 26880 1000 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";

    ExecuteTestCase(ge::DT_FLOAT16, ge::DT_FLOAT4_E2M1, shape, scaleShape, axis, blockSize, expectTilingData);
}

TEST_F(DynamicMxQuantTiling, DynamicMxQuant_tiling_ascendc_float16_fp4e2m1_not_tail_axis)
{
    gert::StorageShape shape = {{60, 14, 64, 128}, {60, 14, 64, 128}};
    gert::StorageShape scaleShape = {{60, 14, 1, 128, 2}, {60, 14, 1, 128, 2}};
    int64_t axis = -2;
    int64_t blockSize = 64;
    string expectTilingData =
        "64 60 7 7 0 420 2 2 4 274877906984 64 1 64 0 0 840 128 215040 1011 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";

    ExecuteTestCase(ge::DT_FLOAT16, ge::DT_FLOAT4_E2M1, shape, scaleShape, axis, blockSize, expectTilingData);
}

TEST_F(DynamicMxQuantTiling, DynamicMxQuant_tiling_ascendc_float16_fp4e1m2_not_tail_axis)
{
    gert::StorageShape shape = {{1024, 512, 64, 32}, {1024, 512, 64, 32}};
    gert::StorageShape scaleShape = {{4, 512, 64, 32, 2}, {4, 512, 64, 32, 2}};
    int64_t axis = 0;
    int64_t blockSize = 128;
    string expectTilingData =
        "64 64 800 752 2 6394 164 124 4 549755813929 128 8 128 0 0 1 1048576 8388608 1111 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";

    ExecuteTestCase(ge::DT_FLOAT16, ge::DT_FLOAT4_E1M2, shape, scaleShape, axis, blockSize, expectTilingData);
}

TEST_F(DynamicMxQuantTiling, DynamicMxQuant_tiling_ascendc_float16_fp4e1m2_tail_axis)
{
    gert::StorageShape shape = {{1024, 512, 64, 128}, {1024, 512, 64, 128}};
    gert::StorageShape scaleShape = {{1024, 512, 64, 1, 2}, {1024, 512, 64, 1, 2}};
    int64_t axis = 3;
    int64_t blockSize = 64;
    string expectTilingData =
        "64 64 3178 3147 0 203361 330 64 4 274877906985 64 2 64 0 1 33554432 1 67108864 1100 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 0 ";

    ExecuteTestCase(ge::DT_FLOAT16, ge::DT_FLOAT4_E1M2, shape, scaleShape, axis, blockSize, expectTilingData);
}

TEST_F(DynamicMxQuantTiling, DynamicMxQuant_tiling_ascendc_bfloat16_fp8e4m3fn_not_tail_axis_true)
{
    gert::StorageShape shape = {{60, 14, 16, 128}, {60, 14, 16, 128}};
    gert::StorageShape scaleShape = {{60, 14, 16, 1, 2}, {60, 14, 16, 1, 2}};
    int64_t axis = -1;
    int64_t blockSize = 64;
    string expectTilingData =
        "64 41 2 2 0 82 330 150 4 274877906980 64 2 64 0 1 13440 1 26880 2200 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";

    ExecuteTestCase(ge::DT_BF16, ge::DT_FLOAT8_E4M3FN, shape, scaleShape, axis, blockSize, expectTilingData);
}

TEST_F(DynamicMxQuantTiling, DynamicMxQuant_tiling_ascendc_bfloat16_fp8e4m3fn_not_tail_axis_false)
{
    gert::StorageShape shape = {{60, 14, 32, 128}, {60, 14, 32, 128}};
    gert::StorageShape scaleShape = {{60, 14, 1, 128, 2}, {60, 14, 1, 128, 2}};
    int64_t axis = -2;
    int64_t blockSize = 32;
    string expectTilingData =
        "64 56 3 3 0 168 5 5 4 137438953508 32 1 32 0 0 840 128 215040 2211 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";

    ExecuteTestCase(ge::DT_BF16, ge::DT_FLOAT8_E4M3FN, shape, scaleShape, axis, blockSize, expectTilingData);
}

TEST_F(DynamicMxQuantTiling, DynamicMxQuant_tiling_ascendc_bfloat16_fp8e4m3fn_not_tail_axis_optimized)
{
    gert::StorageShape shape = {{1024, 512, 64, 32}, {1024, 512, 64, 32}};
    gert::StorageShape scaleShape = {{1024, 512, 1, 32, 2}, {1024, 512, 1, 32, 2}};
    int64_t axis = 2;
    int64_t blockSize = 32;
    string expectTilingData =
        "64 64 820 769 0 52429 20 16 4 137438953508 32 2 32 0 0 524288 32 33554432 2221 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";

    ExecuteTestCase(ge::DT_BF16, ge::DT_FLOAT8_E4M3FN, shape, scaleShape, axis, blockSize, expectTilingData);
}

TEST_F(DynamicMxQuantTiling, DynamicMxQuant_tiling_ascendc_bfloat16_fp8e4m3fn_tail_axis)
{
    gert::StorageShape shape = {{60, 14, 16, 128}, {60, 14, 16, 128}};
    gert::StorageShape scaleShape = {{60, 14, 16, 2, 2}, {60, 14, 16, 2, 2}};
    int64_t axis = -1;
    int64_t blockSize = 32;
    string expectTilingData =
        "64 58 0 0 0 0 0 0 4 137438953508 32 4 0 0 1 13440 1 53760 2240 116 4 2 1 2 4 57 0 128 100 0 0 0 0 ";

    ExecuteTestCase(ge::DT_BF16, ge::DT_FLOAT8_E4M3FN, shape, scaleShape, axis, blockSize, expectTilingData);
}

TEST_F(DynamicMxQuantTiling, DynamicMxQuant_tiling_ascendc_bfloat16_fp8e5m2_not_tail_axis_true)
{
    gert::StorageShape shape = {{15, 60, 16, 128}, {15, 60, 16, 128}};
    gert::StorageShape scaleShape = {{15, 60, 16, 1, 2}, {15, 60, 16, 1, 2}};
    int64_t axis = -1;
    int64_t blockSize = 64;
    string expectTilingData =
        "64 44 2 2 0 88 330 90 4 274877906979 64 2 64 0 1 14400 1 28800 2300 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";

    ExecuteTestCase(ge::DT_BF16, ge::DT_FLOAT8_E5M2, shape, scaleShape, axis, blockSize, expectTilingData);
}

TEST_F(DynamicMxQuantTiling, DynamicMxQuant_tiling_ascendc_bfloat16_fp8e5m2_not_tail_axis_false)
{
    gert::StorageShape shape = {{15, 60, 32, 128}, {15, 60, 32, 128}};
    gert::StorageShape scaleShape = {{15, 60, 1, 128, 2}, {15, 60, 1, 128, 2}};
    int64_t axis = -2;
    int64_t blockSize = 32;
    string expectTilingData =
        "64 60 3 3 0 180 5 5 4 137438953507 32 1 32 0 0 900 128 230400 2311 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";

    ExecuteTestCase(ge::DT_BF16, ge::DT_FLOAT8_E5M2, shape, scaleShape, axis, blockSize, expectTilingData);
}

TEST_F(DynamicMxQuantTiling, DynamicMxQuant_tiling_ascendc_bfloat16_fp8e5m2_not_tail_axis_optimized)
{
    gert::StorageShape shape = {{1, 300, 64, 32}, {1, 300, 64, 32}};
    gert::StorageShape scaleShape = {{1, 300, 1, 32, 2}, {1, 300, 1, 32, 2}};
    int64_t axis = 2;
    int64_t blockSize = 32;
    string expectTilingData =
        "64 38 2 1 0 75 8 8 4 137438953507 32 2 32 0 0 300 32 19200 2321 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";

    ExecuteTestCase(ge::DT_BF16, ge::DT_FLOAT8_E5M2, shape, scaleShape, axis, blockSize, expectTilingData);
}

TEST_F(DynamicMxQuantTiling, DynamicMxQuant_tiling_ascendc_bfloat16_fp8e5m2_tail_axis)
{
    gert::StorageShape shape = {{15, 60, 16, 128}, {15, 60, 16, 128}};
    gert::StorageShape scaleShape = {{15, 60, 16, 2, 2}, {15, 60, 16, 2, 2}};
    int64_t axis = -1;
    int64_t blockSize = 32;
    string expectTilingData =
        "64 63 0 0 0 0 0 0 4 137438953507 32 4 0 0 1 14400 1 57600 2340 116 4 2 1 1 4 62 0 128 16 0 0 0 0 ";

    ExecuteTestCase(ge::DT_BF16, ge::DT_FLOAT8_E5M2, shape, scaleShape, axis, blockSize, expectTilingData);
}

TEST_F(DynamicMxQuantTiling, DynamicMxQuant_tiling_ascendc_error_inDtype)
{
    gert::StorageShape shape = {{1024, 512, 64, 128}, {1024, 512, 64, 128}};
    gert::StorageShape scaleShape = {{1024, 512, 64, 1, 2}, {1024, 512, 64, 1, 2}};
    int64_t axis = 3;
    int64_t blockSize = 64;
    string expectTilingData = "";

    ExecuteTestCase(
        ge::DT_FLOAT, ge::DT_FLOAT4_E1M2, shape, scaleShape, axis, blockSize, expectTilingData, ge::GRAPH_FAILED);
}

TEST_F(DynamicMxQuantTiling, DynamicMxQuant_tiling_ascendc_error_outDtype)
{
    gert::StorageShape shape = {{1024, 512, 64, 128}, {1024, 512, 64, 128}};
    gert::StorageShape scaleShape = {{1024, 512, 64, 1, 2}, {1024, 512, 64, 1, 2}};
    int64_t axis = 3;
    int64_t blockSize = 64;
    string expectTilingData = "";

    ExecuteTestCase(
        ge::DT_FLOAT16, ge::DT_FLOAT16, shape, scaleShape, axis, blockSize, expectTilingData, ge::GRAPH_FAILED);
}

TEST_F(DynamicMxQuantTiling, DynamicMxQuant_tiling_ascendc_error_outrank)
{
    gert::StorageShape shape = {{1024, 512, 64, 128}, {1024, 512, 64, 128}};
    gert::StorageShape scaleShape = {{1024, 512, 64, 2}, {1024, 512, 64, 2}};
    int64_t axis = 3;
    int64_t blockSize = 64;
    string expectTilingData = "";

    ExecuteTestCase(
        ge::DT_FLOAT16, ge::DT_FLOAT8_E5M2, shape, scaleShape, axis, blockSize, expectTilingData, ge::GRAPH_FAILED);
}
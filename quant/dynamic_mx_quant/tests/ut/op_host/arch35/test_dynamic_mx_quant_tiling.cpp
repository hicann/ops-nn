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
#include "../../../../op_host/arch35/dynamic_mx_quant_tiling_arch35.h"

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

template <typename T>
static string to_string(void* buf, size_t size)
{
    std::string result;
    const size_t metadata_size = 2 * sizeof(float);
    size_t data_size = size - metadata_size;                                  // 减去最后两个float数据类型
    const T* data = reinterpret_cast<const T*>(buf);
    size_t len = data_size / sizeof(T);
    for (size_t i = 0; i < len; i++) {
        result += std::to_string(data[i]);
        result += " ";
    }
    const float* meta_ptr = reinterpret_cast<const float*>(buf + data_size);
    float dstTypeMaxFloat = meta_ptr[0];
    float invDstTypeMaxFloat = meta_ptr[1];
    int32_t dstTypeMax = static_cast<int32_t>(dstTypeMaxFloat);
    int32_t invDstTypeMax = static_cast<int32_t>(invDstTypeMaxFloat);
    result += std::to_string(dstTypeMax) + " " + std::to_string(invDstTypeMax);
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
    map<string, string> soc_versions = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};

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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_versions);

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
                           {"scale_alg", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                           {"dst_type_max", Ops::NN::AnyValue::CreateFrom<float>(0.0)}})
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

    auto raw_tiling_data = tiling_context->GetRawTilingData();
    auto tiling_data_result = to_string<int64_t>(raw_tiling_data->GetData(), raw_tiling_data->GetDataSize());
    EXPECT_EQ(tiling_data_result, expectTilingData);
}

TEST_F(DynamicMxQuantTiling, DynamicMxQuant_tiling_ascendc_bfloat16_fp4e2m1_tail_axis)
{
    gert::StorageShape shape = {{60, 14, 16, 128}, {60, 14, 16, 128}};
    gert::StorageShape scaleShape = {{60, 14, 16, 2, 2}, {60, 14, 16, 2, 2}};
    int64_t axis = -1;
    int64_t blockSize = 32;
    string expectTilingData = "10 253952 4 32 64 64 64 1 13440 128 2 128 210 210 1528 0 0";

    ExecuteTestCase(ge::DT_BF16, ge::DT_FLOAT4_E2M1, shape, scaleShape, axis, blockSize, expectTilingData);
}

TEST_F(DynamicMxQuantTiling, DynamicMxQuant_tiling_ascendc_bfloat16_fp4e2m1_not_tail_axis)
{
    gert::StorageShape shape = {{60, 14, 32, 128}, {60, 14, 32, 128}};
    gert::StorageShape scaleShape = {{60, 14, 1, 128, 2}, {60, 14, 1, 128, 2}};
    int64_t axis = -2;
    int64_t blockSize = 32;
    string expectTilingData =
        "64 60 32 0 32 10040 32 128 128 1 1 1 1 840 5 1680 10 168 0 0 0 0 0 0 0 0 0 0 0";

    ExecuteTestCase(ge::DT_BF16, ge::DT_FLOAT4_E2M1, shape, scaleShape, axis, blockSize, expectTilingData);
}

TEST_F(DynamicMxQuantTiling, DynamicMxQuant_tiling_ascendc_bfloat16_fp4e1m2_not_tail_axis)
{
    gert::StorageShape shape = {{1024, 512, 64, 32}, {1024, 512, 64, 32}};
    gert::StorageShape scaleShape = {{1024, 512, 1, 32, 2}, {1024, 512, 1, 32, 2}};
    int64_t axis = 2;
    int64_t blockSize = 32;
    string expectTilingData =
        "64 64 820 769 0 52429 20 16 4 41 32 0 2 32 0 0 524288 32 33554432 121 0 0 0 0";

    ExecuteTestCase(ge::DT_BF16, ge::DT_FLOAT4_E1M2, shape, scaleShape, axis, blockSize, expectTilingData);
}

TEST_F(DynamicMxQuantTiling, DynamicMxQuant_tiling_ascendc_bfloat16_fp4e1m2_tail_axis)
{
    gert::StorageShape shape = {{1024, 512, 64, 32}, {1024, 512, 64, 32}};
    gert::StorageShape scaleShape = {{1024, 512, 64, 1, 2}, {1024, 512, 64, 1, 2}};
    int64_t axis = 3;
    int64_t blockSize = 32;
    string expectTilingData =
        "10 253952 4 32 64 64 64 1 33554432 32 2 32 524288 524288 1528 0 0";

    ExecuteTestCase(ge::DT_BF16, ge::DT_FLOAT4_E1M2, shape, scaleShape, axis, blockSize, expectTilingData);
}

TEST_F(DynamicMxQuantTiling, DynamicMxQuant_tiling_ascendc_float16_fp4e2m1_tail_axis)
{
    gert::StorageShape shape = {{60, 14, 16, 128}, {60, 14, 16, 128}};
    gert::StorageShape scaleShape = {{60, 14, 16, 1, 2}, {60, 14, 16, 1, 2}};
    int64_t axis = -1;
    int64_t blockSize = 64;
    string expectTilingData =
        "64 41 2 2 0 82 328 312 4 40 64 0 2 64 0 1 13440 1 26880 100 0 0 0 0";

    ExecuteTestCase(ge::DT_FLOAT16, ge::DT_FLOAT4_E2M1, shape, scaleShape, axis, blockSize, expectTilingData);
}

TEST_F(DynamicMxQuantTiling, DynamicMxQuant_tiling_ascendc_float16_fp4e2m1_not_tail_axis)
{
    gert::StorageShape shape = {{60, 14, 64, 128}, {60, 14, 64, 128}};
    gert::StorageShape scaleShape = {{60, 14, 1, 128, 2}, {60, 14, 1, 128, 2}};
    int64_t axis = -2;
    int64_t blockSize = 64;
    string expectTilingData =
        "64 60 7 7 0 420 2 2 4 40 64 0 1 64 0 0 840 128 215040 111 0 0 0 0";

    ExecuteTestCase(ge::DT_FLOAT16, ge::DT_FLOAT4_E2M1, shape, scaleShape, axis, blockSize, expectTilingData);
}

TEST_F(DynamicMxQuantTiling, DynamicMxQuant_tiling_ascendc_float16_fp4e1m2_not_tail_axis)
{
    gert::StorageShape shape = {{1024, 512, 64, 32}, {1024, 512, 64, 32}};
    gert::StorageShape scaleShape = {{4, 512, 64, 32, 2}, {4, 512, 64, 32, 2}};
    int64_t axis = 0;
    int64_t blockSize = 128;
    string expectTilingData =
        "64 64 128 0 128 20040 1024 1048576 1048576 4 8192 4 0 0 1 0 0 32768 512 512 0 128 128 256 256 32768 8 0 0";

    ExecuteTestCase(ge::DT_FLOAT16, ge::DT_FLOAT4_E1M2, shape, scaleShape, axis, blockSize, expectTilingData);
}

TEST_F(DynamicMxQuantTiling, DynamicMxQuant_tiling_ascendc_float16_fp4e1m2_tail_axis)
{
    gert::StorageShape shape = {{1024, 512, 64, 128}, {1024, 512, 64, 128}};
    gert::StorageShape scaleShape = {{1024, 512, 64, 1, 2}, {1024, 512, 64, 1, 2}};
    int64_t axis = 3;
    int64_t blockSize = 64;
    string expectTilingData =
        "64 64 3197 3190 0 204601 328 64 4 41 64 0 2 64 0 1 33554432 1 67108864 100 0 0 0 0";

    ExecuteTestCase(ge::DT_FLOAT16, ge::DT_FLOAT4_E1M2, shape, scaleShape, axis, blockSize, expectTilingData);
}

TEST_F(DynamicMxQuantTiling, DynamicMxQuant_tiling_ascendc_bfloat16_fp8e4m3fn_not_tail_axis_true)
{
    gert::StorageShape shape = {{60, 14, 16, 128}, {60, 14, 16, 128}};
    gert::StorageShape scaleShape = {{60, 14, 16, 1, 2}, {60, 14, 16, 1, 2}};
    int64_t axis = -1;
    int64_t blockSize = 64;
    string expectTilingData =
        "64 41 2 2 0 82 328 312 4 36 64 0 2 64 0 1 13440 1 26880 100 0 0 0 0";

    ExecuteTestCase(ge::DT_BF16, ge::DT_FLOAT8_E4M3FN, shape, scaleShape, axis, blockSize, expectTilingData);
}

TEST_F(DynamicMxQuantTiling, DynamicMxQuant_tiling_ascendc_bfloat16_fp8e4m3fn_not_tail_axis_false)
{
    gert::StorageShape shape = {{60, 14, 32, 128}, {60, 14, 32, 128}};
    gert::StorageShape scaleShape = {{60, 14, 1, 128, 2}, {60, 14, 1, 128, 2}};
    int64_t axis = -2;
    int64_t blockSize = 32;
    string expectTilingData =
        "64 60 32 0 32 10040 32 128 128 1 1 1 1 840 5 1680 10 168 0 0 0 0 0 0 0 0 0 0 0";

    ExecuteTestCase(ge::DT_BF16, ge::DT_FLOAT8_E4M3FN, shape, scaleShape, axis, blockSize, expectTilingData);
}

TEST_F(DynamicMxQuantTiling, DynamicMxQuant_tiling_ascendc_bfloat16_fp8e4m3fn_not_tail_axis_optimized)
{
    gert::StorageShape shape = {{1024, 512, 64, 32}, {1024, 512, 64, 32}};
    gert::StorageShape scaleShape = {{1024, 512, 1, 32, 2}, {1024, 512, 1, 32, 2}};
    int64_t axis = 2;
    int64_t blockSize = 32;
    string expectTilingData =
        "64 64 820 769 0 52429 20 16 4 36 32 0 2 32 0 0 524288 32 33554432 121 0 0 0 0";

    ExecuteTestCase(ge::DT_BF16, ge::DT_FLOAT8_E4M3FN, shape, scaleShape, axis, blockSize, expectTilingData);
}

TEST_F(DynamicMxQuantTiling, DynamicMxQuant_tiling_ascendc_bfloat16_fp8e4m3fn_tail_axis)
{
    gert::StorageShape shape = {{60, 14, 16, 128}, {60, 14, 16, 128}};
    gert::StorageShape scaleShape = {{60, 14, 16, 2, 2}, {60, 14, 16, 2, 2}};
    int64_t axis = -1;
    int64_t blockSize = 32;
    string expectTilingData =
        "10 253952 4 32 64 64 64 1 13440 128 2 128 210 210 1280 0 0";

    ExecuteTestCase(ge::DT_BF16, ge::DT_FLOAT8_E4M3FN, shape, scaleShape, axis, blockSize, expectTilingData);
}

TEST_F(DynamicMxQuantTiling, DynamicMxQuant_tiling_ascendc_bfloat16_fp8e5m2_not_tail_axis_true)
{
    gert::StorageShape shape = {{15, 60, 16, 128}, {15, 60, 16, 128}};
    gert::StorageShape scaleShape = {{15, 60, 16, 1, 2}, {15, 60, 16, 1, 2}};
    int64_t axis = -1;
    int64_t blockSize = 64;
    string expectTilingData =
        "64 44 2 2 0 88 328 264 4 35 64 0 2 64 0 1 14400 1 28800 100 0 0 0 0";

    ExecuteTestCase(ge::DT_BF16, ge::DT_FLOAT8_E5M2, shape, scaleShape, axis, blockSize, expectTilingData);
}

TEST_F(DynamicMxQuantTiling, DynamicMxQuant_tiling_ascendc_bfloat16_fp8e5m2_not_tail_axis_false)
{
    gert::StorageShape shape = {{15, 60, 32, 128}, {15, 60, 32, 128}};
    gert::StorageShape scaleShape = {{15, 60, 1, 128, 2}, {15, 60, 1, 128, 2}};
    int64_t axis = -2;
    int64_t blockSize = 32;
    string expectTilingData =
        "64 60 32 0 32 10040 32 128 128 1 1 1 1 900 5 1800 10 180 0 0 0 0 0 0 0 0 0 0 0";

    ExecuteTestCase(ge::DT_BF16, ge::DT_FLOAT8_E5M2, shape, scaleShape, axis, blockSize, expectTilingData);
}

TEST_F(DynamicMxQuantTiling, DynamicMxQuant_tiling_ascendc_bfloat16_fp8e5m2_not_tail_axis_optimized)
{
    gert::StorageShape shape = {{1, 300, 64, 32}, {1, 300, 64, 32}};
    gert::StorageShape scaleShape = {{1, 300, 1, 32, 2}, {1, 300, 1, 32, 2}};
    int64_t axis = 2;
    int64_t blockSize = 32;
    string expectTilingData =
        "64 38 2 1 0 75 8 8 4 35 32 0 2 32 0 0 300 32 19200 121 0 0 0 0";

    ExecuteTestCase(ge::DT_BF16, ge::DT_FLOAT8_E5M2, shape, scaleShape, axis, blockSize, expectTilingData);
}

TEST_F(DynamicMxQuantTiling, DynamicMxQuant_tiling_ascendc_bfloat16_fp8e5m2_tail_axis)
{
    gert::StorageShape shape = {{15, 60, 16, 128}, {15, 60, 16, 128}};
    gert::StorageShape scaleShape = {{15, 60, 16, 2, 2}, {15, 60, 16, 2, 2}};
    int64_t axis = -1;
    int64_t blockSize = 32;
    string expectTilingData =
        "10 253952 4 32 64 64 64 1 14400 128 2 128 225 225 1280 0 0";

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

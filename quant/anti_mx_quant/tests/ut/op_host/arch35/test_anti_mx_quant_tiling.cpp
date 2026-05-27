/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
#include "../../../../op_host/arch35/anti_mx_quant_tiling_arch35.h"

using namespace ut_util;
using namespace std;
using namespace ge;

class AntiMxQuantTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AntiMxQuantTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AntiMxQuantTiling TearDown" << std::endl;
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
    ge::DataType inDtype, ge::DataType outDtype, gert::StorageShape shape, gert::StorageShape scaleShape,
    int64_t axis, int64_t dstType, string expectTilingData, ge::graphStatus status = ge::GRAPH_SUCCESS)
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
    optiling::AntiMxQuantCompileInfo compile_info;

    std::string op_type("AntiMxQuant");
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
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1})
                      .InputShapes({&shape, &scaleShape})
                      .OutputShapes({&shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, inDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, outDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"axis", Ops::NN::AnyValue::CreateFrom(axis)},
                           {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(dstType)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), status);
    if (status == ge::GRAPH_FAILED) {
        return;
    }
    // check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    auto block_dim = tiling_context->GetBlockDim();

    auto raw_tiling_data = tiling_context->GetRawTilingData();
    auto tiling_data_result = to_string<int64_t>(raw_tiling_data->GetData(), raw_tiling_data->GetDataSize());
    if (!expectTilingData.empty()) {
        EXPECT_EQ(tiling_data_result, expectTilingData);
    }
}

// Test case 1: FP8_E5M2 -> FP16, tail axis
TEST_F(AntiMxQuantTiling, AntiMxQuant_tiling_fp8e5m2_to_fp16_tail_axis)
{
    gert::StorageShape shape = {{60, 14, 16, 128}, {60, 14, 16, 128}};
    gert::StorageShape scaleShape = {{60, 14, 16, 2, 2}, {60, 14, 16, 2, 2}};
    int64_t axis = -1;
    int64_t dstType = 1;  // FP16
    string expectTilingData = "253952 1 64 64 64 1 13440 128 1 128 210 210 1280 ";

    ExecuteTestCase(ge::DT_FLOAT8_E5M2, ge::DT_FLOAT16, shape, scaleShape, axis, dstType, expectTilingData);
}

// Test case 2: FP8_E4M3FN -> BF16, tail axis
TEST_F(AntiMxQuantTiling, AntiMxQuant_tiling_fp8e4m3fn_to_bf16_tail_axis)
{
    gert::StorageShape shape = {{60, 14, 16, 128}, {60, 14, 16, 128}};
    gert::StorageShape scaleShape = {{60, 14, 16, 2, 2}, {60, 14, 16, 2, 2}};
    int64_t axis = -1;
    int64_t dstType = 27;  // BF16
    string expectTilingData = "253952 27 64 64 64 1 13440 128 1 128 210 210 1280 ";

    ExecuteTestCase(ge::DT_FLOAT8_E4M3FN, ge::DT_BF16, shape, scaleShape, axis, dstType, expectTilingData);
}

// Test case 3: FP4_E2M1 -> FP32, tail axis
TEST_F(AntiMxQuantTiling, AntiMxQuant_tiling_fp4e2m1_to_fp32_tail_axis)
{
    gert::StorageShape shape = {{60, 14, 16, 128}, {60, 14, 16, 128}};
    gert::StorageShape scaleShape = {{60, 14, 16, 2, 2}, {60, 14, 16, 2, 2}};
    int64_t axis = -1;
    int64_t dstType = 0;  // FP32
    string expectTilingData = "253952 0 64 64 64 1 13440 128 1 128 210 210 864 ";

    ExecuteTestCase(ge::DT_FLOAT4_E2M1, ge::DT_FLOAT, shape, scaleShape, axis, dstType, expectTilingData);
}

// Test case 4: FP4_E1M2 -> FP16, tail axis
TEST_F(AntiMxQuantTiling, AntiMxQuant_tiling_fp4e1m2_to_fp16_tail_axis)
{
    gert::StorageShape shape = {{1024, 512, 64, 128}, {1024, 512, 64, 128}};
    gert::StorageShape scaleShape = {{1024, 512, 64, 2, 2}, {1024, 512, 64, 2, 2}};
    int64_t axis = -1;
    int64_t dstType = 1;  // FP16
    string expectTilingData = "253952 1 64 64 64 1 33554432 128 1 128 524288 524288 1536 ";

    ExecuteTestCase(ge::DT_FLOAT4_E1M2, ge::DT_FLOAT16, shape, scaleShape, axis, dstType, expectTilingData);
}

// Error case 1: invalid input dtype
TEST_F(AntiMxQuantTiling, AntiMxQuant_tiling_error_inDtype)
{
    gert::StorageShape shape = {{60, 14, 16, 128}, {60, 14, 16, 128}};
    gert::StorageShape scaleShape = {{60, 14, 16, 2, 2}, {60, 14, 16, 2, 2}};
    int64_t axis = -1;
    int64_t dstType = 1;
    string expectTilingData = "253952 1 64 64 64 1 13440 128 1 128 210 210 1280 ";

    ExecuteTestCase(
        ge::DT_FLOAT16, ge::DT_FLOAT16, shape, scaleShape, axis, dstType, expectTilingData, ge::GRAPH_FAILED);
}

// Error case 2: invalid output dtype
TEST_F(AntiMxQuantTiling, AntiMxQuant_tiling_error_outDtype)
{
    gert::StorageShape shape = {{60, 14, 16, 128}, {60, 14, 16, 128}};
    gert::StorageShape scaleShape = {{60, 14, 16, 2, 2}, {60, 14, 16, 2, 2}};
    int64_t axis = -1;
    int64_t dstType = 1;
    string expectTilingData = "253952 1 64 64 64 1 13440 128 1 128 210 210 1280 ";

    ExecuteTestCase(
        ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E5M2, shape, scaleShape, axis, dstType, expectTilingData, ge::GRAPH_FAILED);
}

// Error case 3: invalid mxscale shape
TEST_F(AntiMxQuantTiling, AntiMxQuant_tiling_error_mxscale_shape)
{
    gert::StorageShape shape = {{60, 14, 16, 128}, {60, 14, 16, 128}};
    // Wrong mxscale shape: should be {60, 14, 16, 2, 2}
    gert::StorageShape scaleShape = {{60, 14, 16, 1, 2}, {60, 14, 16, 1, 2}};
    int64_t axis = -1;
    int64_t dstType = 1;
    string expectTilingData = "253952 1 64 64 64 1 13440 128 1 128 210 210 1280 ";

    ExecuteTestCase(
        ge::DT_FLOAT8_E5M2, ge::DT_FLOAT16, shape, scaleShape, axis, dstType, expectTilingData, ge::GRAPH_FAILED);
}

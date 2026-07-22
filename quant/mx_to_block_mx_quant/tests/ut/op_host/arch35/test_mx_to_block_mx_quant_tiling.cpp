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
#include "../../../../op_host/arch35/mx_to_block_mx_quant_tiling_arch35.h"

using namespace ut_util;
using namespace std;
using namespace ge;

class MxToBlockMxQuantTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "MxToBlockMxQuantTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "MxToBlockMxQuantTiling TearDown" << std::endl; }
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

static void ExecuteTestCase(ge::DataType inDtype, ge::DataType outDtype, gert::StorageShape xShape,
                            gert::StorageShape mxScaleShape, gert::StorageShape yShape, gert::StorageShape scale1Shape,
                            gert::StorageShape scale2Shape, int64_t dstType, string expectTilingData,
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
    map<string, string> soc_versions = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};

    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();

    // compile info
    optiling::MxToBlockMxQuantCompileInfo compile_info;

    std::string op_type("MxToBlockMxQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 3)
                             .Inputs({const_cast<char*>("{}"), reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_versions);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(2, 3)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&xShape, &mxScaleShape})
                      .OutputShapes({&yShape, &scale1Shape, &scale2Shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, inDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, outDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(dstType)}})
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

// Test case 1: FP4_E2M1 -> FP8_E5M2, row aligned to 64
TEST_F(MxToBlockMxQuantTiling, MxToBlockMxQuant_tiling_fp4e2m1_to_fp8e5m2_row_aligned)
{
    gert::StorageShape xShape = {{64, 64}, {64, 64}};
    gert::StorageShape mxScaleShape = {{64, 1, 2}, {64, 1, 2}};
    gert::StorageShape yShape = {{64, 64}, {64, 64}};
    gert::StorageShape scale1Shape = {{64, 1, 2}, {64, 1, 2}};
    gert::StorageShape scale2Shape = {{1, 64, 2}, {1, 64, 2}};
    int64_t dstType = 35; // FLOAT8_E5M2
    string expectTilingData = "253952 35 64 1 1 64 64 2 0 1 1 64 64 1 1 0 1 0 ";

    ExecuteTestCase(ge::DT_FLOAT4_E2M1, ge::DT_FLOAT8_E5M2, xShape, mxScaleShape, yShape, scale1Shape, scale2Shape,
                    dstType, expectTilingData);
}

// Test case 2: FP4_E2M1 -> FP8_E4M3FN, row not aligned to 64
TEST_F(MxToBlockMxQuantTiling, MxToBlockMxQuant_tiling_fp4e2m1_to_fp8e4m3fn_row_not_aligned)
{
    gert::StorageShape xShape = {{128, 512}, {128, 512}};
    gert::StorageShape mxScaleShape = {{128, 8, 2}, {128, 8, 2}};
    gert::StorageShape yShape = {{128, 512}, {128, 512}};
    gert::StorageShape scale1Shape = {{128, 8, 2}, {128, 8, 2}};
    gert::StorageShape scale2Shape = {{2, 512, 2}, {2, 512, 2}};
    int64_t dstType = 36; // FLOAT8_E4M3FN
    string expectTilingData = "253952 36 64 2 1 128 512 16 0 2 1 64 512 2 1 0 2 0 ";

    ExecuteTestCase(ge::DT_FLOAT4_E2M1, ge::DT_FLOAT8_E4M3FN, xShape, mxScaleShape, yShape, scale1Shape, scale2Shape,
                    dstType, expectTilingData);
}

// Test case 3: FP4_E1M2 -> FP8_E5M2, large shape multi-core
TEST_F(MxToBlockMxQuantTiling, MxToBlockMxQuant_tiling_fp4e1m2_to_fp8e5m2_large)
{
    // x: [1024, 2048], large enough to use many cores
    gert::StorageShape xShape = {{1024, 2048}, {1024, 2048}};
    gert::StorageShape mxScaleShape = {{1024, 32, 2}, {1024, 32, 2}};
    gert::StorageShape yShape = {{1024, 2048}, {1024, 2048}};
    gert::StorageShape scale1Shape = {{1024, 32, 2}, {1024, 32, 2}};
    gert::StorageShape scale2Shape = {{16, 2048, 2}, {16, 2048, 2}};
    int64_t dstType = 35; // FLOAT8_E5M2
    string expectTilingData = "253952 35 64 64 1 1024 2048 64 0 16 4 64 512 64 1 0 64 0 ";

    ExecuteTestCase(ge::DT_FLOAT4_E1M2, ge::DT_FLOAT8_E5M2, xShape, mxScaleShape, yShape, scale1Shape, scale2Shape,
                    dstType, expectTilingData);
}

// Error case 1: invalid dstType (not 35 or 36)
TEST_F(MxToBlockMxQuantTiling, MxToBlockMxQuant_tiling_error_invalid_dstType)
{
    gert::StorageShape xShape = {{64, 64}, {64, 64}};
    gert::StorageShape mxScaleShape = {{64, 1, 2}, {64, 1, 2}};
    gert::StorageShape yShape = {{64, 64}, {64, 64}};
    gert::StorageShape scale1Shape = {{64, 1, 2}, {64, 1, 2}};
    gert::StorageShape scale2Shape = {{1, 64, 2}, {1, 64, 2}};
    int64_t dstType = 1; // FP16, not supported

    ExecuteTestCase(ge::DT_FLOAT4_E2M1, ge::DT_FLOAT8_E5M2, xShape, mxScaleShape, yShape, scale1Shape, scale2Shape,
                    dstType, "", ge::GRAPH_FAILED);
}

// Error case 2: invalid input dtype
TEST_F(MxToBlockMxQuantTiling, MxToBlockMxQuant_tiling_error_invalid_input_dtype)
{
    gert::StorageShape xShape = {{64, 64}, {64, 64}};
    gert::StorageShape mxScaleShape = {{64, 1, 2}, {64, 1, 2}};
    gert::StorageShape yShape = {{64, 64}, {64, 64}};
    gert::StorageShape scale1Shape = {{64, 1, 2}, {64, 1, 2}};
    gert::StorageShape scale2Shape = {{1, 64, 2}, {1, 64, 2}};
    int64_t dstType = 35;

    ExecuteTestCase(ge::DT_FLOAT16, ge::DT_FLOAT8_E5M2, xShape, mxScaleShape, yShape, scale1Shape, scale2Shape, dstType,
                    "", ge::GRAPH_FAILED);
}

// Error case 3: invalid mxscale shape
TEST_F(MxToBlockMxQuantTiling, MxToBlockMxQuant_tiling_error_invalid_mxscale_shape)
{
    gert::StorageShape xShape = {{64, 64}, {64, 64}};
    gert::StorageShape mxScaleShape = {{64, 3, 2}, {64, 3, 2}};
    gert::StorageShape yShape = {{64, 64}, {64, 64}};
    gert::StorageShape scale1Shape = {{64, 1, 2}, {64, 1, 2}};
    gert::StorageShape scale2Shape = {{1, 64, 2}, {1, 64, 2}};
    int64_t dstType = 35;

    ExecuteTestCase(ge::DT_FLOAT4_E2M1, ge::DT_FLOAT8_E5M2, xShape, mxScaleShape, yShape, scale1Shape, scale2Shape,
                    dstType, "", ge::GRAPH_FAILED);
}

// Error case 4: output y shape mismatch
TEST_F(MxToBlockMxQuantTiling, MxToBlockMxQuant_tiling_error_y_shape_mismatch)
{
    gert::StorageShape xShape = {{64, 64}, {64, 64}};
    gert::StorageShape mxScaleShape = {{64, 1, 2}, {64, 1, 2}};
    gert::StorageShape yShape = {{64, 128}, {64, 128}};
    gert::StorageShape scale1Shape = {{64, 1, 2}, {64, 1, 2}};
    gert::StorageShape scale2Shape = {{1, 64, 2}, {1, 64, 2}};
    int64_t dstType = 35;

    ExecuteTestCase(ge::DT_FLOAT4_E2M1, ge::DT_FLOAT8_E5M2, xShape, mxScaleShape, yShape, scale1Shape, scale2Shape,
                    dstType, "", ge::GRAPH_FAILED);
}

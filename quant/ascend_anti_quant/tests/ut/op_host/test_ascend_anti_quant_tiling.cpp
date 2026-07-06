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
#include "platform/platform_infos_def.h"
#include "ut_op_util.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"

using namespace std;
using namespace ge;
using namespace ut_util;

class AscendAntiQuantTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "AscendAntiQuantTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "AscendAntiQuantTiling TearDown" << std::endl; }
};

struct AscendAntiQuantCompileInfo {
    int32_t vectorCoreNum = 0;
    uint64_t ubSize = 0;
};

static void InitPlatForm(fe::PlatFormInfos& platform_info, map<string, string>& soc_infos,
                         map<string, string>& aicore_spec, map<string, string>& intrinsics)
{
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
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);
    platform_info.Init();
}

static void DoTilingCase(ge::DataType inDtype, ge::DataType outDtype, int64_t dstTypeAttr, bool sqrtMode,
                         const gert::StorageShape& xShape, ge::graphStatus expected)
{
    fe::PlatFormInfos platform_info;
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    InitPlatForm(platform_info, soc_infos, aicore_spec, intrinsics);
    map<string, string> socversions = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
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

    gert::StorageShape outShape = xShape;

    AscendAntiQuantCompileInfo compile_info;
    std::string op_type("AscendAntiQuant");
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

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(16 * 1024 * 1024);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1})
                      .InputShapes({const_cast<gert::StorageShape*>(&xShape)})
                      .OutputShapes({const_cast<gert::StorageShape*>(&outShape)})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, inDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, outDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"scale", Ops::NN::AnyValue::CreateFrom<float>(0.5f)},
                                  {"offset", Ops::NN::AnyValue::CreateFrom<float>(1.0f)},
                                  {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(dstTypeAttr)},
                                  {"sqrt_mode", Ops::NN::AnyValue::CreateFrom<bool>(sqrtMode)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    tiling_context->GetPlatformInfo()->SetPlatformRes("version", socversions);

    EXPECT_EQ(tiling_func(tiling_context), expected);
}

TEST_F(AscendAntiQuantTiling, ascend_anti_quant_tiling_int8_fp16)
{
    gert::StorageShape xShape = {{4, 2, 10240}, {4, 2, 10240}};
    DoTilingCase(ge::DT_INT8, ge::DT_FLOAT16, /*dstType=*/1, /*sqrt_mode=*/false, xShape, ge::GRAPH_SUCCESS);
}

TEST_F(AscendAntiQuantTiling, ascend_anti_quant_tiling_int8_fp32_sqrt)
{
    gert::StorageShape xShape = {{4, 2, 10240}, {4, 2, 10240}};
    DoTilingCase(ge::DT_INT8, ge::DT_FLOAT, /*dstType=*/0, /*sqrt_mode=*/true, xShape, ge::GRAPH_SUCCESS);
}

TEST_F(AscendAntiQuantTiling, ascend_anti_quant_tiling_hifloat8_fp16)
{
    gert::StorageShape xShape = {{4, 2, 10240}, {4, 2, 10240}};
    DoTilingCase(ge::DT_HIFLOAT8, ge::DT_FLOAT16, /*dstType=*/1, /*sqrt_mode=*/false, xShape, ge::GRAPH_SUCCESS);
}

TEST_F(AscendAntiQuantTiling, ascend_anti_quant_tiling_hifloat8_fp32)
{
    gert::StorageShape xShape = {{2048}, {2048}};
    DoTilingCase(ge::DT_HIFLOAT8, ge::DT_FLOAT, /*dstType=*/0, /*sqrt_mode=*/false, xShape, ge::GRAPH_SUCCESS);
}

TEST_F(AscendAntiQuantTiling, ascend_anti_quant_tiling_fp8_e5m2_fp16)
{
    gert::StorageShape xShape = {{8, 1024}, {8, 1024}};
    DoTilingCase(ge::DT_FLOAT8_E5M2, ge::DT_FLOAT16, /*dstType=*/1, /*sqrt_mode=*/false, xShape, ge::GRAPH_SUCCESS);
}

TEST_F(AscendAntiQuantTiling, ascend_anti_quant_tiling_fp8_e4m3_fp32)
{
    gert::StorageShape xShape = {{16, 512}, {16, 512}};
    DoTilingCase(ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT, /*dstType=*/0, /*sqrt_mode=*/false, xShape, ge::GRAPH_SUCCESS);
}

TEST_F(AscendAntiQuantTiling, ascend_anti_quant_tiling_small_shape)
{
    gert::StorageShape xShape = {{128}, {128}};
    DoTilingCase(ge::DT_INT8, ge::DT_FLOAT16, /*dstType=*/1, /*sqrt_mode=*/false, xShape, ge::GRAPH_SUCCESS);
}

TEST_F(AscendAntiQuantTiling, ascend_anti_quant_tiling_dtype_mismatch_fail)
{
    // attr dtype = FLOAT16(1) but output y dtype = FLOAT(0) -> should fail.
    gert::StorageShape xShape = {{4, 2, 1024}, {4, 2, 1024}};
    DoTilingCase(ge::DT_INT8, ge::DT_FLOAT, /*dstType=*/1, /*sqrt_mode=*/false, xShape, ge::GRAPH_FAILED);
}

TEST_F(AscendAntiQuantTiling, ascend_anti_quant_tiling_unsupported_input_dtype_fail)
{
    // Input dtype DT_FLOAT16 is not supported.
    gert::StorageShape xShape = {{4, 1024}, {4, 1024}};
    DoTilingCase(ge::DT_FLOAT16, ge::DT_FLOAT16, /*dstType=*/1, /*sqrt_mode=*/false, xShape, ge::GRAPH_FAILED);
}

TEST_F(AscendAntiQuantTiling, ascend_anti_quant_tiling_unsupported_output_dtype_fail)
{
    // Output dtype DT_BF16 is not supported.
    gert::StorageShape xShape = {{4, 1024}, {4, 1024}};
    DoTilingCase(ge::DT_INT8, ge::DT_BF16, /*dstType=*/27, /*sqrt_mode=*/false, xShape, ge::GRAPH_FAILED);
}

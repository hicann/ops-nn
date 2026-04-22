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
#include "log/log.h"
#include <gtest/gtest.h>
#include "register/op_impl_registry.h"
#include "platform/platform_infos_def.h"
#include "ut_op_common.h"
#include "ut_op_util.h"
#include "../../../../op_host/arch35/swiglu_mx_quant_with_dual_axis_tiling_arch35.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"

using namespace std;

class SwigluMxQuantWithDualAxisTilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "SwigluMxQuantWithDualAxisTilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "SwigluMxQuantWithDualAxisTilingTest TearDown" << std::endl;
    }
};

static void ExecuteTestCase(
    ge::DataType xDtype, ge::DataType y1Dtype, ge::DataType mxScale1Dtype,
    ge::DataType y2Dtype, ge::DataType mxScale2Dtype,
    gert::StorageShape xShape, gert::StorageShape y1Shape, gert::StorageShape mxScale1Shape,
    gert::StorageShape y2Shape, gert::StorageShape mxScale2Shape,
    bool activate_left, const string& round_mode, int64_t dst_type, int64_t scale_alg,
    ge::graphStatus status = ge::GRAPH_SUCCESS)
{
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

    optiling::SwigluMxQuantWithDualAxisCompileInfo compile_info;

    std::string op_type("SwigluMxQuantWithDualAxis");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", socversions);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(1, 4)
                      .IrInstanceNum({1})
                      .InputShapes({&xShape})
                      .OutputShapes({&y1Shape, &mxScale1Shape, &y2Shape, &mxScale2Shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, y1Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, mxScale1Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, y2Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, mxScale2Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"activate_left", Ops::NN::AnyValue::CreateFrom<bool>(activate_left)},
                           {"round_mode", Ops::NN::AnyValue::CreateFrom<string>(round_mode)},
                           {"scale_alg", Ops::NN::AnyValue::CreateFrom<int64_t>(scale_alg)},
                           {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(dst_type)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context, nullptr);
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    tiling_context->GetPlatformInfo()->SetPlatformRes("version", socversions);

    EXPECT_EQ(tiling_func(tiling_context), status);
}

TEST_F(SwigluMxQuantWithDualAxisTilingTest, test_tiling_fp16_to_fp8_e4m3fn)
{
    ge::DataType xDtype = ge::DT_FLOAT16;
    ge::DataType y1Dtype = ge::DT_FLOAT8_E4M3FN;
    ge::DataType mxScale1Dtype = ge::DT_FLOAT8_E8M0;
    ge::DataType y2Dtype = ge::DT_FLOAT8_E4M3FN;
    ge::DataType mxScale2Dtype = ge::DT_FLOAT8_E8M0;
    gert::StorageShape xShape = {{8, 8192}, {8, 8192}};
    gert::StorageShape y1Shape = {{8, 4096}, {8, 4096}};
    gert::StorageShape mxScale1Shape = {{8, 64, 2}, {8, 64, 2}};
    gert::StorageShape y2Shape = {{8, 4096}, {8, 4096}};
    gert::StorageShape mxScale2Shape = {{1, 4096, 2}, {1, 4096, 2}};
    ExecuteTestCase(
        xDtype, y1Dtype, mxScale1Dtype, y2Dtype, mxScale2Dtype,
        xShape, y1Shape, mxScale1Shape, y2Shape, mxScale2Shape,
        false, "rint", 36, 0,
        ge::GRAPH_SUCCESS);
}

TEST_F(SwigluMxQuantWithDualAxisTilingTest, test_tiling_bf16_to_fp8_e5m2)
{
    ge::DataType xDtype = ge::DT_BF16;
    ge::DataType y1Dtype = ge::DT_FLOAT8_E5M2;
    ge::DataType mxScale1Dtype = ge::DT_FLOAT8_E8M0;
    ge::DataType y2Dtype = ge::DT_FLOAT8_E5M2;
    ge::DataType mxScale2Dtype = ge::DT_FLOAT8_E8M0;
    gert::StorageShape xShape = {{4, 2048}, {4, 2048}};
    gert::StorageShape y1Shape = {{4, 1024}, {4, 1024}};
    gert::StorageShape mxScale1Shape = {{4, 16, 2}, {4, 16, 2}};
    gert::StorageShape y2Shape = {{4, 1024}, {4, 1024}};
    gert::StorageShape mxScale2Shape = {{1, 1024, 2}, {1, 1024, 2}};
    ExecuteTestCase(
        xDtype, y1Dtype, mxScale1Dtype, y2Dtype, mxScale2Dtype,
        xShape, y1Shape, mxScale1Shape, y2Shape, mxScale2Shape,
        false, "rint", 35, 0,
        ge::GRAPH_SUCCESS);
}

TEST_F(SwigluMxQuantWithDualAxisTilingTest, test_tiling_fp16_to_fp4_e2m1)
{
    ge::DataType xDtype = ge::DT_FLOAT16;
    ge::DataType y1Dtype = ge::DT_FLOAT4_E2M1;
    ge::DataType mxScale1Dtype = ge::DT_FLOAT8_E8M0;
    ge::DataType y2Dtype = ge::DT_FLOAT4_E2M1;
    ge::DataType mxScale2Dtype = ge::DT_FLOAT8_E8M0;
    gert::StorageShape xShape = {{4, 4096}, {4, 4096}};
    gert::StorageShape y1Shape = {{4, 2048}, {4, 2048}};
    gert::StorageShape mxScale1Shape = {{4, 32, 2}, {4, 32, 2}};
    gert::StorageShape y2Shape = {{4, 2048}, {4, 2048}};
    gert::StorageShape mxScale2Shape = {{1, 2048, 2}, {1, 2048, 2}};
    ExecuteTestCase(
        xDtype, y1Dtype, mxScale1Dtype, y2Dtype, mxScale2Dtype,
        xShape, y1Shape, mxScale1Shape, y2Shape, mxScale2Shape,
        false, "rint", 40, 0,
        ge::GRAPH_SUCCESS);
}

TEST_F(SwigluMxQuantWithDualAxisTilingTest, test_tiling_bf16_to_fp4_e1m2)
{
    ge::DataType xDtype = ge::DT_BF16;
    ge::DataType y1Dtype = ge::DT_FLOAT4_E1M2;
    ge::DataType mxScale1Dtype = ge::DT_FLOAT8_E8M0;
    ge::DataType y2Dtype = ge::DT_FLOAT4_E1M2;
    ge::DataType mxScale2Dtype = ge::DT_FLOAT8_E8M0;
    gert::StorageShape xShape = {{2, 2048}, {2, 2048}};
    gert::StorageShape y1Shape = {{2, 1024}, {2, 1024}};
    gert::StorageShape mxScale1Shape = {{2, 16, 2}, {2, 16, 2}};
    gert::StorageShape y2Shape = {{2, 1024}, {2, 1024}};
    gert::StorageShape mxScale2Shape = {{1, 1024, 2}, {1, 1024, 2}};
    ExecuteTestCase(
        xDtype, y1Dtype, mxScale1Dtype, y2Dtype, mxScale2Dtype,
        xShape, y1Shape, mxScale1Shape, y2Shape, mxScale2Shape,
        false, "floor", 41, 0,
        ge::GRAPH_SUCCESS);
}

TEST_F(SwigluMxQuantWithDualAxisTilingTest, test_tiling_invalid_round_mode_fp8)
{
    ge::DataType xDtype = ge::DT_FLOAT16;
    ge::DataType y1Dtype = ge::DT_FLOAT8_E4M3FN;
    ge::DataType mxScale1Dtype = ge::DT_FLOAT8_E8M0;
    ge::DataType y2Dtype = ge::DT_FLOAT8_E4M3FN;
    ge::DataType mxScale2Dtype = ge::DT_FLOAT8_E8M0;
    gert::StorageShape xShape = {{4, 4096}, {4, 4096}};
    gert::StorageShape y1Shape = {{4, 2048}, {4, 2048}};
    gert::StorageShape mxScale1Shape = {{4, 32, 2}, {4, 32, 2}};
    gert::StorageShape y2Shape = {{4, 2048}, {4, 2048}};
    gert::StorageShape mxScale2Shape = {{1, 2048, 2}, {1, 2048, 2}};
    ExecuteTestCase(
        xDtype, y1Dtype, mxScale1Dtype, y2Dtype, mxScale2Dtype,
        xShape, y1Shape, mxScale1Shape, y2Shape, mxScale2Shape,
        false, "floor", 36, 0,
        ge::GRAPH_FAILED);
}

TEST_F(SwigluMxQuantWithDualAxisTilingTest, test_tiling_invalid_dst_type)
{
    ge::DataType xDtype = ge::DT_FLOAT16;
    ge::DataType y1Dtype = ge::DT_FLOAT8_E4M3FN;
    ge::DataType mxScale1Dtype = ge::DT_FLOAT8_E8M0;
    ge::DataType y2Dtype = ge::DT_FLOAT8_E4M3FN;
    ge::DataType mxScale2Dtype = ge::DT_FLOAT8_E8M0;
    gert::StorageShape xShape = {{4, 4096}, {4, 4096}};
    gert::StorageShape y1Shape = {{4, 2048}, {4, 2048}};
    gert::StorageShape mxScale1Shape = {{4, 32, 2}, {4, 32, 2}};
    gert::StorageShape y2Shape = {{4, 2048}, {4, 2048}};
    gert::StorageShape mxScale2Shape = {{1, 2048, 2}, {1, 2048, 2}};
    ExecuteTestCase(
        xDtype, y1Dtype, mxScale1Dtype, y2Dtype, mxScale2Dtype,
        xShape, y1Shape, mxScale1Shape, y2Shape, mxScale2Shape,
        false, "rint", 99, 0,
        ge::GRAPH_FAILED);
}

TEST_F(SwigluMxQuantWithDualAxisTilingTest, test_tiling_scale_alg_1)
{
    ge::DataType xDtype = ge::DT_FLOAT16;
    ge::DataType y1Dtype = ge::DT_FLOAT8_E4M3FN;
    ge::DataType mxScale1Dtype = ge::DT_FLOAT8_E8M0;
    ge::DataType y2Dtype = ge::DT_FLOAT8_E4M3FN;
    ge::DataType mxScale2Dtype = ge::DT_FLOAT8_E8M0;
    gert::StorageShape xShape = {{4, 4096}, {4, 4096}};
    gert::StorageShape y1Shape = {{4, 2048}, {4, 2048}};
    gert::StorageShape mxScale1Shape = {{4, 32, 2}, {4, 32, 2}};
    gert::StorageShape y2Shape = {{4, 2048}, {4, 2048}};
    gert::StorageShape mxScale2Shape = {{1, 2048, 2}, {1, 2048, 2}};
    ExecuteTestCase(
        xDtype, y1Dtype, mxScale1Dtype, y2Dtype, mxScale2Dtype,
        xShape, y1Shape, mxScale1Shape, y2Shape, mxScale2Shape,
        false, "rint", 36, 1,
        ge::GRAPH_SUCCESS);
}
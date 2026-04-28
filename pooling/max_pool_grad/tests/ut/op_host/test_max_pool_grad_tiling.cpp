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

using namespace ut_util;
using namespace std;
using namespace ge;

namespace optiling {
struct MaxPoolGradWithArgmaxCompileInfo {
    platform_ascendc::SocVersion socVersion;
    uint64_t coreNum;
    uint64_t ubSize;
};
} // namespace optiling

class MaxPoolGradTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MaxPoolGradTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MaxPoolGradTiling TearDown" << std::endl;
    }
};

static void ExecuteTilingTestCase(
    gert::StorageShape x1Shape, gert::StorageShape x2Shape, gert::StorageShape gradShape, gert::StorageShape yShape,
    std::vector<int64_t> ksize, std::vector<int64_t> strides, std::string padding, std::string data_format,
    ge::DataType dtype, ge::graphStatus expect_status)
{
    string compile_info_string = R"({
         "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                           "Intrinsic_fix_pipe_l0c2out": false,
                           "Intrinsic_data_move_l12ub": true,
                           "Intrinsic_data_move_l0c2ub": true,
                           "Intrinsic_data_move_out2l1_nd2nz": false,
                           "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                           "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                           "CORE_NUM": 64}
                           })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);
    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}};
    map<string, string> npuarchs = {{"NpuArch", "3510"}};

    fe::PlatFormInfos platform_info;
    platform_info.Init();

    optiling::MaxPoolGradWithArgmaxCompileInfo compile_info;

    std::string op_type("MaxPoolGrad");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs(std::vector<void*>{&compile_info})
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "version", soc_version_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);

    ge::Format format = (data_format == "NHWC") ? ge::FORMAT_NHWC : ge::FORMAT_NCHW;

    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&x1Shape, &x2Shape, &gradShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, dtype, format, format)
                      .NodeInputTd(1, dtype, format, format)
                      .NodeInputTd(2, dtype, format, format)
                      .NodeOutputTd(0, dtype, format, format)
                      .NodeAttrs(
                          {{"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(ksize)},
                           {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(strides)},
                           {"padding", Ops::NN::AnyValue::CreateFrom<std::string>(padding)},
                           {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>(data_format)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);

    EXPECT_EQ(tiling_func(tiling_context), expect_status);
}

// ============================================================
// Failure cases first (TDD)
// ============================================================

TEST_F(MaxPoolGradTiling, MaxPoolGrad_tiling_fail_unsupported_dtype_int32)
{
    // x1(1,1,4,4) x2(1,1,2,2) grad(1,1,2,2) y(1,1,4,4) ksize=2 strides=2 padding=VALID
    gert::StorageShape x1Shape = {{1, 1, 4, 4}, {1, 1, 4, 4}};
    gert::StorageShape x2Shape = {{1, 1, 2, 2}, {1, 1, 2, 2}};
    gert::StorageShape gradShape = {{1, 1, 2, 2}, {1, 1, 2, 2}};
    gert::StorageShape yShape = {{1, 1, 4, 4}, {1, 1, 4, 4}};
    ExecuteTilingTestCase(
        x1Shape, x2Shape, gradShape, yShape, {1, 2, 2, 1}, {1, 2, 2, 1}, "VALID", "NCHW", ge::DT_INT32,
        ge::GRAPH_FAILED);
}

TEST_F(MaxPoolGradTiling, MaxPoolGrad_tiling_fail_unsupported_dtype_double)
{
    gert::StorageShape x1Shape = {{1, 1, 4, 4}, {1, 1, 4, 4}};
    gert::StorageShape x2Shape = {{1, 1, 2, 2}, {1, 1, 2, 2}};
    gert::StorageShape gradShape = {{1, 1, 2, 2}, {1, 1, 2, 2}};
    gert::StorageShape yShape = {{1, 1, 4, 4}, {1, 1, 4, 4}};
    ExecuteTilingTestCase(
        x1Shape, x2Shape, gradShape, yShape, {1, 2, 2, 1}, {1, 2, 2, 1}, "VALID", "NCHW", ge::DT_DOUBLE,
        ge::GRAPH_FAILED);
}

TEST_F(MaxPoolGradTiling, MaxPoolGrad_tiling_fail_invalid_padding_mode)
{
    gert::StorageShape x1Shape = {{1, 1, 4, 4}, {1, 1, 4, 4}};
    gert::StorageShape x2Shape = {{1, 1, 2, 2}, {1, 1, 2, 2}};
    gert::StorageShape gradShape = {{1, 1, 2, 2}, {1, 1, 2, 2}};
    gert::StorageShape yShape = {{1, 1, 4, 4}, {1, 1, 4, 4}};
    ExecuteTilingTestCase(
        x1Shape, x2Shape, gradShape, yShape, {1, 2, 2, 1}, {1, 2, 2, 1}, "INVALID_PAD", "NCHW", ge::DT_FLOAT,
        ge::GRAPH_FAILED);
}

TEST_F(MaxPoolGradTiling, MaxPoolGrad_tiling_fail_invalid_dim_num)
{
    // 3D input instead of 4D
    gert::StorageShape x1Shape = {{1, 4, 4}, {1, 4, 4}};
    gert::StorageShape x2Shape = {{1, 2, 2}, {1, 2, 2}};
    gert::StorageShape gradShape = {{1, 2, 2}, {1, 2, 2}};
    gert::StorageShape yShape = {{1, 4, 4}, {1, 4, 4}};
    ExecuteTilingTestCase(
        x1Shape, x2Shape, gradShape, yShape, {1, 2, 2, 1}, {1, 2, 2, 1}, "VALID", "NCHW", ge::DT_FLOAT,
        ge::GRAPH_FAILED);
}

TEST_F(MaxPoolGradTiling, MaxPoolGrad_tiling_fail_grad_shape_mismatch)
{
    // grad shape != x2 shape
    gert::StorageShape x1Shape = {{1, 1, 4, 4}, {1, 1, 4, 4}};
    gert::StorageShape x2Shape = {{1, 1, 2, 2}, {1, 1, 2, 2}};
    gert::StorageShape gradShape = {{1, 1, 3, 3}, {1, 1, 3, 3}};
    gert::StorageShape yShape = {{1, 1, 4, 4}, {1, 1, 4, 4}};
    ExecuteTilingTestCase(
        x1Shape, x2Shape, gradShape, yShape, {1, 2, 2, 1}, {1, 2, 2, 1}, "VALID", "NCHW", ge::DT_FLOAT,
        ge::GRAPH_FAILED);
}

// ============================================================
// Success cases
// ============================================================

TEST_F(MaxPoolGradTiling, MaxPoolGrad_tiling_success_fp32_valid_nchw)
{
    gert::StorageShape x1Shape = {{1, 1, 4, 4}, {1, 1, 4, 4}};
    gert::StorageShape x2Shape = {{1, 1, 2, 2}, {1, 1, 2, 2}};
    gert::StorageShape gradShape = {{1, 1, 2, 2}, {1, 1, 2, 2}};
    gert::StorageShape yShape = {{1, 1, 4, 4}, {1, 1, 4, 4}};
    ExecuteTilingTestCase(
        x1Shape, x2Shape, gradShape, yShape, {1, 2, 2, 1}, {1, 2, 2, 1}, "VALID", "NCHW", ge::DT_FLOAT,
        ge::GRAPH_SUCCESS);
}

TEST_F(MaxPoolGradTiling, MaxPoolGrad_tiling_success_fp16_same_nchw)
{
    gert::StorageShape x1Shape = {{2, 3, 8, 8}, {2, 3, 8, 8}};
    gert::StorageShape x2Shape = {{2, 3, 4, 4}, {2, 3, 4, 4}};
    gert::StorageShape gradShape = {{2, 3, 4, 4}, {2, 3, 4, 4}};
    gert::StorageShape yShape = {{2, 3, 8, 8}, {2, 3, 8, 8}};
    ExecuteTilingTestCase(
        x1Shape, x2Shape, gradShape, yShape, {1, 2, 2, 1}, {1, 2, 2, 1}, "SAME", "NCHW", ge::DT_FLOAT16,
        ge::GRAPH_SUCCESS);
}

TEST_F(MaxPoolGradTiling, MaxPoolGrad_tiling_success_bf16_valid_nchw)
{
    gert::StorageShape x1Shape = {{1, 8, 16, 16}, {1, 8, 16, 16}};
    gert::StorageShape x2Shape = {{1, 8, 8, 8}, {1, 8, 8, 8}};
    gert::StorageShape gradShape = {{1, 8, 8, 8}, {1, 8, 8, 8}};
    gert::StorageShape yShape = {{1, 8, 16, 16}, {1, 8, 16, 16}};
    ExecuteTilingTestCase(
        x1Shape, x2Shape, gradShape, yShape, {1, 2, 2, 1}, {1, 2, 2, 1}, "VALID", "NCHW", ge::DT_BF16,
        ge::GRAPH_SUCCESS);
}

TEST_F(MaxPoolGradTiling, MaxPoolGrad_tiling_success_fp32_same_nchw_large)
{
    gert::StorageShape x1Shape = {{4, 16, 32, 32}, {4, 16, 32, 32}};
    gert::StorageShape x2Shape = {{4, 16, 16, 16}, {4, 16, 16, 16}};
    gert::StorageShape gradShape = {{4, 16, 16, 16}, {4, 16, 16, 16}};
    gert::StorageShape yShape = {{4, 16, 32, 32}, {4, 16, 32, 32}};
    ExecuteTilingTestCase(
        x1Shape, x2Shape, gradShape, yShape, {1, 3, 3, 1}, {1, 2, 2, 1}, "SAME", "NCHW", ge::DT_FLOAT,
        ge::GRAPH_SUCCESS);
}

TEST_F(MaxPoolGradTiling, MaxPoolGrad_tiling_success_fp16_valid_nchw_k3)
{
    gert::StorageShape x1Shape = {{1, 1, 8, 8}, {1, 1, 8, 8}};
    gert::StorageShape x2Shape = {{1, 1, 2, 2}, {1, 1, 2, 2}};
    gert::StorageShape gradShape = {{1, 1, 2, 2}, {1, 1, 2, 2}};
    gert::StorageShape yShape = {{1, 1, 8, 8}, {1, 1, 8, 8}};
    ExecuteTilingTestCase(
        x1Shape, x2Shape, gradShape, yShape, {1, 3, 3, 1}, {1, 3, 3, 1}, "VALID", "NCHW", ge::DT_FLOAT16,
        ge::GRAPH_SUCCESS);
}

// ============================================================
// Big kernel cases
// ============================================================

TEST_F(MaxPoolGradTiling, MaxPoolGrad_tiling_success_big_kernel_fp16_valid_nchw_k16_s16)
{
    gert::StorageShape x1Shape = {{1, 4, 64, 64}, {1, 4, 64, 64}};
    gert::StorageShape x2Shape = {{1, 4, 4, 4}, {1, 4, 4, 4}};
    gert::StorageShape gradShape = {{1, 4, 4, 4}, {1, 4, 4, 4}};
    gert::StorageShape yShape = {{1, 4, 64, 64}, {1, 4, 64, 64}};

    ExecuteTilingTestCase(
        x1Shape, x2Shape, gradShape, yShape,
        {1, 1, 16, 16},
        {1, 1, 16, 16},
        "VALID",
        "NCHW",
        ge::DT_FLOAT16,
        ge::GRAPH_SUCCESS);
}

TEST_F(MaxPoolGradTiling, MaxPoolGrad_tiling_success_big_kernel_bf16_valid_nchw_k20_s10)
{
    gert::StorageShape x1Shape = {{1, 4, 80, 80}, {1, 4, 80, 80}};
    gert::StorageShape x2Shape = {{1, 4, 7, 7}, {1, 4, 7, 7}};
    gert::StorageShape gradShape = {{1, 4, 7, 7}, {1, 4, 7, 7}};
    gert::StorageShape yShape = {{1, 4, 80, 80}, {1, 4, 80, 80}};

    ExecuteTilingTestCase(
        x1Shape, x2Shape, gradShape, yShape,
        {1, 1, 20, 20},
        {1, 1, 10, 10},
        "VALID",
        "NCHW",
        ge::DT_BF16,
        ge::GRAPH_SUCCESS);
}

TEST_F(MaxPoolGradTiling, MaxPoolGrad_tiling_success_big_kernel_bf16_same_nchw_k18_10_large_w)
{
    gert::StorageShape x1Shape = {{4, 4, 150, 3417}, {4, 4, 150, 3417}};
    gert::StorageShape x2Shape = {{4, 4, 15, 285}, {4, 4, 15, 285}};
    gert::StorageShape gradShape = {{4, 4, 15, 285}, {4, 4, 15, 285}};
    gert::StorageShape yShape = {{4, 4, 150, 3417}, {4, 4, 150, 3417}};

    ExecuteTilingTestCase(
        x1Shape, x2Shape, gradShape, yShape,
        {1, 1, 18, 10},
        {1, 1, 10, 12},
        "SAME",
        "NCHW",
        ge::DT_BF16,
        ge::GRAPH_SUCCESS);
}

// ============================================================
// Small kernel cases
// ============================================================

TEST_F(MaxPoolGradTiling, MaxPoolGrad_tiling_success_small_kernel_fp32_valid_nchw_k2_s2)
{
    gert::StorageShape x1Shape = {{2, 16, 64, 128}, {2, 16, 64, 128}};
    gert::StorageShape x2Shape = {{2, 16, 32, 64}, {2, 16, 32, 64}};
    gert::StorageShape gradShape = {{2, 16, 32, 64}, {2, 16, 32, 64}};
    gert::StorageShape yShape = {{2, 16, 64, 128}, {2, 16, 64, 128}};

    ExecuteTilingTestCase(
        x1Shape, x2Shape, gradShape, yShape,
        {1, 1, 2, 2},
        {1, 1, 2, 2},
        "VALID",
        "NCHW",
        ge::DT_FLOAT,
        ge::GRAPH_SUCCESS);
}

TEST_F(MaxPoolGradTiling, MaxPoolGrad_tiling_success_small_kernel_fp32_valid_nchw_k3_5_s2_4)
{
    gert::StorageShape x1Shape = {{1, 32, 35, 137}, {1, 32, 35, 137}};
    gert::StorageShape x2Shape = {{1, 32, 17, 34}, {1, 32, 17, 34}};
    gert::StorageShape gradShape = {{1, 32, 17, 34}, {1, 32, 17, 34}};
    gert::StorageShape yShape = {{1, 32, 35, 137}, {1, 32, 35, 137}};

    ExecuteTilingTestCase(
        x1Shape, x2Shape, gradShape, yShape,
        {1, 1, 3, 5},
        {1, 1, 2, 4},
        "VALID",
        "NCHW",
        ge::DT_FLOAT,
        ge::GRAPH_SUCCESS);
}

TEST_F(MaxPoolGradTiling, MaxPoolGrad_tiling_success_small_kernel_fp32_same_nchw_k2_s2_odd_shape)
{
    gert::StorageShape x1Shape = {{1, 24, 33, 65}, {1, 24, 33, 65}};
    gert::StorageShape x2Shape = {{1, 24, 17, 33}, {1, 24, 17, 33}};
    gert::StorageShape gradShape = {{1, 24, 17, 33}, {1, 24, 17, 33}};
    gert::StorageShape yShape = {{1, 24, 33, 65}, {1, 24, 33, 65}};

    ExecuteTilingTestCase(
        x1Shape, x2Shape, gradShape, yShape,
        {1, 1, 2, 2},
        {1, 1, 2, 2},
        "SAME",
        "NCHW",
        ge::DT_FLOAT,
        ge::GRAPH_SUCCESS);
}
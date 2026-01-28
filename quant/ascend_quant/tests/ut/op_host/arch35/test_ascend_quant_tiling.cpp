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
 * \file test_ascend_quant_ascendc_tiling.cpp
 * \brief
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
#include "../../../../op_host/arch35/ascend_quant_tiling.h"
#include "../../../../op_host/arch35/ascend_quant_tiling_arch35.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"

using namespace std;

class AscendQuantRegbaseTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AscendQuantRegbaseTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AscendQuantRegbaseTiling TearDown" << std::endl;
    }
};

static void ExecuteTestCase(
    ge::DataType xDtype, ge::DataType yDtype, gert::StorageShape xShape, gert::StorageShape yShape, float scale,
    float offset, bool sqrt_mode, string round_mode, int64_t dst_type, int64_t expectTilingKey, string expectTilingData,
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
    map<string, string> socversions = {{"Short_SoC_version", "Ascend910_95"}};
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();

    // compile info
    optiling::AscendQuantCompileInfo compile_info;

    std::string op_type("AscendQuant");
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", socversions);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1})
                      .InputShapes({&xShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, yDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"scale", Ops::NN::AnyValue::CreateFrom<float>(scale)},
                           {"offset", Ops::NN::AnyValue::CreateFrom<float>(offset)},
                           {"sqrt_mode", Ops::NN::AnyValue::CreateFrom<bool>(sqrt_mode)},
                           {"round_mode", Ops::NN::AnyValue::CreateFrom<string>(round_mode)},
                           {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(dst_type)}})
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

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), status);
    if (status == ge::GRAPH_FAILED) {
        return;
    }
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, expectTilingKey);
}

TEST_F(AscendQuantRegbaseTiling, test_tiling_round_001)
{
    ge::DataType xDtype = ge::DT_FLOAT;
    ge::DataType yDtype = ge::DT_INT8;
    gert::StorageShape xShape = {{1, 64, 2, 64}, {1, 64, 2, 64}};
    gert::StorageShape yShape = {{1, 64, 2, 64}, {1, 64, 2, 64}};
    float scale = 2.0;
    float offset = 1.0;
    bool sqrt_mode = true;
    string round_mode = "Round";
    int64_t dst_type = 2;
    int64_t expectTilingKey = 8;
    string expectTilingData = "64 8192 128 128 128 0 4575657222490554368 ";
    ExecuteTestCase(
        xDtype, yDtype, xShape, yShape, scale, offset, sqrt_mode, round_mode, dst_type, expectTilingKey,
        expectTilingData, ge::GRAPH_SUCCESS);
}

TEST_F(AscendQuantRegbaseTiling, test_tiling_floor_002)
{
    ge::DataType xDtype = ge::DT_FLOAT;
    ge::DataType yDtype = ge::DT_INT8;
    gert::StorageShape xShape = {{1, 64, 2, 64}, {1, 64, 2, 64}};
    gert::StorageShape yShape = {{1, 64, 2, 64}, {1, 64, 2, 64}};
    float scale = 2.0;
    float offset = 1.0;
    bool sqrt_mode = true;
    string round_mode = "Floor";
    int64_t dst_type = 2;
    int64_t expectTilingKey = 9;
    string expectTilingData = "64 8192 128 128 128 1 4575657222490554368 ";
    ExecuteTestCase(
        xDtype, yDtype, xShape, yShape, scale, offset, sqrt_mode, round_mode, dst_type, expectTilingKey,
        expectTilingData, ge::GRAPH_SUCCESS);
}

TEST_F(AscendQuantRegbaseTiling, test_tiling_ceil_003)
{
    ge::DataType xDtype = ge::DT_FLOAT;
    ge::DataType yDtype = ge::DT_INT8;
    gert::StorageShape xShape = {{1, 64, 2, 64}, {1, 64, 2, 64}};
    gert::StorageShape yShape = {{1, 64, 2, 64}, {1, 64, 2, 64}};
    float scale = 2.0;
    float offset = 1.0;
    bool sqrt_mode = true;
    string round_mode = "Ceil";
    int64_t dst_type = 2;
    int64_t expectTilingKey = 10;
    string expectTilingData = "64 8192 128 128 128 2 4575657222490554368 ";
    ExecuteTestCase(
        xDtype, yDtype, xShape, yShape, scale, offset, sqrt_mode, round_mode, dst_type, expectTilingKey,
        expectTilingData, ge::GRAPH_SUCCESS);
}

TEST_F(AscendQuantRegbaseTiling, test_tiling_trunc_004)
{
    ge::DataType xDtype = ge::DT_FLOAT;
    ge::DataType yDtype = ge::DT_INT8;
    gert::StorageShape xShape = {{1, 64, 2, 64}, {1, 64, 2, 64}};
    gert::StorageShape yShape = {{1, 64, 2, 64}, {1, 64, 2, 64}};
    float scale = 2.0;
    float offset = 1.0;
    bool sqrt_mode = true;
    string round_mode = "Trunc";
    int64_t dst_type = 2;
    int64_t expectTilingKey = 11;
    string expectTilingData = "64 8192 128 128 128 3 4575657222490554368 ";
    ExecuteTestCase(
        xDtype, yDtype, xShape, yShape, scale, offset, sqrt_mode, round_mode, dst_type, expectTilingKey,
        expectTilingData, ge::GRAPH_SUCCESS);
}

TEST_F(AscendQuantRegbaseTiling, test_tiling_fp32_int4_005)
{
    ge::DataType xDtype = ge::DT_FLOAT;
    ge::DataType yDtype = ge::DT_INT4;
    gert::StorageShape xShape = {{1, 64, 2, 64}, {1, 64, 2, 64}};
    gert::StorageShape yShape = {{1, 64, 2, 64}, {1, 64, 2, 64}};
    float scale = 2.0;
    float offset = 1.0;
    bool sqrt_mode = true;
    string round_mode = "Round";
    int64_t dst_type = 29;
    int64_t expectTilingKey = 8;
    string expectTilingData = "64 8192 128 128 128 0 4575657222490554368 ";
    ExecuteTestCase(
        xDtype, yDtype, xShape, yShape, scale, offset, sqrt_mode, round_mode, dst_type, expectTilingKey,
        expectTilingData, ge::GRAPH_SUCCESS);
}

TEST_F(AscendQuantRegbaseTiling, test_tiling_fp16_int8_006)
{
    ge::DataType xDtype = ge::DT_FLOAT16;
    ge::DataType yDtype = ge::DT_INT8;
    gert::StorageShape xShape = {{1, 64, 2, 64}, {1, 64, 2, 64}};
    gert::StorageShape yShape = {{1, 64, 2, 64}, {1, 64, 2, 64}};
    float scale = 2.0;
    float offset = 1.0;
    bool sqrt_mode = true;
    string round_mode = "Round";
    int64_t dst_type = 2;
    int64_t expectTilingKey = 8;
    string expectTilingData = "64 8192 128 128 128 0 4575657222490554368 ";
    ExecuteTestCase(
        xDtype, yDtype, xShape, yShape, scale, offset, sqrt_mode, round_mode, dst_type, expectTilingKey,
        expectTilingData, ge::GRAPH_SUCCESS);
}

TEST_F(AscendQuantRegbaseTiling, test_tiling_fp16_int4_007)
{
    ge::DataType xDtype = ge::DT_FLOAT16;
    ge::DataType yDtype = ge::DT_INT4;
    gert::StorageShape xShape = {{1, 64, 2, 64}, {1, 64, 2, 64}};
    gert::StorageShape yShape = {{1, 64, 2, 64}, {1, 64, 2, 64}};
    float scale = 2.0;
    float offset = 1.0;
    bool sqrt_mode = true;
    string round_mode = "Round";
    int64_t dst_type = 29;
    int64_t expectTilingKey = 8;
    string expectTilingData = "64 8192 128 128 128 0 4575657222490554368 ";
    ExecuteTestCase(
        xDtype, yDtype, xShape, yShape, scale, offset, sqrt_mode, round_mode, dst_type, expectTilingKey,
        expectTilingData, ge::GRAPH_SUCCESS);
}

TEST_F(AscendQuantRegbaseTiling, test_tiling_invalid_x_dtype)
{
    ge::DataType xDtype = ge::DT_INT8;
    ge::DataType yDtype = ge::DT_INT8;
    gert::StorageShape xShape = {{1, 64, 2, 64}, {1, 64, 2, 64}};
    gert::StorageShape yShape = {{1, 64, 2, 64}, {1, 64, 2, 64}};
    float scale = 2.0;
    float offset = 1.0;
    bool sqrt_mode = true;
    string round_mode = "Round";
    int64_t dst_type = 2;
    int64_t expectTilingKey = -1;
    string expectTilingData = " ";
    ExecuteTestCase(
        xDtype, yDtype, xShape, yShape, scale, offset, sqrt_mode, round_mode, dst_type, expectTilingKey,
        expectTilingData, ge::GRAPH_FAILED);
}

TEST_F(AscendQuantRegbaseTiling, test_tiling_invalid_y_dtype)
{
    ge::DataType xDtype = ge::DT_FLOAT;
    ge::DataType yDtype = ge::DT_FLOAT;
    gert::StorageShape xShape = {{1, 64, 2, 64}, {1, 64, 2, 64}};
    gert::StorageShape yShape = {{1, 64, 2, 64}, {1, 64, 2, 64}};
    float scale = 2.0;
    float offset = 1.0;
    bool sqrt_mode = true;
    string round_mode = "Round";
    int64_t dst_type = 0;
    int64_t expectTilingKey = -1;
    string expectTilingData = " ";
    ExecuteTestCase(
        xDtype, yDtype, xShape, yShape, scale, offset, sqrt_mode, round_mode, dst_type, expectTilingKey,
        expectTilingData, ge::GRAPH_FAILED);
}

TEST_F(AscendQuantRegbaseTiling, test_tiling_invalid_dst_type)
{
    ge::DataType xDtype = ge::DT_FLOAT;
    ge::DataType yDtype = ge::DT_INT8;
    gert::StorageShape xShape = {{1, 64, 2, 64}, {1, 64, 2, 64}};
    gert::StorageShape yShape = {{1, 64, 2, 64}, {1, 64, 2, 64}};
    float scale = 2.0;
    float offset = 1.0;
    bool sqrt_mode = true;
    string round_mode = "Round";
    int64_t dst_type = 1;
    int64_t expectTilingKey = -1;
    string expectTilingData = " ";
    ExecuteTestCase(
        xDtype, yDtype, xShape, yShape, scale, offset, sqrt_mode, round_mode, dst_type, expectTilingKey,
        expectTilingData, ge::GRAPH_FAILED);
}

TEST_F(AscendQuantRegbaseTiling, test_tiling_y_dst_type_not_same)
{
    ge::DataType xDtype = ge::DT_FLOAT;
    ge::DataType yDtype = ge::DT_INT8;
    gert::StorageShape xShape = {{1, 64, 2, 64}, {1, 64, 2, 64}};
    gert::StorageShape yShape = {{1, 64, 2, 64}, {1, 64, 2, 64}};
    float scale = 2.0;
    float offset = 1.0;
    bool sqrt_mode = true;
    string round_mode = "Round";
    int64_t dst_type = 29;
    int64_t expectTilingKey = -1;
    string expectTilingData = " ";
    ExecuteTestCase(
        xDtype, yDtype, xShape, yShape, scale, offset, sqrt_mode, round_mode, dst_type, expectTilingKey,
        expectTilingData, ge::GRAPH_FAILED);
}

TEST_F(AscendQuantRegbaseTiling, test_tiling_invalid_round_mode)
{
    ge::DataType xDtype = ge::DT_FLOAT;
    ge::DataType yDtype = ge::DT_INT8;
    gert::StorageShape xShape = {{1, 64, 2, 64}, {1, 64, 2, 64}};
    gert::StorageShape yShape = {{1, 64, 2, 64}, {1, 64, 2, 64}};
    float scale = 2.0;
    float offset = 1.0;
    bool sqrt_mode = true;
    string round_mode = "None";
    int64_t dst_type = 2;
    int64_t expectTilingKey = -1;
    string expectTilingData = " ";
    ExecuteTestCase(
        xDtype, yDtype, xShape, yShape, scale, offset, sqrt_mode, round_mode, dst_type, expectTilingKey,
        expectTilingData, ge::GRAPH_FAILED);
}

TEST_F(AscendQuantRegbaseTiling, test_tiling_invalid_x_shape)
{
    ge::DataType xDtype = ge::DT_FLOAT;
    ge::DataType yDtype = ge::DT_INT4;
    gert::StorageShape xShape = {{1, 64, 2, 63}, {1, 64, 2, 63}};
    gert::StorageShape yShape = {{1, 64, 2, 63}, {1, 64, 2, 63}};
    float scale = 2.0;
    float offset = 1.0;
    bool sqrt_mode = true;
    string round_mode = "Round";
    int64_t dst_type = 29;
    int64_t expectTilingKey = -1;
    string expectTilingData = " ";
    ExecuteTestCase(
        xDtype, yDtype, xShape, yShape, scale, offset, sqrt_mode, round_mode, dst_type, expectTilingKey,
        expectTilingData, ge::GRAPH_FAILED);
}

TEST_F(AscendQuantRegbaseTiling, test_tiling_invalid_x_dim)
{
    ge::DataType xDtype = ge::DT_FLOAT;
    ge::DataType yDtype = ge::DT_INT8;
    gert::StorageShape xShape = {{1, 2, 3, 4, 5, 6, 7, 8, 9}, {1, 2, 3, 4, 5, 6, 7, 8, 9}};
    gert::StorageShape yShape = {{1, 2, 3, 4, 5, 6, 7, 8, 9}, {1, 2, 3, 4, 5, 6, 7, 8, 9}};
    float scale = 2.0;
    float offset = 1.0;
    bool sqrt_mode = true;
    string round_mode = "Round";
    int64_t dst_type = 2;
    int64_t expectTilingKey = -1;
    string expectTilingData = " ";
    ExecuteTestCase(
        xDtype, yDtype, xShape, yShape, scale, offset, sqrt_mode, round_mode, dst_type, expectTilingKey,
        expectTilingData, ge::GRAPH_FAILED);
}

TEST_F(AscendQuantRegbaseTiling, test_tiling_empty_x)
{
    ge::DataType xDtype = ge::DT_FLOAT;
    ge::DataType yDtype = ge::DT_INT8;
    gert::StorageShape xShape = {{0}, {0}};
    gert::StorageShape yShape = {{0}, {0}};
    float scale = 2.0;
    float offset = 1.0;
    bool sqrt_mode = true;
    string round_mode = "Round";
    int64_t dst_type = 2;
    int64_t expectTilingKey = -1;
    string expectTilingData = " ";
    ExecuteTestCase(
        xDtype, yDtype, xShape, yShape, scale, offset, sqrt_mode, round_mode, dst_type, expectTilingKey,
        expectTilingData, ge::GRAPH_FAILED);
}

TEST_F(AscendQuantRegbaseTiling, test_tiling_x_y_shape_not_same)
{
    ge::DataType xDtype = ge::DT_FLOAT;
    ge::DataType yDtype = ge::DT_INT8;
    gert::StorageShape xShape = {{1, 2, 3}, {1, 2, 3}};
    gert::StorageShape yShape = {{1, 2, 4}, {1, 2, 4}};
    float scale = 2.0;
    float offset = 1.0;
    bool sqrt_mode = true;
    string round_mode = "Round";
    int64_t dst_type = 2;
    int64_t expectTilingKey = -1;
    string expectTilingData = " ";
    ExecuteTestCase(
        xDtype, yDtype, xShape, yShape, scale, offset, sqrt_mode, round_mode, dst_type, expectTilingKey,
        expectTilingData, ge::GRAPH_FAILED);
}

TEST_F(AscendQuantRegbaseTiling, test_tiling_fp8_e5m2_round)
{
    ge::DataType xDtype = ge::DT_FLOAT;
    ge::DataType yDtype = ge::DT_FLOAT8_E5M2;
    gert::StorageShape xShape = {{1, 64, 2, 64}, {1, 64, 2, 64}};
    gert::StorageShape yShape = {{1, 64, 2, 64}, {1, 64, 2, 64}};
    float scale = 2.0;
    float offset = 1.0;
    bool sqrt_mode = true;
    string round_mode = "Round";
    int64_t dst_type = 35;
    int64_t expectTilingKey = 12;
    string expectTilingData = "64 8192 128 128 128 4 4575657222490554368 ";
    ExecuteTestCase(
        xDtype, yDtype, xShape, yShape, scale, offset, sqrt_mode, round_mode, dst_type, expectTilingKey,
        expectTilingData, ge::GRAPH_SUCCESS);
}

TEST_F(AscendQuantRegbaseTiling, test_tiling_fp8_e4m3_round)
{
    ge::DataType xDtype = ge::DT_FLOAT;
    ge::DataType yDtype = ge::DT_FLOAT8_E4M3FN;
    gert::StorageShape xShape = {{1, 64, 2, 64}, {1, 64, 2, 64}};
    gert::StorageShape yShape = {{1, 64, 2, 64}, {1, 64, 2, 64}};
    float scale = 2.0;
    float offset = 1.0;
    bool sqrt_mode = true;
    string round_mode = "Round";
    int64_t dst_type = 36;
    int64_t expectTilingKey = 12;
    string expectTilingData = "64 8192 128 128 128 4 4575657222490554368 ";
    ExecuteTestCase(
        xDtype, yDtype, xShape, yShape, scale, offset, sqrt_mode, round_mode, dst_type, expectTilingKey,
        expectTilingData, ge::GRAPH_SUCCESS);
}

TEST_F(AscendQuantRegbaseTiling, test_tiling_hi8_round)
{
    ge::DataType xDtype = ge::DT_FLOAT;
    ge::DataType yDtype = ge::DT_HIFLOAT8;
    gert::StorageShape xShape = {{1, 64, 2, 64}, {1, 64, 2, 64}};
    gert::StorageShape yShape = {{1, 64, 2, 64}, {1, 64, 2, 64}};
    float scale = 2.0;
    float offset = 1.0;
    bool sqrt_mode = true;
    string round_mode = "Round";
    int64_t dst_type = 34;
    int64_t expectTilingKey = 8;
    string expectTilingData = "64 8192 128 128 128 0 4575657222490554368 ";
    ExecuteTestCase(
        xDtype, yDtype, xShape, yShape, scale, offset, sqrt_mode, round_mode, dst_type, expectTilingKey,
        expectTilingData, ge::GRAPH_SUCCESS);
}

TEST_F(AscendQuantRegbaseTiling, test_tiling_hi8_hybrid)
{
    ge::DataType xDtype = ge::DT_FLOAT;
    ge::DataType yDtype = ge::DT_HIFLOAT8;
    gert::StorageShape xShape = {{1, 64, 2, 64}, {1, 64, 2, 64}};
    gert::StorageShape yShape = {{1, 64, 2, 64}, {1, 64, 2, 64}};
    float scale = 2.0;
    float offset = 1.0;
    bool sqrt_mode = true;
    string round_mode = "Hybrid";
    int64_t dst_type = 34;
    int64_t expectTilingKey = 13;
    string expectTilingData = "64 8192 128 128 128 5 4575657222490554368 ";
    ExecuteTestCase(
        xDtype, yDtype, xShape, yShape, scale, offset, sqrt_mode, round_mode, dst_type, expectTilingKey,
        expectTilingData, ge::GRAPH_SUCCESS);
}

TEST_F(AscendQuantRegbaseTiling, test_tiling_fp8_invalid_round_mode)
{
    ge::DataType xDtype = ge::DT_FLOAT;
    ge::DataType yDtype = ge::DT_FLOAT8_E5M2;
    gert::StorageShape xShape = {{1, 64, 2, 64}, {1, 64, 2, 64}};
    gert::StorageShape yShape = {{1, 64, 2, 64}, {1, 64, 2, 64}};
    float scale = 2.0;
    float offset = 1.0;
    bool sqrt_mode = true;
    string round_mode = "None";
    int64_t dst_type = 35;
    int64_t expectTilingKey = -1;
    string expectTilingData = " ";
    ExecuteTestCase(
        xDtype, yDtype, xShape, yShape, scale, offset, sqrt_mode, round_mode, dst_type, expectTilingKey,
        expectTilingData, ge::GRAPH_FAILED);
}

TEST_F(AscendQuantRegbaseTiling, test_tiling_hi8_invalid_round_mode)
{
    ge::DataType xDtype = ge::DT_FLOAT;
    ge::DataType yDtype = ge::DT_HIFLOAT8;
    gert::StorageShape xShape = {{1, 64, 2, 64}, {1, 64, 2, 64}};
    gert::StorageShape yShape = {{1, 64, 2, 64}, {1, 64, 2, 64}};
    float scale = 2.0;
    float offset = 1.0;
    bool sqrt_mode = true;
    string round_mode = "None";
    int64_t dst_type = 34;
    int64_t expectTilingKey = -1;
    string expectTilingData = " ";
    ExecuteTestCase(
        xDtype, yDtype, xShape, yShape, scale, offset, sqrt_mode, round_mode, dst_type, expectTilingKey,
        expectTilingData, ge::GRAPH_FAILED);
}
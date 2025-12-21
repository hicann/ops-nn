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
 * \file test_dynamic_block_quant_tiling.cpp
 * \brief
 */

#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "platform/platform_infos_def.h"
#include "ut_op_util.h"
#include "quant/dynamic_block_quant/op_host/dynamic_block_quant_tiling.h"
#include "kernel_run_context_facker.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "test_cube_util.h"

using namespace std;
using namespace ge;
using namespace ut_util;

class DynamicBlockQuantTiling : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "DynamicBlockQuantTiling SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "DynamicBlockQuantTiling TearDown" << std::endl;
    }
};

template <typename T>
static string to_string(void *buf, size_t size)
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

static void ExecuteTestCase(ge::DataType inDtype, ge::DataType outDtype, gert::StorageShape shape,
    gert::StorageShape scaleShape, float minScale, string roundMode, int64_t rowBlockSize, int64_t colBlockSize, string expectTilingData,
    ge::graphStatus status = ge::GRAPH_SUCCESS) {
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
    optiling::DynamicBlockQuantCompileInfo compile_info;

    std::string op_type("DynamicBlockQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder = gert::KernelRunContextFaker()
                        .KernelIONum(2, 1)
                        .Inputs({const_cast<char *>("{}"), reinterpret_cast<void *>(&platform_info)})
                        .Outputs({&compile_info})
                        .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                    .NodeIoNum(1, 2)
                    .IrInstanceNum({1})
                    .InputShapes({&shape})
                    .OutputShapes({&shape, &scaleShape})
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                    .NodeInputTd(0, inDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, outDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({{"min_scale", Ops::NN::AnyValue::CreateFrom<float>(minScale)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<string>(roundMode)},
                                {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(outDtype)},
                                {"row_block_size", Ops::NN::AnyValue::CreateFrom(rowBlockSize)},
                                {"col_block_size", Ops::NN::AnyValue::CreateFrom(colBlockSize)}})
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
    auto tiling_data_result = to_string<int64_t>(raw_tiling_data->GetData(), raw_tiling_data->GetDataSize() - 608);
    EXPECT_EQ(tiling_data_result, expectTilingData);
}

static void ExecuteTestCaseA2(ge::DataType inDtype, ge::DataType outDtype, gert::StorageShape shape,
    gert::StorageShape scaleShape, float minScale, string roundMode, int64_t rowBlockSize, int64_t colBlockSize, string expectTilingData,
    ge::graphStatus status = ge::GRAPH_SUCCESS)
{
    string compile_info_string = R"({
                                        "hardware_info": {
                                            "BT_SIZE": 0,
                                            "load3d_constraints": "1",
                                            "Intrinsic_fix_pipe_l0c2out": false,
                                            "Intrinsic_data_move_l12ub": true,
                                            "Intrinsic_data_move_l0c2ub": true,
                                            "Intrinsic_data_move_out2l1_nd2nz": false,
                                            "UB_SIZE": 196608,
                                            "L2_SIZE": 33554432,
                                            "L1_SIZE": 524288,
                                            "L0A_SIZE": 65536,
                                            "L0B_SIZE": 65536,
                                            "L0C_SIZE": 131072,
                                            "CORE_NUM": 48
                                        }
                                    })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_versions = {{"Short_SoC_version", "Ascend910B"}};

    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();

    // compile info
    optiling::DynamicBlockQuantCompileInfo compile_info;

    std::string op_type("DynamicBlockQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder = gert::KernelRunContextFaker()
                        .KernelIONum(2, 1)
                        .Inputs({const_cast<char *>("{}"), reinterpret_cast<void *>(&platform_info)})
                        .Outputs({&compile_info})
                        .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_versions);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector *>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                    .NodeIoNum(1, 2)
                    .IrInstanceNum({1})
                    .InputShapes({&shape})
                    .OutputShapes({&shape, &scaleShape})
                    .CompileInfo(&compile_info)
                    .PlatformInfo(reinterpret_cast<char *>(&platform_info))
                    .NodeInputTd(0, inDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(0, outDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                    .NodeAttrs({{"min_scale", Ops::NN::AnyValue::CreateFrom<float>(minScale)},
                                {"round_mode", Ops::NN::AnyValue::CreateFrom<string>(roundMode)},
                                {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(outDtype)},
                                {"row_block_size", Ops::NN::AnyValue::CreateFrom(rowBlockSize)},
                                {"col_block_size", Ops::NN::AnyValue::CreateFrom(colBlockSize)}})
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
    auto tiling_data_result = to_string<int64_t>(raw_tiling_data->GetData(), raw_tiling_data->GetDataSize() - 600);
    EXPECT_EQ(tiling_data_result, expectTilingData);
}

TEST_F(DynamicBlockQuantTiling, DynamicBlockQuant_tiling_ascendc_float16_fp8e5m2) {
    gert::StorageShape shape = {{128, 256}, {128, 256}};
    gert::StorageShape scaleShape = {{1, 2}, {1, 2}};
    float minScale = 0.0;
    string roundMode = "rint";
    int64_t rowBlockSize = 128;
    int64_t colBlockSize = 128;
    string expectTilingData = "1100 64 253952 256 0 1 35 128 128 128 256 1 2 1 1 128 128 2 1 2 1 1 1 1 1 2 0 0 ";

    ExecuteTestCase(ge::DT_FLOAT16, ge::DT_FLOAT8_E5M2, shape, scaleShape, minScale, roundMode, rowBlockSize, colBlockSize, expectTilingData);
}

TEST_F(DynamicBlockQuantTiling, DynamicBlockQuant_tiling_ascendc_float16_fp8e4m3) {
    gert::StorageShape shape = {{128, 256}, {128, 256}};
    gert::StorageShape scaleShape = {{1, 2}, {1, 2}};
    float minScale = 0.0;
    string roundMode = "rint";
    int64_t rowBlockSize = 128;
    int64_t colBlockSize = 128;
    string expectTilingData = "1110 64 253952 256 0 1 36 128 128 128 256 1 2 1 1 128 128 2 1 2 1 1 1 1 1 2 0 0 ";

    ExecuteTestCase(ge::DT_FLOAT16, ge::DT_FLOAT8_E4M3FN, shape, scaleShape, minScale, roundMode, rowBlockSize, colBlockSize, expectTilingData);
}

TEST_F(DynamicBlockQuantTiling, DynamicBlockQuant_tiling_ascendc_float16_hifloat8) {
    gert::StorageShape shape = {{128, 256}, {128, 256}};
    gert::StorageShape scaleShape = {{1, 2}, {1, 2}};
    float minScale = 0.0;
    string roundMode = "round";
    int64_t rowBlockSize = 128;
    int64_t colBlockSize = 128;
    string expectTilingData = "4120 64 253952 256 0 4 34 128 128 128 256 1 2 1 1 128 128 2 1 2 1 1 1 1 1 2 0 0 ";

    ExecuteTestCase(ge::DT_FLOAT16, ge::DT_HIFLOAT8, shape, scaleShape, minScale, roundMode, rowBlockSize, colBlockSize, expectTilingData);
}

TEST_F(DynamicBlockQuantTiling, DynamicBlockQuant_tiling_ascendc_bfloat16_fp8e5m2) {
    gert::StorageShape shape = {{128, 256}, {128, 256}};
    gert::StorageShape scaleShape = {{1, 2}, {1, 2}};
    float minScale = 0.0;
    string roundMode = "rint";
    int64_t rowBlockSize = 128;
    int64_t colBlockSize = 128;
    string expectTilingData = "1200 64 253952 256 0 1 35 128 128 128 256 1 2 1 1 128 128 2 1 2 1 1 1 1 1 2 0 0 ";

    ExecuteTestCase(ge::DT_BF16, ge::DT_FLOAT8_E5M2, shape, scaleShape, minScale, roundMode, rowBlockSize, colBlockSize, expectTilingData);
}

TEST_F(DynamicBlockQuantTiling, DynamicBlockQuant_tiling_ascendc_bfloat16_fp8e4m3) {
    gert::StorageShape shape = {{128, 256}, {128, 256}};
    gert::StorageShape scaleShape = {{1, 2}, {1, 2}};
    float minScale = 0.0;
    string roundMode = "rint";
    int64_t rowBlockSize = 128;
    int64_t colBlockSize = 128;
    string expectTilingData = "1210 64 253952 256 0 1 36 128 128 128 256 1 2 1 1 128 128 2 1 2 1 1 1 1 1 2 0 0 ";

    ExecuteTestCase(ge::DT_BF16, ge::DT_FLOAT8_E4M3FN, shape, scaleShape, minScale, roundMode, rowBlockSize, colBlockSize, expectTilingData);
}

TEST_F(DynamicBlockQuantTiling, DynamicBlockQuant_tiling_ascendc_bfloat16_hifloat8) {
    gert::StorageShape shape = {{128, 256}, {128, 256}};
    gert::StorageShape scaleShape = {{1, 2}, {1, 2}};
    float minScale = 0.0;
    string roundMode = "round";
    int64_t rowBlockSize = 128;
    int64_t colBlockSize = 128;
    string expectTilingData = "4220 64 253952 256 0 4 34 128 128 128 256 1 2 1 1 128 128 2 1 2 1 1 1 1 1 2 0 0 ";

    ExecuteTestCase(ge::DT_BF16, ge::DT_HIFLOAT8, shape, scaleShape, minScale, roundMode, rowBlockSize, colBlockSize, expectTilingData);
}

TEST_F(DynamicBlockQuantTiling, DynamicBlockQuant_tiling_ascendc_float16_fp8e5m2_one_row) {
    gert::StorageShape shape = {{128, 256}, {128, 256}};
    gert::StorageShape scaleShape = {{128, 2}, {128, 2}};
    float minScale = 0.0;
    string roundMode = "rint";
    int64_t rowBlockSize = 1;
    int64_t colBlockSize = 128;
    string expectTilingData = "1101 64 253952 256 0 1 35 1 128 128 256 128 2 1 198 1 25344 64 32 2 4 1 4 1 32 2 0 0 ";

    ExecuteTestCase(ge::DT_FLOAT16, ge::DT_FLOAT8_E5M2, shape, scaleShape, minScale, roundMode, rowBlockSize, colBlockSize, expectTilingData);
}

TEST_F(DynamicBlockQuantTiling, DynamicBlockQuant_tiling_ascendc_float16_fp8e4m3_one_row) {
    gert::StorageShape shape = {{128, 256}, {128, 256}};
    gert::StorageShape scaleShape = {{128, 2}, {128, 2}};
    float minScale = 0.0;
    string roundMode = "rint";
    int64_t rowBlockSize = 1;
    int64_t colBlockSize = 128;
    string expectTilingData = "1111 64 253952 256 0 1 36 1 128 128 256 128 2 1 198 1 25344 64 32 2 4 1 4 1 32 2 0 0 ";

    ExecuteTestCase(ge::DT_FLOAT16, ge::DT_FLOAT8_E4M3FN, shape, scaleShape, minScale, roundMode, rowBlockSize, colBlockSize, expectTilingData);
}

TEST_F(DynamicBlockQuantTiling, DynamicBlockQuant_tiling_ascendc_float16_hifloat8_one_row) {
    gert::StorageShape shape = {{128, 256}, {128, 256}};
    gert::StorageShape scaleShape = {{128, 2}, {128, 2}};
    float minScale = 0.0;
    string roundMode = "round";
    int64_t rowBlockSize = 1;
    int64_t colBlockSize = 128;
    string expectTilingData = "4121 64 253952 256 0 4 34 1 128 128 256 128 2 1 198 1 25344 64 32 2 4 1 4 1 32 2 0 0 ";

    ExecuteTestCase(ge::DT_FLOAT16, ge::DT_HIFLOAT8, shape, scaleShape, minScale, roundMode, rowBlockSize, colBlockSize, expectTilingData);
}

TEST_F(DynamicBlockQuantTiling, DynamicBlockQuant_tiling_ascendc_bfloat16_fp8e5m2_one_row) {
    gert::StorageShape shape = {{128, 256}, {128, 256}};
    gert::StorageShape scaleShape = {{128, 2}, {128, 2}};
    float minScale = 0.0;
    string roundMode = "rint";
    int64_t rowBlockSize = 1;
    int64_t colBlockSize = 128;
    string expectTilingData = "1201 64 253952 256 0 1 35 1 128 128 256 128 2 1 198 1 25344 64 32 2 4 1 4 1 32 2 0 0 ";

    ExecuteTestCase(ge::DT_BF16, ge::DT_FLOAT8_E5M2, shape, scaleShape, minScale, roundMode, rowBlockSize, colBlockSize, expectTilingData);
}

TEST_F(DynamicBlockQuantTiling, DynamicBlockQuant_tiling_ascendc_bfloat16_fp8e4m3_one_row) {
    gert::StorageShape shape = {{128, 256}, {128, 256}};
    gert::StorageShape scaleShape = {{128, 2}, {128, 2}};
    float minScale = 0.0;
    string roundMode = "rint";
    int64_t rowBlockSize = 1;
    int64_t colBlockSize = 128;
    string expectTilingData = "1211 64 253952 256 0 1 36 1 128 128 256 128 2 1 198 1 25344 64 32 2 4 1 4 1 32 2 0 0 ";

    ExecuteTestCase(ge::DT_BF16, ge::DT_FLOAT8_E4M3FN, shape, scaleShape, minScale, roundMode, rowBlockSize, colBlockSize, expectTilingData);
}

TEST_F(DynamicBlockQuantTiling, DynamicBlockQuant_tiling_ascendc_bfloat16_hifloat8_one_row) {
    gert::StorageShape shape = {{128, 256}, {128, 256}};
    gert::StorageShape scaleShape = {{128, 2}, {128, 2}};
    float minScale = 0.0;
    string roundMode = "round";
    int64_t rowBlockSize = 1;
    int64_t colBlockSize = 128;
    string expectTilingData = "4221 64 253952 256 0 4 34 1 128 128 256 128 2 1 198 1 25344 64 32 2 4 1 4 1 32 2 0 0 ";

    ExecuteTestCase(ge::DT_BF16, ge::DT_HIFLOAT8, shape, scaleShape, minScale, roundMode, rowBlockSize, colBlockSize, expectTilingData);
}

TEST_F(DynamicBlockQuantTiling, DynamicBlockQuant_tiling_ascendc_float16_fp8e5m2_unsupport_round_mode) {
    gert::StorageShape shape = {{128, 256}, {128, 256}};
    gert::StorageShape scaleShape = {{1, 2}, {1, 2}};
    float minScale = 0.0;
    string roundMode = "round";
    int64_t rowBlockSize = 128;
    int64_t colBlockSize = 128;
    string expectTilingData = "64 45 0 0 0 0 0 0 4 137438953512 32 0 0 0 0 204 152 4 2 1 1 4 44 0 128 64 0 0 0 0 ";

    ExecuteTestCase(ge::DT_FLOAT16, ge::DT_FLOAT8_E5M2, shape, scaleShape, minScale, roundMode, rowBlockSize, colBlockSize, expectTilingData,
                    ge::GRAPH_FAILED);
}

TEST_F(DynamicBlockQuantTiling, DynamicBlockQuant_tiling_ascendc_float16_hifloat8_unsupport_round_mode) {
    gert::StorageShape shape = {{128, 256}, {128, 256}};
    gert::StorageShape scaleShape = {{1, 2}, {1, 2}};
    float minScale = 0.0;
    string roundMode = "rint";
    int64_t rowBlockSize = 128;
    int64_t colBlockSize = 128;
    string expectTilingData = "64 45 0 0 0 0 0 0 4 137438953512 32 0 0 0 0 204 152 4 2 1 1 4 44 0 128 64 0 0 0 0 ";

    ExecuteTestCase(ge::DT_FLOAT16, ge::DT_HIFLOAT8, shape, scaleShape, minScale, roundMode, rowBlockSize, colBlockSize, expectTilingData,
                    ge::GRAPH_FAILED);
}

TEST_F(DynamicBlockQuantTiling, DynamicBlockQuant_tiling_ascendc_float16_fp8e5m2_unsupport_rowBlockSize) {
    gert::StorageShape shape = {{128, 256}, {128, 256}};
    gert::StorageShape scaleShape = {{1, 2}, {1, 2}};
    float minScale = 0.0;
    string roundMode = "rint";
    int64_t rowBlockSize = 2;
    int64_t colBlockSize = 128;
    string expectTilingData = "64 45 0 0 0 0 0 0 4 137438953512 32 0 0 0 0 204 152 4 2 1 1 4 44 0 128 64 0 0 0 0 ";

    ExecuteTestCase(ge::DT_FLOAT16, ge::DT_FLOAT8_E5M2, shape, scaleShape, minScale, roundMode, rowBlockSize, colBlockSize, expectTilingData,
                    ge::GRAPH_FAILED);
}

TEST_F(DynamicBlockQuantTiling, DynamicBlockQuant_tiling_ascendc_float16_fp8e5m2_unsupport_colBlockSize) {
    gert::StorageShape shape = {{128, 256}, {128, 256}};
    gert::StorageShape scaleShape = {{1, 2}, {1, 2}};
    float minScale = 0.0;
    string roundMode = "rint";
    int64_t rowBlockSize = 128;
    int64_t colBlockSize = 2;
    string expectTilingData = "64 45 0 0 0 0 0 0 4 137438953512 32 0 0 0 0 204 152 4 2 1 1 4 44 0 128 64 0 0 0 0 ";

    ExecuteTestCase(ge::DT_FLOAT16, ge::DT_FLOAT8_E5M2, shape, scaleShape, minScale, roundMode, rowBlockSize, colBlockSize, expectTilingData,
                    ge::GRAPH_FAILED);
}

TEST_F(DynamicBlockQuantTiling, DynamicBlockQuant_tiling_ascendc_unsupport_input_dtype) {
    gert::StorageShape shape = {{128, 256}, {128, 256}};
    gert::StorageShape scaleShape = {{1, 2}, {1, 2}};
    float minScale = 0.0;
    string roundMode = "rint";
    int64_t rowBlockSize = 128;
    int64_t colBlockSize = 128;
    string expectTilingData = "64 45 0 0 0 0 0 0 4 137438953512 32 0 0 0 0 204 152 4 2 1 1 4 44 0 128 64 0 0 0 0 ";

    ExecuteTestCase(ge::DT_FLOAT, ge::DT_FLOAT8_E5M2, shape, scaleShape, minScale, roundMode, rowBlockSize, colBlockSize, expectTilingData,
                    ge::GRAPH_FAILED);
}

TEST_F(DynamicBlockQuantTiling, DynamicBlockQuant_tiling_ascendc_unsupport_output_dtype) {
    gert::StorageShape shape = {{128, 256}, {128, 256}};
    gert::StorageShape scaleShape = {{1, 2}, {1, 2}};
    float minScale = 0.0;
    string roundMode = "rint";
    int64_t rowBlockSize = 128;
    int64_t colBlockSize = 128;
    string expectTilingData = "64 45 0 0 0 0 0 0 4 137438953512 32 0 0 0 0 204 152 4 2 1 1 4 44 0 128 64 0 0 0 0 ";

    ExecuteTestCase(ge::DT_FLOAT16, ge::DT_FLOAT, shape, scaleShape, minScale, roundMode, rowBlockSize, colBlockSize, expectTilingData,
                    ge::GRAPH_FAILED);
}

TEST_F(DynamicBlockQuantTiling, DynamicBlockQuant_tiling_ascendc_unsupport_dim_num) {
    gert::StorageShape shape = {{128, 256, 128}, {128, 256, 128}};
    gert::StorageShape scaleShape = {{1, 2, 1}, {1, 2, 1}};
    float minScale = 0.0;
    string roundMode = "rint";
    int64_t rowBlockSize = 128;
    int64_t colBlockSize = 128;
    string expectTilingData = "64 45 0 0 0 0 0 0 4 137438953512 32 0 0 0 0 204 152 4 2 1 1 4 44 0 128 64 0 0 0 0 ";

    ExecuteTestCase(ge::DT_FLOAT16, ge::DT_FLOAT, shape, scaleShape, minScale, roundMode, rowBlockSize, colBlockSize, expectTilingData,
                    ge::GRAPH_FAILED);
}

TEST_F(DynamicBlockQuantTiling, DynamicBlockQuant_tiling_ascendc_unsupport_min_scale) {
    gert::StorageShape shape = {{128, 256}, {128, 256}};
    gert::StorageShape scaleShape = {{1, 2}, {1, 2}};
    float minScale = -1.0;
    string roundMode = "rint";
    int64_t rowBlockSize = 128;
    int64_t colBlockSize = 128;
    string expectTilingData = "64 45 0 0 0 0 0 0 4 137438953512 32 0 0 0 0 204 152 4 2 1 1 4 44 0 128 64 0 0 0 0 ";

    ExecuteTestCase(ge::DT_FLOAT16, ge::DT_FLOAT, shape, scaleShape, minScale, roundMode, rowBlockSize, colBlockSize, expectTilingData,
                    ge::GRAPH_FAILED);
}

TEST_F(DynamicBlockQuantTiling, DynamicBlockQuant_tiling_ascendc_float16_unsupport_shape) {
    gert::StorageShape shape = {{128, 256}, {128, 256}};
    gert::StorageShape scaleShape = {{1, 1}, {1, 1}};
    float minScale = 0.0;
    string roundMode = "rint";
    int64_t rowBlockSize = 128;
    int64_t colBlockSize = 128;
    string expectTilingData = "1100 64 253952 256 0 1 35 128 128 128 256 1 2 1 1 128 128 2 1 2 1 1 1 1 1 2 0 0 ";

    ExecuteTestCase(ge::DT_FLOAT16, ge::DT_FLOAT8_E5M2, shape, scaleShape, minScale, roundMode, rowBlockSize, colBlockSize,
                    expectTilingData, ge::GRAPH_FAILED);
}

TEST_F(DynamicBlockQuantTiling, DynamicBlockQuant_tiling_ascendc_bfloat16_1) {
    gert::StorageShape shape = {{40, 256}, {40, 256}};
    gert::StorageShape scaleShape = {{40, 2}, {40, 2}};
    float minScale = 0.0;
    string roundMode = "rint";
    int64_t rowBlockSize = 1;
    int64_t colBlockSize = 128;
    string expectTilingData = "10 48 196352 0 0 1 2 1 128 40 256 0 0 0 0 0 0 48 0 0 0 0 0 0 0 0 0 0 0 ";

    ExecuteTestCaseA2(ge::DT_BF16, ge::DT_INT8, shape, scaleShape, minScale, roundMode, rowBlockSize, colBlockSize, expectTilingData);
}

TEST_F(DynamicBlockQuantTiling, DynamicBlockQuant_tiling_ascendc_fploat16_1) {
    gert::StorageShape shape = {{40, 256}, {40, 256}};
    gert::StorageShape scaleShape = {{40, 2}, {40, 2}};
    float minScale = 0.0;
    string roundMode = "rint";
    int64_t rowBlockSize = 1;
    int64_t colBlockSize = 128;
    string expectTilingData = "0 48 196352 0 0 1 2 1 128 40 256 0 0 0 0 0 0 48 0 0 0 0 0 0 0 0 0 0 0 ";

    ExecuteTestCaseA2(ge::DT_FLOAT16, ge::DT_INT8, shape, scaleShape, minScale, roundMode, rowBlockSize, colBlockSize, expectTilingData);
}
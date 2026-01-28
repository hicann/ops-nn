/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_grouped_dynamic_block_quant_tiling.cpp
 * \brief
 */

#include <iostream>
#include <vector>
#include <string_view>

#include <gtest/gtest.h>
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "platform/platform_infos_def.h"
#include "ut_op_util.h"
#include "quant/grouped_dynamic_block_quant/op_host/grouped_dynamic_block_quant_tiling.h"
#include "kernel_run_context_facker.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "test_cube_util.h"

using namespace std;
using namespace ge;
using namespace ut_util;

class GroupedDynamicBlockQuantTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "GroupedDynamicBlockQuantTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "GroupedDynamicBlockQuantTiling TearDown" << std::endl;
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
    ge::DataType inDtype, ge::DataType outDtype, gert::StorageShape shape, gert::StorageShape listShape,
    gert::StorageShape scaleShape, float minScale, string roundMode, int64_t rowBlockSize, int64_t colBlockSize,
    int64_t groupListType, string expectTilingData, ge::graphStatus status = ge::GRAPH_SUCCESS)
{
    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, 
                          "Intrinsic_data_move_l12ub": true, 
                          "Intrinsic_data_move_l0c2ub": true, 
                          "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 253952, 
                          "L2_SIZE": 33554432, 
                          "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, 
                          "L0B_SIZE": 65536, 
                          "L0C_SIZE": 131072,
                          "CORE_NUM": 64}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;

    GetPlatFormInfos(compile_info_string.data(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();

    // compile info
    optiling::GroupedDynamicBlockQuantCompileInfo compile_info;

    std::string op_type("GroupedDynamicBlockQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 2)
            .Inputs({compile_info_string.data(), reinterpret_cast<void*>(&platform_info)})
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

    string_view sv(
        reinterpret_cast<const char*>(&platform_info),
        sizeof(platform_info)
    );
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(2, 2)
                      .IrInstanceNum({2})
                      .InputShapes({&shape, &listShape})
                      .OutputShapes({&shape, &scaleShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(sv.data())
                      .NodeInputTd(0, inDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, outDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"min_scale", Ops::NN::AnyValue::CreateFrom<float>(minScale)},
                           {"round_mode", Ops::NN::AnyValue::CreateFrom<string>(roundMode)},
                           {"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(outDtype)},
                           {"row_block_size", Ops::NN::AnyValue::CreateFrom(rowBlockSize)},
                           {"col_block_size", Ops::NN::AnyValue::CreateFrom(colBlockSize)},
                           {"group_list_type", Ops::NN::AnyValue::CreateFrom(groupListType)}})
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

    auto tiling_data_result = to_string<int64_t>(raw_tiling_data->GetData(), raw_tiling_data->GetDataSize());
    EXPECT_EQ(tiling_data_result, expectTilingData);
}

TEST_F(GroupedDynamicBlockQuantTiling, GroupedDynamicBlockQuant_tiling_small_block_float16_fp8e5m2)
{
    ge::DataType inDtype = ge::DT_FLOAT16;
    ge::DataType outDtype = ge::DT_FLOAT8_E5M2;
    gert::StorageShape shape = {{128, 256}, {128, 256}};
    gert::StorageShape listShape = {{1}, {1}};
    gert::StorageShape scaleShape = {{2, 2}, {2, 2}};
    float minScale = 0.0;
    string roundMode = "rint";
    int64_t rowBlockSize = 128;
    int64_t colBlockSize = 128;
    int64_t group_list_type = 0;
    string expectTilingData = "1121 64 253952 256 2 2 0 0 1 35 128 128 1 128 256 2 2 2 1 128 128 1 0 256 ";
    ge::graphStatus status = ge::GRAPH_SUCCESS;

    ExecuteTestCase(
        inDtype, outDtype, shape, listShape, scaleShape, minScale, roundMode, rowBlockSize, colBlockSize,
        group_list_type, expectTilingData, status);
}

TEST_F(GroupedDynamicBlockQuantTiling, GroupedDynamicBlockQuant_tiling_small_block_float16_fp8e4m3fn)
{
    ge::DataType inDtype = ge::DT_FLOAT16;
    ge::DataType outDtype = ge::DT_FLOAT8_E4M3FN;
    gert::StorageShape shape = {{128, 256}, {128, 256}};
    gert::StorageShape listShape = {{1}, {1}};
    gert::StorageShape scaleShape = {{2, 2}, {2, 2}};
    float minScale = 0.0;
    string roundMode = "rint";
    int64_t rowBlockSize = 128;
    int64_t colBlockSize = 128;
    int64_t group_list_type = 0;
    string expectTilingData = "1111 64 253952 256 2 2 0 0 1 36 128 128 1 128 256 2 2 2 1 128 128 1 0 256 ";
    ge::graphStatus status = ge::GRAPH_SUCCESS;

    ExecuteTestCase(
        inDtype, outDtype, shape, listShape, scaleShape, minScale, roundMode, rowBlockSize, colBlockSize,
        group_list_type, expectTilingData, status);
}

TEST_F(GroupedDynamicBlockQuantTiling, GroupedDynamicBlockQuant_tiling_small_block_float16_hifloat8_round)
{
    ge::DataType inDtype = ge::DT_FLOAT16;
    ge::DataType outDtype = ge::DT_HIFLOAT8;
    gert::StorageShape shape = {{128, 256}, {128, 256}};
    gert::StorageShape listShape = {{1}, {1}};
    gert::StorageShape scaleShape = {{2, 2}, {2, 2}};
    float minScale = 0.0;
    string roundMode = "round";
    int64_t rowBlockSize = 128;
    int64_t colBlockSize = 128;
    int64_t group_list_type = 0;
    string expectTilingData = "1134 64 253952 256 2 2 0 0 4 34 128 128 1 128 256 2 2 2 1 128 128 1 0 256 ";
    ge::graphStatus status = ge::GRAPH_SUCCESS;

    ExecuteTestCase(
        inDtype, outDtype, shape, listShape, scaleShape, minScale, roundMode, rowBlockSize, colBlockSize,
        group_list_type, expectTilingData, status);
}

TEST_F(GroupedDynamicBlockQuantTiling, GroupedDynamicBlockQuant_tiling_small_block_float16_hifloat8_hybrid)
{
    ge::DataType inDtype = ge::DT_FLOAT16;
    ge::DataType outDtype = ge::DT_HIFLOAT8;
    gert::StorageShape shape = {{128, 256}, {128, 256}};
    gert::StorageShape listShape = {{1}, {1}};
    gert::StorageShape scaleShape = {{2, 2}, {2, 2}};
    float minScale = 0.0;
    string roundMode = "hybrid";
    int64_t rowBlockSize = 128;
    int64_t colBlockSize = 128;
    int64_t group_list_type = 0;
    string expectTilingData = "1137 64 253952 256 2 2 0 0 7 34 128 128 1 128 256 2 2 2 1 128 128 1 0 256 ";
    ge::graphStatus status = ge::GRAPH_SUCCESS;

    ExecuteTestCase(
        inDtype, outDtype, shape, listShape, scaleShape, minScale, roundMode, rowBlockSize, colBlockSize,
        group_list_type, expectTilingData, status);
}

TEST_F(GroupedDynamicBlockQuantTiling, GroupedDynamicBlockQuant_tiling_small_block_bfloat16_fp8e5m2)
{
    ge::DataType inDtype = ge::DT_BF16;
    ge::DataType outDtype = ge::DT_FLOAT8_E5M2;
    gert::StorageShape shape = {{128, 256}, {128, 256}};
    gert::StorageShape listShape = {{1}, {1}};
    gert::StorageShape scaleShape = {{2, 2}, {2, 2}};
    float minScale = 0.0;
    string roundMode = "rint";
    int64_t rowBlockSize = 128;
    int64_t colBlockSize = 128;
    int64_t group_list_type = 0;
    string expectTilingData = "1221 64 253952 256 2 2 0 0 1 35 128 128 1 128 256 2 2 2 1 128 128 1 0 256 ";
    ge::graphStatus status = ge::GRAPH_SUCCESS;

    ExecuteTestCase(
        inDtype, outDtype, shape, listShape, scaleShape, minScale, roundMode, rowBlockSize, colBlockSize,
        group_list_type, expectTilingData, status);
}

TEST_F(GroupedDynamicBlockQuantTiling, GroupedDynamicBlockQuant_tiling_small_block_bfloat16_fp8e4m3fn)
{
    ge::DataType inDtype = ge::DT_BF16;
    ge::DataType outDtype = ge::DT_FLOAT8_E4M3FN;
    gert::StorageShape shape = {{128, 256}, {128, 256}};
    gert::StorageShape listShape = {{1}, {1}};
    gert::StorageShape scaleShape = {{2, 2}, {2, 2}};
    float minScale = 0.0;
    string roundMode = "rint";
    int64_t rowBlockSize = 128;
    int64_t colBlockSize = 128;
    int64_t group_list_type = 0;
    string expectTilingData = "1211 64 253952 256 2 2 0 0 1 36 128 128 1 128 256 2 2 2 1 128 128 1 0 256 ";
    ge::graphStatus status = ge::GRAPH_SUCCESS;

    ExecuteTestCase(
        inDtype, outDtype, shape, listShape, scaleShape, minScale, roundMode, rowBlockSize, colBlockSize,
        group_list_type, expectTilingData, status);
}

TEST_F(GroupedDynamicBlockQuantTiling, GroupedDynamicBlockQuant_tiling_small_block_bfloat16_hifloat8_round)
{
    ge::DataType inDtype = ge::DT_BF16;
    ge::DataType outDtype = ge::DT_HIFLOAT8;
    gert::StorageShape shape = {{128, 256}, {128, 256}};
    gert::StorageShape listShape = {{1}, {1}};
    gert::StorageShape scaleShape = {{2, 2}, {2, 2}};
    float minScale = 0.0;
    string roundMode = "round";
    int64_t rowBlockSize = 128;
    int64_t colBlockSize = 128;
    int64_t group_list_type = 0;
    string expectTilingData = "1234 64 253952 256 2 2 0 0 4 34 128 128 1 128 256 2 2 2 1 128 128 1 0 256 ";
    ge::graphStatus status = ge::GRAPH_SUCCESS;

    ExecuteTestCase(
        inDtype, outDtype, shape, listShape, scaleShape, minScale, roundMode, rowBlockSize, colBlockSize,
        group_list_type, expectTilingData, status);
}

TEST_F(GroupedDynamicBlockQuantTiling, GroupedDynamicBlockQuant_tiling_small_block_bfloat16_hifloat8_hybrid)
{
    ge::DataType inDtype = ge::DT_BF16;
    ge::DataType outDtype = ge::DT_HIFLOAT8;
    gert::StorageShape shape = {{128, 256}, {128, 256}};
    gert::StorageShape listShape = {{1}, {1}};
    gert::StorageShape scaleShape = {{2, 2}, {2, 2}};
    float minScale = 0.0;
    string roundMode = "hybrid";
    int64_t rowBlockSize = 128;
    int64_t colBlockSize = 128;
    int64_t group_list_type = 0;
    string expectTilingData = "1237 64 253952 256 2 2 0 0 7 34 128 128 1 128 256 2 2 2 1 128 128 1 0 256 ";
    ge::graphStatus status = ge::GRAPH_SUCCESS;

    ExecuteTestCase(
        inDtype, outDtype, shape, listShape, scaleShape, minScale, roundMode, rowBlockSize, colBlockSize,
        group_list_type, expectTilingData, status);
}

// =============================================================================================================
TEST_F(GroupedDynamicBlockQuantTiling, GroupedDynamicBlockQuant_tiling_large_block_float16_fp8e5m2)
{
    ge::DataType inDtype = ge::DT_FLOAT16;
    ge::DataType outDtype = ge::DT_FLOAT8_E5M2;
    gert::StorageShape shape = {{8192, 256}, {8192, 256}};
    gert::StorageShape listShape = {{8}, {8}};
    gert::StorageShape scaleShape = {{40, 1}, {40, 1}};
    float minScale = 0.0;
    string roundMode = "rint";
    int64_t rowBlockSize = 256;
    int64_t colBlockSize = 256;
    int64_t group_list_type = 0;
    string expectTilingData = "2121 64 253952 256 8 8 0 0 1 35 256 256 1 8192 256 40 1 1 8 0 256 1 0 165 ";
    ge::graphStatus status = ge::GRAPH_SUCCESS;

    ExecuteTestCase(
        inDtype, outDtype, shape, listShape, scaleShape, minScale, roundMode, rowBlockSize, colBlockSize,
        group_list_type, expectTilingData, status);
}

TEST_F(GroupedDynamicBlockQuantTiling, GroupedDynamicBlockQuant_tiling_large_block_float16_fp8e4m3fn)
{
    ge::DataType inDtype = ge::DT_FLOAT16;
    ge::DataType outDtype = ge::DT_FLOAT8_E4M3FN;
    gert::StorageShape shape = {{8192, 256}, {8192, 256}};
    gert::StorageShape listShape = {{8}, {8}};
    gert::StorageShape scaleShape = {{40, 1}, {40, 1}};
    float minScale = 0.0;
    string roundMode = "rint";
    int64_t rowBlockSize = 256;
    int64_t colBlockSize = 256;
    int64_t group_list_type = 0;
    string expectTilingData = "2111 64 253952 256 8 8 0 0 1 36 256 256 1 8192 256 40 1 1 8 0 256 1 0 165 ";
    ge::graphStatus status = ge::GRAPH_SUCCESS;

    ExecuteTestCase(
        inDtype, outDtype, shape, listShape, scaleShape, minScale, roundMode, rowBlockSize, colBlockSize,
        group_list_type, expectTilingData, status);
}

TEST_F(GroupedDynamicBlockQuantTiling, GroupedDynamicBlockQuant_tiling_large_block_float16_hifloat8_round)
{
    ge::DataType inDtype = ge::DT_FLOAT16;
    ge::DataType outDtype = ge::DT_HIFLOAT8;
    gert::StorageShape shape = {{8192, 256}, {8192, 256}};
    gert::StorageShape listShape = {{8}, {8}};
    gert::StorageShape scaleShape = {{40, 1}, {40, 1}};
    float minScale = 0.0;
    string roundMode = "round";
    int64_t rowBlockSize = 256;
    int64_t colBlockSize = 256;
    int64_t group_list_type = 0;
    string expectTilingData = "2134 64 253952 256 8 8 0 0 4 34 256 256 1 8192 256 40 1 1 8 0 256 1 0 165 ";
    ge::graphStatus status = ge::GRAPH_SUCCESS;

    ExecuteTestCase(
        inDtype, outDtype, shape, listShape, scaleShape, minScale, roundMode, rowBlockSize, colBlockSize,
        group_list_type, expectTilingData, status);
}

TEST_F(GroupedDynamicBlockQuantTiling, GroupedDynamicBlockQuant_tiling_large_block_float16_hifloat8_hybrid)
{
    ge::DataType inDtype = ge::DT_FLOAT16;
    ge::DataType outDtype = ge::DT_HIFLOAT8;
    gert::StorageShape shape = {{8192, 256}, {8192, 256}};
    gert::StorageShape listShape = {{8}, {8}};
    gert::StorageShape scaleShape = {{40, 1}, {40, 1}};
    float minScale = 0.0;
    string roundMode = "hybrid";
    int64_t rowBlockSize = 256;
    int64_t colBlockSize = 256;
    int64_t group_list_type = 0;
    string expectTilingData = "2137 64 253952 256 8 8 0 0 7 34 256 256 1 8192 256 40 1 1 8 0 256 1 0 165 ";
    ge::graphStatus status = ge::GRAPH_SUCCESS;

    ExecuteTestCase(
        inDtype, outDtype, shape, listShape, scaleShape, minScale, roundMode, rowBlockSize, colBlockSize,
        group_list_type, expectTilingData, status);
}

TEST_F(GroupedDynamicBlockQuantTiling, GroupedDynamicBlockQuant_tiling_large_block_bfloat16_fp8e5m2)
{
    ge::DataType inDtype = ge::DT_BF16;
    ge::DataType outDtype = ge::DT_FLOAT8_E5M2;
    gert::StorageShape shape = {{8192, 256}, {8192, 256}};
    gert::StorageShape listShape = {{8}, {8}};
    gert::StorageShape scaleShape = {{40, 1}, {40, 1}};
    float minScale = 0.0;
    string roundMode = "rint";
    int64_t rowBlockSize = 256;
    int64_t colBlockSize = 256;
    int64_t group_list_type = 0;
    string expectTilingData = "2221 64 253952 256 8 8 0 0 1 35 256 256 1 8192 256 40 1 1 8 0 256 1 0 165 ";
    ge::graphStatus status = ge::GRAPH_SUCCESS;

    ExecuteTestCase(
        inDtype, outDtype, shape, listShape, scaleShape, minScale, roundMode, rowBlockSize, colBlockSize,
        group_list_type, expectTilingData, status);
}

TEST_F(GroupedDynamicBlockQuantTiling, GroupedDynamicBlockQuant_tiling_large_block_bfloat16_fp8e4m3fn)
{
    ge::DataType inDtype = ge::DT_BF16;
    ge::DataType outDtype = ge::DT_FLOAT8_E4M3FN;
    gert::StorageShape shape = {{8192, 256}, {8192, 256}};
    gert::StorageShape listShape = {{8}, {8}};
    gert::StorageShape scaleShape = {{40, 1}, {40, 1}};
    float minScale = 0.0;
    string roundMode = "rint";
    int64_t rowBlockSize = 256;
    int64_t colBlockSize = 256;
    int64_t group_list_type = 0;
    string expectTilingData = "2211 64 253952 256 8 8 0 0 1 36 256 256 1 8192 256 40 1 1 8 0 256 1 0 165 ";
    ge::graphStatus status = ge::GRAPH_SUCCESS;

    ExecuteTestCase(
        inDtype, outDtype, shape, listShape, scaleShape, minScale, roundMode, rowBlockSize, colBlockSize,
        group_list_type, expectTilingData, status);
}

TEST_F(GroupedDynamicBlockQuantTiling, GroupedDynamicBlockQuant_tiling_large_block_bfloat16_hifloat8_round)
{
    ge::DataType inDtype = ge::DT_BF16;
    ge::DataType outDtype = ge::DT_HIFLOAT8;
    gert::StorageShape shape = {{8192, 256}, {8192, 256}};
    gert::StorageShape listShape = {{8}, {8}};
    gert::StorageShape scaleShape = {{40, 1}, {40, 1}};
    float minScale = 0.0;
    string roundMode = "round";
    int64_t rowBlockSize = 256;
    int64_t colBlockSize = 256;
    int64_t group_list_type = 0;
    string expectTilingData = "2234 64 253952 256 8 8 0 0 4 34 256 256 1 8192 256 40 1 1 8 0 256 1 0 165 ";
    ge::graphStatus status = ge::GRAPH_SUCCESS;

    ExecuteTestCase(
        inDtype, outDtype, shape, listShape, scaleShape, minScale, roundMode, rowBlockSize, colBlockSize,
        group_list_type, expectTilingData, status);
}

TEST_F(GroupedDynamicBlockQuantTiling, GroupedDynamicBlockQuant_tiling_large_block_bfloat16_hifloat8_hybrid)
{
    ge::DataType inDtype = ge::DT_BF16;
    ge::DataType outDtype = ge::DT_HIFLOAT8;
    gert::StorageShape shape = {{8192, 256}, {8192, 256}};
    gert::StorageShape listShape = {{8}, {8}};
    gert::StorageShape scaleShape = {{40, 1}, {40, 1}};
    float minScale = 0.0;
    string roundMode = "hybrid";
    int64_t rowBlockSize = 256;
    int64_t colBlockSize = 256;
    int64_t group_list_type = 0;
    string expectTilingData = "2237 64 253952 256 8 8 0 0 7 34 256 256 1 8192 256 40 1 1 8 0 256 1 0 165 ";
    ge::graphStatus status = ge::GRAPH_SUCCESS;

    ExecuteTestCase(
        inDtype, outDtype, shape, listShape, scaleShape, minScale, roundMode, rowBlockSize, colBlockSize,
        group_list_type, expectTilingData, status);
}

// =============================================================================================================
TEST_F(GroupedDynamicBlockQuantTiling, GroupedDynamicBlockQuant_tiling_small_block_float16_fp8e5m2_fail_scale_shape)
{
    ge::DataType inDtype = ge::DT_FLOAT16;
    ge::DataType outDtype = ge::DT_FLOAT8_E5M2;
    gert::StorageShape shape = {{128, 256}, {128, 256}};
    gert::StorageShape listShape = {{1}, {1}};
    gert::StorageShape scaleShape = {{1, 2}, {1, 2}};
    float minScale = 0.0;
    string roundMode = "rint";
    int64_t rowBlockSize = 128;
    int64_t colBlockSize = 128;
    int64_t group_list_type = 0;
    string expectTilingData = "1121 64 253952 256 2 2 0 0 1 35 128 128 1 128 256 2 2 2 1 128 128 1 0 128 ";
    ge::graphStatus status = ge::GRAPH_FAILED;

    ExecuteTestCase(
        inDtype, outDtype, shape, listShape, scaleShape, minScale, roundMode, rowBlockSize, colBlockSize,
        group_list_type, expectTilingData, status);
}

TEST_F(GroupedDynamicBlockQuantTiling, GroupedDynamicBlockQuant_tiling_small_block_float16_fp8e5m2_fail_row_block_size)
{
    ge::DataType inDtype = ge::DT_FLOAT16;
    ge::DataType outDtype = ge::DT_FLOAT8_E5M2;
    gert::StorageShape shape = {{128, 256}, {128, 256}};
    gert::StorageShape listShape = {{1}, {1}};
    gert::StorageShape scaleShape = {{2, 2}, {2, 2}};
    float minScale = 0.0;
    string roundMode = "rint";
    int64_t rowBlockSize = 32;
    int64_t colBlockSize = 128;
    int64_t group_list_type = 0;
    string expectTilingData = "1121 64 253952 256 2 2 0 0 1 35 128 128 1 128 256 2 2 2 1 128 128 1 0 128 ";
    ge::graphStatus status = ge::GRAPH_FAILED;

    ExecuteTestCase(
        inDtype, outDtype, shape, listShape, scaleShape, minScale, roundMode, rowBlockSize, colBlockSize,
        group_list_type, expectTilingData, status);
}

TEST_F(GroupedDynamicBlockQuantTiling, GroupedDynamicBlockQuant_tiling_small_block_float16_fp8e5m2_fail_col_block_size)
{
    ge::DataType inDtype = ge::DT_FLOAT16;
    ge::DataType outDtype = ge::DT_FLOAT8_E5M2;
    gert::StorageShape shape = {{128, 256}, {128, 256}};
    gert::StorageShape listShape = {{1}, {1}};
    gert::StorageShape scaleShape = {{2, 2}, {2, 2}};
    float minScale = 0.0;
    string roundMode = "rint";
    int64_t rowBlockSize = 128;
    int64_t colBlockSize = 32;
    int64_t group_list_type = 0;
    string expectTilingData = "1121 64 253952 256 2 2 0 0 1 35 128 128 1 128 256 2 2 2 1 128 128 1 0 128 ";
    ge::graphStatus status = ge::GRAPH_FAILED;

    ExecuteTestCase(
        inDtype, outDtype, shape, listShape, scaleShape, minScale, roundMode, rowBlockSize, colBlockSize,
        group_list_type, expectTilingData, status);
}

TEST_F(GroupedDynamicBlockQuantTiling, GroupedDynamicBlockQuant_tiling_small_block_float16_fp8e5m2_fail_round_mode)
{
    ge::DataType inDtype = ge::DT_FLOAT16;
    ge::DataType outDtype = ge::DT_FLOAT8_E5M2;
    gert::StorageShape shape = {{128, 256}, {128, 256}};
    gert::StorageShape listShape = {{1}, {1}};
    gert::StorageShape scaleShape = {{2, 2}, {2, 2}};
    float minScale = 0.0;
    string roundMode = "round";
    int64_t rowBlockSize = 128;
    int64_t colBlockSize = 128;
    int64_t group_list_type = 0;
    string expectTilingData = "1124 64 253952 256 2 2 0 0 4 35 128 128 1 128 256 2 2 2 1 128 128 1 0 128 ";
    ge::graphStatus status = ge::GRAPH_FAILED;

    ExecuteTestCase(
        inDtype, outDtype, shape, listShape, scaleShape, minScale, roundMode, rowBlockSize, colBlockSize,
        group_list_type, expectTilingData, status);
}

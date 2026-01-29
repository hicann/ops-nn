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
 * \file test_quant_update_scatter_regbase_tiling.cpp
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
#include "../../../../op_host/arch35/quant_update_scatter_tiling_arch35.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"

using namespace std;
using namespace ge;
using namespace ut_util;

class QuantUpdateScatterRegBaseTiling : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "QuantUpdateScatterRegBaseTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "QuantUpdateScatterRegBaseTiling TearDown" << std::endl;
    }
};

static void ExecuteTestCase(
    std::vector<ge::DataType> dtypes, gert::StorageShape varShape, gert::StorageShape indicesShape,
    gert::StorageShape updatesShape, gert::StorageShape scalesShape, gert::StorageShape zeroPointsShape,
    gert::StorageShape outShape, string reduce, int64_t axis, int64_t quant_axis, bool reciprocal_scale,
    string round_mode, ge::graphStatus status = ge::GRAPH_SUCCESS)
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
    map<string, string> socversions = {{"Short_SoC_version", "Ascend950"}};
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();

    // compile info
    optiling::QuantUpdateScatterCompileInfo compile_info;

    std::string op_type("QuantUpdateScatter");
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
                      .NodeIoNum(5, 1)
                      .IrInstanceNum({1, 1, 1, 1, 1})
                      .InputShapes({&varShape, &indicesShape, &updatesShape, &scalesShape, &zeroPointsShape})
                      .OutputShapes({&outShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, dtypes[0], ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, dtypes[1], ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, dtypes[2], ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, dtypes[3], ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, dtypes[4], ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, dtypes[5], ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"reduce", Ops::NN::AnyValue::CreateFrom<string>(reduce)},
                           {"axis", Ops::NN::AnyValue::CreateFrom<int64_t>(axis)},
                           {"quant_axis", Ops::NN::AnyValue::CreateFrom<int64_t>(quant_axis)},
                           {"reciprocal_scale", Ops::NN::AnyValue::CreateFrom<bool>(reciprocal_scale)},
                           {"round_mode", Ops::NN::AnyValue::CreateFrom<string>(round_mode)}})
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
}

TEST_F(QuantUpdateScatterRegBaseTiling, QuantUpdateScatter1)
{
    gert::StorageShape varShape = {{24, 4096, 128}, {24, 4096, 128}};
    gert::StorageShape indicesShape = {{24}, {24}};
    gert::StorageShape updatesShape = {{24, 1, 128}, {24, 1, 128}};
    gert::StorageShape scalesShape = {{128}, {128}};
    gert::StorageShape zeroPointsShape = {{128}, {128}};
    gert::StorageShape outShape = {{24, 4096, 128}, {24, 4096, 128}};
    int64_t axis = -2;
    string reduce = "update";
    int64_t quant_axis = -1;
    bool reciprocal_scale = false;
    string round_mode = "rint";

    ExecuteTestCase(
        {ge::DT_INT8, ge::DT_INT32, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_INT8}, varShape, indicesShape,
        updatesShape, scalesShape, zeroPointsShape, outShape, reduce, axis, quant_axis, reciprocal_scale, round_mode,
        ge::GRAPH_SUCCESS);
}

TEST_F(QuantUpdateScatterRegBaseTiling, QuantUpdateScatter2)
{
    gert::StorageShape varShape = {{24, 4096, 2}, {24, 4096, 2}};
    gert::StorageShape indicesShape = {{24}, {24}};
    gert::StorageShape updatesShape = {{24, 1, 2}, {24, 1, 2}};
    gert::StorageShape scalesShape = {{2}, {2}};
    gert::StorageShape zeroPointsShape = {{2}, {2}};
    gert::StorageShape outShape = {{24, 4096, 2}, {24, 4096, 2}};
    int64_t axis = -2;
    string reduce = "update";
    int64_t quant_axis = -1;
    bool reciprocal_scale = false;
    string round_mode = "rint";

    ExecuteTestCase(
        {ge::DT_INT8, ge::DT_INT32, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_INT8}, varShape, indicesShape,
        updatesShape, scalesShape, zeroPointsShape, outShape, reduce, axis, quant_axis, reciprocal_scale, round_mode,
        ge::GRAPH_SUCCESS);
}

TEST_F(QuantUpdateScatterRegBaseTiling, QuantUpdateScatter3)
{
    gert::StorageShape varShape = {{24, 4096, 128}, {24, 4096, 128}};
    gert::StorageShape indicesShape = {{24}, {24}};
    gert::StorageShape updatesShape = {{24, 1, 128}, {24, 1, 128}};
    gert::StorageShape scalesShape = {{128}, {128}};
    gert::StorageShape zeroPointsShape = {{128}, {128}};
    gert::StorageShape outShape = {{24, 4096, 128}, {24, 4096, 128}};
    int64_t axis = -2;
    string reduce = "update";
    int64_t quant_axis = -1;
    bool reciprocal_scale = false;
    string round_mode = "rint";

    ExecuteTestCase(
        {ge::DT_INT8, ge::DT_INT32, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_INT8}, varShape, indicesShape,
        updatesShape, scalesShape, zeroPointsShape, outShape, reduce, axis, quant_axis, reciprocal_scale, round_mode,
        ge::GRAPH_SUCCESS);
}

TEST_F(QuantUpdateScatterRegBaseTiling, QuantUpdateScatter4)
{
    gert::StorageShape varShape = {{24, 4096, 2}, {24, 4096, 2}};
    gert::StorageShape indicesShape = {{24}, {24}};
    gert::StorageShape updatesShape = {{24, 1, 2}, {24, 1, 2}};
    gert::StorageShape scalesShape = {{2}, {2}};
    gert::StorageShape zeroPointsShape = {{2}, {2}};
    gert::StorageShape outShape = {{24, 4096, 2}, {24, 4096, 2}};
    int64_t axis = -2;
    string reduce = "update";
    int64_t quant_axis = -1;
    bool reciprocal_scale = false;
    string round_mode = "rint";
    ExecuteTestCase(
        {ge::DT_INT8, ge::DT_INT32, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_INT8}, varShape, indicesShape,
        updatesShape, scalesShape, zeroPointsShape, outShape, reduce, axis, quant_axis, reciprocal_scale, round_mode,
        ge::GRAPH_SUCCESS);
}

TEST_F(QuantUpdateScatterRegBaseTiling, QuantUpdateScatter5)
{
    gert::StorageShape varShape = {{24, 4096, 128}, {24, 4096, 128}};
    gert::StorageShape indicesShape = {{1, 2}, {1, 2}};
    gert::StorageShape updatesShape = {{1, 1024, 128}, {1, 1024, 128}};
    gert::StorageShape scalesShape = {{128}, {128}};
    gert::StorageShape zeroPointsShape = {{128}, {128}};
    gert::StorageShape outShape = {{24, 4096, 128}, {24, 4096, 128}};
    int64_t axis = -2;
    string reduce = "update";
    int64_t quant_axis = -1;
    bool reciprocal_scale = false;
    string round_mode = "rint";
    ExecuteTestCase(
        {ge::DT_INT8, ge::DT_INT32, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_INT8}, varShape, indicesShape,
        updatesShape, scalesShape, zeroPointsShape, outShape, reduce, axis, quant_axis, reciprocal_scale, round_mode,
        ge::GRAPH_SUCCESS);
}

TEST_F(QuantUpdateScatterRegBaseTiling, QuantUpdateScatter6)
{
    gert::StorageShape varShape = {{24, 4096, 2}, {24, 4096, 2}};
    gert::StorageShape indicesShape = {{1, 2}, {1, 2}};
    gert::StorageShape updatesShape = {{1, 1024, 2}, {1, 1024, 2}};
    gert::StorageShape scalesShape = {{2}, {2}};
    gert::StorageShape zeroPointsShape = {{2}, {2}};
    gert::StorageShape outShape = {{24, 4096, 2}, {24, 4096, 2}};
    int64_t axis = -2;
    string reduce = "update";
    int64_t quant_axis = -1;
    bool reciprocal_scale = false;
    string round_mode = "rint";
    ExecuteTestCase(
        {ge::DT_INT8, ge::DT_INT32, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_INT8}, varShape, indicesShape,
        updatesShape, scalesShape, zeroPointsShape, outShape, reduce, axis, quant_axis, reciprocal_scale, round_mode,
        ge::GRAPH_SUCCESS);
}

TEST_F(QuantUpdateScatterRegBaseTiling, QuantUpdateScatter7)
{
    gert::StorageShape varShape = {{128, 48, 2048}, {128, 48, 2048}};
    gert::StorageShape indicesShape = {{128}, {128}};
    gert::StorageShape updatesShape = {{128, 24, 2048}, {128, 24, 2048}};
    gert::StorageShape scalesShape = {{2048}, {2048}};
    gert::StorageShape zeroPointsShape = {{2048}, {2048}};
    gert::StorageShape outShape = {{128, 48, 2048}, {128, 48, 2048}};
    int64_t axis = -2;
    string reduce = "update";
    int64_t quant_axis = -1;

    bool reciprocal_scale = false;
    string round_mode = "rint";
    ExecuteTestCase(
        {ge::DT_INT8, ge::DT_INT32, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_INT8}, varShape, indicesShape,
        updatesShape, scalesShape, zeroPointsShape, outShape, reduce, axis, quant_axis, reciprocal_scale, round_mode,
        ge::GRAPH_SUCCESS);
}

TEST_F(QuantUpdateScatterRegBaseTiling, QuantUpdateScatter8)
{
    gert::StorageShape varShape = {{128, 48, 2048}, {128, 48, 2048}};
    gert::StorageShape indicesShape = {{128}, {128}};
    gert::StorageShape updatesShape = {{128, 24, 2048}, {128, 24, 2048}};
    gert::StorageShape scalesShape = {{2048}, {2048}};
    gert::StorageShape zeroPointsShape = {{2048}, {2048}};
    gert::StorageShape outShape = {{128, 48, 2048}, {128, 48, 2048}};
    int64_t axis = -2;
    string reduce = "update";
    int64_t quant_axis = -1;

    bool reciprocal_scale = false;
    string round_mode = "rint";

    ExecuteTestCase(
        {ge::DT_INT8, ge::DT_INT32, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8}, varShape, indicesShape,
        updatesShape, scalesShape, zeroPointsShape, outShape, reduce, axis, quant_axis, reciprocal_scale, round_mode,
        ge::GRAPH_SUCCESS);
}

TEST_F(QuantUpdateScatterRegBaseTiling, QuantUpdateScatter9)
{
    gert::StorageShape varShape = {{24, 4096, 81920}, {24, 4096, 81920}};
    gert::StorageShape indicesShape = {{24}, {24}};
    gert::StorageShape updatesShape = {{24, 1, 81920}, {24, 1, 81920}};
    gert::StorageShape scalesShape = {{81920}, {81920}};
    gert::StorageShape zeroPointsShape = {{81920}, {81920}};
    gert::StorageShape outShape = {{24, 4096, 81920}, {24, 4096, 81920}};
    int64_t axis = -2;
    string reduce = "update";
    int64_t quant_axis = -1;

    bool reciprocal_scale = false;
    string round_mode = "rint";

    ExecuteTestCase(
        {ge::DT_INT8, ge::DT_INT32, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8}, varShape, indicesShape,
        updatesShape, scalesShape, zeroPointsShape, outShape, reduce, axis, quant_axis, reciprocal_scale, round_mode,
        ge::GRAPH_SUCCESS);
}

TEST_F(QuantUpdateScatterRegBaseTiling, QuantUpdateScatter_empty_tensor)
{
    gert::StorageShape varShape = {{128, 48, 0}, {128, 48, 0}};
    gert::StorageShape indicesShape = {{128}, {128}};
    gert::StorageShape updatesShape = {{128, 24, 2048}, {128, 24, 2048}};
    gert::StorageShape scalesShape = {{2048}, {2048}};
    gert::StorageShape zeroPointsShape = {{2048}, {2048}};
    gert::StorageShape outShape = {{128, 48, 2048}, {128, 48, 2048}};
    int64_t axis = -2;
    string reduce = "update";
    int64_t quant_axis = -1;

    bool reciprocal_scale = false;
    string round_mode = "rint";
    ExecuteTestCase(
        {ge::DT_INT8, ge::DT_INT32, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8}, varShape, indicesShape,
        updatesShape, scalesShape, zeroPointsShape, outShape, reduce, axis, quant_axis, reciprocal_scale, round_mode,
        ge::GRAPH_FAILED);
}

TEST_F(QuantUpdateScatterRegBaseTiling, QuantUpdateScatter_var_error_dtype)
{
    gert::StorageShape varShape = {{128, 48, 2048}, {128, 48, 2048}};
    gert::StorageShape indicesShape = {{128}, {128}};
    gert::StorageShape updatesShape = {{128, 24, 2048}, {128, 24, 2048}};
    gert::StorageShape scalesShape = {{2048}, {2048}};
    gert::StorageShape zeroPointsShape = {{2048}, {2048}};
    gert::StorageShape outShape = {{128, 48, 2048}, {128, 48, 2048}};
    int64_t axis = -2;
    string reduce = "update";
    int64_t quant_axis = -1;

    bool reciprocal_scale = false;
    string round_mode = "rint";
    ExecuteTestCase(
        {ge::DT_INT32, ge::DT_INT32, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8}, varShape, indicesShape,
        updatesShape, scalesShape, zeroPointsShape, outShape, reduce, axis, quant_axis, reciprocal_scale, round_mode,
        ge::GRAPH_FAILED);
}

TEST_F(QuantUpdateScatterRegBaseTiling, QuantUpdateScatter_index_error_dtype)
{
    gert::StorageShape varShape = {{128, 48, 2048}, {128, 48, 2048}};
    gert::StorageShape indicesShape = {{128}, {128}};
    gert::StorageShape updatesShape = {{128, 24, 2048}, {128, 24, 2048}};
    gert::StorageShape scalesShape = {{2048}, {2048}};
    gert::StorageShape zeroPointsShape = {{2048}, {2048}};
    gert::StorageShape outShape = {{128, 48, 2048}, {128, 48, 2048}};
    int64_t axis = -2;
    string reduce = "update";
    int64_t quant_axis = -1;

    bool reciprocal_scale = false;
    string round_mode = "rint";
    ExecuteTestCase(
        {ge::DT_INT8, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8}, varShape, indicesShape,
        updatesShape, scalesShape, zeroPointsShape, outShape, reduce, axis, quant_axis, reciprocal_scale, round_mode,
        ge::GRAPH_FAILED);
}

TEST_F(QuantUpdateScatterRegBaseTiling, QuantUpdateScatter_update_error_dtype)
{
    gert::StorageShape varShape = {{128, 48, 2048}, {128, 48, 2048}};
    gert::StorageShape indicesShape = {{128}, {128}};
    gert::StorageShape updatesShape = {{128, 24, 2048}, {128, 24, 2048}};
    gert::StorageShape scalesShape = {{2048}, {2048}};
    gert::StorageShape zeroPointsShape = {{2048}, {2048}};
    gert::StorageShape outShape = {{128, 48, 2048}, {128, 48, 2048}};
    int64_t axis = -2;
    string reduce = "update";
    int64_t quant_axis = -1;

    bool reciprocal_scale = false;
    string round_mode = "rint";
    ExecuteTestCase(
        {ge::DT_INT8, ge::DT_INT32, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8}, varShape, indicesShape,
        updatesShape, scalesShape, zeroPointsShape, outShape, reduce, axis, quant_axis, reciprocal_scale, round_mode,
        ge::GRAPH_FAILED);
}

TEST_F(QuantUpdateScatterRegBaseTiling, QuantUpdateScatter_scale_error_dtype)
{
    gert::StorageShape varShape = {{128, 48, 2048}, {128, 48, 2048}};
    gert::StorageShape indicesShape = {{128}, {128}};
    gert::StorageShape updatesShape = {{128, 24, 2048}, {128, 24, 2048}};
    gert::StorageShape scalesShape = {{2048}, {2048}};
    gert::StorageShape zeroPointsShape = {{2048}, {2048}};
    gert::StorageShape outShape = {{128, 48, 2048}, {128, 48, 2048}};
    int64_t axis = -2;
    string reduce = "update";
    int64_t quant_axis = -1;

    bool reciprocal_scale = false;
    string round_mode = "rint";
    ExecuteTestCase(
        {ge::DT_INT8, ge::DT_INT32, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8}, varShape, indicesShape,
        updatesShape, scalesShape, zeroPointsShape, outShape, reduce, axis, quant_axis, reciprocal_scale, round_mode,
        ge::GRAPH_FAILED);
}

TEST_F(QuantUpdateScatterRegBaseTiling, QuantUpdateScatter_quant_zero_points_error_dtype)
{
    gert::StorageShape varShape = {{128, 48, 2048}, {128, 48, 2048}};
    gert::StorageShape indicesShape = {{128}, {128}};
    gert::StorageShape updatesShape = {{128, 24, 2048}, {128, 24, 2048}};
    gert::StorageShape scalesShape = {{2048}, {2048}};
    gert::StorageShape zeroPointsShape = {{2048}, {2048}};
    gert::StorageShape outShape = {{128, 48, 2048}, {128, 48, 2048}};
    int64_t axis = -2;
    string reduce = "update";
    int64_t quant_axis = -1;

    bool reciprocal_scale = false;
    string round_mode = "rint";
    ExecuteTestCase(
        {ge::DT_INT8, ge::DT_INT32, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT8, ge::DT_INT8}, varShape, indicesShape,
        updatesShape, scalesShape, zeroPointsShape, outShape, reduce, axis, quant_axis, reciprocal_scale, round_mode,
        ge::GRAPH_FAILED);
}

TEST_F(QuantUpdateScatterRegBaseTiling, QuantUpdateScatter_update_error_rank)
{
    gert::StorageShape varShape = {{128, 48}, {128, 48}};
    gert::StorageShape indicesShape = {{128}, {128}};
    gert::StorageShape updatesShape = {{128, 24}, {128, 24}};
    gert::StorageShape scalesShape = {{2048}, {2048}};
    gert::StorageShape zeroPointsShape = {{2048}, {2048}};
    gert::StorageShape outShape = {{128, 48, 2048}, {128, 48, 2048}};
    int64_t axis = -2;
    string reduce = "update";
    int64_t quant_axis = -1;

    bool reciprocal_scale = false;
    string round_mode = "rint";
    ExecuteTestCase(
        {ge::DT_INT8, ge::DT_INT32, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8}, varShape, indicesShape,
        updatesShape, scalesShape, zeroPointsShape, outShape, reduce, axis, quant_axis, reciprocal_scale, round_mode,
        ge::GRAPH_FAILED);
}

TEST_F(QuantUpdateScatterRegBaseTiling, QuantUpdateScatter_indices_shape_error)
{
    gert::StorageShape varShape = {{128, 48, 2048}, {128, 48, 2048}};
    gert::StorageShape indicesShape = {{128, 1}, {128, 1}};
    gert::StorageShape updatesShape = {{128, 24, 2048}, {128, 24, 2048}};
    gert::StorageShape scalesShape = {{2048}, {2048}};
    gert::StorageShape zeroPointsShape = {{2048}, {2048}};
    gert::StorageShape outShape = {{128, 48, 2048}, {128, 48, 2048}};
    int64_t axis = -2;
    string reduce = "update";
    int64_t quant_axis = -1;

    bool reciprocal_scale = false;
    string round_mode = "rint";
    ExecuteTestCase(
        {ge::DT_INT8, ge::DT_INT32, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8}, varShape, indicesShape,
        updatesShape, scalesShape, zeroPointsShape, outShape, reduce, axis, quant_axis, reciprocal_scale, round_mode,
        ge::GRAPH_FAILED);
}

TEST_F(QuantUpdateScatterRegBaseTiling, QuantUpdateScatter_attr_reduce_error)
{
    gert::StorageShape varShape = {{128, 48, 2048}, {128, 48, 2048}};
    gert::StorageShape indicesShape = {{128}, {128}};
    gert::StorageShape updatesShape = {{128, 24, 2048}, {128, 24, 2048}};
    gert::StorageShape scalesShape = {{2048}, {2048}};
    gert::StorageShape zeroPointsShape = {{2048}, {2048}};
    gert::StorageShape outShape = {{128, 48, 2048}, {128, 48, 2048}};
    int64_t axis = -2;
    string reduce = "Update";
    int64_t quant_axis = -1;

    bool reciprocal_scale = false;
    string round_mode = "rint";
    ExecuteTestCase(
        {ge::DT_INT8, ge::DT_INT32, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8}, varShape, indicesShape,
        updatesShape, scalesShape, zeroPointsShape, outShape, reduce, axis, quant_axis, reciprocal_scale, round_mode,
        ge::GRAPH_FAILED);
}

TEST_F(QuantUpdateScatterRegBaseTiling, QuantUpdateScatter_attr_axis_error)
{
    gert::StorageShape varShape = {{128, 48, 2048}, {128, 48, 2048}};
    gert::StorageShape indicesShape = {{128}, {128}};
    gert::StorageShape updatesShape = {{128, 24, 2048}, {128, 24, 2048}};
    gert::StorageShape scalesShape = {{2048}, {2048}};
    gert::StorageShape zeroPointsShape = {{2048}, {2048}};
    gert::StorageShape outShape = {{128, 48, 2048}, {128, 48, 2048}};
    int64_t axis = -1;
    string reduce = "update";
    int64_t quant_axis = -1;

    bool reciprocal_scale = false;
    string round_mode = "rint";
    ExecuteTestCase(
        {ge::DT_INT8, ge::DT_INT32, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8}, varShape, indicesShape,
        updatesShape, scalesShape, zeroPointsShape, outShape, reduce, axis, quant_axis, reciprocal_scale, round_mode,
        ge::GRAPH_FAILED);
}

TEST_F(QuantUpdateScatterRegBaseTiling, QuantUpdateScatter_attr_quant_axis_error)
{
    gert::StorageShape varShape = {{128, 48, 2048}, {128, 48, 2048}};
    gert::StorageShape indicesShape = {{128}, {128}};
    gert::StorageShape updatesShape = {{128, 24, 2048}, {128, 24, 2048}};
    gert::StorageShape scalesShape = {{2048}, {2048}};
    gert::StorageShape zeroPointsShape = {{2048}, {2048}};
    gert::StorageShape outShape = {{128, 48, 2048}, {128, 48, 2048}};
    int64_t axis = -2;
    string reduce = "update";
    int64_t quant_axis = -2;

    bool reciprocal_scale = false;
    string round_mode = "rint";
    ExecuteTestCase(
        {ge::DT_INT8, ge::DT_INT32, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8}, varShape, indicesShape,
        updatesShape, scalesShape, zeroPointsShape, outShape, reduce, axis, quant_axis, reciprocal_scale, round_mode,
        ge::GRAPH_FAILED);
}

TEST_F(QuantUpdateScatterRegBaseTiling, QuantUpdateScatter_var_update_shape_dim_diff)
{
    gert::StorageShape varShape = {{24, 4096, 128}, {24, 4096, 128}};
    gert::StorageShape indicesShape = {{24, 2}, {24, 2}};
    gert::StorageShape updatesShape = {{1024, 128}, {1024, 128}};
    gert::StorageShape scalesShape = {{128}, {128}};
    gert::StorageShape zeroPointsShape = {{128}, {128}};
    gert::StorageShape outShape = {{24, 4096, 128}, {24, 4096, 128}};
    int64_t axis = -2;
    string reduce = "update";
    int64_t quant_axis = -1;

    bool reciprocal_scale = false;
    string round_mode = "rint";
    ExecuteTestCase(
        {ge::DT_INT8, ge::DT_INT32, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8}, varShape, indicesShape,
        updatesShape, scalesShape, zeroPointsShape, outShape, reduce, axis, quant_axis, reciprocal_scale, round_mode,
        ge::GRAPH_FAILED);
}

TEST_F(QuantUpdateScatterRegBaseTiling, QuantUpdateScatter_update_index_shape_value_diff)
{
    gert::StorageShape varShape = {{24, 4096, 128}, {24, 4096, 128}};
    gert::StorageShape indicesShape = {{1, 2}, {1, 2}};
    gert::StorageShape updatesShape = {{12, 1024, 128}, {12, 1024, 128}};
    gert::StorageShape scalesShape = {{128}, {128}};
    gert::StorageShape zeroPointsShape = {{128}, {128}};
    gert::StorageShape outShape = {{24, 4096, 128}, {24, 4096, 128}};
    int64_t axis = -2;
    string reduce = "update";
    int64_t quant_axis = -1;

    bool reciprocal_scale = false;
    string round_mode = "rint";
    ExecuteTestCase(
        {ge::DT_INT8, ge::DT_INT32, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8}, varShape, indicesShape,
        updatesShape, scalesShape, zeroPointsShape, outShape, reduce, axis, quant_axis, reciprocal_scale, round_mode,
        ge::GRAPH_FAILED);
}

TEST_F(QuantUpdateScatterRegBaseTiling, QuantUpdateScatter_update_shape_larger_var)
{
    gert::StorageShape varShape = {{24, 4096, 128}, {24, 4096, 128}};
    gert::StorageShape indicesShape = {{24, 2}, {24, 2}};
    gert::StorageShape updatesShape = {{100, 1024, 128}, {100, 1024, 128}};
    gert::StorageShape scalesShape = {{128}, {128}};
    gert::StorageShape zeroPointsShape = {{128}, {128}};
    gert::StorageShape outShape = {{24, 4096, 128}, {24, 4096, 128}};
    int64_t axis = -2;
    string reduce = "update";
    int64_t quant_axis = -1;

    bool reciprocal_scale = false;
    string round_mode = "rint";
    ExecuteTestCase(
        {ge::DT_INT8, ge::DT_INT32, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8}, varShape, indicesShape,
        updatesShape, scalesShape, zeroPointsShape, outShape, reduce, axis, quant_axis, reciprocal_scale, round_mode,
        ge::GRAPH_FAILED);
}

TEST_F(QuantUpdateScatterRegBaseTiling, QuantUpdateScatter_update_shape_larger_var2)
{
    gert::StorageShape varShape = {{24, 4096, 128}, {24, 4096, 128}};
    gert::StorageShape indicesShape = {{24, 2}, {24, 2}};
    gert::StorageShape updatesShape = {{24, 5000, 128}, {24, 5000, 128}};
    gert::StorageShape scalesShape = {{128}, {128}};
    gert::StorageShape zeroPointsShape = {{128}, {128}};
    gert::StorageShape outShape = {{24, 4096, 128}, {24, 4096, 128}};
    int64_t axis = -2;
    string reduce = "update";
    int64_t quant_axis = -1;

    bool reciprocal_scale = false;
    string round_mode = "rint";
    ExecuteTestCase(
        {ge::DT_INT8, ge::DT_INT32, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8}, varShape, indicesShape,
        updatesShape, scalesShape, zeroPointsShape, outShape, reduce, axis, quant_axis, reciprocal_scale, round_mode,
        ge::GRAPH_FAILED);
}

TEST_F(QuantUpdateScatterRegBaseTiling, QuantUpdateScatter_round_mode_illegal)
{
    gert::StorageShape varShape = {{24, 1, 128}, {24, 1, 128}};
    gert::StorageShape indicesShape = {{24, 2}, {24, 2}};
    gert::StorageShape updatesShape = {{24, 1, 128}, {24, 1, 128}};
    gert::StorageShape scalesShape = {{128}, {128}};
    gert::StorageShape zeroPointsShape = {{128}, {128}};
    gert::StorageShape outShape = {{24, 1, 128}, {24, 1, 128}};
    int64_t axis = -2;
    string reduce = "update";
    int64_t quant_axis = -1;
    bool reciprocal_scale = false;
    string round_mode = "trunc";
    ExecuteTestCase(
        {ge::DT_INT8, ge::DT_INT32, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8}, varShape, indicesShape,
        updatesShape, scalesShape, zeroPointsShape, outShape, reduce, axis, quant_axis, reciprocal_scale, round_mode,
        ge::GRAPH_FAILED);
}

TEST_F(QuantUpdateScatterRegBaseTiling, QuantUpdateScatter_round_mode_not_match)
{
    gert::StorageShape varShape = {{24, 1, 128}, {24, 1, 128}};
    gert::StorageShape indicesShape = {{24, 2}, {24, 2}};
    gert::StorageShape updatesShape = {{24, 1, 128}, {24, 1, 128}};
    gert::StorageShape scalesShape = {{128}, {128}};
    gert::StorageShape zeroPointsShape = {{128}, {128}};
    gert::StorageShape outShape = {{24, 1, 128}, {24, 1, 128}};
    int64_t axis = -2;
    string reduce = "update";
    int64_t quant_axis = -1;
    bool reciprocal_scale = false;
    string round_mode = "hybrid";
    ExecuteTestCase(
        {ge::DT_INT8, ge::DT_INT32, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8}, varShape, indicesShape,
        updatesShape, scalesShape, zeroPointsShape, outShape, reduce, axis, quant_axis, reciprocal_scale, round_mode,
        ge::GRAPH_FAILED);
}

TEST_F(QuantUpdateScatterRegBaseTiling, QuantUpdateScatter_round_mode_match)
{
    gert::StorageShape varShape = {{24, 4096, 2}, {24, 4096, 2}};
    gert::StorageShape indicesShape = {{1, 2}, {1, 2}};
    gert::StorageShape updatesShape = {{1, 1024, 2}, {1, 1024, 2}};
    gert::StorageShape scalesShape = {{2}, {2}};
    gert::StorageShape zeroPointsShape = {{2}, {2}};
    gert::StorageShape outShape = {{24, 4096, 2}, {24, 4096, 2}};
    int64_t axis = -2;
    string reduce = "update";
    int64_t quant_axis = -1;
    bool reciprocal_scale = false;
    string round_mode = "hybrid";
    ExecuteTestCase(
        {ge::DT_HIFLOAT8, ge::DT_INT32, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_HIFLOAT8}, varShape, indicesShape,
        updatesShape, scalesShape, zeroPointsShape, outShape, reduce, axis, quant_axis, reciprocal_scale, round_mode,
        ge::GRAPH_SUCCESS);
}

TEST_F(QuantUpdateScatterRegBaseTiling, QuantUpdateScatter_axis_not_neg2_shape_error)
{
    gert::StorageShape varShape = {{24, 2, 4096, 2}, {24, 2, 4096, 2}};
    gert::StorageShape indicesShape = {{1, 2}, {1, 2}};
    gert::StorageShape updatesShape = {{1, 2, 1024, 2}, {1, 2, 1024, 2}};
    gert::StorageShape scalesShape = {{2}, {2}};
    gert::StorageShape zeroPointsShape = {{2}, {2}};
    gert::StorageShape outShape = {{24, 2, 4096, 2}, {24, 2, 4096, 2}};
    int64_t axis = -3;
    string reduce = "update";
    int64_t quant_axis = -1;
    bool reciprocal_scale = false;
    string round_mode = "hybrid";
    ExecuteTestCase(
        {ge::DT_HIFLOAT8, ge::DT_INT32, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_HIFLOAT8}, varShape, indicesShape,
        updatesShape, scalesShape, zeroPointsShape, outShape, reduce, axis, quant_axis, reciprocal_scale, round_mode,
        ge::GRAPH_FAILED);
}

TEST_F(QuantUpdateScatterRegBaseTiling, QuantUpdateScatter_axis_not_neg2)
{
    gert::StorageShape varShape = {{24, 2, 4096, 2}, {24, 2, 4096, 2}};
    gert::StorageShape indicesShape = {{1, 2}, {1, 2}};
    gert::StorageShape updatesShape = {{1, 2, 4096, 2}, {1, 2, 4096, 2}};
    gert::StorageShape scalesShape = {{2}, {2}};
    gert::StorageShape zeroPointsShape = {{2}, {2}};
    gert::StorageShape outShape = {{24, 2, 4096, 2}, {24, 2, 4096, 2}};
    int64_t axis = -3;
    string reduce = "update";
    int64_t quant_axis = -1;
    bool reciprocal_scale = false;
    string round_mode = "hybrid";

    ExecuteTestCase(
        {ge::DT_HIFLOAT8, ge::DT_INT32, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_HIFLOAT8}, varShape, indicesShape,
        updatesShape, scalesShape, zeroPointsShape, outShape, reduce, axis, quant_axis, reciprocal_scale, round_mode,
        ge::GRAPH_SUCCESS);
}
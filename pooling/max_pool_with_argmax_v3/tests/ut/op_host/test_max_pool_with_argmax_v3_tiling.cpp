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
#include "../../../op_host/arch35/max_pool_with_argmax_v3_tiling.h"

using namespace ut_util;
using namespace std;
using namespace ge;

class MaxPoolWithArgmaxV3Tiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MaxPoolWithArgmaxV3Tiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MaxPoolWithArgmaxV3Tiling TearDown" << std::endl;
    }
};

static void ExecuteTestCase(
    gert::StorageShape xShape, gert::StorageShape yShape, gert::StorageShape argmaxShape, std::vector<int64_t> ksize,
    std::vector<int64_t> strides, std::vector<int64_t> pads, std::vector<int64_t> dilation, ge::DataType dtype,
    int64_t index_dtype, bool ceil_mode, std::string data_format, uint64_t except_tilingkey, std::string expect)
{
    dlog_setlevel(0, 0, 0);

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
    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend910_95"}};

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::MaxPoolWithArgmaxV3CompileInfo compile_info;

    std::string op_type("MaxPoolWithArgmaxV3");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "version", soc_version_infos);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(1, 2)
                      .IrInstanceNum({1})
                      .InputShapes({&xShape})
                      .OutputShapes({&yShape, &argmaxShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(ksize)},
                           {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(strides)},
                           {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(pads)},
                           {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(index_dtype)},
                           {"dilation", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(dilation)},
                           {"ceil_mode", Ops::NN::AnyValue::CreateFrom<bool>(ceil_mode)},
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

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, except_tilingkey);
    // auto tilingData = tiling_context->GetRawTilingData();
    // ASSERT_NE(tilingData, nullptr);
    // dlog_setlevel(0, 3, 0);
    // auto tiling_data_result = to_string<int64_t>(tilingData->GetData(), tilingData->GetDataSize());
    // std::cout << tiling_data_result << std::endl;
    // EXPECT_EQ(tiling_data_result, expect);
}

TEST_F(MaxPoolWithArgmaxV3Tiling, MaxPoolWithArgmaxV3Tiling_NCHW_Test1)
{
    dlog_setlevel(0, 0, 0);
    gert::StorageShape xShape = {{2, 3, 64, 64}, {2, 3, 64, 64}};
    gert::StorageShape yShape = {{2, 3, 1, 1}, {2, 3, 1, 1}};
    gert::StorageShape argmaxShape = {{2, 3, 1, 1}, {2, 3, 1, 1}};

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

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::MaxPoolWithArgmaxV3CompileInfo compile_info;

    std::string op_type("MaxPoolWithArgmaxV3");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
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

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(1, 2)
                      .IrInstanceNum({1})
                      .InputShapes({&xShape})
                      .OutputShapes({&yShape, &argmaxShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({64, 64})},
                           {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({64, 64})},
                           {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0})},
                           {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(3)},
                           {"dilation", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
                           {"ceil_mode", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                           {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")}})
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
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 400001);
    // auto tilingData = tiling_context->GetRawTilingData();
    // ASSERT_NE(tilingData, nullptr);
    // dlog_setlevel(0, 3, 0);
    // EXPECT_EQ(to_string<int64_t>(tilingData->GetData(), tilingData->GetDataSize()),
    //           "64 64 1 1 64 64 64 64 0 0 1 1 6 10 7 1 0 0 0 0 30016 256 512 ");
}

TEST_F(MaxPoolWithArgmaxV3Tiling, MaxPoolWithArgmaxV3Tiling_NHWC_Test2)
{
    dlog_setlevel(0, 0, 0);
    gert::StorageShape xShape = {{2, 3, 64, 64}, {2, 3, 64, 64}};
    gert::StorageShape yShape = {{2, 3, 1, 1}, {2, 3, 1, 1}};
    gert::StorageShape argmaxShape = {{2, 3, 1, 1}, {2, 3, 1, 1}};

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

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::MaxPoolWithArgmaxV3CompileInfo compile_info;

    std::string op_type("MaxPoolWithArgmaxV3");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
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

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(1, 2)
                      .IrInstanceNum({1})
                      .InputShapes({&xShape})
                      .OutputShapes({&yShape, &argmaxShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({64, 64})},
                           {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({64, 64})},
                           {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0})},
                           {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(3)},
                           {"dilation", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
                           {"ceil_mode", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                           {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NHWC")}})
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
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 800001);
    // auto tilingData = tiling_context->GetRawTilingData();
    // ASSERT_NE(tilingData, nullptr);
    // EXPECT_EQ(to_string<int64_t>(tilingData->GetData(), tilingData->GetDataSize()),
    //           "64 3 64 3 1 64 64 64 64 0 0 1 1 1 1 2 1 1 3 1 1 1 64 64 1 1 1 6 114688 256 256 0 1 7 1 10 64 64 1
    //           800001 ");
    // dlog_setlevel(0, 3, 0);
}

TEST_F(MaxPoolWithArgmaxV3Tiling, MaxPoolWithArgmaxV3Tiling_NHWC_Test3)
{
    dlog_setlevel(0, 0, 0);
    gert::StorageShape xShape = {{2, 3, 64}, {2, 3, 64}};
    gert::StorageShape yShape = {{2, 3, 1}, {2, 3, 1}};
    gert::StorageShape argmaxShape = {{2, 3, 1}, {2, 3, 1}};

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

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::MaxPoolWithArgmaxV3CompileInfo compile_info;

    std::string op_type("MaxPoolWithArgmaxV3");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
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

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(1, 2)
                      .IrInstanceNum({1})
                      .InputShapes({&xShape})
                      .OutputShapes({&yShape, &argmaxShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({64, 64})},
                           {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({64, 64})},
                           {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0})},
                           {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(3)},
                           {"dilation", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
                           {"ceil_mode", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                           {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NHWC")}})
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
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
    dlog_setlevel(0, 3, 0);
}

TEST_F(MaxPoolWithArgmaxV3Tiling, MaxPoolWithArgmaxV3Tiling_NHWC_Test4)
{
    dlog_setlevel(0, 0, 0);
    gert::StorageShape xShape = {{2, 3, 0, 64}, {2, 3, 0, 64}};
    gert::StorageShape yShape = {{2, 3, 0, 1}, {2, 3, 0, 1}};
    gert::StorageShape argmaxShape = {{2, 3, 0, 1}, {2, 3, 0, 1}};

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

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::MaxPoolWithArgmaxV3CompileInfo compile_info;

    std::string op_type("MaxPoolWithArgmaxV3");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
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

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(1, 2)
                      .IrInstanceNum({1})
                      .InputShapes({&xShape})
                      .OutputShapes({&yShape, &argmaxShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({64, 64})},
                           {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({64, 64})},
                           {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0})},
                           {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(3)},
                           {"dilation", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
                           {"ceil_mode", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                           {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NHWC")}})
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
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
    dlog_setlevel(0, 3, 0);
}

TEST_F(MaxPoolWithArgmaxV3Tiling, MaxPoolWithArgmaxV3Tiling_NHWC_Test5)
{
    dlog_setlevel(0, 0, 0);
    gert::StorageShape xShape = {{2, 3, 64, 64}, {2, 3, 64, 64}};
    gert::StorageShape yShape = {{2, 3, 1, 0}, {2, 3, 1, 0}};
    gert::StorageShape argmaxShape = {{2, 3, 1, 0}, {2, 3, 1, 0}};

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

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::MaxPoolWithArgmaxV3CompileInfo compile_info;

    std::string op_type("MaxPoolWithArgmaxV3");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
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

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(1, 2)
                      .IrInstanceNum({1})
                      .InputShapes({&xShape})
                      .OutputShapes({&yShape, &argmaxShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({64, 64})},
                           {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({64, 64})},
                           {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0})},
                           {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(3)},
                           {"dilation", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
                           {"ceil_mode", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                           {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("ND")}})
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
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
    dlog_setlevel(0, 3, 0);
}

TEST_F(MaxPoolWithArgmaxV3Tiling, MaxPoolWithArgmaxV3Tiling_NHWC_Test6)
{
    dlog_setlevel(0, 0, 0);
    gert::StorageShape xShape = {{2, 3, 64, 64}, {2, 3, 64, 64}};
    gert::StorageShape yShape = {{2, 3, 1, 1}, {2, 3, 1, 1}};
    gert::StorageShape argmaxShape = {{2, 3, 1, 1}, {2, 3, 1, 1}};

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

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::MaxPoolWithArgmaxV3CompileInfo compile_info;

    std::string op_type("MaxPoolWithArgmaxV3");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
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

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(1, 2)
                      .IrInstanceNum({1})
                      .InputShapes({&xShape})
                      .OutputShapes({&yShape, &argmaxShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0})},
                           {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({64, 64})},
                           {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0})},
                           {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(3)},
                           {"dilation", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
                           {"ceil_mode", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                           {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NHWC")}})
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
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
    dlog_setlevel(0, 3, 0);
}

TEST_F(MaxPoolWithArgmaxV3Tiling, MaxPoolWithArgmaxV3Tiling_NHWC_Test7)
{
    dlog_setlevel(0, 0, 0);
    gert::StorageShape xShape = {{2, 3, 64, 64}, {2, 3, 64, 64}};
    gert::StorageShape yShape = {{2, 3, 1, 1}, {2, 3, 1, 1}};
    gert::StorageShape argmaxShape = {{2, 3, 1, 1}, {2, 3, 1, 1}};

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

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::MaxPoolWithArgmaxV3CompileInfo compile_info;

    std::string op_type("MaxPoolWithArgmaxV3");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
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

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(1, 2)
                      .IrInstanceNum({1})
                      .InputShapes({&xShape})
                      .OutputShapes({&yShape, &argmaxShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({64, 64})},
                           {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0})},
                           {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0})},
                           {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(3)},
                           {"dilation", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({1, 1})},
                           {"ceil_mode", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                           {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NHWC")}})
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
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
    dlog_setlevel(0, 3, 0);
}

TEST_F(MaxPoolWithArgmaxV3Tiling, MaxPoolWithArgmaxV3Tiling_NHWC_Test8)
{
    dlog_setlevel(0, 0, 0);
    gert::StorageShape xShape = {{2, 3, 64, 64}, {2, 3, 64, 64}};
    gert::StorageShape yShape = {{2, 3, 1, 1}, {2, 3, 1, 1}};
    gert::StorageShape argmaxShape = {{2, 3, 1, 1}, {2, 3, 1, 1}};

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

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::MaxPoolWithArgmaxV3CompileInfo compile_info;

    std::string op_type("MaxPoolWithArgmaxV3");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
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

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(1, 2)
                      .IrInstanceNum({1})
                      .InputShapes({&xShape})
                      .OutputShapes({&yShape, &argmaxShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"ksize", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({64, 64})},
                           {"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({64, 64})},
                           {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0})},
                           {"dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(3)},
                           {"dilation", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>({0, 0})},
                           {"ceil_mode", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                           {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NHWC")}})
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
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
    dlog_setlevel(0, 3, 0);
}

TEST_F(MaxPoolWithArgmaxV3Tiling, MaxPoolWithArgmaxV3Tiling_NCHW_Test9)
{
    gert::StorageShape xShape = {{4, 163, 1024, 600}, {4, 163, 1024, 600}};
    gert::StorageShape yShape = {{4, 163, 2, 2}, {4, 163, 2, 2}};
    gert::StorageShape argmaxShape = {{4, 163, 2, 2}, {4, 163, 2, 2}};
    std::vector<int64_t> ksize = {324, 457};
    std::vector<int64_t> strides = {858, 457};
    std::vector<int64_t> pads = {30, 132};
    std::vector<int64_t> dilation = {1, 1};
    ge::DataType dtype = ge::DT_FLOAT;
    int64_t index_dtype = 3;
    bool ceil_mode = false;
    std::string data_format = "NCHW";
    uint64_t except_tilingkey = 311110;
    std::string expect = "1024 600 2 2 457 324 457 858 132 30 1 1 40 48 2608 64 29184 0 ";
    ExecuteTestCase(
        xShape, yShape, argmaxShape, ksize, strides, pads, dilation, dtype, index_dtype, ceil_mode, data_format,
        except_tilingkey, expect);
}

TEST_F(MaxPoolWithArgmaxV3Tiling, MaxPoolWithArgmaxV3Tiling_NCHW_Test10)
{
    gert::StorageShape xShape = {{4, 833, 1024, 640}, {4, 833, 1024, 640}};
    gert::StorageShape yShape = {{4, 833, 2, 2}, {4, 833, 2, 2}};
    gert::StorageShape argmaxShape = {{4, 833, 2, 2}, {4, 833, 2, 2}};
    std::vector<int64_t> ksize = {455, 513};
    std::vector<int64_t> strides = {455, 256};
    std::vector<int64_t> pads = {106, 163};
    std::vector<int64_t> dilation = {1, 1};
    ge::DataType dtype = ge::DT_FLOAT;
    int64_t index_dtype = 9;
    bool ceil_mode = true;
    std::string data_format = "NCHW";
    uint64_t except_tilingkey = 311110;
    std::string expect = "1024 640 2 2 513 455 256 455 163 106 1 1 208 16 13328 64 29184 0 ";
    ExecuteTestCase(
        xShape, yShape, argmaxShape, ksize, strides, pads, dilation, dtype, index_dtype, ceil_mode, data_format,
        except_tilingkey, expect);
}

TEST_F(MaxPoolWithArgmaxV3Tiling, MaxPoolWithArgmaxV3Tiling_Gather_Test0)
{
    gert::StorageShape xShape = {{4, 16, 32, 32}, {4, 16, 32, 32}};
    gert::StorageShape yShape = {{4, 16, 16, 16}, {4, 16, 16, 16}};
    gert::StorageShape argmaxShape = {{4, 16, 16, 16}, {4, 16, 16, 16}};
    std::vector<int64_t> ksize = {2, 2};
    std::vector<int64_t> strides = {2, 2};
    std::vector<int64_t> pads = {0, 0};
    std::vector<int64_t> dilation = {1, 1};
    ge::DataType dtype = ge::DT_FLOAT;
    int64_t index_dtype = 3;
    bool ceil_mode = true;
    std::string data_format = "NCHW";
    uint64_t except_tilingkey = 300001;
    std::string expect = "32 32 16 16 2 2 2 2 0 0 2 2 32 16 16 1 16 16 1 1 1 32 8192 2048 2048 0 1 1 ";
    ExecuteTestCase(
        xShape, yShape, argmaxShape, ksize, strides, pads, dilation, dtype, index_dtype, ceil_mode, data_format,
        except_tilingkey, expect);
}

TEST_F(MaxPoolWithArgmaxV3Tiling, MaxPoolWithArgmaxV3Tiling_Gather_Test1)
{
    gert::StorageShape xShape = {{4, 16, 30, 30}, {4, 16, 30, 30}};
    gert::StorageShape yShape = {{4, 16, 16, 16}, {4, 16, 16, 16}};
    gert::StorageShape argmaxShape = {{4, 16, 16, 16}, {4, 16, 16, 16}};
    std::vector<int64_t> ksize = {2, 2};
    std::vector<int64_t> strides = {2, 2};
    std::vector<int64_t> pads = {1, 1};
    std::vector<int64_t> dilation = {1, 1};
    ge::DataType dtype = ge::DT_FLOAT;
    int64_t index_dtype = 3;
    bool ceil_mode = false;
    std::string data_format = "NCHW";
    uint64_t except_tilingkey = 300002;
    std::string expect = "30 30 16 16 2 2 2 2 1 1 2 2 32 16 16 1 16 16 1 1 1 32 8192 2048 2048 1 1 1 ";
    ExecuteTestCase(
        xShape, yShape, argmaxShape, ksize, strides, pads, dilation, dtype, index_dtype, ceil_mode, data_format,
        except_tilingkey, expect);
}
TEST_F(MaxPoolWithArgmaxV3Tiling, MaxPoolWithArgmaxV3Tiling_NCHW_MulCore_Float_Test01)
{
    gert::StorageShape xShape = {{1, 1, 68, 22}, {1, 1, 68, 22}};
    gert::StorageShape yShape = {{1, 1, 3, 4}, {1, 1, 3, 4}};
    gert::StorageShape argmaxShape = {{1, 1, 3, 4}, {1, 1, 3, 4}};
    std::vector<int64_t> ksize = {64, 16};
    std::vector<int64_t> strides = {2, 2};
    std::vector<int64_t> pads = {0, 0};
    std::vector<int64_t> dilation = {1, 1};
    ge::DataType dtype = ge::DT_FLOAT;
    int64_t index_dtype = 3;
    bool ceil_mode = true;
    std::string data_format = "NCHW";
    uint64_t except_tilingkey = 400001;
    std::string expect = "68 22 3 4 16 64 2 2 0 0 1 1 12 5 13 12 0 0 0 0 30016 256 512 ";
    ExecuteTestCase(
        xShape, yShape, argmaxShape, ksize, strides, pads, dilation, dtype, index_dtype, ceil_mode, data_format,
        except_tilingkey, expect);
}

TEST_F(MaxPoolWithArgmaxV3Tiling, MaxPoolWithArgmaxV3Tiling_NCHW_MulCore_Float_Test02)
{
    gert::StorageShape xShape = {{1, 1, 4, 522}, {1, 1, 4, 522}};
    gert::StorageShape yShape = {{1, 1, 2, 4}, {1, 1, 2, 4}};
    gert::StorageShape argmaxShape = {{1, 1, 2, 4}, {1, 1, 2, 4}};
    std::vector<int64_t> ksize = {2, 516};
    std::vector<int64_t> strides = {2, 2};
    std::vector<int64_t> pads = {0, 0};
    std::vector<int64_t> dilation = {1, 1};
    ge::DataType dtype = ge::DT_FLOAT;
    int64_t index_dtype = 3;
    bool ceil_mode = true;
    std::string data_format = "NCHW";
    uint64_t except_tilingkey = 400001;
    std::string expect = "4 522 2 4 516 2 2 2 0 0 1 1 8 8 0 0 1 129 129 4 30016 256 512 ";
    ExecuteTestCase(
        xShape, yShape, argmaxShape, ksize, strides, pads, dilation, dtype, index_dtype, ceil_mode, data_format,
        except_tilingkey, expect);
}

TEST_F(MaxPoolWithArgmaxV3Tiling, MaxPoolWithArgmaxV3Tiling_NCHW_MulCore_Float_Test03)
{
    gert::StorageShape xShape = {{1, 1, 68, 22}, {1, 1, 68, 22}};
    gert::StorageShape yShape = {{1, 1, 3, 4}, {1, 1, 3, 4}};
    gert::StorageShape argmaxShape = {{1, 1, 3, 4}, {1, 1, 3, 4}};
    std::vector<int64_t> ksize = {64, 16};
    std::vector<int64_t> strides = {2, 2};
    std::vector<int64_t> pads = {0, 0};
    std::vector<int64_t> dilation = {1, 1};
    ge::DataType dtype = ge::DT_FLOAT;
    int64_t index_dtype = 9;
    bool ceil_mode = true;
    std::string data_format = "NCHW";
    uint64_t except_tilingkey = 400002;
    std::string expect = "68 22 3 4 16 64 2 2 0 0 1 1 12 5 13 12 0 0 0 0 30016 256 512 ";
    ExecuteTestCase(
        xShape, yShape, argmaxShape, ksize, strides, pads, dilation, dtype, index_dtype, ceil_mode, data_format,
        except_tilingkey, expect);
}

TEST_F(MaxPoolWithArgmaxV3Tiling, MaxPoolWithArgmaxV3Tiling_NCHW_MulCore_Float_Test04)
{
    gert::StorageShape xShape = {{1, 1, 4, 522}, {1, 1, 4, 522}};
    gert::StorageShape yShape = {{1, 1, 2, 4}, {1, 1, 2, 4}};
    gert::StorageShape argmaxShape = {{1, 1, 2, 4}, {1, 1, 2, 4}};
    std::vector<int64_t> ksize = {2, 516};
    std::vector<int64_t> strides = {2, 2};
    std::vector<int64_t> pads = {0, 0};
    std::vector<int64_t> dilation = {1, 1};
    ge::DataType dtype = ge::DT_FLOAT;
    int64_t index_dtype = 9;
    bool ceil_mode = true;
    std::string data_format = "NCHW";
    uint64_t except_tilingkey = 400002;
    std::string expect = "4 522 2 4 516 2 2 2 0 0 1 1 8 8 0 0 1 129 129 4 30016 256 512 ";
    ExecuteTestCase(
        xShape, yShape, argmaxShape, ksize, strides, pads, dilation, dtype, index_dtype, ceil_mode, data_format,
        except_tilingkey, expect);
}

TEST_F(MaxPoolWithArgmaxV3Tiling, MaxPoolWithArgmaxV3Tiling_NCHW_MulCore_Bfloat16_Test01)
{
    gert::StorageShape xShape = {{1, 1, 68, 22}, {1, 1, 68, 22}};
    gert::StorageShape yShape = {{1, 1, 3, 4}, {1, 1, 3, 4}};
    gert::StorageShape argmaxShape = {{1, 1, 3, 4}, {1, 1, 3, 4}};
    std::vector<int64_t> ksize = {64, 16};
    std::vector<int64_t> strides = {2, 2};
    std::vector<int64_t> pads = {0, 0};
    std::vector<int64_t> dilation = {1, 1};
    ge::DataType dtype = ge::DT_BF16;
    int64_t index_dtype = 3;
    bool ceil_mode = true;
    std::string data_format = "NCHW";
    uint64_t except_tilingkey = 300001;
    std::string expect = "68 22 3 4 64 16 2 2 0 0 1 1 1 1 1 3 1 1 4 1 1 12 2048 32 64 0 1 1 ";
    ExecuteTestCase(
        xShape, yShape, argmaxShape, ksize, strides, pads, dilation, dtype, index_dtype, ceil_mode, data_format,
        except_tilingkey, expect);
}

TEST_F(MaxPoolWithArgmaxV3Tiling, MaxPoolWithArgmaxV3Tiling_NCHW_MulCore_Bfloat16_Test02)
{
    gert::StorageShape xShape = {{1, 1, 4, 522}, {1, 1, 4, 522}};
    gert::StorageShape yShape = {{1, 1, 2, 4}, {1, 1, 2, 4}};
    gert::StorageShape argmaxShape = {{1, 1, 2, 4}, {1, 1, 2, 4}};
    std::vector<int64_t> ksize = {2, 516};
    std::vector<int64_t> strides = {2, 2};
    std::vector<int64_t> pads = {0, 0};
    std::vector<int64_t> dilation = {1, 1};
    ge::DataType dtype = ge::DT_BF16;
    int64_t index_dtype = 3;
    bool ceil_mode = true;
    std::string data_format = "NCHW";
    uint64_t except_tilingkey = 400003;
    std::string expect = "4 522 2 4 516 2 2 2 0 0 1 1 8 8 0 0 1 129 129 4 30016 256 512 ";
    ExecuteTestCase(
        xShape, yShape, argmaxShape, ksize, strides, pads, dilation, dtype, index_dtype, ceil_mode, data_format,
        except_tilingkey, expect);
}

TEST_F(MaxPoolWithArgmaxV3Tiling, MaxPoolWithArgmaxV3Tiling_NCHW_MulCore_Bfloat16_Test03)
{
    gert::StorageShape xShape = {{1, 1, 4, 522}, {1, 1, 4, 522}};
    gert::StorageShape yShape = {{1, 1, 2, 4}, {1, 1, 2, 4}};
    gert::StorageShape argmaxShape = {{1, 1, 2, 4}, {1, 1, 2, 4}};
    std::vector<int64_t> ksize = {2, 516};
    std::vector<int64_t> strides = {2, 2};
    std::vector<int64_t> pads = {0, 0};
    std::vector<int64_t> dilation = {1, 1};
    ge::DataType dtype = ge::DT_BF16;
    int64_t index_dtype = 9;
    bool ceil_mode = true;
    std::string data_format = "NCHW";
    uint64_t except_tilingkey = 400004;
    std::string expect = "4 522 2 4 516 2 2 2 0 0 1 1 8 8 0 0 1 129 129 4 30016 256 512 ";
    ExecuteTestCase(
        xShape, yShape, argmaxShape, ksize, strides, pads, dilation, dtype, index_dtype, ceil_mode, data_format,
        except_tilingkey, expect);
}

TEST_F(MaxPoolWithArgmaxV3Tiling, MaxPoolWithArgmaxV3Tiling_NCHW_MulCore_Half_Test01)
{
    gert::StorageShape xShape = {{1, 1, 68, 22}, {1, 1, 68, 22}};
    gert::StorageShape yShape = {{1, 1, 3, 4}, {1, 1, 3, 4}};
    gert::StorageShape argmaxShape = {{1, 1, 3, 4}, {1, 1, 3, 4}};
    std::vector<int64_t> ksize = {64, 16};
    std::vector<int64_t> strides = {2, 2};
    std::vector<int64_t> pads = {0, 0};
    std::vector<int64_t> dilation = {1, 1};
    ge::DataType dtype = ge::DT_FLOAT16;
    int64_t index_dtype = 3;
    bool ceil_mode = true;
    std::string data_format = "NCHW";
    uint64_t except_tilingkey = 300001;
    std::string expect = "68 22 3 4 64 16 2 2 0 0 1 1 1 1 1 3 1 1 4 1 1 12 2048 32 64 0 1 1 ";
    ExecuteTestCase(
        xShape, yShape, argmaxShape, ksize, strides, pads, dilation, dtype, index_dtype, ceil_mode, data_format,
        except_tilingkey, expect);
}

TEST_F(MaxPoolWithArgmaxV3Tiling, MaxPoolWithArgmaxV3Tiling_NCHW_MulCore_Half_Test02)
{
    gert::StorageShape xShape = {{1, 1, 4, 522}, {1, 1, 4, 522}};
    gert::StorageShape yShape = {{1, 1, 2, 4}, {1, 1, 2, 4}};
    gert::StorageShape argmaxShape = {{1, 1, 2, 4}, {1, 1, 2, 4}};
    std::vector<int64_t> ksize = {2, 516};
    std::vector<int64_t> strides = {2, 2};
    std::vector<int64_t> pads = {0, 0};
    std::vector<int64_t> dilation = {1, 1};
    ge::DataType dtype = ge::DT_FLOAT16;
    int64_t index_dtype = 3;
    bool ceil_mode = true;
    std::string data_format = "NCHW";
    uint64_t except_tilingkey = 400005;
    std::string expect = "4 522 2 4 516 2 2 2 0 0 1 1 8 8 0 0 1 129 129 4 30016 256 512 ";
    ExecuteTestCase(
        xShape, yShape, argmaxShape, ksize, strides, pads, dilation, dtype, index_dtype, ceil_mode, data_format,
        except_tilingkey, expect);
}

TEST_F(MaxPoolWithArgmaxV3Tiling, MaxPoolWithArgmaxV3Tiling_NCHW_MulCore_Half_Test03)
{
    gert::StorageShape xShape = {{1, 1, 4, 522}, {1, 1, 4, 522}};
    gert::StorageShape yShape = {{1, 1, 2, 4}, {1, 1, 2, 4}};
    gert::StorageShape argmaxShape = {{1, 1, 2, 4}, {1, 1, 2, 4}};
    std::vector<int64_t> ksize = {2, 516};
    std::vector<int64_t> strides = {2, 2};
    std::vector<int64_t> pads = {0, 0};
    std::vector<int64_t> dilation = {1, 1};
    ge::DataType dtype = ge::DT_FLOAT16;
    int64_t index_dtype = 9;
    bool ceil_mode = true;
    std::string data_format = "NCHW";
    uint64_t except_tilingkey = 400006;
    std::string expect = "4 522 2 4 516 2 2 2 0 0 1 1 8 8 0 0 1 129 129 4 30016 256 512 ";
    ExecuteTestCase(
        xShape, yShape, argmaxShape, ksize, strides, pads, dilation, dtype, index_dtype, ceil_mode, data_format,
        except_tilingkey, expect);
}

TEST_F(MaxPoolWithArgmaxV3Tiling, MaxPoolWithArgmaxV3Tiling_SIMT_Test01)
{
    gert::StorageShape xShape = {{4, 4, 2665, 4}, {4, 4, 2665, 4}};
    gert::StorageShape yShape = {{4, 4, 738, 2}, {4, 4, 738, 2}};
    gert::StorageShape argmaxShape = {{4, 4, 738, 2}, {4, 4, 738, 2}};
    std::vector<int64_t> ksize = {456, 4};
    std::vector<int64_t> strides = {3, 3};
    std::vector<int64_t> pads = {1, 1};
    std::vector<int64_t> dilation = {1, 1};
    ge::DataType dtype = ge::DT_BF16;
    int64_t index_dtype = 3;
    bool ceil_mode = true;
    std::string data_format = "NCHW";
    uint64_t except_tilingkey = 500001;
    std::string expect = "256 64 4 4 2665 4 738 2 456 4 3 3 1 1 1 1 1 ";
    ExecuteTestCase(
        xShape, yShape, argmaxShape, ksize, strides, pads, dilation, dtype, index_dtype, ceil_mode, data_format,
        except_tilingkey, expect);
}

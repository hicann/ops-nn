/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_gather_nd_simt_tiling.cpp
 * \brief
 */

#include <iostream>
#include <vector>
#include <gtest/gtest.h>

#include "log/log.h"
#include "kernel_run_context_facker.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "test_cube_util.h"
#include "register/op_impl_registry.h"
#include "ut_op_util.h"
#include "ut_op_common.h"
#include "platform/platform_infos_def.h"
#include "../../../../op_host/gather_nd_tiling.h"
#include "../../../../op_host/gather_nd_tiling_arch35.h"

class AscendGatherNdTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AscendGatherNdTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AscendGatherNdTest TearDown" << std::endl;
    }
};

TEST_F(AscendGatherNdTest, test_tiling_1)
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
                                            "CORE_NUM": 48,
                                            "version": "Ascend950"
                                        }
                                    })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::GatherNdCompileInfo compile_info;

    std::string op_type("GatherNd");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape x = {{1000, 1000, 1}, {1000, 1000, 1}};
    gert::StorageShape indices = {{1039, 2}, {1039, 2}};
    gert::StorageShape y = {{1039, 1}, {1039, 1}};
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&x, &indices})
                      .OutputShapes({&y})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"negative_index_support", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
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
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 240);
}

TEST_F(AscendGatherNdTest, test_tiling_2)
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
                                            "CORE_NUM": 48,
                                            "version": "Ascend910B"
                                        }
                                    })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::GatherNdCompileInfo compile_info;

    std::string op_type("GatherNd");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape x = {{1000, 1000, 1}, {1000, 1000, 1}};
    gert::StorageShape indices = {{1039, 2}, {1039, 2}};
    gert::StorageShape y = {{1039, 1}, {1039, 1}};
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&x, &indices})
                      .OutputShapes({&y})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"negative_index_support", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
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
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 440);
}

// --------------

struct GatherNdParam {
    ge::DataType xDtype;
    ge::DataType indicesDtype;
    gert::StorageShape xShape;
    gert::StorageShape indicesShape;
    gert::StorageShape yShape;
    bool negativeIndexSupport;
};

static void ExecuteTestCase(
    const GatherNdParam& opsParamInfos, uint64_t expectTilingKey, ge::graphStatus status = ge::GRAPH_SUCCESS)
{
    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 40}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}};
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    // compile info
    optiling::GatherNdCompileInfo compile_info;

    std::string op_type("GatherNd");
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "version", soc_version_infos);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);

    gert::StorageShape xShape = opsParamInfos.xShape;
    gert::StorageShape indicesShape = opsParamInfos.indicesShape;
    gert::StorageShape yShape = opsParamInfos.yShape;

    auto holder =
        gert::TilingContextFaker()
            .NodeIoNum(2, 1)
            .IrInstanceNum({1, 1})
            .InputShapes({&xShape, &indicesShape})
            .OutputShapes({&yShape})
            .CompileInfo(&compile_info)
            .PlatformInfo(reinterpret_cast<char*>(&platform_info))
            .NodeInputTd(0, opsParamInfos.xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeInputTd(1, opsParamInfos.indicesDtype, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeOutputTd(0, opsParamInfos.xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
            .NodeAttrs(
                {{"negative_index_support", Ops::NN::AnyValue::CreateFrom<bool>(opsParamInfos.negativeIndexSupport)}})
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
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, expectTilingKey);
    auto raw_tiling_data = tiling_context->GetRawTilingData();
}

TEST_F(AscendGatherNdTest, GatherNd_tiling_ascendc_B64_indices64_noneg_size64)
{
    GatherNdParam opsParamInfos;
    opsParamInfos.xDtype = ge::DT_INT64;
    opsParamInfos.xShape = {{8, 127, 127}, {8, 127, 127}};
    opsParamInfos.indicesDtype = ge::DT_INT64;
    opsParamInfos.indicesShape = {{10000, 1700, 2}, {10000, 1700, 2}};
    opsParamInfos.yShape = {{10000, 1700, 127}, {10000, 1700, 127}};
    uint64_t expectTilingKey = 881;
    opsParamInfos.negativeIndexSupport = false;
    ExecuteTestCase(opsParamInfos, expectTilingKey);
}

TEST_F(AscendGatherNdTest, GatherNd_tiling_ascendc_B64_indices64_noneg_size32)
{
    GatherNdParam opsParamInfos;
    opsParamInfos.xDtype = ge::DT_INT32;
    opsParamInfos.xShape = {{8, 4, 2}, {8, 4, 2}};
    opsParamInfos.indicesDtype = ge::DT_INT64;
    opsParamInfos.indicesShape = {{4, 2}, {4, 2}};
    opsParamInfos.yShape = {{4, 2}, {4, 2}};
    uint64_t expectTilingKey = 10000000001231100033;
    opsParamInfos.negativeIndexSupport = false;
    ExecuteTestCase(opsParamInfos, expectTilingKey);
}

TEST_F(AscendGatherNdTest, GatherNd_tiling_ascendc_B64_indices32_noneg_size32)
{
    GatherNdParam opsParamInfos;
    opsParamInfos.xDtype = ge::DT_INT32;
    opsParamInfos.xShape = {{8, 4, 2}, {8, 4, 2}};
    opsParamInfos.indicesDtype = ge::DT_INT32;
    opsParamInfos.indicesShape = {{4, 2}, {4, 2}};
    opsParamInfos.yShape = {{4, 2}, {4, 2}};
    uint64_t expectTilingKey = 10000000001231100032;
    opsParamInfos.negativeIndexSupport = false;
    ExecuteTestCase(opsParamInfos, expectTilingKey);
}

TEST_F(AscendGatherNdTest, GatherNd_tiling_ascendc_B64_indices32_noneg_size64)
{
    GatherNdParam opsParamInfos;
    opsParamInfos.xDtype = ge::DT_INT64;
    opsParamInfos.xShape = {{8, 127, 127}, {8, 127, 127}};
    opsParamInfos.indicesDtype = ge::DT_INT32;
    opsParamInfos.indicesShape = {{10000, 1700, 2}, {10000, 1700, 2}};
    opsParamInfos.yShape = {{10000, 1700, 127}, {10000, 1700, 127}};
    uint64_t expectTilingKey = 841;
    opsParamInfos.negativeIndexSupport = false;
    ExecuteTestCase(opsParamInfos, expectTilingKey);
}

TEST_F(AscendGatherNdTest, GatherNd_tiling_ascendc_B32_indices64_noneg_size64)
{
    GatherNdParam opsParamInfos;
    opsParamInfos.xDtype = ge::DT_INT32;
    opsParamInfos.xShape = {{8, 127, 127}, {8, 127, 127}};
    opsParamInfos.indicesDtype = ge::DT_INT64;
    opsParamInfos.indicesShape = {{10000, 1700, 2}, {10000, 1700, 2}};
    opsParamInfos.yShape = {{10000, 1700, 127}, {10000, 1700, 127}};
    uint64_t expectTilingKey = 481;
    opsParamInfos.negativeIndexSupport = false;
    ExecuteTestCase(opsParamInfos, expectTilingKey);
}

TEST_F(AscendGatherNdTest, GatherNd_tiling_ascendc_B32_indices64_noneg_size32)
{
    GatherNdParam opsParamInfos;
    opsParamInfos.xDtype = ge::DT_INT32;
    opsParamInfos.xShape = {{8, 4, 1}, {8, 4, 1}};
    opsParamInfos.indicesDtype = ge::DT_INT64;
    opsParamInfos.indicesShape = {{4, 2}, {4, 2}};
    opsParamInfos.yShape = {{4, 1}, {4, 1}};
    uint64_t expectTilingKey = 10000000001231100023;
    opsParamInfos.negativeIndexSupport = false;
    ExecuteTestCase(opsParamInfos, expectTilingKey);
}

TEST_F(AscendGatherNdTest, GatherNd_tiling_ascendc_B32_indices32_noneg_size64)
{
    GatherNdParam opsParamInfos;
    opsParamInfos.xDtype = ge::DT_INT32;
    opsParamInfos.xShape = {{8, 127, 127}, {8, 127, 127}};
    opsParamInfos.indicesDtype = ge::DT_INT32;
    opsParamInfos.indicesShape = {{10000, 1700, 2}, {10000, 1700, 2}};
    opsParamInfos.yShape = {{10000, 1700, 2}, {10000, 1700, 2}};
    uint64_t expectTilingKey = 441;
    opsParamInfos.negativeIndexSupport = false;
    ExecuteTestCase(opsParamInfos, expectTilingKey);
}

TEST_F(AscendGatherNdTest, GatherNd_tiling_ascendc_B64_inddices32_noneg_size32)
{
    GatherNdParam opsParamInfos;
    opsParamInfos.xDtype = ge::DT_INT64;
    opsParamInfos.xShape = {{8, 4, 1}, {8, 4, 1}};
    opsParamInfos.indicesDtype = ge::DT_INT32;
    opsParamInfos.indicesShape = {{2, 2}, {2, 2}};
    opsParamInfos.yShape = {{2, 1}, {2, 1}};
    uint64_t expectTilingKey = 10000000001231100032;
    opsParamInfos.negativeIndexSupport = false;
    ExecuteTestCase(opsParamInfos, expectTilingKey);
}

TEST_F(AscendGatherNdTest, GatherNd_tiling_ascendc_B16_indices64_noneg_size64)
{
    GatherNdParam opsParamInfos;
    opsParamInfos.xDtype = ge::DT_FLOAT16;
    opsParamInfos.xShape = {{8, 127, 127}, {8, 127, 127}};
    opsParamInfos.indicesDtype = ge::DT_INT64;
    opsParamInfos.indicesShape = {{10000, 1700, 2}, {10000, 1700, 2}};
    opsParamInfos.yShape = {{10000, 1700, 2}, {10000, 1700, 2}};
    uint64_t expectTilingKey = 281;
    opsParamInfos.negativeIndexSupport = false;
    ExecuteTestCase(opsParamInfos, expectTilingKey);
}

TEST_F(AscendGatherNdTest, GatherNd_tiling_ascendc_B16_indices64_noneg_size32)
{
    GatherNdParam opsParamInfos;
    opsParamInfos.xDtype = ge::DT_FLOAT16;
    opsParamInfos.xShape = {{8, 4, 1}, {8, 4, 1}};
    opsParamInfos.indicesDtype = ge::DT_INT64;
    opsParamInfos.indicesShape = {{2, 2}, {2, 2}};
    opsParamInfos.yShape = {{2, 1}, {2, 1}};
    uint64_t expectTilingKey = 10000000001231100013;
    opsParamInfos.negativeIndexSupport = false;
    ExecuteTestCase(opsParamInfos, expectTilingKey);
}

TEST_F(AscendGatherNdTest, GatherNd_tiling_ascendc_B16_indices32_noneg_size64)
{
    GatherNdParam opsParamInfos;
    opsParamInfos.xDtype = ge::DT_FLOAT16;
    opsParamInfos.xShape = {{8, 127, 127}, {8, 127, 127}};
    opsParamInfos.indicesDtype = ge::DT_INT32;
    opsParamInfos.indicesShape = {{10000, 1700, 2}, {10000, 1700, 2}};
    opsParamInfos.yShape = {{10000, 1700, 2}, {10000, 1700, 2}};
    uint64_t expectTilingKey = 241;
    opsParamInfos.negativeIndexSupport = false;
    ExecuteTestCase(opsParamInfos, expectTilingKey);
}

TEST_F(AscendGatherNdTest, GatherNd_tiling_ascendc_B8_indices64_noneg_size64)
{
    GatherNdParam opsParamInfos;
    opsParamInfos.xDtype = ge::DT_INT8;
    opsParamInfos.xShape = {{8, 4, 127}, {8, 4, 127}};
    opsParamInfos.indicesDtype = ge::DT_INT64;
    opsParamInfos.indicesShape = {{10000, 1700, 2}, {10000, 1700, 2}};
    opsParamInfos.yShape = {{10000, 1700, 127}, {10000, 1700, 127}};
    uint64_t expectTilingKey = 181;
    opsParamInfos.negativeIndexSupport = false;
    ExecuteTestCase(opsParamInfos, expectTilingKey);
}

TEST_F(AscendGatherNdTest, GatherNd_tiling_ascendc_B8_indices64_noneg_size32)
{
    GatherNdParam opsParamInfos;
    opsParamInfos.xDtype = ge::DT_INT8;
    opsParamInfos.xShape = {{8, 4, 1}, {8, 4, 1}};
    opsParamInfos.indicesDtype = ge::DT_INT64;
    opsParamInfos.indicesShape = {{2, 2}, {2, 2}};
    opsParamInfos.yShape = {{2, 1}, {2, 1}};
    uint64_t expectTilingKey = 10000000001231100003;
    opsParamInfos.negativeIndexSupport = false;
    ExecuteTestCase(opsParamInfos, expectTilingKey);
}

TEST_F(AscendGatherNdTest, GatherNd_tiling_ascendc_B8_indices32_noneg_size32)
{
    GatherNdParam opsParamInfos;
    opsParamInfos.xDtype = ge::DT_INT8;
    opsParamInfos.xShape = {{8, 4, 1}, {8, 4, 1}};
    opsParamInfos.indicesDtype = ge::DT_INT32;
    opsParamInfos.indicesShape = {{2, 2}, {2, 2}};
    opsParamInfos.yShape = {{2, 1}, {2, 1}};
    uint64_t expectTilingKey = 10000000001231100002;
    opsParamInfos.negativeIndexSupport = false;
    ExecuteTestCase(opsParamInfos, expectTilingKey);
}

TEST_F(AscendGatherNdTest, GatherNd_tiling_ascendc_B8_indices32_noneg_size64)
{
    GatherNdParam opsParamInfos;
    opsParamInfos.xDtype = ge::DT_INT8;
    opsParamInfos.xShape = {{8, 127, 127}, {8, 127, 127}};
    opsParamInfos.indicesDtype = ge::DT_INT32;
    opsParamInfos.indicesShape = {{10000, 1700, 2}, {10000, 1700, 2}};
    opsParamInfos.yShape = {{10000, 1700, 2}, {10000, 1700, 2}};
    uint64_t expectTilingKey = 141;
    opsParamInfos.negativeIndexSupport = false;
    ExecuteTestCase(opsParamInfos, expectTilingKey);
}

TEST_F(AscendGatherNdTest, GatherNd_tiling_ascendc_B64_inddices64_supportneg_size64)
{
    GatherNdParam opsParamInfos;
    opsParamInfos.xDtype = ge::DT_INT64;
    opsParamInfos.xShape = {{8, 127, 127}, {8, 127, 127}};
    opsParamInfos.indicesDtype = ge::DT_INT64;
    opsParamInfos.indicesShape = {{10000, 1700, 2}, {10000, 1700, 2}};
    opsParamInfos.yShape = {{10000, 1700, 127}, {10000, 1700, 127}};
    uint64_t expectTilingKey = 881;
    opsParamInfos.negativeIndexSupport = true;
    ExecuteTestCase(opsParamInfos, expectTilingKey);
}

TEST_F(AscendGatherNdTest, GatherNd_tiling_ascendc_B64_inddices64_supportneg_size32)
{
    GatherNdParam opsParamInfos;
    opsParamInfos.xDtype = ge::DT_INT64;
    opsParamInfos.xShape = {{8, 4, 32}, {8, 4, 32}};
    opsParamInfos.indicesDtype = ge::DT_INT64;
    opsParamInfos.indicesShape = {{4, 2}, {4, 2}};
    opsParamInfos.yShape = {{4, 32}, {4, 32}};
    uint64_t expectTilingKey = 10000000001231100133;
    opsParamInfos.negativeIndexSupport = true;
    ExecuteTestCase(opsParamInfos, expectTilingKey);
}

TEST_F(AscendGatherNdTest, GatherNd_tiling_ascendc_B64_inddices32_supportneg_size32)
{
    GatherNdParam opsParamInfos;
    opsParamInfos.xDtype = ge::DT_INT64;
    opsParamInfos.xShape = {{8, 4, 32}, {8, 4, 32}};
    opsParamInfos.indicesDtype = ge::DT_INT32;
    opsParamInfos.indicesShape = {{4, 2}, {4, 2}};
    opsParamInfos.yShape = {{4, 32}, {4, 32}};
    uint64_t expectTilingKey = 10000000001231100132;
    opsParamInfos.negativeIndexSupport = true;
    ExecuteTestCase(opsParamInfos, expectTilingKey);
}

TEST_F(AscendGatherNdTest, GatherNd_tiling_ascendc_B64_inddices32_supportneg_size64)
{
    GatherNdParam opsParamInfos;
    opsParamInfos.xDtype = ge::DT_INT64;
    opsParamInfos.xShape = {{8, 127, 127}, {8, 127, 127}};
    opsParamInfos.indicesDtype = ge::DT_INT32;
    opsParamInfos.indicesShape = {{10000, 1700, 2}, {10000, 1700, 2}};
    opsParamInfos.yShape = {{10000, 1700, 127}, {10000, 1700, 127}};
    uint64_t expectTilingKey = 841;
    opsParamInfos.negativeIndexSupport = true;
    ExecuteTestCase(opsParamInfos, expectTilingKey);
}

TEST_F(AscendGatherNdTest, GatherNd_tiling_ascendc_B32_inddices64_supportneg_size64)
{
    GatherNdParam opsParamInfos;
    opsParamInfos.xDtype = ge::DT_INT32;
    opsParamInfos.xShape = {{8, 127, 127}, {8, 127, 127}};
    opsParamInfos.indicesDtype = ge::DT_INT64;
    opsParamInfos.indicesShape = {{10000, 1700, 2}, {10000, 1700, 2}};
    opsParamInfos.yShape = {{10000, 1700, 127}, {10000, 1700, 127}};
    uint64_t expectTilingKey = 481;
    opsParamInfos.negativeIndexSupport = true;
    ExecuteTestCase(opsParamInfos, expectTilingKey);
}

TEST_F(AscendGatherNdTest, GatherNd_tiling_ascendc_B32_inddices64_supportneg_size32)
{
    GatherNdParam opsParamInfos;
    opsParamInfos.xDtype = ge::DT_INT32;
    opsParamInfos.xShape = {{8, 4, 32}, {8, 4, 32}};
    opsParamInfos.indicesDtype = ge::DT_INT64;
    opsParamInfos.indicesShape = {{4, 2}, {4, 2}};
    opsParamInfos.yShape = {{4, 32}, {4, 32}};
    uint64_t expectTilingKey = 10000000001231100123;
    opsParamInfos.negativeIndexSupport = true;
    ExecuteTestCase(opsParamInfos, expectTilingKey);
}

TEST_F(AscendGatherNdTest, GatherNd_tiling_ascendc_B32_inddices32_supportneg_size32)
{
    GatherNdParam opsParamInfos;
    opsParamInfos.xDtype = ge::DT_INT32;
    opsParamInfos.xShape = {{8, 4, 32}, {8, 4, 32}};
    opsParamInfos.indicesDtype = ge::DT_INT32;
    opsParamInfos.indicesShape = {{4, 2}, {4, 2}};
    opsParamInfos.yShape = {{4, 32}, {4, 32}};
    uint64_t expectTilingKey = 10000000001231100122;
    opsParamInfos.negativeIndexSupport = true;
    ExecuteTestCase(opsParamInfos, expectTilingKey);
}

TEST_F(AscendGatherNdTest, GatherNd_tiling_ascendc_B32_inddices32_supportneg_size64)
{
    GatherNdParam opsParamInfos;
    opsParamInfos.xDtype = ge::DT_INT32;
    opsParamInfos.xShape = {{8, 127, 127}, {8, 127, 127}};
    opsParamInfos.indicesDtype = ge::DT_INT32;
    opsParamInfos.indicesShape = {{10000, 1700, 2}, {10000, 1700, 2}};
    opsParamInfos.yShape = {{10000, 1700, 127}, {10000, 1700, 127}};
    uint64_t expectTilingKey = 441;
    opsParamInfos.negativeIndexSupport = true;
    ExecuteTestCase(opsParamInfos, expectTilingKey);
}

TEST_F(AscendGatherNdTest, GatherNd_tiling_ascendc_B16_inddices64_supportneg_size64)
{
    GatherNdParam opsParamInfos;
    opsParamInfos.xDtype = ge::DT_FLOAT16;
    opsParamInfos.xShape = {{8, 127, 127}, {8, 127, 127}};
    opsParamInfos.indicesDtype = ge::DT_INT64;
    opsParamInfos.indicesShape = {{10000, 1700, 2}, {10000, 1700, 2}};
    opsParamInfos.yShape = {{10000, 1700, 127}, {10000, 1700, 127}};
    uint64_t expectTilingKey = 281;
    opsParamInfos.negativeIndexSupport = true;
    ExecuteTestCase(opsParamInfos, expectTilingKey);
}

TEST_F(AscendGatherNdTest, GatherNd_tiling_ascendc_B16_inddices64_supportneg_size32)
{
    GatherNdParam opsParamInfos;
    opsParamInfos.xDtype = ge::DT_FLOAT16;
    opsParamInfos.xShape = {{8, 4, 32}, {8, 4, 32}};
    opsParamInfos.indicesDtype = ge::DT_INT64;
    opsParamInfos.indicesShape = {{4, 2}, {4, 2}};
    opsParamInfos.yShape = {{4, 32}, {4, 32}};
    uint64_t expectTilingKey = 10000000001231100113;
    opsParamInfos.negativeIndexSupport = true;
    ExecuteTestCase(opsParamInfos, expectTilingKey);
}

TEST_F(AscendGatherNdTest, GatherNd_tiling_ascendc_B16_inddices32_supportneg_size32)
{
    GatherNdParam opsParamInfos;
    opsParamInfos.xDtype = ge::DT_FLOAT16;
    opsParamInfos.xShape = {{8, 4, 32}, {8, 4, 32}};
    opsParamInfos.indicesDtype = ge::DT_INT32;
    opsParamInfos.indicesShape = {{4, 2}, {4, 2}};
    opsParamInfos.yShape = {{4, 32}, {4, 32}};
    uint64_t expectTilingKey = 10000000001231100112;
    opsParamInfos.negativeIndexSupport = true;
    ExecuteTestCase(opsParamInfos, expectTilingKey);
}

TEST_F(AscendGatherNdTest, GatherNd_tiling_ascendc_B16_inddices32_supportneg_size64)
{
    GatherNdParam opsParamInfos;
    opsParamInfos.xDtype = ge::DT_FLOAT16;
    opsParamInfos.xShape = {{8, 127, 127}, {8, 127, 127}};
    opsParamInfos.indicesDtype = ge::DT_INT32;
    opsParamInfos.indicesShape = {{10000, 1700, 2}, {10000, 1700, 2}};
    opsParamInfos.yShape = {{10000, 1700, 127}, {10000, 1700, 127}};
    uint64_t expectTilingKey = 241;
    opsParamInfos.negativeIndexSupport = true;
    ExecuteTestCase(opsParamInfos, expectTilingKey);
}

TEST_F(AscendGatherNdTest, GatherNd_tiling_ascendc_B8_indices64_supportneg_size64)
{
    GatherNdParam opsParamInfos;
    opsParamInfos.xDtype = ge::DT_INT8;
    opsParamInfos.xShape = {{8, 127, 127}, {8, 127, 127}};
    opsParamInfos.indicesDtype = ge::DT_INT64;
    opsParamInfos.indicesShape = {{10000, 1700, 2}, {10000, 1700, 2}};
    opsParamInfos.yShape = {{10000, 1700, 127}, {10000, 1700, 127}};
    uint64_t expectTilingKey = 181;
    opsParamInfos.negativeIndexSupport = true;
    ExecuteTestCase(opsParamInfos, expectTilingKey);
}

TEST_F(AscendGatherNdTest, GatherNd_tiling_ascendc_B8_indices64_supportneg_size32)
{
    GatherNdParam opsParamInfos;
    opsParamInfos.xDtype = ge::DT_INT8;
    opsParamInfos.xShape = {{8, 4, 32}, {8, 4, 32}};
    opsParamInfos.indicesDtype = ge::DT_INT64;
    opsParamInfos.indicesShape = {{4, 2}, {4, 2}};
    opsParamInfos.yShape = {{4, 32}, {4, 32}};
    uint64_t expectTilingKey = 10000000001231100103;
    opsParamInfos.negativeIndexSupport = true;
    ExecuteTestCase(opsParamInfos, expectTilingKey);
}

TEST_F(AscendGatherNdTest, GatherNd_tiling_ascendc_B8_inddices32_supportneg_size32)
{
    GatherNdParam opsParamInfos;
    opsParamInfos.xDtype = ge::DT_INT8;
    opsParamInfos.xShape = {{8, 4, 32}, {8, 4, 32}};
    opsParamInfos.indicesDtype = ge::DT_INT32;
    opsParamInfos.indicesShape = {{4, 2}, {4, 2}};
    opsParamInfos.yShape = {{4, 32}, {4, 32}};
    uint64_t expectTilingKey = 10000000001231100102;
    opsParamInfos.negativeIndexSupport = true;
    ExecuteTestCase(opsParamInfos, expectTilingKey);
}

TEST_F(AscendGatherNdTest, GatherNd_tiling_ascendc_B8_inddices32_supportneg_size64)
{
    GatherNdParam opsParamInfos;
    opsParamInfos.xDtype = ge::DT_INT8;
    opsParamInfos.xShape = {{8, 127, 127}, {8, 127, 127}};
    opsParamInfos.indicesDtype = ge::DT_INT32;
    opsParamInfos.indicesShape = {{10000, 1700, 2}, {10000, 1700, 2}};
    opsParamInfos.yShape = {{10000, 1700, 127}, {10000, 1700, 127}};
    uint64_t expectTilingKey = 141;
    opsParamInfos.negativeIndexSupport = true;
    ExecuteTestCase(opsParamInfos, expectTilingKey);
}

TEST_F(AscendGatherNdTest, GatherNd_tiling_ascendc_indices_zero)
{
    GatherNdParam opsParamInfos;
    opsParamInfos.xDtype = ge::DT_INT8;
    opsParamInfos.xShape = {{8, 127, 127}, {8, 127, 127}};
    opsParamInfos.indicesDtype = ge::DT_INT32;
    opsParamInfos.indicesShape = {{0, 1700, 2}, {0, 1700, 2}};
    opsParamInfos.yShape = {{0, 1700, 127}, {0, 1700, 127}};
    uint64_t expectTilingKey = 10000000000231110000;
    opsParamInfos.negativeIndexSupport = true;
    ExecuteTestCase(opsParamInfos, expectTilingKey);
}

TEST_F(AscendGatherNdTest, GatherNd_tiling_ascendc_simd_inddices32_supportneg_size64)
{
    GatherNdParam opsParamInfos;
    opsParamInfos.xDtype = ge::DT_INT8;
    opsParamInfos.xShape = {{8, 127, 127}, {8, 127, 127}};
    opsParamInfos.indicesDtype = ge::DT_INT32;
    opsParamInfos.indicesShape = {{128, 12, 1}, {128, 12, 1}};
    opsParamInfos.yShape = {{128, 12, 127, 127}, {128, 12, 127, 127}};
    uint64_t expectTilingKey = 3100;
    opsParamInfos.negativeIndexSupport = true;
    ExecuteTestCase(opsParamInfos, expectTilingKey);
}

TEST_F(AscendGatherNdTest, GatherNd_tiling_ascendc_simd_all_load)
{
    GatherNdParam opsParamInfos;
    opsParamInfos.xDtype = ge::DT_FLOAT16;
    opsParamInfos.xShape = {{64, 6, 49}, {64, 6, 49}};
    opsParamInfos.indicesDtype = ge::DT_INT32;
    opsParamInfos.indicesShape = {{10, 33, 24, 1}, {10, 33, 24, 1}};
    opsParamInfos.yShape = {{10, 33, 24, 6, 49}, {10, 33, 24, 6, 49}};
    uint64_t expectTilingKey = 3000;
    opsParamInfos.negativeIndexSupport = false;
    ExecuteTestCase(opsParamInfos, expectTilingKey);
}

TEST_F(AscendGatherNdTest, GatherNd_tiling_ascendc_simd_all_load_vgather)
{
    GatherNdParam opsParamInfos;
    opsParamInfos.xDtype = ge::DT_FLOAT16;
    opsParamInfos.xShape = {{64, 6, 41}, {64, 6, 41}};
    opsParamInfos.indicesDtype = ge::DT_INT32;
    opsParamInfos.indicesShape = {{10, 33, 24, 1}, {10, 33, 24, 1}};
    opsParamInfos.yShape = {{10, 33, 24, 6, 41}, {10, 33, 24, 6, 41}};
    uint64_t expectTilingKey = 4000;
    opsParamInfos.negativeIndexSupport = false;
    ExecuteTestCase(opsParamInfos, expectTilingKey);
}
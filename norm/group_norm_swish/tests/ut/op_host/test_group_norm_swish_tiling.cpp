/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <limits>
#include "test_cube_util.h"
#include "kernel_run_context_facker.h"
#include "platform/platform_infos_def.h"
#include "../../../op_host/group_norm_swish_tiling.h"

class GroupNormSwishTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "GroupNormSwishTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "GroupNormSwishTiling TearDown" << std::endl; }
};

TEST_F(GroupNormSwishTiling, group_norm_swish_tiling_111)
{
    uint32_t N = 1024;
    uint32_t C = 32;
    uint32_t H = 1;
    uint32_t W = 1;
    uint32_t G = 32;
    gert::StorageShape x_shape = {{N, C, H, W}, {N, C, H, W}};
    gert::StorageShape gamma_shape = {{C}, {C}};
    gert::StorageShape beta_shape = {{C}, {C}};
    gert::StorageShape y_shape = {{N, C, H, W}, {N, C, H, W}};
    gert::StorageShape mean_shape = {{N * G}, {N * G}};
    gert::StorageShape rstd_shape = {{N * G}, {N * G}};

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 48}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::GroupNormSwishCompileInfo compile_info;

    std::string op_type("GroupNormSwish");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
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

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(3, 3)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&x_shape, &gamma_shape, &beta_shape})
                      .OutputShapes({&y_shape, &mean_shape, &rstd_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({
                          {"num_groups", Ops::NN::AnyValue::CreateFrom<int64_t>(32)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                          {"eps", Ops::NN::AnyValue::CreateFrom<float>(0.00001)},
                          {"activate_swish", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"swish_scale", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                      })
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
    ASSERT_EQ(tiling_key, 111);
}

TEST_F(GroupNormSwishTiling, group_norm_swish_tiling_112)
{
    uint32_t N = 32;
    uint32_t C = 64;
    uint32_t H = 1024;
    uint32_t W = 1;
    uint32_t G = 32;
    gert::StorageShape x_shape = {{N, C, H, W}, {N, C, H, W}};
    gert::StorageShape gamma_shape = {{C}, {C}};
    gert::StorageShape beta_shape = {{C}, {C}};
    gert::StorageShape y_shape = {{N, C, H, W}, {N, C, H, W}};
    gert::StorageShape mean_shape = {{N * G}, {N * G}};
    gert::StorageShape rstd_shape = {{N * G}, {N * G}};

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 48}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::GroupNormSwishCompileInfo compile_info;

    std::string op_type("GroupNormSwish");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
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

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(3, 3)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&x_shape, &gamma_shape, &beta_shape})
                      .OutputShapes({&y_shape, &mean_shape, &rstd_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({
                          {"num_groups", Ops::NN::AnyValue::CreateFrom<int64_t>(32)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                          {"eps", Ops::NN::AnyValue::CreateFrom<float>(0.00001)},
                          {"activate_swish", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"swish_scale", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                      })
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
    ASSERT_EQ(tiling_key, 112);
}

TEST_F(GroupNormSwishTiling, group_norm_swish_tiling_112_310P)
{
    uint32_t N = 32;
    uint32_t C = 64;
    uint32_t H = 1024;
    uint32_t W = 1;
    uint32_t G = 32;
    gert::StorageShape x_shape = {{N, C, H, W}, {N, C, H, W}};
    gert::StorageShape gamma_shape = {{C}, {C}};
    gert::StorageShape beta_shape = {{C}, {C}};
    gert::StorageShape y_shape = {{N, C, H, W}, {N, C, H, W}};
    gert::StorageShape mean_shape = {{N * G}, {N * G}};
    gert::StorageShape rstd_shape = {{N * G}, {N * G}};

    string compile_info_string = R"({"hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
                          "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 262144, "L2_SIZE": 16777216, "L1_SIZE": 1048576,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144,
                          "cube_core_cnt": 8, "vector_core_cnt": 7, "core_type_list": "CubeCore,VectorCore"}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> socversions = {{"Short_SoC_version", "Ascend310P"}, {"NpuArch", "2202"}};
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::GroupNormSwishCompileInfo compile_info;

    std::string op_type("GroupNormSwish");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(compile_info_string.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", socversions);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(3, 3)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&x_shape, &gamma_shape, &beta_shape})
                      .OutputShapes({&y_shape, &mean_shape, &rstd_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({
                          {"num_groups", Ops::NN::AnyValue::CreateFrom<int64_t>(32)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                          {"eps", Ops::NN::AnyValue::CreateFrom<float>(0.00001)},
                          {"activate_swish", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"swish_scale", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                      })
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", socversions);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 112);
}

TEST_F(GroupNormSwishTiling, group_norm_swish_tiling_113)
{
    uint32_t N = 32;
    uint32_t C = 128;
    uint32_t H = 8192;
    uint32_t W = 1;
    uint32_t G = 32;
    gert::StorageShape x_shape = {{N, C, H, W}, {N, C, H, W}};
    gert::StorageShape gamma_shape = {{C}, {C}};
    gert::StorageShape beta_shape = {{C}, {C}};
    gert::StorageShape y_shape = {{N, C, H, W}, {N, C, H, W}};
    gert::StorageShape mean_shape = {{N * G}, {N * G}};
    gert::StorageShape rstd_shape = {{N * G}, {N * G}};

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 48}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::GroupNormSwishCompileInfo compile_info;

    std::string op_type("GroupNormSwish");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
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

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(3, 3)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&x_shape, &gamma_shape, &beta_shape})
                      .OutputShapes({&y_shape, &mean_shape, &rstd_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({
                          {"num_groups", Ops::NN::AnyValue::CreateFrom<int64_t>(32)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                          {"eps", Ops::NN::AnyValue::CreateFrom<float>(0.00001)},
                          {"activate_swish", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"swish_scale", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                      })
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
    ASSERT_EQ(tiling_key, 113);
}

TEST_F(GroupNormSwishTiling, group_norm_swish_tiling_114)
{
    uint32_t N = 40;
    uint32_t C = 8200 * 32;
    uint32_t H = 64;
    uint32_t W = 1;
    uint32_t G = 32;
    gert::StorageShape x_shape = {{N, C, H, W}, {N, C, H, W}};
    gert::StorageShape gamma_shape = {{C}, {C}};
    gert::StorageShape beta_shape = {{C}, {C}};
    gert::StorageShape y_shape = {{N, C, H, W}, {N, C, H, W}};
    gert::StorageShape mean_shape = {{N * G}, {N * G}};
    gert::StorageShape rstd_shape = {{N * G}, {N * G}};

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 48}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::GroupNormSwishCompileInfo compile_info;

    std::string op_type("GroupNormSwish");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
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

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(3, 3)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&x_shape, &gamma_shape, &beta_shape})
                      .OutputShapes({&y_shape, &mean_shape, &rstd_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({
                          {"num_groups", Ops::NN::AnyValue::CreateFrom<int64_t>(32)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                          {"eps", Ops::NN::AnyValue::CreateFrom<float>(0.00001)},
                          {"activate_swish", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"swish_scale", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                      })
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
    ASSERT_EQ(tiling_key, 114);
}

TEST_F(GroupNormSwishTiling, group_norm_swish_tiling_121)
{
    uint32_t N = 1024;
    uint32_t C = 32;
    uint32_t H = 1;
    uint32_t W = 1;
    uint32_t G = 32;
    gert::StorageShape x_shape = {{N, C, H, W}, {N, C, H, W}};
    gert::StorageShape gamma_shape = {{C}, {C}};
    gert::StorageShape beta_shape = {{C}, {C}};
    gert::StorageShape y_shape = {{N, C, H, W}, {N, C, H, W}};
    gert::StorageShape mean_shape = {{N * G}, {N * G}};
    gert::StorageShape rstd_shape = {{N * G}, {N * G}};

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 48}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::GroupNormSwishCompileInfo compile_info;

    std::string op_type("GroupNormSwish");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
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

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(3, 3)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&x_shape, &gamma_shape, &beta_shape})
                      .OutputShapes({&y_shape, &mean_shape, &rstd_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({
                          {"num_groups", Ops::NN::AnyValue::CreateFrom<int64_t>(32)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                          {"eps", Ops::NN::AnyValue::CreateFrom<float>(0.00001)},
                          {"activate_swish", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"swish_scale", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                      })
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
    ASSERT_EQ(tiling_key, 121);
}

TEST_F(GroupNormSwishTiling, group_norm_swish_tiling_122)
{
    uint32_t N = 32;
    uint32_t C = 64;
    uint32_t H = 1024;
    uint32_t W = 1;
    uint32_t G = 32;
    gert::StorageShape x_shape = {{N, C, H, W}, {N, C, H, W}};
    gert::StorageShape gamma_shape = {{C}, {C}};
    gert::StorageShape beta_shape = {{C}, {C}};
    gert::StorageShape y_shape = {{N, C, H, W}, {N, C, H, W}};
    gert::StorageShape mean_shape = {{N * G}, {N * G}};
    gert::StorageShape rstd_shape = {{N * G}, {N * G}};

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 48}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::GroupNormSwishCompileInfo compile_info;

    std::string op_type("GroupNormSwish");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
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

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(3, 3)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&x_shape, &gamma_shape, &beta_shape})
                      .OutputShapes({&y_shape, &mean_shape, &rstd_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({
                          {"num_groups", Ops::NN::AnyValue::CreateFrom<int64_t>(32)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                          {"eps", Ops::NN::AnyValue::CreateFrom<float>(0.00001)},
                          {"activate_swish", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"swish_scale", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                      })
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
    ASSERT_EQ(tiling_key, 122);
}

TEST_F(GroupNormSwishTiling, group_norm_swish_tiling_123)
{
    uint32_t N = 32;
    uint32_t C = 128;
    uint32_t H = 8192;
    uint32_t W = 1;
    uint32_t G = 32;
    gert::StorageShape x_shape = {{N, C, H, W}, {N, C, H, W}};
    gert::StorageShape gamma_shape = {{C}, {C}};
    gert::StorageShape beta_shape = {{C}, {C}};
    gert::StorageShape y_shape = {{N, C, H, W}, {N, C, H, W}};
    gert::StorageShape mean_shape = {{N * G}, {N * G}};
    gert::StorageShape rstd_shape = {{N * G}, {N * G}};

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 48}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::GroupNormSwishCompileInfo compile_info;

    std::string op_type("GroupNormSwish");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
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

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(3, 3)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&x_shape, &gamma_shape, &beta_shape})
                      .OutputShapes({&y_shape, &mean_shape, &rstd_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({
                          {"num_groups", Ops::NN::AnyValue::CreateFrom<int64_t>(32)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                          {"eps", Ops::NN::AnyValue::CreateFrom<float>(0.00001)},
                          {"activate_swish", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"swish_scale", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                      })
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
    ASSERT_EQ(tiling_key, 123);
}

TEST_F(GroupNormSwishTiling, group_norm_swish_tiling_124)
{
    uint32_t N = 40;
    uint32_t C = 8200 * 32;
    uint32_t H = 64;
    uint32_t W = 1;
    uint32_t G = 32;
    gert::StorageShape x_shape = {{N, C, H, W}, {N, C, H, W}};
    gert::StorageShape gamma_shape = {{C}, {C}};
    gert::StorageShape beta_shape = {{C}, {C}};
    gert::StorageShape y_shape = {{N, C, H, W}, {N, C, H, W}};
    gert::StorageShape mean_shape = {{N * G}, {N * G}};
    gert::StorageShape rstd_shape = {{N * G}, {N * G}};

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 48}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::GroupNormSwishCompileInfo compile_info;

    std::string op_type("GroupNormSwish");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
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

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(3, 3)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&x_shape, &gamma_shape, &beta_shape})
                      .OutputShapes({&y_shape, &mean_shape, &rstd_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({
                          {"num_groups", Ops::NN::AnyValue::CreateFrom<int64_t>(32)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                          {"eps", Ops::NN::AnyValue::CreateFrom<float>(0.00001)},
                          {"activate_swish", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"swish_scale", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                      })
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
    ASSERT_EQ(tiling_key, 124);
}

TEST_F(GroupNormSwishTiling, group_norm_swish_tiling_211)
{
    uint32_t N = 1024;
    uint32_t C = 32;
    uint32_t H = 1;
    uint32_t W = 1;
    uint32_t G = 32;
    gert::StorageShape x_shape = {{N, C, H, W}, {N, C, H, W}};
    gert::StorageShape gamma_shape = {{C}, {C}};
    gert::StorageShape beta_shape = {{C}, {C}};
    gert::StorageShape y_shape = {{N, C, H, W}, {N, C, H, W}};
    gert::StorageShape mean_shape = {{N * G}, {N * G}};
    gert::StorageShape rstd_shape = {{N * G}, {N * G}};

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 48}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::GroupNormSwishCompileInfo compile_info;

    std::string op_type("GroupNormSwish");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
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

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(3, 3)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&x_shape, &gamma_shape, &beta_shape})
                      .OutputShapes({&y_shape, &mean_shape, &rstd_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({
                          {"num_groups", Ops::NN::AnyValue::CreateFrom<int64_t>(32)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                          {"eps", Ops::NN::AnyValue::CreateFrom<float>(0.00001)},
                          {"activate_swish", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"swish_scale", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                      })
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
    ASSERT_EQ(tiling_key, 211);
}

TEST_F(GroupNormSwishTiling, group_norm_swish_tiling_212)
{
    uint32_t N = 32;
    uint32_t C = 64;
    uint32_t H = 1024;
    uint32_t W = 1;
    uint32_t G = 32;
    gert::StorageShape x_shape = {{N, C, H, W}, {N, C, H, W}};
    gert::StorageShape gamma_shape = {{C}, {C}};
    gert::StorageShape beta_shape = {{C}, {C}};
    gert::StorageShape y_shape = {{N, C, H, W}, {N, C, H, W}};
    gert::StorageShape mean_shape = {{N * G}, {N * G}};
    gert::StorageShape rstd_shape = {{N * G}, {N * G}};

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 48}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::GroupNormSwishCompileInfo compile_info;

    std::string op_type("GroupNormSwish");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
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

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(3, 3)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&x_shape, &gamma_shape, &beta_shape})
                      .OutputShapes({&y_shape, &mean_shape, &rstd_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({
                          {"num_groups", Ops::NN::AnyValue::CreateFrom<int64_t>(32)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                          {"eps", Ops::NN::AnyValue::CreateFrom<float>(0.00001)},
                          {"activate_swish", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"swish_scale", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                      })
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
    ASSERT_EQ(tiling_key, 212);
}

TEST_F(GroupNormSwishTiling, group_norm_swish_tiling_213)
{
    uint32_t N = 32;
    uint32_t C = 128;
    uint32_t H = 8192;
    uint32_t W = 1;
    uint32_t G = 32;
    gert::StorageShape x_shape = {{N, C, H, W}, {N, C, H, W}};
    gert::StorageShape gamma_shape = {{C}, {C}};
    gert::StorageShape beta_shape = {{C}, {C}};
    gert::StorageShape y_shape = {{N, C, H, W}, {N, C, H, W}};
    gert::StorageShape mean_shape = {{N * G}, {N * G}};
    gert::StorageShape rstd_shape = {{N * G}, {N * G}};

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 48}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::GroupNormSwishCompileInfo compile_info;

    std::string op_type("GroupNormSwish");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
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

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(3, 3)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&x_shape, &gamma_shape, &beta_shape})
                      .OutputShapes({&y_shape, &mean_shape, &rstd_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({
                          {"num_groups", Ops::NN::AnyValue::CreateFrom<int64_t>(32)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                          {"eps", Ops::NN::AnyValue::CreateFrom<float>(0.00001)},
                          {"activate_swish", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"swish_scale", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                      })
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
    ASSERT_EQ(tiling_key, 213);
}

TEST_F(GroupNormSwishTiling, group_norm_swish_tiling_214)
{
    uint32_t N = 40;
    uint32_t C = 8200 * 32;
    uint32_t H = 64;
    uint32_t W = 1;
    uint32_t G = 32;
    gert::StorageShape x_shape = {{N, C, H, W}, {N, C, H, W}};
    gert::StorageShape gamma_shape = {{C}, {C}};
    gert::StorageShape beta_shape = {{C}, {C}};
    gert::StorageShape y_shape = {{N, C, H, W}, {N, C, H, W}};
    gert::StorageShape mean_shape = {{N * G}, {N * G}};
    gert::StorageShape rstd_shape = {{N * G}, {N * G}};

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 48}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    optiling::GroupNormSwishCompileInfo compile_info;

    std::string op_type("GroupNormSwish");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
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

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(3, 3)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&x_shape, &gamma_shape, &beta_shape})
                      .OutputShapes({&y_shape, &mean_shape, &rstd_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({
                          {"num_groups", Ops::NN::AnyValue::CreateFrom<int64_t>(32)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                          {"eps", Ops::NN::AnyValue::CreateFrom<float>(0.00001)},
                          {"activate_swish", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"swish_scale", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                      })
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
    ASSERT_EQ(tiling_key, 214);
}

TEST_F(GroupNormSwishTiling, group_norm_swish_tiling_y_shape_mismatch_failed)
{
    uint32_t N = 2;
    uint32_t C = 32;
    uint32_t H = 2;
    uint32_t W = 2;
    uint32_t G = 32;
    gert::StorageShape x_shape = {{N, C, H, W}, {N, C, H, W}};
    gert::StorageShape gamma_shape = {{C}, {C}};
    gert::StorageShape beta_shape = {{C}, {C}};
    // y numel(4) < x numel(256): 出参越界写场景(issue #7320)，应被拦截
    gert::StorageShape y_shape = {{N, W}, {N, W}};
    gert::StorageShape mean_shape = {{N * G}, {N * G}};
    gert::StorageShape rstd_shape = {{N * G}, {N * G}};

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 48}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    optiling::GroupNormSwishCompileInfo compile_info;

    std::string op_type("GroupNormSwish");
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
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(3, 3)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&x_shape, &gamma_shape, &beta_shape})
                      .OutputShapes({&y_shape, &mean_shape, &rstd_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({
                          {"num_groups", Ops::NN::AnyValue::CreateFrom<int64_t>(32)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                          {"eps", Ops::NN::AnyValue::CreateFrom<float>(0.00001)},
                          {"activate_swish", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"swish_scale", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                      })
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // y shape mismatch 必须返回失败
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(GroupNormSwishTiling, group_norm_swish_tiling_rank_9_failed)
{
    uint32_t N = 2;
    uint32_t C = 32;
    gert::StorageShape x_shape = {{N, C, 1, 1, 1, 1, 1, 1, 1}, {N, C, 1, 1, 1, 1, 1, 1, 1}};
    gert::StorageShape gamma_shape = {{C}, {C}};
    gert::StorageShape beta_shape = {{C}, {C}};
    gert::StorageShape y_shape = {{N, C, 1, 1, 1, 1, 1, 1, 1}, {N, C, 1, 1, 1, 1, 1, 1, 1}};
    gert::StorageShape mean_shape = {{N * C}, {N * C}};
    gert::StorageShape rstd_shape = {{N * C}, {N * C}};

    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 48}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    optiling::GroupNormSwishCompileInfo compile_info;

    std::string op_type("GroupNormSwish");
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
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(3, 3)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&x_shape, &gamma_shape, &beta_shape})
                      .OutputShapes({&y_shape, &mean_shape, &rstd_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({
                          {"num_groups", Ops::NN::AnyValue::CreateFrom<int64_t>(32)},
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                          {"eps", Ops::NN::AnyValue::CreateFrom<float>(0.00001)},
                          {"activate_swish", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                          {"swish_scale", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                      })
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // rank=9 超出资料声明的 2-8 维，必须返回失败(issue #7322)
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(GroupNormSwishTiling, group_norm_swish_tiling_empty_x_failed)
{
    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 48}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    optiling::GroupNormSwishCompileInfo compile_info;

    std::string op_type("GroupNormSwish");
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

    const std::vector<std::vector<uint32_t>> empty_shapes = {{0, 32, 4, 4}, {2, 0, 4, 4}, {2, 32, 0, 4}, {2, 32, 4, 0}};
    for (const auto& shape : empty_shapes) {
        uint32_t N = shape[0];
        uint32_t C = shape[1];
        uint32_t H = shape[2];
        uint32_t W = shape[3];
        uint32_t G = 2;
        gert::StorageShape x_shape = {{N, C, H, W}, {N, C, H, W}};
        gert::StorageShape gamma_shape = {{C}, {C}};
        gert::StorageShape beta_shape = {{C}, {C}};
        gert::StorageShape y_shape = {{N, C, H, W}, {N, C, H, W}};
        gert::StorageShape mean_shape = {{N * G}, {N * G}};
        gert::StorageShape rstd_shape = {{N * G}, {N * G}};

        auto param = gert::TilingData::CreateCap(4096);
        auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
        auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
        ASSERT_NE(param, nullptr);
        auto holder = gert::TilingContextFaker()
                          .NodeIoNum(3, 3)
                          .IrInstanceNum({1, 1, 1})
                          .InputShapes({&x_shape, &gamma_shape, &beta_shape})
                          .OutputShapes({&y_shape, &mean_shape, &rstd_shape})
                          .CompileInfo(&compile_info)
                          .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                          .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeAttrs({
                              {"num_groups", Ops::NN::AnyValue::CreateFrom<int64_t>(G)},
                              {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                              {"eps", Ops::NN::AnyValue::CreateFrom<float>(0.00001)},
                              {"activate_swish", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                              {"swish_scale", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                          })
                          .TilingData(param.get())
                          .Workspace(ws_size)
                          .Build();

        gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
        ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
        tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
        tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
        tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
        tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

        EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED)
            << "empty x shape: [" << N << ", " << C << ", " << H << ", " << W << "]";
    }
}

TEST_F(GroupNormSwishTiling, group_norm_swish_tiling_eps_invalid_failed)
{
    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 48}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    optiling::GroupNormSwishCompileInfo compile_info;

    std::string op_type("GroupNormSwish");
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

    // eps 为 NaN/Inf/负数时，tiling 必须返回失败(issue cann/ops-nn#8559)
    const std::vector<float> invalidEpsList = {std::numeric_limits<float>::quiet_NaN(),
                                               std::numeric_limits<float>::infinity(), -1.0f};
    for (const float eps : invalidEpsList) {
        uint32_t N = 1024;
        uint32_t C = 32;
        uint32_t H = 1;
        uint32_t W = 1;
        uint32_t G = 32;
        gert::StorageShape x_shape = {{N, C, H, W}, {N, C, H, W}};
        gert::StorageShape gamma_shape = {{C}, {C}};
        gert::StorageShape beta_shape = {{C}, {C}};
        gert::StorageShape y_shape = {{N, C, H, W}, {N, C, H, W}};
        gert::StorageShape mean_shape = {{N * G}, {N * G}};
        gert::StorageShape rstd_shape = {{N * G}, {N * G}};

        auto param = gert::TilingData::CreateCap(4096);
        auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
        auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
        ASSERT_NE(param, nullptr);
        auto holder = gert::TilingContextFaker()
                          .NodeIoNum(3, 3)
                          .IrInstanceNum({1, 1, 1})
                          .InputShapes({&x_shape, &gamma_shape, &beta_shape})
                          .OutputShapes({&y_shape, &mean_shape, &rstd_shape})
                          .CompileInfo(&compile_info)
                          .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                          .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeAttrs({
                              {"num_groups", Ops::NN::AnyValue::CreateFrom<int64_t>(G)},
                              {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                              {"eps", Ops::NN::AnyValue::CreateFrom<float>(eps)},
                              {"activate_swish", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                              {"swish_scale", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                          })
                          .TilingData(param.get())
                          .Workspace(ws_size)
                          .Build();

        gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
        ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
        tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
        tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
        tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
        tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

        EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED) << "invalid eps: " << eps;
    }

    // eps 为正零或负零时，tiling 仍应成功(issue cann/ops-nn#8559)
    for (const float eps : {0.0f, -0.0f}) {
        uint32_t N = 1024;
        uint32_t C = 32;
        uint32_t H = 1;
        uint32_t W = 1;
        uint32_t G = 32;
        gert::StorageShape x_shape = {{N, C, H, W}, {N, C, H, W}};
        gert::StorageShape gamma_shape = {{C}, {C}};
        gert::StorageShape beta_shape = {{C}, {C}};
        gert::StorageShape y_shape = {{N, C, H, W}, {N, C, H, W}};
        gert::StorageShape mean_shape = {{N * G}, {N * G}};
        gert::StorageShape rstd_shape = {{N * G}, {N * G}};

        auto param = gert::TilingData::CreateCap(4096);
        auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
        auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
        ASSERT_NE(param, nullptr);
        auto holder = gert::TilingContextFaker()
                          .NodeIoNum(3, 3)
                          .IrInstanceNum({1, 1, 1})
                          .InputShapes({&x_shape, &gamma_shape, &beta_shape})
                          .OutputShapes({&y_shape, &mean_shape, &rstd_shape})
                          .CompileInfo(&compile_info)
                          .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                          .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeAttrs({
                              {"num_groups", Ops::NN::AnyValue::CreateFrom<int64_t>(G)},
                              {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                              {"eps", Ops::NN::AnyValue::CreateFrom<float>(eps)},
                              {"activate_swish", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                              {"swish_scale", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                          })
                          .TilingData(param.get())
                          .Workspace(ws_size)
                          .Build();

        gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
        ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
        tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
        tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
        tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
        tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

        EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS) << "valid eps: " << eps;
    }
}

TEST_F(GroupNormSwishTiling, group_norm_swish_tiling_swish_scale_invalid_failed)
{
    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 48}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    optiling::GroupNormSwishCompileInfo compile_info;

    std::string op_type("GroupNormSwish");
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

    uint32_t N = 1024;
    uint32_t C = 32;
    uint32_t H = 1;
    uint32_t W = 1;
    uint32_t G = 32;
    gert::StorageShape x_shape = {{N, C, H, W}, {N, C, H, W}};
    gert::StorageShape gamma_shape = {{C}, {C}};
    gert::StorageShape beta_shape = {{C}, {C}};
    gert::StorageShape y_shape = {{N, C, H, W}, {N, C, H, W}};
    gert::StorageShape mean_shape = {{N * G}, {N * G}};
    gert::StorageShape rstd_shape = {{N * G}, {N * G}};

    // activateSwish 为 true 时，swishScale 为 NaN/Inf，tiling 必须返回失败(issue cann/ops-nn#8559)
    const std::vector<float> invalidScaleList = {std::numeric_limits<float>::quiet_NaN(),
                                                 std::numeric_limits<float>::infinity()};
    for (const float swishScale : invalidScaleList) {
        auto param = gert::TilingData::CreateCap(4096);
        auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
        auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
        ASSERT_NE(param, nullptr);
        auto holder = gert::TilingContextFaker()
                          .NodeIoNum(3, 3)
                          .IrInstanceNum({1, 1, 1})
                          .InputShapes({&x_shape, &gamma_shape, &beta_shape})
                          .OutputShapes({&y_shape, &mean_shape, &rstd_shape})
                          .CompileInfo(&compile_info)
                          .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                          .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeAttrs({
                              {"num_groups", Ops::NN::AnyValue::CreateFrom<int64_t>(G)},
                              {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                              {"eps", Ops::NN::AnyValue::CreateFrom<float>(0.00001)},
                              {"activate_swish", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                              {"swish_scale", Ops::NN::AnyValue::CreateFrom<float>(swishScale)},
                          })
                          .TilingData(param.get())
                          .Workspace(ws_size)
                          .Build();

        gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
        ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
        tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
        tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
        tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
        tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

        EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED) << "invalid swish_scale: " << swishScale;
    }

    // activateSwish 为 false 时，swishScale 不参与计算，取 NaN 不应被拦截
    {
        auto param = gert::TilingData::CreateCap(4096);
        auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
        auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
        ASSERT_NE(param, nullptr);
        auto holder = gert::TilingContextFaker()
                          .NodeIoNum(3, 3)
                          .IrInstanceNum({1, 1, 1})
                          .InputShapes({&x_shape, &gamma_shape, &beta_shape})
                          .OutputShapes({&y_shape, &mean_shape, &rstd_shape})
                          .CompileInfo(&compile_info)
                          .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                          .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeAttrs({
                              {"num_groups", Ops::NN::AnyValue::CreateFrom<int64_t>(G)},
                              {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                              {"eps", Ops::NN::AnyValue::CreateFrom<float>(0.00001)},
                              {"activate_swish", Ops::NN::AnyValue::CreateFrom<bool>(false)},
                              {"swish_scale",
                               Ops::NN::AnyValue::CreateFrom<float>(std::numeric_limits<float>::quiet_NaN())},
                          })
                          .TilingData(param.get())
                          .Workspace(ws_size)
                          .Build();

        gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
        ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
        tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
        tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
        tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
        tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

        EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    }
}

// 属性缺失时 GetAttrPointer 返回空指针，tiling 必须返回失败而不是解引用空指针
TEST_F(GroupNormSwishTiling, group_norm_swish_tiling_attrs_missing_failed)
{
    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 48}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    optiling::GroupNormSwishCompileInfo compile_info;

    std::string op_type("GroupNormSwish");
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

    uint32_t N = 1024;
    uint32_t C = 32;
    uint32_t H = 1;
    uint32_t W = 1;
    uint32_t G = 32;
    gert::StorageShape x_shape = {{N, C, H, W}, {N, C, H, W}};
    gert::StorageShape gamma_shape = {{C}, {C}};
    gert::StorageShape beta_shape = {{C}, {C}};
    gert::StorageShape y_shape = {{N, C, H, W}, {N, C, H, W}};
    gert::StorageShape mean_shape = {{N * G}, {N * G}};
    gert::StorageShape rstd_shape = {{N * G}, {N * G}};

    // eps 属性缺失：GetAttrPointer(ATTR_IDX_EPSILON) 返回空指针，tiling 返回失败
    {
        auto param = gert::TilingData::CreateCap(4096);
        auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
        auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
        ASSERT_NE(param, nullptr);
        auto holder = gert::TilingContextFaker()
                          .NodeIoNum(3, 3)
                          .IrInstanceNum({1, 1, 1})
                          .InputShapes({&x_shape, &gamma_shape, &beta_shape})
                          .OutputShapes({&y_shape, &mean_shape, &rstd_shape})
                          .CompileInfo(&compile_info)
                          .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                          .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeAttrs({
                              {"num_groups", Ops::NN::AnyValue::CreateFrom<int64_t>(G)},
                              {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                          })
                          .TilingData(param.get())
                          .Workspace(ws_size)
                          .Build();

        gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
        ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
        tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
        tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
        tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
        tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

        EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED) << "eps attr is missing";
    }

    // activate_swish 属性缺失：GetAttrPointer(ATTR_IDX_ACTIVATE_SWISH) 返回空指针，tiling 返回失败
    {
        auto param = gert::TilingData::CreateCap(4096);
        auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
        auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
        ASSERT_NE(param, nullptr);
        auto holder = gert::TilingContextFaker()
                          .NodeIoNum(3, 3)
                          .IrInstanceNum({1, 1, 1})
                          .InputShapes({&x_shape, &gamma_shape, &beta_shape})
                          .OutputShapes({&y_shape, &mean_shape, &rstd_shape})
                          .CompileInfo(&compile_info)
                          .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                          .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeAttrs({
                              {"num_groups", Ops::NN::AnyValue::CreateFrom<int64_t>(G)},
                              {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                              {"eps", Ops::NN::AnyValue::CreateFrom<float>(0.00001)},
                          })
                          .TilingData(param.get())
                          .Workspace(ws_size)
                          .Build();

        gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
        ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
        tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
        tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
        tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
        tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

        EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED) << "activate_swish attr is missing";
    }

    // swish_scale 属性缺失：无论 activate_swish 取值如何，指针均需判空
    // (SetAttrParams 无条件读取 swishScale，因此 activate_swish=false 也不能放行空指针)
    for (const bool activateSwish : {true, false}) {
        auto param = gert::TilingData::CreateCap(4096);
        auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
        auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
        ASSERT_NE(param, nullptr);
        auto holder = gert::TilingContextFaker()
                          .NodeIoNum(3, 3)
                          .IrInstanceNum({1, 1, 1})
                          .InputShapes({&x_shape, &gamma_shape, &beta_shape})
                          .OutputShapes({&y_shape, &mean_shape, &rstd_shape})
                          .CompileInfo(&compile_info)
                          .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                          .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                          .NodeAttrs({
                              {"num_groups", Ops::NN::AnyValue::CreateFrom<int64_t>(G)},
                              {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCHW")},
                              {"eps", Ops::NN::AnyValue::CreateFrom<float>(0.00001)},
                              {"activate_swish", Ops::NN::AnyValue::CreateFrom<bool>(activateSwish)},
                          })
                          .TilingData(param.get())
                          .Workspace(ws_size)
                          .Build();

        gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
        ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
        tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
        tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
        tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
        tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

        EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED)
            << "swish_scale attr is missing, activate_swish = " << activateSwish;
    }
}

TEST_F(GroupNormSwishTiling, group_norm_swish_tiling_data_format_validation)
{
    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 48}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);
    fe::PlatFormInfos platform_info;
    ASSERT_TRUE(platform_info.Init());
    platform_info.SetPlatformRes("SoCInfo", soc_infos);
    platform_info.SetPlatformRes("AICoreSpec", aicore_spec);
    platform_info.SetCoreNumByCoreType("AICore");
    platform_info.SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    optiling::GroupNormSwishCompileInfo compile_info;
    auto op_impl = gert::OpImplRegistry::GetInstance().GetOpImpl("GroupNormSwish");
    ASSERT_NE(op_impl, nullptr);
    ASSERT_NE(op_impl->tiling, nullptr);

    gert::StorageShape x_shape = {{2, 3, 4}, {2, 3, 4}};
    gert::StorageShape gamma_shape = {{3}, {3}};
    gert::StorageShape beta_shape = {{3}, {3}};
    gert::StorageShape y_shape = {{2, 3, 4}, {2, 3, 4}};
    gert::StorageShape mean_shape = {{2, 1}, {2, 1}};
    gert::StorageShape rstd_shape = {{2, 1}, {2, 1}};
    const auto run_case = [&](const std::string& data_format, ge::graphStatus expected) {
        SCOPED_TRACE(data_format);
        auto param = gert::TilingData::CreateCap(4096);
        auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
        ASSERT_NE(param, nullptr);
        ASSERT_NE(workspace_size_holder, nullptr);
        auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
        gert::TilingContextFaker faker;
        faker.NodeIoNum(3, 3)
            .IrInstanceNum({1, 1, 1})
            .InputShapes({&x_shape, &gamma_shape, &beta_shape})
            .OutputShapes({&y_shape, &mean_shape, &rstd_shape})
            .CompileInfo(&compile_info)
            .PlatformInfo(reinterpret_cast<char*>(&platform_info))
            .NodeAttrs({
                {"num_groups", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>(data_format)},
                {"eps", Ops::NN::AnyValue::CreateFrom<float>(1e-5f)},
                {"activate_swish", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                {"swish_scale", Ops::NN::AnyValue::CreateFrom<float>(1.0f)},
            })
            .TilingData(param.get())
            .Workspace(ws_size);
        for (int32_t i = 0; i < 3; ++i) {
            faker.NodeInputTd(i, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
            faker.NodeOutputTd(i, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND);
        }
        auto holder = faker.Build();
        EXPECT_EQ(op_impl->tiling(holder.GetContext<gert::TilingContext>()), expected);
    };

    run_case("NCHW", ge::GRAPH_SUCCESS);
    for (const std::string data_format : {"ABCDEFG", "NHWC", ""}) {
        run_case(data_format, ge::GRAPH_FAILED);
    }
}

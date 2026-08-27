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
#include "../../../op_host/flat_quant_tiling.h"
#include "platform/platform_infos_def.h"
#include "ut_op_util.h"

using namespace ut_util;
using namespace std;
using namespace ge;

namespace {
class FlatQuantTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "FlatQuantTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "FlatQuantTiling TearDown" << std::endl; }
};

TEST_F(FlatQuantTiling, flat_quant_tiling_001)
{
    gert::StorageShape x_shape = {{16, 64, 64}, {16, 64, 64}};
    gert::StorageShape kronecker_p1_shape = {{64, 64}, {64, 64}};
    gert::StorageShape kronecker_p2_shape = {{64, 64}, {64, 64}};
    gert::StorageShape out_shape = {{16, 64, 64}, {16, 64, 64}};
    gert::StorageShape quant_scale_shape = {{16}, {16}};

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
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    struct FlatQuantCompileInfo {
        int64_t aicNum = 24;
        int64_t aivNum = 48;
        NpuArch npuArch = NpuArch::DAV_2201;
    } compile_info;

    std::string op_type("FlatQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(8192);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType("FlatQuant")
                      .NodeIoNum(3, 2)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&x_shape, &kronecker_p1_shape, &kronecker_p2_shape})
                      .OutputShapes({&out_shape, &quant_scale_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeAttrs({{"clip_ratio", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                                  {"dst_dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(ge::DT_INT4)},
                                  {"dst_type_max", Ops::NN::AnyValue::CreateFrom<float>(0.0)},
                                  {"group_list_type", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT4, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);

    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 2);
}

TEST_F(FlatQuantTiling, flat_quant_tiling_002)
{
    gert::StorageShape x_shape = {{16, 64, 64}, {16, 64, 64}};
    gert::StorageShape kronecker_p1_shape = {{64, 64}, {64, 64}};
    gert::StorageShape kronecker_p2_shape = {{64, 64}, {64, 64}};
    gert::StorageShape out_shape = {{16, 64, 64}, {16, 64, 64}};
    gert::StorageShape quant_scale_shape = {{16}, {16}};

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
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    struct FlatQuantCompileInfo {
        int64_t aicNum = 24;
        int64_t aivNum = 48;
        NpuArch npuArch = NpuArch::DAV_2201;
    } compile_info;

    std::string op_type("FlatQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(8192);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType("FlatQuant")
                      .NodeIoNum(3, 2)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&x_shape, &kronecker_p1_shape, &kronecker_p2_shape})
                      .OutputShapes({&out_shape, &quant_scale_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeAttrs({{"clip_ratio", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                                  {"dst_dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(ge::DT_INT4)},
                                  {"dst_type_max", Ops::NN::AnyValue::CreateFrom<float>(0.0)},
                                  {"group_list_type", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT4, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);

    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 2);
}

TEST_F(FlatQuantTiling, flat_quant_tiling_003)
{
    gert::StorageShape x_shape = {{16, 64, 64}, {16, 64, 64}};
    gert::StorageShape kronecker_p1_shape = {{64, 64}, {64, 64}};
    gert::StorageShape kronecker_p2_shape = {{64, 64}, {64, 64}};
    gert::StorageShape out_shape = {{16, 64, 64}, {16, 64, 64}};
    gert::StorageShape quant_scale_shape = {{16}, {16}};

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
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    struct FlatQuantCompileInfo {
        int64_t aicNum = 24;
        int64_t aivNum = 48;
        NpuArch npuArch = NpuArch::DAV_2201;
    } compile_info;

    std::string op_type("FlatQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(8192);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType("FlatQuant")
                      .NodeIoNum(3, 2)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&x_shape, &kronecker_p1_shape, &kronecker_p2_shape})
                      .OutputShapes({&out_shape, &quant_scale_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeAttrs({{"clip_ratio", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                                  {"dst_dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(ge::DT_INT4)},
                                  {"dst_type_max", Ops::NN::AnyValue::CreateFrom<float>(0.0)},
                                  {"group_list_type", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT4, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);

    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 2);
}

TEST_F(FlatQuantTiling, flat_quant_tiling_004)
{
    gert::StorageShape x_shape = {{1, 64, 64}, {1, 64, 64}};
    gert::StorageShape kronecker_p1_shape = {{64, 64}, {64, 64}};
    gert::StorageShape kronecker_p2_shape = {{64, 64}, {64, 64}};
    gert::StorageShape out_shape = {{1, 64, 64}, {1, 64, 64}};
    gert::StorageShape quant_scale_shape = {{1}, {1}};

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
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    struct FlatQuantCompileInfo {
        int64_t aicNum = 24;
        int64_t aivNum = 48;
        NpuArch npuArch = NpuArch::DAV_2201;
    } compile_info;

    std::string op_type("FlatQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(8192);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType("FlatQuant")
                      .NodeIoNum(3, 2)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&x_shape, &kronecker_p1_shape, &kronecker_p2_shape})
                      .OutputShapes({&out_shape, &quant_scale_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeAttrs({{"clip_ratio", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                                  {"dst_dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(ge::DT_INT4)},
                                  {"dst_type_max", Ops::NN::AnyValue::CreateFrom<float>(0.0)},
                                  {"group_list_type", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT4, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);

    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 1);
}

TEST_F(FlatQuantTiling, flat_quant_tiling_005)
{
    gert::StorageShape x_shape = {{16, 64, 64}, {16, 64, 64}};
    gert::StorageShape kronecker_p1_shape = {{64, 64}, {64, 64}};
    gert::StorageShape kronecker_p2_shape = {{64, 64}, {64, 64}};
    gert::StorageShape out_shape = {{16, 64, 64}, {16, 64, 64}};
    gert::StorageShape quant_scale_shape = {{16}, {16}};

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
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    struct FlatQuantCompileInfo {
        int64_t aicNum = 24;
        int64_t aivNum = 48;
        NpuArch npuArch = NpuArch::DAV_2201;
    } compile_info;

    std::string op_type("FlatQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(8192);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType("FlatQuant")
                      .NodeIoNum(3, 2)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&x_shape, &kronecker_p1_shape, &kronecker_p2_shape})
                      .OutputShapes({&out_shape, &quant_scale_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeAttrs({{"clip_ratio", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                                  {"dst_dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(ge::DT_INT4)},
                                  {"dst_type_max", Ops::NN::AnyValue::CreateFrom<float>(0.0)},
                                  {"group_list_type", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT4, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(FlatQuantTiling, flat_quant_tiling_006)
{
    gert::StorageShape x_shape = {{64, 64}, {64, 64}};
    gert::StorageShape kronecker_p1_shape = {{64, 64}, {64, 64}};
    gert::StorageShape kronecker_p2_shape = {{64, 64}, {64, 64}};
    gert::StorageShape out_shape = {{64, 64}, {64, 64}};
    gert::StorageShape quant_scale_shape = {{1}, {1}};

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
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    struct FlatQuantCompileInfo {
        int64_t aicNum = 24;
        int64_t aivNum = 48;
        NpuArch npuArch = NpuArch::DAV_2201;
    } compile_info;

    std::string op_type("FlatQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(8192);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType("FlatQuant")
                      .NodeIoNum(3, 2)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&x_shape, &kronecker_p1_shape, &kronecker_p2_shape})
                      .OutputShapes({&out_shape, &quant_scale_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeAttrs({{"clip_ratio", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                                  {"dst_dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(ge::DT_INT4)},
                                  {"dst_type_max", Ops::NN::AnyValue::CreateFrom<float>(0.0)},
                                  {"group_list_type", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT4, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(FlatQuantTiling, flat_quant_tiling_007)
{
    gert::StorageShape x_shape = {{16, 64, 64}, {16, 64, 64}};
    gert::StorageShape kronecker_p1_shape = {{32, 32}, {32, 32}};
    gert::StorageShape kronecker_p2_shape = {{64, 64}, {64, 64}};
    gert::StorageShape out_shape = {{16, 64, 64}, {16, 64, 64}};
    gert::StorageShape quant_scale_shape = {{16}, {16}};

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
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    struct FlatQuantCompileInfo {
        int64_t aicNum = 24;
        int64_t aivNum = 48;
        NpuArch npuArch = NpuArch::DAV_2201;
    } compile_info;

    std::string op_type("FlatQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(8192);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType("FlatQuant")
                      .NodeIoNum(3, 2)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&x_shape, &kronecker_p1_shape, &kronecker_p2_shape})
                      .OutputShapes({&out_shape, &quant_scale_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeAttrs({{"clip_ratio", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                                  {"dst_dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(ge::DT_INT4)},
                                  {"dst_type_max", Ops::NN::AnyValue::CreateFrom<float>(0.0)},
                                  {"group_list_type", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT4, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(FlatQuantTiling, flat_quant_tiling_008)
{
    gert::StorageShape x_shape = {{16, 256, 256}, {16, 256, 256}};
    gert::StorageShape kronecker_p1_shape = {{256, 256}, {256, 256}};
    gert::StorageShape kronecker_p2_shape = {{256, 256}, {256, 256}};
    gert::StorageShape out_shape = {{16, 256, 256}, {16, 256, 256}};
    gert::StorageShape quant_scale_shape = {{16}, {16}};

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
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    struct FlatQuantCompileInfo {
        int64_t aicNum = 24;
        int64_t aivNum = 48;
        NpuArch npuArch = NpuArch::DAV_2201;
    } compile_info;

    std::string op_type("FlatQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(8192);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType("FlatQuant")
                      .NodeIoNum(3, 2)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&x_shape, &kronecker_p1_shape, &kronecker_p2_shape})
                      .OutputShapes({&out_shape, &quant_scale_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeAttrs({{"clip_ratio", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                                  {"dst_dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(ge::DT_INT4)},
                                  {"dst_type_max", Ops::NN::AnyValue::CreateFrom<float>(0.0)},
                                  {"group_list_type", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT4, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
}

TEST_F(FlatQuantTiling, flat_quant_tiling_009)
{
    gert::StorageShape x_shape = {{16, 64, 64}, {16, 64, 64}};
    gert::StorageShape kronecker_p1_shape = {{64, 64}, {64, 64}};
    gert::StorageShape kronecker_p2_shape = {{64, 64}, {64, 64}};
    gert::StorageShape out_shape = {{16, 64, 64}, {16, 64, 64}};
    gert::StorageShape quant_scale_shape = {{16}, {16}};

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
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    struct FlatQuantCompileInfo {
        int64_t aicNum = 24;
        int64_t aivNum = 48;
        NpuArch npuArch = NpuArch::DAV_2201;
    } compile_info;

    std::string op_type("FlatQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(8192);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType("FlatQuant")
                      .NodeIoNum(3, 2)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&x_shape, &kronecker_p1_shape, &kronecker_p2_shape})
                      .OutputShapes({&out_shape, &quant_scale_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeAttrs({{"clip_ratio", Ops::NN::AnyValue::CreateFrom<float>(1.5)},
                                  {"dst_dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(ge::DT_INT4)},
                                  {"dst_type_max", Ops::NN::AnyValue::CreateFrom<float>(0.0)},
                                  {"group_list_type", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT4, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(FlatQuantTiling, flat_quant_tiling_010)
{
    gert::StorageShape x_shape = {{16, 64, 64}, {16, 64, 64}};
    gert::StorageShape kronecker_p1_shape = {{64, 64}, {64, 64}};
    gert::StorageShape kronecker_p2_shape = {{64, 64}, {64, 64}};
    gert::StorageShape group_list_shape = {{1}, {1}};
    gert::StorageShape out_shape = {{16, 64, 64}, {16, 64, 64}};
    gert::StorageShape quant_scale_shape = {{16}, {16}};

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
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    struct FlatQuantCompileInfo {
        int64_t aicNum = 24;
        int64_t aivNum = 48;
        NpuArch npuArch = NpuArch::DAV_2201;
    } compile_info;

    std::string op_type("FlatQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(8192);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType("FlatQuant")
                      .NodeIoNum(4, 2)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&x_shape, &kronecker_p1_shape, &kronecker_p2_shape, &group_list_shape})
                      .OutputShapes({&out_shape, &quant_scale_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeAttrs({{"clip_ratio", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                                  {"dst_dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(ge::DT_INT4)},
                                  {"dst_type_max", Ops::NN::AnyValue::CreateFrom<float>(0.0)},
                                  {"group_list_type", Ops::NN::AnyValue::CreateFrom<int64_t>(1)}})
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT4, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
}

TEST_F(FlatQuantTiling, 910d_flat_quant_int4_tiling_001)
{
    gert::StorageShape x_shape = {{16, 64, 128}, {16, 64, 128}};
    gert::StorageShape kronecker_p1_shape = {{64, 64}, {64, 64}};
    gert::StorageShape kronecker_p2_shape = {{128, 128}, {128, 128}};
    gert::StorageShape out_shape = {{16, 64, 128}, {16, 64, 128}};
    gert::StorageShape quant_scale_shape = {{16}, {16}};

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
                                            "CORE_NUM": 64
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
    struct FlatQuantCompileInfo {
        int64_t aicNum = 32;
        int64_t aivNum = 64;
        NpuArch npuArch = NpuArch::DAV_3510;
    } compile_info;

    std::string op_type("FlatQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(8192);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType("FlatQuant")
                      .NodeIoNum(3, 2)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&x_shape, &kronecker_p1_shape, &kronecker_p2_shape})
                      .OutputShapes({&out_shape, &quant_scale_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeAttrs({{"clip_ratio", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                                  {"dst_dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(ge::DT_INT4)},
                                  {"dst_type_max", Ops::NN::AnyValue::CreateFrom<float>(0.0)},
                                  {"group_list_type", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT4, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
}

TEST_F(FlatQuantTiling, 910d_flat_quant_int4_tiling_002)
{
    gert::StorageShape x_shape = {{16, 256, 256}, {16, 256, 256}};
    gert::StorageShape kronecker_p1_shape = {{256, 256}, {256, 256}};
    gert::StorageShape kronecker_p2_shape = {{256, 256}, {256, 256}};
    gert::StorageShape out_shape = {{16, 256, 256}, {16, 256, 256}};
    gert::StorageShape quant_scale_shape = {{16}, {16}};

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
                                            "CORE_NUM": 64
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
    struct FlatQuantCompileInfo {
        int64_t aicNum = 32;
        int64_t aivNum = 64;
        NpuArch npuArch = NpuArch::DAV_3510;
    } compile_info;

    std::string op_type("FlatQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(8192);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType("FlatQuant")
                      .NodeIoNum(3, 2)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&x_shape, &kronecker_p1_shape, &kronecker_p2_shape})
                      .OutputShapes({&out_shape, &quant_scale_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeAttrs({{"clip_ratio", Ops::NN::AnyValue::CreateFrom<float>(1.5)},
                                  {"dst_dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(ge::DT_INT4)},
                                  {"dst_type_max", Ops::NN::AnyValue::CreateFrom<float>(0.0)},
                                  {"group_list_type", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_INT4, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(FlatQuantTiling, 910d_flat_quant_mxfp4_tiling_001)
{
    gert::StorageShape x_shape = {{16, 64, 64}, {16, 64, 64}};
    gert::StorageShape kronecker_p1_shape = {{64, 64}, {64, 64}};
    gert::StorageShape kronecker_p2_shape = {{64, 64}, {64, 64}};
    gert::StorageShape out_shape = {{16, 4096}, {16, 4096}};
    gert::StorageShape quant_scale_shape = {{16, 64, 2}, {16, 64, 2}};

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
                                            "CORE_NUM": 64
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
    struct FlatQuantCompileInfo {
        int64_t aicNum = 32;
        int64_t aivNum = 64;
        NpuArch npuArch = NpuArch::DAV_3510;
    } compile_info;

    std::string op_type("FlatQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(8192);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType("FlatQuant")
                      .NodeIoNum(3, 2)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&x_shape, &kronecker_p1_shape, &kronecker_p2_shape})
                      .OutputShapes({&out_shape, &quant_scale_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeAttrs({{"clip_ratio", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                                  {"dst_dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(ge::DT_FLOAT4_E2M1)},
                                  {"dst_type_max", Ops::NN::AnyValue::CreateFrom<float>(0.0)},
                                  {"group_list_type", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
}
TEST_F(FlatQuantTiling, 910d_flat_quant_mxfp4_tiling_002)
{
    gert::StorageShape x_shape = {{16, 128, 64}, {16, 128, 64}};
    gert::StorageShape kronecker_p1_shape = {{128, 128}, {128, 128}};
    gert::StorageShape kronecker_p2_shape = {{64, 64}, {64, 64}};
    gert::StorageShape out_shape = {{16, 8192}, {16, 8192}};
    gert::StorageShape quant_scale_shape = {{16, 128, 2}, {16, 128, 2}};

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
                                            "CORE_NUM": 64
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
    struct FlatQuantCompileInfo {
        int64_t aicNum = 32;
        int64_t aivNum = 64;
        NpuArch npuArch = NpuArch::DAV_3510;
    } compile_info;

    std::string op_type("FlatQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(8192);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType("FlatQuant")
                      .NodeIoNum(3, 2)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&x_shape, &kronecker_p1_shape, &kronecker_p2_shape})
                      .OutputShapes({&out_shape, &quant_scale_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeAttrs({{"clip_ratio", Ops::NN::AnyValue::CreateFrom<float>(0.5)},
                                  {"dst_dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(ge::DT_FLOAT4_E2M1)},
                                  {"dst_type_max", Ops::NN::AnyValue::CreateFrom<float>(0.0)},
                                  {"group_list_type", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
}

TEST_F(FlatQuantTiling, 910d_flat_quant_mxfp4_tiling_003)
{
    gert::StorageShape x_shape = {{16, 192, 190}, {16, 192, 190}};
    gert::StorageShape kronecker_p1_shape = {{192, 192}, {192, 192}};
    gert::StorageShape kronecker_p2_shape = {{190, 190}, {190, 190}};
    gert::StorageShape out_shape = {{16, 36480}, {16, 36480}};
    gert::StorageShape quant_scale_shape = {{16, 570, 2}, {16, 570, 2}};

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
                                            "CORE_NUM": 64
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
    struct FlatQuantCompileInfo {
        int64_t aicNum = 32;
        int64_t aivNum = 64;
        NpuArch npuArch = NpuArch::DAV_3510;
    } compile_info;

    std::string op_type("FlatQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(8192);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType("FlatQuant")
                      .NodeIoNum(3, 2)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&x_shape, &kronecker_p1_shape, &kronecker_p2_shape})
                      .OutputShapes({&out_shape, &quant_scale_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeAttrs({{"clip_ratio", Ops::NN::AnyValue::CreateFrom<float>(0.6)},
                                  {"dst_dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(ge::DT_FLOAT4_E2M1)},
                                  {"dst_type_max", Ops::NN::AnyValue::CreateFrom<float>(0.0)},
                                  {"group_list_type", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
}

TEST_F(FlatQuantTiling, 910d_flat_quant_mxfp4_tiling_004)
{
    gert::StorageShape x_shape = {{16, 128, 190}, {16, 128, 190}};
    gert::StorageShape kronecker_p1_shape = {{128, 128}, {128, 128}};
    gert::StorageShape kronecker_p2_shape = {{64, 64}, {64, 64}};
    gert::StorageShape out_shape = {{16, 24320}, {16, 24320}};
    gert::StorageShape quant_scale_shape = {{16, 380, 2}, {16, 380, 2}};

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
                                            "CORE_NUM": 64
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
    struct FlatQuantCompileInfo {
        int64_t aicNum = 32;
        int64_t aivNum = 64;
        NpuArch npuArch = NpuArch::DAV_3510;
    } compile_info;

    std::string op_type("FlatQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(8192);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType("FlatQuant")
                      .NodeIoNum(3, 2)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&x_shape, &kronecker_p1_shape, &kronecker_p2_shape})
                      .OutputShapes({&out_shape, &quant_scale_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeAttrs({{"clip_ratio", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                                  {"dst_dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(ge::DT_FLOAT4_E2M1)},
                                  {"dst_type_max", Ops::NN::AnyValue::CreateFrom<float>(0.0)},
                                  {"group_list_type", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}
TEST_F(FlatQuantTiling, 910d_flat_quant_mxfp4_tiling_005)
{
    gert::StorageShape x_shape = {{16, 128, 160}, {16, 128, 160}};
    gert::StorageShape kronecker_p1_shape = {{128, 128}, {128, 128}};
    gert::StorageShape kronecker_p2_shape = {{160, 160}, {160, 160}};
    gert::StorageShape out_shape = {{16, 20480}, {16, 20480}};
    gert::StorageShape quant_scale_shape = {{16, 320, 2}, {16, 320, 2}};

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
                                            "CORE_NUM": 64
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
    struct FlatQuantCompileInfo {
        int64_t aicNum = 32;
        int64_t aivNum = 64;
        NpuArch npuArch = NpuArch::DAV_3510;
    } compile_info;

    std::string op_type("FlatQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(8192);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType("FlatQuant")
                      .NodeIoNum(3, 2)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&x_shape, &kronecker_p1_shape, &kronecker_p2_shape})
                      .OutputShapes({&out_shape, &quant_scale_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeAttrs({{"clip_ratio", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                                  {"dst_dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(ge::DT_INT4)},
                                  {"dst_type_max", Ops::NN::AnyValue::CreateFrom<float>(0.0)},
                                  {"group_list_type", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}
TEST_F(FlatQuantTiling, 910d_flat_quant_mxfp4_tiling_006)
{
    gert::StorageShape x_shape = {{16, 128, 160}, {16, 128, 160}};
    gert::StorageShape kronecker_p1_shape = {{128, 128}, {128, 128}};
    gert::StorageShape kronecker_p2_shape = {{0, 0}, {0, 0}};
    gert::StorageShape out_shape = {{16, 20480}, {16, 20480}};
    gert::StorageShape quant_scale_shape = {{16, 320, 2}, {16, 320, 2}};

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
                                            "CORE_NUM": 64
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
    struct FlatQuantCompileInfo {
        int64_t aicNum = 32;
        int64_t aivNum = 64;
        NpuArch npuArch = NpuArch::DAV_3510;
    } compile_info;

    std::string op_type("FlatQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(8192);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType("FlatQuant")
                      .NodeIoNum(3, 2)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&x_shape, &kronecker_p1_shape, &kronecker_p2_shape})
                      .OutputShapes({&out_shape, &quant_scale_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeAttrs({{"clip_ratio", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                                  {"dst_dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(ge::DT_FLOAT4_E2M1)},
                                  {"dst_type_max", Ops::NN::AnyValue::CreateFrom<float>(0.0)},
                                  {"group_list_type", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
}

TEST_F(FlatQuantTiling, 910d_flat_quant_mxfp4_tiling_007)
{
    gert::StorageShape x_shape = {{16, 128, 160}, {16, 128, 160}};
    gert::StorageShape kronecker_p1_shape = {{128, 128}, {128, 128}};
    gert::StorageShape kronecker_p2_shape = {{0, 160}, {0, 160}};
    gert::StorageShape out_shape = {{16, 20480}, {16, 20480}};
    gert::StorageShape quant_scale_shape = {{16, 320, 2}, {16, 320, 2}};

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
                                            "CORE_NUM": 64
                                        }
                                    })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    struct FlatQuantCompileInfo {
        int64_t aicNum = 32;
        int64_t aivNum = 64;
        NpuArch npuArch = NpuArch::DAV_3510;
    } compile_info;

    std::string op_type("FlatQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(8192);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType("FlatQuant")
                      .NodeIoNum(3, 2)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&x_shape, &kronecker_p1_shape, &kronecker_p2_shape})
                      .OutputShapes({&out_shape, &quant_scale_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeAttrs({{"clip_ratio", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                                  {"dst_dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(ge::DT_FLOAT4_E2M1)},
                                  {"dst_type_max", Ops::NN::AnyValue::CreateFrom<float>(0.0)},
                                  {"group_list_type", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(FlatQuantTiling, 910d_flat_quant_mxfp4_tiling_008)
{
    gert::StorageShape x_shape = {{16, 128, 160}, {16, 128, 160}};
    gert::StorageShape kronecker_p1_shape = {{128, 128}, {128, 128}};
    gert::StorageShape kronecker_p2_shape = {{160, 0}, {160, 0}};
    gert::StorageShape out_shape = {{16, 20480}, {16, 20480}};
    gert::StorageShape quant_scale_shape = {{16, 320, 2}, {16, 320, 2}};

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
                                            "CORE_NUM": 64
                                        }
                                    })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    struct FlatQuantCompileInfo {
        int64_t aicNum = 32;
        int64_t aivNum = 64;
        NpuArch npuArch = NpuArch::DAV_3510;
    } compile_info;

    std::string op_type("FlatQuant");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(8192);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType("FlatQuant")
                      .NodeIoNum(3, 2)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&x_shape, &kronecker_p1_shape, &kronecker_p2_shape})
                      .OutputShapes({&out_shape, &quant_scale_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeAttrs({{"clip_ratio", Ops::NN::AnyValue::CreateFrom<float>(1.0)},
                                  {"dst_dtype", Ops::NN::AnyValue::CreateFrom<int64_t>(ge::DT_FLOAT4_E2M1)},
                                  {"dst_type_max", Ops::NN::AnyValue::CreateFrom<float>(0.0)},
                                  {"group_list_type", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}
} // namespace

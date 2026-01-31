/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_scatter_elements_v2_tiling.cpp
 * \brief
 */

#include <iostream>
#include <vector>
#include <thread>
#include <nlohmann/json.hpp>
#include <gtest/gtest.h>
#include "log/log.h"
#include "graph/graph.h"
#include "kernel_run_context_facker.h"

#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "test_cube_util.h"
#include "register/op_impl_registry.h"
#include "ut_op_util.h"
#include "ut_op_common.h"
#include "platform/platform_infos_def.h"
#include "../../../op_host/scatter_elements_v2_tiling.h"
#include "../../../op_host/scatter_elements_v2_asc_tiling.h"

using namespace std;
using namespace ge;

class ScatterElementsV2Tiling : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "ScatterElementsV2Tiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "ScatterElementsV2Tiling TearDown" << std::endl;
    }
};

static string to_string(const std::stringstream& tiling_data)
{
    auto data = tiling_data.str();
    string result;
    int32_t tmp = 0;
    for (size_t i = 0; i < data.length(); i += sizeof(int32_t)) {
        memcpy(&tmp, data.c_str() + i, sizeof(tmp));
        result += std::to_string(tmp);
        result += " ";
    }

    return result;
}

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

TEST_F(ScatterElementsV2Tiling, test_scatter_elements_v2_float32)
{
    std::string op_type("ScatterElementsV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    string compile_info_string = R"({"hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                                                       "Intrinsic_fix_pipe_l0c2out": false,
                                                       "Intrinsic_data_move_l12ub": true,
                                                       "Intrinsic_data_move_l0c2ub": true,
                                                       "Intrinsic_data_move_out2l1_nd2nz": false,
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
    struct ScatterElementsV2CompileInfo {
        int32_t totalCoreNum = 30;
        uint64_t ubSizePlatForm = 0;
        uint64_t workspaceSize = 0;
    } compile_info;
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
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape input_shape = {{4096, 5933}, {4096, 5933}};
    gert::StorageShape indic_shape = {{4096, 5933}, {4096, 5933}};
    gert::StorageShape src_shape = {{4096, 5933}, {4096, 5933}};
    gert::StorageShape output_shape = {{4096, 5933}, {4096, 5933}};

    // tilingParseFunc simulate
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&input_shape, &indic_shape, &src_shape})
                      .OutputShapes({&output_shape})
                      .CompileInfo(&compile_info)
                      .NodeAttrs(
                          {{"axis", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                           {"reduction", Ops::NN::AnyValue::CreateFrom<std::string>("add")}})
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
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
    ASSERT_EQ(tiling_key, 2);
}

TEST_F(ScatterElementsV2Tiling, test_scatter_elements_v2_float32_few)
{
    std::string op_type("ScatterElementsV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    string compile_info_string = R"({"hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                                                       "Intrinsic_fix_pipe_l0c2out": false,
                                                       "Intrinsic_data_move_l12ub": true,
                                                       "Intrinsic_data_move_l0c2ub": true,
                                                       "Intrinsic_data_move_out2l1_nd2nz": false,
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
    struct ScatterElementsV2CompileInfo {
        int32_t totalCoreNum = 30;
        uint64_t ubSizePlatForm = 0;
        uint64_t workspaceSize = 0;
    } compile_info;
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
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape input_shape = {{10, 5933}, {10, 5933}};
    gert::StorageShape indic_shape = {{10, 5933}, {10, 5933}};
    gert::StorageShape src_shape = {{10, 5933}, {10, 5933}};
    gert::StorageShape output_shape = {{10, 5933}, {10, 5933}};

    // tilingParseFunc simulate
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&input_shape, &indic_shape, &src_shape})
                      .OutputShapes({&output_shape})
                      .CompileInfo(&compile_info)
                      .NodeAttrs(
                          {{"axis", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                           {"reduction", Ops::NN::AnyValue::CreateFrom<std::string>("add")}})
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
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
    ASSERT_EQ(tiling_key, 2);
}

TEST_F(ScatterElementsV2Tiling, test_scatter_elements_v2_float16)
{
    std::string op_type("ScatterElementsV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    string compile_info_string = R"({"hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                                                       "Intrinsic_fix_pipe_l0c2out": false,
                                                       "Intrinsic_data_move_l12ub": true,
                                                       "Intrinsic_data_move_l0c2ub": true,
                                                       "Intrinsic_data_move_out2l1_nd2nz": false,
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
    struct ScatterElementsV2CompileInfo {
        int32_t totalCoreNum = 30;
        uint64_t ubSizePlatForm = 0;
        uint64_t workspaceSize = 0;
    } compile_info;
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
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape input_shape = {{4096, 20000}, {4096, 20000}};
    gert::StorageShape indic_shape = {{4096, 2000}, {4096, 2000}};
    gert::StorageShape src_shape = {{4096, 2000}, {4096, 2000}};
    gert::StorageShape output_shape = {{4096, 20000}, {4096, 20000}};

    // tilingParseFunc simulate
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&input_shape, &indic_shape, &src_shape})
                      .OutputShapes({&output_shape})
                      .CompileInfo(&compile_info)
                      .NodeAttrs(
                          {{"axis", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                           {"reduction", Ops::NN::AnyValue::CreateFrom<std::string>("none")}})
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
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
    ASSERT_EQ(tiling_key, 1);
}

TEST_F(ScatterElementsV2Tiling, test_scatter_elements_v2_bfloat16)
{
    std::string op_type("ScatterElementsV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    string compile_info_string = R"({"hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                                                       "Intrinsic_fix_pipe_l0c2out": false,
                                                       "Intrinsic_data_move_l12ub": true,
                                                       "Intrinsic_data_move_l0c2ub": true,
                                                       "Intrinsic_data_move_out2l1_nd2nz": false,
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
    struct ScatterElementsV2CompileInfo {
        int32_t totalCoreNum = 30;
        uint64_t ubSizePlatForm = 0;
        uint64_t workspaceSize = 0;
    } compile_info;
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
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape input_shape = {{6144, 18192}, {6144, 18192}};
    gert::StorageShape indic_shape = {{4096, 18192}, {4096, 18192}};
    gert::StorageShape src_shape = {{4096, 18192}, {4096, 18192}};
    gert::StorageShape output_shape = {{6144, 18192}, {6144, 18192}};

    // tilingParseFunc simulate
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&input_shape, &indic_shape, &src_shape})
                      .OutputShapes({&output_shape})
                      .CompileInfo(&compile_info)
                      .NodeAttrs(
                          {{"axis", Ops::NN::AnyValue::CreateFrom<int64_t>(1)},
                           {"reduction", Ops::NN::AnyValue::CreateFrom<std::string>("add")}})
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
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
    ASSERT_EQ(tiling_key, 2);
}

TEST_F(ScatterElementsV2Tiling, test_scatter_elements_v2_mul_error)
{
    std::string op_type("ScatterElementsV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    string compile_info_string = R"({"hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                                                       "Intrinsic_fix_pipe_l0c2out": false,
                                                       "Intrinsic_data_move_l12ub": true,
                                                       "Intrinsic_data_move_l0c2ub": true,
                                                       "Intrinsic_data_move_out2l1_nd2nz": false,
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
    struct ScatterElementsV2CompileInfo {
        int32_t totalCoreNum = 30;
        uint64_t ubSizePlatForm = 0;
        uint64_t workspaceSize = 0;
    } compile_info;
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
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape input_shape = {{6144, 18192}, {6144, 18192}};
    gert::StorageShape indic_shape = {{4096, 18192}, {4096, 18192}};
    gert::StorageShape src_shape = {{4096, 18192}, {4096, 18192}};
    gert::StorageShape output_shape = {{6144, 18192}, {6144, 18192}};

    // tilingParseFunc simulate
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&input_shape, &indic_shape, &src_shape})
                      .OutputShapes({&output_shape})
                      .CompileInfo(&compile_info)
                      .NodeAttrs(
                          {{"axis", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                           {"reduction", Ops::NN::AnyValue::CreateFrom<std::string>("mul")}})
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
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
    EXPECT_NE(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
}

TEST_F(ScatterElementsV2Tiling, test_scatter_elements_v2_other_error)
{
    std::string op_type("ScatterElementsV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    string compile_info_string = R"({"hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                                                       "Intrinsic_fix_pipe_l0c2out": false,
                                                       "Intrinsic_data_move_l12ub": true,
                                                       "Intrinsic_data_move_l0c2ub": true,
                                                       "Intrinsic_data_move_out2l1_nd2nz": false,
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
    struct ScatterElementsV2CompileInfo {
        int32_t totalCoreNum = 30;
        uint64_t ubSizePlatForm = 0;
        uint64_t workspaceSize = 0;
    } compile_info;
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
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape input_shape = {{6144, 18192}, {6144, 18192}};
    gert::StorageShape indic_shape = {{4096, 3192}, {4096, 3192}};
    gert::StorageShape src_shape = {{4096, 3192}, {4096, 3192}};
    gert::StorageShape output_shape = {{6144, 18192}, {6144, 18192}};

    // tilingParseFunc simulate
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&input_shape, &indic_shape, &src_shape})
                      .OutputShapes({&output_shape})
                      .CompileInfo(&compile_info)
                      .NodeAttrs(
                          {{"axis", Ops::NN::AnyValue::CreateFrom<int64_t>(-1)},
                           {"reduction", Ops::NN::AnyValue::CreateFrom<std::string>("other")}})
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
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
    EXPECT_NE(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
}

static void ExecuteTestCase(ge::DataType data_dtype, ge::DataType indices_dtype, ge::DataType updates_dtype,
                            gert::StorageShape data_shape, gert::StorageShape indices_shape,
                            gert::StorageShape updates_shape, int64_t axis, string reduction, uint64_t tilingKeyValue,
                            string expectTilingData, int32_t deterministic, ge::graphStatus status = ge::GRAPH_SUCCESS)
{
    string compile_info_string = R"({
        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                          "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                          "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                          "CORE_NUM": 64}
                          })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}};
    map<string, string> npuarchs = {{"NpuArch", "3510"}};
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();

    // compile info
    optiling::ScatterElementsV2CompileInfo compile_info;

    std::string op_type("ScatterElementsV2");
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&data_shape, &indices_shape, &updates_shape})
                      .OutputShapes({&data_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, data_dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, indices_dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, updates_dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, data_dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .DeterministicInfo(reinterpret_cast<int32_t*>(deterministic))
                      .NodeAttrs({{"axis", Ops::NN::AnyValue::CreateFrom<int64_t>(axis)},
                                  {"reduction", Ops::NN::AnyValue::CreateFrom<string>(reduction)}})
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

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), status);
    if (status == ge::GRAPH_FAILED) {
        return;
    }
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    auto block_dim = tiling_context->GetBlockDim();
    auto raw_tiling_data = tiling_context->GetRawTilingData();
    auto tiling_data_result = to_string<int64_t>(raw_tiling_data->GetData(), raw_tiling_data->GetDataSize());
    ASSERT_EQ(tiling_key, tilingKeyValue);
    EXPECT_EQ(tiling_data_result, expectTilingData);
}

TEST_F(ScatterElementsV2Tiling, test_tiling_ascendc_int8_int64_dim1)
{
    gert::StorageShape shape1 = {{10}, {10}};
    gert::StorageShape shape2 = {{10}, {10}};
    gert::StorageShape shape3 = {{10}, {10}};
    string expectTilingData = "1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 65536 10 10 10 1 1 1 0 0 0 1 1 0 1 ";
    uint64_t tilingKeyValue = 1001001;

    ExecuteTestCase(ge::DT_INT8, ge::DT_INT64, ge::DT_INT8, shape1, shape2, shape3, 0, "none", tilingKeyValue, expectTilingData, 0);
}

TEST_F(ScatterElementsV2Tiling, test_tiling_ascendc_int8_int64_dim1_determ)
{
    gert::StorageShape shape1 = {{10}, {10}};
    gert::StorageShape shape2 = {{10}, {10}};
    gert::StorageShape shape3 = {{10}, {10}};
    string expectTilingData = "1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 65536 10 10 10 1 10 1 1 12 1 10 1 1 43980465111041 ";
    uint64_t tilingKeyValue = 1001001;

    ExecuteTestCase(ge::DT_INT8, ge::DT_INT64, ge::DT_INT8, shape1, shape2, shape3, 0, "none", tilingKeyValue, expectTilingData, 1);
}

TEST_F(ScatterElementsV2Tiling, test_tiling_ascendc_int32_int64_dim2)
{
    gert::StorageShape shape1 = {{3, 5}, {3, 5}};
    gert::StorageShape shape2 = {{2, 3}, {2, 3}};
    gert::StorageShape shape3 = {{2, 5}, {2, 5}};
    string expectTilingData = "5 1 1 1 1 1 1 3 1 1 1 1 1 1 5 1 1 1 1 1 1 16384 6 15 10 1 1 1 0 0 0 1 1 0 65538 ";
    uint64_t tilingKeyValue = 1001004;

    ExecuteTestCase(ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, shape1, shape2, shape3, 1, "none", tilingKeyValue, expectTilingData, 0);
}

TEST_F(ScatterElementsV2Tiling, test_tiling_ascendc_bfloat16_int32_dim2)
{
    gert::StorageShape shape1 = {{10, 38, 26}, {10, 38, 26}};
    gert::StorageShape shape2 = {{10, 18, 26}, {10, 18, 26}};
    gert::StorageShape shape3 = {{10, 28, 96}, {10, 28, 96}};
    string expectTilingData = "988 26 1 1 1 1 1 468 26 1 1 1 1 1 2688 96 1 1 1 1 1 32768 4680 9880 26880 1 1 1 0 0 0 1 1 0 131075 ";
    uint64_t tilingKeyValue = 1000002;

    ExecuteTestCase(ge::DT_BF16, ge::DT_INT32, ge::DT_BF16, shape1, shape2, shape3, 2, "none", tilingKeyValue, expectTilingData, 0);
}

TEST_F(ScatterElementsV2Tiling, test_tiling_ascendc_bool_int32_dim4) 
 { 
     gert::StorageShape shape1 = {{10, 2, 38, 26}, {10, 2, 38, 26}}; 
     gert::StorageShape shape2 = {{10, 2, 18, 26}, {10, 2, 18, 26}}; 
     gert::StorageShape shape3 = {{10, 2, 28, 96}, {10, 2, 28, 96}}; 
     string expectTilingData = "1976 988 26 1 1 1 1 936 468 26 1 1 1 1 5376 2688 96 1 1 1 1 65536 9360 19760 53760 1 1 1 0 0 0 1 1 0 196612 "; 
     uint64_t tilingKeyValue = 1000001; 
 
     ExecuteTestCase(ge::DT_BOOL, ge::DT_INT32, ge::DT_BOOL, shape1, shape2, shape3, 3, "none", tilingKeyValue, expectTilingData, 0); 
 }

TEST_F(ScatterElementsV2Tiling, test_tiling_ascendc_int64_int32_dim4)
{
    gert::StorageShape shape1 = {{10, 2, 18, 26}, {10, 2, 18, 26}};
    gert::StorageShape shape2 = {{10, 2, 18, 26}, {10, 2, 18, 26}};
    gert::StorageShape shape3 = {{10, 2, 18, 26}, {10, 2, 18, 26}};
    string expectTilingData = "936 468 26 1 1 1 1 936 468 26 1 1 1 1 936 468 26 1 1 1 1 8192 9360 9360 9360 1 1 1 0 0 0 1 1 0 131076 ";
    uint64_t tilingKeyValue = 1000008;

    ExecuteTestCase(ge::DT_INT64, ge::DT_INT32, ge::DT_INT64, shape1, shape2, shape3, 2, "none", tilingKeyValue, expectTilingData, 0);
}

TEST_F(ScatterElementsV2Tiling, test_tiling_ascendc_float_int64_dim5)
{
    gert::StorageShape shape1 = {{10, 2, 18, 26, 5}, {10, 2, 18, 26, 5}};
    gert::StorageShape shape2 = {{10, 2, 18, 26, 5}, {10, 2, 18, 26, 5}};
    gert::StorageShape shape3 = {{10, 2, 18, 26, 5}, {10, 2, 18, 26, 5}};
    string expectTilingData = "4680 1 1 1 1 1 1 4680 1 1 1 1 1 1 4680 1 1 1 1 1 1 16384 46800 46800 46800 1 1 1 0 0 0 1 1 0 2 ";
    uint64_t tilingKeyValue = 1001100;

    ExecuteTestCase(ge::DT_FLOAT, ge::DT_INT64, ge::DT_FLOAT, shape1, shape2, shape3, 0, "add", tilingKeyValue, expectTilingData, 0);
}

TEST_F(ScatterElementsV2Tiling, test_tiling_ascendc_float_int64_dim5_determ)
{
    gert::StorageShape shape1 = {{10, 2, 18, 26, 5}, {10, 2, 18, 26, 5}};
    gert::StorageShape shape2 = {{10, 2, 18, 26, 5}, {10, 2, 18, 26, 5}};
    gert::StorageShape shape3 = {{10, 2, 18, 26, 5}, {10, 2, 18, 26, 5}};
    string expectTilingData = "4680 1 1 1 1 1 1 4680 1 1 1 1 1 1 4680 1 1 1 1 1 1 16384 46800 46800 46800 1 10 4680 64 74 18 10 74 1 43980465111042 ";
    uint64_t tilingKeyValue = 1001100;

    ExecuteTestCase(ge::DT_FLOAT, ge::DT_INT64, ge::DT_FLOAT, shape1, shape2, shape3, 0, "add", tilingKeyValue, expectTilingData, 1);
}

TEST_F(ScatterElementsV2Tiling, test_tiling_ascendc_int16_int32_dim6)
{
    gert::StorageShape shape1 = {{2, 2, 8, 16, 5, 3}, {2, 2, 8, 16, 5, 3}};
    gert::StorageShape shape2 = {{2, 2, 8, 9, 5, 3}, {2, 2, 8, 9, 5, 3}};
    gert::StorageShape shape3 = {{2, 2, 8, 19, 5, 3}, {2, 2, 8, 19, 5, 3}};
    string expectTilingData = "3840 1920 240 15 1 1 1 2160 1080 135 15 1 1 1 4560 2280 285 15 1 1 1 8192 4320 7680 9120 1 1 1 0 0 0 1 1 0 131077 ";
    uint64_t tilingKeyValue = 1000106;

    ExecuteTestCase(ge::DT_INT16, ge::DT_INT32, ge::DT_INT16, shape1, shape2, shape3, 2, "add", tilingKeyValue, expectTilingData, 0);
}


TEST_F(ScatterElementsV2Tiling, test_tiling_ascendc_int16_int32_dim6_determ)
{
    gert::StorageShape shape1 = {{2, 2, 8, 16, 5, 3}, {2, 2, 8, 16, 5, 3}};
    gert::StorageShape shape2 = {{2, 2, 8, 9, 5, 3}, {2, 2, 8, 9, 5, 3}};
    gert::StorageShape shape3 = {{2, 2, 8, 19, 5, 3}, {2, 2, 8, 19, 5, 3}};
    string expectTilingData = "3840 1920 240 15 1 1 1 2160 1080 135 15 1 1 1 4560 2280 285 15 1 1 1 8192 4320 7680 9120 1 1 1 0 0 0 1 1 0 131077 ";
    uint64_t tilingKeyValue = 1000106;

    ExecuteTestCase(ge::DT_INT16, ge::DT_INT32, ge::DT_INT16, shape1, shape2, shape3, 2, "add", tilingKeyValue, expectTilingData, 1);
}

TEST_F(ScatterElementsV2Tiling, test_tiling_ascendc_bool_int64_dim2)
{
    gert::StorageShape shape1 = {{2, 2, 8, 16, 5, 3}, {2, 2, 8, 16, 5, 3}};
    gert::StorageShape shape2 = {{2, 2, 8, 9, 5, 3}, {2, 2, 8, 9, 5, 3}};
    gert::StorageShape shape3 = {{2, 2, 8, 19, 5, 3}, {2, 2, 8, 19, 5, 3}};
    string expectTilingData = "3840 1920 240 15 1 1 1 2160 1080 135 15 1 1 1 4560 2280 285 15 1 1 1 16384 4320 7680 9120 1 1 1 0 0 0 1 1 0 131077 ";
    uint64_t tilingKeyValue = 1001112;

    ExecuteTestCase(ge::DT_BOOL, ge::DT_INT64, ge::DT_BOOL, shape1, shape2, shape3, 2, "add", tilingKeyValue, expectTilingData, 0);
}

TEST_F(ScatterElementsV2Tiling, test_tiling_ascendc_uint8_int64_dim7)
{
    gert::StorageShape shape1 = {{1, 3, 8, 16, 5, 3, 4}, {1, 3, 8, 16, 5, 3, 4}};
    gert::StorageShape shape2 = {{1, 3, 8, 16, 5, 3, 4}, {1, 3, 8, 16, 5, 3, 4}};
    gert::StorageShape shape3 = {{1, 3, 8, 16, 5, 3, 4}, {1, 3, 8, 16, 5, 3, 4}};
    string expectTilingData = "23040 7680 960 60 1 1 1 23040 7680 960 60 1 1 1 23040 7680 960 60 1 1 1 65536 23040 23040 23040 1 1 1 0 0 0 1 1 0 196613 ";
    uint64_t tilingKeyValue = 1001204;

    ExecuteTestCase(ge::DT_UINT8, ge::DT_INT64, ge::DT_UINT8, shape1, shape2, shape3, 3, "mul", tilingKeyValue, expectTilingData, 0);
}

TEST_F(ScatterElementsV2Tiling, test_tiling_ascendc_uint8_int64_dim7_determ)
{
    gert::StorageShape shape1 = {{1, 3, 8, 16, 5, 3, 4}, {1, 3, 8, 16, 5, 3, 4}};
    gert::StorageShape shape2 = {{1, 3, 8, 16, 5, 3, 4}, {1, 3, 8, 16, 5, 3, 4}};
    gert::StorageShape shape3 = {{1, 3, 8, 16, 5, 3, 4}, {1, 3, 8, 16, 5, 3, 4}};
    string expectTilingData = "23040 7680 960 60 1 1 1 23040 7680 960 60 1 1 1 23040 7680 960 60 1 1 1 65536 23040 23040 23040 1 1 1 0 0 0 1 1 0 196613 ";
    uint64_t tilingKeyValue = 1001204;

    ExecuteTestCase(ge::DT_UINT8, ge::DT_INT64, ge::DT_UINT8, shape1, shape2, shape3, 3, "mul", tilingKeyValue, expectTilingData, 1);
}

TEST_F(ScatterElementsV2Tiling, test_tiling_ascendc_float16_int32_dim8)
{
    gert::StorageShape shape1 = {{4, 3, 6, 5, 1, 5, 3, 4}, {4, 3, 6, 5, 1, 5, 3, 4}};
    gert::StorageShape shape2 = {{4, 3, 6, 5, 1, 5, 3, 4}, {4, 3, 6, 5, 1, 5, 3, 4}};
    gert::StorageShape shape3 = {{4, 3, 6, 5, 1, 5, 3, 4}, {4, 3, 6, 5, 1, 5, 3, 4}};
    string expectTilingData = "5400 1800 300 60 60 12 4 5400 1800 300 60 60 12 4 5400 1800 300 60 60 12 4 32768 21600 21600 21600 1 1 1 0 0 0 1 1 0 393224 ";
    uint64_t tilingKeyValue = 1000201;

    ExecuteTestCase(ge::DT_FLOAT16, ge::DT_INT32, ge::DT_FLOAT16, shape1, shape2, shape3, 6, "mul", tilingKeyValue, expectTilingData, 0);
}

TEST_F(ScatterElementsV2Tiling, test_tiling_ascendc_dim_invalid)
{
    gert::StorageShape shape1 = {{3, 5}, {3, 5}};
    gert::StorageShape shape2 = {{2, 3}, {2, 3}};
    gert::StorageShape shape3 = {{2, 5}, {2, 5}};
    string expectTilingData = "5 1 1 1 1 1 1 3 1 1 1 1 1 1 5 1 1 1 1 1 1 114688 6 15 10 2 1 1 1 0 0 0 1 1 0 2";
    uint64_t tilingKeyValue = 1001004;

    ExecuteTestCase(ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, shape1, shape2, shape3, 2, "none", tilingKeyValue, expectTilingData, 0, ge::GRAPH_FAILED);
}

TEST_F(ScatterElementsV2Tiling, test_tiling_ascendc_indices_data_dim_not_same)
{
    gert::StorageShape shape1 = {{3, 5}, {3, 5}};
    gert::StorageShape shape2 = {{2, 3, 1}, {2, 3, 1}};
    gert::StorageShape shape3 = {{2, 5}, {2, 5}};
    string expectTilingData = "5 1 1 1 1 1 1 3 1 1 1 1 1 1 5 1 1 1 1 1 1 114688 6 15 10 2 1 1 1 0 0 0 1 1 0 0";
    uint64_t tilingKeyValue = 1001004;

    ExecuteTestCase(ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, shape1, shape2, shape3, 0, "none", tilingKeyValue, expectTilingData, 0, ge::GRAPH_FAILED);
}

TEST_F(ScatterElementsV2Tiling, test_tiling_ascendc_indices_updates_dim_not_same)
{
    gert::StorageShape shape1 = {{3, 5}, {3, 5}};
    gert::StorageShape shape2 = {{2, 3}, {2, 3}};
    gert::StorageShape shape3 = {{2, 5, 1}, {2, 5, 1}};
    string expectTilingData = "5 1 1 1 1 1 1 3 1 1 1 1 1 1 5 1 1 1 1 1 1 114688 6 15 10 2 1 1 1 0 0 0 1 1 0 0";
    uint64_t tilingKeyValue = 1001004;

    ExecuteTestCase(ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, shape1, shape2, shape3, 0, "none", tilingKeyValue, expectTilingData, 0, ge::GRAPH_FAILED);
}

TEST_F(ScatterElementsV2Tiling, test_tiling_ascendc_indices_updates_shape_invalid)
{
    gert::StorageShape shape1 = {{3, 5}, {3, 5}};
    gert::StorageShape shape2 = {{2, 3}, {2, 3}};
    gert::StorageShape shape3 = {{2, 2}, {2, 2}};
    string expectTilingData = "5 1 1 1 1 1 1 3 1 1 1 1 1 1 5 1 1 1 1 1 1 114688 6 15 10 2 1 1 1 0 0 0 1 1 0 0";
    uint64_t tilingKeyValue = 1001004;

    ExecuteTestCase(ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, shape1, shape2, shape3, 0, "none", tilingKeyValue, expectTilingData, 0, ge::GRAPH_FAILED);
}

TEST_F(ScatterElementsV2Tiling, test_tiling_ascendc_indices_data_shape_invalid)
{
    gert::StorageShape shape1 = {{3, 5}, {3, 5}};
    gert::StorageShape shape2 = {{4, 6}, {4, 6}};
    gert::StorageShape shape3 = {{2, 5}, {2, 5}};
    string expectTilingData = "5 1 1 1 1 1 1 3 1 1 1 1 1 1 5 1 1 1 1 1 1 114688 6 15 10 2 1 1 1 0 0 0 1 1 0 0";
    uint64_t tilingKeyValue = 1001004;

    ExecuteTestCase(ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, shape1, shape2, shape3, 0, "none", tilingKeyValue, expectTilingData, 0, ge::GRAPH_FAILED);
}

TEST_F(ScatterElementsV2Tiling, test_tiling_ascendc_reduction_invalid)
{
    gert::StorageShape shape1 = {{3, 5}, {3, 5}};
    gert::StorageShape shape2 = {{2, 3}, {2, 3}};
    gert::StorageShape shape3 = {{2, 5}, {2, 5}};
    string expectTilingData = "5 1 1 1 1 1 1 3 1 1 1 1 1 1 5 1 1 1 1 1 1 114688 6 15 10 2 1 1 1 0 0 0 1 1 0 0";
    uint64_t tilingKeyValue = 1001004;

    ExecuteTestCase(ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, shape1, shape2, shape3, 0, "sum", tilingKeyValue, expectTilingData, 0, ge::GRAPH_FAILED);
}

TEST_F(ScatterElementsV2Tiling, test_tiling_ascendc_reduction_none_dtype_invalid)
{
    gert::StorageShape shape1 = {{3, 5}, {3, 5}};
    gert::StorageShape shape2 = {{2, 3}, {2, 3}};
    gert::StorageShape shape3 = {{2, 5}, {2, 5}};
    string expectTilingData = "5 1 1 1 1 1 1 3 1 1 1 1 1 1 5 1 1 1 1 1 1 114688 6 15 10 2 1 1 1 0 0 0 1 1 0 0";
    uint64_t tilingKeyValue = 1001004;

    ExecuteTestCase(ge::DT_UINT32, ge::DT_INT64, ge::DT_UINT32, shape1, shape2, shape3, 0, "none", tilingKeyValue, expectTilingData, 0, ge::GRAPH_FAILED);
}

TEST_F(ScatterElementsV2Tiling, test_tiling_ascendc_reduction_add_dtype_invalid) 
{ 
    gert::StorageShape shape1 = {{3, 5}, {3, 5}}; 
    gert::StorageShape shape2 = {{2, 3}, {2, 3}}; 
    gert::StorageShape shape3 = {{2, 5}, {2, 5}}; 
    string expectTilingData = "5 1 1 1 1 1 1 3 1 1 1 1 1 1 5 1 1 1 1 1 1 114688 6 15 10 2 1 1 1 0 0 0 1 1 0 0"; 
    uint64_t tilingKeyValue = 1001004; 

    ExecuteTestCase(ge::DT_DOUBLE, ge::DT_INT64, ge::DT_DOUBLE, shape1, shape2, shape3, 0, "add", tilingKeyValue, expectTilingData, 0, ge::GRAPH_FAILED); 
} 

TEST_F(ScatterElementsV2Tiling, test_tiling_ascendc_reduction_mul_dtype_invalid) 
{ 
    gert::StorageShape shape1 = {{3, 5}, {3, 5}}; 
    gert::StorageShape shape2 = {{2, 3}, {2, 3}}; 
    gert::StorageShape shape3 = {{2, 5}, {2, 5}}; 
    string expectTilingData = "5 1 1 1 1 1 1 3 1 1 1 1 1 1 5 1 1 1 1 1 1 114688 6 15 10 2 1 1 1 0 0 0 1 1 0 0"; 
    uint64_t tilingKeyValue = 1001004; 

    ExecuteTestCase(ge::DT_BOOL, ge::DT_INT64, ge::DT_BOOL, shape1, shape2, shape3, 0, "mul", tilingKeyValue, expectTilingData, 0, ge::GRAPH_FAILED); 
}

TEST_F(ScatterElementsV2Tiling, test_tiling_ascendc_indices_dtype_invalid)
{
    gert::StorageShape shape1 = {{3, 5}, {3, 5}};
    gert::StorageShape shape2 = {{2, 3}, {2, 3}};
    gert::StorageShape shape3 = {{2, 5}, {2, 5}};
    string expectTilingData = "5 1 1 1 1 1 1 3 1 1 1 1 1 1 5 1 1 1 1 1 1 114688 6 15 10 2 1 1 1 0 0 0 1 1 0 1";
    uint64_t tilingKeyValue = 1001004;

    ExecuteTestCase(ge::DT_INT16, ge::DT_INT16, ge::DT_INT16, shape1, shape2, shape3, 0, "none", tilingKeyValue, expectTilingData, 0, ge::GRAPH_FAILED);
}

TEST_F(ScatterElementsV2Tiling, test_tiling_ascendc_indices_data_dtype_not_same)
{
    gert::StorageShape shape1 = {{3, 5}, {3, 5}};
    gert::StorageShape shape2 = {{2, 3}, {2, 3}};
    gert::StorageShape shape3 = {{2, 5}, {2, 5}};
    string expectTilingData = "5 1 1 1 1 1 1 3 1 1 1 1 1 1 5 1 1 1 1 1 1 114688 6 15 10 2 1 1 1 0 0 0 1 1 0 1";
    uint64_t tilingKeyValue = 1001004;

    ExecuteTestCase(ge::DT_INT64, ge::DT_INT32, ge::DT_INT32, shape1, shape2, shape3, 0, "none", tilingKeyValue, expectTilingData, 0, ge::GRAPH_FAILED);
}

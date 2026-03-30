/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

 /*!
 * \file test_adaptive_avg_pool3d_grad_tiling.cpp
 * \brief
 */

#include <iostream>
#include <vector>
#include <gtest/gtest.h>

#include "kernel_run_context_facker.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "test_cube_util.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "ut_op_util.h"
#include "ut_op_common.h"
#include "platform/platform_infos_def.h"
#include "platform/platform_info.h"

using namespace std;
using namespace ge;

struct AdaptiveAvgPool3dGradTilingTestParam {
    string case_name;

    std::initializer_list<int64_t> y_grad_shape;
    std::initializer_list<int64_t> x_shape;
    std::initializer_list<int64_t> x_grad_shape;

    ge::DataType data_type;

    uint32_t expected_block_dim;
    uint64_t expected_tiling_key;
    string expected_tiling_data;
};

struct AdaptiveAvgPool3dGradCompileInfo {
    uint32_t coreNum = 0;
    uint64_t sysWorkspaceSize = 0;
    uint64_t ubSizePlatForm = 0;
};

class AdaptiveAvgPool3dGradTilingTest : public testing::TestWithParam<AdaptiveAvgPool3dGradTilingTestParam>
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "AdaptiveAvgPool3dGradTilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AdaptiveAvgPool3dGradTilingTest TearDown" << std::endl;
    }
};

static string TilingData2Str(const gert::TilingData* tiling_data)
{
    auto data = tiling_data->GetData();

    stringstream ss;
    for (size_t i = 0; i < tiling_data->GetDataSize(); i += sizeof(int64_t)) {
        ss << std::to_string((reinterpret_cast<const int64_t*>(tiling_data->GetData())[i / sizeof(int64_t)])) << " ";
    }

    return ss.str();
}

TEST_P(AdaptiveAvgPool3dGradTilingTest, test_case_adaptive_avg_pool3d_grad_tiling)
{
    AdaptiveAvgPool3dGradTilingTestParam param = GetParam();

    gert::StorageShape y_grad_shape = {param.y_grad_shape, param.y_grad_shape};
    gert::StorageShape x_shape = {param.x_shape, param.x_shape};
    gert::StorageShape x_grad_shape = {param.x_grad_shape, param.x_grad_shape};

    string op_type("AdaptiveAvgPool3dGrad");
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
                                                      "CORE_NUM": 40}
                                    })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    AdaptiveAvgPool3dGradCompileInfo compile_info;
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
    auto tiling_data = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(tiling_data, nullptr);

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&y_grad_shape, &x_shape})
                      .OutputShapes({&x_grad_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, param.data_type, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, param.data_type, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, param.data_type, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(tiling_data.get())
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
    // check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, param.expected_tiling_key);
    auto block_dim = tiling_context->GetBlockDim();
    ASSERT_EQ(block_dim, param.expected_block_dim);
    auto tiling_data_result = TilingData2Str(tiling_context->GetRawTilingData());
    ASSERT_EQ(tiling_data_result, param.expected_tiling_data);
}

static AdaptiveAvgPool3dGradTilingTestParam cases[] = {
    {"test_case_float32_tiling",
     {64, 24, 24, 8192},
     {8192, 16, 12, 12},
     {16, 12, 12, 8192},
     ge::DT_FLOAT,
     40,
     0,
     "8192 16 12 12 64 24 24 40 922 906 2 1 8192 8192 1 0 "},
    {"test_case_cast_tiling",
     {64, 24, 24, 8192},
     {8192, 16, 12, 12},
     {16, 12, 12, 8192},
     ge::DT_FLOAT16,
     40,
     10,
     "8192 16 12 12 64 24 24 40 922 906 1 1 8192 8192 1 0 "},

    {"test_case_nc_large_float32_tiling",
     {64, 24, 24, 40960},
     {40960, 16, 12, 12},
     {16, 12, 12, 40960},
     ge::DT_FLOAT,
     40,
     1,
     "40960 16 12 12 64 24 24 40 1844 1812 1 2 20480 20480 1 0 "},
    {"test_case_nc_large_cast_tiling",
     {64, 24, 24, 40960},
     {40960, 16, 12, 12},
     {16, 12, 12, 40960},
     ge::DT_BF16,
     40,
     21,
     "40960 16 12 12 64 24 24 40 3687 3663 1 4 13648 16 1 0 "},
};

INSTANTIATE_TEST_CASE_P(AdaptiveAvgPool3dGrad, AdaptiveAvgPool3dGradTilingTest, testing::ValuesIn(cases));


namespace {
static void SetAscend950GlobalPlatformInfo()
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optiCompilationInfo;

    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = "Ascend950";
    optiCompilationInfo.soc_version = "Ascend950";

    fe::PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
    fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
}
} // namespace

class AdaptiveAvgPool3dGradTiling950Test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AdaptiveAvgPool3dGradTiling950Test SetUp" << std::endl;
        SetAscend950GlobalPlatformInfo();
    }

    static void TearDownTestCase()
    {
        std::cout << "AdaptiveAvgPool3dGradTiling950Test TearDown" << std::endl;
    }
};

static void ExecuteAdaptiveAvgPool3dGrad950TestCase(gert::StorageShape yGradShape,
                                                    gert::StorageShape xShape,
                                                    gert::StorageShape xGradShape,
                                                    ge::DataType dtype,
                                                    uint64_t expect_tiling_key)
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

    std::map<std::string, std::string> soc_version_infos = {
        {"Short_SoC_version", "Ascend950"},
        {"NpuArch", "3510"}
    };

    fe::PlatFormInfos platform_info;
    platform_info.Init();

    AdaptiveAvgPool3dGradCompileInfo compile_info;

    std::string op_type("AdaptiveAvgPool3dGrad");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

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

    auto tiling_data = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(tiling_data, nullptr);

    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&yGradShape, &xShape})
                      .OutputShapes({&xGradShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, dtype, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
                      .NodeInputTd(1, dtype, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
                      .NodeOutputTd(0, dtype, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
                      .NodeAttrs({
                          {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")}
                      })
                      .TilingData(tiling_data.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes(
        "version", soc_version_infos);

    std::cout << "[950] before tiling_func" << std::endl;
    auto ret = tiling_func(tiling_context);
    std::cout << "[950] after tiling_func, ret=" << ret << std::endl;

    ASSERT_EQ(ret, ge::GRAPH_SUCCESS);

    auto real_tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(real_tiling_key, expect_tiling_key);

    auto raw_tiling = tiling_context->GetRawTilingData();
    ASSERT_NE(raw_tiling, nullptr);

    dlog_setlevel(0, 3, 0);
}

TEST_F(AdaptiveAvgPool3dGradTiling950Test, adaptive_avg_pool3d_grad_tilingkey_258_ascend950)
{
    std::cout << "run case: adaptive_avg_pool3d_grad_tilingkey_258_ascend950" << std::endl;

    gert::StorageShape yGradShape = {{1, 1, 1, 1, 32768}, {1, 1, 1, 1, 32768}};
    gert::StorageShape xShape = {{1, 1, 1, 1, 16777216}, {1, 1, 1, 1, 16777216}};
    gert::StorageShape xGradShape = {{1, 1, 1, 1, 16777216}, {1, 1, 1, 1, 16777216}};

    ExecuteAdaptiveAvgPool3dGrad950TestCase(
        yGradShape,
        xShape,
        xGradShape,
        ge::DT_FLOAT,
        17);
}

TEST_F(AdaptiveAvgPool3dGradTiling950Test, adaptive_avg_pool3d_grad_tilingkey_257_ascend950)
{
    std::cout << "run case: adaptive_avg_pool3d_grad_tilingkey_257_ascend950" << std::endl;

    gert::StorageShape yGradShape = {{3, 30, 37, 6, 29}, {3, 30, 37, 6, 29}};
    gert::StorageShape xShape = {{3, 30, 5, 1, 24}, {3, 30, 5, 1, 24}};
    gert::StorageShape xGradShape = {{3, 30, 5, 1, 24}, {3, 30, 5, 1, 24}};

    ExecuteAdaptiveAvgPool3dGrad950TestCase(
        yGradShape,
        xShape,
        xGradShape,
        ge::DT_FLOAT,
        18);
}
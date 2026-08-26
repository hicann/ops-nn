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
#include "platform/platform_info.h"

using namespace std;
using namespace ge;

struct AdaptiveAvgPool3dTilingTestParam {
    string case_name;

    std::initializer_list<int64_t> x_shape;
    std::initializer_list<int64_t> y_shape;
    std::initializer_list<int64_t> output_size;
    std::string data_format;
    ge::DataType data_type;

    uint64_t expected_tiling_key;
};

struct AdaptiveAvgPool3dCompileInfo {
    int32_t totalCoreNum = 0;
    uint32_t sysWorkspaceSize = 0;
    uint64_t ubSizePlatForm = 0;
};

class AdaptiveAvgPool3dTilingTest : public testing::TestWithParam<AdaptiveAvgPool3dTilingTestParam> {
protected:
    static void SetUpTestCase() { std::cout << "AdaptiveAvgPool3dTilingTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "AdaptiveAvgPool3dTilingTest TearDown" << std::endl; }
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

class AdaptiveAvgPool3dTiling950Test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AdaptiveAvgPool3dTiling950Test SetUp" << std::endl;
        SetAscend950GlobalPlatformInfo();
    }

    static void TearDownTestCase() { std::cout << "AdaptiveAvgPool3dTiling950Test TearDown" << std::endl; }
};

static void ExecuteAdaptiveAvgPool3d950TestCase(gert::StorageShape xShape, gert::StorageShape yShape,
                                                std::vector<int64_t> outputSize, std::string dataFormat,
                                                ge::DataType dataType, uint64_t expectTilingKey)
{
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

    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    std::map<std::string, std::string> npu_arch_infos = {{"NpuArch", "3510"}};
    fe::PlatFormInfos platform_info;
    platform_info.Init();

    AdaptiveAvgPool3dCompileInfo compile_info;

    std::string op_type("AdaptiveAvgPool3d");
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
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version",
                                                                                            soc_version_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", npu_arch_infos);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto tiling_data = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(tiling_data, nullptr);

    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1})
                      .InputShapes({&xShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, dataType, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"output_size", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(outputSize)},
                                  {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>(dataFormat)}})
                      .TilingData(tiling_data.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", npu_arch_infos);

    std::cout << "[950] before tiling_func" << std::endl;
    auto ret = tiling_func(tiling_context);
    std::cout << "[950] after tiling_func, ret=" << ret << std::endl;

    ASSERT_EQ(ret, ge::GRAPH_SUCCESS);

    auto real_tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(real_tiling_key, expectTilingKey);

    auto raw_tiling = tiling_context->GetRawTilingData();
    ASSERT_NE(raw_tiling, nullptr);
}

TEST_F(AdaptiveAvgPool3dTiling950Test, adaptive_avg_pool3d_gather_trans_float32_ascend950)
{
    std::cout << "run case: adaptive_avg_pool3d_gather_trans_float32_ascend950" << std::endl;

    gert::StorageShape xShape = {{19, 2883, 6, 6, 6}, {19, 2883, 6, 6, 6}};
    gert::StorageShape yShape = {{19, 2883, 5, 5, 6}, {19, 2883, 5, 5, 6}};
    std::vector<int64_t> outputSize = {5, 5, 6};
    std::string dataFormat = "NCDHW";
    ge::DataType dataType = ge::DT_FLOAT;

    ExecuteAdaptiveAvgPool3d950TestCase(xShape, yShape, outputSize, dataFormat, dataType, 32);
}
TEST_F(AdaptiveAvgPool3dTiling950Test, adaptive_avg_pool3d_gather_trans_bfloat16_ascend950)
{
    std::cout << "run case: adaptive_avg_pool3d_gather_trans_bfloat16_ascend950" << std::endl;

    gert::StorageShape xShape = {{10791, 67, 4, 2, 2}, {10791, 67, 4, 2, 2}};
    gert::StorageShape yShape = {{10791, 67, 2, 1, 2}, {10791, 67, 2, 1, 2}};
    std::vector<int64_t> outputSize = {2, 1, 2};
    std::string dataFormat = "NCDHW";
    ge::DataType dataType = ge::DT_BF16;

    ExecuteAdaptiveAvgPool3d950TestCase(xShape, yShape, outputSize, dataFormat, dataType, 32);
}
TEST_F(AdaptiveAvgPool3dTiling950Test, adaptive_avg_pool3d_gather_trans_float16_ascend950)
{
    std::cout << "run case: adaptive_avg_pool3d_gather_trans_float16_ascend950" << std::endl;

    gert::StorageShape xShape = {{4, 13776, 18, 2, 3}, {4, 13776, 18, 2, 3}};
    gert::StorageShape yShape = {{4, 13776, 9, 2, 2}, {4, 13776, 9, 2, 2}};
    std::vector<int64_t> outputSize = {9, 2, 2};
    std::string dataFormat = "NCDHW";
    ge::DataType dataType = ge::DT_FLOAT16;

    ExecuteAdaptiveAvgPool3d950TestCase(xShape, yShape, outputSize, dataFormat, dataType, 32);
}

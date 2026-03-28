/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_adaptive_avg_pool2d_tiling.cpp
 * \brief Tiling测试 - TestAdaptiveAvgPool2dTiling
 */

#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include <string>

#include "../../../../op_host/arch35/adaptive_avg_pool2d_tiling.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "platform/platform_infos_def.h"
#include "ut_op_util.h"

using namespace ut_util;
using namespace std;
using namespace ge;

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

class AdaptiveAvgPool2dTiling950Test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AdaptiveAvgPool2dTiling950Test SetUp" << std::endl;
        SetAscend950GlobalPlatformInfo();
    }

    static void TearDownTestCase()
    {
        std::cout << "AdaptiveAvgPool2dTiling950Test TearDown" << std::endl;
    }
};

static void ExecuteAdaptiveAvgPool2d950TestCase(gert::StorageShape xShape, gert::StorageShape yShape, 
                                                    std::vector<int64_t> outputSize, 
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

    optiling::AdaptiveAvgPool2dCompileInfo compile_info;

    std::string op_type("AdaptiveAvgPool2d");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(1, 1)
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
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1})
                      .InputShapes({&xShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, dtype, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeOutputTd(0, dtype, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
                      .NodeAttrs({
                          {"output_size", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(outputSize)}
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

// 测试用例1: simt - 4D
TEST_F(AdaptiveAvgPool2dTiling950Test, test_simt_4d_enough)
{
    gert::StorageShape x_shape = {{3, 1, 79, 109}, {3, 1, 79, 109}};
    gert::StorageShape y_shape = {{3, 1, 68, 46}, {3, 1, 68, 46}};
    ExecuteAdaptiveAvgPool2d950TestCase(x_shape, y_shape, {68, 46}, ge::DT_FLOAT16, 2);
}

// 测试用例2: simt - 3D
TEST_F(AdaptiveAvgPool2dTiling950Test, test_simt_3d_enough)
{
    gert::StorageShape x_shape = {{2, 9226, 3}, {2, 9226, 3}};
    gert::StorageShape y_shape = {{2, 88, 53}, {2, 88, 53}};
    ExecuteAdaptiveAvgPool2d950TestCase(x_shape, y_shape, {88, 53}, ge::DT_FLOAT16, 2);
}
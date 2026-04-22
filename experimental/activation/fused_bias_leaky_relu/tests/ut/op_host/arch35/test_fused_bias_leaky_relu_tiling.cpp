/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
#include "kernel_run_context_faker.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "test_cube_util.h"
#include "register/op_impl_registry.h"
#include "ut_op_util.h"
#include "ut_op_common.h"
#include "platform/platform_infos_def.h"

using namespace ut_util;
using namespace std;
using namespace ge;

struct FusedBiasLeakyReluTilingTestParam {
    string caseName;
    vector<int64_t> xShape;
    ge::DataType dtype;
    float negativeSlope;
    float scale;
};

class FusedBiasLeakyReluTiling : public testing::TestWithParam<FusedBiasLeakyReluTilingTestParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "FusedBiasLeakyReluTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "FusedBiasLeakyReluTiling TearDown" << std::endl;
    }
};

static FusedBiasLeakyReluTilingTestParam tilingCases[] = {
    {"float32_small", {8}, ge::DT_FLOAT, 0.2f, 1.414213562373f},
    {"float32_medium", {1024}, ge::DT_FLOAT, 0.2f, 1.414213562373f},
    {"float32_large", {8192, 1024}, ge::DT_FLOAT, 0.2f, 1.414213562373f},
    {"float32_4d", {2, 4, 8, 16}, ge::DT_FLOAT, 0.2f, 1.414213562373f},
    {"float16_small", {8}, ge::DT_FLOAT16, 0.2f, 1.414213562373f},
    {"float16_medium", {1024}, ge::DT_FLOAT16, 0.2f, 1.414213562373f},
    {"float16_large", {8192, 1024}, ge::DT_FLOAT16, 0.2f, 1.414213562373f},
    {"negative_slope_0", {1024}, ge::DT_FLOAT, 0.0f, 1.0f},
    {"scale_1", {1024}, ge::DT_FLOAT, 0.2f, 1.0f},
    {"negative_slope_large", {1024}, ge::DT_FLOAT, 0.5f, 1.414213562373f},
};

static void RunTilingTest(const FusedBiasLeakyReluTilingTestParam& param)
{
    gert::StorageShape x_shape = {{param.xShape}, {param.xShape}};
    gert::StorageShape bias_shape = {{param.xShape}, {param.xShape}};
    gert::StorageShape y_shape = {{param.xShape}, {param.xShape}};
    
    string compile_info_string = R"({
            "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                            "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, 
                            "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                            "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                            "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                            "CORE_NUM": 24}
                            })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();

    struct FusedBiasLeakyReluTilingCompileInfo {};
    FusedBiasLeakyReluTilingCompileInfo compile_info;

    std::string op_type("FusedBiasLeakyRelu");
    auto op_impl = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str());
    ASSERT_NE(op_impl, nullptr);
    auto tiling_func = op_impl->tiling;
    ASSERT_NE(tiling_func, nullptr);

    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);

    auto param_holder = gert::TilingData::CreateCap(4096);
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
    ASSERT_NE(param_holder, nullptr);
    
    auto holder = gert::TilingContextFaker()
                        .SetOpType("FusedBiasLeakyRelu")
                        .NodeIoNum(2, 1)
                        .IrInstanceNum({1, 1})
                        .InputShapes({&x_shape, &bias_shape})
                        .OutputShapes({&y_shape})
                        .CompileInfo(&compile_info)
                        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                        .NodeInputTd(0, param.dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(1, param.dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeOutputTd(0, param.dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                        .Attrs({param.negativeSlope, param.scale})
                        .TilingData(param_holder.get())
                        .Workspace(ws_size)
                        .Build();
    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context, nullptr);
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
}

TEST_P(FusedBiasLeakyReluTiling, tiling_test)
{
    FusedBiasLeakyReluTilingTestParam param = GetParam();
    RunTilingTest(param);
}

INSTANTIATE_TEST_CASE_P(FusedBiasLeakyRelu, FusedBiasLeakyReluTiling, testing::ValuesIn(tilingCases));

TEST_F(FusedBiasLeakyReluTiling, empty_tensor)
{
    gert::StorageShape x_shape = {{0}, {0}};
    gert::StorageShape bias_shape = {{0}, {0}};
    gert::StorageShape y_shape = {{0}, {0}};
    
    string compile_info_string = R"({
            "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                            "UB_SIZE": 196608, "CORE_NUM": 24}
                            })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();

    struct FusedBiasLeakyReluTilingCompileInfo {};
    FusedBiasLeakyReluTilingCompileInfo compile_info;

    std::string op_type("FusedBiasLeakyRelu");
    auto op_impl = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str());
    ASSERT_NE(op_impl, nullptr);
    auto tiling_func = op_impl->tiling;
    ASSERT_NE(tiling_func, nullptr);

    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);

    auto param_holder = gert::TilingData::CreateCap(4096);
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());

    auto holder = gert::TilingContextFaker()
                        .SetOpType("FusedBiasLeakyRelu")
                        .NodeIoNum(2, 1)
                        .IrInstanceNum({1, 1})
                        .InputShapes({&x_shape, &bias_shape})
                        .OutputShapes({&y_shape})
                        .CompileInfo(&compile_info)
                        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                        .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .Attrs({0.2f, 1.414213562373f})
                        .TilingData(param_holder.get())
                        .Workspace(ws_size)
                        .Build();
    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context, nullptr);
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
}
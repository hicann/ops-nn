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
 * \file test_p_relu_tiling.cpp
 * \brief p_relu tiling ut test
 */

#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include "log/log.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "platform/platform_infos_def.h"
#include "ut_op_util.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include "../../../../op_host/arch35/p_relu_tiling_arch35.h"

using namespace ut_util;
using namespace std;
using namespace ge;

class PReluTilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "PReluTilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "PReluTilingTest TearDown" << std::endl;
    }
};

template <typename T>
static string to_string(void* buf, size_t size)
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

static void ExecuteTestCase(
    ge::DataType xDtype, ge::DataType weightDtype, ge::DataType yDtype, gert::StorageShape& xShape,
    gert::StorageShape& weightShape, gert::StorageShape& yShape, ge::graphStatus expectStatus = ge::GRAPH_SUCCESS)
{
    std::string compile_info_string = R"({
        "hardware_info": {
            "BT_SIZE":0, "load3d_constraints": "1",
            "Intrinsic_fix_pipe_l0c2out": false, 
            "Intrinsic_data_move_l12ub": true,
            "Intrinsic_data_move_l0c2ub": true, 
            "Intrinsic_data_move_out2l1_nd2nz": false,
            "UB_SIZE": 253952, "L2_SIZE": 33554432, "L1_SIZE": 524288,
            "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, 
            "CORE_NUM": 64}
            })";
    std::map<std::string, std::string> soc_infos;
    std::map<std::string, std::string> aicore_spec;
    std::map<std::string, std::string> intrinsics;
    std::map<std::string, std::string> socversions = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();

    Ops::Base::BroadcastCompileInfo compile_info;

    std::string op_type("PRelu");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

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
        "version", socversions);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());

    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&xShape, &weightShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, weightDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, yDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);

    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    tiling_context->GetPlatformInfo()->SetPlatformRes("version", socversions);

    EXPECT_EQ(tiling_func(tiling_context), expectStatus);
}

TEST_F(PReluTilingTest, test_p_relu_tiling_fp16_weight_channel)
{
    gert::StorageShape xShape = {{4, 16, 4, 4}, {4, 16, 4, 4}};
    gert::StorageShape weightShape = {{1, 16, 1, 1}, {1, 16, 1, 1}};
    gert::StorageShape yShape = {{4, 16, 4, 4}, {4, 16, 4, 4}};
    ExecuteTestCase(ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, xShape, weightShape, yShape, ge::GRAPH_SUCCESS);
}

TEST_F(PReluTilingTest, test_p_relu_tiling_fp16_weight_scalar)
{
    gert::StorageShape xShape = {{4, 16, 4, 4}, {4, 16, 4, 4}};
    gert::StorageShape weightShape = {{1}, {1}};
    gert::StorageShape yShape = {{4, 16, 4, 4}, {4, 16, 4, 4}};
    ExecuteTestCase(ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, xShape, weightShape, yShape, ge::GRAPH_SUCCESS);
}

TEST_F(PReluTilingTest, test_p_relu_tiling_fp32_weight_channel)
{
    gert::StorageShape xShape = {{4, 16, 4, 4}, {4, 16, 4, 4}};
    gert::StorageShape weightShape = {{1, 16, 1, 1}, {1, 16, 1, 1}};
    gert::StorageShape yShape = {{4, 16, 4, 4}, {4, 16, 4, 4}};
    ExecuteTestCase(ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, xShape, weightShape, yShape, ge::GRAPH_SUCCESS);
}

TEST_F(PReluTilingTest, test_p_relu_tiling_fp32_weight_scalar)
{
    gert::StorageShape xShape = {{4, 16, 4, 4}, {4, 16, 4, 4}};
    gert::StorageShape weightShape = {{1}, {1}};
    gert::StorageShape yShape = {{4, 16, 4, 4}, {4, 16, 4, 4}};
    ExecuteTestCase(ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, xShape, weightShape, yShape, ge::GRAPH_SUCCESS);
}

TEST_F(PReluTilingTest, test_p_relu_tiling_bf16_weight_channel)
{
    gert::StorageShape xShape = {{4, 16, 4, 4}, {4, 16, 4, 4}};
    gert::StorageShape weightShape = {{1, 16, 1, 1}, {1, 16, 1, 1}};
    gert::StorageShape yShape = {{4, 16, 4, 4}, {4, 16, 4, 4}};
    ExecuteTestCase(ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, xShape, weightShape, yShape, ge::GRAPH_SUCCESS);
}

TEST_F(PReluTilingTest, test_p_relu_tiling_bf16_weight_scalar)
{
    gert::StorageShape xShape = {{4, 16, 4, 4}, {4, 16, 4, 4}};
    gert::StorageShape weightShape = {{1}, {1}};
    gert::StorageShape yShape = {{4, 16, 4, 4}, {4, 16, 4, 4}};
    ExecuteTestCase(ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, xShape, weightShape, yShape, ge::GRAPH_SUCCESS);
}

TEST_F(PReluTilingTest, test_p_relu_tiling_5d_fp16)
{
    gert::StorageShape xShape = {{2, 4, 4, 4, 7}, {2, 4, 4, 4, 7}};
    gert::StorageShape weightShape = {{1, 4, 1, 1, 1}, {1, 4, 1, 1, 1}};
    gert::StorageShape yShape = {{2, 4, 4, 4, 7}, {2, 4, 4, 4, 7}};
    ExecuteTestCase(ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, xShape, weightShape, yShape, ge::GRAPH_SUCCESS);
}

TEST_F(PReluTilingTest, test_p_relu_tiling_failed_dtype_mismatch_x_weight)
{
    gert::StorageShape xShape = {{4, 16, 4, 4}, {4, 16, 4, 4}};
    gert::StorageShape weightShape = {{1, 16, 1, 1}, {1, 16, 1, 1}};
    gert::StorageShape yShape = {{4, 16, 4, 4}, {4, 16, 4, 4}};
    ExecuteTestCase(ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT16, xShape, weightShape, yShape, ge::GRAPH_FAILED);
}

TEST_F(PReluTilingTest, test_p_relu_tiling_failed_dtype_mismatch_x_y)
{
    gert::StorageShape xShape = {{4, 16, 4, 4}, {4, 16, 4, 4}};
    gert::StorageShape weightShape = {{1, 16, 1, 1}, {1, 16, 1, 1}};
    gert::StorageShape yShape = {{4, 16, 4, 4}, {4, 16, 4, 4}};
    ExecuteTestCase(ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, xShape, weightShape, yShape, ge::GRAPH_FAILED);
}

TEST_F(PReluTilingTest, test_p_relu_tiling_failed_unsupported_dtype)
{
    gert::StorageShape xShape = {{4, 16, 4, 4}, {4, 16, 4, 4}};
    gert::StorageShape weightShape = {{1, 16, 1, 1}, {1, 16, 1, 1}};
    gert::StorageShape yShape = {{4, 16, 4, 4}, {4, 16, 4, 4}};
    ExecuteTestCase(ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, xShape, weightShape, yShape, ge::GRAPH_FAILED);
}
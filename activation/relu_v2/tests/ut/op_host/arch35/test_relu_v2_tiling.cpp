/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_nll_loss_tiling.cpp
 * \brief dynamic tiling test of nll_loss
 */
#include <iostream>
#include <fstream>
#include <vector>
#include "log/log.h"
#include <gtest/gtest.h>
#include "register/op_impl_registry.h"
#include "platform/platform_infos_def.h"
#include "ut_op_common.h"
#include "ut_op_util.h"
#include "activation/relu_v2/op_host/arch35/relu_v2_tiling_arch35.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"

using namespace ut_util;
using namespace std;
using namespace ge;

class ReluV2TilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "ReluV2TilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "ReluV2TilingTest TearDown" << std::endl;
    }
};

static string TilingData2Str(const gert::TilingData* tiling_data)
{
    auto data = tiling_data->GetData();
    string result;
    for (size_t i = 0; i < tiling_data->GetDataSize(); i += sizeof(int64_t)) {
        result += std::to_string((reinterpret_cast<const int64_t*>(tiling_data->GetData())[i / sizeof(int64_t)]));
        result += " ";
    }

    return result;
}

TEST_F(ReluV2TilingTest, test_tiling_failed_bf16_fp32_007)
{
    std::string op_type("ReluV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape Shape = {{1, 64, 2, 64}, {1, 64, 2, 64}};

    fe::PlatFormInfos platform_info;
    platform_info.Init();

    optiling::ReluV2CompileInfo compile_info;
    compile_info.coreNum = 64;
    compile_info.ubSize = 262144;

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(1, 2)
                      .IrInstanceNum({1})
                      .InputShapes({&Shape})
                      .OutputShapes({&Shape, &Shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_UINT1, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();
    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(ReluV2TilingTest, test_tiling_failed_empty_tensor_008)
{
    std::string op_type("ReluV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape Shape = {{1, 0, 2, 64}, {1, 0, 2, 64}};
    fe::PlatFormInfos platform_info;
    platform_info.Init();

    optiling::ReluV2CompileInfo compile_info;
    compile_info.coreNum = 64;
    compile_info.ubSize = 262144;

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(1, 2)
                      .IrInstanceNum({1})
                      .InputShapes({&Shape})
                      .OutputShapes({&Shape, &Shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_UINT1, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();
    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(ReluV2TilingTest, test_tiling_failed_unsupport_type_009)
{
    std::string op_type("ReluV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape Shape = {{1, 1, 2, 64}, {1, 1, 2, 64}};
    fe::PlatFormInfos platform_info;
    platform_info.Init();

    optiling::ReluV2CompileInfo compile_info;
    compile_info.coreNum = 64;
    compile_info.ubSize = 262144;

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(1, 2)
                      .IrInstanceNum({1})
                      .InputShapes({&Shape})
                      .OutputShapes({&Shape, &Shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_DOUBLE, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_DOUBLE, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_UINT1, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();
    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}
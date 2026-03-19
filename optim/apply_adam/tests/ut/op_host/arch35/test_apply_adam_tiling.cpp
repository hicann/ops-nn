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
 * \file test_apply_adam_tiling.cpp
 * \brief
 */

#include <gtest/gtest.h>

#include <fstream>
#include <iostream>
#include <vector>

#include "../../../../op_host/arch35/apply_adam_tiling.h"
#include "ut_op_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"

using namespace ut_util;
using namespace std;
using namespace ge;

class TestApplyAdamTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "TestApplyAdamTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "TestApplyAdamTiling TearDown" << std::endl;
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

static void InitPlatForm(
    fe::PlatFormInfos& platFormInfo, map<string, string>& socInfos, map<string, string>& aicoreSpec,
    map<string, string>& intrinsics, map<string, string>& socVersion)
{
    string compile_info_string = R"({
         "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                           "Intrinsic_fix_pipe_l0c2out": false,
                           "Intrinsic_data_move_l12ub": true,
                           "Intrinsic_data_move_l0c2ub": true,
                           "Intrinsic_data_move_out2l1_nd2nz": false,
                           "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                           "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                           "CORE_NUM": 64, "socVersion": "Ascend950"}})";
    GetPlatFormInfos(compile_info_string.c_str(), socInfos, aicoreSpec, intrinsics, socVersion);

    // platform info
    platFormInfo.Init();
}

struct ApplyAdamUtCompileInfo {};

static void DoApplyAdamTilingCase(
    std::initializer_list<int64_t>& inputShape, ge::DataType tensorDtype, ge::Format inputFormat, bool use_nesterov,
    int64_t expectKey, std::string& expectStr)
{
    // init platform
    fe::PlatFormInfos platFormInfo;
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    map<string, string> socVersion = {{"Short_SoC_version", "ASCEND950"}};
    InitPlatForm(platFormInfo, socInfos, aicoreSpec, intrinsics, socVersion);

    // scalar inputs must be float32
    ge::DataType scalarDtype = ge::DT_FLOAT;

    std::string opType("ApplyAdam");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);

    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling_parse;

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    gert::StorageShape tensorShape = {inputShape, inputShape};
    gert::StorageShape oneShape = {{1}, {1}};

    ApplyAdamUtCompileInfo compileInfo;

    auto holder = gert::TilingContextFaker()
                      .SetOpType(opType)
                      .NodeIoNum(10, 1)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes(
                          {&tensorShape, &tensorShape, &tensorShape, &oneShape, &oneShape, &oneShape, &oneShape,
                           &oneShape, &oneShape, &tensorShape})
                      .OutputShapes({&tensorShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platFormInfo))
                      .NodeInputTd(0, tensorDtype, inputFormat, inputFormat)
                      .NodeInputTd(1, tensorDtype, inputFormat, inputFormat)
                      .NodeInputTd(2, tensorDtype, inputFormat, inputFormat)
                      .NodeInputTd(3, scalarDtype, inputFormat, inputFormat)
                      .NodeInputTd(4, scalarDtype, inputFormat, inputFormat)
                      .NodeInputTd(5, scalarDtype, inputFormat, inputFormat)
                      .NodeInputTd(6, scalarDtype, inputFormat, inputFormat)
                      .NodeInputTd(7, scalarDtype, inputFormat, inputFormat)
                      .NodeInputTd(8, scalarDtype, inputFormat, inputFormat)
                      .NodeInputTd(9, tensorDtype, inputFormat, inputFormat)
                      .NodeOutputTd(0, tensorDtype, inputFormat, inputFormat)
                      .NodeAttrs(
                          {{"use_nesterov", Ops::NN::AnyValue::CreateFrom<bool>(use_nesterov)},
                           {"use_locking", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    tiling_context->GetPlatformInfo()->SetPlatformRes("version", socVersion);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);

    auto tiling_data_result = TilingData2Str(tiling_context->GetRawTilingData());
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, expectKey);
}

TEST_F(TestApplyAdamTiling, apply_adam_testcase_float32)
{
    // FLOAT
    std::initializer_list<int64_t> inputShape = {64, 10, 10, 32};
    auto tensorDtype = ge::DT_FLOAT;
    auto inputFormat = ge::FORMAT_ND;
    bool use_nesterov = false;
    auto expectKey = 103;
    std::string expectStr = "";
    DoApplyAdamTilingCase(inputShape, tensorDtype, inputFormat, use_nesterov, expectKey, expectStr);
}

TEST_F(TestApplyAdamTiling, apply_adam_testcase_float16)
{
    // FLOAT16
    std::initializer_list<int64_t> inputShape = {64, 10, 10, 32};
    auto tensorDtype = ge::DT_FLOAT16;
    auto inputFormat = ge::FORMAT_ND;
    bool use_nesterov = false;
    auto expectKey = 101;
    std::string expectStr = "";
    DoApplyAdamTilingCase(inputShape, tensorDtype, inputFormat, use_nesterov, expectKey, expectStr);
}

TEST_F(TestApplyAdamTiling, apply_adam_testcase_bfloat16)
{
    // BFLOAT16
    std::initializer_list<int64_t> inputShape = {64, 10, 10, 32};
    auto tensorDtype = ge::DT_BF16;
    auto inputFormat = ge::FORMAT_ND;
    bool use_nesterov = false;
    auto expectKey = 102;
    std::string expectStr = "";
    DoApplyAdamTilingCase(inputShape, tensorDtype, inputFormat, use_nesterov, expectKey, expectStr);
}

TEST_F(TestApplyAdamTiling, apply_adam_testcase_float32_nesterov)
{
    // FLOAT with nesterov
    std::initializer_list<int64_t> inputShape = {64, 10, 10, 32};
    auto tensorDtype = ge::DT_FLOAT;
    auto inputFormat = ge::FORMAT_ND;
    bool use_nesterov = true;
    auto expectKey = 103;
    std::string expectStr = "";
    DoApplyAdamTilingCase(inputShape, tensorDtype, inputFormat, use_nesterov, expectKey, expectStr);
}
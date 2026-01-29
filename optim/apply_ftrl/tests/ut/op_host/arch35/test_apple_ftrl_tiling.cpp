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
 * \file test_apply_ftrl_tiling.cpp
 * \brief
 */

#include <gtest/gtest.h>

#include <fstream>
#include <iostream>
#include <vector>

#include "../../../../op_host/apply_ftrl_tiling.h"
#include "ut_op_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"

using namespace ut_util;
using namespace std;
using namespace ge;

class TestApplyFtrlTiling : public testing::Test {
   protected:
    static void SetUpTestCase() { std::cout << "TestApplyFtrlTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "TestApplyFtrlTiling TearDown" << std::endl; }
};

static string TilingData2Str(const gert::TilingData* tiling_data) {
    auto data = tiling_data->GetData();
    string result;
    for (size_t i = 0; i < tiling_data->GetDataSize(); i += sizeof(int64_t)) {
        result += std::to_string((reinterpret_cast<const int64_t*>(tiling_data->GetData())[i / sizeof(int64_t)]));
        result += " ";
    }

    return result;
}

static void InitPlatForm(fe::PlatFormInfos& platFormInfo, map<string, string>& socInfos,
                         map<string, string>& aicoreSpec, map<string, string>& intrinsics,
                         map<string, string>& socVersion) {
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

struct ApplyFtrlUtCompileInfo {
};

static void DoApplyFtrlTilingCase(std::initializer_list<int64_t>& inputShape, ge::DataType inputDtype,
                                  ge::Format inputFormat, std::string& expectStr) {
    // init platform
    fe::PlatFormInfos platFormInfo;
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    map<string, string> socVersion = {{"Short_SoC_version", "ASCEND950"}};
    InitPlatForm(platFormInfo, socInfos, aicoreSpec, intrinsics, socVersion);

    std::string opType("ApplyFtrl");
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

    ApplyFtrlUtCompileInfo compileInfo;

    auto holder = gert::TilingContextFaker()
                      .SetOpType(opType)
                      .NodeIoNum(8, 1)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&tensorShape, &tensorShape, &tensorShape, &tensorShape, &oneShape, &oneShape, &oneShape, &oneShape})
                      .OutputShapes({&tensorShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platFormInfo))
                      .NodeInputTd(0, inputDtype, inputFormat, inputFormat)
                      .NodeInputTd(1, inputDtype, inputFormat, inputFormat)
                      .NodeInputTd(2, inputDtype, inputFormat, inputFormat)
                      .NodeInputTd(3, inputDtype, inputFormat, inputFormat)
                      .NodeInputTd(4, inputDtype, inputFormat, inputFormat)
                      .NodeInputTd(5, inputDtype, inputFormat, inputFormat)
                      .NodeInputTd(6, inputDtype, inputFormat, inputFormat)
                      .NodeInputTd(7, inputDtype, inputFormat, inputFormat)
                      .NodeOutputTd(0, inputDtype, inputFormat, inputFormat)
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
    EXPECT_EQ(tiling_data_result, expectStr);
}

// TEST_F(TestApplyFtrlTiling, apply_ftrl_testcase_float32) {
//     // FLOAT
//     std::initializer_list<int64_t> inputShape = {16, 26, 16, 19};
//     auto inputDtype = ge::DT_FLOAT;
//     auto inputFormat = ge::FORMAT_ND;
//     std::string expectStr =
//         "1 126464 4080 31 2816 2 2 1264 1248 2816 0 0 ";
//     DoApplyFtrlTilingCase(inputShape, inputDtype, inputFormat, expectStr);
// }

// TEST_F(TestApplyFtrlTiling, apply_ftrl_testcase_float16) {
//     // FLOAT16
//     std::initializer_list<int64_t> inputShape = {3761, 4, 44, 4};
//     auto inputDtype = ge::DT_FLOAT16;
//     auto inputFormat = ge::FORMAT_ND;
//     std::string expectStr =
//         "1 2647744 41376 64 2560 17 17 416 96 2560 0 0 ";
//     DoApplyFtrlTilingCase(inputShape, inputDtype, inputFormat, expectStr);
// }

// TEST_F(TestApplyFtrlTiling, apply_ftrl_testcase_bfloat16) {
//     // BFLOAT16
//     std::initializer_list<int64_t> inputShape = {7, 2, 7, 8, 10};
//     auto inputDtype = ge::DT_BF16;
//     auto inputFormat = ge::FORMAT_ND;
//     std::string expectStr =
//         "1 7840 3920 2 2560 2 2 1360 1360 2560 0 0 ";
//     DoApplyFtrlTilingCase(inputShape, inputDtype, inputFormat, expectStr);
// }
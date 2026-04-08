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
 * \file test_sigmoid_cross_entropy_with_logits_tiling.cpp
 * \brief SigmoidCrossEntropyWithLogits 算子 Tiling 单元测试
 */

#include <gtest/gtest.h>

#include <fstream>
#include <iostream>
#include <vector>

#include "../../../../op_host/arch35/sigmoid_cross_entropy_with_logits_tiling.h"
#include "ut_op_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"

using namespace ut_util;
using namespace std;
using namespace ge;

class TestSigmoidCrossEntropyWithLogitsTiling : public testing::Test {
   protected:
    static void SetUpTestCase() { std::cout << "TestSigmoidCrossEntropyWithLogitsTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "TestSigmoidCrossEntropyWithLogitsTiling TearDown" << std::endl; }
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

struct SigmoidCrossEntropyWithLogitsUtCompileInfo {
};

static void DoSigmoidCrossEntropyWithLogitsTilingCase(std::initializer_list<int64_t>& inputShape, ge::DataType inputDtype,
                                      ge::Format inputFormat, int64_t expectKey, std::string& expectStr) {
    // init platform
    fe::PlatFormInfos platFormInfo;
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    map<string, string> socVersion = {{"Short_SoC_version", "ASCEND950"}};
    InitPlatForm(platFormInfo, socInfos, aicoreSpec, intrinsics, socVersion);

    std::string opType("SigmoidCrossEntropyWithLogits");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);

    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling_parse;

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    gert::StorageShape tensorShape = {inputShape, inputShape};

    SigmoidCrossEntropyWithLogitsUtCompileInfo compileInfo;

    auto holder = gert::TilingContextFaker()
                        .SetOpType(opType)
                        .NodeIoNum(2, 1)
                        .IrInstanceNum({1, 1})
                        .InputShapes({&tensorShape, &tensorShape})
                        .OutputShapes({&tensorShape})
                        .CompileInfo(&compileInfo)
                        .PlatformInfo(reinterpret_cast<char*>(&platFormInfo))
                        .NodeInputTd(0, inputDtype, inputFormat, inputFormat)
                        .NodeInputTd(1, inputDtype, inputFormat, inputFormat)
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
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, expectKey);
}

// FP32: TPL_FP32=3, tilingKey=3
TEST_F(TestSigmoidCrossEntropyWithLogitsTiling, sigmoid_cross_entropy_with_logits_testcase_float32) {
    std::initializer_list<int64_t> inputShape = {16, 26, 16, 19};
    auto inputDtype = ge::DT_FLOAT;
    auto inputFormat = ge::FORMAT_ND;
    auto expectKey = 3;  // TPL_FP32
    std::string expectStr = "";
    DoSigmoidCrossEntropyWithLogitsTilingCase(inputShape, inputDtype, inputFormat, expectKey, expectStr);
}

TEST_F(TestSigmoidCrossEntropyWithLogitsTiling, sigmoid_cross_entropy_with_logits_testcase_float32_1d) {
    std::initializer_list<int64_t> inputShape = {1024};
    auto inputDtype = ge::DT_FLOAT;
    auto inputFormat = ge::FORMAT_ND;
    auto expectKey = 3;  // TPL_FP32
    std::string expectStr = "";
    DoSigmoidCrossEntropyWithLogitsTilingCase(inputShape, inputDtype, inputFormat, expectKey, expectStr);
}

TEST_F(TestSigmoidCrossEntropyWithLogitsTiling, sigmoid_cross_entropy_with_logits_testcase_float32_2d) {
    std::initializer_list<int64_t> inputShape = {128, 256};
    auto inputDtype = ge::DT_FLOAT;
    auto inputFormat = ge::FORMAT_ND;
    auto expectKey = 3;  // TPL_FP32
    std::string expectStr = "";
    DoSigmoidCrossEntropyWithLogitsTilingCase(inputShape, inputDtype, inputFormat, expectKey, expectStr);
}

// FP16: TPL_FP16=1, tilingKey=1
TEST_F(TestSigmoidCrossEntropyWithLogitsTiling, sigmoid_cross_entropy_with_logits_testcase_float16) {
    std::initializer_list<int64_t> inputShape = {3761, 4, 44, 4};
    auto inputDtype = ge::DT_FLOAT16;
    auto inputFormat = ge::FORMAT_ND;
    auto expectKey = 1;  // TPL_FP16
    std::string expectStr = "";
    DoSigmoidCrossEntropyWithLogitsTilingCase(inputShape, inputDtype, inputFormat, expectKey, expectStr);
}

TEST_F(TestSigmoidCrossEntropyWithLogitsTiling, sigmoid_cross_entropy_with_logits_testcase_float16_1d) {
    std::initializer_list<int64_t> inputShape = {2048};
    auto inputDtype = ge::DT_FLOAT16;
    auto inputFormat = ge::FORMAT_ND;
    auto expectKey = 1;  // TPL_FP16
    std::string expectStr = "";
    DoSigmoidCrossEntropyWithLogitsTilingCase(inputShape, inputDtype, inputFormat, expectKey, expectStr);
}

TEST_F(TestSigmoidCrossEntropyWithLogitsTiling, sigmoid_cross_entropy_with_logits_testcase_float16_3d) {
    std::initializer_list<int64_t> inputShape = {32, 64, 128};
    auto inputDtype = ge::DT_FLOAT16;
    auto inputFormat = ge::FORMAT_ND;
    auto expectKey = 1;  // TPL_FP16
    std::string expectStr = "";
    DoSigmoidCrossEntropyWithLogitsTilingCase(inputShape, inputDtype, inputFormat, expectKey, expectStr);
}

// BF16: TPL_BF16=2, tilingKey=2
TEST_F(TestSigmoidCrossEntropyWithLogitsTiling, sigmoid_cross_entropy_with_logits_testcase_bfloat16) {
    std::initializer_list<int64_t> inputShape = {7, 2, 7, 8, 10};
    auto inputDtype = ge::DT_BF16;
    auto inputFormat = ge::FORMAT_ND;
    auto expectKey = 2;  // TPL_BF16
    std::string expectStr = "";
    DoSigmoidCrossEntropyWithLogitsTilingCase(inputShape, inputDtype, inputFormat, expectKey, expectStr);
}

TEST_F(TestSigmoidCrossEntropyWithLogitsTiling, sigmoid_cross_entropy_with_logits_testcase_bfloat16_1d) {
    std::initializer_list<int64_t> inputShape = {512};
    auto inputDtype = ge::DT_BF16;
    auto inputFormat = ge::FORMAT_ND;
    auto expectKey = 2;  // TPL_BF16
    std::string expectStr = "";
    DoSigmoidCrossEntropyWithLogitsTilingCase(inputShape, inputDtype, inputFormat, expectKey, expectStr);
}

TEST_F(TestSigmoidCrossEntropyWithLogitsTiling, sigmoid_cross_entropy_with_logits_testcase_bfloat16_4d) {
    std::initializer_list<int64_t> inputShape = {16, 32, 64, 128};
    auto inputDtype = ge::DT_BF16;
    auto inputFormat = ge::FORMAT_ND;
    auto expectKey = 2;  // TPL_BF16
    std::string expectStr = "";
    DoSigmoidCrossEntropyWithLogitsTilingCase(inputShape, inputDtype, inputFormat, expectKey, expectStr);
}

// Large shape test
TEST_F(TestSigmoidCrossEntropyWithLogitsTiling, sigmoid_cross_entropy_with_logits_testcase_float32_large) {
    std::initializer_list<int64_t> inputShape = {1024, 1024};
    auto inputDtype = ge::DT_FLOAT;
    auto inputFormat = ge::FORMAT_ND;
    auto expectKey = 3;  // TPL_FP32
    std::string expectStr = "";
    DoSigmoidCrossEntropyWithLogitsTilingCase(inputShape, inputDtype, inputFormat, expectKey, expectStr);
}

TEST_F(TestSigmoidCrossEntropyWithLogitsTiling, sigmoid_cross_entropy_with_logits_testcase_float16_large) {
    std::initializer_list<int64_t> inputShape = {2048, 2048};
    auto inputDtype = ge::DT_FLOAT16;
    auto inputFormat = ge::FORMAT_ND;
    auto expectKey = 1;  // TPL_FP16
    std::string expectStr = "";
    DoSigmoidCrossEntropyWithLogitsTilingCase(inputShape, inputDtype, inputFormat, expectKey, expectStr);
}

TEST_F(TestSigmoidCrossEntropyWithLogitsTiling, sigmoid_cross_entropy_with_logits_testcase_bfloat16_large) {
    std::initializer_list<int64_t> inputShape = {512, 512, 512};
    auto inputDtype = ge::DT_BF16;
    auto inputFormat = ge::FORMAT_ND;
    auto expectKey = 2;  // TPL_BF16
    std::string expectStr = "";
    DoSigmoidCrossEntropyWithLogitsTilingCase(inputShape, inputDtype, inputFormat, expectKey, expectStr);
}

// Small shape test
TEST_F(TestSigmoidCrossEntropyWithLogitsTiling, sigmoid_cross_entropy_with_logits_testcase_float32_small) {
    std::initializer_list<int64_t> inputShape = {8, 8};
    auto inputDtype = ge::DT_FLOAT;
    auto inputFormat = ge::FORMAT_ND;
    auto expectKey = 3;  // TPL_FP32
    std::string expectStr = "";
    DoSigmoidCrossEntropyWithLogitsTilingCase(inputShape, inputDtype, inputFormat, expectKey, expectStr);
}

TEST_F(TestSigmoidCrossEntropyWithLogitsTiling, sigmoid_cross_entropy_with_logits_testcase_float16_small) {
    std::initializer_list<int64_t> inputShape = {4, 4};
    auto inputDtype = ge::DT_FLOAT16;
    auto inputFormat = ge::FORMAT_ND;
    auto expectKey = 1;  // TPL_FP16
    std::string expectStr = "";
    DoSigmoidCrossEntropyWithLogitsTilingCase(inputShape, inputDtype, inputFormat, expectKey, expectStr);
}
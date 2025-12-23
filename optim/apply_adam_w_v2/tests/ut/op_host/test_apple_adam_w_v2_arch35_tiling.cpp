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
 * \file test_apple_adam_w_v2_arch35.cpp
 * \brief
 */

#include <gtest/gtest.h>

#include <fstream>
#include <iostream>
#include <vector>

#include "../../../op_host/apply_adam_w_v2_tiling.h"
#include "../../../op_host/apply_adam_w_v2_tiling_arch35.h"
#include "ut_op_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"

using namespace ut_util;
using namespace std;
using namespace ge;

class TestApplyAdamWV2RegbaseTiling : public testing::Test {
   protected:
    static void SetUpTestCase() { std::cout << "TestApplyAdamWV2RegbaseTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "TestApplyAdamWV2RegbaseTiling TearDown" << std::endl; }
};

static string TilingData2Str(const gert::TilingData* tiling_data) {
    auto data = tiling_data->GetData();
    string result;
    for (size_t i = tiling_data->GetDataSize() - sizeof(int64_t); i < tiling_data->GetDataSize(); i += sizeof(int64_t)) {
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
                           "CORE_NUM": 64, "socVersion": "Ascend910_95"}})";
    GetPlatFormInfos(compile_info_string.c_str(), socInfos, aicoreSpec, intrinsics, socVersion);

    // platform info
    platFormInfo.Init();
}

static void DoApplyAdamWV2RegbaseAmsGradTilingCase(std::initializer_list<int64_t>& inputShape,
                                                    ge::DataType inputDtype, ge::DataType gradDtype,
                                                    ge::DataType stepDtype, float lr, float beta1, float beta2,
                                                    float weightDecay, float eps, bool amsgrad, bool maximize,
                                                    std::string& expectStr) {
    // init platform
    fe::PlatFormInfos platFormInfo;
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    map<string, string> socVersion = {{"Short_SoC_version", "ASCEND910_95"}};
    InitPlatForm(platFormInfo, socInfos, aicoreSpec, intrinsics, socVersion);

    std::string opType("ApplyAdamWV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);

    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling_parse;

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    gert::StorageShape varShape = {inputShape, inputShape};
    gert::StorageShape mShape = {inputShape, inputShape};
    gert::StorageShape vShape = {inputShape, inputShape};
    gert::StorageShape gradShape = {inputShape, inputShape};
    gert::StorageShape stepShape = {{1}, {1}};

    // compile info
    optiling::ApplyAdamWV2CompileInfo compileInfo;
    compileInfo.totalCoreNum = 64;
    compileInfo.ubSize = 262114;
    compileInfo.isRegbase = true;

    gert::StorageShape maxGradNormShape = {inputShape, inputShape};
    auto holder = gert::TilingContextFaker()
                      .SetOpType(opType)
                      .NodeIoNum(6, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1})
                      .InputShapes({&varShape, &mShape, &vShape, &gradShape, &stepShape, &maxGradNormShape})
                      .OutputShapes({&varShape, &mShape, &vShape, &maxGradNormShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platFormInfo))
                      .NodeInputTd(0, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, gradDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, stepDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, gradDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, gradDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"lr", Ops::NN::AnyValue::CreateFrom<float>(lr)},
                                  {"beta1", Ops::NN::AnyValue::CreateFrom<float>(beta1)},
                                  {"beta2", Ops::NN::AnyValue::CreateFrom<float>(beta2)},
                                  {"weight_decay", Ops::NN::AnyValue::CreateFrom<float>(weightDecay)},
                                  {"eps", Ops::NN::AnyValue::CreateFrom<float>(eps)},
                                  {"amsgrad", Ops::NN::AnyValue::CreateFrom<bool>(amsgrad)},
                                  {"maximize", Ops::NN::AnyValue::CreateFrom<bool>(maximize)}})
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

static void DoApplyAdamWV2RegbaseTilingCase(std::initializer_list<int64_t>& inputShape,
                                                    ge::DataType inputDtype, ge::DataType gradDtype,
                                                    ge::DataType stepDtype, float lr, float beta1, float beta2,
                                                    float weightDecay, float eps, bool amsgrad, bool maximize,
                                                    std::string& expectStr) {
    // init platform
    fe::PlatFormInfos platFormInfo;
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    map<string, string> socVersion = {{"Short_SoC_version", "ASCEND910_95"}};
    InitPlatForm(platFormInfo, socInfos, aicoreSpec, intrinsics, socVersion);

    std::string opType("ApplyAdamWV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);

    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling_parse;

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    gert::StorageShape varShape = {inputShape, inputShape};
    gert::StorageShape mShape = {inputShape, inputShape};
    gert::StorageShape vShape = {inputShape, inputShape};
    gert::StorageShape gradShape = {inputShape, inputShape};
    gert::StorageShape stepShape = {{1}, {1}};

    // compile info
    optiling::ApplyAdamWV2CompileInfo compileInfo;
    compileInfo.totalCoreNum = 64;
    compileInfo.ubSize = 262114;
    compileInfo.isRegbase = true;

    auto holder = gert::TilingContextFaker()
                      .SetOpType(opType)
                      .NodeIoNum(5, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1})
                      .InputShapes({&varShape, &mShape, &vShape, &gradShape, &stepShape})
                      .OutputShapes({&varShape, &mShape, &vShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platFormInfo))
                      .NodeInputTd(0, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, gradDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, stepDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"lr", Ops::NN::AnyValue::CreateFrom<float>(lr)},
                                  {"beta1", Ops::NN::AnyValue::CreateFrom<float>(beta1)},
                                  {"beta2", Ops::NN::AnyValue::CreateFrom<float>(beta2)},
                                  {"weight_decay", Ops::NN::AnyValue::CreateFrom<float>(weightDecay)},
                                  {"eps", Ops::NN::AnyValue::CreateFrom<float>(eps)},
                                  {"amsgrad", Ops::NN::AnyValue::CreateFrom<bool>(amsgrad)},
                                  {"maximize", Ops::NN::AnyValue::CreateFrom<bool>(maximize)}})
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

TEST_F(TestApplyAdamWV2RegbaseTiling, adamw_v2_float_int64_amsgrad) {
    // FLOAT
    std::initializer_list<int64_t> inputShape = {2048, 16384};
    float lr = 0.0;
    float beta1 = 0.0;
    float beta2 = 0.0;
    float weightDecay = 0.0;
    float eps = 0.0;
    bool amsgrad = true;

    bool maximize = false;
    std::string expectStr =
        "4575657221408423936 ";
    DoApplyAdamWV2RegbaseAmsGradTilingCase(inputShape, ge::DT_FLOAT /*inputDtype*/, ge::DT_FLOAT /*gradDtype*/, ge::DT_INT64 /*stepDtype*/,
        lr, beta1, beta2, weightDecay, eps, amsgrad, maximize, expectStr);
}

TEST_F(TestApplyAdamWV2RegbaseTiling, adamw_v2_float_int64) {
    // FLOAT
    std::initializer_list<int64_t> inputShape = {2048, 16384};
    float lr = 0.1;
    float beta1 = 0.3;
    float beta2 = 0.3;
    float weightDecay = 0.3;
    float eps = 0.2;
    bool amsgrad = false;

    bool maximize = false;
    std::string expectStr =
        "4575657222453644493 ";
    DoApplyAdamWV2RegbaseTilingCase(inputShape, ge::DT_FLOAT /*inputDtype*/, ge::DT_FLOAT /*gradDtype*/, ge::DT_INT64 /*stepDtype*/,
        lr, beta1, beta2, weightDecay, eps, amsgrad, maximize, expectStr);
}

TEST_F(TestApplyAdamWV2RegbaseTiling, adamw_v2_float16_int64_amsgrad) {
    // FLOAT
    std::initializer_list<int64_t> inputShape = {2048, 16384};
    float lr = 0.0;
    float beta1 = 0.0;
    float beta2 = 0.0;
    float weightDecay = 0.0;
    float eps = 0.0;
    bool amsgrad = true;

    bool maximize = false;
    std::string expectStr =
        "4575657221408423936 ";
    DoApplyAdamWV2RegbaseAmsGradTilingCase(inputShape, ge::DT_FLOAT16 /*inputDtype*/, ge::DT_FLOAT16 /*gradDtype*/, ge::DT_INT64 /*stepDtype*/,
        lr, beta1, beta2, weightDecay, eps, amsgrad, maximize, expectStr);
}

TEST_F(TestApplyAdamWV2RegbaseTiling, adamw_v2_float16_int64) {
    // FLOAT
    std::initializer_list<int64_t> inputShape = {2048, 16384};
    float lr = 0.1;
    float beta1 = 0.3;
    float beta2 = 0.3;
    float weightDecay = 0.3;
    float eps = 0.2;
    bool amsgrad = false;

    bool maximize = true;
    std::string expectStr =
        "-4647714814401131315 ";
    DoApplyAdamWV2RegbaseTilingCase(inputShape, ge::DT_FLOAT16 /*inputDtype*/, ge::DT_FLOAT16 /*gradDtype*/, ge::DT_INT64 /*stepDtype*/,
        lr, beta1, beta2, weightDecay, eps, amsgrad, maximize, expectStr);
}

TEST_F(TestApplyAdamWV2RegbaseTiling, adamw_v2_bfloat16_int64_amsgrad) {
    // FLOAT
    std::initializer_list<int64_t> inputShape = {2048, 16384};
    float lr = 0.0;
    float beta1 = 0.0;
    float beta2 = 0.0;
    float weightDecay = 0.0;
    float eps = 0.0;
    bool amsgrad = true;

    bool maximize = false;
    std::string expectStr =
        "4575657221408423936 ";
    DoApplyAdamWV2RegbaseAmsGradTilingCase(inputShape, ge::DT_FLOAT16 /*inputDtype*/, ge::DT_FLOAT16 /*gradDtype*/, ge::DT_INT64 /*stepDtype*/,
        lr, beta1, beta2, weightDecay, eps, amsgrad, maximize, expectStr);
}

TEST_F(TestApplyAdamWV2RegbaseTiling, adamw_v2_bfloat16_int64) {
    // FLOAT
    std::initializer_list<int64_t> inputShape = {2048, 16384};
    float lr = 0.1;
    float beta1 = 0.3;
    float beta2 = 0.3;
    float weightDecay = 0.3;
    float eps = 0.2;
    bool amsgrad = false;

    bool maximize = true;
    std::string expectStr =
        "-4647714814401131315 ";
    DoApplyAdamWV2RegbaseTilingCase(inputShape, ge::DT_BF16 /*inputDtype*/, ge::DT_BF16 /*gradDtype*/, ge::DT_INT64 /*stepDtype*/,
        lr, beta1, beta2, weightDecay, eps, amsgrad, maximize, expectStr);
}

TEST_F(TestApplyAdamWV2RegbaseTiling, adamw_v2_mixtype_int64_amsgrad_01) {
    // FLOAT
    std::initializer_list<int64_t> inputShape = {2048, 16384};
    float lr = 0.0;
    float beta1 = 0.0;
    float beta2 = 0.0;
    float weightDecay = 0.0;
    float eps = 0.0;
    bool amsgrad = true;

    bool maximize = false;
    std::string expectStr =
        "4575657221408423936 ";
    DoApplyAdamWV2RegbaseAmsGradTilingCase(inputShape, ge::DT_FLOAT /*inputDtype*/, ge::DT_FLOAT16 /*gradDtype*/, ge::DT_INT64 /*stepDtype*/,
        lr, beta1, beta2, weightDecay, eps, amsgrad, maximize, expectStr);
}

TEST_F(TestApplyAdamWV2RegbaseTiling, adamw_v2_mixtype_int64_01) {
    // FLOAT
    std::initializer_list<int64_t> inputShape = {2048, 16384};
    float lr = 0.1;
    float beta1 = 0.3;
    float beta2 = 0.3;
    float weightDecay = 0.3;
    float eps = 0.2;
    bool amsgrad = false;

    bool maximize = true;
    std::string expectStr =
        "-4647714814401131315 ";
    DoApplyAdamWV2RegbaseTilingCase(inputShape, ge::DT_FLOAT /*inputDtype*/, ge::DT_BF16 /*gradDtype*/, ge::DT_INT64 /*stepDtype*/,
        lr, beta1, beta2, weightDecay, eps, amsgrad, maximize, expectStr);
}

TEST_F(TestApplyAdamWV2RegbaseTiling, adamw_v2_mixtype_int64_amsgrad_02) {
    // FLOAT
    std::initializer_list<int64_t> inputShape = {2048, 16384};
    float lr = 0.0;
    float beta1 = 0.0;
    float beta2 = 0.0;
    float weightDecay = 0.0;
    float eps = 0.0;
    bool amsgrad = true;

    bool maximize = false;
    std::string expectStr =
        "4575657221408423936 ";
    DoApplyAdamWV2RegbaseAmsGradTilingCase(inputShape, ge::DT_FLOAT /*inputDtype*/, ge::DT_FLOAT16 /*gradDtype*/, ge::DT_INT64 /*stepDtype*/,
        lr, beta1, beta2, weightDecay, eps, amsgrad, maximize, expectStr);
}

TEST_F(TestApplyAdamWV2RegbaseTiling, adamw_v2_mixtype_int64_02) {
    // FLOAT
    std::initializer_list<int64_t> inputShape = {2048, 16384};
    float lr = 0.1;
    float beta1 = 0.3;
    float beta2 = 0.3;
    float weightDecay = 0.3;
    float eps = 0.2;
    bool amsgrad = false;

    bool maximize = true;
    std::string expectStr =
        "-4647714814401131315 ";
    DoApplyAdamWV2RegbaseTilingCase(inputShape, ge::DT_FLOAT /*inputDtype*/, ge::DT_BF16 /*gradDtype*/, ge::DT_INT64 /*stepDtype*/,
        lr, beta1, beta2, weightDecay, eps, amsgrad, maximize, expectStr);
}



TEST_F(TestApplyAdamWV2RegbaseTiling, adamw_v2_float_float_amsgrad) {
    // FLOAT
    std::initializer_list<int64_t> inputShape = {2048, 16384};
    float lr = 0.0;
    float beta1 = 0.0;
    float beta2 = 0.0;
    float weightDecay = 0.0;
    float eps = 0.0;
    bool amsgrad = true;

    bool maximize = false;
    std::string expectStr =
        "4575657221408423936 ";
    DoApplyAdamWV2RegbaseAmsGradTilingCase(inputShape, ge::DT_FLOAT /*inputDtype*/, ge::DT_FLOAT /*gradDtype*/, ge::DT_FLOAT /*stepDtype*/,
        lr, beta1, beta2, weightDecay, eps, amsgrad, maximize, expectStr);
}

TEST_F(TestApplyAdamWV2RegbaseTiling, adamw_v2_float_float) {
    // FLOAT
    std::initializer_list<int64_t> inputShape = {2048, 16384};
    float lr = 0.1;
    float beta1 = 0.3;
    float beta2 = 0.3;
    float weightDecay = 0.3;
    float eps = 0.2;
    bool amsgrad = false;

    bool maximize = false;
    std::string expectStr =
        "4575657222453644493 ";
    DoApplyAdamWV2RegbaseTilingCase(inputShape, ge::DT_FLOAT /*inputDtype*/, ge::DT_FLOAT /*gradDtype*/, ge::DT_FLOAT /*stepDtype*/,
        lr, beta1, beta2, weightDecay, eps, amsgrad, maximize, expectStr);
}

TEST_F(TestApplyAdamWV2RegbaseTiling, adamw_v2_float16_float_amsgrad) {
    // FLOAT
    std::initializer_list<int64_t> inputShape = {2048, 16384};
    float lr = 0.0;
    float beta1 = 0.0;
    float beta2 = 0.0;
    float weightDecay = 0.0;
    float eps = 0.0;
    bool amsgrad = true;

    bool maximize = false;
    std::string expectStr =
        "4575657221408423936 ";
    DoApplyAdamWV2RegbaseAmsGradTilingCase(inputShape, ge::DT_FLOAT16 /*inputDtype*/, ge::DT_FLOAT16 /*gradDtype*/, ge::DT_FLOAT /*stepDtype*/,
        lr, beta1, beta2, weightDecay, eps, amsgrad, maximize, expectStr);
}

TEST_F(TestApplyAdamWV2RegbaseTiling, adamw_v2_float16_float) {
    // FLOAT
    std::initializer_list<int64_t> inputShape = {2048, 16384};
    float lr = 0.1;
    float beta1 = 0.3;
    float beta2 = 0.3;
    float weightDecay = 0.3;
    float eps = 0.2;
    bool amsgrad = false;

    bool maximize = true;
    std::string expectStr =
        "-4647714814401131315 ";
    DoApplyAdamWV2RegbaseTilingCase(inputShape, ge::DT_FLOAT16 /*inputDtype*/, ge::DT_FLOAT16 /*gradDtype*/, ge::DT_FLOAT /*stepDtype*/,
        lr, beta1, beta2, weightDecay, eps, amsgrad, maximize, expectStr);
}

TEST_F(TestApplyAdamWV2RegbaseTiling, adamw_v2_bfloat16_float_amsgrad) {
    // FLOAT
    std::initializer_list<int64_t> inputShape = {2048, 16384};
    float lr = 0.0;
    float beta1 = 0.0;
    float beta2 = 0.0;
    float weightDecay = 0.0;
    float eps = 0.0;
    bool amsgrad = true;

    bool maximize = false;
    std::string expectStr =
        "4575657221408423936 ";
    DoApplyAdamWV2RegbaseAmsGradTilingCase(inputShape, ge::DT_FLOAT16 /*inputDtype*/, ge::DT_FLOAT16 /*gradDtype*/, ge::DT_FLOAT /*stepDtype*/,
        lr, beta1, beta2, weightDecay, eps, amsgrad, maximize, expectStr);
}

TEST_F(TestApplyAdamWV2RegbaseTiling, adamw_v2_bfloat16_float) {
    // FLOAT
    std::initializer_list<int64_t> inputShape = {2048, 16384};
    float lr = 0.1;
    float beta1 = 0.3;
    float beta2 = 0.3;
    float weightDecay = 0.3;
    float eps = 0.2;
    bool amsgrad = false;

    bool maximize = true;
    std::string expectStr =
        "-4647714814401131315 ";
    DoApplyAdamWV2RegbaseTilingCase(inputShape, ge::DT_BF16 /*inputDtype*/, ge::DT_BF16 /*gradDtype*/, ge::DT_FLOAT /*stepDtype*/,
        lr, beta1, beta2, weightDecay, eps, amsgrad, maximize, expectStr);
}

TEST_F(TestApplyAdamWV2RegbaseTiling, adamw_v2_mixtype_float_amsgrad_01) {
    // FLOAT
    std::initializer_list<int64_t> inputShape = {2048, 16384};
    float lr = 0.0;
    float beta1 = 0.0;
    float beta2 = 0.0;
    float weightDecay = 0.0;
    float eps = 0.0;
    bool amsgrad = true;

    bool maximize = false;
    std::string expectStr =
        "4575657221408423936 ";
    DoApplyAdamWV2RegbaseAmsGradTilingCase(inputShape, ge::DT_FLOAT /*inputDtype*/, ge::DT_FLOAT16 /*gradDtype*/, ge::DT_FLOAT /*stepDtype*/,
        lr, beta1, beta2, weightDecay, eps, amsgrad, maximize, expectStr);
}

TEST_F(TestApplyAdamWV2RegbaseTiling, adamw_v2_mixtype_float_01) {
    // FLOAT
    std::initializer_list<int64_t> inputShape = {2048, 16384};
    float lr = 0.1;
    float beta1 = 0.3;
    float beta2 = 0.3;
    float weightDecay = 0.3;
    float eps = 0.2;
    bool amsgrad = false;

    bool maximize = true;
    std::string expectStr =
        "-4647714814401131315 ";
    DoApplyAdamWV2RegbaseTilingCase(inputShape, ge::DT_FLOAT /*inputDtype*/, ge::DT_BF16 /*gradDtype*/, ge::DT_FLOAT /*stepDtype*/,
        lr, beta1, beta2, weightDecay, eps, amsgrad, maximize, expectStr);
}

TEST_F(TestApplyAdamWV2RegbaseTiling, adamw_v2_mixtype_float_amsgrad_02) {
    // FLOAT
    std::initializer_list<int64_t> inputShape = {2048, 16384};
    float lr = 0.0;
    float beta1 = 0.0;
    float beta2 = 0.0;
    float weightDecay = 0.0;
    float eps = 0.0;
    bool amsgrad = true;

    bool maximize = false;
    std::string expectStr =
        "4575657221408423936 ";
    DoApplyAdamWV2RegbaseAmsGradTilingCase(inputShape, ge::DT_FLOAT /*inputDtype*/, ge::DT_FLOAT16 /*gradDtype*/, ge::DT_FLOAT /*stepDtype*/,
        lr, beta1, beta2, weightDecay, eps, amsgrad, maximize, expectStr);
}

TEST_F(TestApplyAdamWV2RegbaseTiling, adamw_v2_mixtype_float_02) {
    // FLOAT
    std::initializer_list<int64_t> inputShape = {2048, 16384};
    float lr = 0.1;
    float beta1 = 0.3;
    float beta2 = 0.3;
    float weightDecay = 0.3;
    float eps = 0.2;
    bool amsgrad = false;

    bool maximize = true;
    std::string expectStr =
        "-4647714814401131315 ";
    DoApplyAdamWV2RegbaseTilingCase(inputShape, ge::DT_FLOAT /*inputDtype*/, ge::DT_BF16 /*gradDtype*/, ge::DT_FLOAT /*stepDtype*/,
        lr, beta1, beta2, weightDecay, eps, amsgrad, maximize, expectStr);
}

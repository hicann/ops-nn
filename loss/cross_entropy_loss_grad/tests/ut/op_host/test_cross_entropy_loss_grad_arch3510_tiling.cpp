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
* \file test_cross_entropy_loss_grad_arch3510_tiling.cpp
* \brief
*/

#include <iostream>
#include <fstream>
#include <vector>
#include <gtest/gtest.h>
#include "log/log.h"
#include "kernel_run_context_facker.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "test_cube_util.h"
#include "ut_op_util.h"
#include "ut_op_common.h"
#include "platform/platform_infos_def.h"

#include "../../../op_host/cross_entropy_loss_grad_tiling.h"

using namespace ut_util;
using namespace std;
using namespace ge;

class CrossEntropyLossGradRegbaseTiling : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "CrossEntropyLossGradRegbaseTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "CrossEntropyLossGradRegbaseTiling TearDown" << std::endl;
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

static void InitPlatForm(fe::PlatFormInfos& platFormInfo, map<string, string>& socInfos,
                         map<string, string>& aicoreSpec, map<string, string>& intrinsics, map<string, string>& socVersion)
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

static void DoCELossGradRegbaseNoWeightTilingCase(std::initializer_list<int64_t>& inputShape1, std::initializer_list<int64_t>& inputShape2, std::initializer_list<int64_t>& inputShape3,
                                       std::initializer_list<int64_t>& outputShape, ge::DataType inputDtype, ge::DataType targetDtype, std::string& reduction, int64_t ignore, float labelSmooth, std::string& expectStr)
{
    // init platform
    fe::PlatFormInfos platFormInfo;
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    map<string, string> socVersion = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    InitPlatForm(platFormInfo, socInfos, aicoreSpec, intrinsics, socVersion);

    std::string opType("CrossEntropyLossGrad");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);

    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling_parse;

    // tiling_parse_func
    struct CrossEntropyLossGradCompileInfo {
    };
    CrossEntropyLossGradCompileInfo compileInfo;
    string compileInfoStr = R"({
    "device_id": null})";
    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compileInfoStr.c_str()), reinterpret_cast<void*>(&platFormInfo)})
            .Outputs({&compileInfo})
            .Build();
    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", socVersion);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);


    gert::StorageShape gradLossShape = {inputShape1, inputShape1};
    gert::StorageShape logProbShape = {inputShape2, inputShape2};
    gert::StorageShape targetShape = {inputShape3, inputShape3};
    gert::StorageShape outShape = {outputShape, outputShape};
    auto holder = gert::TilingContextFaker()
                      .SetOpType(opType)
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&gradLossShape, &logProbShape, &targetShape})
                      .OutputShapes({&outShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platFormInfo))
                      .NodeInputTd(0, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, targetDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"reduction", Ops::NN::AnyValue::CreateFrom<std::string>(reduction)},
                                  {"ignore_index", Ops::NN::AnyValue::CreateFrom<int64_t>(ignore)},
                                  {"label_smoothing", Ops::NN::AnyValue::CreateFrom<float>(labelSmooth)},
                                  {"lse_square_scale_for_zloss", Ops::NN::AnyValue::CreateFrom<float>(0.0)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", socVersion);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);

    auto tiling_data_result = TilingData2Str(tiling_context->GetRawTilingData());
    EXPECT_EQ(tiling_data_result, expectStr);
}

static void DoCELossGradRegbaseHasWeightTilingCase(std::initializer_list<int64_t>& inputShape1, std::initializer_list<int64_t>& inputShape2, std::initializer_list<int64_t>& inputShape3,
                                                std::initializer_list<int64_t>& inputShape4, std::initializer_list<int64_t>& outputShape, ge::DataType inputDtype, ge::DataType targetDtype,
                                                std::string& reduction, int64_t ignore, float labelSmooth, std::string& expectStr)
{
    // init platform
    fe::PlatFormInfos platFormInfo;
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    map<string, string> socVersion = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    InitPlatForm(platFormInfo, socInfos, aicoreSpec, intrinsics, socVersion);

    std::string opType("CrossEntropyLossGrad");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);

    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling_parse;

    // tiling_parse_func
    struct CrossEntropyLossGradCompileInfo {
    };
    CrossEntropyLossGradCompileInfo compileInfo;
    string compileInfoStr = R"({
    "device_id": null})";
    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compileInfoStr.c_str()), reinterpret_cast<void*>(&platFormInfo)})
            .Outputs({&compileInfo})
            .Build();
    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", socVersion);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);


    gert::StorageShape gradLossShape = {inputShape1, inputShape1};
    gert::StorageShape logProbShape = {inputShape2, inputShape2};
    gert::StorageShape targetShape = {inputShape3, inputShape3};
    gert::StorageShape weightShape = {inputShape4, inputShape4};
    gert::StorageShape outShape = {outputShape, outputShape};
    auto holder = gert::TilingContextFaker()
                      .SetOpType(opType)
                      .NodeIoNum(4, 1)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&gradLossShape, &logProbShape, &targetShape, &weightShape})
                      .OutputShapes({&outShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platFormInfo))
                      .NodeInputTd(0, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, targetDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"reduction", Ops::NN::AnyValue::CreateFrom<std::string>(reduction)},
                                  {"ignore_index", Ops::NN::AnyValue::CreateFrom<int64_t>(ignore)},
                                  {"label_smoothing", Ops::NN::AnyValue::CreateFrom<float>(labelSmooth)},
                                  {"lse_square_scale_for_zloss", Ops::NN::AnyValue::CreateFrom<float>(0.0)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", socVersion);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);

    auto tiling_data_result = TilingData2Str(tiling_context->GetRawTilingData());
    EXPECT_EQ(tiling_data_result, expectStr);
}

TEST_F(CrossEntropyLossGradRegbaseTiling, cross_entropy_loss_grad_fp32_int32_mean)
{
    // FLOAT
    std::initializer_list<int64_t> inputShape1 = {1};
    std::initializer_list<int64_t> inputShape2 = {2048, 16384};
    std::initializer_list<int64_t> inputShape3 = {2048};
    std::initializer_list<int64_t> outputShape = {2048, 16384};
    std::string reduction = "mean";
    int64_t ignoreIdx = -100;
    float labelSmoothing = 0.0;
    std::string expectStr =
        "2048 16384 1 -100 0 64 64 0 32 32 16384 1 0 16384 128 128 128 0 128 0 32 256 0 0 0 0 0 0 ";
    DoCELossGradRegbaseNoWeightTilingCase(inputShape1, inputShape2, inputShape3, outputShape, ge::DT_FLOAT /*inputdtype*/, ge::DT_INT32 /*targetdtype*/, reduction, ignoreIdx, labelSmoothing, expectStr);
}


TEST_F(CrossEntropyLossGradRegbaseTiling, cross_entropy_loss_grad_fp32_int32_sum)
{
    // FLOAT
    std::initializer_list<int64_t> inputShape1 = {1};
    std::initializer_list<int64_t> inputShape2 = {2048, 16384};
    std::initializer_list<int64_t> inputShape3 = {2048};
    std::initializer_list<int64_t> outputShape = {2048, 16384};
    std::string reduction = "sum";
    int64_t ignoreIdx = -100;
    float labelSmoothing = 0.1;
    std::string expectStr =
        "2048 16384 2 -100 1036831949 64 64 0 32 32 16384 1 0 16384 128 128 128 0 128 0 0 0 0 0 0 0 0 0 ";
    DoCELossGradRegbaseNoWeightTilingCase(inputShape1, inputShape2, inputShape3, outputShape, ge::DT_FLOAT /*inputdtype*/, ge::DT_INT32 /*targetdtype*/, reduction, ignoreIdx, labelSmoothing, expectStr);
}

TEST_F(CrossEntropyLossGradRegbaseTiling, cross_entropy_loss_grad_fp32_int32_none)
{
    // FLOAT
    std::initializer_list<int64_t> inputShape1 = {2048};
    std::initializer_list<int64_t> inputShape2 = {2048, 16384};
    std::initializer_list<int64_t> inputShape3 = {2048};
    std::initializer_list<int64_t> outputShape = {2048, 16384};
    std::string reduction = "none";
    int64_t ignoreIdx = -100;
    float labelSmoothing = 0.0;
    std::string expectStr =
        "2048 16384 0 -100 0 64 64 0 32 32 16384 1 0 16384 128 128 128 128 128 0 0 0 0 0 0 0 0 0 ";
    DoCELossGradRegbaseNoWeightTilingCase(inputShape1, inputShape2, inputShape3, outputShape, ge::DT_FLOAT /*inputdtype*/, ge::DT_INT32 /*targetdtype*/, reduction, ignoreIdx, labelSmoothing, expectStr);
}

TEST_F(CrossEntropyLossGradRegbaseTiling, cross_entropy_loss_grad_fp16_int32_mean)
{
    // FLOAT
    std::initializer_list<int64_t> inputShape1 = {1};
    std::initializer_list<int64_t> inputShape2 = {2048, 16384};
    std::initializer_list<int64_t> inputShape3 = {2048};
    std::initializer_list<int64_t> outputShape = {2048, 16384};
    std::string reduction = "mean";
    int64_t ignoreIdx = -100;
    float labelSmoothing = 0.2;
    std::string expectStr =
        "2048 16384 1 -100 1045220557 64 64 0 32 32 16384 0 0 16384 64 64 64 0 64 0 32 256 0 0 0 0 0 0 ";
    DoCELossGradRegbaseNoWeightTilingCase(inputShape1, inputShape2, inputShape3, outputShape, ge::DT_FLOAT16 /*inputdtype*/, ge::DT_INT32 /*targetdtype*/, reduction, ignoreIdx, labelSmoothing, expectStr);
}


TEST_F(CrossEntropyLossGradRegbaseTiling, cross_entropy_loss_grad_fp16_int32_sum)
{
    // FLOAT
    std::initializer_list<int64_t> inputShape1 = {1};
    std::initializer_list<int64_t> inputShape2 = {2048, 16384};
    std::initializer_list<int64_t> inputShape3 = {2048};
    std::initializer_list<int64_t> outputShape = {2048, 16384};
    std::string reduction = "sum";
    int64_t ignoreIdx = -100;
    float labelSmoothing = 0.0;
    std::string expectStr =
        "2048 16384 2 -100 0 64 64 0 32 32 16384 0 0 16384 64 64 0 0 64 0 0 0 0 0 0 0 0 0 ";
    DoCELossGradRegbaseNoWeightTilingCase(inputShape1, inputShape2, inputShape3, outputShape, ge::DT_FLOAT16 /*inputdtype*/, ge::DT_INT32 /*targetdtype*/, reduction, ignoreIdx, labelSmoothing, expectStr);
}

TEST_F(CrossEntropyLossGradRegbaseTiling, cross_entropy_loss_grad_fp16_int32_none)
{
    // FLOAT
    std::initializer_list<int64_t> inputShape1 = {2048};
    std::initializer_list<int64_t> inputShape2 = {2048, 16384};
    std::initializer_list<int64_t> inputShape3 = {2048};
    std::initializer_list<int64_t> outputShape = {2048, 16384};
    std::string reduction = "none";
    int64_t ignoreIdx = -100;
    float labelSmoothing = 0.0;
    std::string expectStr =
        "2048 16384 0 -100 0 64 64 0 32 32 16384 0 0 16384 64 64 0 32 64 0 0 0 0 0 0 0 0 0 ";
    DoCELossGradRegbaseNoWeightTilingCase(inputShape1, inputShape2, inputShape3, outputShape, ge::DT_FLOAT16 /*inputdtype*/, ge::DT_INT32 /*targetdtype*/, reduction, ignoreIdx, labelSmoothing, expectStr);
}

TEST_F(CrossEntropyLossGradRegbaseTiling, cross_entropy_loss_grad_bf16_int32_mean)
{
    // FLOAT
    std::initializer_list<int64_t> inputShape1 = {1};
    std::initializer_list<int64_t> inputShape2 = {2048, 16384};
    std::initializer_list<int64_t> inputShape3 = {2048};
    std::initializer_list<int64_t> outputShape = {2048, 16384};
    std::string reduction = "mean";
    int64_t ignoreIdx = -100;
    float labelSmoothing = 0.6;
    std::string expectStr =
        "2048 16384 1 -100 1058642330 64 64 0 32 32 16384 0 0 16384 64 64 64 0 64 0 32 256 0 0 0 0 0 0 ";
    DoCELossGradRegbaseNoWeightTilingCase(inputShape1, inputShape2, inputShape3, outputShape, ge::DT_BF16 /*inputdtype*/, ge::DT_INT32 /*targetdtype*/, reduction, ignoreIdx, labelSmoothing, expectStr);
}

TEST_F(CrossEntropyLossGradRegbaseTiling, cross_entropy_loss_grad_bf16_int32_mean_full)
{
    // FLOAT
    std::initializer_list<int64_t> inputShape1 = {1};
    std::initializer_list<int64_t> inputShape2 = {2048, 163};
    std::initializer_list<int64_t> inputShape3 = {2048};
    std::initializer_list<int64_t> outputShape = {2048, 163};
    std::string reduction = "mean";
    int64_t ignoreIdx = -100;
    float labelSmoothing = 0.6;
    std::string expectStr =
        "2048 163 1 -100 1058642330 64 64 0 32 32 176 0 0 30096 704 704 704 0 704 0 32 256 0 0 0 0 0 0 ";
    DoCELossGradRegbaseNoWeightTilingCase(inputShape1, inputShape2, inputShape3, outputShape, ge::DT_BF16 /*inputdtype*/, ge::DT_INT32 /*targetdtype*/, reduction, ignoreIdx, labelSmoothing, expectStr);
}


TEST_F(CrossEntropyLossGradRegbaseTiling, cross_entropy_loss_grad_bf16_int32_sum)
{
    // FLOAT
    std::initializer_list<int64_t> inputShape1 = {1};
    std::initializer_list<int64_t> inputShape2 = {2048, 16384};
    std::initializer_list<int64_t> inputShape3 = {2048};
    std::initializer_list<int64_t> outputShape = {2048, 16384};
    std::string reduction = "sum";
    int64_t ignoreIdx = -100;
    float labelSmoothing = 0.3;
    std::string expectStr =
        "2048 16384 2 -100 1050253722 64 64 0 32 32 16384 0 0 16384 64 64 64 0 64 0 0 0 0 0 0 0 0 0 ";
    DoCELossGradRegbaseNoWeightTilingCase(inputShape1, inputShape2, inputShape3, outputShape, ge::DT_BF16 /*inputdtype*/, ge::DT_INT32 /*targetdtype*/, reduction, ignoreIdx, labelSmoothing, expectStr);
}

TEST_F(CrossEntropyLossGradRegbaseTiling, cross_entropy_loss_grad_bf16_int32_none)
{
    // FLOAT
    std::initializer_list<int64_t> inputShape1 = {2048};
    std::initializer_list<int64_t> inputShape2 = {2048, 1638400};
    std::initializer_list<int64_t> inputShape3 = {2048};
    std::initializer_list<int64_t> outputShape = {2048, 1638400};
    std::string reduction = "none";
    int64_t ignoreIdx = 2;
    float labelSmoothing = 0.0;
    std::string expectStr =
        "2048 1638400 0 2 0 64 64 0 32 32 20352 80 10240 20352 128 128 128 64 128 0 0 0 0 0 0 0 0 0 ";
    DoCELossGradRegbaseNoWeightTilingCase(inputShape1, inputShape2, inputShape3, outputShape, ge::DT_BF16 /*inputdtype*/, ge::DT_INT32 /*targetdtype*/, reduction, ignoreIdx, labelSmoothing, expectStr);
}

TEST_F(CrossEntropyLossGradRegbaseTiling, cross_entropy_loss_grad_bf16_int32_none_weight)
{
    // FLOAT
    std::initializer_list<int64_t> inputShape1 = {2048};
    std::initializer_list<int64_t> inputShape2 = {2048, 1638400};
    std::initializer_list<int64_t> inputShape3 = {2048};
    std::initializer_list<int64_t> inputShape4 = {1638400};
    std::initializer_list<int64_t> outputShape = {2048, 1638400};
    std::string reduction = "none";
    int64_t ignoreIdx = 2;
    float labelSmoothing = 0.0;
    std::string expectStr =
        "2048 1638400 0 2 0 64 64 0 32 32 15232 107 8576 15232 128 128 128 64 128 128 0 0 0 0 0 0 0 0 ";
    DoCELossGradRegbaseHasWeightTilingCase(inputShape1, inputShape2, inputShape3, inputShape4, outputShape, ge::DT_BF16 /*inputdtype*/, ge::DT_INT32 /*targetdtype*/, reduction, ignoreIdx, labelSmoothing, expectStr);
}

TEST_F(CrossEntropyLossGradRegbaseTiling, cross_entropy_loss_grad_fp16_int32_sum_weight)
{
    // FLOAT
    std::initializer_list<int64_t> inputShape1 = {1};
    std::initializer_list<int64_t> inputShape2 = {2048, 1638400};
    std::initializer_list<int64_t> inputShape3 = {2048};
    std::initializer_list<int64_t> inputShape4 = {1638400};
    std::initializer_list<int64_t> outputShape = {2048, 1638400};
    std::string reduction = "sum";
    int64_t ignoreIdx = 2;
    float labelSmoothing = 0.0;
    std::string expectStr =
        "2048 1638400 2 2 0 64 64 0 32 32 15232 107 8576 15232 128 128 128 0 128 128 0 0 0 0 0 0 0 0 ";
    DoCELossGradRegbaseHasWeightTilingCase(inputShape1, inputShape2, inputShape3, inputShape4, outputShape, ge::DT_FLOAT16 /*inputdtype*/, ge::DT_INT32 /*targetdtype*/, reduction, ignoreIdx, labelSmoothing, expectStr);
}

TEST_F(CrossEntropyLossGradRegbaseTiling, cross_entropy_loss_grad_fp32_int32_mean_weight)
{
    // FLOAT
    std::initializer_list<int64_t> inputShape1 = {1};
    std::initializer_list<int64_t> inputShape2 = {2048, 1638400};
    std::initializer_list<int64_t> inputShape3 = {2048};
    std::initializer_list<int64_t> inputShape4 = {1638400};
    std::initializer_list<int64_t> outputShape = {2048, 1638400};
    std::string reduction = "mean";
    int64_t ignoreIdx = 2;
    float labelSmoothing = 0.3;
    std::string expectStr =
        "2048 1638400 1 2 1050253722 64 64 0 32 32 1664 984 1024 1664 128 128 128 0 128 128 32 256 512 1664 1024 473 1024 576 ";
    DoCELossGradRegbaseHasWeightTilingCase(inputShape1, inputShape2, inputShape3, inputShape4, outputShape, ge::DT_FLOAT /*inputdtype*/, ge::DT_INT32 /*targetdtype*/, reduction, ignoreIdx, labelSmoothing, expectStr);
}

TEST_F(CrossEntropyLossGradRegbaseTiling, cross_entropy_loss_grad_empty)
{
    // FLOAT
    std::initializer_list<int64_t> inputShape1 = {1};
    std::initializer_list<int64_t> inputShape2 = {2048, 0};
    std::initializer_list<int64_t> inputShape3 = {2048};
    std::initializer_list<int64_t> inputShape4 = {0};
    std::initializer_list<int64_t> outputShape = {2048, 0};
    std::string reduction = "mean";
    int64_t ignoreIdx = 2;
    float labelSmoothing = 0.0;
    std::string expectStr =
        "0 0 1 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    DoCELossGradRegbaseHasWeightTilingCase(inputShape1, inputShape2, inputShape3, inputShape4, outputShape, ge::DT_FLOAT /*inputdtype*/, ge::DT_INT32 /*targetdtype*/, reduction, ignoreIdx, labelSmoothing, expectStr);
}
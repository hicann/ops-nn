/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file test_cross_entropy_loss_regbase_tiling.cpp
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

 #include "../../../op_host/cross_entropy_loss_tiling.h"

using namespace ut_util;
using namespace std;
using namespace ge;

class CrossEntropyLossRegBaseTiling : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "CrossEntropyLossRegBaseTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "CrossEntropyLossRegBaseTiling TearDown" << std::endl;
    }
};

static void InitPlatForm(
    fe::PlatFormInfos& platFormInfo, map<string, string>& socInfos, map<string, string>& aicoreSpec,
    map<string, string>& intrinsics)
{
    string hardwareInfo = R"({
        "hardware_info": {"UB_SIZE": 253952, "CORE_NUM": 64}
                          })";
    GetPlatFormInfos(hardwareInfo.c_str(), socInfos, aicoreSpec, intrinsics);

    platFormInfo.Init();
}

static string to_string(const std::stringstream& tiling_data)
{
    auto data = tiling_data.str();
    string result;
    int32_t tmp = 0;

    for (size_t i = 0; i < data.length(); i += sizeof(int32_t)) {
        memcpy(&tmp, data.c_str() + i, sizeof(tmp));
        result += std::to_string(tmp);
        result += " ";
    }

    return result;
}

static void DoCrossEntropyLossTilingCase(
    std::initializer_list<int64_t>& inputShape, std::initializer_list<int64_t>& targetShape,
    std::initializer_list<int64_t>& weightShape, std::initializer_list<int64_t>& lossShape, ge::DataType inputDtype,
    ge::DataType tDtype, std::string reduction, int ignore_index, float label_smoothing)
{
    // init platform
    fe::PlatFormInfos platFormInfo;
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend910_95"}};
    InitPlatForm(platFormInfo, socInfos, aicoreSpec, intrinsics);

    // compile info
    struct CrossEntropyLossCompileInfo {
    };
    CrossEntropyLossCompileInfo compileInfo;

    std::string opType("CrossEntropyLoss");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling_parse;

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
    kernel_holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version_infos);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    gert::StorageShape xShape = {inputShape, inputShape};
    gert::StorageShape tShape = {targetShape, targetShape};
    gert::StorageShape wShape = {weightShape, weightShape};
    gert::StorageShape oShape = {lossShape, lossShape};
    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(3, 4)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&xShape, &tShape, &wShape})
                      .OutputShapes({&oShape, &xShape, &oShape, &tShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platFormInfo))
                      .NodeInputTd(0, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, tDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"reduction", Ops::NN::AnyValue::CreateFrom<std::string>(reduction)},
                           {"ignore_index", Ops::NN::AnyValue::CreateFrom<int64_t>(ignore_index)},
                           {"label_smoothing", Ops::NN::AnyValue::CreateFrom<float>(label_smoothing)},
                           {"lse_square_scale_for_zloss", Ops::NN::AnyValue::CreateFrom<float>(0.0f)},
                           {"return_zloss", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context, nullptr);
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    auto block_dim = tiling_context->GetBlockDim();
    std::cout << tiling_key << std::endl;
    std::cout << block_dim << std::endl;
    auto raw_tiling_data = tiling_context->GetRawTilingData();
    auto tiling_data_result = to_string<int64_t>(raw_tiling_data->GetData(), raw_tiling_data->GetDataSize());
    std::cout << tiling_data_result << std::endl;
}

TEST_F(CrossEntropyLossRegBaseTiling, test_tiling_fp32_none)
{
    std::initializer_list<int64_t> inputShape = {49, 112595};
    std::initializer_list<int64_t> targetShape = {49};
    std::initializer_list<int64_t> weightShape = {112595};
    std::initializer_list<int64_t> lossShape = {49};
    ge::DataType inputDtype = ge::DT_FLOAT;
    ge::DataType tDtype = ge::DT_INT32;
    string reduction = "none";
    int ignore_index = -100;
    float label_smoothing = 0.0f;
    DoCrossEntropyLossTilingCase(
        inputShape, targetShape, weightShape, lossShape, inputDtype, tDtype, reduction, ignore_index, label_smoothing);
}

TEST_F(CrossEntropyLossRegBaseTiling, test_tiling_fp32_mean)
{
    std::initializer_list<int64_t> inputShape = {49, 112595};
    std::initializer_list<int64_t> targetShape = {49};
    std::initializer_list<int64_t> weightShape = {112595};
    std::initializer_list<int64_t> lossShape = {1};
    ge::DataType inputDtype = ge::DT_FLOAT;
    ge::DataType tDtype = ge::DT_INT32;
    string reduction = "mean";
    int ignore_index = -100;
    float label_smoothing = 0.0f;
    DoCrossEntropyLossTilingCase(
        inputShape, targetShape, weightShape, lossShape, inputDtype, tDtype, reduction, ignore_index, label_smoothing);
}

TEST_F(CrossEntropyLossRegBaseTiling, test_tiling_fp32_sum)
{
    std::initializer_list<int64_t> inputShape = {49, 112595};
    std::initializer_list<int64_t> targetShape = {49};
    std::initializer_list<int64_t> weightShape = {112595};
    std::initializer_list<int64_t> lossShape = {1};
    ge::DataType inputDtype = ge::DT_FLOAT;
    ge::DataType tDtype = ge::DT_INT32;
    string reduction = "sum";
    int ignore_index = -100;
    float label_smoothing = 0.0f;
    DoCrossEntropyLossTilingCase(
        inputShape, targetShape, weightShape, lossShape, inputDtype, tDtype, reduction, ignore_index, label_smoothing);
}

TEST_F(CrossEntropyLossRegBaseTiling, test_tiling_fp16_none)
{
    std::initializer_list<int64_t> inputShape = {4096, 4096};
    std::initializer_list<int64_t> targetShape = {4096};
    std::initializer_list<int64_t> weightShape = {4096};
    std::initializer_list<int64_t> lossShape = {4096};
    ge::DataType inputDtype = ge::DT_FLOAT16;
    ge::DataType tDtype = ge::DT_INT32;
    string reduction = "none";
    int ignore_index = -100;
    float label_smoothing = 0.0f;
    DoCrossEntropyLossTilingCase(
        inputShape, targetShape, weightShape, lossShape, inputDtype, tDtype, reduction, ignore_index, label_smoothing);
}

TEST_F(CrossEntropyLossRegBaseTiling, test_tiling_fp16_mean)
{
    std::initializer_list<int64_t> inputShape = {4096, 4096};
    std::initializer_list<int64_t> targetShape = {4096};
    std::initializer_list<int64_t> weightShape = {4096};
    std::initializer_list<int64_t> lossShape = {1};
    ge::DataType inputDtype = ge::DT_FLOAT16;
    ge::DataType tDtype = ge::DT_INT32;
    string reduction = "mean";
    int ignore_index = -100;
    float label_smoothing = 0.0f;
    DoCrossEntropyLossTilingCase(
        inputShape, targetShape, weightShape, lossShape, inputDtype, tDtype, reduction, ignore_index, label_smoothing);
}

TEST_F(CrossEntropyLossRegBaseTiling, test_tiling_fp16_sumn)
{
    std::initializer_list<int64_t> inputShape = {4096, 4096};
    std::initializer_list<int64_t> targetShape = {4096};
    std::initializer_list<int64_t> weightShape = {4096};
    std::initializer_list<int64_t> lossShape = {1};
    ge::DataType inputDtype = ge::DT_FLOAT16;
    ge::DataType tDtype = ge::DT_INT32;
    string reduction = "sum";
    int ignore_index = -100;
    float label_smoothing = 0.0f;
    DoCrossEntropyLossTilingCase(
        inputShape, targetShape, weightShape, lossShape, inputDtype, tDtype, reduction, ignore_index, label_smoothing);
}

TEST_F(CrossEntropyLossRegBaseTiling, test_tiling_bfp16_none)
{
    std::initializer_list<int64_t> inputShape = {4096, 4096};
    std::initializer_list<int64_t> targetShape = {4096};
    std::initializer_list<int64_t> weightShape = {4096};
    std::initializer_list<int64_t> lossShape = {4096};
    ge::DataType inputDtype = ge::DT_BF16;
    ge::DataType tDtype = ge::DT_INT32;
    string reduction = "none";
    int ignore_index = -100;
    float label_smoothing = 0.0f;
    DoCrossEntropyLossTilingCase(
        inputShape, targetShape, weightShape, lossShape, inputDtype, tDtype, reduction, ignore_index, label_smoothing);
}

TEST_F(CrossEntropyLossRegBaseTiling, test_tiling_bfp16_mean)
{
    std::initializer_list<int64_t> inputShape = {4096, 4096};
    std::initializer_list<int64_t> targetShape = {4096};
    std::initializer_list<int64_t> weightShape = {4096};
    std::initializer_list<int64_t> lossShape = {1};
    ge::DataType inputDtype = ge::DT_BF16;
    ge::DataType tDtype = ge::DT_INT32;
    string reduction = "mean";
    int ignore_index = -100;
    float label_smoothing = 0.0f;
    DoCrossEntropyLossTilingCase(
        inputShape, targetShape, weightShape, lossShape, inputDtype, tDtype, reduction, ignore_index, label_smoothing);
}

TEST_F(CrossEntropyLossRegBaseTiling, test_tiling_bfp16_sumn)
{
    std::initializer_list<int64_t> inputShape = {4096, 4096};
    std::initializer_list<int64_t> targetShape = {4096};
    std::initializer_list<int64_t> weightShape = {4096};
    std::initializer_list<int64_t> lossShape = {1};
    ge::DataType inputDtype = ge::DT_BF16;
    ge::DataType tDtype = ge::DT_INT32;
    string reduction = "sum";
    int ignore_index = -100;
    float label_smoothing = 0.0f;
    DoCrossEntropyLossTilingCase(
        inputShape, targetShape, weightShape, lossShape, inputDtype, tDtype, reduction, ignore_index, label_smoothing);
}

TEST_F(CrossEntropyLossRegBaseTiling, test_tiling_bfp16_int32_none)
{
    std::initializer_list<int64_t> inputShape = {4096, 4096};
    std::initializer_list<int64_t> targetShape = {4096};
    std::initializer_list<int64_t> weightShape = {4096};
    std::initializer_list<int64_t> lossShape = {4096};
    ge::DataType inputDtype = ge::DT_BF16;
    ge::DataType tDtype = ge::DT_INT32;
    string reduction = "none";
    int ignore_index = 1;
    float label_smoothing = 1.0f;
    DoCrossEntropyLossTilingCase(
        inputShape, targetShape, weightShape, lossShape, inputDtype, tDtype, reduction, ignore_index, label_smoothing);
}

TEST_F(CrossEntropyLossRegBaseTiling, test_tiling_bfp16_int32_mean)
{
    std::initializer_list<int64_t> inputShape = {4096, 4096};
    std::initializer_list<int64_t> targetShape = {4096};
    std::initializer_list<int64_t> weightShape = {4096};
    std::initializer_list<int64_t> lossShape = {1};
    ge::DataType inputDtype = ge::DT_BF16;
    ge::DataType tDtype = ge::DT_INT32;
    string reduction = "mean";
    int ignore_index = 1;
    float label_smoothing = 1.0f;
    DoCrossEntropyLossTilingCase(
        inputShape, targetShape, weightShape, lossShape, inputDtype, tDtype, reduction, ignore_index, label_smoothing);
}

TEST_F(CrossEntropyLossRegBaseTiling, test_tiling_bfp16_int32_sumn)
{
    std::initializer_list<int64_t> inputShape = {4096, 4096};
    std::initializer_list<int64_t> targetShape = {4096};
    std::initializer_list<int64_t> weightShape = {4096};
    std::initializer_list<int64_t> lossShape = {1};
    ge::DataType inputDtype = ge::DT_BF16;
    ge::DataType tDtype = ge::DT_INT32;
    string reduction = "sum";
    int ignore_index = 1;
    float label_smoothing = 1.0f;
    DoCrossEntropyLossTilingCase(
        inputShape, targetShape, weightShape, lossShape, inputDtype, tDtype, reduction, ignore_index, label_smoothing);
}

TEST_F(CrossEntropyLossRegBaseTiling, test_tiling_bfp16_int32_none_full_load)
{
    std::initializer_list<int64_t> inputShape = {4096, 1035};
    std::initializer_list<int64_t> targetShape = {4096};
    std::initializer_list<int64_t> weightShape = {1035};
    std::initializer_list<int64_t> lossShape = {4096};
    ge::DataType inputDtype = ge::DT_BF16;
    ge::DataType tDtype = ge::DT_INT32;
    string reduction = "none";
    int ignore_index = 1;
    float label_smoothing = 1.0f;
    DoCrossEntropyLossTilingCase(
        inputShape, targetShape, weightShape, lossShape, inputDtype, tDtype, reduction, ignore_index, label_smoothing);
}

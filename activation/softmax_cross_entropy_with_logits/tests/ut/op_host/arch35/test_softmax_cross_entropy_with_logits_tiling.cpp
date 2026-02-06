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
 * \file test_softmax_cross_entropy_with_logits_regbase_split_r_tiling.cpp
 * \brief
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

using namespace ut_util;
using namespace std;
using namespace ge;

class SoftmaxCrossEntropyWithLogitsRegBaseSplitRTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "SoftmaxCrossEntropyWithLogitsRegBaseSplitRTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "SoftmaxCrossEntropyWithLogitsRegBaseSplitRTiling TearDown" << std::endl;
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
    map<string, string>& intrinsics)
{
    string hardwareInfo = R"({
        "hardware_info": {"UB_SIZE": 253952, "CORE_NUM": 64}
                          })";
    GetPlatFormInfos(hardwareInfo.c_str(), socInfos, aicoreSpec, intrinsics);

    platFormInfo.Init();
}

static void DoSCEWithLogtisSplitRTilingCase(
    std::initializer_list<int64_t>& featuresShape, std::initializer_list<int64_t>& labelsShape,
    std::initializer_list<int64_t>& lossShape, std::initializer_list<int64_t>& backPropShape, ge::DataType inputDtype,
    std::string& expectStr)
{
    // init platform
    fe::PlatFormInfos platFormInfo;
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    InitPlatForm(platFormInfo, socInfos, aicoreSpec, intrinsics);

    // compile info
    struct SCEWithLogitsCompileInfo {
        bool isAscendC{false};
    };
    SCEWithLogitsCompileInfo compileInfo;
    compileInfo.isAscendC = true;

    std::string opType("SoftmaxCrossEntropyWithLogits");
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

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    gert::StorageShape fShape = {featuresShape, featuresShape};
    gert::StorageShape lShape = {labelsShape, labelsShape};
    gert::StorageShape out1Shape = {lossShape, lossShape};
    gert::StorageShape out2Shape = {backPropShape, backPropShape};
    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(2, 2)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&fShape, &lShape})
                      .OutputShapes({&out1Shape, &out2Shape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platFormInfo))
                      .NodeInputTd(0, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
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
    auto raw_tiling_data = tiling_context->GetRawTilingData();
    auto tiling_data_result = TilingData2Str(raw_tiling_data);

    EXPECT_EQ(tiling_data_result, expectStr);
}

TEST_F(SoftmaxCrossEntropyWithLogitsRegBaseSplitRTiling, test_tiling_fp32_elewise)
{
    std::initializer_list<int64_t> featuresShape = {49, 112595};
    std::initializer_list<int64_t> labelsShape = {49, 112595};
    std::initializer_list<int64_t> lossShape = {49};
    std::initializer_list<int64_t> backPropShape = {49, 112595};
    ge::DataType inputDtype = ge::DT_FLOAT;
    std::string expectStr = "49 49 112595 1 1 1472 8 0 0 1 1 77 723 728 13 64 0 0 ";

    DoSCEWithLogtisSplitRTilingCase(featuresShape, labelsShape, lossShape, backPropShape, inputDtype, expectStr);
}

TEST_F(SoftmaxCrossEntropyWithLogitsRegBaseSplitRTiling, test_tiling_fp16_elewise)
{
    std::initializer_list<int64_t> featuresShape = {49, 112595};
    std::initializer_list<int64_t> labelsShape = {49, 112595};
    std::initializer_list<int64_t> lossShape = {49};
    std::initializer_list<int64_t> backPropShape = {49, 112595};
    ge::DataType inputDtype = ge::DT_FLOAT16;
    std::string expectStr = "49 49 112595 1 1 1152 16 0 0 1 1 98 851 864 34 64 0 0 ";

    DoSCEWithLogtisSplitRTilingCase(featuresShape, labelsShape, lossShape, backPropShape, inputDtype, expectStr);
}

TEST_F(SoftmaxCrossEntropyWithLogitsRegBaseSplitRTiling, test_tiling_bf16_elewise)
{
    std::initializer_list<int64_t> featuresShape = {49, 112595};
    std::initializer_list<int64_t> labelsShape = {49, 112595};
    std::initializer_list<int64_t> lossShape = {49};
    std::initializer_list<int64_t> backPropShape = {49, 112595};
    ge::DataType inputDtype = ge::DT_BF16;
    std::string expectStr = "49 49 112595 1 1 1152 16 0 0 1 1 98 851 864 34 64 0 0 ";

    DoSCEWithLogtisSplitRTilingCase(featuresShape, labelsShape, lossShape, backPropShape, inputDtype, expectStr);
}

TEST_F(SoftmaxCrossEntropyWithLogitsRegBaseSplitRTiling, test_tiling_fp32_brc)
{
    std::initializer_list<int64_t> featuresShape = {49, 1};
    std::initializer_list<int64_t> labelsShape = {49, 112595};
    std::initializer_list<int64_t> lossShape = {49};
    std::initializer_list<int64_t> backPropShape = {49, 112595};
    ge::DataType inputDtype = ge::DT_FLOAT;
    std::string expectStr = "49 49 112595 1 1 1472 8 0 0 1 1 77 723 728 13 64 1 0 ";

    DoSCEWithLogtisSplitRTilingCase(featuresShape, labelsShape, lossShape, backPropShape, inputDtype, expectStr);
}

TEST_F(SoftmaxCrossEntropyWithLogitsRegBaseSplitRTiling, test_tiling_fp16_brc)
{
    std::initializer_list<int64_t> featuresShape = {49, 112595};
    std::initializer_list<int64_t> labelsShape = {1, 112595};
    std::initializer_list<int64_t> lossShape = {49};
    std::initializer_list<int64_t> backPropShape = {49, 112595};
    ge::DataType inputDtype = ge::DT_FLOAT16;
    std::string expectStr = "49 49 112595 1 1 1152 16 0 0 1 1 98 851 864 34 64 0 0 ";

    DoSCEWithLogtisSplitRTilingCase(featuresShape, labelsShape, lossShape, backPropShape, inputDtype, expectStr);
}

TEST_F(SoftmaxCrossEntropyWithLogitsRegBaseSplitRTiling, test_tiling_bf16_brc)
{
    std::initializer_list<int64_t> featuresShape = {49, 112595};
    std::initializer_list<int64_t> labelsShape = {1};
    std::initializer_list<int64_t> lossShape = {49};
    std::initializer_list<int64_t> backPropShape = {49, 112595};
    ge::DataType inputDtype = ge::DT_FLOAT16;
    std::string expectStr = "49 49 112595 1 1 1152 16 0 0 1 1 98 851 864 34 64 0 0 ";

    DoSCEWithLogtisSplitRTilingCase(featuresShape, labelsShape, lossShape, backPropShape, inputDtype, expectStr);
}

TEST_F(SoftmaxCrossEntropyWithLogitsRegBaseSplitRTiling, test_tiling_NHWC)
{
    std::initializer_list<int64_t> featuresShape = {2, 3, 4, 5};
    std::initializer_list<int64_t> labelsShape = {2, 3, 4, 5};
    std::initializer_list<int64_t> lossShape = {24};
    std::initializer_list<int64_t> backPropShape = {2, 3, 4, 5};
    ge::DataType inputDtype = ge::DT_FLOAT;
    std::string expectStr = "24 24 5 1 1 8 8 0 0 1 1 0 0 0 0 0 0 0 ";

    DoSCEWithLogtisSplitRTilingCase(featuresShape, labelsShape, lossShape, backPropShape, inputDtype, expectStr);
}

TEST_F(SoftmaxCrossEntropyWithLogitsRegBaseSplitRTiling, test_tiling_NHWC_brc)
{
    std::initializer_list<int64_t> featuresShape = {2, 3, 4, 5};
    std::initializer_list<int64_t> labelsShape = {2, 3, 4, 1};
    std::initializer_list<int64_t> lossShape = {24};
    std::initializer_list<int64_t> backPropShape = {2, 3, 4, 5};
    ge::DataType inputDtype = ge::DT_FLOAT;
    std::string expectStr = "24 24 5 1 1 8 8 0 0 1 1 0 0 0 0 0 0 1 ";

    DoSCEWithLogtisSplitRTilingCase(featuresShape, labelsShape, lossShape, backPropShape, inputDtype, expectStr);
}
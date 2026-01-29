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
 * \file test_l2_loss_tiling.cpp
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
#include "atvoss/elewise/elewise_tiling.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include "../../../../op_host/arch35/l2_loss_tiling.h"

using namespace std;
using namespace ge;
using namespace ut_util;

class L2LossDavidTiling : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "L2LossDavidTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "L2LossDavidTiling TearDown" << std::endl;
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
                         map<string, string>& aicoreSpec, map<string, string>& intrinsics)
{
    string hardwareInfo = R"({
        "hardware_info": {"UB_SIZE": 253952, "CORE_NUM": 64}
                          })";
    GetPlatFormInfos(hardwareInfo.c_str(), socInfos, aicoreSpec, intrinsics);

    platFormInfo.Init();
}

static void DoL2LossTilingCase(std::initializer_list<int64_t>& inputShape, std::initializer_list<int64_t>& outputShape, 
                               ge::DataType inputDtype, std::string& expectStr)
{
    // init platform
    fe::PlatFormInfos platFormInfo;
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    std::map<std::string, std::string> soc_version_infos = { { "Short_SoC_version", "Ascend950" } };
    InitPlatForm(platFormInfo, socInfos, aicoreSpec, intrinsics);

    Ops::Base::ReduceOpCompileInfo compileInfo;
    std::string opType("L2Loss");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);

    string compileInfoStr = R"({"device_id": null})";

    auto kernelHolder = gert::KernelRunContextFaker()
                            .KernelIONum(2, 1)
                            .Inputs({const_cast<char*>(compileInfoStr.c_str()), reinterpret_cast<void*>(&platFormInfo)})
                            .Outputs({&compileInfo})
                            .Build();

    ASSERT_TRUE(kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                           intrinsics);
    auto tilingParseFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling_parse;
    ASSERT_EQ(tilingParseFunc(kernelHolder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    gert::StorageShape xShape = {inputShape, inputShape};
    gert::StorageShape yShape = {outputShape, outputShape};
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;

    // tilingFunc simulate
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(16 * 4096);
    auto wsSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1})
                      .InputShapes({&xShape})
                      .OutputShapes({&yShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platFormInfo))
                      .NodeInputTd(0, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(wsSize)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tilingFunc(tiling_context), ge::GRAPH_SUCCESS);

    auto tiling_data_result = TilingData2Str(tiling_context->GetRawTilingData());
}

TEST_F(L2LossDavidTiling, l2_loss_david_tiling1)
{
    // FLOAT
    std::initializer_list<int64_t> inputShape = {2048, 1, 48};
    std::initializer_list<int64_t> outputShape = {};

    std::string expectStr =
        "1 1 1 1 55 1792 55 1 44032 3975177272524013632 1 98304 0 0 0 0 0 0 0 98304 1 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 ";
    DoL2LossTilingCase(inputShape, outputShape, ge::DT_FLOAT /*inputdtype*/, expectStr);
}

TEST_F(L2LossDavidTiling, l2_loss_david_tiling2)
{
    // HALF
    std::initializer_list<int64_t> inputShape = {2048, 1, 48};
    std::initializer_list<int64_t> outputShape = {};

    std::string expectStr =
        "1 1 1 1 55 1792 55 1 26624 3975177272524013632 1 98304 0 0 0 0 0 0 0 98304 1 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 ";
    DoL2LossTilingCase(inputShape, outputShape, ge::DT_FLOAT16 /*inputdtype*/, expectStr);
}

TEST_F(L2LossDavidTiling, l2_loss_david_tiling3)
{
    // BF16
    std::initializer_list<int64_t> inputShape = {2048, 1, 48};
    std::initializer_list<int64_t> outputShape = {};

    std::string expectStr =
        "1 1 1 1 55 1792 55 1 26624 3975177272524013632 1 98304 0 0 0 0 0 0 0 98304 1 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 ";
    DoL2LossTilingCase(inputShape, outputShape, ge::DT_BF16 /*inputdtype*/, expectStr);
}

TEST_F(L2LossDavidTiling, l2_loss_david_tiling4)
{
    // 2D
    std::initializer_list<int64_t> inputShape = {32, 48};
    std::initializer_list<int64_t> outputShape = {};

    std::string expectStr =
        "1 1 1 1 1 1 1 1 44032 4191350054637797440 1 1536 0 0 0 0 0 0 0 1536 1 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 ";
    DoL2LossTilingCase(inputShape, outputShape, ge::DT_FLOAT /*inputdtype*/, expectStr);
}

TEST_F(L2LossDavidTiling, l2_loss_david_tiling5)
{
    // 4D
    std::initializer_list<int64_t> inputShape = {32, 48, 32, 32};
    std::initializer_list<int64_t> outputShape = {};

    std::string expectStr =
        "1 1 1 3 164 9600 55 1 44032 3831062084448157760 1 1572864 0 0 0 0 0 0 0 1572864 1 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 ";
    DoL2LossTilingCase(inputShape, outputShape, ge::DT_FLOAT /*inputdtype*/, expectStr);
}

TEST_F(L2LossDavidTiling, l2_loss_david_tiling6)
{
    // 5D
    std::initializer_list<int64_t> inputShape = {32, 48, 32, 32, 1};
    std::initializer_list<int64_t> outputShape = {};

    std::string expectStr =
        "1 1 1 3 164 9600 55 1 44032 3831062084448157760 1 1572864 0 0 0 0 0 0 0 1572864 1 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 ";
    DoL2LossTilingCase(inputShape, outputShape, ge::DT_FLOAT /*inputdtype*/, expectStr);
}

TEST_F(L2LossDavidTiling, l2_loss_david_tiling7)
{
    // A
    std::initializer_list<int64_t> inputShape = {88,};
    std::initializer_list<int64_t> outputShape = {};

    std::string expectStr =
        "1 1 1 1 1 1 1 1 44032 4339832369755193408 1 88 0 0 0 0 0 0 0 88 1 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 ";
    DoL2LossTilingCase(inputShape, outputShape, ge::DT_FLOAT /*inputdtype*/, expectStr);
}


TEST_F(L2LossDavidTiling, l2_loss_david_tiling8)
{
    // EMPTY
    std::initializer_list<int64_t> inputShape = {};
    std::initializer_list<int64_t> outputShape = {};

    std::string expectStr =
        "1 1 1 1 1 1 1 1 33792 4575657221408424000 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 ";
    DoL2LossTilingCase(inputShape, outputShape, ge::DT_FLOAT /*inputdtype*/, expectStr);
}
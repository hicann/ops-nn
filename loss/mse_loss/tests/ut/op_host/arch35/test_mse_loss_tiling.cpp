/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <gtest/gtest.h>
#include "log/log.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "platform/platform_infos_def.h"
#include "ut_op_util.h"
#include "loss/mse_loss/op_host/arch35/mse_loss_tiling.h"

using namespace std;
using namespace ge;
using namespace ut_util;

class MseLossDavidTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MseLossDavidTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MseLossDavidTiling TearDown" << std::endl;
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

static void DoMseLossTilingCase(
    std::initializer_list<int64_t>& predictShape, std::initializer_list<int64_t>& lableShape,
    std::initializer_list<int64_t>& outputShape, ge::DataType inputDtype, std::string reduction, std::string& expectStr)
{
    // init platform
    fe::PlatFormInfos platFormInfo;
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}};
    InitPlatForm(platFormInfo, socInfos, aicoreSpec, intrinsics);

    optiling::MseLossCompileInfo compileInfo;
    compileInfo.coreNum = 64;
    std::string opType("MseLoss");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str()), nullptr);

    string compileInfoStr = R"({
        "device_id": null})";

    auto kernelHolder = gert::KernelRunContextFaker()
                            .KernelIONum(2, 1)
                            .Inputs({const_cast<char*>(compileInfoStr.c_str()), reinterpret_cast<void*>(&platFormInfo)})
                            .Outputs({&compileInfo})
                            .Build();

    ASSERT_TRUE(kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);
    // kernelHolder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version_infos);
    auto tilingParseFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling_parse;
    ASSERT_EQ(tilingParseFunc(kernelHolder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    gert::StorageShape xShape = {predictShape, predictShape};
    gert::StorageShape aShape = {lableShape, lableShape};
    gert::StorageShape oSahpe = {outputShape, outputShape};
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;

    // tilingFunc simulate
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(16 * 4096);
    auto wsSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({1, 1})
                      .InputShapes({&xShape, &aShape})
                      .OutputShapes({&oSahpe})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platFormInfo))
                      .NodeInputTd(0, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"reduction", Ops::NN::AnyValue::CreateFrom<std::string>(reduction)}})
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
    // reduce模板有sum/mean校验tilingdata
    // EXPECT_EQ(tiling_data_result, expectStr);
}

TEST_F(MseLossDavidTiling, mse_loss_david_tiling1)
{
    // test empty tensor
    std::initializer_list<int64_t> predictShape = {2048, 1, 48};
    std::initializer_list<int64_t> labelShape = {2048, 1, 48};
    std::initializer_list<int64_t> outputShape = {2048, 1, 48};
    std::string reduction = "none";

    std::string expectStr =
        "1 98304 1536 64 9024 1 1 1536 1536 9024 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 ";
    DoMseLossTilingCase(
        predictShape, labelShape, outputShape, ge::DT_FLOAT /*inputdtype*/, reduction /*noopWithEmptyAxes*/, expectStr);
}

TEST_F(MseLossDavidTiling, mse_loss_david_tiling2)
{
    // test empty tensor
    std::initializer_list<int64_t> predictShape = {2048, 48};
    std::initializer_list<int64_t> labelShape = {2048, 48};
    std::initializer_list<int64_t> outputShape = {1};
    std::string reduction = "sum";

    std::string expectStr =
        "0 0 0 0 0 0 0 0 0 0 1 1 1 1 55 1792 55 1 31488 3975177272524013632 1 98304 0 0 0 0 0 0 0 98304 1 0 0 0 0 0 0 "
        "0 1 1 0 0 0 0 0 0 0 ";
    DoMseLossTilingCase(
        predictShape, labelShape, outputShape, ge::DT_FLOAT /*inputdtype*/, reduction /*noopWithEmptyAxes*/, expectStr);
}

TEST_F(MseLossDavidTiling, mse_loss_david_tiling3)
{
    // test empty tensor
    std::initializer_list<int64_t> predictShape = {2048, 48};
    std::initializer_list<int64_t> labelShape = {2048, 48};
    std::initializer_list<int64_t> outputShape = {1};
    std::string reduction = "mean";

    std::string expectStr =
        "0 0 0 0 0 0 0 0 0 0 1 1 1 1 55 1792 55 1 31488 3975177272524013632 1 98304 0 0 0 0 0 0 0 98304 1 0 0 0 0 0 0 "
        "0 1 1 0 0 0 0 0 0 0 ";
    DoMseLossTilingCase(
        predictShape, labelShape, outputShape, ge::DT_FLOAT /*inputdtype*/, reduction /*noopWithEmptyAxes*/, expectStr);
}

TEST_F(MseLossDavidTiling, mse_loss_david_tiling4)
{
    // test empty tensor
    std::initializer_list<int64_t> predictShape = {4, 64};
    std::initializer_list<int64_t> labelShape = {4, 64};
    std::initializer_list<int64_t> outputShape = {1};
    std::string reduction = "mean";

    std::string expectStr =
        "0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 31488 4287426845256712256 1 256 0 0 0 0 0 0 0 256 1 0 0 0 0 0 0 0 1 1 0 0 "
        "0 0 0 0 0 ";
    DoMseLossTilingCase(
        predictShape, labelShape, outputShape, ge::DT_FLOAT /*inputdtype*/, reduction /*noopWithEmptyAxes*/, expectStr);
}

TEST_F(MseLossDavidTiling, mse_loss_david_tiling5)
{
    // test empty tensor
    std::initializer_list<int64_t> predictShape = {4, 64};
    std::initializer_list<int64_t> labelShape = {4, 64};
    std::initializer_list<int64_t> outputShape = {4, 64};
    std::string reduction = "none";

    std::string expectStr =
        "1 256 256 1 9024 1 1 256 256 9024 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    DoMseLossTilingCase(
        predictShape, labelShape, outputShape, ge::DT_FLOAT /*inputdtype*/, reduction /*noopWithEmptyAxes*/, expectStr);
}

TEST_F(MseLossDavidTiling, mse_loss_david_tiling6)
{
    // test empty tensor
    std::initializer_list<int64_t> predictShape = {4, 64};
    std::initializer_list<int64_t> labelShape = {4, 64};
    std::initializer_list<int64_t> outputShape = {1};
    std::string reduction = "sum";

    std::string expectStr =
        "0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 31488 4287426845256712256 1 256 0 0 0 0 0 0 0 256 1 0 0 0 0 0 0 0 1 1 0 0 "
        "0 0 0 0 0 ";
    DoMseLossTilingCase(
        predictShape, labelShape, outputShape, ge::DT_FLOAT /*inputdtype*/, reduction /*noopWithEmptyAxes*/, expectStr);
}

TEST_F(MseLossDavidTiling, mse_loss_david_tiling7)
{
    // test empty tensor
    std::initializer_list<int64_t> predictShape = {1, 1, 1};
    std::initializer_list<int64_t> labelShape = {1, 1, 1};
    std::initializer_list<int64_t> outputShape = {1};
    std::string reduction = "mean";

    std::string expectStr =
        "0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 26368 4575657221408424000 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 "
        "0 0 0 ";
    DoMseLossTilingCase(
        predictShape, labelShape, outputShape, ge::DT_FLOAT /*inputdtype*/, reduction /*noopWithEmptyAxes*/, expectStr);
}

TEST_F(MseLossDavidTiling, mse_loss_david_tiling8)
{
    // test empty tensor
    std::initializer_list<int64_t> predictShape = {1, 1, 1};
    std::initializer_list<int64_t> labelShape = {1, 1, 1};
    std::initializer_list<int64_t> outputShape = {1};
    std::string reduction = "sum";

    std::string expectStr =
        "0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 26368 4575657221408424000 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 "
        "0 0 0 ";
    DoMseLossTilingCase(
        predictShape, labelShape, outputShape, ge::DT_FLOAT /*inputdtype*/, reduction /*noopWithEmptyAxes*/, expectStr);
}

TEST_F(MseLossDavidTiling, mse_loss_david_tiling9)
{
    // test empty tensor
    std::initializer_list<int64_t> predictShape = {1, 1, 1};
    std::initializer_list<int64_t> labelShape = {1, 1, 1};
    std::initializer_list<int64_t> outputShape = {1, 1, 1};
    std::string reduction = "none";

    std::string expectStr =
        "1 1 8 1 9024 1 1 8 1 9024 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ";
    DoMseLossTilingCase(
        predictShape, labelShape, outputShape, ge::DT_FLOAT /*inputdtype*/, reduction /*noopWithEmptyAxes*/, expectStr);
}

TEST_F(MseLossDavidTiling, mse_loss_david_tiling10)
{
    // test empty tensor
    std::initializer_list<int64_t> predictShape = {2048, 1, 48};
    std::initializer_list<int64_t> labelShape = {2048, 1, 48};
    std::initializer_list<int64_t> outputShape = {2048, 1, 48};
    std::string reduction = "none";

    std::string expectStr =
        "1 98304 2048 48 7040 1 1 2048 2048 7040 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 ";
    DoMseLossTilingCase(
        predictShape, labelShape, outputShape, ge::DT_FLOAT16 /*inputdtype*/, reduction /*noopWithEmptyAxes*/,
        expectStr);
}

TEST_F(MseLossDavidTiling, mse_loss_david_tiling11)
{
    // test empty tensor
    std::initializer_list<int64_t> predictShape = {2048, 1, 48};
    std::initializer_list<int64_t> labelShape = {2048, 1, 48};
    std::initializer_list<int64_t> outputShape = {2048, 1, 48};
    std::string reduction = "none";

    std::string expectStr =
        "1 98304 2048 48 7040 1 1 2048 2048 7040 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 "
        "0 0 ";
    DoMseLossTilingCase(
        predictShape, labelShape, outputShape, ge::DT_BF16 /*inputdtype*/, reduction /*noopWithEmptyAxes*/, expectStr);
}

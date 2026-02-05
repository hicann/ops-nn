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
 * \file test_sigmoid_cross_entropy_with_logits_v2_ascendc_tiling.cpp
 * \brief
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
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "../../../op_host/arch35/sigmoid_cross_entropy_with_logits_v2_tiling.h"

using namespace std;
using namespace ge;
using namespace ut_util;

class SigmoidCrossEntropyWithLogitsV2AscendCTiling : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "SigmoidCrossEntropyWithLogitsV2AscendCTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "SigmoidCrossEntropyWithLogitsV2AscendCTiling TearDown" << std::endl;
    }
};

static string TilingData2Str(const gert::TilingData* tiling_data)
{
    auto data = tiling_data->GetData();
    string result;
    for (size_t i = 0; i < tiling_data->GetDataSize(); i += sizeof(int64_t)) {
        result += std::to_string((reinterpret_cast<const int64_t*>(data)[i / sizeof(int64_t)]));
        result += " ";
    }

    return result;
}


static void InitPlatForm(
    fe::PlatFormInfos& platFormInfo, map<string, string>& socInfos,
    map<string, string>& aicoreSpec, map<string, string>& intrinsics)
{
    string hardwareInfo = R"({
      "hardware_info": {"UB_SIZE": 253952, "CORE_NUM": 64}
                        })";
    GetPlatFormInfos(hardwareInfo.c_str(), socInfos, aicoreSpec, intrinsics);

    platFormInfo.Init();
}

static void DoSigmoidCEWithLogitsV2AscendcTilingCase(
    std::initializer_list<int64_t>& inputShape, std::initializer_list<int64_t>& outputShape,
    ge::DataType inputDtype, ge::DataType outputDtype, std::string& reduction, std::string& expectStr)
{
    // init platform
    fe::PlatFormInfos platFormInfo;
    map<string, string> socInfos;
    map<string, string> aicoreSpec;
    map<string, string> intrinsics;
    map<string, string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}};
    InitPlatForm(platFormInfo, socInfos, aicoreSpec, intrinsics);

    optiling::SigmoidCEWithLogitsV2CompileInfo compileInfo;
    std::string opType("SigmoidCrossEntropyWithLogitsV2");
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
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);
    kernelHolder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "version", soc_version_infos);

    auto tilingParseFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling_parse;
    ASSERT_EQ(tilingParseFunc(kernelHolder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    gert::StorageShape xShape = {inputShape, inputShape};
    gert::StorageShape yShape = {inputShape, inputShape};
    gert::StorageShape weightShape = {inputShape, inputShape};
    gert::StorageShape posWeightShape = {inputShape, inputShape};
    gert::StorageShape outShape = {outputShape, outputShape};
    auto tilingFunc = gert::OpImplRegistry::GetInstance().GetOpImpl(opType.c_str())->tiling;

    // tilingFunc simulate
    auto workspaceSizeHoler = gert::ContinuousVector::Create<size_t>(16 * 4096);
    auto wsSize = reinterpret_cast<gert::ContinuousVector*>(workspaceSizeHoler.get());
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);

    auto holder = gert::TilingContextFaker()
                      .SetOpType(opType)
                      .NodeIoNum(4, 1)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&xShape, &yShape, &weightShape, &posWeightShape})
                      .OutputShapes({&outShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platFormInfo))
                      .NodeInputTd(0, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, inputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, outputDtype, ge::FORMAT_ND, ge::FORMAT_ND)
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
    EXPECT_EQ(tiling_data_result, tiling_data_result);
}

TEST_F(SigmoidCrossEntropyWithLogitsV2AscendCTiling, sigmoid_cross_entropy_with_logits_v2_david_tiling1)
{
    // FLOAT
    std::initializer_list<int64_t> inputShape = {2048, 1, 48};
    std::initializer_list<int64_t> outputShape = {2048, 1, 48};
    std::string reduction = "none";

    std::string expectStr = "0 13194139536384 32 1 1 0 0 ";
    DoSigmoidCEWithLogitsV2AscendcTilingCase(inputShape, outputShape,
        ge::DT_FLOAT /*inputdtype*/, ge::DT_FLOAT /*outputdtype*/, reduction, expectStr);
}

TEST_F(SigmoidCrossEntropyWithLogitsV2AscendCTiling, sigmoid_cross_entropy_with_logits_v2_david_tiling2)
{
    // HALF
    std::initializer_list<int64_t> inputShape = {2048, 1, 48};
    std::initializer_list<int64_t> outputShape = {2048, 1, 48};
    std::string reduction = "none";

    std::string expectStr = "0 13194139536384 32 1 1 0 0 ";
    DoSigmoidCEWithLogitsV2AscendcTilingCase(inputShape, outputShape,
        ge::DT_FLOAT16 /*inputdtype*/, ge::DT_FLOAT /*outputdtype*/, reduction, expectStr);
}

TEST_F(SigmoidCrossEntropyWithLogitsV2AscendCTiling, sigmoid_cross_entropy_with_logits_v2_david_tiling3)
{
    // BF16
    std::initializer_list<int64_t> inputShape = {2048, 1, 48};
    std::initializer_list<int64_t> outputShape = {2048, 1, 48};
    std::string reduction = "none";

    std::string expectStr = "0 13194139536384 32 1 1 0 0 ";
    DoSigmoidCEWithLogitsV2AscendcTilingCase(inputShape, outputShape,
        ge::DT_BF16 /*inputdtype*/, ge::DT_BF16 /*outputdtype*/, reduction, expectStr);
}

TEST_F(SigmoidCrossEntropyWithLogitsV2AscendCTiling, sigmoid_cross_entropy_with_logits_v2_david_tiling4)
{
    // 2D
    std::initializer_list<int64_t> inputShape = {32, 48};
    std::initializer_list<int64_t> outputShape = {32, 48};
    std::string reduction = "none";

    std::string expectStr = "0 549755814016 12 1 1 0 0 ";
    DoSigmoidCEWithLogitsV2AscendcTilingCase(inputShape, outputShape,
        ge::DT_FLOAT16 /*inputdtype*/, ge::DT_FLOAT16 /*outputdtype*/, reduction, expectStr);
}

TEST_F(SigmoidCrossEntropyWithLogitsV2AscendCTiling, sigmoid_cross_entropy_with_logits_v2_david_tiling5)
{
    // 4D
    std::initializer_list<int64_t> inputShape = {32, 48, 32, 32};
    std::initializer_list<int64_t> outputShape = {32, 48, 32, 32};
    std::string reduction = "none";

    std::string expectStr = "0 6597069770624 57 7 5 0 0 ";
    DoSigmoidCEWithLogitsV2AscendcTilingCase(inputShape, outputShape,
        ge::DT_FLOAT /*inputdtype*/, ge::DT_FLOAT /*outputdtype*/, reduction, expectStr);
}

/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <map>
#include <string>

#include <gtest/gtest.h>
#include "exe_graph/runtime/storage_shape.h"
#include "kernel_run_context_facker.h"
#include "platform/platform_infos_def.h"
#include "register/op_impl_registry.h"
#include "test_cube_util.h"
#include "ut_op_util.h"

namespace {

ge::graphStatus RunTiling(ge::DataType predictDtype, ge::DataType labelDtype, ge::DataType doutDtype,
                          ge::DataType gradientDtype, const std::string& reduction)
{
    std::map<std::string, std::string> socInfos;
    std::map<std::string, std::string> aicoreSpec;
    std::map<std::string, std::string> intrinsics;
    std::map<std::string, std::string> socVersion = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    const std::string hardwareInfo = R"({"hardware_info":{"UB_SIZE":253952,"CORE_NUM":64,"socVersion":"Ascend950"}})";
    GetPlatFormInfos(hardwareInfo.c_str(), socInfos, aicoreSpec, intrinsics, socVersion);

    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    gert::StorageShape inputShape = {{2, 2}, {2, 2}};
    gert::StorageShape outputShape = {{2, 2}, {2, 2}};
    auto tilingData = gert::TilingData::CreateCap(4096);
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto workspace = reinterpret_cast<gert::ContinuousVector*>(workspaceHolder.get());
    uint8_t compileInfo = 0;

    auto holder = gert::TilingContextFaker()
                      .SetOpType("SoftMarginLossGrad")
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&inputShape, &inputShape, &inputShape})
                      .OutputShapes({&outputShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, predictDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, labelDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, doutDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, gradientDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"reduction", Ops::NN::AnyValue::CreateFrom<std::string>(reduction)}})
                      .TilingData(tilingData.get())
                      .Workspace(workspace)
                      .Build();

    auto* context = holder.GetContext<gert::TilingContext>();
    context->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    context->GetPlatformInfo()->SetPlatformRes("version", socVersion);

    const auto* opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("SoftMarginLossGrad");
    if (opImpl == nullptr || opImpl->tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    return opImpl->tiling(context);
}

TEST(SoftMarginLossGradTiling, AcceptsValidParams)
{
    EXPECT_EQ(RunTiling(ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, "mean"), ge::GRAPH_SUCCESS);
}

TEST(SoftMarginLossGradTiling, RejectsInvalidReduction)
{
    EXPECT_EQ(RunTiling(ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, "invalid"), ge::GRAPH_FAILED);
}

TEST(SoftMarginLossGradTiling, RejectsMismatchedDtypes)
{
    EXPECT_EQ(RunTiling(ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, "mean"), ge::GRAPH_FAILED);
    EXPECT_EQ(RunTiling(ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT, "mean"), ge::GRAPH_FAILED);
    EXPECT_EQ(RunTiling(ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT, "mean"), ge::GRAPH_FAILED);
    EXPECT_EQ(RunTiling(ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT16, "mean"), ge::GRAPH_FAILED);
}

} // namespace

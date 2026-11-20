/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <type_traits>
#include <iostream>
#include <string>
#include "experiment/metadef/common/util/error_manager/error_manager.h"
#include "platform/platform_infos_def.h"
#include "tiling_context_faker.h"
#include "op_impl_registry.h"
#include "../../../op_host/op_tiling/rms_norm_hi_f4_quant_tiling.h"
#include "op_host/op_const_def_nn.h"
using namespace std;
using namespace ge;
namespace {
using ErrSnapshot = std::decay_t<decltype(ErrorManager::GetInstance().GetRawErrorMessages())>;

ErrSnapshot TakeSnapshot() { return ErrorManager::GetInstance().GetRawErrorMessages(); }

string Report(const ErrSnapshot& errors)
{
    string text;
    for (const auto& error : errors) {
        text += error.error_id + "{";
        for (const auto& field : error.args_map) {
            text += field.first + "=" + field.second + ";";
        }
        text += "} ";
    }
    return text;
}

void ClearErrors()
{
    EXPECT_EQ(ErrorManager::GetInstance().Init(), 0);
    (void)ErrorManager::GetInstance().GetRawErrorMessages();
}

struct ExposedRmsHi4Tiling : optiling::RmsNormHiF4QuantRFullLoadTiling {
    using RmsNormHiF4QuantRFullLoadTiling::GetShapeAttrsInfo;
    using RmsNormHiF4QuantRFullLoadTiling::RmsNormHiF4QuantRFullLoadTiling;
};

gert::KernelRunContextHolder MakeRmsHi4Context(optiling::RmsNormHiF4QuantCompileInfo* compileInfo,
                                               gert::StorageShape& xShape, gert::StorageShape& gammaShape,
                                               ge::DataType xDtype, ge::DataType yDtype)
{
    static auto tilingData = gert::TilingData::CreateCap(4096);
    static auto workspaceHolder = gert::ContinuousVector::Create<size_t>(4096);
    static fe::PlatFormInfos platform;
    platform.Init();
    auto* workspace = reinterpret_cast<gert::ContinuousVector*>(workspaceHolder.get());
    static gert::StorageShape betaShape{{64}, {64}};
    static gert::StorageShape yShape{{8, 64}, {8, 64}};
    static gert::StorageShape scaleShape{{8, 1, 1}, {8, 1, 1}};
    static gert::StorageShape rstdShape{{8}, {8}};
    return gert::TilingContextFaker()
        .SetOpType("RmsNormHiF4Quant")
        .NodeIoNum(3, 3)
        .IrInstanceNum({1, 1, 1})
        .InputShapes({&xShape, &gammaShape, &betaShape})
        .OutputShapes({&yShape, &scaleShape, &rstdShape})
        .NodeInputTd(0, xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, yDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(1, Ops::NN::DT_HIFLOAT4_SCALE, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeAttrs({})
        .CompileInfo(compileInfo)
        .PlatformInfo(reinterpret_cast<char*>(&platform))
        .TilingData(tilingData.get())
        .Workspace(workspace)
        .Build();
}
} // namespace

TEST(RmsNormHiF4QuantPropagationErrorReporting, Target13540_CheckInputShapeDimNegativeDim)
{
    optiling::RmsNormHiF4QuantCompileInfo info{};
    static gert::StorageShape xBad{{8, -1}, {8, -1}};
    static gert::StorageShape gammaOk{{64}, {64}};
    auto holder = MakeRmsHi4Context(&info, xBad, gammaOk, ge::DT_FLOAT16, ge::DT_HIFLOAT4);
    auto* context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(context, nullptr);
    ASSERT_NE(context->GetInputShape(optiling::X_INDEX), nullptr);
    ExposedRmsHi4Tiling tiling(context);
    ClearErrors();
    const auto status = tiling.CheckInputShapeDim();
    const auto errors = TakeSnapshot();
    std::cout << "[g18][13540] level=CALLEE_REAL_REPORT return=" << status << " entry=CheckInputShapeDim"
              << " count=" << errors.size() << " errors: " << Report(errors) << std::endl;
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    ASSERT_EQ(errors.size(), 1U);
    EXPECT_EQ(errors[0].error_id, "EZ0009");
    ASSERT_EQ(errors[0].args_map.count("param_name"), 1U);
    EXPECT_EQ(errors[0].args_map.at("param_name"), "x");
    ASSERT_EQ(errors[0].args_map.count("reason"), 1U);
    EXPECT_EQ(errors[0].args_map.at("reason"), "All dimensions must be nonnegative");
}

TEST(RmsNormHiF4QuantPropagationErrorReporting, Target13578_GetShapeAttrsInfoRankTooLarge)
{
    optiling::RmsNormHiF4QuantCompileInfo info{};
    static gert::StorageShape xRank8{{1, 1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1, 1}};
    static gert::StorageShape gammaOk{{64}, {64}};
    auto holder = MakeRmsHi4Context(&info, xRank8, gammaOk, ge::DT_FLOAT16, ge::DT_HIFLOAT4);
    auto* context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(context, nullptr);
    ASSERT_NE(context->GetInputShape(optiling::X_INDEX), nullptr);
    ExposedRmsHi4Tiling tiling(context);
    ClearErrors();
    const auto status = tiling.GetShapeAttrsInfo();
    const auto errors = TakeSnapshot();
    std::cout << "[g18][13578] level=OUTER_CALLER_LOWER_FAILURE return=" << status << " entry=GetShapeAttrsInfo"
              << " count=" << errors.size() << " errors: " << Report(errors) << std::endl;
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    ASSERT_EQ(errors.size(), 1U);
    EXPECT_EQ(errors[0].error_id, "EZ0012");
    ASSERT_EQ(errors[0].args_map.count("param_name"), 1U);
    EXPECT_EQ(errors[0].args_map.at("param_name"), "x");
    EXPECT_EQ(errors[0].args_map.count("reason"), 1U);
    EXPECT_EQ(errors[0].args_map.at("reason"), "Rank must be in [1, 7]");
    ASSERT_EQ(errors[0].args_map.count("incorrect_dim"), 1U);
    EXPECT_EQ(errors[0].args_map.at("incorrect_dim"), "8");
}

TEST(RmsNormHiF4QuantPropagationErrorReporting, Target13579_GetShapeAttrsInfoGammaRankNotOne)
{
    optiling::RmsNormHiF4QuantCompileInfo info{};
    static gert::StorageShape xOk{{8, 64}, {8, 64}};
    static gert::StorageShape gammaRank2{{8, 64}, {8, 64}};
    auto holder = MakeRmsHi4Context(&info, xOk, gammaRank2, ge::DT_FLOAT16, ge::DT_HIFLOAT4);
    auto* context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(context, nullptr);
    ASSERT_NE(context->GetInputShape(optiling::X_INDEX), nullptr);
    ASSERT_NE(context->GetInputShape(optiling::GAMMA_INDEX), nullptr);
    ExposedRmsHi4Tiling tiling(context);
    ClearErrors();
    const auto status = tiling.GetShapeAttrsInfo();
    const auto errors = TakeSnapshot();
    std::cout << "[g18][13579] level=OUTER_CALLER_LOWER_FAILURE return=" << status << " entry=GetShapeAttrsInfo"
              << " count=" << errors.size() << " errors: " << Report(errors) << std::endl;
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    ASSERT_EQ(errors.size(), 1U);
    EXPECT_EQ(errors[0].error_id, "EZ0012");
    ASSERT_EQ(errors[0].args_map.count("param_name"), 1U);
    EXPECT_EQ(errors[0].args_map.at("param_name"), "gamma");
    ASSERT_EQ(errors[0].args_map.count("reason"), 1U);
    EXPECT_EQ(errors[0].args_map.at("reason"), "Rank must be 1");
    ASSERT_EQ(errors[0].args_map.count("incorrect_dim"), 1U);
    EXPECT_EQ(errors[0].args_map.at("incorrect_dim"), "2");
}

TEST(RmsNormHiF4QuantPropagationErrorReporting, Target13580_GetShapeAttrsInfoXDtypeInvalid)
{
    optiling::RmsNormHiF4QuantCompileInfo info{};
    static gert::StorageShape xOk{{8, 64}, {8, 64}};
    static gert::StorageShape gammaOk{{64}, {64}};
    auto holder = MakeRmsHi4Context(&info, xOk, gammaOk, ge::DT_INT32, ge::DT_HIFLOAT4);
    auto* context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(context, nullptr);
    ASSERT_NE(context->GetInputTensor(optiling::X_INDEX), nullptr);
    ASSERT_NE(context->GetInputTensor(optiling::GAMMA_INDEX), nullptr);
    ExposedRmsHi4Tiling tiling(context);
    ClearErrors();
    const auto status = tiling.GetShapeAttrsInfo();
    const auto errors = TakeSnapshot();
    std::cout << "[g18][13580] level=OUTER_CALLER_LOWER_FAILURE return=" << status << " entry=GetShapeAttrsInfo"
              << " count=" << errors.size() << " errors: " << Report(errors) << std::endl;
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    ASSERT_EQ(errors.size(), 1U);
    EXPECT_FALSE(errors[0].error_id.empty());
    ASSERT_EQ(errors[0].args_map.count("param_name"), 1U);
    EXPECT_EQ(errors[0].args_map.at("param_name"), "x");
    ASSERT_EQ(errors[0].args_map.count("reason"), 1U);
    EXPECT_NE(errors[0].args_map.at("reason").find("Supported dtypes"), string::npos);
}

TEST(RmsNormHiF4QuantPropagationErrorReporting, Target13581_GetShapeAttrsInfoOutputDtypeInvalid)
{
    optiling::RmsNormHiF4QuantCompileInfo info{};
    static gert::StorageShape xOk{{8, 64}, {8, 64}};
    static gert::StorageShape gammaOk{{64}, {64}};
    auto holder = MakeRmsHi4Context(&info, xOk, gammaOk, ge::DT_FLOAT16, ge::DT_FLOAT16);
    auto* context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(context, nullptr);
    ASSERT_NE(context->GetOutputDesc(optiling::Y_INDEX), nullptr);
    ExposedRmsHi4Tiling tiling(context);
    ClearErrors();
    const auto status = tiling.GetShapeAttrsInfo();
    const auto errors = TakeSnapshot();
    std::cout << "[g18][13581] level=OUTER_CALLER_LOWER_FAILURE return=" << status << " entry=GetShapeAttrsInfo"
              << " count=" << errors.size() << " errors: " << Report(errors) << std::endl;
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    ASSERT_EQ(errors.size(), 1U);
    EXPECT_FALSE(errors[0].error_id.empty());
    ASSERT_EQ(errors[0].args_map.count("param_name"), 1U);
    EXPECT_EQ(errors[0].args_map.at("param_name"), "y");
    ASSERT_EQ(errors[0].args_map.count("reason"), 1U);
    EXPECT_EQ(errors[0].args_map.at("reason"), "Output y dtype must be HIFLOAT4");
}

TEST(RmsNormHiF4QuantPropagationErrorReporting, MissingPlatformReportsOneFormalError)
{
    optiling::RmsNormHiF4QuantCompileInfo info{};
    gert::StorageShape input{{8, 64}, {8, 64}};
    gert::StorageShape gamma{{64}, {64}};
    auto holder = MakeRmsHi4Context(&info, input, gamma, ge::DT_FLOAT16, ge::DT_HIFLOAT4);
    auto* context = holder.GetContext<gert::TilingContext>();
    auto* kernel = holder.GetContext<gert::KernelContext>();
    ASSERT_NE(context, nullptr);
    ASSERT_NE(context->GetComputeNodeInfo(), nullptr);
    const size_t slot = context->GetComputeNodeInfo()->GetInputsNum() + context->GetComputeNodeInfo()->GetOutputsNum() +
                        1U;
    ASSERT_NE(kernel->MutableInput(slot), nullptr);
    kernel->MutableInput(slot)->Set(nullptr, nullptr);
    ASSERT_EQ(context->GetPlatformInfo(), nullptr);
    ExposedRmsHi4Tiling tiling(context);
    ClearErrors();
    EXPECT_EQ(tiling.GetShapeAttrsInfo(), ge::GRAPH_FAILED);
    const auto errors = TakeSnapshot();
    ASSERT_EQ(errors.size(), 1U);
    EXPECT_EQ(errors[0].error_id, "EZ1008");
    EXPECT_EQ(errors[0].args_map.at("reason"), "Missing platform metadata for vector resources");
}

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
#include <vector>
#include "experiment/metadef/common/util/error_manager/error_manager.h"
#include "platform/platform_infos_def.h"
#include "tiling_context_faker.h"
#include "op_impl_registry.h"
#include "../../../op_host/op_tiling/add_rms_norm_hi_f4_quant_tiling.h"
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

struct ExposedAddHi4Tiling : optiling::AddRmsNormHiF4QuantRFullLoadTiling {
    using AddRmsNormHiF4QuantRFullLoadTiling::AddRmsNormHiF4QuantRFullLoadTiling;
    void SetBetaFlag(uint32_t value) { betaFlag_ = value; }
    uint32_t GetBetaFlag() const { return betaFlag_; }
};

gert::KernelRunContextHolder MakeAddHi4Context(optiling::AddRmsNormHiF4QuantCompileInfo* compileInfo)
{
    static auto tilingData = gert::TilingData::CreateCap(4096);
    static auto workspaceHolder = gert::ContinuousVector::Create<size_t>(4096);
    static fe::PlatFormInfos platform;
    platform.Init();
    auto* workspace = reinterpret_cast<gert::ContinuousVector*>(workspaceHolder.get());
    static gert::StorageShape x1{{8, 64}, {8, 64}};
    static gert::StorageShape x2{{8, 64}, {8, 64}};
    static gert::StorageShape gamma{{64}, {64}};
    static gert::StorageShape beta{{64}, {64}};
    static gert::StorageShape y{{8, 64}, {8, 64}};
    static gert::StorageShape xOut{{8, 64}, {8, 64}};
    static gert::StorageShape mxScale{{8, 1}, {8, 1}};
    static gert::StorageShape rstd{{8}, {8}};
    return gert::TilingContextFaker()
        .SetOpType("AddRmsNormHiF4Quant")
        .NodeIoNum(4, 4)
        .IrInstanceNum({1, 1, 1, 1})
        .InputShapes({&x1, &x2, &gamma, &beta})
        .OutputShapes({&y, &xOut, &mxScale, &rstd})
        .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, ge::DT_INT8, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeAttrs({})
        .CompileInfo(compileInfo)
        .PlatformInfo(reinterpret_cast<char*>(&platform))
        .TilingData(tilingData.get())
        .Workspace(workspace)
        .Build();
}
} // namespace

TEST(AddRmsNormHiF4QuantMissingShapeErrorReporting, Target902_BetaFlagTrueButOptionalBetaShapeMissing)
{
    optiling::AddRmsNormHiF4QuantCompileInfo info{};
    auto holder = MakeAddHi4Context(&info);
    auto* context = holder.GetContext<gert::TilingContext>();
    std::cout << "[p55][factory] context=" << context << std::endl;
    ASSERT_NE(context, nullptr);
    ASSERT_NE(context->GetInputShape(optiling::X1_INDEX), nullptr);
    ASSERT_NE(context->GetInputShape(optiling::X2_INDEX), nullptr);
    ASSERT_NE(context->GetInputShape(optiling::GAMMA_INDEX), nullptr);
    ASSERT_NE(context->GetOptionalInputShape(optiling::BETA_INDEX), nullptr);
    auto* kernel = holder.GetContext<gert::KernelContext>();
    ASSERT_NE(kernel, nullptr);
    ASSERT_NE(kernel->MutableInput(optiling::BETA_INDEX), nullptr);
    kernel->MutableInput(optiling::BETA_INDEX)->Set(nullptr, nullptr);
    ASSERT_EQ(context->GetOptionalInputShape(optiling::BETA_INDEX), nullptr);
    ASSERT_NE(context->GetInputShape(optiling::X1_INDEX), nullptr);
    ASSERT_NE(context->GetInputShape(optiling::X2_INDEX), nullptr);
    ASSERT_NE(context->GetInputShape(optiling::GAMMA_INDEX), nullptr);
    ExposedAddHi4Tiling tiling(context);
    tiling.SetBetaFlag(1U);
    ASSERT_EQ(tiling.GetBetaFlag(), 1U);
    ClearErrors();
    const auto status = tiling.CheckInputShapeValue();
    const auto errors = TakeSnapshot();
    std::cout << "[p55][902] level=FUNCTION_STATE_INJECTION"
              << " return=" << status << " entry=CheckInputShapeValue"
              << " betaFlag=" << tiling.GetBetaFlag()
              << " betaShape=" << context->GetOptionalInputShape(optiling::BETA_INDEX) << " count=" << errors.size()
              << " errors: " << Report(errors) << std::endl;
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    ASSERT_EQ(errors.size(), 1U);
    EXPECT_EQ(errors[0].error_id, "EZ1008");
    ASSERT_EQ(errors[0].args_map.count("reason"), 1U);
    EXPECT_NE(errors[0].args_map.at("reason").find("Missing tiling context object: betaShape"), std::string::npos);
    ASSERT_EQ(errors[0].args_map.count("opName"), 1U);
    EXPECT_FALSE(errors[0].args_map.at("opName").empty());
}

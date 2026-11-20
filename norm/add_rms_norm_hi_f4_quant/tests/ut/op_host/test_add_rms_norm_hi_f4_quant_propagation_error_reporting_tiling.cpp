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

struct ExposedGetShapeAttrsInfo : optiling::AddRmsNormHiF4QuantRFullLoadTiling {
    using AddRmsNormHiF4QuantRFullLoadTiling::AddRmsNormHiF4QuantRFullLoadTiling;
    ge::graphStatus RunGetShapeAttrsInfo() { return GetShapeAttrsInfo(); }
};

struct Cfg {
    gert::StorageShape* x1;
    gert::StorageShape* x2;
    gert::StorageShape* gamma;
    gert::StorageShape* beta;
    gert::StorageShape* y;
    gert::StorageShape* xOut;
    gert::StorageShape* scale;
    gert::StorageShape* rstd;
    ge::DataType x1Dtype;
    ge::DataType x2Dtype;
    ge::DataType gammaDtype;
    ge::DataType betaDtype;
    ge::DataType yDtype;
    ge::DataType xOutDtype;
    ge::DataType scaleDtype;
    ge::DataType rstdDtype;
};

gert::KernelRunContextHolder MakeAddHi4Context(const Cfg& cfg, optiling::AddRmsNormHiF4QuantCompileInfo* compileInfo)
{
    static auto tilingData = gert::TilingData::CreateCap(4096);
    static auto workspaceHolder = gert::ContinuousVector::Create<size_t>(4096);
    static fe::PlatFormInfos platform;
    platform.Init();
    auto* workspace = reinterpret_cast<gert::ContinuousVector*>(workspaceHolder.get());
    return gert::TilingContextFaker()
        .SetOpType("AddRmsNormHiF4Quant")
        .NodeIoNum(4, 4)
        .IrInstanceNum({1, 1, 1, 1})
        .InputShapes({cfg.x1, cfg.x2, cfg.gamma, cfg.beta})
        .OutputShapes({cfg.y, cfg.xOut, cfg.scale, cfg.rstd})
        .NodeInputTd(0, cfg.x1Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(1, cfg.x2Dtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(2, cfg.gammaDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(3, cfg.betaDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, cfg.yDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(1, cfg.xOutDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(2, cfg.scaleDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(3, cfg.rstdDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeAttrs({})
        .CompileInfo(compileInfo)
        .PlatformInfo(reinterpret_cast<char*>(&platform))
        .TilingData(tilingData.get())
        .Workspace(workspace)
        .Build();
}
} // namespace

TEST(AddRmsNormHiF4QuantPropagationErrorReporting, Target941_InputShapeDimInvalidWrapperBranch)
{
    static gert::StorageShape x1{{1, 1, 1, 1, 1, 1, 1, 2}, {1, 1, 1, 1, 1, 1, 1, 2}};
    static gert::StorageShape x2{{1, 1, 1, 1, 1, 1, 1, 2}, {1, 1, 1, 1, 1, 1, 1, 2}};
    static gert::StorageShape gamma{{2}, {2}};
    static gert::StorageShape beta{{2}, {2}};
    static gert::StorageShape y{{1, 1}, {1, 1}};
    static gert::StorageShape xOut{{1, 1}, {1, 1}};
    static gert::StorageShape mxScale{{1, 1}, {1, 1}};
    static gert::StorageShape rstd{{1}, {1}};
    optiling::AddRmsNormHiF4QuantCompileInfo info{};
    Cfg cfg{&x1,
            &x2,
            &gamma,
            &beta,
            &y,
            &xOut,
            &mxScale,
            &rstd,
            ge::DT_FLOAT16,
            ge::DT_FLOAT16,
            ge::DT_FLOAT16,
            ge::DT_FLOAT16,
            ge::DT_HIFLOAT4,
            ge::DT_FLOAT16,
            Ops::NN::DT_HIFLOAT4_SCALE,
            ge::DT_FLOAT};
    auto holder = MakeAddHi4Context(cfg, &info);
    auto* context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(context, nullptr);
    ASSERT_NE(context->GetInputShape(optiling::X1_INDEX), nullptr);
    ASSERT_NE(context->GetInputShape(optiling::X2_INDEX), nullptr);
    ASSERT_NE(context->GetInputShape(optiling::GAMMA_INDEX), nullptr);
    ASSERT_EQ(context->GetInputShape(optiling::X1_INDEX)->GetStorageShape().GetDimNum(), 8U);
    ExposedGetShapeAttrsInfo tiling(context);
    ClearErrors();
    const auto status = tiling.RunGetShapeAttrsInfo();
    const auto errors = TakeSnapshot();
    std::cout << "[g18][941] level=WRAPPER_LOWER_FAILURE entry=GetShapeAttrsInfo"
              << " log_site=arch92/add_rms_norm_hi_f4_quant_tiling_base.cpp:357"
              << " return=" << status << " count=" << errors.size() << " errors: " << Report(errors) << std::endl;
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    ASSERT_EQ(errors.size(), 1U);
    EXPECT_EQ(errors[0].error_id, "EZ0013");
    ASSERT_EQ(errors[0].args_map.count("param_names"), 1U);
    EXPECT_NE(errors[0].args_map.at("param_names").find("x1"), std::string::npos);
    ASSERT_EQ(errors[0].args_map.count("incorrect_dims"), 1U);
    EXPECT_NE(errors[0].args_map.at("incorrect_dims").find("8"), std::string::npos);
    ASSERT_EQ(errors[0].args_map.count("reason"), 1U);
    EXPECT_NE(errors[0].args_map.at("reason").find("Ranks must be in"), std::string::npos);
    ASSERT_EQ(errors[0].args_map.count("op_name"), 1U);
    EXPECT_FALSE(errors[0].args_map.at("op_name").empty());
}

TEST(AddRmsNormHiF4QuantPropagationErrorReporting, Target942_InputShapeValueInvalidWrapperBranch)
{
    static gert::StorageShape x1{{8, 64}, {8, 64}};
    static gert::StorageShape x2{{8, 128}, {8, 128}};
    static gert::StorageShape gamma{{64}, {64}};
    static gert::StorageShape beta{{64}, {64}};
    static gert::StorageShape y{{8, 64}, {8, 64}};
    static gert::StorageShape xOut{{8, 64}, {8, 64}};
    static gert::StorageShape mxScale{{8, 1}, {8, 1}};
    static gert::StorageShape rstd{{8}, {8}};
    optiling::AddRmsNormHiF4QuantCompileInfo info{};
    Cfg cfg{&x1,
            &x2,
            &gamma,
            &beta,
            &y,
            &xOut,
            &mxScale,
            &rstd,
            ge::DT_FLOAT16,
            ge::DT_FLOAT16,
            ge::DT_FLOAT16,
            ge::DT_FLOAT16,
            ge::DT_HIFLOAT4,
            ge::DT_FLOAT16,
            Ops::NN::DT_HIFLOAT4_SCALE,
            ge::DT_FLOAT};
    auto holder = MakeAddHi4Context(cfg, &info);
    auto* context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(context, nullptr);
    ASSERT_NE(context->GetInputShape(optiling::X1_INDEX), nullptr);
    ASSERT_NE(context->GetInputShape(optiling::X2_INDEX), nullptr);
    ASSERT_EQ(context->GetInputShape(optiling::X1_INDEX)->GetStorageShape().GetDimNum(), 2U);
    ASSERT_EQ(context->GetInputShape(optiling::X2_INDEX)->GetStorageShape().GetDimNum(), 2U);
    ExposedGetShapeAttrsInfo tiling(context);
    ClearErrors();
    const auto status = tiling.RunGetShapeAttrsInfo();
    const auto errors = TakeSnapshot();
    std::cout << "[g18][942] level=WRAPPER_LOWER_FAILURE entry=GetShapeAttrsInfo"
              << " log_site=arch92/add_rms_norm_hi_f4_quant_tiling_base.cpp:360"
              << " return=" << status << " count=" << errors.size() << " errors: " << Report(errors) << std::endl;
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    ASSERT_EQ(errors.size(), 1U);
    ASSERT_EQ(errors[0].args_map.count("op_name"), 1U);
    EXPECT_FALSE(errors[0].args_map.at("op_name").empty());
    ASSERT_EQ(errors[0].args_map.count("reason"), 1U);
}

TEST(AddRmsNormHiF4QuantPropagationErrorReporting, Target943_InputDtypeInvalidWrapperBranch)
{
    static gert::StorageShape x1{{8, 64}, {8, 64}};
    static gert::StorageShape x2{{8, 64}, {8, 64}};
    static gert::StorageShape gamma{{64}, {64}};
    static gert::StorageShape beta{{64}, {64}};
    static gert::StorageShape y{{8, 64}, {8, 64}};
    static gert::StorageShape xOut{{8, 64}, {8, 64}};
    static gert::StorageShape mxScale{{8, 1}, {8, 1}};
    static gert::StorageShape rstd{{8}, {8}};
    optiling::AddRmsNormHiF4QuantCompileInfo info{};
    Cfg cfg{&x1,
            &x2,
            &gamma,
            &beta,
            &y,
            &xOut,
            &mxScale,
            &rstd,
            ge::DT_FLOAT16,
            ge::DT_BF16,
            ge::DT_FLOAT16,
            ge::DT_FLOAT16,
            ge::DT_HIFLOAT4,
            ge::DT_FLOAT16,
            Ops::NN::DT_HIFLOAT4_SCALE,
            ge::DT_FLOAT};
    auto holder = MakeAddHi4Context(cfg, &info);
    auto* context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(context, nullptr);
    ASSERT_NE(context->GetInputShape(optiling::X1_INDEX), nullptr);
    ASSERT_NE(context->GetInputShape(optiling::X2_INDEX), nullptr);
    ASSERT_NE(context->GetInputShape(optiling::GAMMA_INDEX), nullptr);
    ExposedGetShapeAttrsInfo tiling(context);
    ClearErrors();
    const auto status = tiling.RunGetShapeAttrsInfo();
    const auto errors = TakeSnapshot();
    std::cout << "[g18][943] level=WRAPPER_LOWER_FAILURE entry=GetShapeAttrsInfo"
              << " log_site=arch92/add_rms_norm_hi_f4_quant_tiling_base.cpp:361"
              << " return=" << status << " count=" << errors.size() << " errors: " << Report(errors) << std::endl;
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    ASSERT_EQ(errors.size(), 1U);
    ASSERT_EQ(errors[0].args_map.count("op_name"), 1U);
    EXPECT_FALSE(errors[0].args_map.at("op_name").empty());
    ASSERT_EQ(errors[0].args_map.count("reason"), 1U);
    EXPECT_NE(errors[0].args_map.at("reason").find("dtypes must be equal"), std::string::npos);
}

TEST(AddRmsNormHiF4QuantPropagationErrorReporting, Target944_OutputDtypeInvalidWrapperBranch)
{
    static gert::StorageShape x1{{8, 64}, {8, 64}};
    static gert::StorageShape x2{{8, 64}, {8, 64}};
    static gert::StorageShape gamma{{64}, {64}};
    static gert::StorageShape beta{{64}, {64}};
    static gert::StorageShape y{{8, 64}, {8, 64}};
    static gert::StorageShape xOut{{8, 64}, {8, 64}};
    static gert::StorageShape mxScale{{8, 1}, {8, 1}};
    static gert::StorageShape rstd{{8}, {8}};
    optiling::AddRmsNormHiF4QuantCompileInfo info{};
    Cfg cfg{&x1,
            &x2,
            &gamma,
            &beta,
            &y,
            &xOut,
            &mxScale,
            &rstd,
            ge::DT_FLOAT16,
            ge::DT_FLOAT16,
            ge::DT_FLOAT16,
            ge::DT_FLOAT16,
            ge::DT_FLOAT16,
            ge::DT_FLOAT16,
            Ops::NN::DT_HIFLOAT4_SCALE,
            ge::DT_FLOAT};
    auto holder = MakeAddHi4Context(cfg, &info);
    auto* context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(context, nullptr);
    ASSERT_NE(context->GetOutputDesc(optiling::Y_INDEX), nullptr);
    ASSERT_NE(context->GetOutputDesc(optiling::Y_INDEX)->GetDataType(), ge::DT_HIFLOAT4);
    ExposedGetShapeAttrsInfo tiling(context);
    ClearErrors();
    const auto status = tiling.RunGetShapeAttrsInfo();
    const auto errors = TakeSnapshot();
    std::cout << "[g18][944] level=WRAPPER_LOWER_FAILURE entry=GetShapeAttrsInfo"
              << " log_site=arch92/add_rms_norm_hi_f4_quant_tiling_base.cpp:364"
              << " return=" << status << " count=" << errors.size() << " errors: " << Report(errors) << std::endl;
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    ASSERT_EQ(errors.size(), 1U);
    EXPECT_EQ(errors[0].args_map.count("reason"), 1U);
    EXPECT_EQ(errors[0].args_map.at("reason"), "Output y dtype must be HIFLOAT4");
    ASSERT_EQ(errors[0].args_map.count("op_name"), 1U);
    EXPECT_FALSE(errors[0].args_map.at("op_name").empty());
}

TEST(AddRmsNormHiF4QuantPropagationErrorReporting, MissingPlatformReportsOneFormalError)
{
    optiling::AddRmsNormHiF4QuantCompileInfo info{};
    gert::StorageShape input{{8, 64}, {8, 64}};
    gert::StorageShape gamma{{64}, {64}};
    gert::StorageShape scale{{8, 1, 1}, {8, 1, 1}};
    gert::StorageShape rstd{{8}, {8}};
    Cfg cfg{&input,         &input,         &gamma,          &gamma,         &input,
            &input,         &scale,         &rstd,           ge::DT_FLOAT16, ge::DT_FLOAT16,
            ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_HIFLOAT4, ge::DT_FLOAT16, Ops::NN::DT_HIFLOAT4_SCALE,
            ge::DT_FLOAT};
    auto holder = MakeAddHi4Context(cfg, &info);
    auto* context = holder.GetContext<gert::TilingContext>();
    auto* kernel = holder.GetContext<gert::KernelContext>();
    ASSERT_NE(context, nullptr);
    ASSERT_NE(context->GetComputeNodeInfo(), nullptr);
    const size_t slot = context->GetComputeNodeInfo()->GetInputsNum() + context->GetComputeNodeInfo()->GetOutputsNum() +
                        1U;
    ASSERT_NE(kernel->MutableInput(slot), nullptr);
    kernel->MutableInput(slot)->Set(nullptr, nullptr);
    ASSERT_EQ(context->GetPlatformInfo(), nullptr);
    ExposedGetShapeAttrsInfo tiling(context);
    ClearErrors();
    EXPECT_EQ(tiling.RunGetShapeAttrsInfo(), ge::GRAPH_FAILED);
    const auto errors = TakeSnapshot();
    ASSERT_EQ(errors.size(), 1U);
    EXPECT_EQ(errors[0].error_id, "EZ1008");
    EXPECT_EQ(errors[0].args_map.at("reason"), "Missing platform metadata for vector resources");
}

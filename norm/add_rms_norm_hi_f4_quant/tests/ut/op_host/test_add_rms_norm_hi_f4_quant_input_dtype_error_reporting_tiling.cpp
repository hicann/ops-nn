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
#include "exe_graph/runtime/kernel_context.h"
#include "tiling_context_faker.h"
#include "op_impl_registry.h"
#include "op_host/op_const_def_nn.h"
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

struct ExposedGetShapeAttrsInfo : optiling::AddRmsNormHiF4QuantRFullLoadTiling {
    using AddRmsNormHiF4QuantRFullLoadTiling::AddRmsNormHiF4QuantRFullLoadTiling;
    ge::graphStatus RunGetShapeAttrsInfo() { return GetShapeAttrsInfo(); }
};

struct ExposedCheckInputDtype : optiling::AddRmsNormHiF4QuantRFullLoadTiling {
    using AddRmsNormHiF4QuantRFullLoadTiling::AddRmsNormHiF4QuantRFullLoadTiling;
    ge::graphStatus RunCheckInputDtype() { return CheckInputDtype(); }
};

struct Cfg {
    gert::StorageShape* x1;
    gert::StorageShape* x2;
    gert::StorageShape* gamma;
    gert::StorageShape* beta;
    gert::StorageShape* y;
    gert::StorageShape* xOut;
    gert::StorageShape* mxScale;
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
        .OutputShapes({cfg.y, cfg.xOut, cfg.mxScale, cfg.rstd})
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

TEST(AddRmsNormHiF4QuantInputDtypeErrorReporting, Target900_GAMMA_RANK)
{
    static gert::StorageShape x1{{8, 64}, {8, 64}};
    static gert::StorageShape x2{{8, 64}, {8, 64}};
    static gert::StorageShape gamma{{8, 64}, {8, 64}};
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
    ExposedGetShapeAttrsInfo tiling(context);
    ClearErrors();
    std::cout << "[AddRmsNormHiF4QuantInputDtypeErrorReporting][900] BEGIN scan_id=900 group=G03 line=84" << std::endl;
    const ge::graphStatus status = tiling.RunGetShapeAttrsInfo();
    const ErrSnapshot errors = TakeSnapshot();
    std::cout << "[AddRmsNormHiF4QuantInputDtypeErrorReporting][900] return=" << status << " count=" << errors.size()
              << " errors: " << Report(errors) << std::endl;
    std::cout << "[AddRmsNormHiF4QuantInputDtypeErrorReporting][900] END" << std::endl;
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    ASSERT_EQ(errors.size(), 1U);
    EXPECT_EQ(errors[0].error_id, "EZ0012");
    EXPECT_EQ(errors[0].args_map.count("op_name"), 1U);
    EXPECT_FALSE(errors[0].args_map.at("op_name").empty());
    EXPECT_EQ(errors[0].args_map.at("param_name"), "gamma");
    EXPECT_EQ(errors[0].args_map.at("reason"), "Rank must be 1");
}

TEST(AddRmsNormHiF4QuantInputDtypeErrorReporting, Target907_X_DTYPE_MISMATCH)
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
    ExposedGetShapeAttrsInfo tiling(context);
    ClearErrors();
    std::cout << "[AddRmsNormHiF4QuantInputDtypeErrorReporting][907] BEGIN scan_id=907 group=G07 line=118" << std::endl;
    const ge::graphStatus status = tiling.RunGetShapeAttrsInfo();
    const ErrSnapshot errors = TakeSnapshot();
    std::cout << "[AddRmsNormHiF4QuantInputDtypeErrorReporting][907] return=" << status << " count=" << errors.size()
              << " errors: " << Report(errors) << std::endl;
    std::cout << "[AddRmsNormHiF4QuantInputDtypeErrorReporting][907] END" << std::endl;
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    ASSERT_EQ(errors.size(), 1U);
    EXPECT_EQ(errors[0].error_id, "EZ0021");
    EXPECT_EQ(errors[0].args_map.count("op_name"), 1U);
    EXPECT_FALSE(errors[0].args_map.at("op_name").empty());
    EXPECT_EQ(errors[0].args_map.at("param_names"), "x1, x2");
    EXPECT_EQ(errors[0].args_map.at("reason"), "Input x1 and x2 dtypes must be equal");
}

TEST(AddRmsNormHiF4QuantInputDtypeErrorReporting, Target908_X_DTYPE_UNSUPPORTED)
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
            ge::DT_INT8,
            ge::DT_INT8,
            ge::DT_FLOAT16,
            ge::DT_FLOAT16,
            ge::DT_HIFLOAT4,
            ge::DT_FLOAT16,
            Ops::NN::DT_HIFLOAT4_SCALE,
            ge::DT_FLOAT};
    auto holder = MakeAddHi4Context(cfg, &info);
    auto* context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(context, nullptr);
    ExposedGetShapeAttrsInfo tiling(context);
    ClearErrors();
    std::cout << "[AddRmsNormHiF4QuantInputDtypeErrorReporting][908] BEGIN scan_id=908 group=G06 line=122" << std::endl;
    const ge::graphStatus status = tiling.RunGetShapeAttrsInfo();
    const ErrSnapshot errors = TakeSnapshot();
    std::cout << "[AddRmsNormHiF4QuantInputDtypeErrorReporting][908] return=" << status << " count=" << errors.size()
              << " errors: " << Report(errors) << std::endl;
    std::cout << "[AddRmsNormHiF4QuantInputDtypeErrorReporting][908] END" << std::endl;
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    ASSERT_EQ(errors.size(), 1U);
    EXPECT_EQ(errors[0].error_id, "EZ0020");
    EXPECT_EQ(errors[0].args_map.count("op_name"), 1U);
    EXPECT_FALSE(errors[0].args_map.at("op_name").empty());
    EXPECT_EQ(errors[0].args_map.at("param_name"), "x1");
    EXPECT_EQ(errors[0].args_map.at("reason"), "Supported dtypes are float16 and bfloat16");
}

TEST(AddRmsNormHiF4QuantInputDtypeErrorReporting, Target909_GAMMA_DTYPE_UNSUPPORTED)
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
    Cfg cfg{&x1,         &x2,         &gamma,          &beta,          &y,
            &xOut,       &mxScale,    &rstd,           ge::DT_FLOAT16, ge::DT_FLOAT16,
            ge::DT_INT8, ge::DT_INT8, ge::DT_HIFLOAT4, ge::DT_FLOAT16, Ops::NN::DT_HIFLOAT4_SCALE,
            ge::DT_FLOAT};
    auto holder = MakeAddHi4Context(cfg, &info);
    auto* context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(context, nullptr);
    ExposedGetShapeAttrsInfo tiling(context);
    ClearErrors();
    std::cout << "[AddRmsNormHiF4QuantInputDtypeErrorReporting][909] BEGIN scan_id=909 group=G06 line=126" << std::endl;
    const ge::graphStatus status = tiling.RunGetShapeAttrsInfo();
    const ErrSnapshot errors = TakeSnapshot();
    std::cout << "[AddRmsNormHiF4QuantInputDtypeErrorReporting][909] return=" << status << " count=" << errors.size()
              << " errors: " << Report(errors) << std::endl;
    std::cout << "[AddRmsNormHiF4QuantInputDtypeErrorReporting][909] END" << std::endl;
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    ASSERT_EQ(errors.size(), 1U);
    EXPECT_EQ(errors[0].error_id, "EZ0020");
    EXPECT_EQ(errors[0].args_map.count("op_name"), 1U);
    EXPECT_FALSE(errors[0].args_map.at("op_name").empty());
    EXPECT_EQ(errors[0].args_map.at("param_name"), "gamma");
    EXPECT_EQ(errors[0].args_map.at("reason"), "Supported dtypes are float16, bfloat16 and float32");
}

TEST(AddRmsNormHiF4QuantInputDtypeErrorReporting, Target910_BETA_DTYPE_MISMATCH)
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
            ge::DT_BF16,
            ge::DT_HIFLOAT4,
            ge::DT_FLOAT16,
            Ops::NN::DT_HIFLOAT4_SCALE,
            ge::DT_FLOAT};
    auto holder = MakeAddHi4Context(cfg, &info);
    auto* context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(context, nullptr);
    ExposedGetShapeAttrsInfo tiling(context);
    ClearErrors();
    std::cout << "[AddRmsNormHiF4QuantInputDtypeErrorReporting][910] BEGIN scan_id=910 group=G07 line=132" << std::endl;
    const ge::graphStatus status = tiling.RunGetShapeAttrsInfo();
    const ErrSnapshot errors = TakeSnapshot();
    std::cout << "[AddRmsNormHiF4QuantInputDtypeErrorReporting][910] return=" << status << " count=" << errors.size()
              << " errors: " << Report(errors) << std::endl;
    std::cout << "[AddRmsNormHiF4QuantInputDtypeErrorReporting][910] END" << std::endl;
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    ASSERT_EQ(errors.size(), 1U);
    EXPECT_EQ(errors[0].error_id, "EZ0021");
    EXPECT_EQ(errors[0].args_map.count("op_name"), 1U);
    EXPECT_FALSE(errors[0].args_map.at("op_name").empty());
    EXPECT_EQ(errors[0].args_map.at("param_names"), "gamma, beta");
    EXPECT_EQ(errors[0].args_map.at("reason"), "Input gamma and beta dtypes must be equal");
}

TEST(AddRmsNormHiF4QuantInputDtypeErrorReporting, Target912_Y_DTYPE)
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
    ExposedGetShapeAttrsInfo tiling(context);
    ClearErrors();
    std::cout << "[AddRmsNormHiF4QuantInputDtypeErrorReporting][912] BEGIN scan_id=912 group=G06 line=151" << std::endl;
    const ge::graphStatus status = tiling.RunGetShapeAttrsInfo();
    const ErrSnapshot errors = TakeSnapshot();
    std::cout << "[AddRmsNormHiF4QuantInputDtypeErrorReporting][912] return=" << status << " count=" << errors.size()
              << " errors: " << Report(errors) << std::endl;
    std::cout << "[AddRmsNormHiF4QuantInputDtypeErrorReporting][912] END" << std::endl;
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    ASSERT_EQ(errors.size(), 1U);
    EXPECT_EQ(errors[0].error_id, "EZ0020");
    EXPECT_EQ(errors[0].args_map.count("op_name"), 1U);
    EXPECT_FALSE(errors[0].args_map.at("op_name").empty());
    EXPECT_EQ(errors[0].args_map.at("param_name"), "y");
    EXPECT_EQ(errors[0].args_map.at("reason"), "Output y dtype must be HIFLOAT4");
}

TEST(AddRmsNormHiF4QuantInputDtypeErrorReporting, Target915_XOUT_DTYPE)
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
            ge::DT_HIFLOAT4,
            ge::DT_INT8,
            Ops::NN::DT_HIFLOAT4_SCALE,
            ge::DT_FLOAT};
    auto holder = MakeAddHi4Context(cfg, &info);
    auto* context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(context, nullptr);
    ExposedGetShapeAttrsInfo tiling(context);
    ClearErrors();
    std::cout << "[AddRmsNormHiF4QuantInputDtypeErrorReporting][915] BEGIN scan_id=915 group=G07 line=160" << std::endl;
    const ge::graphStatus status = tiling.RunGetShapeAttrsInfo();
    const ErrSnapshot errors = TakeSnapshot();
    std::cout << "[AddRmsNormHiF4QuantInputDtypeErrorReporting][915] return=" << status << " count=" << errors.size()
              << " errors: " << Report(errors) << std::endl;
    std::cout << "[AddRmsNormHiF4QuantInputDtypeErrorReporting][915] END" << std::endl;
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    ASSERT_EQ(errors.size(), 1U);
    EXPECT_EQ(errors[0].error_id, "EZ0021");
    EXPECT_EQ(errors[0].args_map.count("op_name"), 1U);
    EXPECT_FALSE(errors[0].args_map.at("op_name").empty());
    EXPECT_EQ(errors[0].args_map.at("param_names"), "x, x1");
    EXPECT_EQ(errors[0].args_map.at("reason"), "Output x dtype must equal the input dtype");
}

TEST(AddRmsNormHiF4QuantInputDtypeErrorReporting, Target916_SCALE_DTYPE)
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
            ge::DT_HIFLOAT4,
            ge::DT_FLOAT16,
            ge::DT_INT8,
            ge::DT_FLOAT};
    auto holder = MakeAddHi4Context(cfg, &info);
    auto* context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(context, nullptr);
    ExposedGetShapeAttrsInfo tiling(context);
    ClearErrors();
    std::cout << "[AddRmsNormHiF4QuantInputDtypeErrorReporting][916] BEGIN scan_id=916 group=G06 line=164" << std::endl;
    const ge::graphStatus status = tiling.RunGetShapeAttrsInfo();
    const ErrSnapshot errors = TakeSnapshot();
    std::cout << "[AddRmsNormHiF4QuantInputDtypeErrorReporting][916] return=" << status << " count=" << errors.size()
              << " errors: " << Report(errors) << std::endl;
    std::cout << "[AddRmsNormHiF4QuantInputDtypeErrorReporting][916] END" << std::endl;
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    ASSERT_EQ(errors.size(), 1U);
    EXPECT_EQ(errors[0].error_id, "EZ0020");
    EXPECT_EQ(errors[0].args_map.count("op_name"), 1U);
    EXPECT_FALSE(errors[0].args_map.at("op_name").empty());
    EXPECT_EQ(errors[0].args_map.at("param_name"), "scale");
    EXPECT_EQ(errors[0].args_map.at("reason"), "Output scale dtype must be HIFLOAT4_SCALE");
}

TEST(AddRmsNormHiF4QuantInputDtypeErrorReporting, Target904_X1_TENSOR_NULL)
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
            ge::DT_HIFLOAT4,
            ge::DT_FLOAT16,
            Ops::NN::DT_HIFLOAT4_SCALE,
            ge::DT_FLOAT};
    auto holder = MakeAddHi4Context(cfg, &info);
    auto* context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(context, nullptr);
    gert::KernelContext* kernelContext = reinterpret_cast<gert::KernelContext*>(context);
    ASSERT_NE(kernelContext, nullptr);
    ClearErrors();
    kernelContext->MutableInput(0)->Set(nullptr, nullptr);
    ASSERT_EQ(context->GetInputTensor(0), nullptr);
    ASSERT_NE(context->GetInputTensor(1), nullptr);
    ExposedCheckInputDtype tiling(context);
    std::cout << "[AddRmsNormHiF4QuantInputDtypeErrorReporting][904] BEGIN scan_id=904 group=G13 line=110" << std::endl;
    const ge::graphStatus status = tiling.RunCheckInputDtype();
    const ErrSnapshot errors = TakeSnapshot();
    std::cout << "[AddRmsNormHiF4QuantInputDtypeErrorReporting][904] return=" << status << " count=" << errors.size()
              << " errors: " << Report(errors) << std::endl;
    std::cout << "[AddRmsNormHiF4QuantInputDtypeErrorReporting][904] END" << std::endl;
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    ASSERT_EQ(errors.size(), 1U);
    EXPECT_EQ(errors[0].error_id, "EZ1008");
    EXPECT_EQ(errors[0].args_map.count("opName"), 1U);
    EXPECT_FALSE(errors[0].args_map.at("opName").empty());
    EXPECT_EQ(errors[0].args_map.at("reason"), "Missing tiling context object: context_->GetInputTensor(X1_INDEX)");
}

TEST(AddRmsNormHiF4QuantInputDtypeErrorReporting, Target905_X2_TENSOR_NULL)
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
            ge::DT_HIFLOAT4,
            ge::DT_FLOAT16,
            Ops::NN::DT_HIFLOAT4_SCALE,
            ge::DT_FLOAT};
    auto holder = MakeAddHi4Context(cfg, &info);
    auto* context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(context, nullptr);
    gert::KernelContext* kernelContext = reinterpret_cast<gert::KernelContext*>(context);
    ASSERT_NE(kernelContext, nullptr);
    ClearErrors();
    kernelContext->MutableInput(1)->Set(nullptr, nullptr);
    ASSERT_EQ(context->GetInputTensor(1), nullptr);
    ASSERT_NE(context->GetInputTensor(0), nullptr);
    ExposedCheckInputDtype tiling(context);
    std::cout << "[AddRmsNormHiF4QuantInputDtypeErrorReporting][905] BEGIN scan_id=905 group=G13 line=111" << std::endl;
    const ge::graphStatus status = tiling.RunCheckInputDtype();
    const ErrSnapshot errors = TakeSnapshot();
    std::cout << "[AddRmsNormHiF4QuantInputDtypeErrorReporting][905] return=" << status << " count=" << errors.size()
              << " errors: " << Report(errors) << std::endl;
    std::cout << "[AddRmsNormHiF4QuantInputDtypeErrorReporting][905] END" << std::endl;
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    ASSERT_EQ(errors.size(), 1U);
    EXPECT_EQ(errors[0].error_id, "EZ1008");
    EXPECT_EQ(errors[0].args_map.count("opName"), 1U);
    EXPECT_FALSE(errors[0].args_map.at("opName").empty());
    EXPECT_EQ(errors[0].args_map.at("reason"), "Missing tiling context object: context_->GetInputTensor(X2_INDEX)");
}

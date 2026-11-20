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

struct ExposedAddHi4 : optiling::AddRmsNormHiF4QuantRFullLoadTiling {
    using AddRmsNormHiF4QuantRFullLoadTiling::AddRmsNormHiF4QuantRFullLoadTiling;
    ge::graphStatus RunCheckOutputDtype() { return CheckOutputDtype(); }
    ge::graphStatus RunCheckOutputShapeValue() { return CheckOutputShapeValue(); }
    ge::graphStatus RunGetShapeAttrsInfo() { return GetShapeAttrsInfo(); }
    void InjectShapeState(uint64_t numCol, uint32_t rstdFlag)
    {
        numCol_ = numCol;
        rstdFlag_ = rstdFlag;
    }
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
        .NodeAttrs({{"eps", Ops::NN::AnyValue::CreateFrom<float>(1e-6f)},
                    {"output_rstd", Ops::NN::AnyValue::CreateFrom<bool>(true)},
                    {"scale_alg", Ops::NN::AnyValue::CreateFrom<int64_t>(0)}})
        .CompileInfo(compileInfo)
        .PlatformInfo(reinterpret_cast<char*>(&platform))
        .TilingData(tilingData.get())
        .Workspace(workspace)
        .Build();
}

TEST(AddRmsNormHiF4QuantOutputMiscErrorReporting, Case919CheckOutputDtypeRstdMustBeFloat32)
{
    gert::StorageShape x1 = {{8, 64}, {8, 64}};
    gert::StorageShape x2 = {{8, 64}, {8, 64}};
    gert::StorageShape gamma = {{64}, {64}};
    gert::StorageShape beta = {{64}, {64}};
    gert::StorageShape y = {{8, 64}, {8, 64}};
    gert::StorageShape xOut = {{8, 64}, {8, 64}};
    gert::StorageShape scale = {{8, 1, 1}, {8, 1, 1}};
    gert::StorageShape rstd = {{8, 1}, {8, 1}};
    optiling::AddRmsNormHiF4QuantCompileInfo compileInfo{};
    auto holder = MakeAddHi4Context(
        Cfg{&x1, &x2, &gamma, &beta, &y, &xOut, &scale, &rstd, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
            ge::DT_FLOAT16, Ops::NN::DT_HIFLOAT4, ge::DT_FLOAT16, Ops::NN::DT_HIFLOAT4_SCALE, ge::DT_FLOAT16},
        &compileInfo);
    auto* context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(context, nullptr);
    ExposedAddHi4 tiling(context);
    ClearErrors();
    const auto status = tiling.RunCheckOutputDtype();
    const auto errors = TakeSnapshot();
    std::cout << "[add-misc][919] ACTUAL return=" << status << " count=" << errors.size() << " " << Report(errors)
              << std::endl;
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    ASSERT_EQ(errors.size(), 1U);
    EXPECT_EQ(errors[0].error_id, "EZ0020");
    EXPECT_EQ(errors[0].args_map.at("param_name"), "rstd");
    EXPECT_EQ(errors[0].args_map.at("reason"), "Output rstd dtype must be float32");
}

TEST(AddRmsNormHiF4QuantOutputMiscErrorReporting, Case933CheckOutputShapeValueLastDimOdd)
{
    gert::StorageShape x1 = {{8, 63}, {8, 63}};
    gert::StorageShape x2 = {{8, 63}, {8, 63}};
    gert::StorageShape gamma = {{63}, {63}};
    gert::StorageShape beta = {{63}, {63}};
    gert::StorageShape y = {{8, 63}, {8, 63}};
    gert::StorageShape xOut = {{8, 63}, {8, 63}};
    gert::StorageShape scale = {{8, 1, 1}, {8, 1, 1}};
    gert::StorageShape rstd = {{8, 1}, {8, 1}};
    optiling::AddRmsNormHiF4QuantCompileInfo compileInfo{};
    auto holder = MakeAddHi4Context(
        Cfg{&x1, &x2, &gamma, &beta, &y, &xOut, &scale, &rstd, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16,
            ge::DT_FLOAT16, Ops::NN::DT_HIFLOAT4, ge::DT_FLOAT16, Ops::NN::DT_HIFLOAT4_SCALE, ge::DT_FLOAT},
        &compileInfo);
    auto* context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(context, nullptr);
    ExposedAddHi4 tiling(context);
    tiling.InjectShapeState(63U, 0U); // numCol_ = 63, rstdFlag_ = 0
    ClearErrors();
    const auto status = tiling.RunCheckOutputShapeValue();
    const auto errors = TakeSnapshot();
    std::cout << "[add-misc][933] ACTUAL return=" << status << " count=" << errors.size() << " " << Report(errors)
              << std::endl;
    EXPECT_EQ(status, ge::GRAPH_FAILED);
    ASSERT_EQ(errors.size(), 1U);
    EXPECT_EQ(errors[0].error_id, "EZ0009");
    EXPECT_EQ(errors[0].args_map.at("param_name"), "x1");
    EXPECT_EQ(errors[0].args_map.at("reason"), "Last input dimension must be even for packed HiF4 output");
}

} // namespace

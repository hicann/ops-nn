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

struct ExposedCheckOutputShapeValue : optiling::AddRmsNormHiF4QuantRFullLoadTiling {
    using AddRmsNormHiF4QuantRFullLoadTiling::AddRmsNormHiF4QuantRFullLoadTiling;
    ge::graphStatus RunCheckOutputShapeValue() { return CheckOutputShapeValue(); }
    void InjectShapeState(uint32_t rstdFlag)
    {
        numCol_ = 64U; // HIF4_BLOCK_SIZE baseline: R = 64 -> expectedLastDim = 1
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
        .NodeAttrs({})
        .CompileInfo(compileInfo)
        .PlatformInfo(reinterpret_cast<char*>(&platform))
        .TilingData(tilingData.get())
        .Workspace(workspace)
        .Build();
}

void RunShapeCase(const string& caseName, const vector<int64_t>& scaleDims, const vector<int64_t>& rstdDims,
                  uint32_t rstdFlag, const string& expectedCode, const string& expectedParamNames,
                  const string& expectedIncorrectDims, const string& expectedReason)
{
    SCOPED_TRACE(caseName);
    gert::StorageShape x1Shape({8, 64}, {8, 64});
    gert::StorageShape x2Shape({8, 64}, {8, 64});
    gert::StorageShape gammaShape({64}, {64});
    gert::StorageShape betaShape({64}, {64});
    gert::StorageShape yShape({8, 64},
                              {8, 64}); // baseline y mirrors x1 (CheckOutputShapeValue expects yshape == xshape)
    gert::StorageShape xOutShape({8, 64}, {8, 64});
    gert::StorageShape scaleShape;
    for (auto d : scaleDims) {
        scaleShape.MutableOriginShape().AppendDim(d);
        scaleShape.MutableStorageShape().AppendDim(d);
    }
    gert::StorageShape rstdShape;
    for (auto d : rstdDims) {
        rstdShape.MutableOriginShape().AppendDim(d);
        rstdShape.MutableStorageShape().AppendDim(d);
    }

    optiling::AddRmsNormHiF4QuantCompileInfo compileInfo{2, 262144};
    Cfg cfg{&x1Shape,       &x2Shape,       &gammaShape,     &betaShape,     &yShape,
            &xOutShape,     &scaleShape,    &rstdShape,      ge::DT_FLOAT16, ge::DT_FLOAT16,
            ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_HIFLOAT4, ge::DT_FLOAT16, Ops::NN::DT_HIFLOAT4_SCALE,
            ge::DT_FLOAT};
    auto holder = MakeAddHi4Context(cfg, &compileInfo);
    auto* context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(context, nullptr);

    ExposedCheckOutputShapeValue tiler(context);
    tiler.InjectShapeState(rstdFlag);

    ClearErrors(); // BEFORE the single original call
    const ge::graphStatus status = tiler.RunCheckOutputShapeValue();
    const ErrSnapshot errors = TakeSnapshot(); // exactly one consume AFTER

    cout << "[" << caseName << "] actual return = " << status << ", actual count = " << errors.size()
         << ", errors = " << Report(errors) << endl;

    EXPECT_EQ(status, ge::GRAPH_FAILED);
    ASSERT_EQ(errors.size(), 1U); // formal count 1
    const auto& error = errors.front();
    cout << "[" << caseName << "] error_id = " << error.error_id << endl;
    for (const auto& field : error.args_map) {
        cout << "[" << caseName << "] args_map[" << field.first << "] = " << field.second << endl;
    }
    EXPECT_EQ(error.error_id, expectedCode);
    if (expectedCode == "EZ0013") {
        ASSERT_TRUE(error.args_map.count("param_names") > 0);
        EXPECT_EQ(error.args_map.at("param_names"), expectedParamNames);
        ASSERT_TRUE(error.args_map.count("incorrect_dims") > 0);
        EXPECT_EQ(error.args_map.at("incorrect_dims"), expectedIncorrectDims);
    } else { // EZ0009
        ASSERT_TRUE(error.args_map.count("param_name") > 0);
        EXPECT_EQ(error.args_map.at("param_name"), expectedParamNames);
    }
    ASSERT_TRUE(error.args_map.count("reason") > 0);
    EXPECT_EQ(error.args_map.at("reason"), expectedReason);
}
} // namespace

TEST(AddRmsNormHiF4QuantOutputShapeErrorReporting, Case921ScaleRankMismatch)
{
    RunShapeCase("921", {8, 1}, {8, 1}, 0U, "EZ0013", "scale, x1", "2, 2",
                 "Output scale rank must equal input rank plus 1");
}

TEST(AddRmsNormHiF4QuantOutputShapeErrorReporting, Case922ScaleDim0Mismatch)
{
    RunShapeCase("922", {7, 1, 1}, {8, 1}, 0U, "EZ0009", "scale", "", "Dimension 0 must equal x1 dimension 8");
}

TEST(AddRmsNormHiF4QuantOutputShapeErrorReporting, Case923ScalePenultimateMismatch)
{
    RunShapeCase("923", {8, 2, 1}, {8, 1}, 0U, "EZ0009", "scale", "",
                 "Penultimate dimension must equal ceil(input last dimension / 64) = 1");
}

TEST(AddRmsNormHiF4QuantOutputShapeErrorReporting, Case924ScaleLastDimMismatch)
{
    RunShapeCase("924", {8, 1, 2}, {8, 1}, 0U, "EZ0009", "scale", "", "Last dimension must be 1");
}

TEST(AddRmsNormHiF4QuantOutputShapeErrorReporting, Case926RstdRankMismatch)
{
    RunShapeCase("926", {8, 1, 1}, {8}, 1U, "EZ0013", "rstd, x1", "1, 2", "Output rstd rank must equal input rank");
}

TEST(AddRmsNormHiF4QuantOutputShapeErrorReporting, Case927RstdDim1NotOne)
{
    RunShapeCase("927", {8, 1, 1}, {8, 2}, 1U, "EZ0009", "rstd", "", "Dimension 1 must be 1");
}

TEST(AddRmsNormHiF4QuantOutputShapeErrorReporting, Case928RstdDim0Mismatch)
{
    RunShapeCase("928", {8, 1, 1}, {7, 1}, 1U, "EZ0009", "rstd", "", "Dimension 0 must equal x1 dimension 8");
}
